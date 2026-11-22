"""Full experiment matrix runner (Phases 7–8).

Runs every system on every dataset, sequentially, resumably, with a cost ceiling.
Each run writes its own JSONL + meta; this script only orchestrates and reports.

Usage::

    python -m src.experiments.run_matrix --limit 40                  # pilot
    python -m src.experiments.run_matrix                             # full 500
    python -m src.experiments.run_matrix --systems pandarag P1c --datasets hotpotqa

Ordering is deliberate: P4 runs before P5 and P6 so the persistent reranker score
cache is warm for later systems on the same questions. That affects wall-clock and
API spend, never the scores themselves (the cache is keyed by model+query+document).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from src import config

LOG_DIR = config.RESULTS_DIR / "logs"
RUN_LOG = LOG_DIR / "matrix.log"

DATASETS = ("hotpotqa", "2wiki", "musique")

#: system id -> (module, extra args)
SYSTEMS: dict[str, tuple[str, list[str]]] = {
    "P0": ("src.pipelines.corpus_baselines", ["--baseline", "P0"]),
    "P1c": ("src.pipelines.corpus_baselines", ["--baseline", "P1c"]),
    "P2c": ("src.pipelines.corpus_baselines", ["--baseline", "P2c"]),
    "P3c": ("src.pipelines.corpus_baselines", ["--baseline", "P3c"]),
    "P1n": ("src.pipelines.corpus_baselines", ["--baseline", "P1n"]),
    "P4": ("src.pipelines.sac_rag.pipeline", []),
    "P5": ("src.pipelines.sac_rag.pipeline", ["--compression"]),
    "P6": ("src.pipelines.pandarag.pipeline", []),
}

DEFAULT_ORDER = ("P0", "P1c", "P2c", "P3c", "P1n", "P4", "P5", "P6")

RESULT_DIRS = {
    "P0": "baseline_no_rag",
    "P1c": "baseline_dense_corpus",
    "P2c": "baseline_hybrid_corpus",
    "P3c": "baseline_adaptive_transferred",
    "P1n": "baseline_dense_paper_naive",
    "P4": "sac_rag_adaptive",
    "P5": "sac_rag_full",
    "P6": "pandarag",
}


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def meta_path(system: str, dataset: str) -> Path:
    return config.RESULTS_DIR / RESULT_DIRS[system] / f"{dataset}.meta.json"


def spent_usd() -> float:
    """Live spend across every meta file that reports it."""
    total = 0.0
    for path in config.RESULTS_DIR.glob("*/*.meta.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        aggregate = payload.get("aggregate")
        if isinstance(aggregate, dict):
            total += float(
                aggregate.get("total_api_cost_usd_live", aggregate.get("total_api_cost_usd", 0.0))
            )
    return round(total, 4)


def completed(system: str, dataset: str, limit: int | None) -> bool:
    path = meta_path(system, dataset)
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate", {})
    expected = limit if limit else payload.get("n_requested")
    return int(aggregate.get("n_success", 0)) >= int(expected or 0)


def run_one(system: str, dataset: str, *, limit: int | None, concurrency: int) -> None:
    module, extra = SYSTEMS[system]
    command = [
        sys.executable,
        "-m",
        module,
        "--dataset",
        dataset,
        *extra,
        "--concurrency",
        str(concurrency),
        "--quiet",
    ]
    if limit:
        command.extend(["--limit", str(limit)])
    log(f"START {system} {dataset} limit={limit} :: {' '.join(command[3:])}")
    started = time.perf_counter()
    completed_process = subprocess.run(command, cwd=config.REPO_ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if completed_process.returncode != 0:
        log(f"FAIL {system} {dataset} after {elapsed:.0f}s: {completed_process.stderr.strip()[-600:]}")
        return
    path = meta_path(system, dataset)
    if not path.exists():
        log(f"FAIL {system} {dataset}: no meta written")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregate = payload["aggregate"]
    log(
        f"DONE {system} {dataset} n={aggregate['n_success']}/{aggregate['n_questions']} "
        f"EM={aggregate['exact_match']} F1={aggregate['f1']} R@10={aggregate['recall@10']} "
        f"live_cost=${aggregate.get('total_api_cost_usd_live', aggregate['total_api_cost_usd'])} "
        f"in {elapsed/60:.1f}min"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the experiment matrix")
    parser.add_argument("--systems", nargs="*", default=list(DEFAULT_ORDER))
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--max-cost-usd", type=float, default=35.0)
    parser.add_argument("--force", action="store_true", help="re-run even if the meta is complete")
    args = parser.parse_args(argv)

    log(
        f"matrix start systems={args.systems} datasets={args.datasets} "
        f"limit={args.limit} concurrency={args.concurrency} ceiling=${args.max_cost_usd}"
    )
    for dataset in args.datasets:
        for system in args.systems:
            if not args.force and completed(system, dataset, args.limit):
                log(f"SKIP {system} {dataset}: already complete")
                continue
            current = spent_usd()
            if current >= args.max_cost_usd:
                log(f"STOP: spent ${current} >= ceiling ${args.max_cost_usd}")
                return 2
            run_one(system, dataset, limit=args.limit, concurrency=args.concurrency)
    log(f"matrix finished; live spend ≈ ${spent_usd()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
