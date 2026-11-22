"""Reduced-sample ablation experiments (A0–A5).

A0 is SAC-RAG Adaptive itself (``src.pipelines.sac_rag.pipeline``); A1–A5 are the
PandaRAG pipeline with components switched off/on. All variants run on the **same
frozen stratified question subset**, so a difference is attributable to the
component and not to a different sample.

Ablations are explicitly labelled reduced-sample: they are never mixed into the
full-sample primary results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from src import config
from src.analysis import loaders
from src.pipelines.pandarag.pipeline import ABLATIONS
from src.pipelines.shared.sampling import stratified_sample

ABLATION_ROOT = config.RESULTS_DIR / "ablations"
SUMMARY_FILE = ABLATION_ROOT / "summary.json"
SAMPLE_SIZE = 100
SEED = 42


def _run(command: Sequence[str]) -> None:
    started = time.perf_counter()
    completed = subprocess.run(list(command), cwd=config.REPO_ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr[-1500:]}"
        )


def run_variant(variant: str, dataset: str, sample: list[str], *, concurrency: int) -> None:
    directory = ABLATION_ROOT / variant
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{dataset}.jsonl"
    if target.exists() and sum(1 for _ in target.open()) >= len(sample):
        print(f"skip {variant}/{dataset} (already complete)", flush=True)
        return

    if variant == "A0":
        # A0 = SAC-RAG Adaptive on the same subset.
        ids = set(sample)

        # Restrict through a temporary sample file is not supported by the SAC-RAG
        # CLI, so drive it in-process with the same spec machinery.
        from src.pipelines.shared.runner import RunSpec, run_pipeline
        from src.pipelines.sac_rag.pipeline import execute_question, load_sac_config
        from src.pipelines.sac_rag.classifier import load_frozen_labels as labels_for

        sac_config = load_sac_config()
        labels = labels_for(dataset)
        spec = RunSpec(
            pipeline="ablation_A0",
            dataset=dataset,
            concurrency=concurrency,
            config_fingerprint=None,
            sample_ids=ids,
            results_file=target,
            meta_file=directory / f"{dataset}.meta.json",
            extra_meta={"variant": "A0", "note": "SAC-RAG Adaptive on the ablation subset", "sample_size": len(sample)},
        )

        def question_fn(example, service, client):
            return execute_question(
                example,
                service,
                client,
                semantic_type=labels[example.example_id]["question_type"],
                sac_config=sac_config,
                compression_enabled=False,
            )

        run_pipeline(spec, question_fn, semantic_types={k: v["question_type"] for k, v in labels.items()}, quiet=True)
        return

    _run(
        [
            sys.executable,
            "-m",
            "src.pipelines.pandarag.pipeline",
            "--dataset",
            dataset,
            "--ablation",
            variant,
            "--sample-size",
            str(len(sample)),
            "--results-dir",
            str(directory),
            "--concurrency",
            str(concurrency),
            "--quiet",
        ]
    )


def summarise() -> dict[str, Any]:
    rows = []
    for variant in sorted(ABLATION_ROOT.glob("A*")):
        per_dataset = []
        for dataset in loaders.DATASETS:
            path = variant / f"{dataset}.jsonl"
            if not path.exists():
                continue
            records = [json.loads(line) for line in path.open() if line.strip()]
            records = [row for row in records if row.get("success")]
            if not records:
                continue
            per_dataset.append(
                {
                    "dataset": dataset,
                    "n": len(records),
                    "em": round(mean(row["exact_match"] for row in records), 4),
                    "f1": round(mean(row["f1"] for row in records), 4),
                    "tokens": round(mean((row["input_tokens"] or 0) + (row["output_tokens"] or 0) for row in records), 1),
                    "cost": round(mean(row.get("api_cost_usd") or 0 for row in records), 8),
                }
            )
        if not per_dataset:
            continue
        rows.append(
            {
                "variant": variant.name,
                "n": sum(item["n"] for item in per_dataset),
                "em": round(mean(item["em"] for item in per_dataset), 4),
                "f1": round(mean(item["f1"] for item in per_dataset), 4),
                "tokens": round(mean(item["tokens"] for item in per_dataset), 1),
                "cost_per_question": round(mean(item["cost"] for item in per_dataset), 8),
                "per_dataset": per_dataset,
                "components": ABLATIONS.get(variant.name).as_dict() if variant.name in ABLATIONS else {
                    "note": "A0 = SAC-RAG Adaptive (all PandaRAG components off)"
                },
            }
        )

    payload = {
        "design": "reduced-sample ablation",
        "sample_size_per_dataset": SAMPLE_SIZE,
        "seed": SEED,
        "sample": "stratified by frozen semantic type, identical across variants",
        "rows": rows,
    }
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run reduced-sample ablations A0–A5")
    parser.add_argument("--datasets", nargs="*", default=list(loaders.DATASETS))
    parser.add_argument("--variants", nargs="*", default=["A0", "A1", "A2", "A3", "A4", "A5"])
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(argv)

    for dataset in args.datasets:
        sample = stratified_sample(dataset, args.sample_size)
        print(f"{dataset}: sample {len(sample)} questions", flush=True)
        for variant in args.variants:
            print(f"  running {variant}/{dataset} ...", flush=True)
            run_variant(variant, dataset, sample, concurrency=args.concurrency)
    payload = summarise()
    print(json.dumps(payload["rows"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
