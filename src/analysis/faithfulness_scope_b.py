"""Faithfulness judging for the Scope-B pipelines, using the frozen protocol.

Reuses ``faithfulness-judge/1.0`` wholesale — same system prompt, same judge prompt
template, same claim labels, same deterministic scoring — and only changes *which*
records are fed in and how the context is obtained (Scope-B records store
``context_text``, the exact block the generator saw).

Sampling: a deterministic stratified subset per (system, dataset) so the judge bill
is bounded; the sample is identical across systems. Judged entries are written to
``results/analysis_scope_b/``.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from src import config
from src.analysis import loaders
from src.faithfulness_eval import (
    JUDGE_MAX_TOKENS,
    JUDGE_SYSTEM_PROMPT,
    PROTOCOL_VERSION,
    build_judge_prompt,
    normalise_claims,
    parse_judge_json,
    score_claims,
)
from src.llm import DeepSeekClient
from src.pipelines.shared.sampling import stratified_sample

OUT_DIR = config.RESULTS_DIR / "analysis_scope_b"
DETAIL_FILE = OUT_DIR / "faithfulness.jsonl"
SUMMARY_FILE = OUT_DIR / "faithfulness_summary.json"
DEFAULT_SAMPLE = 100


def judge_record(client: DeepSeekClient, record: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "system": record.get("pipeline"),
        "dataset": record.get("dataset"),
        "question_id": record.get("question_id"),
        "success": bool(record.get("success")),
        "n_claims": 0,
        "faithfulness": None,
        "hallucination_score": None,
        "parse_ok": None,
    }
    answer = (record.get("prediction") or "").strip()
    if not record.get("success") or not answer:
        entry["parse_ok"] = False
        entry["note"] = "no answer to judge"
        return entry

    prompt = build_judge_prompt(
        record.get("question", ""),
        record.get("context_text") or "Context:\n(none)",
        answer,
    )
    try:
        response = client.generate(prompt, system=JUDGE_SYSTEM_PROMPT, max_tokens=JUDGE_MAX_TOKENS)
    except Exception as error:  # recorded, never hidden
        entry["parse_ok"] = False
        entry["error"] = f"{type(error).__name__}: {error}"
        return entry

    entry["judge_cost_usd"] = response.cost_usd
    entry["judge_input_tokens"] = response.prompt_tokens
    entry["judge_output_tokens"] = response.completion_tokens
    try:
        payload = parse_judge_json(response.text)
        claims = normalise_claims(payload)
        scores = score_claims(claims)
        entry.update(scores)
        entry["parse_ok"] = True
        entry["claims"] = claims
    except Exception as error:
        entry["parse_ok"] = False
        entry["error"] = f"{type(error).__name__}: {error}"
        entry["raw"] = (response.text or "")[:600]
    return entry


def run(
    *,
    systems: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    sample: int = DEFAULT_SAMPLE,
    concurrency: int = 4,
    quiet: bool = False,
) -> dict[str, Any]:
    systems = list(systems or [s for s in loaders.SYSTEMS if loaders.SYSTEMS[s].family != "closed_book"])
    datasets = list(datasets or loaders.DATASETS)
    client = DeepSeekClient()

    jobs: list[dict[str, Any]] = []
    for dataset in datasets:
        wanted = set(stratified_sample(dataset, sample))
        for system in systems:
            for record in loaders.load(system, dataset):
                if record["question_id"] in wanted:
                    jobs.append(record)

    entries: list[dict[str, Any]] = []
    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Write incrementally: a rate-limit stall must not lose completed judgements.
    sink = DETAIL_FILE.open("w", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(judge_record, client, record): record for record in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            entry = future.result()
            entries.append(entry)
            sink.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            sink.flush()
            if not quiet and index % 100 == 0:
                print(f"  judged {index}/{len(jobs)}", flush=True)

    sink.close()
    summary = _summarise(entries)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "judge_system_prompt": JUDGE_SYSTEM_PROMPT,
        "judge_max_output_tokens": JUDGE_MAX_TOKENS,
        "judge_model": config.DEEPSEEK_MODEL,
        "sample_per_system_dataset": sample,
        "sample_rule": "stratified by frozen semantic type (identical across systems)",
        "systems": systems,
        "datasets": datasets,
        "n_entries": len(entries),
        "runtime_s": round(time.perf_counter() - started, 1),
        "isolation": (
            "Judge input is question + the exact context block the generator received + answer. "
            "Gold answers, gold documents, EM/F1, system identity and K are never passed."
        ),
    }
    (OUT_DIR / "faithfulness.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return summary


def _summarise(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for system in sorted({entry["system"] for entry in entries if entry.get("system")}):
        rows = [entry for entry in entries if entry.get("system") == system]
        scored = [row for row in rows if row.get("faithfulness") is not None]
        summary[system] = {
            "n_judged": len(rows),
            "n_scored": len(scored),
            "faithfulness": round(mean(row["faithfulness"] for row in scored), 4) if scored else None,
            "hallucination_score": round(mean(row["hallucination_score"] for row in scored), 4) if scored else None,
            "unsupported_claim_rate": round(
                mean((row.get("n_unsupported", 0) or 0) / max(1, row["n_claims"]) for row in scored), 4
            )
            if scored
            else None,
            "contradicted_claim_rate": round(
                mean((row.get("n_contradicted", 0) or 0) / max(1, row["n_claims"]) for row in scored), 4
            )
            if scored
            else None,
            "fully_grounded_rate": round(
                mean(1.0 if row.get("grounding_label") == "fully" else 0.0 for row in scored), 4
            )
            if scored
            else None,
            "parse_failures": sum(1 for row in rows if row.get("parse_ok") is False),
            "judge_cost_usd": round(sum(row.get("judge_cost_usd") or 0 for row in rows), 6),
        }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Faithfulness judging for Scope-B results")
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(argv)
    summary = run(
        systems=args.systems,
        datasets=args.datasets,
        sample=args.sample,
        concurrency=args.concurrency,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
