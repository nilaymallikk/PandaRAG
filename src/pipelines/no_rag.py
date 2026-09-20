"""No-RAG generation baseline: question -> DeepSeek -> answer.

No retrieval is performed here: no FAISS, no BM25, no vector database, no
context from ``data/hotpotqa_500.json`` is supplied to the model. The dataset
is read only for the question text and the gold answer used in evaluation.

Generation settings are the shared ones from :mod:`src.config` (model,
thinking mode, temperature, output budget); this pipeline introduces no
pipeline-specific generation settings. The frozen prompt
(``GENERATION_SYSTEM_PROMPT`` + ``Question: ... / Answer:``) is the template
all four pipelines share.

Usage::

    python -m src.pipelines.no_rag --limit 10     # smoke test (10 questions)
    python -m src.pipelines.no_rag                # full 500-question run

Artifacts::

    results/no_rag.jsonl       raw per-question result records
    results/no_rag.meta.json   run metadata + aggregate metrics
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from src import answer_eval, config, dataset
from src.llm import DeepSeekClient, LLMResponse


def build_prompt(question: str) -> str:
    """Frozen No-RAG user message: the question, nothing else."""
    return f"Question: {question.strip()}\nAnswer:"


def build_record(
    example: dataset.HotpotExample,
    response: LLMResponse,
    *,
    client: DeepSeekClient,
) -> dict[str, Any]:
    """Map one example + one LLM response to the unified result record."""
    if response.ok:
        prediction: str | None = response.text
        scores = answer_eval.score_answer(prediction, example.answer)
    else:
        prediction = None
        scores = {"exact_match": 0.0, "f1": 0.0}
    return {
        "question_id": example.example_id,
        "question": example.question,
        "question_type": example.question_type,
        "level": example.level,
        "gold_answer": example.answer,
        "prediction": prediction,
        "retrieval_method": "none",
        "retrieval_k": None,
        "retrieved_docs": [],
        "retrieval_doc_recall": None,
        "retrieval_sentence_recall": None,
        "retrieval_complete": None,
        "exact_match": scores["exact_match"],
        "f1": scores["f1"],
        "input_tokens": response.prompt_tokens,
        "output_tokens": response.completion_tokens,
        "cached_input_tokens": response.cached_input_tokens,
        "reasoning_tokens": response.reasoning_tokens,
        "finish_reason": response.finish_reason,
        "truncated": response.truncated,
        "llm_calls": response.llm_calls,
        "from_cache": response.from_cache,
        "latency_seconds": response.latency_s,
        "api_cost_usd": response.cost_usd,
        "model": response.model or client.model,
        "thinking": client.thinking,
        "error": response.error,
        "success": response.ok,
        "timestamp_utc": response.created_at,
    }


def _latency_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "total": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean": round(mean(ordered), 6),
        "median": round(median(ordered), 6),
        "p95": round(ordered[p95_index], 6),
        "total": round(sum(ordered), 6),
    }

def build_metadata(
    records: Sequence[Mapping[str, Any]],
    *,
    client: DeepSeekClient,
    started_at: str,
    finished_at: str,
    runtime_s: float,
) -> dict[str, Any]:
    """Aggregate run metadata from raw per-question records."""
    successful = [record for record in records if record["success"]]
    failed = [record for record in records if not record["success"]]
    per_question_scores = [
        {"exact_match": record["exact_match"], "f1": record["f1"]}
        for record in records
    ]
    overall = answer_eval.aggregate_answer_scores(per_question_scores)
    successful_scores = [
        {"exact_match": record["exact_match"], "f1": record["f1"]}
        for record in successful
    ]
    successful_only = answer_eval.aggregate_answer_scores(successful_scores)
    fingerprint = dataset.dataset_fingerprint()
    llm_settings = client.settings()
    latency = _latency_stats([float(record["latency_seconds"]) for record in records])
    from_cache_count = sum(1 for record in records if record["from_cache"])
    return {
        "pipeline": "no_rag",
        "retrieval_method": "none",
        "dataset": fingerprint,
        "dataset_path": str(fingerprint["path"]),
        "dataset_sha256": fingerprint["sha256"],
        "n_questions": len(records),
        "n": len(records),
        "model": llm_settings.get("model", getattr(client, "model", None)),
        "thinking": llm_settings.get("thinking", getattr(client, "thinking", None)),
        "temperature": llm_settings.get("temperature"),
        "max_output_tokens": llm_settings.get("max_output_tokens"),
        "generation_config": {
            "model": llm_settings.get("model", getattr(client, "model", None)),
            "thinking": llm_settings.get("thinking", getattr(client, "thinking", None)),
            "temperature": llm_settings.get("temperature"),
            "max_output_tokens": llm_settings.get("max_output_tokens"),
            "system_prompt": config.GENERATION_SYSTEM_PROMPT,
            "user_template": "Question: {question}\nAnswer:",
        },
        "timestamp": finished_at,
        "llm": llm_settings,
        "generation_prompt": {
            "system": config.GENERATION_SYSTEM_PROMPT,
            "user_template": "Question: {question}\nAnswer:",
        },
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "runtime_s": round(runtime_s, 4),
        "successful": len(successful),
        "failed": len(failed),
        "failed_ids": [record["question_id"] for record in failed],
        "total_llm_calls": sum(int(record["llm_calls"]) for record in records),
        "from_cache": from_cache_count,
        "cache_stats": {
            "hits": from_cache_count,
            "misses": len(records) - from_cache_count,
            "hit_rate": round(from_cache_count / len(records), 6) if records else 0.0,
        },
        "total_input_tokens": sum(int(record["input_tokens"]) for record in records),
        "total_output_tokens": sum(int(record["output_tokens"]) for record in records),
        "total_cached_input_tokens": sum(
            int(record["cached_input_tokens"]) for record in records
        ),
        "total_api_cost_usd": round(
            sum(float(record["api_cost_usd"]) for record in records), 8
        ),
        "mean_input_tokens": (
            round(mean(int(record["input_tokens"]) for record in records), 4)
            if records
            else 0.0
        ),
        "mean_output_tokens": (
            round(mean(int(record["output_tokens"]) for record in records), 4)
            if records
            else 0.0
        ),
        "latency_s": latency,
        "mean_latency_seconds": latency["mean"],
        "median_latency_seconds": latency["median"],
        "total_latency_seconds": latency["total"],
        "answer_metrics_successful_only": successful_only,
        "answer_metrics_all": overall,
        "exact_match": overall["exact_match"],
        "f1": overall["f1"],
        "llm_calls": sum(int(record["llm_calls"]) for record in records),
        "llm_used": True,
        "results_file": str(config.NO_RAG_RESULTS_FILE),
    }

def save_records(
    records: Sequence[Mapping[str, Any]], path: Path | None = None
) -> Path:
    """Write raw per-question records as JSONL."""
    target = Path(path or config.NO_RAG_RESULTS_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


def save_metadata(metadata: Mapping[str, Any], path: Path | None = None) -> Path:
    """Write run metadata as JSON."""
    target = Path(path or config.NO_RAG_RESULTS_META_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return target


def load_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Read raw per-question records back from JSONL."""
    target = Path(path or config.NO_RAG_RESULTS_FILE)
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_records_by_id(
    path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Raw records keyed by ``question_id``."""
    return {record["question_id"]: record for record in load_records(path)}


def run_pipeline(
    *,
    limit: int | None = None,
    client: DeepSeekClient | None = None,
    results_file: Path | None = None,
    meta_file: Path | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the No-RAG baseline and persist raw records + metadata."""
    active_client = client or DeepSeekClient()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    examples = dataset.load_dataset()
    if limit is not None:
        examples = examples[:limit]
    records: list[dict[str, Any]] = []
    total = len(examples)
    for index, example in enumerate(examples):
        prompt = build_prompt(example.question)
        response = active_client.generate(
            prompt, system=config.GENERATION_SYSTEM_PROMPT
        )
        records.append(build_record(example, response, client=active_client))
        if not quiet and (index + 1 == total or (index + 1) % 25 == 0):
            done = len([item for item in records if item["success"]])
            print(f"  progress {index + 1}/{total} (successful {done})", flush=True)
    runtime_s = time.perf_counter() - started
    finished_at = datetime.now(timezone.utc).isoformat()
    metadata = build_metadata(
        records,
        client=active_client,
        started_at=started_at,
        finished_at=finished_at,
        runtime_s=runtime_s,
    )
    save_records(records, results_file)
    save_metadata(metadata, meta_file)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: smoke test with ``--limit`` or the full run."""
    parser = argparse.ArgumentParser(
        description="No-RAG baseline: question -> DeepSeek -> answer (no retrieval)."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only the first N questions"
    )
    parser.add_argument("--results-file", type=Path, default=None)
    parser.add_argument("--meta-file", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args(argv)

    client = DeepSeekClient()
    settings = client.settings()
    print("No-RAG baseline (no retrieval)")
    print(f"  model            {settings['model']}")
    print(f"  thinking         {settings['thinking']}")
    print(f"  max_output       {settings['max_output_tokens']}")
    print(f"  temperature      {settings['temperature']}")
    print(f"  dataset sha256   {dataset.dataset_fingerprint()['sha256']}")
    metadata = run_pipeline(
        limit=args.limit,
        client=client,
        results_file=args.results_file,
        meta_file=args.meta_file,
        quiet=args.quiet,
    )
    print(f"\nquestions        {metadata['n_questions']}")
    print(f"successful       {metadata['successful']} (failed {metadata['failed']})")
    print(f"exact_match      {metadata['exact_match']:.4f}")
    print(f"f1               {metadata['f1']:.4f}")
    print(
        f"tokens           in {metadata['total_input_tokens']} / "
        f"out {metadata['total_output_tokens']} "
        f"(cached {metadata['total_cached_input_tokens']})"
    )
    print(f"api_cost_usd     {metadata['total_api_cost_usd']:.8f}")
    print(f"llm_calls        {metadata['total_llm_calls']}")
    print(
        f"latency_s        mean {metadata['latency_s']['mean']:.4f} / "
        f"median {metadata['latency_s']['median']:.4f} / "
        f"total {metadata['latency_s']['total']:.2f}"
    )
    print(f"\nraw results      {args.results_file or config.NO_RAG_RESULTS_FILE}")
    print(f"metadata         {args.meta_file or config.NO_RAG_RESULTS_META_FILE}")
    return 0 if metadata["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
