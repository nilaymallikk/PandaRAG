"""Run the dense-retrieval benchmark over all 500 HotpotQA questions.

The benchmark is retrieval-only: it never calls the DeepSeek API and it never
generates an answer. It evaluates the dense retriever against the HotpotQA
supporting facts and stores the raw rankings so that later RAG pipelines can
reuse exactly the same retrieval output.

Usage::

    python -m src.retrieval_benchmark                # all 500 questions, K<=10
    python -m src.retrieval_benchmark --limit 5      # quick run

Artifacts::

    results/retrieval/dense_retrieval.jsonl        raw per-question rankings
    results/retrieval/dense_retrieval.meta.json    config + aggregate metrics
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src import config, dataset, evaluation
from src.retrieval import DenseRetriever, RetrievalResult


def build_record(
    example: dataset.HotpotExample,
    result: RetrievalResult,
    metrics: Mapping[str, Mapping[str, float]],
    k_max: int,
) -> dict[str, Any]:
    """One raw retrieval record, reusable by the RAG pipelines."""
    return {
        "question_id": example.example_id,
        "question": example.question,
        "question_type": example.question_type,
        "level": example.level,
        "gold_answer": example.answer,
        "supporting_titles": list(example.supporting_titles),
        "supporting_facts": list(example.supporting_fact_ids),
        "k_max": k_max,
        "candidate_paragraphs": len(example.context_titles),
        "retrieved_documents": [hit.to_dict() for hit in result.paragraphs],
        "retrieved_sentences": [hit.to_dict() for hit in result.sentences],
        "metrics": dict(metrics),
        "encode_s": result.encode_s,
        "search_s": result.search_s,
        "retrieval_latency_s": result.latency_s,
    }


def run_benchmark(
    *,
    limit: int | None = None,
    k_max: int = config.RETRIEVAL_MAX_K,
    retriever: DenseRetriever | None = None,
    results_file: Path | None = None,
    meta_file: Path | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Retrieve for every question, save raw results, return the metadata."""
    retriever = retriever or DenseRetriever()
    results_path = Path(results_file or config.DENSE_RESULTS_FILE)
    meta_path = Path(meta_file or config.DENSE_RESULTS_META_FILE)

    examples = dataset.load_dataset()
    if limit is not None:
        examples = examples[:limit]

    model_load_s = retriever.load()
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for example in _progress(examples, quiet=quiet):
        result = retriever.retrieve(example, k=k_max)
        metrics = evaluation.evaluate_retrieval(result, example, config.RETRIEVAL_K_VALUES)
        records.append(build_record(example, result, metrics, k_max))
    runtime_s = time.perf_counter() - started

    aggregate = evaluation.aggregate_retrieval_metrics(
        [record["metrics"] for record in records], config.RETRIEVAL_K_VALUES
    )
    meta: dict[str, Any] = {
        "stage": "dense retrieval benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "llm_calls": 0,
        "llm_used": False,
        "n_questions": len(records),
        "k_values": list(config.RETRIEVAL_K_VALUES),
        "max_k": k_max,
        "runtime_s": round(runtime_s, 3),
        "model_load_s": model_load_s,
        "retriever": retriever.settings(),
        "dataset": dataset.dataset_fingerprint(),
        "metrics": aggregate,
        "latency_s": {
            "encode": evaluation.aggregate_latency([record["encode_s"] for record in records]),
            "search": evaluation.aggregate_latency([record["search_s"] for record in records]),
            "retrieval": evaluation.aggregate_latency(
                [record["retrieval_latency_s"] for record in records]
            ),
        },
        "results_file": str(results_path),
    }

    save_records(records, results_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def _progress(items: Sequence[Any], *, quiet: bool) -> Sequence[Any]:
    """Wrap the question list in a progress bar when tqdm is available."""
    if quiet:
        return items
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items, desc="dense retrieval", unit="q")


def save_records(records: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Write per-question retrieval records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Read the raw retrieval records written by this benchmark."""
    results_path = Path(path or config.DENSE_RESULTS_FILE)
    with results_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_records_by_id(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Raw retrieval records keyed by ``question_id`` for later stages."""
    return {record["question_id"]: record for record in load_records(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dense retrieval benchmark (no LLM calls).")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--k", type=int, default=config.RETRIEVAL_MAX_K, help="max K to rank")
    parser.add_argument("--results-file", type=Path, default=config.DENSE_RESULTS_FILE)
    parser.add_argument("--meta-file", type=Path, default=config.DENSE_RESULTS_META_FILE)
    parser.add_argument("--quiet", action="store_true", help="no progress bar")
    args = parser.parse_args(argv)

    meta = run_benchmark(
        limit=args.limit,
        k_max=args.k,
        results_file=args.results_file,
        meta_file=args.meta_file,
        quiet=args.quiet,
    )

    retriever = meta["retriever"]
    print("\nDense retrieval benchmark (retrieval only, no LLM calls)")
    print(f"  questions        {meta['n_questions']}")
    print(f"  embedding model  {retriever['embedding_model']} ({retriever['embedding_device']})")
    print(f"  index            {retriever['index']}, {retriever['similarity']}")
    print(f"  dataset sha256   {meta['dataset']['sha256']}")
    print(f"  runtime_s        {meta['runtime_s']} (model load {meta['model_load_s']}s)")
    print("\nRetrieval vs HotpotQA supporting facts")
    print(evaluation.format_retrieval_summary(meta["metrics"]))
    latency = meta["latency_s"]
    print(
        "\nLatency per question (s): "
        f"encode mean {latency['encode']['mean']:.4f}, "
        f"search mean {latency['search']['mean']:.4f}, "
        f"retrieval mean {latency['retrieval']['mean']:.4f}, "
        f"retrieval total {latency['retrieval']['total']:.2f}"
    )
    print(f"\nraw results      {meta['results_file']}")
    print(f"metadata         {args.meta_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
