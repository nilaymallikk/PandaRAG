"""Run a retrieval benchmark over all 500 HotpotQA questions.

The benchmark is retrieval-only: it never calls the DeepSeek API and it never
generates an answer. It evaluates a retriever against the HotpotQA supporting
facts and stores the raw rankings so that later RAG pipelines can reuse exactly
the same retrieval output. One runner and one evaluation path serve both
retrieval methods; only the artifacts differ.

Usage::

    python -m src.retrieval_benchmark --method dense       # all 500, K<=10
    python -m src.retrieval_benchmark --method bm25
    python -m src.retrieval_benchmark --method bm25 --limit 5

Artifacts::

    results/retrieval/dense_retrieval.jsonl        dense per-question rankings
    results/retrieval/dense_retrieval.meta.json    dense config + metrics
    results/retrieval/bm25_retrieval.jsonl         BM25 per-question rankings
    results/retrieval/bm25_retrieval.meta.json     BM25 config + metrics
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src import config, dataset, evaluation
from src.retrieval import BM25Retriever, DenseRetriever, RetrievalResult

RETRIEVER_CLASSES: dict[str, type] = {"dense": DenseRetriever, "bm25": BM25Retriever}
METHOD_PATHS: dict[str, tuple[Path, Path]] = {
    "dense": (config.DENSE_RESULTS_FILE, config.DENSE_RESULTS_META_FILE),
    "bm25": (config.BM25_RESULTS_FILE, config.BM25_RESULTS_META_FILE),
}


def build_record(
    example: dataset.HotpotExample,
    result: RetrievalResult,
    metrics: Mapping[str, Mapping[str, float]],
    k_max: int,
    method: str = "dense",
) -> dict[str, Any]:
    """One raw retrieval record, reusable by the RAG pipelines."""
    return {
        "question_id": example.example_id,
        "question": example.question,
        "question_type": example.question_type,
        "level": example.level,
        "gold_answer": example.answer,
        "retrieval_method": method,
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
    method: str = "dense",
    limit: int | None = None,
    k_max: int = config.RETRIEVAL_MAX_K,
    retriever: DenseRetriever | BM25Retriever | None = None,
    results_file: Path | None = None,
    meta_file: Path | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Retrieve for every question, save raw results, return the metadata."""
    if method not in RETRIEVER_CLASSES:
        raise ValueError(f"Unknown retrieval method {method!r}; expected {sorted(RETRIEVER_CLASSES)}.")
    default_results, default_meta = METHOD_PATHS[method]
    retriever = retriever or RETRIEVER_CLASSES[method]()
    results_path = Path(results_file or default_results)
    meta_path = Path(meta_file or default_meta)

    examples = dataset.load_dataset()
    if limit is not None:
        examples = examples[:limit]

    model_load_s = retriever.load()
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for example in _progress(examples, quiet=quiet, method=method):
        result = retriever.retrieve(example, k=k_max)
        metrics = evaluation.evaluate_retrieval(result, example, config.RETRIEVAL_K_VALUES)
        records.append(build_record(example, result, metrics, k_max, method))
    runtime_s = time.perf_counter() - started

    aggregate = evaluation.aggregate_retrieval_metrics(
        [record["metrics"] for record in records], config.RETRIEVAL_K_VALUES
    )
    meta: dict[str, Any] = {
        "stage": f"{method} retrieval benchmark",
        "retrieval_method": method,
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


def _progress(items: Sequence[Any], *, quiet: bool, method: str = "dense") -> Sequence[Any]:
    """Wrap the question list in a progress bar when tqdm is available."""
    if quiet:
        return items
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items, desc=f"{method} retrieval", unit="q")


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
    parser = argparse.ArgumentParser(description="Retrieval benchmark (no LLM calls).")
    parser.add_argument(
        "--method",
        choices=sorted(RETRIEVER_CLASSES),
        default="dense",
        help="retrieval method to benchmark",
    )
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--k", type=int, default=config.RETRIEVAL_MAX_K, help="max K to rank")
    parser.add_argument("--results-file", type=Path, default=None)
    parser.add_argument("--meta-file", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true", help="no progress bar")
    args = parser.parse_args(argv)

    meta = run_benchmark(
        method=args.method,
        limit=args.limit,
        k_max=args.k,
        results_file=args.results_file,
        meta_file=args.meta_file,
        quiet=args.quiet,
    )

    retriever = meta["retriever"]
    latency = meta["latency_s"]
    print(f"\n{args.method.upper()} retrieval benchmark (retrieval only, no LLM calls)")
    print(f"  questions        {meta['n_questions']}")
    if args.method == "bm25":
        print(
            f"  implementation   {retriever['implementation']} {retriever['library_version']} "
            f"(k1={retriever['k1']}, b={retriever['b']}, epsilon={retriever['epsilon']})"
        )
    else:
        print(f"  embedding model  {retriever['embedding_model']} ({retriever['embedding_device']})")
        print(f"  index            {retriever['index']}, {retriever['similarity']}")
    print(f"  dataset sha256   {meta['dataset']['sha256']}")
    print(f"  runtime_s        {meta['runtime_s']} (model load {meta['model_load_s']}s)")
    print("\nRetrieval vs HotpotQA supporting facts")
    print(evaluation.format_retrieval_summary(meta["metrics"]))
    if args.method == "bm25":
        breakdown = (
            f"index+score mean {latency['search']['mean']:.4f}, "
            f"index+score p95 {latency['search']['p95']:.4f}"
        )
    else:
        breakdown = (
            f"encode mean {latency['encode']['mean']:.4f}, "
            f"search mean {latency['search']['mean']:.4f}"
        )
    print(
        f"\nLatency per question (s): {breakdown}, "
        f"retrieval mean {latency['retrieval']['mean']:.4f}, "
        f"retrieval total {latency['retrieval']['total']:.2f}"
    )
    print(f"\nraw results      {meta['results_file']}")
    print(f"metadata         {args.meta_file or METHOD_PATHS[args.method][1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
