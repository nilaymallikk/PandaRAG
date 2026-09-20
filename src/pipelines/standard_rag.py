"""Standard Dense RAG baseline: question -> frozen dense retrieval -> answer.

Pipeline::

    question -> frozen dense retrieval (exactly top-K=5) -> context
    -> DeepSeek (thinking disabled) -> answer

The dense ranking is NOT recomputed and no dense/BM25/RRF/adaptive code is
imported here: per-question rankings (titles, ranks, scores) and K=5 retrieval
metrics are loaded from the frozen artifact
``results/retrieval/dense_retrieval.jsonl``. Paragraph text comes from the
immutable dataset via :class:`src.dataset.HotpotExample.paragraph_text`, joined
to the ranking by title. Generation reuses the shared
``GENERATION_SYSTEM_PROMPT`` plus the deterministic per-question context block
defined below; faithfulness/hallucination is left null (deferred, no LLM
judge, no extra API call).

Usage::

    python -m src.pipelines.standard_rag --limit 10     # smoke test
    python -m src.pipelines.standard_rag                # full 500-question run

Artifacts::

    results/standard_rag.jsonl       raw per-question result records
    results/standard_rag.meta.json   run metadata + aggregate metrics
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


# K=5 is frozen before the run; supplying min(K, pool) documents when a
# candidate pool is smaller than K (5 of 500 questions) is a physical limit
# documented in metadata, not dynamic per-question K selection.
TOP_K = config.STANDARD_RAG_TOP_K
RETRIEVAL_METHOD = "dense"

# The import surface of this module is frozen by the protocol: BM25, FAISS,
# RRF, dense or adaptive-retrieval implementations must never be imported
# here. (Enforced statically by tests/test_standard_rag.py via AST
# inspection; there is intentionally no runtime sys.modules check because the
# shared test suite legitimately imports other retrieval modules first.)
_PROTOCOL_FORBIDDEN_MODULES = (
    "rank_bm25",
    "faiss",
    "sentence_transformers",
    "src.retrieval",
    "src.retrieval_benchmark",
    "src.pipelines.hybrid_rag",
    "src.pipelines.adaptive_rag",
)

_EXPECTED_DATASET_SHA256 = "956405ceb43d73385d60898e1bfc7f45a8184c9a4e1eff45b9a8ec239acfd048"


def load_dense_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Read the frozen dense retrieval artifact (JSONL; no retriever import)."""
    records_path = Path(path or config.DENSE_RESULTS_FILE)
    with records_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_retrieval_artifact(
    records: Sequence[Mapping[str, Any]],
    expected_sha256: str = _EXPECTED_DATASET_SHA256,
) -> dict[str, dict[str, Any]]:
    """Index frozen dense records by question id after frozen-artifact checks."""
    meta_path = config.DENSE_RESULTS_META_FILE
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Dense retrieval metadata not found: {meta_path}") from error
    sha256 = meta.get("dataset", {}).get("sha256")
    if sha256 != expected_sha256:
        raise ValueError("Dense artifact dataset fingerprint mismatch.")
    retriever = meta.get("retriever", {})
    if retriever.get("retrieval_method", "dense") != "dense":
        raise ValueError("Dense retrieval artifact has unexpected retrieval method.")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        question_id = str(record["question_id"])
        if question_id in by_id:
            raise ValueError(f"Duplicate question_id in dense artifact: {question_id!r}.")
        if "5" not in record.get("metrics", {}):
            raise ValueError(f"Dense record {question_id!r} is missing K=5 metrics.")
        by_id[question_id] = dict(record)
    return by_id


def top_k_hits(record: Mapping[str, Any], k: int = TOP_K) -> list[dict[str, Any]]:
    """Return the first ``k`` frozen dense paragraph hits, in frozen order."""
    if k != TOP_K:
        raise ValueError(f"Standard RAG uses exactly K={TOP_K}; got K={k}.")
    hits = list(record.get("retrieved_documents", []))
    return [dict(hit) for hit in hits[:k]]


def format_context(
    hits: Sequence[Mapping[str, Any]],
    paragraphs: Mapping[str, str],
) -> str:
    """Render the deterministic ``Context:`` block for the user prompt."""
    blocks: list[str] = []
    for position, hit in enumerate(hits, start=1):
        title = str(hit["title"])
        if title not in paragraphs:
            raise KeyError(f"Retrieved title {title!r} not in dataset context.")
        blocks.append(f"[Document {position}]\nTitle: {title}\n{paragraphs[title]}")
    return "Context:\n" + "\n\n".join(blocks) if blocks else "Context:\n(none)"


def build_prompt(question: str, context_block: str) -> str:
    """Frozen user message: the question plus the frozen top-K context block."""
    return (
        f"Question: {question.strip()}\n\n"
        f"{context_block}\n\n"
        "Answer the question using the supplied context.\n"
        "Answer:"
    )

def build_record(
    example: dataset.HotpotExample,
    retrieval_record: Mapping[str, Any],
    response: LLMResponse,
    *,
    client: DeepSeekClient,
    k: int = TOP_K,
) -> dict[str, Any]:
    """Map one example + its frozen K=5 retrieval + one LLM response."""
    if k != TOP_K:
        raise ValueError(f"Standard RAG uses exactly K={TOP_K}; got K={k}.")
    hits = top_k_hits(retrieval_record, k)
    paragraphs = {t: example.paragraph_text(t) for t in example.context_titles}
    retrieved_docs = [
        {
            "title": str(hit["title"]),
            "rank": int(hit["rank"]),
            "score": float(hit["score"]),
            "text": paragraphs[str(hit["title"])],
        }
        for hit in hits
    ]
    metrics = retrieval_record["metrics"][str(k)]
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
        "retrieval_method": RETRIEVAL_METHOD,
        "retrieval_k": k,
        "retrieved_docs": retrieved_docs,
        "retrieved_doc_count": len(retrieved_docs),
        "retrieval_doc_recall": float(metrics["document_recall"]),
        "retrieval_sentence_recall": float(metrics["sentence_recall"]),
        "retrieval_complete": bool(metrics["document_complete"]),
        "retrieval_doc_complete": bool(metrics["document_complete"]),
        "retrieval_sentence_complete": bool(metrics["sentence_complete"]),
        "faithfulness": None,
        "hallucination": None,
        "faithfulness_note": "deferred (no LLM judge; no extra API call)",
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
        "total": round(sum(ordered), 4),
    }

def build_metadata(
    records: Sequence[Mapping[str, Any]],
    *,
    client: DeepSeekClient,
    started_at: str,
    finished_at: str,
    runtime_s: float,
    retrieval_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate run metadata; faithfulness stays deferred (null)."""
    successful = [r for r in records if r["success"]]
    failed = [r for r in records if not r["success"]]
    overall = answer_eval.aggregate_answer_scores(
        [{"exact_match": r["exact_match"], "f1": r["f1"]} for r in records]
    )
    successful_only = answer_eval.aggregate_answer_scores(
        [{"exact_match": r["exact_match"], "f1": r["f1"]} for r in successful]
    )
    fingerprint = dataset.dataset_fingerprint()
    llm_settings = client.settings()
    latency = _latency_stats([float(r["latency_seconds"]) for r in records])
    from_cache_count = sum(1 for r in records if r["from_cache"])
    short_pool = sum(1 for r in records if int(r.get("retrieved_doc_count", TOP_K)) < TOP_K)
    if retrieval_meta is None:
        retrieval_meta = json.loads(config.DENSE_RESULTS_META_FILE.read_text(encoding="utf-8"))
    doc_recalls = [float(r["retrieval_doc_recall"]) for r in records if r["retrieval_doc_recall"] is not None]
    sent_recalls = [float(r["retrieval_sentence_recall"]) for r in records if r["retrieval_sentence_recall"] is not None]
    return {
        "pipeline": "standard_rag",
        "retrieval_method": RETRIEVAL_METHOD,
        "retrieval_k": TOP_K,
        "retrieval_artifact": str(config.DENSE_RESULTS_FILE),
        "retrieval_fingerprint": retrieval_meta.get("dataset"),
        "retrieval_config": retrieval_meta.get("retriever"),
        "retrieval_metrics_at_k": {
            "document_recall": round(mean(doc_recalls), 6) if doc_recalls else 0.0,
            "sentence_recall": round(mean(sent_recalls), 6) if sent_recalls else 0.0,
            "document_complete_rate": round(mean([1.0 if r["retrieval_doc_complete"] else 0.0 for r in records]), 6) if records else 0.0,
        },
        "retrieval_short_pool_questions": short_pool,
        "faithfulness": "deferred (no LLM judge; no extra API call)",
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
            "user_template": "Question: {question} + Context [Document i] + instruction + Answer:",
        },
        "timestamp": finished_at,
        "llm": llm_settings,
        "generation_prompt": {
            "system": config.GENERATION_SYSTEM_PROMPT,
            "user_template": "Question: {question} + Context [Document i] + instruction + Answer:",
        },
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "runtime_s": round(runtime_s, 4),
        "successful": len(successful),
        "failed": len(failed),
        "failed_ids": [r["question_id"] for r in failed],
        "total_llm_calls": sum(int(r["llm_calls"]) for r in records),
        "llm_calls": sum(int(r["llm_calls"]) for r in records),
        "llm_used": True,
        "from_cache": from_cache_count,
        "cache_stats": {
            "hits": from_cache_count,
            "misses": len(records) - from_cache_count,
            "hit_rate": round(from_cache_count / len(records), 6) if records else 0.0,
        },
        "total_input_tokens": sum(int(r["input_tokens"]) for r in records),
        "total_output_tokens": sum(int(r["output_tokens"]) for r in records),
        "total_cached_input_tokens": sum(int(r["cached_input_tokens"]) for r in records),
        "total_api_cost_usd": round(sum(float(r["api_cost_usd"]) for r in records), 8),
        "mean_input_tokens": (
            round(mean([int(r["input_tokens"]) for r in records]), 4)
            if records
            else 0.0
        ),
        "mean_output_tokens": (
            round(mean([int(r["output_tokens"]) for r in records]), 4)
            if records
            else 0.0
        ),
        "total_retrieved_docs": sum(
            int(r.get("retrieved_doc_count", 0)) for r in records
        ),
        "mean_retrieved_docs": (
            round(mean([int(r.get("retrieved_doc_count", 0)) for r in records]), 4)
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
        "results_file": str(config.STANDARD_RAG_RESULTS_FILE),
    }

def save_records(records: Sequence[Mapping[str, Any]], path: Path | None = None) -> Path:
    out_path = Path(path or config.STANDARD_RAG_RESULTS_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def save_metadata(metadata: Mapping[str, Any], path: Path | None = None) -> Path:
    out_path = Path(path or config.STANDARD_RAG_RESULTS_META_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def load_records(path: Path | None = None) -> list[dict[str, Any]]:
    records_path = Path(path or config.STANDARD_RAG_RESULTS_FILE)
    with records_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_pipeline(
    *,
    limit: int | None = None,
    client: DeepSeekClient | None = None,
    retrieval_records: Sequence[Mapping[str, Any]] | None = None,
    results_file: Path | None = None,
    meta_file: Path | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    active_client = client or DeepSeekClient()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    raw = list(retrieval_records) if retrieval_records is not None else load_dense_records(config.DENSE_RESULTS_FILE)
    retrieval_by_id = verify_retrieval_artifact(raw)
    retrieval_meta = json.loads(config.DENSE_RESULTS_META_FILE.read_text(encoding="utf-8"))
    examples = dataset.load_dataset()
    if limit is not None:
        examples = examples[:limit]
    records: list[dict[str, Any]] = []
    total = len(examples)
    for index, example in enumerate(examples):
        retrieval_record = retrieval_by_id[example.example_id]
        hits = top_k_hits(retrieval_record)
        paragraphs = {t: example.paragraph_text(t) for t in example.context_titles}
        prompt = build_prompt(example.question, format_context(hits, paragraphs))
        response = active_client.generate(prompt, system=config.GENERATION_SYSTEM_PROMPT)
        records.append(build_record(example, retrieval_record, response, client=active_client, k=TOP_K))
        if not quiet and (index + 1 == total or (index + 1) % 25 == 0):
            done = len([item for item in records if item["success"]])
            print(f"  progress {index + 1}/{total} (successful {done})", flush=True)
    runtime_s = time.perf_counter() - started
    finished_at = datetime.now(timezone.utc).isoformat()
    metadata = build_metadata(records, client=active_client, started_at=started_at, finished_at=finished_at, runtime_s=runtime_s, retrieval_meta=retrieval_meta)
    save_records(records, results_file)
    save_metadata(metadata, meta_file)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standard Dense RAG: frozen top-5 dense retrieval -> DeepSeek.")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--results-file", type=Path, default=None)
    parser.add_argument("--meta-file", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args(argv)
    if TOP_K != config.STANDARD_RAG_TOP_K or TOP_K != 5:
        raise SystemExit(f"Standard RAG requires K=5 (got TOP_K={TOP_K}).")
    client = DeepSeekClient()
    settings = client.settings()
    print("Standard Dense RAG (frozen top-5, no BM25/RRF/adaptive logic)")
    print(f"  model            {settings['model']}")
    print(f"  thinking         {settings['thinking']}")
    print(f"  max_output       {settings['max_output_tokens']}")
    print(f"  temperature      {settings['temperature']}")
    print(f"  retrieval        {RETRIEVAL_METHOD} K={TOP_K} (frozen artifact)")
    print(f"  dataset sha256   {dataset.dataset_fingerprint()['sha256']}")
    metadata = run_pipeline(limit=args.limit, client=client, results_file=args.results_file, meta_file=args.meta_file, quiet=args.quiet)
    print(f"\nquestions        {metadata['n_questions']}")
    print(f"successful       {metadata['successful']} (failed {metadata['failed']})")
    print(f"exact_match      {metadata['exact_match']:.4f}")
    print(f"f1               {metadata['f1']:.4f}")
    print(f"retrieval        doc_recall@5 {metadata['retrieval_metrics_at_k']['document_recall']:.4f} / sent_recall@5 {metadata['retrieval_metrics_at_k']['sentence_recall']:.4f} / complete@5 {metadata['retrieval_metrics_at_k']['document_complete_rate']:.4f}")
    print(f"tokens           in {metadata['total_input_tokens']} / out {metadata['total_output_tokens']} (cached {metadata['total_cached_input_tokens']})")
    print(f"api_cost_usd     {metadata['total_api_cost_usd']:.8f}")
    print(f"llm_calls        {metadata['total_llm_calls']}")
    print(f"latency_s        mean {metadata['latency_s']['mean']:.4f} / median {metadata['latency_s']['median']:.4f} / total {metadata['latency_s']['total']:.2f}")
    print(f"\nraw results      {args.results_file or config.STANDARD_RAG_RESULTS_FILE}")
    print(f"metadata         {args.meta_file or config.STANDARD_RAG_RESULTS_META_FILE}")
    return 0 if metadata["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
