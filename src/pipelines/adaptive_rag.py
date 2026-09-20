"""Adaptive RAG: frozen hybrid ranking + rule-based K=3/K=5 decision -> DeepSeek.

Pipeline::

    question -> frozen hybrid artifact -> top-3 -> A/M/confidence
    -> K=3 if confidence >= 0.5 else K=5 -> context -> DeepSeek -> answer

FROZEN policy (pre-registered; do not change after seeing results):
initial K=3, A = 1-((d1+b1)-2)/18, M = (s3-s4)/W (W = 2/61-2/70, else 0.0
if pool < 4), confidence = 0.5*A + 0.5*M, threshold 0.5. ``confidence`` is
an ordinal heuristic, NOT a probability. Exactly ONE DeepSeek call per
question; no LLM classifier/judge, no recomputed retrieval.

Usage::

    python -m src.pipelines.adaptive_rag --limit 10   # smoke test
    python -m src.pipelines.adaptive_rag              # full 500-question run
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

INITIAL_K = config.ADAPTIVE_RAG_INITIAL_K
ALLOWED_FINAL_K = tuple(config.ADAPTIVE_RAG_ALLOWED_FINAL_K)
W_A = config.ADAPTIVE_RAG_AGREEMENT_WEIGHT
W_M = config.ADAPTIVE_RAG_MARGIN_WEIGHT
THRESHOLD = config.ADAPTIVE_RAG_THRESHOLD
RANK_MIN = config.ADAPTIVE_RAG_AGREEMENT_RANK_MIN
RANK_MAX = config.ADAPTIVE_RAG_AGREEMENT_RANK_MAX
W_NORM = config.ADAPTIVE_RAG_MARGIN_NORMALIZER_W
RETRIEVAL_METHOD = "hybrid_adaptive"

# Frozen import surface: the frozen hybrid artifact is consumed directly.
_PROTOCOL_FORBIDDEN_MODULES = (
    "rank_bm25",
    "faiss",
    "sentence_transformers",
    "src.retrieval",
    "src.retrieval_benchmark",
    "src.pipelines.standard_rag",
    "src.pipelines.hybrid_rag",
    "src.pipelines.no_rag",
)

_EXPECTED_DATASET_SHA256 = "956405ceb43d73385d60898e1bfc7f45a8184c9a4e1eff45b9a8ec239acfd048"
EXPECTED_RRF_CONSTANT = 60.0
EXPECTED_RRF_DEPTH = 10

def _hit_value(hit: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if hit.get(name) is not None:
            return hit[name]
    return None


def agreement(dense_rank_top1: int, bm25_rank_top1: int) -> float:
    """Source agreement A = 1 - ((d1+b1) - 2) / 18 (theory range 2..20)."""
    rank_sum = int(dense_rank_top1) + int(bm25_rank_top1)
    return 1.0 - ((rank_sum - RANK_MIN) / (RANK_MAX - RANK_MIN))


def boundary_margin(pool_size: int, s3: float | None, s4: float | None) -> tuple[float, float]:
    """Boundary margin (M_raw, M); M = (s3-s4)/W if pool >= 4 else 0.0."""
    if int(pool_size) < 4 or s3 is None or s4 is None:
        return 0.0, 0.0
    raw = float(s3) - float(s4)
    return raw, raw / W_NORM


def confidence_score(a: float, m: float) -> float:
    """Frozen confidence = 0.5*A + 0.5*M (heuristic, NOT a probability)."""
    return W_A * float(a) + W_M * float(m)


def decide_final_k(a: float, m: float, threshold: float = THRESHOLD) -> tuple[int, float]:
    """Return (final_k, confidence); confidence >= threshold -> K=3."""
    conf = confidence_score(a, m)
    return (3 if conf >= threshold else 5), conf


def decide_adaptive_k(retrieved_documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Decide final K from retrieval-only fused hits (no gold leakage possible).

    Accepts ONLY the frozen ``retrieved_documents`` list (rank, rrf_score,
    dense_rank, bm25_rank); the full HotpotQA record is never passed here.
    Uses at most the top-5 fused hits plus pool size.
    """
    hits = [dict(h) for h in retrieved_documents]
    pool_size = len(hits)
    if pool_size == 0:
        raise ValueError("Cannot decide K: empty retrieved_documents.")
    ordered = sorted(hits, key=lambda h: int(h["rank"]))[:5]
    top1 = ordered[0]
    d1 = _hit_value(top1, "dense_rank")
    b1 = _hit_value(top1, "bm25_rank")
    if d1 is None or b1 is None:
        raise ValueError("Top-1 fused hit is missing dense/bm25 rank provenance.")
    a = agreement(int(d1), int(b1))
    s3 = float(_hit_value(ordered[2], "rrf_score", "score")) if len(ordered) >= 3 else None
    s4 = float(_hit_value(ordered[3], "rrf_score", "score")) if len(ordered) >= 4 else None
    raw, m = boundary_margin(pool_size, s3, s4)
    final_k, conf = decide_final_k(a, m)
    return {
        "pool_size": pool_size,
        "dense_rank_top1": int(d1),
        "bm25_rank_top1": int(b1),
        "agreement": a,
        "rrf_score_rank3": s3,
        "rrf_score_rank4": s4,
        "boundary_margin_raw": raw,
        "boundary_margin_normalized": m,
        "confidence": conf,
        "threshold": THRESHOLD,
        "final_k": final_k,
        "reason": f"K={final_k}: confidence {'>=' if final_k == 3 else '<'} threshold",
    }

def load_hybrid_records(path: Path | None = None) -> list[dict[str, Any]]:
    records_path = Path(path or config.HYBRID_RESULTS_FILE)
    with records_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_retrieval_artifact(records: Sequence[Mapping[str, Any]], expected_sha256: str = _EXPECTED_DATASET_SHA256) -> dict[str, dict[str, Any]]:
    meta = json.loads(config.HYBRID_RESULTS_META_FILE.read_text(encoding="utf-8"))
    if meta.get("dataset", {}).get("sha256") != expected_sha256:
        raise ValueError("Hybrid artifact dataset fingerprint mismatch.")
    if meta.get("retrieval_method", "hybrid") != "hybrid":
        raise ValueError("Hybrid retrieval artifact has unexpected retrieval method.")
    fusion = meta.get("retriever", {}).get("fusion", {})
    if float(fusion.get("rrf_constant", -1)) != EXPECTED_RRF_CONSTANT:
        raise ValueError("Hybrid artifact RRF constant mismatch.")
    if int(fusion.get("rrf_depth", -1)) != EXPECTED_RRF_DEPTH:
        raise ValueError("Hybrid artifact RRF depth mismatch.")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        qid = str(record["question_id"])
        if qid in by_id:
            raise ValueError(f"Duplicate question_id in hybrid artifact: {qid!r}.")
        if str(record.get("retrieval_method", "hybrid")) != "hybrid":
            raise ValueError(f"Hybrid record {qid!r} has unexpected retrieval method.")
        if "5" not in record.get("metrics", {}) or "3" not in record.get("metrics", {}):
            raise ValueError(f"Hybrid record {qid!r} is missing K=3/K=5 metrics.")
        by_id[qid] = dict(record)
    return by_id


def check_frozen_policy() -> dict[str, Any]:
    if INITIAL_K != 3:
        raise ValueError(f"Adaptive initial K must be frozen at 3 (got {INITIAL_K}).")
    if tuple(ALLOWED_FINAL_K) != (3, 5):
        raise ValueError("Adaptive allowed final K must be frozen at (3, 5).")
    if W_A != 0.5 or W_M != 0.5:
        raise ValueError("Adaptive weights must be frozen at 0.5/0.5.")
    if THRESHOLD != 0.5:
        raise ValueError("Adaptive threshold must be frozen at 0.5.")
    if (RANK_MIN, RANK_MAX) != (2, 20):
        raise ValueError("Adaptive agreement range must be frozen at [2, 20].")
    if abs(W_NORM - ((2.0 / 61.0) - (2.0 / 70.0))) > 1e-15:
        raise ValueError("Adaptive margin normalizer W mismatch.")
    if float(config.RRF_CONSTANT) != EXPECTED_RRF_CONSTANT:
        raise ValueError("RRF constant must be frozen at 60.")
    if int(config.RRF_DEPTH) != EXPECTED_RRF_DEPTH:
        raise ValueError("RRF depth must be frozen at 10.")
    return {"initial_k": INITIAL_K, "allowed_final_k": [3, 5], "agreement_weight": W_A, "margin_weight": W_M, "threshold": THRESHOLD, "agreement_rank_range": [RANK_MIN, RANK_MAX], "margin_normalizer_W": W_NORM, "rrf_constant": EXPECTED_RRF_CONSTANT, "rrf_depth": EXPECTED_RRF_DEPTH}


def top_k_hits(record: Mapping[str, Any], k: int) -> list[dict[str, Any]]:
    if k not in (3, 5):
        raise ValueError(f"Adaptive RAG final K must be 3 or 5; got K={k}.")
    return [dict(h) for h in list(record.get("retrieved_documents", []))[:k]]


def format_context(hits: Sequence[Mapping[str, Any]], paragraphs: Mapping[str, str]) -> str:
    blocks: list[str] = []
    for position, hit in enumerate(hits, start=1):
        title = str(hit["title"])
        if title not in paragraphs:
            raise KeyError(f"Retrieved title {title!r} not in dataset context.")
        blocks.append(f"[Document {position}]\nTitle: {title}\n{paragraphs[title]}")
    return "Context:\n" + "\n\n".join(blocks) if blocks else "Context:\n(none)"


def build_prompt(question: str, context_block: str) -> str:
    return (f"Question: {question.strip()}\n\n{context_block}\n\nAnswer the question using the supplied context.\nAnswer:")

def build_record(
    example: dataset.HotpotExample,
    retrieval_record: Mapping[str, Any],
    decision: Mapping[str, Any],
    response: LLMResponse,
    *,
    client: DeepSeekClient,
) -> dict[str, Any]:
    """Map example + frozen retrieval + adaptive decision + LLM response."""
    final_k = int(decision["final_k"])
    if final_k not in (3, 5):
        raise ValueError(f"Adaptive final K must be 3 or 5; got {final_k}.")
    hits = top_k_hits(retrieval_record, final_k)
    paragraphs = {t: example.paragraph_text(t) for t in example.context_titles}
    retrieved_docs: list[dict[str, Any]] = []
    for hit in hits:
        title = str(hit["title"])
        doc: dict[str, Any] = {
            "title": title,
            "rank": int(hit["rank"]),
            "score": float(hit.get("score", hit.get("rrf_score"))),
            "rrf_score": float(hit.get("rrf_score", hit.get("score"))),
            "text": paragraphs[title],
        }
        doc["dense_rank"] = int(hit["dense_rank"]) if hit.get("dense_rank") is not None else None
        doc["bm25_rank"] = int(hit["bm25_rank"]) if hit.get("bm25_rank") is not None else None
        if hit.get("dense_score") is not None:
            doc["dense_score"] = float(hit["dense_score"])
        if hit.get("bm25_score") is not None:
            doc["bm25_score"] = float(hit["bm25_score"])
        retrieved_docs.append(doc)
    metrics = retrieval_record["metrics"][str(final_k)]
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
        "retrieval_k": final_k,
        "initial_k": INITIAL_K,
        "final_k": final_k,
        "pool_size": int(decision["pool_size"]),
        "actual_retrieved_docs": len(retrieved_docs),
        "retrieved_doc_count": len(retrieved_docs),
        "adaptive_confidence": float(decision["confidence"]),
        "adaptive_threshold": float(decision["threshold"]),
        "adaptive_decision": str(decision["reason"]),
        "adaptive_signals": {
            "dense_rank_top1": decision["dense_rank_top1"],
            "bm25_rank_top1": decision["bm25_rank_top1"],
            "agreement": float(decision["agreement"]),
            "rrf_score_rank3": decision["rrf_score_rank3"],
            "rrf_score_rank4": decision["rrf_score_rank4"],
            "boundary_margin_raw": float(decision["boundary_margin_raw"]),
            "boundary_margin_normalized": float(decision["boundary_margin_normalized"]),
        },
        "adaptive_reason": str(decision["reason"]),
        "adaptive_policy": {
            "initial_k": INITIAL_K,
            "allowed_final_k": [3, 5],
            "agreement_weight": W_A,
            "margin_weight": W_M,
            "threshold": THRESHOLD,
            "agreement_rank_range": [RANK_MIN, RANK_MAX],
            "margin_normalizer_W": W_NORM,
            "rrf_constant": EXPECTED_RRF_CONSTANT,
            "rrf_depth": EXPECTED_RRF_DEPTH,
        },
        "retrieved_docs": retrieved_docs,
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
    """Aggregate run metadata incl. K=3/K=5 split and efficiency stats."""
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
    if retrieval_meta is None:
        retrieval_meta = json.loads(config.HYBRID_RESULTS_META_FILE.read_text(encoding="utf-8"))
    k3 = [r for r in records if int(r["final_k"]) == 3]
    k5 = [r for r in records if int(r["final_k"]) == 5]
    doc_recalls = [float(r["retrieval_doc_recall"]) for r in records if r["retrieval_doc_recall"] is not None]
    sent_recalls = [float(r["retrieval_sentence_recall"]) for r in records if r["retrieval_sentence_recall"] is not None]
    total_in = sum(int(r["input_tokens"]) for r in records)
    total_out = sum(int(r["output_tokens"]) for r in records)
    total_cached = sum(int(r["cached_input_tokens"]) for r in records)
    total_cost = round(sum(float(r["api_cost_usd"]) for r in records), 8)
    return {
        "pipeline": "adaptive_rag",
        "retrieval_method": RETRIEVAL_METHOD,
        "retrieval_source": str(config.HYBRID_RESULTS_FILE),
        "retrieval_artifact": str(config.HYBRID_RESULTS_FILE),
        "retrieval_artifact_sha256": retrieval_meta.get("dataset", {}).get("sha256"),
        "retrieval_fingerprint": retrieval_meta.get("dataset"),
        "retrieval_config": retrieval_meta.get("retriever"),
        "rrf_constant": EXPECTED_RRF_CONSTANT,
        "rrf_depth": EXPECTED_RRF_DEPTH,
        "initial_k": INITIAL_K,
        "allowed_final_k": [3, 5],
        "agreement_weight": W_A,
        "margin_weight": W_M,
        "threshold": THRESHOLD,
        "agreement_rank_range": [RANK_MIN, RANK_MAX],
        "margin_normalizer_W": W_NORM,
        "policy_description": (
            "Initial K=3 over the frozen hybrid ranking; escalate to K=5 iff "
            "confidence < 0.5 where confidence = 0.5*A + 0.5*M, "
            "A = 1-((d1+b1)-2)/18, M = (s3-s4)/W (pool>=4 else 0), "
            "W = 2/61-2/70. Confidence is a heuristic, NOT a probability. "
            "Actual docs = min(final_k, pool_size). One DeepSeek call per question."
        ),
        "retrieval_metrics": {
            "document_recall": round(mean(doc_recalls), 6) if doc_recalls else 0.0,
            "sentence_recall": round(mean(sent_recalls), 6) if sent_recalls else 0.0,
            "document_complete_rate": round(mean([1.0 if r["retrieval_doc_complete"] else 0.0 for r in records]), 6) if records else 0.0,
        },
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
        "k3_count": len(k3),
        "k5_count": len(k5),
        "k3_percentage": round(100.0 * len(k3) / len(records), 4) if records else 0.0,
        "k5_percentage": round(100.0 * len(k5) / len(records), 4) if records else 0.0,
        "mean_final_k": round(mean([int(r["final_k"]) for r in records]), 6) if records else 0.0,
        "mean_retrieved_docs": round(mean([int(r["actual_retrieved_docs"]) for r in records]), 6) if records else 0.0,
        "total_retrieved_docs": sum(int(r["actual_retrieved_docs"]) for r in records),
        "from_cache": from_cache_count,
        "cache_stats": {
            "hits": from_cache_count,
            "misses": len(records) - from_cache_count,
            "hit_rate": round(from_cache_count / len(records), 6) if records else 0.0,
        },
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cached_input_tokens": total_cached,
        "total_tokens": total_in + total_out,
        "total_api_cost_usd": total_cost,
        "latency_s": latency,
        "mean_latency_seconds": latency["mean"],
        "median_latency_seconds": latency["median"],
        "p95_latency_seconds": latency["p95"],
        "total_latency_seconds": latency["total"],
        "answer_metrics_successful_only": successful_only,
        "answer_metrics_all": overall,
        "exact_match": overall["exact_match"],
        "f1": overall["f1"],
        "results_file": str(config.ADAPTIVE_RAG_RESULTS_FILE),
    }

def save_records(records: Sequence[Mapping[str, Any]], path: Path | None = None) -> Path:
    out_path = Path(path or config.ADAPTIVE_RAG_RESULTS_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def save_metadata(metadata: Mapping[str, Any], path: Path | None = None) -> Path:
    out_path = Path(path or config.ADAPTIVE_RAG_RESULTS_META_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def load_records(path: Path | None = None) -> list[dict[str, Any]]:
    records_path = Path(path or config.ADAPTIVE_RAG_RESULTS_FILE)
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
    """Run Adaptive RAG: frozen policy decides K, then one DeepSeek call."""
    check_frozen_policy()
    active_client = client or DeepSeekClient()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    raw = list(retrieval_records) if retrieval_records is not None else load_hybrid_records()
    retrieval_by_id = verify_retrieval_artifact(raw)
    retrieval_meta = json.loads(config.HYBRID_RESULTS_META_FILE.read_text(encoding="utf-8"))
    examples = dataset.load_dataset()
    if limit is not None:
        examples = examples[:limit]
    records: list[dict[str, Any]] = []
    total = len(examples)
    for index, example in enumerate(examples):
        retrieval_record = retrieval_by_id[example.example_id]
        decision = decide_adaptive_k(retrieval_record.get("retrieved_documents", []))
        final_k = int(decision["final_k"])
        hits = top_k_hits(retrieval_record, final_k)
        paragraphs = {t: example.paragraph_text(t) for t in example.context_titles}
        prompt = build_prompt(example.question, format_context(hits, paragraphs))
        response = active_client.generate(prompt, system=config.GENERATION_SYSTEM_PROMPT)
        records.append(build_record(example, retrieval_record, decision, response, client=active_client))
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
    parser = argparse.ArgumentParser(description="Adaptive RAG: frozen K=3/K=5 policy over frozen hybrid ranking -> DeepSeek.")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--results-file", type=Path, default=None)
    parser.add_argument("--meta-file", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args(argv)
    policy = check_frozen_policy()
    client = DeepSeekClient()
    settings = client.settings()
    print("Adaptive RAG (frozen policy: K=3 -> confidence<0.5 -> K=5; one LLM call)")
    print(f"  model            {settings['model']}")
    print(f"  thinking         {settings['thinking']}")
    print(f"  max_output       {settings['max_output_tokens']}")
    print(f"  temperature      {settings['temperature']}")
    print(f"  retrieval        {RETRIEVAL_METHOD} (frozen hybrid; RRF c={policy['rrf_constant']:g}, depth={policy['rrf_depth']})")
    print(f"  policy           w_A={policy['agreement_weight']} w_M={policy['margin_weight']} threshold={policy['threshold']}")
    print(f"  dataset sha256   {dataset.dataset_fingerprint()['sha256']}")
    metadata = run_pipeline(limit=args.limit, client=client, results_file=args.results_file, meta_file=args.meta_file, quiet=args.quiet)
    print(f"\nquestions        {metadata['n_questions']}")
    print(f"successful       {metadata['successful']} (failed {metadata['failed']})")
    print(f"K split          K=3:{metadata['k3_count']} ({metadata['k3_percentage']}%) / K=5:{metadata['k5_count']} ({metadata['k5_percentage']}%)")
    print(f"exact_match      {metadata['exact_match']:.4f}")
    print(f"f1               {metadata['f1']:.4f}")
    print(f"retrieval        doc_recall {metadata['retrieval_metrics']['document_recall']:.4f} / sent_recall {metadata['retrieval_metrics']['sentence_recall']:.4f} / complete {metadata['retrieval_metrics']['document_complete_rate']:.4f}")
    print(f"tokens           in {metadata['total_input_tokens']} / out {metadata['total_output_tokens']} (cached {metadata['total_cached_input_tokens']})")
    print(f"api_cost_usd     {metadata['total_api_cost_usd']:.8f}")
    print(f"llm_calls        {metadata['total_llm_calls']}")
    print(f"latency_s        mean {metadata['latency_s']['mean']:.4f} / median {metadata['latency_s']['median']:.4f} / p95 {metadata['latency_s']['p95']:.4f} / total {metadata['latency_s']['total']:.2f}")
    print(f"\nraw results      {args.results_file or config.ADAPTIVE_RAG_RESULTS_FILE}")
    print(f"metadata         {args.meta_file or config.ADAPTIVE_RAG_RESULTS_META_FILE}")
    return 0 if metadata["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
