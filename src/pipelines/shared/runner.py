"""Per-question execution loop, record schema and run metadata.

Every Scope-B pipeline calls :func:`run_pipeline`. It owns the parts that must be
identical across systems:

* loading the frozen dataset split and the frozen corpus;
* the result record schema (``schema_version: 2``), including answer metrics and
  document-level Recall@K / all-gold@K computed from the ranked list retrieval
  actually produced;
* per-question token/cost/latency accounting, separating generation from
  auxiliary LLM work;
* resumable JSONL writing (a crash never loses completed questions);
* a ``meta.json`` carrying dataset/corpus/config/prompt fingerprints.

Pipelines only supply ``question_fn(example, service, client) -> dict`` returning
their own fields (``prediction``, ``context_documents``, ``retrieved_documents``,
``usage``, plus any pipeline-specific payload).
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable, Sequence

from src import answer_eval, config
from src.datasets import QAExample, get_adapter
from src.evaluation import recall_at_k
from src.llm import DeepSeekClient
from src.pipelines.shared.generator import prompt_fingerprint
from src.pipelines.shared.llm_json import Usage
from src.pipelines.shared.retriever import RetrievalService

SCHEMA_VERSION = 2
K_VALUES = (1, 3, 5, 10)


@dataclass
class RunSpec:
    pipeline: str
    dataset: str
    split: str | None = None
    limit: int | None = None
    concurrency: int = 4
    results_file: Path | None = None
    meta_file: Path | None = None
    config_fingerprint: str | None = None
    extra_meta: dict[str, Any] = field(default_factory=dict)
    resume: bool = True
    #: when set, only these question ids run (used by reduced-sample ablations)
    sample_ids: set[str] | None = None


QuestionFn = Callable[[QAExample, RetrievalService, DeepSeekClient], dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pct(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "mean": round(mean(ordered), 4),
        "median": round(median(ordered), 4),
        "p90": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))], 4),
        "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 4),
        "max": round(ordered[-1], 4),
    }


def build_record(
    *,
    spec: RunSpec,
    example: QAExample,
    payload: dict[str, Any],
    service: RetrievalService,
    semantic_type: str | None,
    prompt_fp: str,
    latency_seconds: float,
) -> dict[str, Any]:
    """Assemble the canonical Scope-B result record."""
    usage: Usage = payload.get("usage") or Usage()
    retrieved: list[dict[str, Any]] = payload.get("retrieved_documents") or []
    ranked_ids = [document["doc_id"] for document in retrieved][: max(K_VALUES)]
    gold_ids = list(example.gold_doc_ids)
    gold_set = set(gold_ids)

    answer_metrics = {
        "exact_match": answer_eval.exact_match(payload.get("prediction"), example.answer),
        "f1": answer_eval.token_f1(payload.get("prediction"), example.answer),
    }
    retrieval_metrics = {
        f"recall@{k}": (None if not gold_set else recall_at_k(ranked_ids, gold_set, k))
        for k in K_VALUES
    }
    retrieval_metrics["all_gold@10"] = (
        None if not gold_set else float(gold_set.issubset(set(ranked_ids[:10])))
    )
    context_documents: list[dict[str, Any]] = payload.get("context_documents") or []

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": spec.pipeline,
        "dataset": example.dataset,
        "split": example.split,
        "question_id": example.example_id,
        "question": example.question,
        "question_type": example.question_type,
        "semantic_type": semantic_type,
        "gold_answer": example.answer,
        "prediction": payload.get("prediction", ""),
        "gold_document_ids": sorted(gold_ids),
        "gold_evidence_kind": example.gold_evidence[0].kind if example.gold_evidence else None,
        "n_gold_documents": len(gold_set),
        **answer_metrics,
        **retrieval_metrics,
        "retrieved_documents": retrieved,
        "retrieved_doc_ids": ranked_ids,
        "n_retrieved": len(ranked_ids),
        "candidate_document_ids": payload.get("candidate_document_ids", ranked_ids),
        "context_documents": context_documents,
        "context_text": payload.get("context_text"),
        "n_context_documents": len(context_documents),
        "context_tokens_estimate": payload.get("context_tokens_estimate"),
        "retrieval_rounds": payload.get("retrieval_rounds", 0),
        "retrieval_executions": payload.get("retrieval_executions"),
        "retrieval_latency_seconds": round(float(payload.get("retrieval_latency_seconds", 0.0)), 4),
        "rerank_latency_seconds": round(float(payload.get("rerank_latency_seconds", 0.0)), 4),
        "generation_latency_seconds": round(float(payload.get("generation_latency_seconds", 0.0)), 4),
        "total_latency_seconds": round(latency_seconds, 4),
        **usage.as_dict(),
        "corpus_sha256": service.corpus_sha256,
        "dataset_sha256": service.dataset_sha256,
        "config_fingerprint": spec.config_fingerprint,
        "prompt_fingerprint": prompt_fp,
        "success": payload.get("error") is None,
        "error": payload.get("error"),
        "from_cache": usage.live_calls == 0 and usage.llm_calls > 0,
        "timestamp_utc": utc_now(),
    }
    for key in ("sac_rag", "pandarag", "baseline"):
        if key in payload:
            record[key] = payload[key]
    return record


def _aggregate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in records if row.get("success")]
    def _mean_of(key: str) -> float | None:
        values = [row[key] for row in ok if row.get(key) is not None]
        return round(mean(values), 4) if values else None

    doc_tokens = [row["input_tokens"] + row["output_tokens"] for row in ok]
    summary = {
        "n_questions": len(records),
        "n_success": len(ok),
        "n_failed": len(records) - len(ok),
        "exact_match": _mean_of("exact_match"),
        "f1": _mean_of("f1"),
        "recall@1": _mean_of("recall@1"),
        "recall@3": _mean_of("recall@3"),
        "recall@5": _mean_of("recall@5"),
        "recall@10": _mean_of("recall@10"),
        "all_gold@10": _mean_of("all_gold@10"),
        "mean_input_tokens": _mean_of("input_tokens"),
        "mean_output_tokens": _mean_of("output_tokens"),
        "mean_total_tokens": round(mean(doc_tokens), 1) if doc_tokens else None,
        "total_api_cost_usd": round(sum(row["api_cost_usd"] for row in ok), 6),
        # Cache accounting: a replayed run must never be reported as live spend.
        "total_api_cost_usd_live": round(
            sum(row["api_cost_usd"] for row in ok if row.get("live_llm_calls", 0) > 0), 6
        ),
        "n_fully_cached": sum(1 for row in ok if row.get("from_cache")),
        "n_with_live_calls": sum(1 for row in ok if row.get("live_llm_calls", 0) > 0),
        "mean_api_cost_usd": round(mean([row["api_cost_usd"] for row in ok]), 8) if ok else None,
        "mean_llm_calls": _mean_of("llm_calls"),
        "mean_latency_seconds": _mean_of("total_latency_seconds"),
        "latency_seconds": _pct([row["total_latency_seconds"] for row in ok]),
        "mean_retrieval_executions": _mean_of("retrieval_executions"),
        "mean_context_documents": _mean_of("n_context_documents"),
        "mean_context_tokens_estimate": _mean_of("context_tokens_estimate"),
        "mean_retrieval_rounds": _mean_of("retrieval_rounds"),
    }
    return summary


def _read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line after a crash is ignored, not repaired
    return rows


def run_pipeline(
    spec: RunSpec,
    question_fn: QuestionFn,
    *,
    service_factory: Callable[[], RetrievalService] | None = None,
    semantic_types: dict[str, Any] | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run one pipeline over one dataset split and write JSONL + meta."""
    adapter = get_adapter(spec.dataset, spec.split)
    examples = sorted(adapter.load(), key=lambda example: example.example_id)
    if spec.sample_ids is not None:
        examples = [example for example in examples if example.example_id in spec.sample_ids]
    if spec.limit:
        examples = examples[: int(spec.limit)]

    if semantic_types is None:
        from src.datasets import load_semantic_types

        semantic_types = load_semantic_types(spec.dataset)

    results_file = spec.results_file or (config.RESULTS_DIR / spec.pipeline / f"{spec.dataset}.jsonl")
    meta_file = spec.meta_file or results_file.with_suffix(".meta.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    existing = _read_existing(results_file) if spec.resume else []
    done_ids = {row["question_id"] for row in existing if row.get("success")}
    todo = [example for example in examples if example.example_id not in done_ids]

    client = DeepSeekClient()
    service = (service_factory or (lambda: RetrievalService(spec.dataset, spec.split)))()
    prompt_fp = prompt_fingerprint()
    write_lock = threading.Lock()
    started = time.perf_counter()
    started_utc = utc_now()

    def process(example: QAExample) -> dict[str, Any]:
        question_started = time.perf_counter()
        try:
            payload = question_fn(example, service, client)
            error = payload.get("error")
        except Exception as exc:  # a pipeline crash must be recorded, not hidden
            payload = {"prediction": "", "usage": Usage(), "error": f"{type(exc).__name__}: {exc}"}
            error = payload["error"]
        latency = time.perf_counter() - question_started
        semantic = semantic_types.get(example.example_id)
        semantic_label = None
        if isinstance(semantic, dict):
            semantic_label = semantic.get("type")
        elif semantic is not None:
            semantic_label = str(semantic)
        record = build_record(
            spec=spec,
            example=example,
            payload=payload,
            service=service,
            semantic_type=semantic_label,
            prompt_fp=prompt_fp,
            latency_seconds=latency,
        )
        if error:
            record["success"] = False
        return record

    records = list(existing)
    completed = 0
    mode = "a" if existing else "w"
    with results_file.open(mode, encoding="utf-8") as sink:
        if spec.concurrency <= 1:
            for example in todo:
                record = process(example)
                with write_lock:
                    sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    sink.flush()
                records.append(record)
                completed += 1
                if not quiet and completed % 25 == 0:
                    print(f"  {spec.pipeline}/{spec.dataset}: {completed}/{len(todo)}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=spec.concurrency) as pool:
                futures = {pool.submit(process, example): example for example in todo}
                for future in as_completed(futures):
                    record = future.result()
                    with write_lock:
                        sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        sink.flush()
                    records.append(record)
                    completed += 1
                    if not quiet and completed % 25 == 0:
                        print(f"  {spec.pipeline}/{spec.dataset}: {completed}/{len(todo)}", flush=True)

    runtime = time.perf_counter() - started
    # One row per question: a resumed run appends a fresh row for a previously
    # failed question, so fold the file before aggregating (prefer the success).
    folded: dict[str, dict[str, Any]] = {}
    for row in records:
        key = str(row["question_id"])
        current = folded.get(key)
        if current is None or (row.get("success") and not current.get("success")):
            folded[key] = row
    aggregate = _aggregate(list(folded.values()))
    aggregate["n_raw_records"] = len(records)
    aggregate["n_unique_questions"] = len(folded)
    meta = {
        "pipeline": spec.pipeline,
        "n_sample_ids": len(spec.sample_ids) if spec.sample_ids is not None else None,
        "dataset": spec.dataset,
        "split": adapter.split,
        "schema_version": SCHEMA_VERSION,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "runtime_s": round(runtime, 2),
        "n_requested": len(examples),
        "n_processed_this_run": completed,
        "resumed_from_existing": len(existing),
        "concurrency": spec.concurrency,
        "aggregate": aggregate,
        "dataset_fingerprint": adapter.fingerprint().as_dict(),
        "corpus": {
            "sha256": service.corpus_sha256,
            "n_documents": service.n_documents,
        },
        "retrieval": {
            "encoder": service.encoder_name,
            "reranker": service.reranker_name,
            "rrf_constant": config.RRF_CONSTANT,
            "rrf_depth": config.RRF_DEPTH,
        },
        "llm": {
            "model": config.DEEPSEEK_MODEL,
            "model_version": config.DEEPSEEK_MODEL_VERSION,
            "temperature": config.TEMPERATURE,
            "prompt_fingerprint": prompt_fp,
            "system_prompt": config.GENERATION_SYSTEM_PROMPT,
            "session_header": dict(config.LLM_EXTRA_HEADERS),
        },
        "config_fingerprint": spec.config_fingerprint,
        "results_file": str(results_file.relative_to(config.REPO_ROOT)),
        **spec.extra_meta,
    }
    meta_file.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    if not quiet:
        print(
            f"{spec.pipeline}/{spec.dataset}: n={aggregate['n_questions']} "
            f"EM={aggregate['exact_match']} F1={aggregate['f1']} "
            f"R@10={aggregate['recall@10']} cost=${aggregate['total_api_cost_usd']} "
            f"in {runtime/60:.1f}min",
            flush=True,
        )
    return meta
