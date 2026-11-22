"""Corpus-matched baselines: P1c, P2c, P3c, P1n.

Why these exist (decision D3): the legacy P0–P3 pipelines retrieve from each
question's ten distractor paragraphs, where Recall@10 is structurally 1.000. P4–P6
retrieve from the pooled corpus. Comparing across two different retrieval problems
would confound every claim, so the baselines are re-run on the *same* corpus.

Their frozen predecessors are untouched; these are new systems in a new namespace.

* **P1c — Standard Dense RAG (corpus):** dense top-5 -> generate.
* **P2c — Hybrid RAG (corpus):** dense+BM25 RRF top-5 -> generate.
* **P3c — Transferred Adaptive RAG:** the original frozen K=3/K=5 confidence policy
  applied to corpus retrieval. The formula, weights (0.5/0.5), threshold (0.5),
  agreement range [2,20] and margin normaliser are imported from
  :mod:`src.pipelines.adaptive_rag` unchanged — nothing is re-tuned. Only the input
  changed, which is why this is labelled a *transferred-policy* experiment and not
  the original system.
* **P1n — paper-specified naive baseline:** dense top-10 retrieved, top-5 fed to the
  generator. Included only because SAC-RAG's Table 3 specifies exactly that
  configuration; it is not a claim about any other baseline.

One transfer detail had to be defined: the frozen policy reads ``dense_rank`` /
``bm25_rank`` of the top-1 fused document, which are always present in the legacy
artifact. On live corpus retrieval a fused document can be absent from one source's
top-10, so a missing source rank is set to ``depth + 1`` (11) — the worst rank a
fused candidate can have — and recorded per question.
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence

from src import config
from src.datasets import QAExample
from src.experiment_config import fingerprint, load_config
from src.llm import DeepSeekClient
from src.pipelines import adaptive_rag as frozen_policy
from src.pipelines.shared.generator import ContextBlock, generate_answer
from src.pipelines.shared.llm_json import Usage
from src.pipelines.shared.retriever import RetrievalService, RetrievedDoc
from src.pipelines.shared.runner import RunSpec, run_pipeline

PIPELINES = {
    "P0": "baseline_no_rag",
    "P1c": "baseline_dense_corpus",
    "P2c": "baseline_hybrid_corpus",
    "P3c": "baseline_adaptive_transferred",
    "P1n": "baseline_dense_paper_naive",
    "P2c10": "baseline_hybrid_corpus_depth10",
}
MISSING_SOURCE_RANK = config.RRF_DEPTH + 1


def _to_block(documents: Sequence[RetrievedDoc]) -> list[dict[str, Any]]:
    return [
        {"doc_id": document.doc_id, "title": document.title, "text": document.text}
        for document in documents
    ]


def _emit(
    example: QAExample,
    documents: Sequence[RetrievedDoc],
    context_documents: Sequence[RetrievedDoc],
    client: DeepSeekClient,
    usage: Usage,
    service: RetrievalService,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    block = ContextBlock(documents=_to_block(context_documents))
    generation = generate_answer(client, example.question, block, usage=usage)
    payload: dict[str, Any] = {
        "prediction": generation.prediction,
        "retrieved_documents": [document.as_record() for document in documents],
        "context_documents": [
            {"doc_id": document.doc_id, "title": document.title} for document in context_documents
        ],
        "context_text": block.text,
        "context_tokens_estimate": block.tokens_estimate,
        "retrieval_rounds": 1,
        "retrieval_executions": 1,
        "rerank_latency_seconds": 0.0,
        "generation_latency_seconds": generation.latency_seconds,
        "usage": usage,
        "error": generation.error,
        "baseline": {
            "strategy": "corpus",
            "context_doc_ids": [document.doc_id for document in context_documents],
            **(extra or {}),
        },
    }
    return payload


def run_p1c(example: QAExample, service: RetrievalService, client: DeepSeekClient) -> dict[str, Any]:
    usage = Usage()
    retrieved = service.search(example.question, "dense", 5)
    return _emit(example, retrieved, retrieved, client, usage, service, extra={"k": 5, "mode": "dense"})


def run_p2c(example: QAExample, service: RetrievalService, client: DeepSeekClient) -> dict[str, Any]:
    usage = Usage()
    retrieved = service.search(example.question, "hybrid", 5)
    return _emit(
        example, retrieved, retrieved, client, usage, service, extra={"k": 5, "mode": "hybrid_rrf"}
    )


def run_p1n(example: QAExample, service: RetrievalService, client: DeepSeekClient) -> dict[str, Any]:
    usage = Usage()
    retrieved = service.search(example.question, "dense", 10)
    fed = retrieved[:5]
    return _emit(
        example,
        retrieved,
        fed,
        client,
        usage,
        service,
        extra={"k_retrieved": 10, "k_fed": 5, "mode": "dense"},
    )


def run_p3c(example: QAExample, service: RetrievalService, client: DeepSeekClient) -> dict[str, Any]:
    """Transferred frozen policy: identical formula, corpus retrieval input."""
    usage = Usage()
    fused = service.search(example.question, "hybrid", 10)

    dense_ranking = service.search(example.question, "dense", config.RRF_DEPTH)
    bm25_ranking = service.search(example.question, "bm25", config.RRF_DEPTH)
    dense_rank_of = {document.doc_id: document.rank for document in dense_ranking}
    bm25_rank_of = {document.doc_id: document.rank for document in bm25_ranking}

    top1 = fused[0]
    d1 = dense_rank_of.get(top1.doc_id, MISSING_SOURCE_RANK)
    b1 = bm25_rank_of.get(top1.doc_id, MISSING_SOURCE_RANK)

    agreement = frozen_policy.agreement(d1, b1)
    s3 = fused[2].retrieval_score if len(fused) >= 3 else None
    s4 = fused[3].retrieval_score if len(fused) >= 4 else None
    margin_raw, margin = frozen_policy.boundary_margin(len(fused), s3, s4)
    final_k, confidence = frozen_policy.decide_final_k(agreement, margin)

    context_documents = fused[:final_k]
    return _emit(
        example,
        fused,
        context_documents,
        client,
        usage,
        service,
        extra={
            "k_initial": config.ADAPTIVE_RAG_INITIAL_K,
            "final_k": final_k,
            "confidence": round(confidence, 6),
            "threshold": config.ADAPTIVE_RAG_THRESHOLD,
            "agreement": round(agreement, 6),
            "dense_rank_top1": d1,
            "bm25_rank_top1": b1,
            "boundary_margin": round(margin, 6),
            "boundary_margin_raw": round(margin_raw, 6),
            "policy": "transferred_frozen_adaptive_rag/1.0",
            "policy_source": "src.pipelines.adaptive_rag (unmodified import)",
        },
    )


def run_p0(example: QAExample, service: RetrievalService, client: DeepSeekClient) -> dict[str, Any]:
    """Closed-book baseline in the Scope-B namespace.

    The frozen legacy artifact (``results/no_rag.jsonl``) is left untouched: it was
    produced by a different prompt and only covers HotpotQA.
    """
    usage = Usage()
    block = ContextBlock(documents=[])
    generation = generate_answer(client, example.question, block, usage=usage)
    return {
        "prediction": generation.prediction,
        "retrieved_documents": [],
        "context_documents": [],
        "context_text": block.text,
        "context_tokens_estimate": block.tokens_estimate,
        "retrieval_rounds": 0,
        "retrieval_executions": 0,
        "rerank_latency_seconds": 0.0,
        "generation_latency_seconds": generation.latency_seconds,
        "usage": usage,
        "error": generation.error,
        "baseline": {"strategy": "closed_book", "k": 0},
    }


def run_p2c10(example: QAExample, service: RetrievalService, client: DeepSeekClient) -> dict[str, Any]:
    """Hybrid baseline with the fusion depth the other systems use.

    P2c (kept, archived) fused source lists already truncated to five. Every other
    hybrid in this project — P3c, P4, P5, P6 and the Phase-1 benchmark — fuses the
    top-10 of each source and *then* truncates. This variant does the same: retrieve
    10, feed 5, so both retrieval and context are directly comparable.
    """
    usage = Usage()
    retrieved = service.search(example.question, "hybrid", config.RRF_DEPTH)
    fed = retrieved[:5]
    return _emit(
        example,
        retrieved,
        fed,
        client,
        usage,
        service,
        extra={
            "k_retrieved": config.RRF_DEPTH,
            "k_fed": 5,
            "mode": "hybrid_rrf_depth10",
            "fusion_depth": config.RRF_DEPTH,
            "corrects": "P2c (fusion depth 5, kept as results/baseline_hybrid_corpus)",
        },
    )


RUNNERS = {
    "P0": run_p0,
    "P1c": run_p1c,
    "P2c": run_p2c,
    "P2c10": run_p2c10,
    "P3c": run_p3c,
    "P1n": run_p1n,
}


def build(
    baseline: str,
    *,
    dataset: str,
    split: str | None = None,
    limit: int | None = None,
    concurrency: int = 4,
    quiet: bool = False,
) -> dict[str, Any]:
    if baseline not in RUNNERS:
        raise SystemExit(f"unknown baseline {baseline!r}; choose from {sorted(RUNNERS)}")
    pipeline_name = PIPELINES[baseline]
    retrieval_config = load_config(config.REPO_ROOT / "configs" / "retrieval" / "pooled_corpus.json")

    spec = RunSpec(
        pipeline=pipeline_name,
        dataset=dataset,
        split=split,
        limit=limit,
        concurrency=concurrency,
        config_fingerprint=fingerprint(retrieval_config),
        extra_meta={
            "pipeline_family": "closed_book" if baseline == "P0" else "corpus_baseline",
            "baseline_id": baseline,
            "resolved_retrieval_config": retrieval_config,
            "note": (
                "Corpus-matched baseline. Not comparable to the legacy per-question "
                "artifacts in results/*.jsonl; see docs/pandarag/phase1_report.md."
            ),
        },
    )
    results_file = config.RESULTS_DIR / pipeline_name / f"{dataset}.jsonl"
    spec.results_file = results_file
    spec.meta_file = results_file.with_suffix(".meta.json")
    return run_pipeline(spec, RUNNERS[baseline], quiet=quiet)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Corpus-matched baselines")
    parser.add_argument("--baseline", required=True, choices=sorted(RUNNERS))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    build(
        args.baseline,
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        concurrency=args.concurrency,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
