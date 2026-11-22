"""SAC-RAG type-specific retrieval strategies.

Paper (Table 3 and §III-C):

* **Fact**      — BM25 top-10 + dense top-10 -> union/dedupe -> BGE rerank -> top-10
* **Definition**— dense only, top-8
* **Reasoning** — decompose -> dense top-10 per sub-question -> union/dedupe ->
  BGE rerank -> top-10

Deviations from Table 3, both recorded in the reproduction protocol:

* Definition **does** rerank in this implementation (the execution directive
  specifies a reranker for the Definition path); Table 3 lists the reranker for
  Fact/Reasoning only. Configurable via ``configs/sac_rag.json``.
* The hybrid Fact path fuses by union-and-rerank exactly as the paper describes,
  so the paper's unvalued ``lambda`` never enters the primary path.

Every strategy returns a :class:`StrategyResult` carrying the candidate pool, the
reranked list and the selected context, so the record can show *why* a document
reached the generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from src.pipelines.shared.retriever import RetrievalService, RetrievedDoc


@dataclass
class StrategyResult:
    strategy: str
    queries: list[str]
    candidates: list[RetrievedDoc]
    ranked: list[RetrievedDoc]
    selected: list[RetrievedDoc]
    retrieval_executions: int
    rerank_latency_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "queries": list(self.queries),
            "n_candidates": len(self.candidates),
            "n_ranked": len(self.ranked),
            "selected_doc_ids": [document.doc_id for document in self.selected],
            "retrieval_executions": self.retrieval_executions,
            "rerank_latency_seconds": round(self.rerank_latency_seconds, 4),
            **self.extra,
        }


def _merge(service: RetrievalService, groups: Sequence[Sequence[RetrievedDoc]]) -> list[RetrievedDoc]:
    """Union by doc_id, keeping the first (best-ranked) occurrence and its provenance."""
    merged: list[RetrievedDoc] = []
    seen: set[str] = set()
    for group in groups:
        for document in group:
            if document.doc_id in seen:
                continue
            seen.add(document.doc_id)
            merged.append(document)
    return merged


def retrieve_fact(
    service: RetrievalService,
    question: str,
    *,
    dense_top_k: int = 10,
    bm25_top_k: int = 10,
    final_k: int = 10,
    rerank: bool = True,
) -> StrategyResult:
    dense = service.search(question, "dense", dense_top_k)
    bm25 = service.search(question, "bm25", bm25_top_k)
    executions = 2
    candidates = _merge(service, [dense, bm25])

    if rerank:
        ranked = service.rerank(question, candidates, top_k=None)
        rerank_latency = service._reranker.latency_seconds if service._reranker else 0.0
    else:
        ranked = sorted(candidates, key=lambda document: (-document.rank, document.doc_id))
        rerank_latency = 0.0

    return StrategyResult(
        strategy="fact",
        queries=[question],
        candidates=candidates,
        ranked=ranked,
        selected=ranked[:final_k],
        retrieval_executions=executions,
        rerank_latency_seconds=rerank_latency,
        extra={"n_dense": len(dense), "n_bm25": len(bm25), "rerank": rerank},
    )


def retrieve_definition(
    service: RetrievalService,
    question: str,
    *,
    dense_top_k: int = 8,
    final_k: int = 8,
    rerank: bool = True,
) -> StrategyResult:
    dense = service.search(question, "dense", dense_top_k)
    candidates = list(dense)
    if rerank:
        ranked = service.rerank(question, candidates, top_k=None)
    else:
        ranked = sorted(candidates, key=lambda document: (-document.rank, document.doc_id))
    return StrategyResult(
        strategy="definition",
        queries=[question],
        candidates=candidates,
        ranked=ranked,
        selected=ranked[:final_k],
        retrieval_executions=1,
        rerank_latency_seconds=service._reranker.latency_seconds if (rerank and service._reranker) else 0.0,
        extra={"n_dense": len(dense), "rerank": rerank},
    )


def retrieve_reasoning(
    service: RetrievalService,
    question: str,
    subquestions: Sequence[str],
    *,
    dense_top_k: int = 10,
    final_k: int = 10,
    rerank: bool = True,
) -> StrategyResult:
    queries = [item for item in subquestions if item.strip()] or [question]
    per_query = [service.search(query, "dense", dense_top_k) for query in queries]
    candidates = _merge(service, per_query)

    if rerank:
        ranked = service.rerank(question, candidates, top_k=None)
        rerank_latency = service._reranker.latency_seconds if service._reranker else 0.0
    else:
        ranked = sorted(candidates, key=lambda document: (-document.rank, document.doc_id))
        rerank_latency = 0.0

    return StrategyResult(
        strategy="reasoning",
        queries=list(queries),
        candidates=candidates,
        ranked=ranked,
        selected=ranked[:final_k],
        retrieval_executions=len(queries),
        rerank_latency_seconds=rerank_latency,
        extra={
            "n_candidates_after_merge": len(candidates),
            "per_query_counts": [len(group) for group in per_query],
            "rerank": rerank,
        },
    )


def retrieve_for_type(
    service: RetrievalService,
    question: str,
    semantic_type: str,
    *,
    config: dict[str, Any],
    decompose_fn=None,
) -> StrategyResult:
    """Dispatch on the classifier label, using the frozen strategy parameters."""
    fact = config["fact"]
    definition = config["definition"]
    reasoning = config["reasoning"]

    if semantic_type == "Fact":
        return retrieve_fact(
            service,
            question,
            dense_top_k=int(fact["dense_top_k"]),
            bm25_top_k=int(fact["bm25_top_k"]),
            final_k=int(fact["final_k"]),
            rerank=bool(fact["rerank"]),
        )
    if semantic_type == "Definition":
        return retrieve_definition(
            service,
            question,
            dense_top_k=int(definition["dense_top_k"]),
            final_k=int(definition["final_k"]),
            rerank=bool(definition["rerank"]),
        )
    subquestions = decompose_fn(question) if decompose_fn else [question]
    return retrieve_reasoning(
        service,
        question,
        subquestions,
        dense_top_k=int(reasoning["dense_top_k"]),
        final_k=int(reasoning["final_k"]),
        rerank=bool(reasoning["rerank"]),
    )
