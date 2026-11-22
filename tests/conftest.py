"""Fixtures shared by the Scope-B pipeline tests.

No network, no model weights: a fake LLM client and a fake retrieval service keep
the suite fast and deterministic while still exercising the real code paths
(schema validation, dedupe, budget arithmetic, prompt assembly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from src.llm import LLMResponse
from src.pipelines.shared.retriever import RetrievedDoc


class FakeClient:
    """Stands in for ``DeepSeekClient``; replays queued texts and records prompts."""

    def __init__(self, responses: Sequence[str] | None = None) -> None:
        self._responses = list(responses or ["{}"])
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.calls = 0

    @property
    def last_prompt(self) -> str:
        return self.prompts[-1] if self.prompts else ""

    def generate(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> LLMResponse:
        self.prompts.append(prompt)
        self.systems.append(system or "")
        text = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return LLMResponse(
            text=text,
            model="fake",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.0001,
            finish_reason="stop",
        )


@dataclass
class FakeService:
    """Minimal retrieval service: deterministic documents, no encoder, no reranker."""

    dataset: str = "hotpotqa"
    split: str = "validation"
    corpus_sha256: str = "corpus-sha"
    dataset_sha256: str = "dataset-sha"
    n_documents: int = 100
    encoder_name: str = "fake-encoder"
    reranker_name: str = "fake-reranker"
    _reranker: Any = None
    documents: dict[str, RetrievedDoc] = field(default_factory=dict)
    search_results: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    search_calls: list[tuple[str, str, int]] = field(default_factory=list)

    def add_documents(self, specs: Sequence[tuple[str, str, str]]) -> None:
        for doc_id, title, text in specs:
            self.documents[doc_id] = RetrievedDoc(doc_id=doc_id, title=title, text=text, rank=1)

    def search(self, query: str, mode: str, k: int = 10) -> list[RetrievedDoc]:
        self.search_calls.append((query, mode, k))
        explicit = self.search_results.get((query, mode))
        ids = explicit if explicit is not None else list(self.documents)
        results = []
        for position, doc_id in enumerate(ids[:k], start=1):
            document = self.documents[doc_id]
            results.append(
                RetrievedDoc(
                    doc_id=doc_id,
                    title=document.title,
                    text=document.text,
                    rank=position,
                    retrieval_score=1.0 / position,
                    source_ranks={mode: position},
                )
            )
        return results

    def rerank(self, query: str, documents: Sequence[RetrievedDoc], *, top_k: int | None = None):
        # Relevance = count of shared tokens, deterministic tie-break by doc_id.
        query_tokens = set(query.lower().split())

        def score(document: RetrievedDoc) -> int:
            return len(query_tokens & set(document.text.lower().split()))

        ordered = sorted(documents, key=lambda document: (-score(document), document.doc_id))
        ranked = [
            RetrievedDoc(
                doc_id=document.doc_id,
                title=document.title,
                text=document.text,
                rank=position,
                rerank_score=float(score(document)),
                retrieval_score=document.retrieval_score,
                source_ranks=dict(document.source_ranks),
            )
            for position, document in enumerate(ordered, start=1)
        ]
        return ranked[:top_k] if top_k is not None else ranked

    def dedupe(self, documents):
        seen, out = set(), []
        for document in documents:
            if document.doc_id in seen:
                continue
            seen.add(document.doc_id)
            out.append(document)
        return out
