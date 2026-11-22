"""Evidence-preserving context builder (PandaRAG component E).

Selects the smallest context that keeps the evidence the system actually found,
under the frozen 4,000-token budget. Priorities, in order:

1. one covering passage per supported requirement (dependency order first);
2. one best candidate for requirements that remain uncovered — the generator is
   handed the best available material for the gap rather than nothing;
3. remaining passages by rerank score, up to the bounded window ``base_top_k``.

The window matters: a context containing only protected passages degenerates into a
one-document prompt (measured: 1.7 documents/question, EM 0.448 on HotpotQA — worse
than plain RAG), because requirement coverage is not a claim of answer sufficiency.
Evidence selection protects passages; it does not shrink the context below the
retrieval budget.

Two failure modes that must not happen, and are tested:

* dropping a covered requirement's evidence because it ranks low lexically — a
  bridge passage often has low overlap with the question;
* exceeding the token budget by silently truncating the requirement list. On
  overflow this builder drops the lowest-priority *passages* and records every
  drop, so the record shows exactly what was sacrificed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.pipelines.pandarag.coverage import Coverage
from src.pipelines.pandarag.evidence_planner import Requirement
from src.pipelines.shared.generator import estimate_tokens
from src.pipelines.shared.retriever import RetrievedDoc

BRIDGE_TEXT_LIMIT = 1800  # keep a bridge passage readable without unbounded growth


@dataclass
class ContextSelection:
    documents: list[dict[str, Any]]
    requirement_documents: dict[str, list[str]]
    tokens_estimate: int
    dropped: list[dict[str, str]] = field(default_factory=list)
    truncated: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_documents": len(self.documents),
            "document_ids": [document["doc_id"] for document in self.documents],
            "requirement_documents": self.requirement_documents,
            "tokens_estimate": self.tokens_estimate,
            "dropped_for_budget": self.dropped,
            "truncated_documents": self.truncated,
        }


def _priority_order(
    requirements: Sequence[Requirement],
    coverage: Coverage,
    candidates: Sequence[RetrievedDoc],
    base_top_k: int = 10,
) -> list[tuple[RetrievedDoc, list[str]]]:
    """Order candidates by requirement attachment, then rerank position."""
    by_id = {document.doc_id: document for document in candidates}
    chosen: list[tuple[RetrievedDoc, list[str]]] = []
    seen: set[str] = set()

    def take(doc_id: str, requirement_id: str) -> None:
        document = by_id.get(doc_id)
        if document is None or doc_id in seen:
            return
        seen.add(doc_id)
        chosen.append((document, [requirement_id]))

    # 1. Covering evidence for supported requirements, dependency order first.
    for requirement in requirements:
        item = coverage.per_requirement.get(requirement.id)
        if item is None or not item.covered:
            continue
        if requirement.depends_on:
            for parent in requirement.depends_on:
                parent_item = coverage.per_requirement.get(parent)
                if parent_item is not None:
                    for doc_id in parent_item.supporting_document_ids:
                        take(doc_id, parent)
        for doc_id in item.supporting_document_ids:
            take(doc_id, requirement.id)

    # 2. Best available candidate for requirements still uncovered.
    for requirement in requirements:
        item = coverage.per_requirement.get(requirement.id)
        if item is not None and item.covered:
            continue
        for document in candidates:
            if document.doc_id not in seen:
                seen.add(document.doc_id)
                chosen.append((document, [requirement.id]))
                break

    # 3. Fill up to the bounded context window with the best remaining passages.
    # The window keeps this a RAG context rather than a single-passage prompt:
    # evidence selection decides which passages are protected, the reranker decides
    # the rest, up to the same budget SAC-RAG uses.
    for document in candidates:
        if len(chosen) >= base_top_k:
            break
        if document.doc_id in seen:
            continue
        seen.add(document.doc_id)
        chosen.append((document, []))

    return chosen


def build_context(
    requirements: Sequence[Requirement],
    coverage: Coverage,
    candidates: Sequence[RetrievedDoc],
    *,
    max_tokens: int = 4000,
    min_passages_per_requirement: int = 1,
    base_top_k: int = 10,
) -> ContextSelection:
    """Assemble the generator context within the token budget."""
    ordered = _priority_order(requirements, coverage, candidates, base_top_k=base_top_k)
    requirement_documents: dict[str, list[str]] = {}
    documents: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    truncated: list[str] = []

    def current_tokens(candidate_documents: Sequence[Mapping[str, Any]]) -> int:
        if not candidate_documents:
            return estimate_tokens("Context:\n(none)")
        rendered = "\n\n".join(
            f"[Document {index}]\nTitle: {document['title']}\n{document['text']}"
            for index, document in enumerate(candidate_documents, start=1)
        )
        return estimate_tokens("Context:\n" + rendered)

    for document, requirement_ids in ordered:
        entry = {"doc_id": document.doc_id, "title": document.title, "text": document.text}
        proposed = documents + [entry]
        if current_tokens(proposed) <= max_tokens:
            documents = proposed
        else:
            # A single passage can blow the whole budget; trim this passage first,
            # then drop it only if even the trimmed version does not fit.
            trimmed_text = document.text[:BRIDGE_TEXT_LIMIT]
            entry_trimmed = {"doc_id": document.doc_id, "title": document.title, "text": trimmed_text}
            if current_tokens(documents + [entry_trimmed]) <= max_tokens:
                documents = documents + [entry_trimmed]
                truncated.append(document.doc_id)
            else:
                dropped.append(
                    {
                        "doc_id": document.doc_id,
                        "title": document.title,
                        "reason": "context token budget",
                    }
                )
                continue
        for requirement_id in requirement_ids:
            requirement_documents.setdefault(requirement_id, [])
            if document.doc_id not in requirement_documents[requirement_id]:
                requirement_documents[requirement_id].append(document.doc_id)

    # Guarantee the stated minimum, even if the budget forced everything else out.
    if min_passages_per_requirement:
        for requirement in requirements:
            if requirement_documents.get(requirement.id):
                continue
            for document in candidates:
                if document.doc_id in {item["doc_id"] for item in documents}:
                    continue
                entry = {"doc_id": document.doc_id, "title": document.title, "text": document.text[:BRIDGE_TEXT_LIMIT]}
                if current_tokens(documents + [entry]) <= max_tokens:
                    documents.append(entry)
                    truncated.append(document.doc_id)
                    requirement_documents.setdefault(requirement.id, []).append(document.doc_id)
                    break

    # Selection is driven by evidence coverage; ORDER is driven by the reranker.
    # Putting a requirement-attached bridge passage above a better-ranked passage
    # discards the relevance signal for no benefit (and empirically hurt answers).
    rank_of = {document.doc_id: index for index, document in enumerate(candidates)}
    documents.sort(key=lambda item: rank_of.get(item["doc_id"], len(rank_of)))

    return ContextSelection(
        documents=documents,
        requirement_documents=requirement_documents,
        tokens_estimate=current_tokens(documents),
        dropped=dropped,
        truncated=truncated,
    )
