"""Rank fusion over corpus-level rankings.

The math is imported from the legacy module (:func:`src.retrieval.rrf_score`) so
the two scopes fuse identically. Fusion uses ranks only — dense cosine scores and
BM25 scores are on different scales and are never mixed.

Candidates are the union of the first ``depth`` items of each source ranking; an
item retrieved by only one source keeps ``None`` semantics by simply being absent
from ``source_ranks``. Ties are broken by ``doc_id`` so the order is
deterministic across runs and machines.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from src import config
from src.corpus_retrieval.schemas import FusedHit, RankedList
from src.retrieval import rrf_score


def fuse(
    rankings: Mapping[str, RankedList],
    *,
    depth: int = config.RRF_DEPTH,
    constant: float = config.RRF_CONSTANT,
    k: int | None = None,
) -> list[FusedHit]:
    """Reciprocal-rank fusion of ``{method: RankedList}``."""
    ranks: dict[str, dict[str, int]] = {}
    scores: dict[str, dict[str, float]] = {}
    doc_ids: set[str] = set()

    for method, ranking in rankings.items():
        source_ranks: dict[str, int] = {}
        source_scores: dict[str, float] = {}
        for position, (doc_id, score) in enumerate(ranking.as_pairs(), start=1):
            if position > depth:
                break
            # First occurrence wins: a duplicate inside one ranking is a bug,
            # and silently letting a later rank overwrite it would hide that bug.
            source_ranks.setdefault(doc_id, position)
            source_scores.setdefault(doc_id, float(score))
            doc_ids.add(doc_id)
        ranks[method] = source_ranks
        scores[method] = source_scores

    fused: list[FusedHit] = []
    for doc_id in doc_ids:
        per_source = {method: ranks[method][doc_id] for method in ranks if doc_id in ranks[method]}
        total = rrf_score(list(per_source.values()), constant=constant)
        fused.append(
            FusedHit(
                doc_id=doc_id,
                rrf_score=total,
                source_ranks=per_source,
                source_scores={m: scores[m][doc_id] for m in per_source if doc_id in scores[m]},
            )
        )

    fused.sort(key=lambda hit: (-hit.rrf_score, hit.doc_id))
    return fused[:k] if k is not None else fused


def fuse_pairs(
    *rankings: Sequence[tuple[str, float]],
    depth: int = config.RRF_DEPTH,
    constant: float = config.RRF_CONSTANT,
) -> list[tuple[str, float]]:
    """Convenience wrapper for ad-hoc ``(doc_id, score)`` sequences in tests."""
    named = {
        f"source{index}": RankedList(
            query="",
            method=f"source{index}",
            doc_ids=tuple(doc_id for doc_id, _ in ranking),
            scores=tuple(score for _, score in ranking),
        )
        for index, ranking in enumerate(rankings)
    }
    return [(hit.doc_id, hit.rrf_score) for hit in fuse(named, depth=depth, constant=constant)]
