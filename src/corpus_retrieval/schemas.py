"""Typed records shared by the corpus, index and retrieval layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from src.datasets.base import Document


@dataclass(frozen=True)
class FusedHit:
    """One fused candidate with per-source provenance preserved."""

    doc_id: str
    rrf_score: float
    source_ranks: Mapping[str, int] = field(default_factory=dict)
    source_scores: Mapping[str, float] = field(default_factory=dict)

    @property
    def rank(self) -> int:
        """Fusion rank is assigned by the caller; kept for symmetry with the legacy records."""
        return int(self.source_ranks.get("rrf", 0))


@dataclass(frozen=True)
class RankedList:
    """A ranked document list for one query from one retriever."""

    query: str
    method: str
    doc_ids: tuple[str, ...]
    scores: tuple[float, ...]

    def as_pairs(self) -> tuple[tuple[str, float], ...]:
        return tuple(zip(self.doc_ids, self.scores))


@dataclass(frozen=True)
class CorpusArtifact:
    """A built corpus: documents + provenance + identity."""

    dataset: str
    split: str
    dataset_sha256: str
    documents: tuple[Document, ...]
    provenance: Mapping[str, Mapping[str, int]]
    stats: Mapping[str, Any]

    @property
    def corpus_sha256(self) -> str:
        return corpus_fingerprint(self.documents)


@dataclass(frozen=True)
class IndexManifest:
    """Identity and configuration of a built index pair."""

    dataset: str
    split: str
    corpus_sha256: str
    n_documents: int
    dense: Mapping[str, Any]
    bm25: Mapping[str, Any]
    built_utc: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "corpus_sha256": self.corpus_sha256,
            "n_documents": self.n_documents,
            "dense": dict(self.dense),
            "bm25": dict(self.bm25),
            "built_utc": self.built_utc,
        }


def corpus_fingerprint(documents) -> str:
    """sha256 over ``doc_id\\x00title\\x00text`` in doc_id order.

    Order-independent (documents are sorted first) so two builds of the same
    content always agree, and any content change always changes the digest.
    """
    import hashlib

    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda d: d.doc_id):
        digest.update(document.doc_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(document.title.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(document.text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
