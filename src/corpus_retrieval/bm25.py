"""Corpus-level BM25 retrieval.

One ``BM25Okapi`` index per ``(dataset, split)`` corpus, built over the same
documents as the dense index. The tokenizer and the ``k1``/``b``/``epsilon``
values are the frozen ones from :mod:`src.config` (identical to the legacy
experiment), so lexical retrieval is directly comparable across the two scopes.

BM25 is not serialisable in a stable way, so the index is rebuilt from the
persisted corpus + parameters at load time and verified against the corpus
fingerprint. Rebuild cost is milliseconds for a few thousand documents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rank_bm25 import BM25Okapi

from src import config
from src.datasets.base import Document
from src.retrieval import tokenize

CONFIG_FILE = "bm25_config.json"
DOCS_FILE = "documents.json"


@dataclass
class BM25Index:
    doc_ids: tuple[str, ...]
    corpus_sha256: str
    k1: float
    b: float
    epsilon: float
    tokenizer: str = "regex-lowercase-alphanumeric"
    _bm25: BM25Okapi | None = None

    def __post_init__(self) -> None:
        if self._bm25 is None:
            raise ValueError("BM25Index must be created with BM25Index.build or BM25Index.load")

    # -- construction -------------------------------------------------------
    @classmethod
    def build(
        cls,
        documents: Sequence[Document],
        *,
        corpus_sha256: str,
        k1: float = config.BM25_K1,
        b: float = config.BM25_B,
        epsilon: float = config.BM25_EPSILON,
    ) -> "BM25Index":
        ordered = sorted(documents, key=lambda document: document.doc_id)
        corpus = [tokenize(document.text_for_embedding) for document in ordered]
        bm25 = BM25Okapi(corpus, k1=k1, b=b, epsilon=epsilon)
        return cls(
            doc_ids=tuple(document.doc_id for document in ordered),
            corpus_sha256=corpus_sha256,
            k1=k1,
            b=b,
            epsilon=epsilon,
            _bm25=bm25,
        )

    # -- search -------------------------------------------------------------
    def search(self, query: str, k: int) -> tuple[tuple[str, ...], tuple[float, ...]]:
        if not self.doc_ids:
            return tuple(), tuple()
        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)
        k = max(1, min(int(k), len(self.doc_ids)))
        # Deterministic ordering: score desc, then doc_id asc (never rely on
        # the underlying argpartition tie order).
        order = sorted(range(len(self.doc_ids)), key=lambda i: (-float(scores[i]), self.doc_ids[i]))[:k]
        return (
            tuple(self.doc_ids[i] for i in order),
            tuple(float(scores[i]) for i in order),
        )

    # -- persistence --------------------------------------------------------
    def save(self, directory: Path, *, extra: dict | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / DOCS_FILE).write_text(json.dumps(list(self.doc_ids)), encoding="utf-8")
        payload = {
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "tokenizer": self.tokenizer,
            "corpus_sha256": self.corpus_sha256,
            "n_documents": len(self.doc_ids),
            "implementation": "rank_bm25.BM25Okapi",
            "rebuild_on_load": True,
        }
        payload.update(extra or {})
        (directory / CONFIG_FILE).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, documents: Sequence[Document]) -> "BM25Index":
        payload = json.loads((directory / CONFIG_FILE).read_text(encoding="utf-8"))
        return cls.build(
            documents,
            corpus_sha256=str(payload["corpus_sha256"]),
            k1=float(payload["k1"]),
            b=float(payload["b"]),
            epsilon=float(payload["epsilon"]),
        )
