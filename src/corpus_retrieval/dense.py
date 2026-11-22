"""Corpus-level dense retrieval (FAISS, exact inner product on L2-normalised vectors).

Differences from the legacy per-question retriever in :mod:`src.retrieval`:

* one index per ``(dataset, split)`` corpus instead of one per question, so
  Recall@K is a real measurement;
* the index is persisted with the document order and the corpus fingerprint, so
  doc ids retrieved later can be verified against the corpus that produced them;
* the encoder is injectable (:class:`Encoder`), which keeps unit tests offline and
  fast — production uses :class:`SentenceTransformerEncoder` (bge-small-en-v1.5).

Metric: cosine similarity. Vectors are L2-normalised, so FAISS ``IndexFlatIP``
returns exactly the cosine score. Exact (flat) search, not ANN — the corpora are
a few thousand documents, so there is no quality/speed trade-off to hide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import faiss
import numpy as np

from src import config
from src.datasets.base import Document

DOCS_FILE = "documents.json"
VECTORS_FILE = "dense.npy"
CONFIG_FILE = "dense_config.json"


class Encoder(Protocol):
    """Anything that turns texts into an ``(n, dim)`` float32 matrix."""

    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Production encoder: ``BAAI/bge-small-en-v1.5`` (the project's frozen embedding model)."""

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        *,
        device: str = config.EMBEDDING_DEVICE,
        query_instruction: str = config.EMBEDDING_QUERY_INSTRUCTION,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._query_instruction = query_instruction
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        # sentence-transformers >=6 renamed the method; keep both paths without
        # pinning the project to a single minor version.
        getter = getattr(self._model, "get_embedding_dimension", None)
        if getter is None:
            getter = self._model.get_sentence_embedding_dimension
        return int(getter())

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                [self._query_instruction + text],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0],
            dtype=np.float32,
        )


@dataclass
class DenseIndex:
    """A persisted exact dense index over one corpus."""

    doc_ids: tuple[str, ...]
    vectors: np.ndarray
    encoder_name: str
    dimension: int
    corpus_sha256: str
    _index: Any = None

    def __post_init__(self) -> None:
        if self.vectors.shape[0] != len(self.doc_ids):
            raise ValueError("dense index: vector count does not match document count")
        if self._index is None:
            self._index = faiss.IndexFlatIP(int(self.dimension))
            self._index.add(self.vectors)

    # -- construction -------------------------------------------------------
    @classmethod
    def build(
        cls,
        documents: Sequence[Document],
        encoder: Encoder,
        *,
        corpus_sha256: str,
        batch_size: int = 256,
    ) -> "DenseIndex":
        ordered = sorted(documents, key=lambda document: document.doc_id)
        texts = [document.text_for_embedding for document in ordered]
        chunks = []
        for start in range(0, len(texts), batch_size):
            chunks.append(encoder.encode_documents(texts[start : start + batch_size]))
        vectors = np.vstack(chunks).astype(np.float32) if chunks else np.zeros((0, encoder.dimension), np.float32)
        return cls(
            doc_ids=tuple(document.doc_id for document in ordered),
            vectors=vectors,
            encoder_name=encoder.name,
            dimension=int(vectors.shape[1]) if vectors.size else int(encoder.dimension),
            corpus_sha256=corpus_sha256,
        )

    # -- search -------------------------------------------------------------
    def search_vector(self, vector: np.ndarray, k: int) -> tuple[tuple[str, ...], tuple[float, ...]]:
        if self.vectors.shape[0] == 0:
            return tuple(), tuple()
        k = max(1, min(int(k), self.vectors.shape[0]))
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        scores, indices = self._index.search(query, k)
        doc_ids = tuple(self.doc_ids[int(i)] for i in indices[0] if int(i) >= 0)
        values = tuple(float(s) for s in scores[0][: len(doc_ids)])
        return doc_ids, values

    def search(self, encoder: Encoder, query: str, k: int) -> tuple[tuple[str, ...], tuple[float, ...]]:
        return self.search_vector(encoder.encode_query(query), k)

    # -- persistence --------------------------------------------------------
    def save(self, directory: Path, *, extra: dict[str, Any] | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / VECTORS_FILE, self.vectors)
        (directory / DOCS_FILE).write_text(json.dumps(list(self.doc_ids)), encoding="utf-8")
        payload = {
            "encoder": self.encoder_name,
            "dimension": self.dimension,
            "corpus_sha256": self.corpus_sha256,
            "n_documents": len(self.doc_ids),
            "metric": "cosine",
            "index_type": "faiss.IndexFlatIP",
            "normalized": True,
        }
        payload.update(extra or {})
        (directory / CONFIG_FILE).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "DenseIndex":
        config_payload = json.loads((directory / CONFIG_FILE).read_text(encoding="utf-8"))
        doc_ids = tuple(json.loads((directory / DOCS_FILE).read_text(encoding="utf-8")))
        vectors = np.load(directory / VECTORS_FILE)
        return cls(
            doc_ids=doc_ids,
            vectors=np.asarray(vectors, dtype=np.float32),
            encoder_name=str(config_payload["encoder"]),
            dimension=int(config_payload["dimension"]),
            corpus_sha256=str(config_payload["corpus_sha256"]),
        )
