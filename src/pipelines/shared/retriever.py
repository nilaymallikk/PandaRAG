"""Shared retrieval service for Scope-B pipelines.

Wraps the Phase-1 corpus (frozen) and indexes with three capabilities the
pipelines need and none of them should re-implement:

* Dense / BM25 / hybrid search over the pooled corpus, with per-query caching
  inside one process.
* Cross-encoder reranking (``BAAI/bge-reranker-base``) with a persistent score
  cache, so P4/P5/P6 do not each pay for the same query–document pair.
* Deterministic ordering and full provenance (source ranks and scores are
  preserved, never overwritten by the reranker).

Retrieval never sees gold. ``RetrievedDoc`` carries only what retrieval produced;
gold lives in the evaluation record.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from src import config
from src.corpus_retrieval import corpus as corpus_module
from src.corpus_retrieval import fusion as fusion_module
from src.corpus_retrieval.dense import DenseIndex, SentenceTransformerEncoder
from src.corpus_retrieval.bm25 import BM25Index
from src.corpus_retrieval.index_manager import load_indexes
from src.corpus_retrieval.schemas import RankedList

RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_CACHE_DIR = config.CACHE_DIR / "rerank"
RERANK_CACHE_FILE = "scores.jsonl"


@dataclass(frozen=True)
class RetrievedDoc:
    doc_id: str
    title: str
    text: str
    rank: int
    rerank_score: float | None = None
    retrieval_score: float | None = None
    source_ranks: dict[str, int] = field(default_factory=dict)

    def as_record(self, *, include_text: bool = True, max_chars: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "doc_id": self.doc_id,
            "title": self.title,
            "rank": self.rank,
            "rerank_score": None if self.rerank_score is None else round(float(self.rerank_score), 6),
            "retrieval_score": None if self.retrieval_score is None else round(float(self.retrieval_score), 6),
            "source_ranks": dict(self.source_ranks),
        }
        if include_text:
            text = self.text
            if max_chars is not None:
                text = text[:max_chars]
            payload["text"] = text
        return payload


class Reranker:
    """BGE cross-encoder with a persistent (query, doc) score cache."""

    def __init__(self, model_name: str = RERANK_MODEL, *, device: str = "cpu", batch_size: int = 64) -> None:
        self.model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()
        self._cache: dict[str, float] = {}
        self._cache_loaded = False
        self._cache_path = RERANK_CACHE_DIR / _slug(model_name) / RERANK_CACHE_FILE
        self.n_scored = 0
        self.n_cache_hits = 0
        self.latency_seconds = 0.0

    # -- cache --------------------------------------------------------------
    def _key(self, query: str, doc_id: str, text: str) -> str:
        payload = "\x00".join((self.model_name, query, doc_id, hashlib.sha1(text.encode()).hexdigest()))
        return hashlib.sha1(payload.encode()).hexdigest()

    def _load_cache(self) -> None:
        if self._cache_loaded:
            return
        self._cache_loaded = True
        if not self._cache_path.exists():
            return
        with self._cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                self._cache[row["key"]] = float(row["score"])

    def _append_cache(self, entries: Iterable[tuple[str, float]]) -> None:
        RERANK_CACHE_DIR.joinpath(_slug(self.model_name)).mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("a", encoding="utf-8") as handle:
            for key, score in entries:
                handle.write(json.dumps({"key": key, "score": score}) + "\n")

    # -- scoring ------------------------------------------------------------
    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self.model_name, device=self._device, max_length=512)
        return self._model

    def score(self, query: str, documents: Sequence[RetrievedDoc]) -> list[float]:
        """Return a relevance score per document (higher = more relevant)."""
        self._load_cache()
        keys = [self._key(query, document.doc_id, document.text) for document in documents]
        missing = [(index, document) for index, document in enumerate(documents) if keys[index] not in self._cache]

        if missing:
            pairs = [(query, document.text[:4000]) for _, document in missing]
            started = time.perf_counter()
            model = self._ensure_model()
            # One lock only around model construction; forward passes are batched by
            # sentence-transformers and torch inference is thread-safe, so scoring
            # can overlap with other threads' LLM calls instead of serialising them.
            raw = model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)
            elapsed = time.perf_counter() - started
            self.latency_seconds += elapsed
            new_entries = []
            for (index, _), value in zip(missing, raw):
                score = float(value)
                self._cache[keys[index]] = score
                new_entries.append((keys[index], score))
            self._append_cache(new_entries)
            self.n_scored += len(missing)
        self.n_cache_hits += len(documents) - len(missing)
        return [self._cache[key] for key in keys]


def _slug(name: str) -> str:
    return name.replace("/", "__")


class RetrievalService:
    """Corpus + indexes + reranker, loaded once per (dataset, split)."""

    def __init__(
        self,
        dataset: str,
        split: str | None = None,
        *,
        encoder: Any | None = None,
        reranker: Reranker | None = None,
        use_reranker: bool = True,
    ) -> None:
        from src.datasets import get_adapter

        adapter = get_adapter(dataset, split)
        self.dataset = dataset
        self.split = adapter.split
        self._artifact = corpus_module.load_corpus(dataset, self.split)
        self._dense, self._bm25, self._manifest = load_indexes(dataset, self.split)
        self._documents = {document.doc_id: document for document in self._artifact.documents}
        self._encoder = encoder or SentenceTransformerEncoder()
        self._reranker = reranker if reranker is not None else (Reranker() if use_reranker else None)
        self._search_cache: dict[tuple[str, str, int], list[RetrievedDoc]] = {}
        self.retrieval_executions = 0

    # -- identity -----------------------------------------------------------
    @property
    def corpus_sha256(self) -> str:
        return self._artifact.corpus_sha256

    @property
    def dataset_sha256(self) -> str:
        return self._artifact.dataset_sha256

    @property
    def n_documents(self) -> int:
        return len(self._documents)

    @property
    def encoder_name(self) -> str:
        return getattr(self._encoder, "name", type(self._encoder).__name__)

    @property
    def reranker_name(self) -> str | None:
        return self._reranker.model_name if self._reranker else None

    def document(self, doc_id: str) -> RetrievedDoc:
        document = self._documents[doc_id]
        return RetrievedDoc(doc_id=document.doc_id, title=document.title, text=document.text, rank=0)

    # -- search -------------------------------------------------------------
    def search(self, query: str, mode: str, k: int = 10) -> list[RetrievedDoc]:
        """Dense, BM25 or hybrid (RRF) search; results are rank-ordered."""
        cache_key = (mode, query, int(k))
        if cache_key in self._search_cache:
            return list(self._search_cache[cache_key])

        self.retrieval_executions += 1
        if mode == "dense":
            ids, scores = self._dense.search(self._encoder, query, k)
            hits = self._wrap(ids, scores, {}, "dense")
        elif mode == "bm25":
            ids, scores = self._bm25.search(query, k)
            hits = self._wrap(ids, scores, {}, "bm25")
        elif mode == "hybrid":
            dense_ids, dense_scores = self._dense.search(self._encoder, query, k)
            bm25_ids, bm25_scores = self._bm25.search(query, k)
            self.retrieval_executions += 1  # hybrid = two individual retriever executions
            rankings = {
                "dense": RankedList(query, "dense", dense_ids, dense_scores),
                "bm25": RankedList(query, "bm25", bm25_ids, bm25_scores),
            }
            fused = fusion_module.fuse(
                rankings, depth=config.RRF_DEPTH, constant=config.RRF_CONSTANT, k=k
            )
            hits = [
                RetrievedDoc(
                    doc_id=hit.doc_id,
                    title=self._documents[hit.doc_id].title,
                    text=self._documents[hit.doc_id].text,
                    rank=position,
                    retrieval_score=hit.rrf_score,
                    source_ranks=dict(hit.source_ranks),
                )
                for position, hit in enumerate(fused, start=1)
            ]
        else:
            raise ValueError(f"unknown retrieval mode {mode!r}")

        self._search_cache[cache_key] = hits
        return list(hits)

    def _wrap(self, ids, scores, source_ranks, source) -> list[RetrievedDoc]:
        return [
            RetrievedDoc(
                doc_id=doc_id,
                title=self._documents[doc_id].title,
                text=self._documents[doc_id].text,
                rank=position,
                retrieval_score=float(score),
                source_ranks={source: position},
            )
            for position, (doc_id, score) in enumerate(zip(ids, scores), start=1)
        ]

    # -- reranking ----------------------------------------------------------
    def rerank(self, query: str, documents: Sequence[RetrievedDoc], *, top_k: int | None = None) -> list[RetrievedDoc]:
        if self._reranker is None:
            raise RuntimeError("reranker disabled for this service")
        if not documents:
            return []
        scores = self._reranker.score(query, documents)
        ordered = sorted(
            zip(documents, scores), key=lambda pair: (-pair[1], pair[0].doc_id)
        )
        reranked = [
            RetrievedDoc(
                doc_id=document.doc_id,
                title=document.title,
                text=document.text,
                rank=position,
                rerank_score=score,
                retrieval_score=document.retrieval_score,
                source_ranks=dict(document.source_ranks),
            )
            for position, (document, score) in enumerate(ordered, start=1)
        ]
        return reranked[:top_k] if top_k is not None else reranked

    def dedupe(self, documents: Iterable[RetrievedDoc]) -> list[RetrievedDoc]:
        seen: set[str] = set()
        unique: list[RetrievedDoc] = []
        for document in documents:
            if document.doc_id in seen:
                continue
            seen.add(document.doc_id)
            unique.append(document)
        return unique
