"""Local retrieval over the HotpotQA candidate contexts (dense and BM25).

Both retrievers are deliberately small and local:

* candidate corpus = the distractor paragraphs shipped with each question in
  ``data/hotpotqa_500.json`` (no Wikipedia download, no hosted vector store);
* dense: embeddings = ``BAAI/bge-small-en-v1.5`` (sentence-transformers, CPU),
  L2 normalised so that inner-product search equals cosine similarity, index =
  FAISS ``IndexFlatIP``, one index per question and granularity;
* lexical: ``rank_bm25.BM25Okapi`` built over exactly the same per-question
  candidate paragraphs / sentences.

Both retrievers return the same :class:`RetrievalResult`, so a single
evaluation path (:mod:`src.evaluation`) and a single benchmark runner
(:mod:`src.retrieval_benchmark`) serve both methods.

Two granularities are produced for every question:

* paragraph level -- the retrieval unit used by the RAG pipelines;
* sentence level -- a diagnostic that shows whether the exact supporting
  sentences (``title`` + ``sent_id``) were ranked, not only their article.

Both rankings are computed once at ``k = 10``; smaller ``K`` are prefixes of the
same ranking, so no re-encoding is needed for K=1/3/5.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any, Callable, Sequence

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src import config
from src.dataset import HotpotExample, iter_sentences, sentence_id


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer used by the lexical retriever.

    Punctuation and whitespace are dropped, no stemming and no stop-word
    removal is applied; the tokenizer is recorded in the result metadata.
    """
    return re.findall(r"[a-z0-9]+", text.lower())



@dataclass(frozen=True)
class RankedHit:
    """One ranked retrieval hit (a paragraph when ``sent_id`` is None)."""

    rank: int
    title: str
    score: float
    sent_id: int | None = None

    @property
    def is_sentence(self) -> bool:
        return self.sent_id is not None

    def identifier(self) -> str:
        """``title`` for paragraphs, ``title::sent_id`` for sentences."""
        return self.title if self.sent_id is None else sentence_id(self.title, self.sent_id)

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {"rank": self.rank, "title": self.title, "score": self.score}
        if self.sent_id is not None:
            record["sent_id"] = self.sent_id
        return record


@dataclass
class RetrievalResult:
    """Ranked paragraph and sentence hits for a single question."""

    example_id: str
    question: str
    k: int
    paragraphs: list[RankedHit] = field(default_factory=list)
    sentences: list[RankedHit] = field(default_factory=list)
    encode_s: float = 0.0
    search_s: float = 0.0

    @property
    def latency_s(self) -> float:
        """Retrieval latency for this question (encoding + FAISS search)."""
        return round(self.encode_s + self.search_s, 6)

    def top_paragraphs(self, k: int) -> list[RankedHit]:
        return self.paragraphs[:k]

    def top_sentences(self, k: int) -> list[RankedHit]:
        return self.sentences[:k]

    def paragraph_titles(self, k: int) -> list[str]:
        return [hit.title for hit in self.top_paragraphs(k)]

    def sentence_ids(self, k: int) -> list[str]:
        return [hit.identifier() for hit in self.top_sentences(k)]


class DenseRetriever:
    """BGE-small-en-v1.5 + FAISS inner-product retriever.

    ``encoder`` may be injected for tests, so ranking logic can be exercised
    without loading the real model.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        device: str | None = None,
        query_instruction: str | None = None,
        batch_size: int | None = None,
        encoder: Any | None = None,
    ) -> None:
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.device = device or config.EMBEDDING_DEVICE
        self.query_instruction = (
            config.EMBEDDING_QUERY_INSTRUCTION
            if query_instruction is None
            else query_instruction
        )
        self.batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        self._encoder = encoder
        self.embedding_dim: int | None = (
            encoder.get_embedding_dimension() if encoder is not None else None
        )

    def load(self) -> float:
        """Load the embedding model if needed; returns the load time in seconds."""
        if self._encoder is not None:
            return 0.0
        started = time.perf_counter()
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self._encoder = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self._encoder.get_embedding_dimension()
        return round(time.perf_counter() - started, 4)

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        """Embed texts (queries get the BGE instruction prefix, documents do not)."""
        self.load()
        if not texts:
            return np.zeros((0, self.embedding_dim or 0), dtype="float32")
        payload = (
            [f"{self.query_instruction}{text}" for text in texts]
            if is_query and self.query_instruction
            else list(texts)
        )
        vectors = self._encoder.encode(
            payload,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    @staticmethod
    def _search(
        vectors: np.ndarray,
        query_vector: np.ndarray,
        k: int,
        records: Sequence[tuple[str, int | None]],
    ) -> list[RankedHit]:
        """Rank ``records`` against the query using a flat inner-product index."""
        if not records or vectors.shape[0] == 0:
            return []
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        scores, positions = index.search(query_vector, min(k, len(records)))
        hits: list[RankedHit] = []
        for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1):
            if position < 0:
                continue
            title, sent_id = records[int(position)]
            hits.append(RankedHit(rank=rank, title=title, score=float(score), sent_id=sent_id))
        return hits

    def retrieve(
        self, example: HotpotExample, k: int = config.RETRIEVAL_MAX_K
    ) -> RetrievalResult:
        """Rank the candidate paragraphs and sentences of ``example``."""
        paragraph_titles = list(example.context_titles)
        paragraph_texts = [example.paragraph_text(title) for title in paragraph_titles]
        sentence_records = [
            (title, sent_id, text) for _, title, sent_id, text in iter_sentences(example)
        ]
        sentence_texts = [text for _, _, text in sentence_records]

        # Paragraphs and sentences are embedded in separate calls: a single
        # combined batch pads every short sentence up to the longest paragraph
        # length, which dominates the runtime on CPU.
        encode_started = time.perf_counter()
        paragraph_vectors = self.embed(paragraph_texts)
        sentence_vectors = self.embed(sentence_texts)
        query_vector = self.embed([example.question], is_query=True)
        encode_s = time.perf_counter() - encode_started

        search_started = time.perf_counter()
        paragraph_hits = self._search(
            paragraph_vectors,
            query_vector,
            k,
            [(title, None) for title in paragraph_titles],
        )
        sentence_hits = self._search(
            sentence_vectors,
            query_vector,
            k,
            [(title, sent_id) for title, sent_id, _ in sentence_records],
        )
        search_s = time.perf_counter() - search_started

        return RetrievalResult(
            example_id=example.example_id,
            question=example.question,
            k=min(k, len(paragraph_titles)),
            paragraphs=paragraph_hits,
            sentences=sentence_hits,
            encode_s=round(encode_s, 6),
            search_s=round(search_s, 6),
        )

    def settings(self) -> dict[str, Any]:
        """Retrieval configuration to store with every result."""
        return {
            "retrieval_method": "dense",
            "embedding_model": self.model_name,
            "embedding_device": self.device,
            "query_instruction": self.query_instruction,
            "similarity": "cosine (L2-normalised inner product)",
            "index": "faiss.IndexFlatIP",
            "granularities": ["paragraph", "sentence"],
            "max_k": config.RETRIEVAL_MAX_K,
        }


def _package_version_or_unknown(distribution: str) -> str:
    """Version of a third-party package, or ``"unknown"`` if not installed."""
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return "unknown"


class BM25Retriever:
    """Lexical ``rank_bm25.BM25Okapi`` retriever over the per-question candidates.

    The index is rebuilt for each question from exactly that question's
    candidate paragraphs (and candidate sentences), mirroring
    :class:`DenseRetriever` so the two retrieval methods are comparable.
    """

    def __init__(
        self,
        *,
        k1: float | None = None,
        b: float | None = None,
        epsilon: float | None = None,
        tokenizer: Callable[[str], list[str]] = tokenize,
    ) -> None:
        self.k1 = config.BM25_K1 if k1 is None else k1
        self.b = config.BM25_B if b is None else b
        self.epsilon = config.BM25_EPSILON if epsilon is None else epsilon
        self.tokenizer = tokenizer

    def load(self) -> float:
        """Kept for parity with :class:`DenseRetriever`: BM25 loads no model."""
        return 0.0

    def _rank(
        self,
        texts: Sequence[str],
        query_tokens: Sequence[str],
        k: int,
        records: Sequence[tuple[str, int | None]],
    ) -> list[RankedHit]:
        """Score and rank ``records`` against the query with BM25."""
        if not records or not texts:
            return []
        corpus = [self.tokenizer(text) for text in texts]
        bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b, epsilon=self.epsilon)
        scores = np.asarray(bm25.get_scores(query_tokens), dtype="float64")
        # Stable ordering keeps equal scores in corpus order (deterministic).
        order = np.argsort(-scores, kind="stable")
        hits: list[RankedHit] = []
        for rank, position in enumerate(order[: min(k, len(records))], start=1):
            title, sent_id = records[int(position)]
            hits.append(
                RankedHit(rank=rank, title=title, score=float(scores[position]), sent_id=sent_id)
            )
        return hits

    def retrieve(
        self, example: HotpotExample, k: int = config.RETRIEVAL_MAX_K
    ) -> RetrievalResult:
        """Rank the candidate paragraphs and sentences of ``example`` with BM25."""
        paragraph_titles = list(example.context_titles)
        paragraph_texts = [example.paragraph_text(title) for title in paragraph_titles]
        sentence_records = [
            (title, sent_id, text) for _, title, sent_id, text in iter_sentences(example)
        ]
        sentence_texts = [text for _, _, text in sentence_records]
        query_tokens = self.tokenizer(example.question)

        search_started = time.perf_counter()
        paragraph_hits = self._rank(
            paragraph_texts, query_tokens, k, [(title, None) for title in paragraph_titles]
        )
        sentence_hits = self._rank(
            sentence_texts,
            query_tokens,
            k,
            [(title, sent_id) for title, sent_id, _ in sentence_records],
        )
        search_s = time.perf_counter() - search_started

        return RetrievalResult(
            example_id=example.example_id,
            question=example.question,
            k=min(k, len(paragraph_titles)),
            paragraphs=paragraph_hits,
            sentences=sentence_hits,
            encode_s=0.0,  # lexical retrieval has no embedding step
            search_s=round(search_s, 6),
        )

    def settings(self) -> dict[str, Any]:
        """Retrieval configuration to store with every result."""
        return {
            "retrieval_method": "bm25",
            "implementation": "rank_bm25.BM25Okapi",
            "library_version": _package_version_or_unknown("rank-bm25"),
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "tokenizer": "lowercase [a-z0-9]+ tokens (no stemming, no stop-word removal)",
            "corpus": "per-question candidate paragraphs / sentences (distractor setting)",
            "similarity": "BM25 score (unbounded, higher is better)",
            "granularities": ["paragraph", "sentence"],
            "max_k": config.RETRIEVAL_MAX_K,
        }
