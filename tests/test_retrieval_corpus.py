"""Corpus-level retrieval layer tests (dense index, BM25, fusion, index manager).

Everything here runs offline: the dense index takes an injectable encoder, so no
model download and no sentence-transformer load is required. Production uses
``SentenceTransformerEncoder``; that path is exercised by the Phase 1 benchmark
script, not by the unit suite.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src import config
from src.corpus_retrieval import benchmark as benchmark_module
from src.corpus_retrieval import corpus as corpus_module
from src.corpus_retrieval import fusion as fusion_module
from src.corpus_retrieval import index_manager
from src.corpus_retrieval.bm25 import BM25Index
from src.corpus_retrieval.dense import DenseIndex
from src.corpus_retrieval.schemas import RankedList, corpus_fingerprint
from src.datasets.base import Document
from src.evaluation import recall_at_k


# ---------------------------------------------------------------------------
# Deterministic offline encoder
# ---------------------------------------------------------------------------


class StubEncoder:
    """Bag-of-words hash encoder: identical text => identical vector, no downloads."""

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def name(self) -> str:
        return "stub-hash-encoder/1.0"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for token in text.lower().replace("\n", " ").split():
            index = int(hashlib.sha1(token.encode()).hexdigest(), 16) % self._dimension
            vector[index] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def encode_documents(self, texts) -> np.ndarray:
        return np.vstack([self._encode(text) for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode(text)


@pytest.fixture()
def documents() -> list[Document]:
    rows = [
        ("d1", "Alpha", "the founder of company x was born in springfield"),
        ("d2", "Beta", "company x was founded by jane doe in 1999"),
        ("d3", "Gamma", "jane doe graduated from springfield university"),
        ("d4", "Delta", "springfield university was founded in 1871"),
        ("d5", "Epsilon", "an unrelated paragraph about rivers"),
    ]
    return [Document(doc_id, title, text, "synthetic", "dev") for doc_id, title, text in rows]


@pytest.fixture()
def corpus_sha(documents) -> str:
    return corpus_fingerprint(documents)


# ---------------------------------------------------------------------------
# Dense index
# ---------------------------------------------------------------------------


def test_dense_index_returns_k_results_sorted(documents, corpus_sha) -> None:
    encoder = StubEncoder()
    index = DenseIndex.build(documents, encoder, corpus_sha256=corpus_sha)

    ids, scores = index.search(encoder, "jane doe university", 3)
    assert len(ids) == 3
    assert list(scores) == sorted(scores, reverse=True)
    assert set(ids) <= {document.doc_id for document in documents}
    assert len(set(ids)) == len(ids)


def test_dense_index_is_deterministic(documents, corpus_sha) -> None:
    encoder = StubEncoder()
    first = DenseIndex.build(documents, encoder, corpus_sha256=corpus_sha)
    second = DenseIndex.build(list(reversed(documents)), encoder, corpus_sha256=corpus_sha)
    assert first.doc_ids == second.doc_ids
    assert np.array_equal(first.vectors, second.vectors)
    assert first.search(encoder, "springfield", 5) == second.search(encoder, "springfield", 5)


def test_dense_index_k_larger_than_corpus_is_clamped(documents, corpus_sha) -> None:
    encoder = StubEncoder()
    index = DenseIndex.build(documents, encoder, corpus_sha256=corpus_sha)
    ids, _ = index.search(encoder, "springfield", 50)
    assert len(ids) == len(documents)


def test_dense_index_rejects_mismatched_vectors(documents, corpus_sha) -> None:
    with pytest.raises(ValueError, match="vector count"):
        DenseIndex(
            doc_ids=("d1", "d2"),
            vectors=np.zeros((3, 4), dtype=np.float32),
            encoder_name="x",
            dimension=4,
            corpus_sha256=corpus_sha,
        )


def test_dense_index_save_load_roundtrip(documents, corpus_sha, tmp_path) -> None:
    encoder = StubEncoder()
    index = DenseIndex.build(documents, encoder, corpus_sha256=corpus_sha)
    index.save(tmp_path / "dense")

    loaded = DenseIndex.load(tmp_path / "dense")
    assert loaded.doc_ids == index.doc_ids
    assert loaded.corpus_sha256 == index.corpus_sha256
    assert loaded.encoder_name == index.encoder_name
    assert loaded.search(encoder, "springfield", 3) == index.search(encoder, "springfield", 3)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def test_bm25_ranks_lexical_matches_first(documents, corpus_sha) -> None:
    index = BM25Index.build(documents, corpus_sha256=corpus_sha)
    ids, scores = index.search("springfield university", 3)
    assert "d4" in ids or "d3" in ids
    assert list(scores) == sorted(scores, reverse=True)


def test_bm25_is_deterministic_on_ties(documents, corpus_sha) -> None:
    index = BM25Index.build(documents, corpus_sha256=corpus_sha)
    assert index.search("completely absent term", 5) == index.search("completely absent term", 5)


def test_bm25_parameters_are_preserved(documents, corpus_sha, tmp_path) -> None:
    index = BM25Index.build(documents, corpus_sha256=corpus_sha, k1=1.2, b=0.6, epsilon=0.1)
    index.save(tmp_path / "bm25")
    loaded = BM25Index.load(tmp_path / "bm25", documents)
    assert (loaded.k1, loaded.b, loaded.epsilon) == (1.2, 0.6, 0.1)
    assert loaded.search("springfield", 2) == index.search("springfield", 2)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def test_rrf_is_symmetric_and_monotone() -> None:
    a = [("d1", 0.9), ("d2", 0.8), ("d3", 0.7)]
    b = [("d3", 12.0), ("d1", 9.0), ("d4", 3.0)]

    forward = fusion_module.fuse_pairs(a, b)
    swapped = fusion_module.fuse_pairs(b, a)
    assert [doc_id for doc_id, _ in forward] == [doc_id for doc_id, _ in swapped]

    scores = dict(forward)
    # d1 appears at rank 1 and 2: 1/61 + 1/62. d3 at 3 and 1: 1/63 + 1/61.
    assert scores["d1"] > scores["d3"]
    assert scores["d1"] == pytest.approx(1 / 61 + 1 / 62)
    assert "d4" in scores


def test_fusion_respects_depth_and_dedupes() -> None:
    dense = RankedList("q", "dense", ("d1", "d2", "d3"), (0.9, 0.8, 0.7))
    bm25 = RankedList("q", "bm25", ("d2", "d2", "d4"), (5.0, 5.0, 1.0))

    fused = fusion_module.fuse({"dense": dense, "bm25": bm25}, depth=2)
    doc_ids = [hit.doc_id for hit in fused]
    assert set(doc_ids) == {"d1", "d2"}  # deduplicated, depth-limited
    assert len(doc_ids) == len(set(doc_ids))
    assert "d3" not in doc_ids  # beyond depth=2 in the dense ranking
    assert "d4" not in doc_ids  # beyond depth=2 in the BM25 ranking
    # d2 is the only candidate found by both sources, so it must rank first.
    assert doc_ids[0] == "d2"
    by_id = {hit.doc_id: hit for hit in fused}
    assert by_id["d2"].source_ranks == {"dense": 2, "bm25": 1}


def test_fusion_ties_break_by_doc_id() -> None:
    dense = RankedList("q", "dense", ("d9", "d8"), (1.0, 0.5))
    bm25 = RankedList("q", "bm25", ("d8", "d9"), (1.0, 0.5))
    fused = fusion_module.fuse({"dense": dense, "bm25": bm25})
    assert [hit.doc_id for hit in fused] == ["d8", "d9"]


# ---------------------------------------------------------------------------
# Index manager
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_corpus(documents, corpus_sha, tmp_path, monkeypatch):
    """A tiny on-disk corpus + index root, isolated from the real cache."""
    from src.corpus_retrieval.schemas import CorpusArtifact

    monkeypatch.setattr(corpus_module, "CORPUS_ROOT", tmp_path / "corpus")
    monkeypatch.setattr(index_manager, "INDEX_ROOT", tmp_path / "indexes")
    artifact = CorpusArtifact(
        dataset="synthetic",
        split="dev",
        dataset_sha256="sha-of-dataset",
        documents=tuple(sorted(documents, key=lambda d: d.doc_id)),
        provenance={},
        stats={"protocol": "test", "n_documents": len(documents), "granularity": "paragraph"},
    )
    corpus_module.write_corpus(artifact)
    return artifact


def test_index_manager_build_load_and_verify(synthetic_corpus) -> None:
    encoder = StubEncoder()
    manifest = index_manager.build_indexes(synthetic_corpus, encoder=encoder, force=True)
    assert manifest.n_documents == len(synthetic_corpus.documents)
    assert manifest.corpus_sha256 == synthetic_corpus.corpus_sha256

    dense, bm25, loaded = index_manager.load_indexes("synthetic", "dev")
    assert loaded.corpus_sha256 == manifest.corpus_sha256
    assert dense.search(encoder, "springfield", 2)[0]
    assert bm25.search("springfield", 2)[0]


def test_index_manager_rejects_stale_corpus(synthetic_corpus, tmp_path) -> None:
    encoder = StubEncoder()
    index_manager.build_indexes(synthetic_corpus, encoder=encoder, force=True)

    manifest_path = index_manager.index_dir("synthetic", "dev") / index_manager.MANIFEST_FILE
    payload = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    payload["corpus_sha256"] = "0" * 64
    manifest_path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="different corpus"):
        index_manager.load_indexes("synthetic", "dev")


# ---------------------------------------------------------------------------
# Metric plumbing used by the benchmark
# ---------------------------------------------------------------------------


def test_recall_matches_paper_definition() -> None:
    gold = {"d1", "d2"}
    assert recall_at_k(["d1", "d2"], gold, 2) == 1.0
    assert recall_at_k(["d1", "d9", "d2"], gold, 3) == 1.0
    assert recall_at_k(["d1", "d9"], gold, 2) == 0.5
    assert recall_at_k(["d9"], gold, 1) == 0.0


def test_aggregate_reports_recall_and_all_gold_coverage() -> None:
    rows = [
        {
            "n_gold_documents": 2,
            "dense": {
                "recall": {"recall@1": 0.5, "recall@10": 1.0},
                "complete": {"complete@1": 0.0, "complete@10": 1.0},
                "n_gold_retrieved": 2,
                "gold_ranks": [1, 2],
            },
            "bm25": {
                "recall": {"recall@1": 0.5, "recall@10": 0.5},
                "complete": {"complete@1": 0.0, "complete@10": 0.0},
                "n_gold_retrieved": 1,
                "gold_ranks": [1],
            },
            "hybrid": {
                "recall": {"recall@1": 0.5, "recall@10": 1.0},
                "complete": {"complete@1": 0.0, "complete@10": 1.0},
                "n_gold_retrieved": 2,
                "gold_ranks": [1, 3],
            },
        }
    ]

    class _Artifact:
        documents = [object()] * 50

    class _Manifest:
        n_documents = 50

    summary = benchmark_module._aggregate(rows, [1, 10], _Artifact(), _Manifest(), 1.23)
    assert summary["dense"]["recall@10"] == 1.0
    assert summary["dense"]["all_gold_coverage@10"] == 1.0
    assert summary["bm25"]["recall@10"] == 0.5
    assert summary["bm25"]["all_gold_coverage@10"] == 0.0
    assert summary["dense"]["suspicious_recall_at_10"] is True
    assert summary["dense"]["gold_rank_max"] == 2


def test_config_reference_values_match_code() -> None:
    """The retrieval config must describe the code that runs, not an aspiration."""
    from src.experiment_config import retrieval_config

    payload = retrieval_config()
    assert payload["bm25"]["k1"] == config.BM25_K1
    assert payload["bm25"]["b"] == config.BM25_B
    assert payload["bm25"]["epsilon"] == config.BM25_EPSILON
    assert payload["fusion"]["constant"] == config.RRF_CONSTANT
    assert payload["fusion"]["depth"] == config.RRF_DEPTH
    assert tuple(payload["k_values"]) == config.RETRIEVAL_K_VALUES
    assert payload["dense"]["model"] == config.EMBEDDING_MODEL_NAME
