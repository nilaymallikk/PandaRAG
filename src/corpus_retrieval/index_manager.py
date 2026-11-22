"""Build, persist, load and verify corpus-level indexes.

Layout::

    cache/corpus/<dataset>/<split>/          corpus.jsonl, provenance.json, manifest.json
    cache/indexes/<dataset>/<split>/dense/   vectors + doc order + config
    cache/indexes/<dataset>/<split>/bm25/    doc order + config (rebuilt on load)
    cache/indexes/<dataset>/<split>/manifest.json

Verification is the point of this module: an index whose corpus fingerprint does
not match the corpus on disk is refused, so a stale index can never silently be
used after the corpus changes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config
from src.corpus_retrieval import corpus as corpus_module
from src.corpus_retrieval.bm25 import BM25Index
from src.corpus_retrieval.dense import DenseIndex, Encoder, SentenceTransformerEncoder
from src.corpus_retrieval.schemas import CorpusArtifact, IndexManifest

INDEX_ROOT = config.CACHE_DIR / "indexes"
MANIFEST_FILE = "manifest.json"
DENSE_DIR = "dense"
BM25_DIR = "bm25"


def index_dir(dataset: str, split: str) -> Path:
    return INDEX_ROOT / dataset / split


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_indexes(
    artifact: CorpusArtifact,
    *,
    encoder: Encoder | None = None,
    force: bool = False,
) -> IndexManifest:
    """Build and persist dense + BM25 indexes for one corpus artifact."""
    directory = index_dir(artifact.dataset, artifact.split)
    manifest_path = directory / MANIFEST_FILE
    corpus_sha = artifact.corpus_sha256

    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("corpus_sha256") == corpus_sha:
            return _manifest_from_dict(existing)

    encoder = encoder or SentenceTransformerEncoder()
    dense = DenseIndex.build(artifact.documents, encoder, corpus_sha256=corpus_sha)
    bm25 = BM25Index.build(artifact.documents, corpus_sha256=corpus_sha)

    dense.save(
        directory / DENSE_DIR,
        extra={"encoder_dimension": int(encoder.dimension), "query_instruction": True},
    )
    bm25.save(directory / BM25_DIR)

    manifest = IndexManifest(
        dataset=artifact.dataset,
        split=artifact.split,
        corpus_sha256=corpus_sha,
        n_documents=len(artifact.documents),
        dense={
            "encoder": dense.encoder_name,
            "dimension": dense.dimension,
            "metric": "cosine",
            "index_type": "faiss.IndexFlatIP",
            "normalized": True,
        },
        bm25={
            "implementation": "rank_bm25.BM25Okapi",
            "k1": bm25.k1,
            "b": bm25.b,
            "epsilon": bm25.epsilon,
            "tokenizer": bm25.tokenizer,
        },
        built_utc=_utc_now(),
    )
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _manifest_from_dict(payload: dict[str, Any]) -> IndexManifest:
    return IndexManifest(
        dataset=payload["dataset"],
        split=payload["split"],
        corpus_sha256=payload["corpus_sha256"],
        n_documents=int(payload["n_documents"]),
        dense=payload["dense"],
        bm25=payload["bm25"],
        built_utc=payload["built_utc"],
    )


def load_indexes(
    dataset: str,
    split: str,
    *,
    verify: bool = True,
) -> tuple[DenseIndex, BM25Index, IndexManifest]:
    """Load persisted indexes, optionally verifying them against the corpus."""
    directory = index_dir(dataset, split)
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(f"indexes not built for {dataset}/{split}: {directory}")

    manifest = _manifest_from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    artifact = corpus_module.load_corpus(dataset, split)

    if verify:
        if manifest.corpus_sha256 != artifact.corpus_sha256:
            raise ValueError(
                f"{dataset}/{split}: index was built for a different corpus "
                f"({manifest.corpus_sha256[:12]}… vs {artifact.corpus_sha256[:12]}…)"
            )
        if manifest.n_documents != len(artifact.documents):
            raise ValueError(f"{dataset}/{split}: index document count does not match the corpus")

    dense = DenseIndex.load(directory / DENSE_DIR)
    bm25 = BM25Index.load(directory / BM25_DIR, artifact.documents)
    return dense, bm25, manifest


def load_manifest(dataset: str, split: str) -> IndexManifest:
    path = index_dir(dataset, split) / MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"index manifest missing: {path}")
    return _manifest_from_dict(json.loads(path.read_text(encoding="utf-8")))


def index_fingerprint(dataset: str, split: str) -> str:
    """The corpus fingerprint an index pair is bound to (used in cache keys)."""
    return load_manifest(dataset, split).corpus_sha256
