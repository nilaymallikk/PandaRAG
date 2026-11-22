"""Build the dense + BM25 indexes for the pooled corpora (Phase 1).

CPU embedding: roughly 0.05 s per document with ``bge-small-en-v1.5``, so a full
rebuild of all three datasets is ~15 minutes. Existing indexes whose corpus
fingerprint still matches are reused unless ``--force``.

Usage::

    python -m src.corpus_retrieval.index_cli
    python -m src.corpus_retrieval.index_cli --dataset musique --force
"""

from __future__ import annotations

import argparse
import time
from typing import Sequence

from src.corpus_retrieval import corpus as corpus_module
from src.corpus_retrieval.index_manager import build_indexes
from src.datasets import available


def build(dataset: str, split: str, *, force: bool = False, encoder=None) -> dict[str, object]:
    artifact = corpus_module.load_corpus(dataset, split)
    started = time.perf_counter()
    manifest = build_indexes(artifact, encoder=encoder, force=force)
    return {
        "dataset": dataset,
        "split": split,
        "n_documents": manifest.n_documents,
        "corpus_sha256": manifest.corpus_sha256,
        "dense_dimension": manifest.dense["dimension"],
        "encoder": manifest.dense["encoder"],
        "elapsed_s": round(time.perf_counter() - started, 1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build corpus indexes")
    parser.add_argument("--dataset", default=None, help="dataset name; default: all")
    parser.add_argument("--force", action="store_true", help="rebuild even if fingerprints match")
    args = parser.parse_args(argv)

    encoder = None  # lazily constructed once by build_indexes
    targets = [(args.dataset, None)] if args.dataset else available()
    for dataset, split in targets:
        print(build(dataset, split, force=args.force, encoder=encoder), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
