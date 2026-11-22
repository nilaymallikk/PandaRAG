"""Build the pooled candidate corpora (Phase 1).

Usage::

    python -m src.corpus_retrieval.corpus_cli                 # all datasets
    python -m src.corpus_retrieval.corpus_cli --dataset 2wiki --force
"""

from __future__ import annotations

import argparse
from typing import Sequence

from src.corpus_retrieval import corpus as corpus_module
from src.datasets import available, get_adapter


def build(dataset: str, split: str, *, force: bool = False) -> dict[str, object]:
    adapter = get_adapter(dataset, split)
    examples = adapter.load()
    fingerprint = adapter.fingerprint()

    directory = corpus_module.corpus_dir(dataset, split)
    if (directory / corpus_module.MANIFEST_FILE).exists() and not force:
        existing = corpus_module.load_manifest(dataset, split)
        if existing.get("corpus_sha256") == corpus_module.build_corpus(
            dataset, split, fingerprint.sha256, examples
        ).corpus_sha256:
            return {"dataset": dataset, "split": split, "status": "unchanged", **existing}

    artifact = corpus_module.build_corpus(dataset, split, fingerprint.sha256, examples)
    corpus_module.write_corpus(artifact)
    audit = corpus_module.validate_evidence_mapping(artifact, examples)
    if not audit["ok"]:
        raise SystemExit(f"{dataset}: evidence mapping failed: {audit['unresolved_examples']}")
    return {
        "dataset": dataset,
        "split": split,
        "status": "built",
        "n_documents": len(artifact.documents),
        "corpus_sha256": artifact.corpus_sha256,
        "n_ambiguous_titles": artifact.stats["n_ambiguous_titles"],
        "n_duplicates_collapsed": artifact.stats["n_duplicate_instances_collapsed"],
        "evidence_items": audit["n_evidence_items"],
        "evidence_resolved": audit["n_resolved"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build pooled candidate corpora")
    parser.add_argument("--dataset", default=None, help="dataset name; default: all")
    parser.add_argument("--force", action="store_true", help="rebuild even if fingerprints match")
    args = parser.parse_args(argv)

    targets = [(args.dataset, None)] if args.dataset else available()
    for dataset, split in targets:
        result = build(dataset, split, force=args.force)
        print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
