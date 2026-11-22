"""Pooled candidate corpus construction (Scope B, decision D2).

Protocol
--------
The corpus for one ``(dataset, split)`` is the union of every paragraph shipped
with every question in that dataset file, deduplicated by **content**:
``doc_id = content_doc_id(dataset, split, title, text)``.

Why this protocol:

* it is closed-book over each dataset's shipped paragraph pool, so every gold
  document is guaranteed present (an open-domain corpus would need downloads and
  would change the research question);
* it makes document-level Recall@K a non-trivial number (3k–7k documents)
  instead of the structural ``1.000`` produced by per-question indices;
* deduplication by content handles MuSiQue, where the same paragraph *title*
  carries different text in different questions (1,116 such titles), without
  silently discarding a paragraph.

What it is not: this is **not** Wikipedia/open-domain retrieval, and its recall
must never be compared with the base paper's Recall@10 (the paper does not
describe its corpus — see the reproduction protocol, DV8).

Gold annotations are used here **only** to compute offline provenance statistics
(``is_gold_for``). They are never part of the runtime corpus record, and the
runtime view of a question (:class:`~src.datasets.base.RuntimeExample`) has no
gold fields at all.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src import config
from src.datasets.base import Document, QAExample
from src.corpus_retrieval.schemas import CorpusArtifact, corpus_fingerprint

PROTOCOL_VERSION = "pooled-candidate-corpus/1.0"
CORPUS_ROOT = config.CACHE_DIR / "corpus"

CORPUS_FILE = "corpus.jsonl"
PROVENANCE_FILE = "provenance.json"
MANIFEST_FILE = "manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_corpus(
    dataset: str,
    split: str,
    dataset_sha256: str,
    examples: Iterable[QAExample],
) -> CorpusArtifact:
    """Build the pooled corpus for one dataset split.

    Deterministic: output documents are sorted by ``doc_id``, so the corpus
    fingerprint does not depend on question order.
    """
    examples = list(examples)
    unique: dict[str, Document] = {}
    provenance: dict[str, dict[str, int]] = {}
    title_to_doc_ids: dict[str, set[str]] = {}

    for example in examples:
        for document in example.documents():
            unique.setdefault(document.doc_id, document)
            record = provenance.setdefault(document.doc_id, {"n_questions": 0, "is_gold_for": 0})
            record["n_questions"] += 1
            title_to_doc_ids.setdefault(document.title, set()).add(document.doc_id)
        for gold_doc_id in example.gold_doc_ids:
            provenance.setdefault(gold_doc_id, {"n_questions": 0, "is_gold_for": 0})
            provenance[gold_doc_id]["is_gold_for"] += 1

    documents = tuple(sorted(unique.values(), key=lambda d: d.doc_id))

    gold_ids = {doc_id for doc_id, record in provenance.items() if record["is_gold_for"] > 0}
    missing_gold = sorted(gold_ids - set(unique))
    if missing_gold:
        raise ValueError(
            f"{dataset}: {len(missing_gold)} gold document ids are absent from the corpus "
            f"(first: {missing_gold[:3]}); evidence mapping is broken"
        )

    ambiguous_titles = {title: sorted(ids) for title, ids in title_to_doc_ids.items() if len(ids) > 1}

    stats: dict[str, Any] = {
        "protocol": PROTOCOL_VERSION,
        "dataset": dataset,
        "split": split,
        "n_questions": len(examples),
        "n_documents": len(documents),
        "n_documents_used_as_gold": len(gold_ids),
        "n_documents_never_gold": len(documents) - len(gold_ids),
        "n_paragraph_instances_seen": sum(record["n_questions"] for record in provenance.values()),
        "n_duplicate_instances_collapsed": sum(record["n_questions"] - 1 for record in provenance.values()),
        "n_titles": len(title_to_doc_ids),
        "n_ambiguous_titles": len(ambiguous_titles),
        "ambiguous_title_examples": dict(sorted(ambiguous_titles.items())[:5]),
        "dedupe": "content hash of (dataset, split, title, paragraph text)",
        "granularity": "paragraph",
        "gold_visible_at_runtime": False,
    }

    return CorpusArtifact(
        dataset=dataset,
        split=split,
        dataset_sha256=dataset_sha256,
        documents=documents,
        provenance=provenance,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def corpus_dir(dataset: str, split: str) -> Path:
    return CORPUS_ROOT / dataset / split


def write_corpus(artifact: CorpusArtifact) -> Path:
    directory = corpus_dir(artifact.dataset, artifact.split)
    directory.mkdir(parents=True, exist_ok=True)

    with (directory / CORPUS_FILE).open("w", encoding="utf-8") as handle:
        for document in artifact.documents:
            handle.write(
                json.dumps(
                    {
                        "doc_id": document.doc_id,
                        "title": document.title,
                        "text": document.text,
                        "dataset": document.dataset,
                        "split": document.split,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    (directory / PROVENANCE_FILE).write_text(
        json.dumps(
            {
                "note": (
                    "Offline statistics only. 'is_gold_for' counts how many questions use the "
                    "document as gold evidence; it must never be loaded by runtime retrieval."
                ),
                "provenance": artifact.provenance,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    manifest = {
        "protocol": PROTOCOL_VERSION,
        "dataset": artifact.dataset,
        "split": artifact.split,
        "dataset_sha256": artifact.dataset_sha256,
        "corpus_sha256": artifact.corpus_sha256,
        "n_documents": len(artifact.documents),
        "built_utc": _utc_now(),
        "stats": artifact.stats,
    }
    (directory / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return directory


def load_corpus(dataset: str, split: str) -> CorpusArtifact:
    directory = corpus_dir(dataset, split)
    manifest_path = directory / MANIFEST_FILE
    corpus_path = directory / CORPUS_FILE
    if not manifest_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(f"corpus not built for {dataset}/{split}: {directory}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            documents.append(
                Document(
                    doc_id=row["doc_id"],
                    title=row["title"],
                    text=row["text"],
                    dataset=row["dataset"],
                    split=row["split"],
                )
            )

    artifact = CorpusArtifact(
        dataset=manifest["dataset"],
        split=manifest["split"],
        dataset_sha256=manifest["dataset_sha256"],
        documents=tuple(documents),
        provenance={},
        stats=manifest["stats"],
    )

    actual = corpus_fingerprint(artifact.documents)
    if actual != manifest["corpus_sha256"]:
        raise ValueError(
            f"{dataset}/{split}: corpus fingerprint mismatch — file says {manifest['corpus_sha256'][:12]}…, "
            f"content hashes to {actual[:12]}…. Refusing to use a modified corpus."
        )
    return artifact


def load_manifest(dataset: str, split: str) -> dict[str, Any]:
    path = corpus_dir(dataset, split) / MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"corpus manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_evidence_mapping(artifact: CorpusArtifact, examples: Iterable[QAExample]) -> dict[str, Any]:
    """Offline audit: every gold evidence item must resolve to a corpus document."""
    ids = {document.doc_id for document in artifact.documents}
    total = 0
    missing: list[tuple[str, str, str]] = []
    for example in examples:
        for evidence in example.gold_evidence:
            total += 1
            if evidence.doc_id not in ids:
                missing.append((example.example_id, evidence.title, evidence.doc_id))
    return {
        "dataset": artifact.dataset,
        "split": artifact.split,
        "n_evidence_items": total,
        "n_resolved": total - len(missing),
        "n_unresolved": len(missing),
        "unresolved_examples": missing[:5],
        "ok": not missing,
    }


def summary_rows(artifacts: Mapping[str, CorpusArtifact]) -> list[dict[str, Any]]:
    """Compact rows for the phase-1 report."""
    rows = []
    for name, artifact in artifacts.items():
        rows.append(
            {
                "dataset": name,
                "split": artifact.split,
                "dataset_sha256": artifact.dataset_sha256,
                "corpus_sha256": artifact.corpus_sha256,
                "n_documents": len(artifact.documents),
                **{
                    key: artifact.stats[key]
                    for key in (
                        "n_questions",
                        "n_paragraph_instances_seen",
                        "n_duplicate_instances_collapsed",
                        "n_titles",
                        "n_ambiguous_titles",
                        "n_documents_used_as_gold",
                    )
                },
            }
        )
    return rows
