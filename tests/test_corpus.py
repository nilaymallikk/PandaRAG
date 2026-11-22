"""Pooled corpus construction tests (Scope B, decision D2).

The corpus is the foundation every later phase depends on, so these tests pin the
properties the protocol claims: content-based ids, deterministic fingerprints,
loss-free deduplication, explicit handling of MuSiQue's ambiguous titles, a
separate provenance file (gold statistics never enter the runtime record), and
detection of any post-hoc corpus tampering.
"""

from __future__ import annotations

import json

import pytest

from src.corpus_retrieval import corpus as corpus_module
from src.datasets import get_adapter

DATASETS = [("hotpotqa", "validation"), ("2wiki", "dev"), ("musique", "validation")]


@pytest.mark.parametrize("dataset,split", DATASETS)
def test_build_is_deterministic(dataset: str, split: str, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(corpus_module, "CORPUS_ROOT", tmp_path / "corpus")
    adapter = get_adapter(dataset, split)
    examples = adapter.load()
    fingerprint = adapter.fingerprint()

    first = corpus_module.build_corpus(dataset, split, fingerprint.sha256, examples)
    second = corpus_module.build_corpus(dataset, split, fingerprint.sha256, list(reversed(examples)))

    assert first.corpus_sha256 == second.corpus_sha256, "corpus must not depend on question order"
    assert [d.doc_id for d in first.documents] == [d.doc_id for d in second.documents]
    assert len({d.doc_id for d in first.documents}) == len(first.documents)


@pytest.mark.parametrize("dataset,split", DATASETS)
def test_corpus_documents_are_unique_and_non_empty(dataset: str, split: str) -> None:
    adapter = get_adapter(dataset, split)
    artifacts = corpus_module.build_corpus(
        dataset, split, adapter.fingerprint().sha256, adapter.load()
    )
    assert artifacts.documents
    for document in artifacts.documents:
        assert document.title.strip()
        assert document.text.strip()
        assert document.dataset == dataset
        assert document.split == split


@pytest.mark.parametrize("dataset,split", DATASETS)
def test_every_gold_document_is_present(dataset: str, split: str) -> None:
    adapter = get_adapter(dataset, split)
    examples = adapter.load()
    artifact = corpus_module.build_corpus(
        dataset, split, adapter.fingerprint().sha256, examples
    )
    audit = corpus_module.validate_evidence_mapping(artifact, examples)
    assert audit["ok"], audit["unresolved_examples"]
    assert audit["n_resolved"] == audit["n_evidence_items"]


def test_hotpotqa_dedupes_repeated_paragraphs() -> None:
    adapter = get_adapter("hotpotqa")
    artifact = corpus_module.build_corpus(
        "hotpotqa", "validation", adapter.fingerprint().sha256, adapter.load()
    )
    stats = artifact.stats
    assert stats["n_documents"] == 4928
    assert stats["n_duplicate_instances_collapsed"] == 28
    # HotpotQA's distractor paragraphs are canonical: no title carries two texts.
    assert stats["n_ambiguous_titles"] == 0


def test_musique_ambiguous_titles_are_kept_not_dropped() -> None:
    adapter = get_adapter("musique")
    artifact = corpus_module.build_corpus(
        "musique", "validation", adapter.fingerprint().sha256, adapter.load()
    )
    stats = artifact.stats
    assert stats["n_ambiguous_titles"] > 0

    by_title: dict[str, set[str]] = {}
    for document in artifact.documents:
        by_title.setdefault(document.title, set()).add(document.doc_id)
    ambiguous = {title: ids for title, ids in by_title.items() if len(ids) > 1}
    assert len(ambiguous) == stats["n_ambiguous_titles"]

    title, doc_ids = sorted(ambiguous.items())[0]
    texts = {d.text for d in artifact.documents if d.title == title}
    assert len(texts) == len(doc_ids), "each ambiguous variant must keep its own text"


def test_provenance_is_separate_from_corpus_records(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(corpus_module, "CORPUS_ROOT", tmp_path / "corpus")
    adapter = get_adapter("hotpotqa")
    artifact = corpus_module.build_corpus(
        "hotpotqa", "validation", adapter.fingerprint().sha256, adapter.load()
    )
    directory = corpus_module.write_corpus(artifact)

    corpus_lines = (directory / corpus_module.CORPUS_FILE).read_text(encoding="utf-8").splitlines()
    assert corpus_lines
    for line in corpus_lines:
        row = json.loads(line)
        # The runtime corpus record exposes exactly these fields; provenance and
        # gold statistics live in a separate file. (Paragraph text may
        # legitimately contain the word "gold", so only keys are inspected.)
        assert set(row) == {"doc_id", "title", "text", "dataset", "split"}
        assert not any("gold" in key for key in row)

    provenance = json.loads((directory / corpus_module.PROVENANCE_FILE).read_text(encoding="utf-8"))
    assert "never be loaded by runtime retrieval" in provenance["note"]


@pytest.mark.parametrize("dataset,split", DATASETS)
def test_write_load_roundtrip_preserves_fingerprint(dataset: str, split: str, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(corpus_module, "CORPUS_ROOT", tmp_path / "corpus")
    adapter = get_adapter(dataset, split)
    original = corpus_module.build_corpus(
        dataset, split, adapter.fingerprint().sha256, adapter.load()
    )
    corpus_module.write_corpus(original)

    loaded = corpus_module.load_corpus(dataset, split)
    assert loaded.corpus_sha256 == original.corpus_sha256
    assert len(loaded.documents) == len(original.documents)
    manifest = corpus_module.load_manifest(dataset, split)
    assert manifest["corpus_sha256"] == original.corpus_sha256
    assert manifest["dataset_sha256"] == adapter.fingerprint().sha256


def test_tampered_corpus_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(corpus_module, "CORPUS_ROOT", tmp_path / "corpus")
    adapter = get_adapter("2wiki")
    artifact = corpus_module.build_corpus(
        "2wiki", "dev", adapter.fingerprint().sha256, adapter.load()
    )
    directory = corpus_module.write_corpus(artifact)

    path = directory / corpus_module.CORPUS_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["text"] = row["text"] + " tampered"
    lines[0] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="corpus fingerprint mismatch"):
        corpus_module.load_corpus("2wiki", "dev")


def test_fingerprint_changes_when_content_changes() -> None:
    from src.datasets.base import Document
    from src.corpus_retrieval.schemas import corpus_fingerprint

    base = [Document("d1", "T", "one", "hotpotqa", "validation")]
    changed = [Document("d1", "T", "two", "hotpotqa", "validation")]
    assert corpus_fingerprint(base) != corpus_fingerprint(changed)
    # Order independence.
    two = base + [Document("d2", "U", "other", "hotpotqa", "validation")]
    assert corpus_fingerprint(two) == corpus_fingerprint(list(reversed(two)))
