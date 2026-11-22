"""Dataset adapter tests (Scope B).

These assert the properties Phase 1 promises: three datasets load through one
canonical interface, ids are stable and unique, evidence resolves, fingerprints
match the committed manifest, and — most importantly — the runtime view of a
question physically cannot leak gold.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src import config
from src.datasets import (
    DatasetError,
    build_examples,
    content_doc_id,
    get_adapter,
    load_semantic_types,
    semantic_type_histogram,
)
from src.datasets.base import REQUIRED_RECORD_FIELDS

DATASETS = [("hotpotqa", "validation"), ("2wiki", "dev"), ("musique", "validation")]

EXPECTED_EVIDENCE_ITEMS = {"hotpotqa": 1212, "2wiki": 1000, "musique": 1322}
EXPECTED_EVIDENCE_KIND = {"hotpotqa": "sentence", "2wiki": "sentence", "musique": "paragraph"}


@pytest.mark.parametrize("dataset,split", DATASETS)
def test_loads_full_split(dataset: str, split: str) -> None:
    adapter = get_adapter(dataset, split)
    examples = adapter.load()
    assert len(examples) == 500
    assert len({example.example_id for example in examples}) == 500
    assert all(example.question for example in examples)
    assert all(example.answer for example in examples)


@pytest.mark.parametrize("dataset,split", DATASETS)
def test_fingerprint_matches_manifest(dataset: str, split: str) -> None:
    adapter = get_adapter(dataset, split)
    fingerprint = adapter.fingerprint()

    manifest_path = config.DATA_DIR / "dataset_manifest.json"
    payload = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    entry = payload["datasets"][adapter.manifest_key]

    assert fingerprint.sha256 == entry["sha256"]
    assert fingerprint.n_questions == entry["n"] == 500
    assert fingerprint.evidence_kind == EXPECTED_EVIDENCE_KIND[dataset]
    assert fingerprint.split == split
    # The file itself must hash to what the fingerprint claims.
    assert hashlib.sha256(adapter.path().read_bytes()).hexdigest() == fingerprint.sha256


@pytest.mark.parametrize("dataset,split", DATASETS)
def test_gold_evidence_resolves_into_corpus(dataset: str, split: str) -> None:
    adapter = get_adapter(dataset, split)
    examples = adapter.load()
    documents = adapter.documents()
    doc_ids = {document.doc_id for document in documents}

    assert sum(len(example.gold_evidence) for example in examples) == EXPECTED_EVIDENCE_ITEMS[dataset]
    for example in examples:
        assert example.gold_evidence, f"{example.example_id}: empty gold set"
        for evidence in example.gold_evidence:
            assert evidence.doc_id in doc_ids
            assert evidence.kind == EXPECTED_EVIDENCE_KIND[dataset]
            assert evidence.title
            if evidence.kind == "sentence":
                assert isinstance(evidence.sent_id, int)
            else:
                assert evidence.sent_id is None


def test_musique_evidence_is_paragraph_level() -> None:
    examples = get_adapter("musique").load()
    evidence = [item for example in examples for item in example.gold_evidence]
    assert evidence and all(item.kind == "paragraph" for item in evidence)
    assert all(item.sent_id is None for item in evidence)
    # A paragraph-level item must carry the whole paragraph, not one sentence.
    sample = evidence[0]
    example = next(e for e in examples if sample.doc_id in e.gold_doc_ids)
    assert sample.text == example.paragraph_text(sample.title)


@pytest.mark.parametrize("dataset,split", DATASETS)
def test_runtime_view_has_no_gold(dataset: str, split: str) -> None:
    example = get_adapter(dataset, split).load()[0]
    runtime = example.runtime()

    for forbidden in ("answer", "gold_evidence", "gold_doc_ids", "gold_sentences"):
        assert not hasattr(runtime, forbidden), f"runtime view leaks {forbidden}"
    assert runtime.example_id == example.example_id
    assert runtime.question == example.question
    assert runtime.documents(), "runtime view must still expose its non-gold context"


@pytest.mark.parametrize("dataset,split", DATASETS)
def test_semantic_type_sidecar_covers_every_question(dataset: str, split: str) -> None:
    examples = get_adapter(dataset, split).load()
    types = load_semantic_types(dataset, examples=examples)
    assert len(types) == 500
    histogram = semantic_type_histogram(dataset)
    assert set(histogram) <= {"Fact", "Definition", "Reasoning"}
    assert sum(histogram.values()) == 500


def test_semantic_type_coverage_mismatch_is_fatal() -> None:
    examples = get_adapter("2wiki").load()
    tampered = examples[:-1]
    with pytest.raises(DatasetError, match="coverage mismatch"):
        load_semantic_types("2wiki", examples=tampered)


def test_registry_rejects_unknown_dataset_and_split() -> None:
    with pytest.raises(DatasetError, match="unknown dataset"):
        get_adapter("nq")
    with pytest.raises(DatasetError, match="unknown split"):
        get_adapter("hotpotqa", "train")


def test_doc_id_is_content_derived() -> None:
    baseline = content_doc_id("hotpotqa", "validation", "Title", "body text")
    assert baseline == content_doc_id("hotpotqa", "validation", "Title", "body text")
    assert baseline != content_doc_id("hotpotqa", "validation", "Title", "other text")
    assert baseline != content_doc_id("hotpotqa", "validation", "Other", "body text")
    # Dataset and split isolation: identical content in another corpus must not collide.
    assert baseline != content_doc_id("2wiki", "validation", "Title", "body text")
    assert baseline != content_doc_id("hotpotqa", "dev", "Title", "body text")


# ---------------------------------------------------------------------------
# Schema rejection paths (shared builder)
# ---------------------------------------------------------------------------


def _record(**overrides):
    record = {
        "id": "q1",
        "question": "Who?",
        "answer": "Someone",
        "type": "bridge",
        "level": "hard",
        "supporting_facts": {"title": ["A"], "sent_id": [0]},
        "context": {"title": ["A", "B"], "sentences": [["first.", "second."], ["other."]]},
    }
    record.update(overrides)
    return record


def test_builder_accepts_minimal_valid_record() -> None:
    examples = build_examples([_record()], dataset="synthetic", split="dev", evidence_kind="sentence")
    assert len(examples) == 1
    assert examples[0].gold_doc_ids and examples[0].hops() == 1


def test_builder_rejects_missing_field() -> None:
    record = _record()
    del record["answer"]
    with pytest.raises(DatasetError, match="missing fields"):
        build_examples([record], dataset="synthetic", split="dev", evidence_kind="sentence")


def test_builder_rejects_duplicate_ids() -> None:
    with pytest.raises(DatasetError, match="duplicate question ids"):
        build_examples([_record(), _record()], dataset="synthetic", split="dev", evidence_kind="sentence")


def test_builder_rejects_unknown_support_title() -> None:
    record = _record(supporting_facts={"title": ["Z"], "sent_id": [0]})
    with pytest.raises(DatasetError, match="not in the context"):
        build_examples([record], dataset="synthetic", split="dev", evidence_kind="sentence")


def test_builder_rejects_out_of_range_sentence() -> None:
    record = _record(supporting_facts={"title": ["A"], "sent_id": [7]})
    with pytest.raises(DatasetError, match="outside paragraph"):
        build_examples([record], dataset="synthetic", split="dev", evidence_kind="sentence")


def test_builder_rejects_paragraph_kind_with_sentence_index() -> None:
    record = _record(supporting_facts={"title": ["A"], "sent_id": [0]})
    with pytest.raises(DatasetError, match="expected sent_id -1"):
        build_examples([record], dataset="synthetic", split="dev", evidence_kind="paragraph")


def test_builder_rejects_empty_gold() -> None:
    record = _record(supporting_facts={"title": [], "sent_id": []})
    with pytest.raises(DatasetError, match="empty gold evidence"):
        build_examples([record], dataset="synthetic", split="dev", evidence_kind="sentence")


def test_builder_rejects_duplicate_titles_in_context() -> None:
    record = _record(
        context={"title": ["A", "A"], "sentences": [["x."], ["y."]]},
        supporting_facts={"title": ["A"], "sent_id": [0]},
    )
    with pytest.raises(DatasetError, match="duplicate paragraph titles"):
        build_examples([record], dataset="synthetic", split="dev", evidence_kind="sentence")


def test_required_fields_cover_every_adapter_field() -> None:
    assert set(REQUIRED_RECORD_FIELDS) == set(_record())
