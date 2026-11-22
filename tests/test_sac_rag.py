"""SAC-RAG Adaptive (P4) and Full SAC-RAG (P5) tests.

Covers the pieces that decide experimental validity: classification source, the
three retrieval strategies, decomposition fallback, reranker ordering, compression
attribution, and the pipeline record shape.
"""

from __future__ import annotations

import json

import pytest

from src.datasets import get_adapter
from src.pipelines.sac_rag import classifier, decomposition
from src.pipelines.sac_rag.context_compression import compress
from src.pipelines.sac_rag.pipeline import execute_question, load_sac_config
from src.pipelines.sac_rag.retrieval_strategies import (
    retrieve_definition,
    retrieve_fact,
    retrieve_reasoning,
)
from src.pipelines.shared.llm_json import Usage
from tests.conftest import FakeClient, FakeService


@pytest.fixture()
def service() -> FakeService:
    fake = FakeService()
    fake.add_documents(
        [
            ("d1", "Cold Harbor", "the battle of cold harbor was fought in june 1864"),
            ("d2", "Bull Run", "the second battle of bull run was fought in august 1862"),
            ("d3", "Rivers", "an unrelated paragraph about rivers and mountains"),
            ("d4", "Lee", "robert e lee commanded the army of northern virginia"),
        ]
    )
    return fake


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_frozen_labels_cover_every_question_and_never_include_gold() -> None:
    labels = classifier.load_frozen_labels("hotpotqa")
    examples = get_adapter("hotpotqa").load()
    assert len(labels) == len(examples) == 500
    assert {item["question_type"] for item in labels.values()} <= set(classifier.LABELS)
    assert all(item["source"] == "frozen_sidecar" for item in labels.values())
    # The sidecar path must not carry an answer field of any kind.
    assert all(set(item) == {"question_type", "source", "prompt_version"} for item in labels.values())


def test_keyword_fallback_matches_published_cue_lists() -> None:
    assert classifier.keyword_fallback("When was the battle fought?") == "Fact"
    assert classifier.keyword_fallback("Which river is longer?") == "Fact"
    assert classifier.keyword_fallback("Who is Ada Lovelace?") == "Definition"
    assert classifier.keyword_fallback("What is a monad?") == "Definition"
    assert classifier.keyword_fallback("Why did the empire collapse?") == "Reasoning"
    assert classifier.keyword_fallback("Tell me about it") == classifier.FALLBACK_LABEL


def test_runtime_classifier_validates_schema_and_rejects_bad_labels() -> None:
    client = FakeClient(['{"question_type": "reasoning", "confidence": 0.91}'])
    result = classifier.classify(client, "Who founded the company that built X?")
    assert result.ok
    assert result.data == {"question_type": "Reasoning", "confidence": 0.91}

    bad = FakeClient(['{"question_type": "vibes", "confidence": 1}'])
    assert not classifier.classify(bad, "anything").ok


def test_runtime_classifier_never_sees_gold_fields() -> None:
    client = FakeClient(['{"question_type": "fact", "confidence": 0.5}'])
    classifier.classify(client, "Which city hosts the museum?")
    prompt = client.last_prompt.lower()
    for forbidden in ("supporting_facts", "gold_answer", "exact_match", "f1 score"):
        assert forbidden not in prompt
    assert prompt.count("question:") == 1


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def test_decomposition_parses_and_caps_subquestions() -> None:
    payload = json.dumps({"subquestions": ["q1", "q2", "q3", "q4", "q5", "q6"]})
    result = decomposition.decompose(FakeClient([payload]), "complex question")
    assert not result.fallback
    assert len(result.subquestions) == decomposition.MAX_SUBQUESTIONS


def test_decomposition_falls_back_to_the_original_question() -> None:
    result = decomposition.decompose(FakeClient(["not json at all"]), "the original question")
    assert result.fallback
    assert result.subquestions == ["the original question"]
    assert result.error


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------


def test_fact_strategy_merges_dense_and_bm25_then_reranks(service: FakeService) -> None:
    result = retrieve_fact(service, "cold harbor 1864", rerank=True)
    assert result.strategy == "fact"
    assert {("cold harbor 1864", "dense"), ("cold harbor 1864", "bm25")} <= {
        (query, mode) for query, mode, _ in service.search_calls
    }
    assert result.retrieval_executions == 2
    assert len({document.doc_id for document in result.candidates}) == len(result.candidates)
    assert result.selected
    assert [document.rank for document in result.selected] == list(range(1, len(result.selected) + 1))


def test_definition_strategy_uses_dense_only(service: FakeService) -> None:
    retrieve_definition(service, "what is a monad")
    modes = [mode for _, mode, _ in service.search_calls]
    assert modes == ["dense"]


def test_definition_strategy_can_skip_the_reranker(service: FakeService) -> None:
    result = retrieve_definition(service, "what is a monad", rerank=False)
    assert result.extra["rerank"] is False
    assert result.ranked


def test_reasoning_strategy_queries_every_subquestion_and_dedupes(service: FakeService) -> None:
    result = retrieve_reasoning(service, "compare A and B", ["when was A", "when was B"])
    queries = [query for query, _, _ in service.search_calls]
    assert queries == ["when was A", "when was B"]
    assert result.retrieval_executions == 2
    assert result.queries == ["when was A", "when was B"]
    ids = [document.doc_id for document in result.candidates]
    assert len(ids) == len(set(ids))


def test_reranker_ordering_is_deterministic(service: FakeService) -> None:
    first = retrieve_fact(service, "bull run 1862")
    second = retrieve_fact(service, "bull run 1862")
    assert [document.doc_id for document in first.selected] == [
        document.doc_id for document in second.selected
    ]


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def _docs(service: FakeService):
    return service.search("question", "dense", 3)


def test_compression_keeps_attribution_and_measures_ratio(service: FakeService) -> None:
    documents = _docs(service)
    payload = json.dumps(
        {"snippets": [{"doc_id": documents[0].doc_id, "text": "the battle was fought in august 1862"}]}
    )
    usage = Usage()
    result = compress(FakeClient([payload]), "when was it fought", documents, usage=usage)
    assert not result.fallback
    assert result.doc_ids_retained == [documents[0].doc_id]
    assert result.compressed_tokens <= result.original_tokens
    assert 0.0 < result.ratio <= 1.0
    assert usage.llm_calls == 1


def test_compression_rejects_unknown_document_ids(service: FakeService) -> None:
    payload = json.dumps({"snippets": [{"doc_id": "made-up", "text": "fabricated"}]})
    result = compress(FakeClient([payload, payload]), "q", _docs(service), usage=Usage())
    assert result.fallback
    assert result.error


def test_compression_failure_returns_uncompressed_context(service: FakeService) -> None:
    documents = _docs(service)
    result = compress(FakeClient(["garbage", "garbage"]), "q", documents, usage=Usage())
    assert result.fallback
    assert [document["doc_id"] for document in result.documents] == [d.doc_id for d in documents]


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


def _first_example(dataset: str = "hotpotqa"):
    return get_adapter(dataset).load()[0]


def test_execute_question_produces_transformative_record_payload(service: FakeService) -> None:
    example = _first_example()
    client = FakeClient(['{"subquestions": ["when was A", "when was B"]}', "Short answer"])
    payload = execute_question(
        example,
        service,
        client,
        semantic_type="Reasoning",
        sac_config=load_sac_config(),
        compression_enabled=False,
    )
    assert payload["prediction"] == "Short answer"
    assert payload["retrieval_rounds"] == 1
    assert payload["sac_rag"]["strategy"]["strategy"] == "reasoning"
    assert payload["sac_rag"]["decomposition"]["fallback"] is False
    assert payload["usage"].llm_calls >= 2
    assert payload["context_documents"]
    # Gold must never reach the runtime payload.
    assert example.answer not in json.dumps(payload.get("context_documents", []))


def test_execute_question_with_compression_marks_it_in_the_payload(service: FakeService) -> None:
    example = _first_example()
    documents = service.search("q", "dense", 2)
    snippet = json.dumps({"snippets": [{"doc_id": documents[0].doc_id, "text": "condensed"}]})
    client = FakeClient([snippet, "Answer"])
    payload = execute_question(
        example,
        service,
        client,
        semantic_type="Fact",
        sac_config=load_sac_config(),
        compression_enabled=True,
    )
    assert payload["sac_rag"]["compression"] is not None
    assert "compression_ratio" in payload["sac_rag"]["compression"]
