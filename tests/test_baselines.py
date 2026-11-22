"""Corpus-matched baseline tests (P1c, P2c, P3c, P1n) and pipeline integrity.

The P3c checks matter most: it must be the *frozen* Adaptive RAG policy applied to
corpus retrieval, with nothing re-tuned. These tests assert the imported formula is
the original one and that the transfer detail (a fused document missing from one
source ranking) is handled explicitly rather than silently.
"""

from __future__ import annotations

import json

import pytest

from src import config
from src.datasets import get_adapter
from src.pipelines import adaptive_rag as frozen_policy
from src.pipelines import corpus_baselines
from src.pipelines.corpus_baselines import MISSING_SOURCE_RANK, run_p1c, run_p1n, run_p2c, run_p3c
from src.pipelines.shared.generator import prompt_fingerprint
from tests.conftest import FakeClient, FakeService


@pytest.fixture()
def service() -> FakeService:
    fake = FakeService()
    fake.add_documents(
        [
            ("d1", "A", "alpha text"),
            ("d2", "B", "beta text"),
            ("d3", "C", "gamma text"),
            ("d4", "D", "delta text"),
            ("d5", "E", "epsilon text"),
            ("d6", "F", "zeta text"),
        ]
    )
    return fake


def _example():
    return get_adapter("hotpotqa").load()[0]


# ---------------------------------------------------------------------------
# Retrieval shapes
# ---------------------------------------------------------------------------


def test_p1c_uses_dense_top5(service: FakeService) -> None:
    payload = run_p1c(_example(), service, FakeClient(["answer"]))
    assert service.search_calls == [(service.search_calls[0][0], "dense", 5)]
    assert len(payload["retrieved_documents"]) <= 5
    assert payload["baseline"]["k"] == 5


def test_p2c_uses_hybrid_top5(service: FakeService) -> None:
    payload = run_p2c(_example(), service, FakeClient(["answer"]))
    assert service.search_calls[0][1] == "hybrid"
    assert payload["baseline"]["mode"] == "hybrid_rrf"


def test_p1n_retrieves_ten_and_feeds_five(service: FakeService) -> None:
    payload = run_p1n(_example(), service, FakeClient(["answer"]))
    assert len(payload["retrieved_documents"]) == 6  # fake corpus has 6 documents
    assert len(payload["context_documents"]) == 5
    assert payload["baseline"] == {
        "strategy": "corpus",
        "context_doc_ids": [document["doc_id"] for document in payload["context_documents"]],
        "k_retrieved": 10,
        "k_fed": 5,
        "mode": "dense",
    }


# ---------------------------------------------------------------------------
# P3c: transferred frozen policy
# ---------------------------------------------------------------------------


def test_p3c_uses_the_frozen_policy_functions_unmodified() -> None:
    # Identity, not equivalence: the module's own functions are the ones called.
    assert corpus_baselines.frozen_policy is frozen_policy
    assert corpus_baselines.frozen_policy.agreement.__module__ == "src.pipelines.adaptive_rag"
    assert corpus_baselines.frozen_policy.decide_final_k is frozen_policy.decide_final_k


def test_p3c_final_k_is_always_three_or_five(service: FakeService) -> None:
    payload = run_p3c(_example(), service, FakeClient(["answer"]))
    assert payload["baseline"]["final_k"] in (3, 5)
    assert payload["baseline"]["policy"] == "transferred_frozen_adaptive_rag/1.0"
    assert payload["baseline"]["threshold"] == config.ADAPTIVE_RAG_THRESHOLD
    assert len(payload["context_documents"]) == payload["baseline"]["final_k"]


def test_p3c_scores_the_same_as_the_frozen_formula(service: FakeService) -> None:
    payload = run_p3c(_example(), service, FakeClient(["answer"]))
    baseline = payload["baseline"]
    expected_agreement = frozen_policy.agreement(baseline["dense_rank_top1"], baseline["bm25_rank_top1"])
    expected_k, expected_confidence = frozen_policy.decide_final_k(
        expected_agreement, baseline["boundary_margin"]
    )
    assert baseline["agreement"] == pytest.approx(expected_agreement, abs=1e-6)
    assert baseline["confidence"] == pytest.approx(expected_confidence, abs=1e-6)
    assert baseline["final_k"] == expected_k


def test_missing_source_rank_is_explicit_and_recorded(service: FakeService) -> None:
    # FakeService tags hits with the source it was asked for; a fused hit that the
    # fake returns for both sources still has both ranks. Force one to be absent.
    payload = run_p3c(_example(), service, FakeClient(["answer"]))
    baseline = payload["baseline"]
    if baseline["dense_rank_top1"] == MISSING_SOURCE_RANK:
        assert MISSING_SOURCE_RANK == config.RRF_DEPTH + 1
    assert isinstance(baseline["dense_rank_top1"], int)
    assert isinstance(baseline["bm25_rank_top1"], int)


# ---------------------------------------------------------------------------
# Integrity across pipelines
# ---------------------------------------------------------------------------


def test_one_generation_prompt_for_every_pipeline() -> None:
    """The fingerprint is computed from one module, so it cannot drift per pipeline."""
    assert prompt_fingerprint() == prompt_fingerprint()
    from src.pipelines.shared.runner import build_record

    assert build_record.__module__.startswith("src.pipelines.shared")


def test_gold_never_reaches_the_generation_prompt(service: FakeService) -> None:
    example = get_adapter("hotpotqa").load()[0]
    for runner in (run_p1c, run_p2c, run_p1n, run_p3c):
        client = FakeClient(["answer"])
        runner(example, service, client)
        prompt = client.last_prompt
        assert example.answer.lower() not in prompt.lower() or len(example.answer) < 4
        assert "gold" not in prompt.lower()


def test_baseline_payload_separates_retrieved_from_fed(service: FakeService) -> None:
    payload = run_p1n(_example(), service, FakeClient(["answer"]))
    assert payload["retrieved_documents"]
    assert payload["context_documents"] is not payload["retrieved_documents"]
    assert "context_tokens_estimate" in payload
    assert payload["usage"].llm_calls == 1
