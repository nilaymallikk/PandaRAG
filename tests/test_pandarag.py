"""PandaRAG tests: planner, evaluator, coverage, controller, context builder.

These assert the mechanisms the paper will claim, especially the ones that are easy
to get wrong silently: per-requirement coverage (not a global score), evidence
support versus relevance, gap-directed retrieval that targets only the uncovered
requirement, budget accounting, and evidence preservation in the final context.
"""

from __future__ import annotations

import json

import pytest

from src.datasets import get_adapter
from src.pipelines.pandarag import controller as controller_module
from src.pipelines.pandarag.context_builder import build_context
from src.pipelines.pandarag.coverage import PairVerdict, summarise
from src.pipelines.pandarag.evidence_evaluator import evaluate
from src.pipelines.pandarag.evidence_planner import (
    Requirement,
    plan_requirements,
    fallback_plan,
    normalise,
)
from src.pipelines.pandarag.pipeline import execute_question, load_pandarag_config
from src.pipelines.sac_rag.pipeline import load_sac_config
from src.pipelines.shared.llm_json import Usage
from src.pipelines.shared.retriever import RetrievedDoc
from tests.conftest import FakeClient, FakeService


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def test_planner_parses_valid_plan_and_keeps_dependencies() -> None:
    payload = json.dumps(
        {
            "requirements": [
                {"id": "E1", "description": "find founder", "search_query": "Who founded X?", "depends_on": []},
                {"id": "E2", "description": "find university", "search_query": "Which university?", "depends_on": ["E1"]},
            ]
        }
    )
    usage = Usage()
    plan = plan_requirements(FakeClient([payload]), "Which university did X's founder attend?", "Reasoning", usage=usage)
    assert not plan.fallback
    assert [requirement.id for requirement in plan.requirements] == ["E1", "E2"]
    assert plan.requirements[1].depends_on == ("E1",)
    assert usage.llm_calls == 1


def test_planner_skips_the_llm_for_simple_types() -> None:
    client = FakeClient(['{"requirements": []}'])
    usage = Usage()
    plan = plan_requirements(client, "When was X founded?", "Fact", usage=usage)
    assert usage.llm_calls == 0, "single-fact questions must not pay for a planner call"
    assert len(plan.requirements) == 1
    assert plan.requirements[0].search_query == "When was X founded?"


def test_planner_falls_back_on_malformed_output() -> None:
    plan = plan_requirements(FakeClient(["not json", "still not json"]), "q?", "Reasoning", usage=Usage())
    assert plan.fallback
    assert len(plan.requirements) == 1
    assert plan.requirements[0].search_query == "q?"
    assert plan.error


def test_planner_rejects_cycles_unknown_dependencies_and_duplicates() -> None:
    cycle = {
        "requirements": [
            {"id": "E1", "description": "a", "search_query": "qa", "depends_on": ["E2"]},
            {"id": "E2", "description": "b", "search_query": "qb", "depends_on": ["E1"]},
        ]
    }
    with pytest.raises(ValueError, match="cycle"):
        normalise(cycle)

    unknown = {"requirements": [{"id": "E1", "description": "a", "search_query": "qa", "depends_on": ["E9"]}]}
    with pytest.raises(ValueError, match="unknown ids"):
        normalise(unknown)

    duplicate = {
        "requirements": [
            {"id": "E1", "description": "a", "search_query": "qa"},
            {"id": "E1", "description": "b", "search_query": "qb"},
        ]
    }
    with pytest.raises(ValueError, match="duplicate"):
        normalise(duplicate)

    empty = {"requirements": [{"id": "E1", "description": "", "search_query": "qa"}]}
    with pytest.raises(ValueError, match="empty"):
        normalise(empty)


def test_planner_reports_overflow_instead_of_discarding_it() -> None:
    payload = {
        "requirements": [
            {"id": f"E{i}", "description": f"d{i}", "search_query": f"q{i}", "depends_on": []}
            for i in range(1, 7)
        ]
    }
    kept, overflow = normalise(payload, max_requirements=4)
    assert len(kept) == 4
    assert len(overflow) == 2
    assert overflow[0]["id"] == "E5"


def test_planner_rejects_dependency_on_a_dropped_requirement() -> None:
    payload = {
        "requirements": [
            {"id": "E1", "description": "a", "search_query": "qa"},
            {"id": "E2", "description": "b", "search_query": "qb"},
            {"id": "E3", "description": "c", "search_query": "qc"},
            {"id": "E4", "description": "d", "search_query": "qd"},
            {"id": "E5", "description": "e", "search_query": "qe", "depends_on": ["E5"]},
        ]
    }
    with pytest.raises(ValueError, match="self"):
        normalise(payload, max_requirements=4)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def _requirements() -> list[Requirement]:
    return [
        Requirement("E1", "founder", "who founded x"),
        Requirement("E2", "university", "which university", ("E1",)),
    ]


def test_coverage_is_per_requirement_not_a_global_score() -> None:
    coverage = summarise(
        _requirements(),
        [
            PairVerdict("E1", "d1", "supported"),
            PairVerdict("E2", "d2", "insufficient"),
        ],
    )
    assert coverage.n_supported == 1
    assert coverage.n_uncovered == 1
    assert coverage.uncovered_ids() == ["E2"]
    assert not coverage.all_covered
    assert coverage.coverage_ratio == pytest.approx(0.5)


def test_coverage_status_precedence_prefers_contradiction() -> None:
    coverage = summarise(
        _requirements(),
        [
            PairVerdict("E1", "d1", "supported"),
            PairVerdict("E1", "d2", "contradicted"),
        ],
    )
    assert coverage.per_requirement["E1"].status == "contradicted"
    assert coverage.n_contradicted == 1


def test_partial_evidence_is_not_counted_as_covered() -> None:
    coverage = summarise(
        _requirements(),
        [PairVerdict("E1", "d1", "supported"), PairVerdict("E2", "d2", "partial")],
    )
    assert coverage.all_covered is False
    assert coverage.uncovered_ids() == ["E2"]
    assert coverage.coverage_ratio == pytest.approx(0.75)


def test_coverage_accumulates_across_rounds() -> None:
    first = [PairVerdict("E1", "d1", "supported")]
    second = first + [PairVerdict("E2", "d3", "supported")]
    assert summarise(_requirements(), first).n_supported == 1
    assert summarise(_requirements(), second).n_supported == 2


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


def _state(**kwargs) -> controller_module.ControllerState:
    return controller_module.ControllerState(**kwargs)


def test_controller_stops_when_everything_is_covered() -> None:
    coverage = summarise(_requirements(), [PairVerdict("E1", "d1", "supported"), PairVerdict("E2", "d2", "supported")])
    decision = controller_module.decide(_requirements(), coverage, _state(), controller_module.Budget())
    assert decision.action == controller_module.STOP
    assert decision.stop_reason == "all_requirements_covered"


def test_controller_targets_the_uncovered_requirement_only() -> None:
    coverage = summarise(_requirements(), [PairVerdict("E1", "d1", "supported"), PairVerdict("E2", "d2", "insufficient")])
    decision = controller_module.decide(_requirements(), coverage, _state(), controller_module.Budget())
    assert decision.action == controller_module.TARGETED_DENSE
    assert decision.target_requirement == "E2"
    assert decision.query == "which university"
    assert decision.top_k >= 1


def test_controller_respects_round_document_and_execution_budgets() -> None:
    coverage = summarise(_requirements(), [PairVerdict("E1", "d1", "insufficient")])
    budget = controller_module.Budget()

    assert controller_module.decide(
        _requirements(), coverage, _state(rounds_used=2), budget
    ).stop_reason == "max_rounds"
    assert controller_module.decide(
        _requirements(), coverage, _state(unique_documents=20), budget
    ).stop_reason == "max_documents"
    assert controller_module.decide(
        _requirements(), coverage, _state(retriever_executions=8), budget
    ).stop_reason == "max_retriever_executions"


def test_controller_stops_when_a_round_does_not_improve_coverage() -> None:
    coverage = summarise(_requirements(), [PairVerdict("E1", "d1", "insufficient")])
    decision = controller_module.decide(
        _requirements(), coverage, _state(rounds_used=1, last_improved=False), controller_module.Budget()
    )
    assert decision.stop_reason == "no_improvement"


def test_controller_does_not_repeat_the_initial_retriever() -> None:
    """A query the initial round already ran on dense must not be re-run on dense."""
    coverage = summarise(_requirements(), [PairVerdict("E1", "d1", "insufficient")])
    state = _state(initial_modes_by_query={"who founded x": ["dense"]})
    decision = controller_module.decide(_requirements(), coverage, state, controller_module.Budget())
    assert decision.action == controller_module.TARGETED_BM25
    assert decision.mode == "bm25"


def test_controller_expands_the_pool_when_both_retrievers_are_exhausted() -> None:
    coverage = summarise(_requirements(), [PairVerdict("E1", "d1", "insufficient")])
    state = _state(initial_modes_by_query={"who founded x": ["dense", "bm25"]})
    decision = controller_module.decide(_requirements(), coverage, state, controller_module.Budget())
    assert decision.action == controller_module.EXPAND_POOL


def test_controller_switches_retriever_after_dense_miss() -> None:
    coverage = summarise(_requirements(), [PairVerdict("E1", "d1", "insufficient")])
    state = _state()
    state.record_attempt("E1", "dense")
    decision = controller_module.decide(_requirements(), coverage, state, controller_module.Budget())
    assert decision.action == controller_module.TARGETED_BM25
    assert decision.mode == "bm25"


def test_controller_never_overspends_the_document_cap() -> None:
    coverage = summarise(_requirements(), [PairVerdict("E1", "d1", "insufficient")])
    decision = controller_module.decide(
        _requirements(), coverage, _state(unique_documents=18), controller_module.Budget()
    )
    assert decision.top_k == 2


def test_controller_stops_on_unresolvable_contradiction() -> None:
    coverage = summarise(_requirements(), [PairVerdict("E1", "d1", "contradicted")])
    decision = controller_module.decide(_requirements(), coverage, _state(), controller_module.Budget())
    assert decision.stop_reason == "contradiction_unresolved"


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def _documents() -> list[RetrievedDoc]:
    return [
        RetrievedDoc("d1", "Founder", "the founder of company x was jane doe", 1, rerank_score=0.9),
        RetrievedDoc("d2", "University", "jane doe graduated from springfield university", 2, rerank_score=0.8),
        RetrievedDoc("d3", "Rivers", "rivers are wet", 3, rerank_score=0.7),
        RetrievedDoc("d4", "Bridge", "was born in springfield", 4, rerank_score=0.1),
    ]


def test_context_builder_preserves_covered_requirement_evidence_even_if_low_ranked() -> None:
    coverage = summarise(
        _requirements(),
        [PairVerdict("E1", "d4", "supported"), PairVerdict("E2", "d2", "supported")],
    )
    selection = build_context(_requirements(), coverage, _documents(), max_tokens=4000)
    ids = [document["doc_id"] for document in selection.documents]
    assert "d4" in ids, "low-relevance bridge evidence must not be dropped"
    assert "d2" in ids
    assert selection.requirement_documents["E1"] == ["d4"]


def test_context_builder_keeps_rerank_order_while_selecting_by_coverage() -> None:
    coverage = summarise(
        _requirements(),
        [PairVerdict("E1", "d4", "supported"), PairVerdict("E2", "d2", "supported")],
    )
    selection = build_context(_requirements(), coverage, _documents(), max_tokens=4000)
    ids = [document["doc_id"] for document in selection.documents]
    # d4 (the lowest-ranked bridge passage) is included, but relevance order wins.
    assert "d4" in ids
    assert ids.index("d2") < ids.index("d4")


def test_context_builder_fills_a_bounded_window_but_never_drops_evidence() -> None:
    coverage = summarise(
        _requirements(),
        [PairVerdict("E1", "d4", "supported"), PairVerdict("E2", "d2", "supported")],
    )
    small = build_context(_requirements(), coverage, _documents(), max_tokens=4000, base_top_k=2)
    large = build_context(_requirements(), coverage, _documents(), max_tokens=4000, base_top_k=10)
    assert {"d4", "d2"} <= {document["doc_id"] for document in small.documents}
    assert len(small.documents) == 2
    assert len(large.documents) == 4  # the whole (tiny) synthetic corpus


def test_context_builder_dedupes_and_respects_the_token_budget() -> None:
    coverage = summarise(
        _requirements(),
        [PairVerdict("E1", "d1", "supported"), PairVerdict("E2", "d2", "supported")],
    )
    # A budget so small that even one passage cannot fit: drops must be recorded,
    # never silent, and the minimum-passage guarantee must not blow the budget.
    selection = build_context(_requirements(), coverage, _documents(), max_tokens=5)
    ids = [document["doc_id"] for document in selection.documents]
    assert len(ids) == len(set(ids))
    assert selection.tokens_estimate <= 5
    assert selection.dropped, "over-budget passages must be recorded, not silently dropped"


def test_context_builder_always_keeps_one_passage_per_requirement() -> None:
    coverage = summarise(_requirements(), [PairVerdict("E1", "d1", "supported")])
    selection = build_context(_requirements(), coverage, _documents(), max_tokens=200)
    assert selection.documents
    assert selection.requirement_documents.get("E1")


# ---------------------------------------------------------------------------
# Evidence evaluator
# ---------------------------------------------------------------------------


def _service_with_docs() -> FakeService:
    fake = FakeService()
    fake.add_documents([("d1", "Founder", "jane doe founded company x"), ("d2", "Other", "unrelated")])
    return fake


def test_evaluator_keeps_supported_and_insufficient_apart() -> None:
    documents = _service_with_docs().search("q", "dense", 2)
    payload = json.dumps(
        {
            "verdicts": [
                {"requirement_id": "E1", "document_id": "d1", "support_status": "supported", "supporting_spans": ["jane doe founded company x"], "reason": "states the founder"},
                {"requirement_id": "E2", "document_id": "d1", "support_status": "insufficient", "reason": "no university"},
            ]
        }
    )
    verdicts = evaluate(FakeClient([payload]), "q", _requirements(), documents, usage=Usage())
    assert not verdicts.fallback
    assert {verdict.requirement_id: verdict.support_status for verdict in verdicts.verdicts} == {
        "E1": "supported",
        "E2": "insufficient",
    }


def test_evaluator_drops_verdicts_for_unknown_ids() -> None:
    documents = _service_with_docs().search("q", "dense", 2)
    payload = json.dumps(
        {
            "verdicts": [
                {"requirement_id": "E9", "document_id": "d1", "support_status": "supported"},
                {"requirement_id": "E1", "document_id": "d9", "support_status": "supported"},
                {"requirement_id": "E1", "document_id": "d1", "support_status": "partial"},
            ]
        }
    )
    result = evaluate(FakeClient([payload]), "q", _requirements(), documents, usage=Usage())
    assert [verdict.requirement_id for verdict in result.verdicts] == ["E1"]
    assert result.verdicts[0].support_status == "partial"


def test_evaluator_never_receives_gold_and_fails_safe() -> None:
    documents = _service_with_docs().search("q", "dense", 2)
    client = FakeClient(["garbage", "garbage"])
    result = evaluate(client, "which university did the founder attend", _requirements(), documents, usage=Usage())
    assert result.fallback
    assert result.verdicts == []
    prompt = client.last_prompt.lower()
    assert "gold" not in prompt
    assert "answer:" not in prompt


# ---------------------------------------------------------------------------
# End-to-end trace (fake client)
# ---------------------------------------------------------------------------


def test_pandarag_records_the_full_trace_and_stops_within_budget() -> None:
    example = get_adapter("hotpotqa").load()[0]
    service = FakeService()
    service.add_documents(
        [
            ("d1", "A", "cold harbor was fought in june 1864"),
            ("d2", "B", "the second battle of bull run was fought in august 1862"),
            ("d3", "C", "unrelated rivers"),
            ("d4", "D", "the second battle of bull run was fought in august 1862 by lee"),
        ]
    )
    # Targeted retrieval for the uncovered requirement must surface a NEW document,
    # otherwise the controller correctly stops with "no_improvement" and no
    # recovery round is recorded.
    service.search_results[("when was cold harbor", "dense")] = ["d1", "d3"]
    service.search_results[("when was bull run", "dense")] = ["d2", "d3"]
    plan = json.dumps(
        {
            "requirements": [
                {"id": "E1", "description": "date of cold harbor", "search_query": "when was cold harbor", "depends_on": []},
                {"id": "E2", "description": "date of bull run", "search_query": "when was bull run", "depends_on": []},
            ]
        }
    )
    verdicts = json.dumps(
        {
            "verdicts": [
                {"requirement_id": "E1", "document_id": "d1", "support_status": "supported", "reason": "date given"},
                {"requirement_id": "E2", "document_id": "d2", "support_status": "insufficient", "reason": "missing"},
            ]
        }
    )
    # call order: planner, initial evaluation, recovery evaluation, generation
    service.search_results[("when was bull run", "bm25")] = ["d2", "d4"]
    improved = json.dumps(
        {
            "verdicts": [
                {"requirement_id": "E2", "document_id": "d4", "support_status": "supported", "reason": "date given"},
            ]
        }
    )
    # call order: planner, initial evaluation, recovery evaluation, generation
    client = FakeClient([plan, verdicts, improved, "final answer"])

    payload = execute_question(
        example,
        service,
        client,
        semantic_type="Reasoning",
        pandarag_config=load_pandarag_config(),
        sac_config=load_sac_config(),
    )
    trace = payload["pandarag"]
    assert trace["plan"]["fallback"] is False
    assert trace["initial_coverage"]["n_supported"] == 1
    assert trace["budget"]["used_additional_rounds"] <= 2
    assert trace["budget"]["used_unique_documents"] <= 20
    assert trace["budget"]["used_retriever_executions"] <= 8
    assert trace["context_selection"]["tokens_estimate"] <= 4000
    assert any(item.get("kind") == "recovery" for item in trace["rounds"])
    assert trace["insufficiency_note"] in (True, False)
    assert payload["prediction"] == "final answer"
