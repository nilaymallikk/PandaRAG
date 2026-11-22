"""Gap-directed retrieval controller (PandaRAG component D) — the contribution.

The controller is deliberately rule-based, not learned: the research claim is about
*where* the remaining budget goes, and that has to be inspectable. Every decision
carries a machine-readable reason and the budget counters that produced it.

Budget semantics (frozen in ``configs/pandarag.json``):

* ``max_additional_rounds = 2`` — two rounds **after** the initial retrieval.
* ``max_cumulative_unique_documents = 20`` — unique documents admitted to the
  candidate pool across all rounds (not the raw pre-rerank union).
* ``max_total_retriever_executions = 8`` — every individual dense or BM25 search
  counts, including the initial round.
* ``max_generator_context_tokens = 4000`` — enforced by the context builder.

Stop conditions are checked before any target is chosen, so the controller can
never overspend by "finishing" a round it cannot afford.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from src.pipelines.pandarag.coverage import Coverage
from src.pipelines.pandarag.evidence_planner import Requirement

STOP = "STOP"
TARGETED_DENSE = "TARGETED_DENSE_RETRIEVAL"
TARGETED_BM25 = "TARGETED_BM25_RETRIEVAL"
EXPAND_POOL = "EXPAND_CANDIDATE_POOL"

_STATUS_PRIORITY = {"partial": 0, "insufficient": 1, "unknown": 2, "contradicted": 3}
MIN_NEW_DOCUMENTS = 5
EXPANDED_TOP_K = 15


@dataclass
class Budget:
    max_additional_rounds: int = 2
    max_cumulative_unique_documents: int = 20
    max_total_retriever_executions: int = 8
    max_generator_context_tokens: int = 4000
    max_requirements: int = 4


@dataclass
class ControllerState:
    rounds_used: int = 0
    unique_documents: int = 0
    retriever_executions: int = 0
    attempted_modes: dict[str, list[str]] = field(default_factory=dict)
    #: retriever modes already used for each *query* by the initial retrieval.
    #: Re-running the same query on the same retriever cannot add candidates, so
    #: the controller must switch modality instead of repeating itself.
    initial_modes_by_query: dict[str, list[str]] = field(default_factory=dict)
    last_improved: bool = True
    history: list[dict[str, Any]] = field(default_factory=list)

    def record_attempt(self, requirement_id: str, mode: str) -> None:
        self.attempted_modes.setdefault(requirement_id, []).append(mode)

    def modes_for(self, requirement_id: str, query: str) -> set[str]:
        used = set(self.attempted_modes.get(requirement_id, []))
        used.update(self.initial_modes_by_query.get(query, []))
        return used


@dataclass
class Decision:
    action: str
    reason: str
    target_requirement: str | None = None
    query: str | None = None
    mode: str | None = None
    top_k: int = MIN_NEW_DOCUMENTS
    stop_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "target_requirement": self.target_requirement,
            "query": self.query,
            "retrieval_mode": self.mode,
            "additional_top_k": self.top_k if self.action != STOP else 0,
            "stop_reason": self.stop_reason,
        }


def _choose(requirements: Sequence[Requirement], coverage: Coverage) -> Requirement | None:
    pending = coverage.needs_retrieval()
    if not pending:
        return None
    pending = sorted(
        pending,
        key=lambda requirement: (
            _STATUS_PRIORITY.get(coverage.per_requirement[requirement.id].status, 9),
            requirement.id,
        ),
    )
    return pending[0]


def decide(
    requirements: Sequence[Requirement],
    coverage: Coverage,
    state: ControllerState,
    budget: Budget,
) -> Decision:
    """Decide the next action for one question."""
    if coverage.all_covered:
        return Decision(STOP, "all requirements are supported", stop_reason="all_requirements_covered")

    if state.rounds_used >= budget.max_additional_rounds:
        return Decision(STOP, "additional round budget exhausted", stop_reason="max_rounds")

    if state.unique_documents >= budget.max_cumulative_unique_documents:
        return Decision(STOP, "unique document budget exhausted", stop_reason="max_documents")

    if state.retriever_executions + 1 > budget.max_total_retriever_executions:
        return Decision(STOP, "retriever execution budget exhausted", stop_reason="max_retriever_executions")

    if state.rounds_used > 0 and not state.last_improved:
        return Decision(
            STOP, "the last round did not increase requirement coverage", stop_reason="no_improvement"
        )

    target = _choose(requirements, coverage)
    if target is None:
        return Decision(STOP, "no requirement can be targeted", stop_reason="no_targetable_requirement")

    status = coverage.per_requirement[target.id].status
    if status == "contradicted":
        return Decision(
            STOP,
            f"requirement {target.id} is contradicted; further retrieval cannot resolve it",
            stop_reason="contradiction_unresolved",
        )

    tried = state.modes_for(target.id, target.search_query)
    if "dense" not in tried:
        mode, action, why = "dense", TARGETED_DENSE, "dense has not been tried for this gap"
    elif "bm25" not in tried:
        mode, action, why = "bm25", TARGETED_BM25, "dense is already exhausted for this query; switching to lexical retrieval"
    else:
        mode, action, why = "dense", EXPAND_POOL, "both retrievers were tried; widening the pool beyond the initial depth"

    remaining = budget.max_cumulative_unique_documents - state.unique_documents
    wanted = MIN_NEW_DOCUMENTS if action != EXPAND_POOL else EXPANDED_TOP_K
    top_k = max(1, min(wanted, remaining))

    return Decision(
        action=action,
        reason=f"requirement {target.id} is {status}; {why}",
        target_requirement=target.id,
        query=target.search_query,
        mode=mode,
        top_k=top_k,
    )
