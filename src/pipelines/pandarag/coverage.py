"""Evidence coverage tracking (PandaRAG component C).

Coverage is tracked **per requirement**, never as a single global retrieval score.
That is the whole point of the contribution: a question can have high average
relevance while one bridge fact is missing, and only requirement-level statuses
reveal it.

Status precedence when several documents bear on one requirement:

    contradicted > supported > partial > insufficient > unknown

Rationale: a contradiction must never be hidden by a supportive document, and a
requirement with no evaluated document is unknown rather than unsupported.

Coverage counts only *evidence*: ``partial`` contributes half, and nothing is ever
credited to a requirement because another requirement was satisfied. (The planner
is instructed not to emit synthesis-only requirements precisely so this stays
honest: a comparison is performed by the generator, not counted as evidence.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from src.pipelines.pandarag.evidence_planner import Requirement

STATUSES = ("supported", "partial", "insufficient", "contradicted", "unknown")
_PRECEDENCE = {"contradicted": 4, "supported": 3, "partial": 2, "insufficient": 1, "unknown": 0}
_COVERING = {"supported"}


@dataclass(frozen=True)
class PairVerdict:
    requirement_id: str
    document_id: str
    support_status: str
    reason: str = ""
    relevance_score: float | None = None
    supporting_spans: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "document_id": self.document_id,
            "support_status": self.support_status,
            "reason": self.reason,
            "relevance_score": self.relevance_score,
            "supporting_spans": list(self.supporting_spans),
        }


@dataclass
class RequirementCoverage:
    requirement_id: str
    status: str
    supporting_document_ids: list[str] = field(default_factory=list)

    @property
    def covered(self) -> bool:
        return self.status in _COVERING

    def as_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "status": self.status,
            "covered": self.covered,
            "supporting_document_ids": list(self.supporting_document_ids),
        }


@dataclass
class Coverage:
    requirements: list[Requirement]
    per_requirement: dict[str, RequirementCoverage]
    verdicts: list[PairVerdict] = field(default_factory=list)

    @property
    def n_supported(self) -> int:
        return sum(1 for item in self.per_requirement.values() if item.status == "supported")

    @property
    def n_partial(self) -> int:
        return sum(1 for item in self.per_requirement.values() if item.status == "partial")

    @property
    def n_contradicted(self) -> int:
        return sum(1 for item in self.per_requirement.values() if item.status == "contradicted")

    @property
    def n_unknown(self) -> int:
        return sum(1 for item in self.per_requirement.values() if item.status == "unknown")

    @property
    def n_uncovered(self) -> int:
        return len(self.per_requirement) - self.n_supported

    @property
    def coverage_ratio(self) -> float:
        if not self.per_requirement:
            return 0.0
        # Partial evidence counts as half — stated explicitly, never implicit.
        score = self.n_supported + 0.5 * self.n_partial
        return round(score / len(self.per_requirement), 4)

    @property
    def all_covered(self) -> bool:
        return bool(self.per_requirement) and self.n_supported == len(self.per_requirement)

    def uncovered_ids(self) -> list[str]:
        return [
            requirement.id
            for requirement in self.requirements
            if not self.per_requirement.get(
                requirement.id, RequirementCoverage(requirement.id, "unknown")
            ).covered
        ]

    def needs_retrieval(self) -> list[Requirement]:
        """Requirements worth another retrieval round, dependencies first."""
        pending: list[Requirement] = []
        for requirement in self.requirements:
            item = self.per_requirement.get(requirement.id)
            if item is not None and item.covered:
                continue
            parents_covered = all(
                self.per_requirement.get(dep) is not None and self.per_requirement[dep].covered
                for dep in requirement.depends_on
            )
            # A dependent requirement cannot be resolved before its parent.
            if requirement.depends_on and not parents_covered:
                continue
            pending.append(requirement)
        return pending

    def as_dict(self) -> dict[str, Any]:
        return {
            "per_requirement": {
                key: value.as_dict() for key, value in sorted(self.per_requirement.items())
            },
            "n_requirements": len(self.per_requirement),
            "n_supported": self.n_supported,
            "n_partial": self.n_partial,
            "n_uncovered": self.n_uncovered,
            "n_contradicted": self.n_contradicted,
            "n_unknown": self.n_unknown,
            "coverage_ratio": self.coverage_ratio,
            "all_covered": self.all_covered,
            "uncovered_requirement_ids": self.uncovered_ids(),
            "n_verdicts": len(self.verdicts),
        }


def summarise(
    requirements: Sequence[Requirement],
    verdicts: Iterable[PairVerdict],
) -> Coverage:
    """Fold pair verdicts into one status per requirement."""
    per: dict[str, RequirementCoverage] = {
        requirement.id: RequirementCoverage(requirement.id, "unknown") for requirement in requirements
    }
    verdicts = list(verdicts)
    for verdict in verdicts:
        if verdict.requirement_id not in per:
            continue
        if verdict.support_status not in _PRECEDENCE:
            continue
        item = per[verdict.requirement_id]
        if _PRECEDENCE[verdict.support_status] > _PRECEDENCE[item.status]:
            item.status = verdict.support_status
        if verdict.support_status in ("supported", "partial") and verdict.document_id not in item.supporting_document_ids:
            item.supporting_document_ids.append(verdict.document_id)

    return Coverage(requirements=list(requirements), per_requirement=per, verdicts=verdicts)


def comparison(old: Coverage | None, new: Coverage) -> dict[str, Any]:
    """Coverage delta between two rounds (used by the controller's stall check)."""
    if old is None:
        return {
            "supported_before": 0,
            "supported_after": new.n_supported,
            "delta_supported": new.n_supported,
            "improved": new.n_supported > 0,
        }
    return {
        "supported_before": old.n_supported,
        "supported_after": new.n_supported,
        "delta_supported": new.n_supported - old.n_supported,
        "improved": new.n_supported > old.n_supported,
    }
