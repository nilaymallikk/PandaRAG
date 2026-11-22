"""Evidence requirement planning (PandaRAG component A).

Turns a question into atomic, independently verifiable evidence requirements. The
planner is the only place that decides *what needs to be known*; it never answers,
never presumes a fact, and never sees gold data.

Validation is strict (unique ids, resolvable dependencies, acyclic graph, non-empty
search queries, cap enforced). Two failure modes are treated differently on
purpose:

* **Overflow** — the model produces more requirements than the cap. The valid ones
  are kept, the remainder is preserved in ``overflow`` and reported, never silently
  discarded.
* **Malformed** — the output cannot be validated at all. Falls back to one
  requirement containing the original question, with ``fallback=True`` recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.llm import DeepSeekClient
from src.pipelines.shared.llm_json import Usage, call_json

PROMPT_VERSION = "pandarag-plan/1.0"
MAX_REQUIREMENTS = 4

SYSTEM_PROMPT = (
    "You plan evidence requirements for a question-answering system. "
    "You never answer the question. Return JSON only."
)

USER_PROMPT = """Plan the evidence needed to answer the question.

Rules:
- Each requirement must be atomic and independently checkable in a document.
- Each requirement needs a standalone search query that can be sent to a search engine.
- Use dependencies when one requirement can only be answered after another is known.
- Use the fewest requirements that cover the question (1 for a simple fact question).
- For a comparison, create one requirement per entity. Do NOT add a separate
  requirement for the comparison itself: the answering step performs it.
- Never create a requirement whose only job is to combine or compare facts that
  other requirements already collect. Every requirement must need its own evidence.
- Do NOT answer the question. Do NOT invent names, universities, dates or facts.
  Only describe what must be found.
- At most {max_n} requirements.

Question type: {semantic_type}
Question: {question}

Return JSON exactly like:
{{"requirements": [{{"id": "E1", "description": "Identify X", "search_query": "Who is X?", "depends_on": []}}]}}"""


@dataclass(frozen=True)
class Requirement:
    id: str
    description: str
    search_query: str
    depends_on: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "search_query": self.search_query,
            "depends_on": list(self.depends_on),
        }


@dataclass
class Plan:
    question_type: str
    requirements: list[Requirement]
    fallback: bool = False
    overflow: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    raw: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_version": PROMPT_VERSION,
            "question_type": self.question_type,
            "requirements": [requirement.as_dict() for requirement in self.requirements],
            "n_requirements": len(self.requirements),
            "fallback": self.fallback,
            "overflow": list(self.overflow),
            "n_overflow": len(self.overflow),
            "error": self.error,
        }

    def search_queries(self) -> list[str]:
        return [requirement.search_query for requirement in self.requirements]


def _detect_cycle(requirements: Sequence[Requirement]) -> str | None:
    graph = {requirement.id: list(requirement.depends_on) for requirement in requirements}
    state: dict[str, int] = {}

    def visit(node: str) -> bool:
        status = state.get(node, 0)
        if status == 1:
            return True
        if status == 2:
            return False
        state[node] = 1
        for neighbour in graph.get(node, ()):
            if visit(neighbour):
                return True
        state[node] = 2
        return False

    for node in graph:
        if visit(node):
            return node
    return None


def normalise(payload: Any, *, max_requirements: int = MAX_REQUIREMENTS) -> tuple[list[Requirement], list[dict[str, Any]]]:
    """Validate and normalise planner output; returns ``(requirements, overflow)``."""
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    raw = payload.get("requirements")
    if not isinstance(raw, list) or not raw:
        raise ValueError("'requirements' must be a non-empty list")

    normalised: list[Requirement] = []
    seen_ids: set[str] = set()
    for position, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"requirement {position} is not an object")
        description = str(item.get("description", "")).strip()
        query = str(item.get("search_query", "")).strip()
        if not description or not query:
            raise ValueError(f"requirement {position} has an empty description or search_query")
        identifier = str(item.get("id", f"E{position}")).strip() or f"E{position}"
        if identifier in seen_ids:
            raise ValueError(f"duplicate requirement id {identifier!r}")
        seen_ids.add(identifier)
        depends = item.get("depends_on") or []
        if not isinstance(depends, list):
            raise ValueError(f"requirement {identifier}: depends_on must be a list")
        normalised.append(
            Requirement(
                id=identifier,
                description=description,
                search_query=query,
                depends_on=tuple(str(dep).strip() for dep in depends if str(dep).strip()),
            )
        )

    known = {requirement.id for requirement in normalised}
    for requirement in normalised:
        unknown = [dep for dep in requirement.depends_on if dep not in known]
        if unknown:
            raise ValueError(f"requirement {requirement.id} depends on unknown ids {unknown}")
        if requirement.id in requirement.depends_on:
            raise ValueError(f"requirement {requirement.id} depends on itself")

    cycle = _detect_cycle(normalised)
    if cycle:
        raise ValueError(f"dependency cycle detected at {cycle!r}")

    kept = normalised[:max_requirements]
    overflow = [requirement.as_dict() for requirement in normalised[max_requirements:]]
    # Dependencies must not point at dropped requirements.
    kept_ids = {requirement.id for requirement in kept}
    for requirement in kept:
        if any(dep not in kept_ids for dep in requirement.depends_on):
            raise ValueError(
                f"requirement {requirement.id} depends on a requirement dropped by the cap"
            )
    return kept, overflow


def fallback_plan(question: str, semantic_type: str, error: str | None = None) -> Plan:
    return Plan(
        question_type=semantic_type,
        requirements=[
            Requirement(
                id="E1",
                description="Answer the question directly",
                search_query=question.strip(),
                depends_on=(),
            )
        ],
        fallback=True,
        error=error or "planner fallback",
    )


def plan_requirements(
    client: DeepSeekClient,
    question: str,
    semantic_type: str,
    *,
    usage: Usage,
    max_requirements: int = MAX_REQUIREMENTS,
    simple_types: Sequence[str] = ("Fact", "Definition"),
) -> Plan:
    """Plan evidence requirements for one question."""
    # A single-fact question does not need an LLM call: one requirement is the
    # documented behaviour for Fact/Definition, and skipping the call saves cost
    # without changing the plan shape.
    if semantic_type in simple_types:
        return Plan(
            question_type=semantic_type,
            requirements=[
                Requirement(
                    id="E1",
                    description="Retrieve the passage that answers the question",
                    search_query=question.strip(),
                    depends_on=(),
                )
            ],
        )

    result = call_json(
        client,
        prompt=USER_PROMPT.format(
            question=question.strip(), semantic_type=semantic_type, max_n=max_requirements
        ),
        system=SYSTEM_PROMPT,
        validator=lambda payload: {"requirements": payload.get("requirements")},
        component="planner",
        usage=usage,
        max_tokens=600,
    )
    if not result.ok or not result.data:
        return fallback_plan(question, semantic_type, result.error)

    try:
        requirements, overflow = normalise(result.data, max_requirements=max_requirements)
    except ValueError as error:
        return fallback_plan(question, semantic_type, str(error))

    return Plan(
        question_type=semantic_type,
        requirements=requirements,
        overflow=overflow,
        raw=result.raw_text,
    )
