"""Sub-question decomposition for the Reasoning strategy (and PandaRAG's planner).

The base paper states that an LLM decomposes a query into "a set of independent
sub-questions" but publishes neither the prompt nor the count. Both are
implementation choices here:

* ``MAX_SUBQUESTIONS = 4`` (documented in the reproduction protocol);
* the prompt states explicitly that no answers or presumed facts may be produced,
  only search queries.

Failure is never fatal: a malformed response yields the one-element fallback
``[original_question]`` with ``fallback=True`` recorded, so a broken decomposition
degrades to plain retrieval instead of losing the question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.llm import DeepSeekClient
from src.pipelines.shared.llm_json import Usage, call_json

PROMPT_VERSION = "sac-rag-decompose/1.0"
MAX_SUBQUESTIONS = 4

SYSTEM_PROMPT = "You break a question into independent search queries. Return JSON only."

USER_PROMPT = """Break the question into at most {max_n} independent sub-questions.

Rules:
- Each sub-question must be answerable by search on its own.
- Do not answer the question. Do not state presumed facts or names you are not given.
- Use the fewest sub-questions that cover the question; one is allowed.
- If the question is already atomic, return it unchanged as the single sub-question.

Question: {question}

Return JSON exactly like: {{"subquestions": ["...", "..."]}}"""


def _validate(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    raw = payload.get("subquestions")
    if not isinstance(raw, list) or not raw:
        raise ValueError("'subquestions' must be a non-empty list")
    cleaned = [str(item).strip() for item in raw if str(item).strip()]
    if not cleaned:
        raise ValueError("all sub-questions were empty")
    return {"subquestions": cleaned[:MAX_SUBQUESTIONS]}


@dataclass
class Decomposition:
    subquestions: list[str]
    fallback: bool
    error: str | None
    raw: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "subquestions": list(self.subquestions),
            "fallback": self.fallback,
            "error": self.error,
            "prompt_version": PROMPT_VERSION,
            "n_subquestions": len(self.subquestions),
        }


def decompose(
    client: DeepSeekClient,
    question: str,
    *,
    usage: Usage | None = None,
    max_subquestions: int = MAX_SUBQUESTIONS,
) -> Decomposition:
    result = call_json(
        client,
        prompt=USER_PROMPT.format(question=question.strip(), max_n=max_subquestions),
        system=SYSTEM_PROMPT,
        validator=_validate,
        component="decomposition",
        usage=usage,
        max_tokens=300,
    )
    if result.ok and result.data:
        return Decomposition(
            subquestions=list(result.data["subquestions"]),
            fallback=False,
            error=None,
            raw=result.raw_text,
        )
    # Safe fallback: one sub-question equal to the original question.
    return Decomposition(
        subquestions=[question.strip()],
        fallback=True,
        error=result.error,
        raw=result.raw_text,
    )


def ensure_original(question: str, subquestions: Sequence[str]) -> list[str]:
    """Guarantee the original question is retrievable even if decomposition drops it."""
    cleaned = [item.strip() for item in subquestions if item.strip()]
    if not cleaned:
        return [question.strip()]
    return cleaned
