"""Evidence requirement evaluator (PandaRAG component B).

For each (requirement, document) pair it decides whether the document actually
*evidences* the requirement — not merely whether it is topically relevant. A
passage about the right person that never names the university is
``insufficient``; that distinction is the reason this component exists.

Design choices, all recorded:

* One batched LLM call per round covering the requirements and the admitted
  candidate documents. Per-pair verdicts come back in one JSON array.
* Deterministic settings (temperature 0, frozen prompt), schema validation, and a
  strict ``doc_id``/``requirement_id`` whitelist: a verdict naming an unknown id is
  dropped rather than trusted.
* No gold answer, no gold evidence, no scores, no dataset labels are ever placed in
  the prompt.
* Failure returns ``unknown`` for every requirement and records the error; the
  controller then sees "no coverage" rather than a fabricated success.

A relevance score from the reranker is *not* used as the support decision
(``support != relevance``). It is supplied only as a candidate-ordering hint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.llm import DeepSeekClient
from src.pipelines.pandarag.coverage import PairVerdict
from src.pipelines.pandarag.evidence_planner import Requirement
from src.pipelines.shared.llm_json import Usage, as_float, call_json
from src.pipelines.shared.retriever import RetrievedDoc

PROMPT_VERSION = "pandarag-support/1.0"
MAX_DOCS_PER_CALL = 10
DOC_CHAR_LIMIT = 700
VALID_STATUSES = ("supported", "partial", "insufficient", "contradicted", "unknown")

SYSTEM_PROMPT = (
    "You judge whether retrieved passages contain the evidence needed by a "
    "requirement. You are strict: topical similarity is not evidence. Return JSON only."
)

USER_PROMPT = """For each requirement, decide which passages actually contain the evidence.

Statuses:
- supported: a passage states the fact or relationship the requirement needs.
- partial: a passage gives part of it (e.g. names the entity but not the value).
- insufficient: the passage is related but does not contain the needed evidence.
- contradicted: a passage states something incompatible with the requirement.
- unknown: you cannot tell from the passage.

Rules:
- Judge evidence, not relevance. Mentioning an entity is not evidence about it.
- Quote the exact span that carries the evidence in "supporting_spans" (may be empty).
- Only evaluate pairs where the passage plausibly relates to the requirement.
  You may omit clearly unrelated pairs.
- Use only the passages. Do not use outside knowledge. Do not answer the question.
- Keep every "reason" under 12 words and every "supporting_spans" entry under 25 words.

Question: {question}

Requirements:
{requirements}

Passages:
{passages}

Return JSON exactly like:
{{"verdicts": [{{"requirement_id": "E1", "document_id": "<passage id>", "support_status": "supported", "supporting_spans": ["..."], "reason": "short reason"}}]}}"""


@dataclass
class EvaluationResult:
    verdicts: list[PairVerdict]
    error: str | None
    fallback: bool
    prompt_version: str = PROMPT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_version": self.prompt_version,
            "n_verdicts": len(self.verdicts),
            "fallback": self.fallback,
            "error": self.error,
        }


def _render_requirements(requirements: Sequence[Requirement]) -> str:
    return "\n".join(
        f"- {requirement.id}: {requirement.description} (search: {requirement.search_query})"
        for requirement in requirements
    )


def _render_passages(documents: Sequence[RetrievedDoc]) -> str:
    return "\n\n".join(
        f"[{document.doc_id}] {document.title}\n{document.text[:DOC_CHAR_LIMIT]}"
        for document in documents
    )


def evaluate(
    client: DeepSeekClient,
    question: str,
    requirements: Sequence[Requirement],
    documents: Sequence[RetrievedDoc],
    *,
    usage: Usage,
) -> EvaluationResult:
    """Return one verdict per (requirement, document) pair the model judged."""
    if not requirements or not documents:
        return EvaluationResult(verdicts=[], error="nothing to evaluate", fallback=True)

    documents = list(documents)[:MAX_DOCS_PER_CALL]
    allowed_requirements = {requirement.id for requirement in requirements}
    allowed_documents = {document.doc_id for document in documents}

    def _validate(payload: Any) -> Mapping[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
        raw = payload.get("verdicts")
        if not isinstance(raw, list):
            raise ValueError("'verdicts' must be a list")
        cleaned = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            requirement_id = str(item.get("requirement_id", "")).strip()
            document_id = str(item.get("document_id", "")).strip()
            status = str(item.get("support_status", "")).strip().lower()
            if requirement_id not in allowed_requirements:
                continue  # whitelist: never trust an invented requirement id
            if document_id not in allowed_documents:
                continue  # whitelist: never trust an invented document id
            if status not in VALID_STATUSES:
                status = "unknown"
            spans = item.get("supporting_spans") or []
            if not isinstance(spans, list):
                spans = []
            cleaned.append(
                {
                    "requirement_id": requirement_id,
                    "document_id": document_id,
                    "support_status": status,
                    "supporting_spans": [str(span)[:500] for span in spans][:5],
                    "reason": str(item.get("reason", ""))[:400],
                }
            )
        if not cleaned:
            raise ValueError("no usable verdicts in response")
        return {"verdicts": cleaned}

    result = call_json(
        client,
        prompt=USER_PROMPT.format(
            question=question.strip(),
            requirements=_render_requirements(requirements),
            passages=_render_passages(documents),
        ),
        system=SYSTEM_PROMPT,
        validator=_validate,
        component="evidence_evaluator",
        usage=usage,
        max_tokens=900,
    )

    if not result.ok or not result.data:
        return EvaluationResult(verdicts=[], error=result.error, fallback=True)

    verdicts = [
        PairVerdict(
            requirement_id=item["requirement_id"],
            document_id=item["document_id"],
            support_status=item["support_status"],
            reason=item["reason"],
            supporting_spans=tuple(item["supporting_spans"]),
        )
        for item in result.data["verdicts"]
    ]
    return EvaluationResult(verdicts=verdicts, error=None, fallback=False)
