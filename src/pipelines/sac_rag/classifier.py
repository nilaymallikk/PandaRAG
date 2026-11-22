"""SAC-RAG question classifier.

Two entry points, deliberately separated:

* :func:`load_frozen_labels` — the classification used by every dataset run. The
  labels were produced by ``sac-rag-classifier/1.3`` and frozen in
  ``data/semantic_types_<dataset>.json`` **before** any answer was generated. Using
  them makes classification reproducible and prevents any possibility of tuning it
  against results. This is a deviation from "classify at runtime" and is recorded
  in the run metadata.
* :func:`classify` — an LLM classifier for unseen questions, with the same schema,
  validation and documented fallback, used if a new question set is ever scored.

Neither path may see the answer, the gold evidence or dataset-native labels
(bridge/comparison), which are a different taxonomy.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.datasets import load_semantic_types
from src.llm import DeepSeekClient
from src.pipelines.shared.llm_json import JSONResult, Usage, call_json

PROMPT_VERSION = "sac-rag-classifier/1.3-runtime"
LABELS = ("Fact", "Definition", "Reasoning")
FALLBACK_LABEL = "Reasoning"  # widest retrieval: safest failure mode

SYSTEM_PROMPT = (
    "You classify questions for a retrieval system. Return JSON only."
)

USER_PROMPT = """Classify the question into exactly one type.

Fact: asks for a specific fact, value, attribute, date, number or entity.
  Cue words: When, Where, How many, Which.
Definition: asks what something is, or for a description/definition.
  Cue words: What is, Who is, Define.
Reasoning: requires combining multiple facts, causal or comparative reasoning,
  or cannot be answered from a single passage.
  Cue words: Why, How, What caused.

Rule of thumb: wherever the answer has to be derived rather than read off one
passage, the question is Reasoning.

Question: {question}

Return JSON exactly like: {{"question_type": "fact|definition|reasoning", "confidence": 0.0-1.0}}"""


def _validate(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    raw = str(payload.get("question_type", "")).strip().lower()
    mapping = {"fact": "Fact", "definition": "Definition", "reasoning": "Reasoning"}
    if raw not in mapping:
        raise ValueError(f"invalid question_type {raw!r}")
    confidence = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return {"question_type": mapping[raw], "confidence": max(0.0, min(1.0, confidence))}


def classify(
    client: DeepSeekClient, question: str, *, usage: Usage | None = None
) -> JSONResult:
    """Classify one question with the LLM (for unseen data only)."""
    return call_json(
        client,
        prompt=USER_PROMPT.format(question=question.strip()),
        system=SYSTEM_PROMPT,
        validator=_validate,
        component="classifier",
        usage=usage,
        max_tokens=120,
    )


def keyword_fallback(question: str) -> str:
    """Deterministic cue-word fallback mirroring the published cue lists."""
    lowered = question.lower().strip()
    if lowered.startswith(("why", "how did", "how does", "what caused", "how many times")):
        return "Reasoning"
    if lowered.startswith(("what is", "who is", "define")):
        return "Definition"
    if lowered.startswith(("when", "where", "how many", "which")):
        return "Fact"
    return FALLBACK_LABEL


def load_frozen_labels(dataset: str) -> dict[str, dict[str, Any]]:
    """Return ``{question_id: {...}}`` from the frozen sidecar, validated for coverage."""
    from src.datasets import get_adapter

    examples = get_adapter(dataset).load()
    labels = load_semantic_types(dataset, examples=examples)
    normalised: dict[str, dict[str, Any]] = {}
    for question_id, value in labels.items():
        if isinstance(value, dict):
            normalised[question_id] = {
                "question_type": str(value.get("type", value.get("question_type", FALLBACK_LABEL))),
                "source": "frozen_sidecar",
                "prompt_version": PROMPT_VERSION,
            }
        else:
            normalised[question_id] = {
                "question_type": str(value),
                "source": "frozen_sidecar",
                "prompt_version": PROMPT_VERSION,
            }
    return normalised
