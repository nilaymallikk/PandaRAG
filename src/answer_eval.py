"""Deterministic HotpotQA-style answer evaluation (Exact Match and token F1).

This module is deliberately separate from :mod:`src.evaluation`, which is
frozen with retrieval-only metrics. Nothing here calls the LLM.

Normalization follows the HotpotQA official evaluation convention:

* lowercase,
* remove punctuation,
* remove the articles ``a`` / ``an`` / ``the``,
* collapse whitespace.

An incorrect answer is *not* a hallucination signal: EM/F1 measure answer
correctness only. No groundedness or hallucination metric is defined here,
because No-RAG supplies no retrieval context to ground against.
"""

from __future__ import annotations

import re
import string
from typing import Sequence

_ARTICLES = frozenset({"a", "an", "the"})
_PUNCTUATION = frozenset(string.punctuation)


def normalize_answer(text: str | None) -> str:
    """Normalize an answer string for EM/F1 comparison."""
    if text is None:
        return ""
    lowered = str(text).lower()
    no_punct = "".join(char for char in lowered if char not in _PUNCTUATION)
    tokens = [token for token in no_punct.split() if token not in _ARTICLES]
    return " ".join(tokens)


def exact_match(prediction: str | None, gold: str | None) -> float:
    """1.0 when the normalized strings are identical, else 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str | None, gold: str | None) -> float:
    """Token-level F1 over normalized answer tokens.

    Both answers normalizing to empty counts as agreement (1.0); exactly one
    side empty counts as 0.0.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    overlap = 0
    for token in gold_tokens:
        if pred_counts.get(token, 0) > 0:
            pred_counts[token] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str | None, gold: str | None) -> dict[str, float]:
    """Score one prediction against its gold answer."""
    return {
        "exact_match": exact_match(prediction, gold),
        "f1": token_f1(prediction, gold),
    }


def aggregate_answer_scores(
    scores: Sequence[dict[str, float]],
) -> dict[str, float | int]:
    """Mean EM/F1 over per-question score dicts (failed calls excluded upstream)."""
    em_values = [float(item["exact_match"]) for item in scores]
    f1_values = [float(item["f1"]) for item in scores]
    count = len(scores)
    return {
        "n_scored": count,
        "exact_match": round(sum(em_values) / count, 6) if count else 0.0,
        "f1": round(sum(f1_values) / count, 6) if count else 0.0,
    }
