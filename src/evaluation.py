"""Evaluation logic for the experiment.

This module currently implements retrieval evaluation against the HotpotQA
supporting facts:

* document level -- were the supporting article titles retrieved?
* sentence level -- were the supporting ``(title, sent_id)`` facts retrieved?

Both are reported as Recall@K (fraction of the supporting items found in the
top-K ranking) together with a strict "all supporting items retrieved" flag.
Answer-quality metrics are added in a later stage; nothing here calls the LLM.
"""

from __future__ import annotations

import math
from statistics import mean, median
from typing import Iterable, Mapping, Sequence

from src import config
from src.dataset import HotpotExample
from src.retrieval import RetrievalResult


def recall_at_k(retrieved: Sequence[str], supporting: Iterable[str], k: int) -> float:
    """Fraction of the supporting items present in the first ``k`` retrieved items.

    Returns ``NaN`` when the supporting set is empty (recall is undefined).
    """
    gold = set(supporting)
    if not gold:
        return math.nan
    if k <= 0:
        return 0.0
    return len(gold & set(retrieved[:k])) / len(gold)


def complete_at_k(retrieved: Sequence[str], supporting: Iterable[str], k: int) -> bool:
    """True when every supporting item was retrieved within the first ``k`` items."""
    return set(supporting).issubset(set(retrieved[:k]))


def document_recall_at_k(result: RetrievalResult, example: HotpotExample, k: int) -> float:
    """Document-level Recall@K over the supporting article titles."""
    return recall_at_k(result.paragraph_titles(k), example.supporting_titles, k)


def sentence_recall_at_k(result: RetrievalResult, example: HotpotExample, k: int) -> float:
    """Sentence-level Recall@K over the supporting ``title::sent_id`` facts."""
    return recall_at_k(result.sentence_ids(k), example.supporting_fact_ids, k)


def evaluate_retrieval(
    result: RetrievalResult,
    example: HotpotExample,
    k_values: Sequence[int] = config.RETRIEVAL_K_VALUES,
) -> dict[str, dict[str, float]]:
    """Per-question retrieval metrics for every K (keyed by ``str(k)``)."""
    metrics: dict[str, dict[str, float]] = {}
    for k in k_values:
        metrics[str(k)] = {
            "k": k,
            "document_recall": document_recall_at_k(result, example, k),
            "document_complete": float(
                complete_at_k(result.paragraph_titles(k), example.supporting_titles, k)
            ),
            "sentence_recall": sentence_recall_at_k(result, example, k),
            "sentence_complete": float(
                complete_at_k(result.sentence_ids(k), example.supporting_fact_ids, k)
            ),
        }
    return metrics


def _nanmean(values: Sequence[float]) -> float:
    usable = [value for value in values if not math.isnan(value)]
    if not usable:
        return math.nan
    return round(mean(usable), 6)


def aggregate_retrieval_metrics(
    per_question: Sequence[Mapping[str, Mapping[str, float]]],
    k_values: Sequence[int] = config.RETRIEVAL_K_VALUES,
) -> dict[str, dict[str, float]]:
    """Mean document/sentence recall and strict retrieval rate for each K."""
    aggregate: dict[str, dict[str, float]] = {}
    for k in k_values:
        rows = [record[str(k)] for record in per_question if str(k) in record]
        aggregate[str(k)] = {
            "k": k,
            "n_questions": len(rows),
            "document_recall": _nanmean([row["document_recall"] for row in rows]),
            "document_complete_rate": _nanmean([row["document_complete"] for row in rows]),
            "sentence_recall": _nanmean([row["sentence_recall"] for row in rows]),
            "sentence_complete_rate": _nanmean([row["sentence_complete"] for row in rows]),
        }
    return aggregate


def aggregate_latency(values: Sequence[float]) -> dict[str, float]:
    """Mean/median/p95/total latency of a list of per-question measurements."""
    if not values:
        return {"mean": math.nan, "median": math.nan, "p95": math.nan, "total": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean": round(mean(ordered), 6),
        "median": round(median(ordered), 6),
        "p95": round(ordered[p95_index], 6),
        "total": round(sum(ordered), 6),
    }


def format_retrieval_summary(aggregate: Mapping[str, Mapping[str, float]]) -> str:
    """Human-readable Recall@K table (document and sentence level)."""
    header = (
        f"{'K':>3}  {'Doc Recall@K':>12}  {'Doc All@K':>9}  "
        f"{'Sent Recall@K':>13}  {'Sent All@K':>10}"
    )
    lines = [header, "-" * len(header)]
    for key, row in aggregate.items():
        lines.append(
            f"{int(row['k']):>3}  {row['document_recall']:>12.4f}  "
            f"{row['document_complete_rate']:>9.4f}  {row['sentence_recall']:>13.4f}  "
            f"{row['sentence_complete_rate']:>10.4f}"
        )
    return "\n".join(lines)
