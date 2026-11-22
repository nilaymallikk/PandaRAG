"""Aggregation tables for the final report.

Every table is computed from raw per-question records through
:mod:`src.analysis.loaders`, so a table can never disagree with the artifacts that
produced it. Paired comparisons use identical question ids (see ``loaders.paired``).
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from src.analysis import loaders, stats

QUESTION_TYPES = ("Fact", "Definition", "Reasoning")


def answer_quality(rows: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """EM / F1 per system per dataset, plus a pooled row per system."""
    rows = rows or loaders.headline_rows()
    table = [dict(row) for row in rows]
    for system in loaders.SYSTEMS:
        per = [row for row in rows if row["system"] == system]
        if not per:
            continue
        table.append(
            {
                "system": system,
                "label": loaders.SYSTEMS[system].label,
                "family": loaders.SYSTEMS[system].family,
                "dataset": "pooled",
                "n": sum(row["n"] for row in per),
                "em": round(mean(row["em"] for row in per if row["em"] is not None), 4),
                "f1": round(mean(row["f1"] for row in per if row["f1"] is not None), 4),
                "recall@10": round(mean(row["recall@10"] for row in per if row["recall@10"] is not None), 4),
                "all_gold@10": round(
                    mean(row["all_gold@10"] for row in per if row["all_gold@10"] is not None), 4
                ),
            }
        )
    return table


def per_question_type() -> list[dict[str, Any]]:
    """EM / F1 by the frozen semantic type, per system per dataset."""
    rows = []
    for system, dataset in loaders.available():
        records = loaders.load(system, dataset)
        for question_type in QUESTION_TYPES:
            subset = [row for row in records if row.get("semantic_type") == question_type]
            if not subset:
                continue
            rows.append(
                {
                    "system": system,
                    "dataset": dataset,
                    "semantic_type": question_type,
                    "n": len(subset),
                    "em": round(mean(row["exact_match"] for row in subset), 4),
                    "f1": round(mean(row["f1"] for row in subset), 4),
                    "recall@10": _mean_optional(row.get("recall@10") for row in subset),
                }
            )
    return rows


def retrieval_quality() -> list[dict[str, Any]]:
    rows = []
    for system, dataset in loaders.available():
        if not loaders.SYSTEMS[system].uses_retrieval:
            continue
        records = loaders.load(system, dataset)
        rows.append(
            {
                "system": system,
                "dataset": dataset,
                "n": len(records),
                "retrieved_documents": _mean_optional(
                    len(row.get("retrieved_doc_ids") or []) for row in records
                ),
                "recall@1": _mean_optional(row.get("recall@1") for row in records),
                "recall@3": _mean_optional(row.get("recall@3") for row in records),
                "recall@5": _mean_optional(row.get("recall@5") for row in records),
                "recall@10": _mean_optional(row.get("recall@10") for row in records),
                "all_gold@10": _mean_optional(row.get("all_gold@10") for row in records),
            }
        )
    return rows


def efficiency() -> list[dict[str, Any]]:
    rows = []
    for system, dataset in loaders.available():
        records = loaders.load(system, dataset)
        meta = loaders.load_meta(system, dataset) or {}
        aggregate = meta.get("aggregate", {})
        rows.append(
            {
                "system": system,
                "dataset": dataset,
                "n": len(records),
                "input_tokens": _mean_optional(row.get("input_tokens") for row in records),
                "output_tokens": _mean_optional(row.get("output_tokens") for row in records),
                "total_tokens": _mean_optional(
                    (row.get("input_tokens") or 0) + (row.get("output_tokens") or 0) for row in records
                ),
                "llm_calls": _mean_optional(row.get("llm_calls") for row in records),
                "retrieval_executions": _mean_optional(row.get("retrieval_executions") for row in records),
                "context_documents": _mean_optional(row.get("n_context_documents") for row in records),
                "context_tokens": _mean_optional(row.get("context_tokens_estimate") for row in records),
                "cost_live_total": aggregate.get("total_api_cost_usd_live"),
                "cost_replay_total": aggregate.get("total_api_cost_usd"),
                "cost_per_question": _mean_optional(row.get("api_cost_usd") for row in records),
                "latency_mean": _mean_optional(row.get("total_latency_seconds") for row in records),
                "latency_p95": _percentile(
                    [row.get("total_latency_seconds") or 0 for row in records], 0.95
                ),
            }
        )
    return rows


def pandarag_behaviour() -> list[dict[str, Any]]:
    """Adaptive-behaviour statistics for P6 (coverage, rounds, targeting, budgets)."""
    rows = []
    for dataset in loaders.DATASETS:
        records = loaders.load("P6", dataset)
        if not records:
            continue
        traces = [row["pandarag"] for row in records if "pandarag" in row]
        if not traces:
            continue
        rounds = [row.get("retrieval_rounds", 0) for row in records]
        rows.append(
            {
                "dataset": dataset,
                "n": len(traces),
                "mean_requirements": round(mean(len(trace["requirements"]) for trace in traces), 3),
                "mean_coverage": round(
                    mean(trace["final_coverage"]["coverage_ratio"] for trace in traces), 4
                ),
                "mean_initial_coverage": round(
                    mean(trace["initial_coverage"]["coverage_ratio"] for trace in traces), 4
                ),
                "all_covered_rate": round(
                    mean(1.0 if trace["final_coverage"]["all_covered"] else 0.0 for trace in traces), 4
                ),
                "targeted_retrieval_rate": round(
                    mean(1.0 if trace["targeted_retrieval_requested"] else 0.0 for trace in traces), 4
                ),
                "recovery_success_rate": _recovery_success(traces),
                "stopped_after_initial": round(mean(1.0 if value <= 1 else 0.0 for value in rounds), 4),
                "budget_exhausted_rate": round(
                    mean(
                        1.0
                        if any(
                            item.get("decision", {}).get("stop_reason") in {
                                "max_rounds",
                                "max_documents",
                                "max_retriever_executions",
                            }
                            for item in trace["rounds"]
                        )
                        else 0.0
                        for trace in traces
                    ),
                    4,
                ),
                "fallback_plan_rate": round(
                    mean(1.0 if trace["plan"]["fallback"] else 0.0 for trace in traces), 4
                ),
                "plan_overflow_rate": round(
                    mean(1.0 if trace["plan"].get("n_overflow", 0) else 0.0 for trace in traces), 4
                ),
                "insufficiency_answer_rate": round(
                    mean(
                        1.0
                        if str(record.get("prediction", "")).strip().lower().startswith("insufficient")
                        else 0.0
                        for record in records
                    ),
                    4,
                ),
                "mean_rounds": round(mean(rounds), 3),
            }
        )
    return rows


def _recovery_success( traces: Sequence[Mapping[str, Any]]) -> float | None:
    recovered = 0
    attempted = 0
    for trace in traces:
        for item in trace["rounds"]:
            if item.get("kind") != "recovery":
                continue
            attempted += 1
            if item.get("improved"):
                recovered += 1
    return round(recovered / attempted, 4) if attempted else None


def primary_comparisons() -> list[dict[str, Any]]:
    """P6 against each pre-registered baseline, per dataset and pooled."""
    rows: list[dict[str, Any]] = []
    for left, right in loaders.PRIMARY_COMPARISONS:
        pooled_diffs: list[list[float]] = []
        pooled_em: list[tuple[bool, bool]] = []
        for dataset in loaders.DATASETS:
            pairs = loaders.paired(left, right, dataset)
            if not pairs:
                continue
            left_records = [pair[0] for pair in pairs]
            right_records = [pair[1] for pair in pairs]
            for metric in ("f1", "input_tokens", "api_cost_usd"):
                row = stats.compare_pair(
                    left_records, right_records, metric=metric, label=f"{left} vs {right}", dataset=dataset
                )
                rows.append(row)
            pooled_diffs.append(
                [float(a.get("f1") or 0) - float(b.get("f1") or 0) for a, b in zip(left_records, right_records)]
            )
            pooled_em.extend(
                (bool(a.get("exact_match")), bool(b.get("exact_match")))
                for a, b in zip(left_records, right_records)
            )

        if pooled_diffs:
            interval = stats.stratified_bootstrap_ci(pooled_diffs)
            mcnemar = stats.mcnemar_from_pairs(pooled_em)
            rows.append(
                {
                    "label": f"{left} vs {right}",
                    "dataset": "pooled",
                    "metric": "f1",
                    "n": interval["n"],
                    "mean_difference": interval["mean"],
                    "ci_low": interval["ci_low"],
                    "ci_high": interval["ci_high"],
                    "effect_size_dz": None,
                    "mcnemar": mcnemar,
                    "p_value": mcnemar["p_value"],
                }
            )
            rows.append(
                {
                    "label": f"{left} vs {right}",
                    "dataset": "pooled",
                    "metric": "exact_match",
                    "n": mcnemar["n_pairs"],
                    "mean_difference": round(
                        mean(1.0 if a else 0.0 for a, _ in pooled_em)
                        - mean(1.0 if b else 0.0 for _, b in pooled_em),
                        4,
                    ),
                    "ci_low": None,
                    "ci_high": None,
                    "mcnemar": mcnemar,
                    "p_value": mcnemar["p_value"],
                }
            )

    # Holm correction over the family of F1 comparisons (per-dataset and pooled),
    # which is the family the paper claims on.
    family = [row for row in rows if row.get("metric") == "f1"]
    adjusted = stats.holm([row["p_value"] for row in family])
    for row, value in zip(family, adjusted):
        row["p_value_holm"] = round(value, 6)
    # EM comparisons get their own correction family.
    em_family = [row for row in rows if row.get("metric") == "exact_match"]
    em_adjusted = stats.holm([row["p_value"] for row in em_family])
    for row, value in zip(em_family, em_adjusted):
        row["p_value_holm"] = round(value, 6)
    return rows


def _mean_optional(values: Iterable[float | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return round(sum(numbers) / len(numbers), 4) if numbers else None


def _percentile(values: Sequence[float], q: float) -> float | None:
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return None
    return round(numbers[min(len(numbers) - 1, int(round(q * (len(numbers) - 1))))], 4)


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    """Render a compact markdown table (``columns`` = [(key, header), ...])."""
    header = "| " + " | ".join(title for _, title in columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, divider]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if value is None:
                cells.append("–")
            elif isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
