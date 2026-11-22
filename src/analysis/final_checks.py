"""Post-audit checks: depth-aware retrieval metrics, failure accounting, ablation inertness.

Read-only. Produces `results/analysis_scope_b/final_checks.json`, consumed by the audit
and research reports.

Three questions it answers:

1. **Fixed-depth vs stopping-depth recall.** Recall@K is only defined when at least K
   documents were retrieved. Systems that stop at 5 (P1c, P2c) have no R@8/R@10; mixing
   them into one column is what made the Phase-1 comparison look inconsistent.
2. **Failure accounting.** Every JSONL contains superseded `success: false` rows from
   HTTP-429 attempts. This enumerates rows, unique questions, final answers, terminal
   failures and the exact metric denominators per system and dataset.
3. **Ablation inertness.** A3 and A4 produced identical results. This traces whether the
   components are active and whether the evidence-preserving builder can ever change the
   context under the frozen budget.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from src import config
from src.analysis import loaders

OUT = config.RESULTS_DIR / "analysis_scope_b" / "final_checks.json"
K_VALUES = (1, 3, 5, 8, 10)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _mean(values: Iterable[float | None]) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    return round(sum(numbers) / len(numbers), 6) if numbers else None


def _recall(record: Mapping[str, Any], k: int) -> float | None:
    gold = set(record.get("gold_document_ids") or [])
    retrieved = list(record.get("retrieved_doc_ids") or [])
    if not gold or len(retrieved) < k:
        return None  # undefined, not zero
    return len(gold & set(retrieved[:k])) / len(gold)


def _context_recall(record: Mapping[str, Any]) -> float | None:
    """Recall over the documents the generator actually received.

    Distinct from candidate recall: P1n and P2c10 retrieve 10 documents but feed 5,
    so their candidate R@10 is not evidence the model ever saw.
    """
    gold = set(record.get("gold_document_ids") or [])
    if not gold:
        return None
    context = {item["doc_id"] for item in (record.get("context_documents") or [])}
    return len(gold & context) / len(gold)


def _all_gold(record: Mapping[str, Any], k: int) -> float | None:
    gold = set(record.get("gold_document_ids") or [])
    retrieved = list(record.get("retrieved_doc_ids") or [])
    if not gold or len(retrieved) < k:
        return None
    return 1.0 if gold.issubset(set(retrieved[:k])) else 0.0


# ---------------------------------------------------------------------------
# 1. depth-aware retrieval metrics
# ---------------------------------------------------------------------------


def depth_metrics() -> dict[str, Any]:
    per_cell = []
    for system, dataset in loaders.available():
        records = loaders.load(system, dataset)
        depths = [len(r.get("retrieved_doc_ids") or []) for r in records]
        cell: dict[str, Any] = {
            "system": system,
            "dataset": dataset,
            "n": len(records),
            "mean_depth": round(mean(depths), 3),
            "min_depth": min(depths),
            "max_depth": max(depths),
            "n_cells_at_or_above": {k: sum(1 for d in depths if d >= k) for k in K_VALUES},
            "fixed_depth": {},
            "stopping_depth": {
                "recall": _mean(_recall(r, len(r.get("retrieved_doc_ids") or [])) for r in records),
                "all_gold": _mean(_all_gold(r, len(r.get("retrieved_doc_ids") or [])) for r in records),
            },
            "context_recall": _mean(_context_recall(r) for r in records),
        }
        for k in K_VALUES:
            cell["fixed_depth"][f"recall@{k}"] = _mean(_recall(r, k) for r in records)
            cell["fixed_depth"][f"all_gold@{k}"] = _mean(_all_gold(r, k) for r in records)
            cell["fixed_depth"][f"n_defined@{k}"] = cell["n_cells_at_or_above"][k]
        per_cell.append(cell)

    # Macro over datasets for systems defined at each depth.
    macro = {}
    for system in loaders.SYSTEMS:
        cells = [c for c in per_cell if c["system"] == system]
        if not cells:
            continue
        macro[system] = {
            "mean_depth": round(mean(c["mean_depth"] for c in cells), 3),
            "fixed_depth": {},
            "stopping_depth": {
                "recall": _mean(c["stopping_depth"]["recall"] for c in cells),
                "all_gold": _mean(c["stopping_depth"]["all_gold"] for c in cells),
            },
            "context_recall": _mean(c["context_recall"] for c in cells),
        }
        for k in K_VALUES:
            values = [c["fixed_depth"][f"recall@{k}"] for c in cells]
            defined = [v for v in values if v is not None]
            macro[system]["fixed_depth"][f"recall@{k}"] = _mean(defined)
            macro[system]["fixed_depth"][f"datasets_defined@{k}"] = len(defined)
            macro[system]["fixed_depth"][f"all_gold@{k}"] = _mean(
                c["fixed_depth"][f"all_gold@{k}"] for c in cells
            )
    return {"per_cell": per_cell, "macro": macro}


# ---------------------------------------------------------------------------
# 2. failure accounting
# ---------------------------------------------------------------------------


def failure_accounting() -> dict[str, Any]:
    system_rows = []
    total_failed_rows = 0
    for system, dataset in loaders.available():
        raw = _rows(loaders.results_path(system, dataset))
        success_rows = [r for r in raw if r.get("success")]
        failed_rows = [r for r in raw if not r.get("success")]
        success_by_id = {}
        for record in success_rows:
            success_by_id.setdefault(record["question_id"], record)
        ids_with_success = set(success_by_id)
        ids_seen = {r["question_id"] for r in raw}
        ids_without_success = sorted(ids_seen - ids_with_success)
        total_failed_rows += len(failed_rows)
        id_counts = Counter(r["question_id"] for r in success_rows)
        duplicate_success_rows = sum(count - 1 for count in id_counts.values())
        system_rows.append(
            {
                "system": system,
                "dataset": dataset,
                "rows_written": len(raw),
                "success_rows": len(success_rows),
                "failed_rows_superseded": len(failed_rows),
                "duplicate_success_rows": duplicate_success_rows,
                "unique_questions": len(ids_seen),
                "unique_questions_with_success": len(ids_with_success),
                "unique_questions_without_success": len(ids_without_success),
                "terminal_failures": len(ids_without_success),
                # Denominator = unique questions with a successful record, never rows:
                # one cell has duplicate successful rows (P2c10/2Wiki) and counting rows
                # would overstate every denominator in that cell.
                "metric_denominator_em_f1": sum(
                    1 for r in success_by_id.values() if r.get("exact_match") is not None
                ),
                "metric_denominator_retrieval": sum(
                    1 for r in success_by_id.values() if r.get("recall@10") is not None
                ),
            }
        )

    log_path = config.RESULTS_DIR / "logs" / "llm_errors.jsonl"
    retries: dict[str, Any] = {}
    if log_path.exists():
        entries = _rows(log_path)
        statuses = Counter(str(e.get("status_code")) for e in entries)
        retries = {
            "log_entries": len(entries),
            "status_codes": dict(statuses.most_common(6)),
            "attempts_histogram": dict(
                sorted(Counter(int(e.get("attempt") or 0) for e in entries).items())
            ),
            "retryable_entries": sum(1 for e in entries if e.get("will_retry")),
            "note": (
                "The error log is process-global: entries carry model, base URL, request hash, "
                "attempt number and status, but no pipeline or question id, so retries cannot be "
                "attributed per system from this file. Per-system superseded failure counts come "
                "from the result JSONL instead."
            ),
        }
    return {
        "per_cell": system_rows,
        "totals": {
            "pairs": len(system_rows),
            "rows_written": sum(r["rows_written"] for r in system_rows),
            "success_rows": sum(r["success_rows"] for r in system_rows),
            "superseded_failed_rows": total_failed_rows,
            "duplicate_success_rows": sum(r["duplicate_success_rows"] for r in system_rows),
            "canonical_records": sum(r["unique_questions"] for r in system_rows),
            "questions_without_success": sum(r["terminal_failures"] for r in system_rows),
        },
        "retry_log": retries,
        "faithfulness_denominators": _faithfulness_denominators(),
    }


def _faithfulness_denominators() -> dict[str, Any]:
    detail = config.RESULTS_DIR / "analysis_scope_b" / "faithfulness.jsonl"
    if not detail.exists():
        return {}
    rows = _rows(detail)
    out = {}
    for system in sorted({r["system"] for r in rows}):
        subset = [r for r in rows if r["system"] == system]
        out[system] = {
            "sampled": len(subset),
            "scored": sum(1 for r in subset if r.get("faithfulness") is not None),
            "no_checkable_claims": sum(1 for r in subset if r.get("grounding_label") == "no_checkable_claims"),
            "parse_failures": sum(1 for r in subset if r.get("parse_ok") is False),
        }
    return out


# ---------------------------------------------------------------------------
# 3. ablation inertness (A3 vs A4)
# ---------------------------------------------------------------------------


def ablation_trace() -> dict[str, Any]:
    root = config.RESULTS_DIR / "ablations"
    pairs = {}
    for left, right in (("A2", "A3"), ("A3", "A4"), ("A1", "A2")):
        identical = {}
        for dataset in loaders.DATASETS:
            left_path = root / left / f"{dataset}.jsonl"
            right_path = root / right / f"{dataset}.jsonl"
            if not (left_path.exists() and right_path.exists()):
                continue
            left_rows = {r["question_id"]: r for r in _rows(left_path)}
            right_rows = {r["question_id"]: r for r in _rows(right_path)}
            shared = sorted(set(left_rows) & set(right_rows))
            same_context = 0
            same_prediction = 0
            for qid in shared:
                a = left_rows[qid]
                b = right_rows[qid]
                if (a.get("context_text") or "") == (b.get("context_text") or ""):
                    same_context += 1
                if (a.get("prediction") or "").strip() == (b.get("prediction") or "").strip():
                    same_prediction += 1
            identical[f"{left}_vs_{right}:{dataset}"] = {
                "n": len(shared),
                "identical_context": same_context,
                "identical_prediction": same_prediction,
            }
        pairs[f"{left}_vs_{right}"] = identical

    # Is the evaluator active, and can a protected document fall outside the window?
    activity = {}
    for variant in ("A2", "A3", "A4"):
        records = []
        for dataset in loaders.DATASETS:
            path = root / variant / f"{dataset}.jsonl"
            if path.exists():
                records.extend(_rows(path))
        successful = [r for r in records if r.get("success")]
        verdicts = sum(
            (r.get("pandarag", {}).get("rounds") or [{}])[0].get("evaluation", {}).get("n_verdicts", 0)
            if r.get("pandarag", {}).get("rounds")
            else 0
            for r in successful
        )
        evaluator_cost = sum(
            float((r.get("usage_by_component") or {}).get("evidence_evaluator", {}).get("cost_usd") or 0)
            for r in successful
        )
        protected_outside_window = 0
        checked = 0
        for record in successful:
            trace = record.get("pandarag") or {}
            selection = trace.get("context_selection") or {}
            requirement_documents = selection.get("requirement_documents") or {}
            if not requirement_documents:
                continue
            admitted = list(record.get("candidate_document_ids") or [])
            window = set(admitted[:10])
            for docs in requirement_documents.values():
                for doc_id in docs:
                    checked += 1
                    if doc_id not in window:
                        protected_outside_window += 1
        activity[variant] = {
            "records": len(successful),
            "evaluator_verdicts_first_round": verdicts,
            "evaluator_cost_usd_total": round(evaluator_cost, 6),
            "protected_doc_requirement_links_checked": checked,
            "protected_links_outside_top10": protected_outside_window,
        }

    return {"context_comparisons": pairs, "component_activity": activity}


def main() -> int:  # pragma: no cover
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "depth_metrics": depth_metrics(),
        "failure_accounting": failure_accounting(),
        "ablation_trace": ablation_trace(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", OUT.relative_to(config.REPO_ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
