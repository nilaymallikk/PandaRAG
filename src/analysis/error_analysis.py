"""Error analysis: classify wrong answers and trace PandaRAG's recovery behaviour.

The taxonomy is the one the directive specifies. Classification is heuristic and
conservative: a category fires only when the stored evidence supports it, and
anything undecidable is labelled ``unclassified`` rather than guessed. Categories
that need a faithfulness judgement are reported only when judging data exists.

Nothing here is used to change a system; it exists to explain the results.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src import config
from src.analysis import loaders

OUT_DIR = config.RESULTS_DIR / "analysis_scope_b"
OUT_FILE = OUT_DIR / "error_analysis.json"

CATEGORIES = (
    "closed_book_knowledge_gap",
    "abstained",
    "missing_evidence_initial",
    "evidence_present_not_recognised",
    "requirement_decomposition",
    "wrong_targeted_query",
    "reranking_failure",
    "evaluator_false_positive",
    "evaluator_false_negative",
    "context_selection_removed",
    "compression_destroyed_evidence",
    "generator_hallucination",
    "generator_formatting",
    "contradictory_evidence",
    "api_or_pipeline_failure",
    "unclassified",
)

LONG_ANSWER_WORDS = 8


def _gold_in(record: Mapping[str, Any], key: str) -> bool:
    gold = set(record.get("gold_document_ids") or [])
    if not gold:
        return False
    candidates = set(record.get(key) or [])
    return gold.issubset(candidates)


def _normalised(text: str | None) -> str:
    return " ".join(str(text or "").lower().split())


def classify(record: Mapping[str, Any], faithfulness: Mapping[str, Any] | None = None) -> str:
    """Return one category for an incorrect record."""
    if not record.get("success"):
        return "api_or_pipeline_failure"

    gold = _normalised(record.get("gold_answer"))
    prediction = _normalised(record.get("prediction"))
    trace = record.get("pandarag") or {}

    if prediction.startswith("insufficient evidence"):
        return "abstained"
    if record.get("pipeline") == "baseline_no_rag":
        return "closed_book_knowledge_gap"

    # Evidence was there but the answer is wrong in a shallow way: formatting.
    if gold and gold in prediction and len(prediction.split()) > LONG_ANSWER_WORDS:
        return "generator_formatting"

    if faithfulness is not None:
        if (faithfulness.get("n_contradicted") or 0) > 0:
            return "contradictory_evidence"
        if (faithfulness.get("n_unsupported") or 0) > 0:
            return "generator_hallucination"

    if trace:
        final = trace.get("final_coverage", {})
        initial = trace.get("initial_coverage", {})
        if (final.get("n_contradicted") or 0) > 0:
            return "contradictory_evidence"

        context_ids = {item["doc_id"] for item in record.get("context_documents") or []}
        admitted = set(record.get("candidate_document_ids") or [])
        gold_ids = set(record.get("gold_document_ids") or [])

        # Compression removed evidence that retrieval had found.
        compression = (trace.get("context_selection") or {}).get("compression")
        if compression and not compression.get("fallback"):
            retained = set(compression.get("doc_ids_retained") or [])
            if gold_ids & admitted and not (gold_ids & retained):
                return "compression_destroyed_evidence"

        # The document was admitted but the context builder left it out.
        if gold_ids & admitted and not (gold_ids & context_ids):
            return "context_selection_removed"

        # Gold evidence was in the context, but coverage said "not covered".
        if (gold_ids & context_ids) and not final.get("all_covered", False):
            return "evaluator_false_negative"

        # Coverage claimed full support, yet the answer is wrong: the evaluator
        # over-claimed support for this question.
        if final.get("all_covered") and (initial.get("n_supported", 0) <= (final.get("n_supported") or 0)):
            return "evaluator_false_positive"

        if initial.get("n_supported", 1) < len(trace.get("requirements") or []) and not trace.get(
            "targeted_retrieval_requested"
        ):
            return "missing_evidence_initial"

        # Retrieval rounds ran but never improved coverage.
        recoveries = [item for item in trace.get("rounds") or [] if item.get("kind") == "recovery"]
        if recoveries and not any(item.get("improved") for item in recoveries):
            if any(item.get("n_new", 0) == 0 for item in recoveries):
                return "wrong_targeted_query"
            return "missing_evidence_initial"

        if trace.get("plan", {}).get("fallback"):
            return "requirement_decomposition"

    # Non-PandaRAG systems (and unresolved PandaRAG cases).
    gold_docs = set(record.get("gold_document_ids") or [])
    retrieved = set(record.get("retrieved_doc_ids") or [])
    admitted = set(record.get("candidate_document_ids") or [])
    if gold_docs:
        if gold_docs.issubset(retrieved):
            # Every required passage was available to the generator.
            return "evidence_present_not_recognised"
        if gold_docs.issubset(admitted) and not gold_docs.issubset(retrieved):
            return "reranking_failure"
        if not (gold_docs & retrieved):
            return "missing_evidence_initial"
        # Partial: at least one required passage never reached the generator.
        return "missing_evidence_initial"
    return "unclassified"


def analyse() -> dict[str, Any]:
    faithfulness: dict[tuple[str, str, str], dict[str, Any]] = {}
    detail = OUT_DIR / "faithfulness.jsonl"
    if detail.exists():
        for line in detail.open(encoding="utf-8"):
            if not line.strip():
                continue
            entry = json.loads(line)
            faithfulness[(entry["system"], entry["dataset"], entry["question_id"])] = entry

    per_system: dict[str, dict[str, Any]] = {}
    traces: list[dict[str, Any]] = []

    for system, dataset in loaders.available():
        records = loaders.load(system, dataset)
        wrong = [row for row in records if not row.get("exact_match")]
        counts: Counter = Counter()
        for record in wrong:
            label = classify(
                record, faithfulness.get((system, dataset, record["question_id"]))
            )
            counts[label] += 1
        per_system.setdefault(system, {"datasets": {}, "counts": Counter(), "n_wrong": 0})
        per_system[system]["datasets"][dataset] = {
            "n": len(records),
            "n_wrong": len(wrong),
            "counts": dict(counts.most_common()),
        }
        per_system[system]["counts"].update(counts)
        per_system[system]["n_wrong"] += len(wrong)

        if system == "P6" and dataset == "hotpotqa":
            recoveries = [
                row
                for row in records
                if row.get("pandarag", {}).get("targeted_retrieval_requested")
            ]
            for row in recoveries[:5]:
                trace = row["pandarag"]
                traces.append(
                    {
                        "question_id": row["question_id"],
                        "question": row["question"],
                        "requirements": trace["requirements"],
                        "initial_coverage": trace["initial_coverage"],
                        "final_coverage": trace["final_coverage"],
                        "rounds": [
                            {
                                "round": item.get("round"),
                                "kind": item.get("kind"),
                                "decision": item.get("decision"),
                                "new_doc_ids": item.get("new_doc_ids"),
                                "improved": item.get("improved"),
                            }
                            for item in trace["rounds"]
                        ],
                        "context_doc_ids": [item["doc_id"] for item in row.get("context_documents") or []],
                        "prediction": row.get("prediction"),
                        "gold_answer": row.get("gold_answer"),
                        "exact_match": row.get("exact_match"),
                        "failure_category": classify(row),
                    }
                )

    payload = {
        "taxonomy": list(CATEGORIES),
        "note": (
            "Heuristic, conservative classification over stored artifacts. Categories that need a "
            "faithfulness judgement are only assigned when that judgement exists. 'unclassified' "
            "means the record does not support any single category."
        ),
        "systems": {
            system: {
                "n_wrong": data["n_wrong"],
                "counts": dict(data["counts"].most_common()),
                "share": {
                    key: round(value / max(1, data["n_wrong"]), 4)
                    for key, value in data["counts"].most_common()
                },
                "datasets": data["datasets"],
            }
            for system, data in per_system.items()
        },
        "pandarag_traces": traces,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> int:  # pragma: no cover
    payload = analyse()
    for system, data in payload["systems"].items():
        print(system, data["n_wrong"], list(data["counts"].items())[:4])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
