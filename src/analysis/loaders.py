"""Loading and pairing Scope-B result artifacts.

One place that knows where each system's results live and how to align two systems
on identical question ids. Every comparison in the paper is paired, so the pairing
logic lives here rather than being re-implemented per table.

A system run is usable only when it has a meta file whose ``n_success`` equals the
number of records; partial or crashed runs are surfaced as such instead of being
silently averaged into a table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src import config

RESULTS = config.RESULTS_DIR


@dataclass(frozen=True)
class System:
    id: str
    directory: str
    label: str
    family: str  # closed_book | corpus_baseline | sac_rag | pandarag
    uses_retrieval: bool = True
    reports_coverage: bool = False


SYSTEMS: dict[str, System] = {
    "P0": System("P0", "baseline_no_rag", "No-RAG", "closed_book", uses_retrieval=False),
    "P1c": System("P1c", "baseline_dense_corpus", "Dense RAG (corpus)", "corpus_baseline"),
    "P2c": System("P2c", "baseline_hybrid_corpus", "Hybrid RAG (fusion depth 5)", "corpus_baseline"),
    "P2c10": System("P2c10", "baseline_hybrid_corpus_depth10", "Hybrid RAG (fusion depth 10)", "corpus_baseline"),
    "P3c": System("P3c", "baseline_adaptive_transferred", "Adaptive RAG (transferred)", "corpus_baseline"),
    "P1n": System("P1n", "baseline_dense_paper_naive", "Naive RAG (paper 10->5)", "corpus_baseline"),
    "P4": System("P4", "sac_rag_adaptive", "SAC-RAG Adaptive", "sac_rag"),
    "P5": System("P5", "sac_rag_full", "Full SAC-RAG", "sac_rag"),
    "P6": System("P6", "pandarag", "PandaRAG", "pandarag", reports_coverage=True),
}

DATASETS = ("hotpotqa", "2wiki", "musique")
PRIMARY_COMPARISONS = (
    ("P6", "P4"),
    ("P6", "P5"),
    ("P6", "P1c"),
    ("P6", "P2c"),
    ("P6", "P2c10"),
    ("P6", "P3c"),
)


def results_path(system: str, dataset: str) -> Path:
    return RESULTS / SYSTEMS[system].directory / f"{dataset}.jsonl"


def meta_path(system: str, dataset: str) -> Path:
    return RESULTS / SYSTEMS[system].directory / f"{dataset}.meta.json"


def load(system: str, dataset: str) -> list[dict[str, Any]]:
    """Load records, one per question id, preferring a successful attempt.

    Resume runs append a new row for a previously failed question instead of
    rewriting the file, so raw failures are preserved. Analysis must use exactly one
    row per question: the successful one when it exists.
    """
    path = results_path(system, dataset)
    if not path.exists():
        return []
    by_id: dict[str, dict[str, Any]] = {}
    for row in _iter_rows(path):
        key = str(row["question_id"])
        existing = by_id.get(key)
        if existing is None or (row.get("success") and not existing.get("success")):
            by_id[key] = row
    return [by_id[key] for key in sorted(by_id)]


def _iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def raw_row_count(system: str, dataset: str) -> int:
    path = results_path(system, dataset)
    return sum(1 for _ in _iter_rows(path)) if path.exists() else 0


def load_meta(system: str, dataset: str) -> dict[str, Any] | None:
    path = meta_path(system, dataset)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_complete(system: str, dataset: str) -> bool:
    meta = load_meta(system, dataset)
    rows = load(system, dataset)
    if meta is None or not rows:
        return False
    return int(meta["aggregate"].get("n_success", 0)) == len(rows)


def available(systems: Iterable[str] = tuple(SYSTEMS), datasets: Iterable[str] = DATASETS) -> list[tuple[str, str]]:
    return [(system, dataset) for dataset in datasets for system in systems if is_complete(system, dataset)]


def index_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["question_id"]): dict(row) for row in rows}


def paired(
    system_a: str, system_b: str, dataset: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Records for both systems on identical question ids, sorted by id."""
    left = index_by_id(load(system_a, dataset))
    right = index_by_id(load(system_b, dataset))
    shared = sorted(set(left) & set(right))
    return [(left[qid], right[qid]) for qid in shared]


def paired_ids(*systems_and_dataset: str) -> list[str]:
    """Question ids common to every listed system on one dataset."""
    *systems, dataset = systems_and_dataset
    sets = []
    for system in systems:
        sets.append(set(index_by_id(load(system, dataset))))
    shared = set.intersection(*sets) if sets else set()
    return sorted(shared)


def headline_rows() -> list[dict[str, Any]]:
    """One row per (system, dataset) with the numbers every table needs."""
    rows = []
    for system, dataset in available():
        records = load(system, dataset)
        meta = load_meta(system, dataset) or {}
        aggregate = meta.get("aggregate", {})
        rows.append(
            {
                "system": system,
                "label": SYSTEMS[system].label,
                "family": SYSTEMS[system].family,
                "dataset": dataset,
                "n": len(records),
                "em": _mean(row.get("exact_match") for row in records),
                "f1": _mean(row.get("f1") for row in records),
                "recall@10": _mean(row.get("recall@10") for row in records),
                "all_gold@10": _mean(row.get("all_gold@10") for row in records),
                "input_tokens": _mean(row.get("input_tokens") for row in records),
                "output_tokens": _mean(row.get("output_tokens") for row in records),
                "total_tokens": _mean(
                    (row.get("input_tokens") or 0) + (row.get("output_tokens") or 0) for row in records
                ),
                "cost_live": aggregate.get("total_api_cost_usd_live"),
                "cost_replay": aggregate.get("total_api_cost_usd"),
                "llm_calls": _mean(row.get("llm_calls") for row in records),
                "retrieval_executions": _mean(row.get("retrieval_executions") for row in records),
                "latency_mean": _mean(row.get("total_latency_seconds") for row in records),
                "latency_p95": _percentile([row.get("total_latency_seconds") or 0 for row in records], 0.95),
                "context_documents": _mean(row.get("n_context_documents") for row in records),
                "context_tokens": _mean(row.get("context_tokens_estimate") for row in records),
                "runtime_s": meta.get("runtime_s"),
            }
        )
    return rows


def _mean(values: Iterable[float | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return round(sum(numbers) / len(numbers), 4) if numbers else None


def _percentile(values: Sequence[float], q: float) -> float | None:
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return None
    index = min(len(numbers) - 1, int(round(q * (len(numbers) - 1))))
    return round(numbers[index], 4)
