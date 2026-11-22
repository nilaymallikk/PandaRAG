"""Deterministic stratified sampling for reduced-sample experiments.

The sample is identical for every variant and every system: proportional allocation
by the frozen semantic type, largest-remainder rounding, and an order inside each
stratum fixed by ``sha1(seed|question_id)``. No answer information is used.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

DEFAULT_SIZE = 100
DEFAULT_SEED = 42


def stratified_sample(dataset: str, size: int = DEFAULT_SIZE, seed: int = DEFAULT_SEED) -> list[str]:
    from src.pipelines.sac_rag.classifier import load_frozen_labels

    labels = load_frozen_labels(dataset)
    by_type: dict[str, list[str]] = defaultdict(list)
    for question_id in sorted(labels):
        by_type[labels[question_id]["question_type"]].append(question_id)

    total = sum(len(items) for items in by_type.values())
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    allocated = 0
    for question_type, items in by_type.items():
        exact = size * len(items) / total
        base = int(exact)
        quotas[question_type] = base
        allocated += base
        remainders.append((exact - base, question_type))
    for _, question_type in sorted(remainders, reverse=True)[: size - allocated]:
        quotas[question_type] += 1

    chosen: list[str] = []
    for question_type, items in by_type.items():
        ordered = sorted(items, key=lambda qid: hashlib.sha1(f"{seed}|{qid}".encode()).hexdigest())
        chosen.extend(ordered[: quotas[question_type]])
    return sorted(chosen)
