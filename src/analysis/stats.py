"""Paired statistical tests for the primary comparisons.

Methods, chosen to match the directive and to stay defensible on binary and
continuous metrics:

* **McNemar exact test** for Exact Match (binary, paired): two-sided binomial over
  the discordant pairs, no continuity fudge, exact for small counts.
* **Paired bootstrap** (2,000 resamples, seeded) for F1 / cost / token differences,
  with the percentile interval. Sampling is over *questions*, so within-question
  pairing is preserved; when pooling datasets, resampling is stratified by dataset
  and the pooled mean is recomputed each resample.
* **Holm–Bonferroni** correction over the primary comparison family.

Nothing here infers causation, and no test is applied to a comparison that was not
pre-registered as primary.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------


def mcnemar_exact(discordant_a_only: int, discordant_b_only: int) -> float:
    """Two-sided exact McNemar p-value (binomial on discordant pairs)."""
    b, c = int(discordant_a_only), int(discordant_b_only)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def mcnemar_from_pairs(pairs: Iterable[tuple[bool, bool]]) -> dict[str, Any]:
    """Counts plus exact p-value for paired binary outcomes ``(a, b)``."""
    both = only_a = only_b = neither = 0
    for a, b in pairs:
        if a and b:
            both += 1
        elif a and not b:
            only_a += 1
        elif b and not a:
            only_b += 1
        else:
            neither += 1
    return {
        "n_pairs": both + only_a + only_b + neither,
        "both_correct": both,
        "only_left_correct": only_a,
        "only_right_correct": only_b,
        "neither_correct": neither,
        "p_value": mcnemar_exact(only_a, only_b),
    }


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Percentile CI for the mean of paired differences."""
    values = [float(value) for value in differences if value is not None]
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    observed = sum(values) / len(values)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    low_index = int((alpha / 2) * resamples)
    high_index = min(resamples - 1, int((1 - alpha / 2) * resamples))
    return {
        "mean": round(observed, 6),
        "ci_low": round(means[low_index], 6),
        "ci_high": round(means[high_index], 6),
        "n": n,
        "resamples": resamples,
    }


def stratified_bootstrap_ci(
    strata: Sequence[Sequence[float]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired bootstrap over several datasets: resample within each stratum, pool means.

    Pooling naive question-level differences would let the largest dataset dominate
    the interval; stratifying keeps each dataset's contribution fixed.
    """
    strata = [[float(value) for value in stratum] for stratum in strata if stratum]
    if not strata:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    weights = [len(stratum) / sum(len(s) for s in strata) for stratum in strata]
    observed = sum(weight * (sum(stratum) / len(stratum)) for weight, stratum in zip(weights, strata))

    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        total = 0.0
        for weight, stratum in zip(weights, strata):
            n = len(stratum)
            subtotal = 0.0
            for _ in range(n):
                subtotal += stratum[rng.randrange(n)]
            total += weight * (subtotal / n)
        means.append(total)
    means.sort()
    return {
        "mean": round(observed, 6),
        "ci_low": round(means[int((alpha / 2) * resamples)], 6),
        "ci_high": round(means[min(resamples - 1, int((1 - alpha / 2) * resamples))], 6),
        "n": sum(len(stratum) for stratum in strata),
        "strata": len(strata),
        "resamples": resamples,
    }


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values, order preserved."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(indexed):
        value = min(1.0, (m - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


# ---------------------------------------------------------------------------
# Comparison rows
# ---------------------------------------------------------------------------


def _diff(metric: str) -> Callable[[dict, dict], float]:
    if metric == "exact_match":
        return lambda left, right: float(left.get("exact_match") or 0) - float(right.get("exact_match") or 0)
    if metric == "f1":
        return lambda left, right: float(left.get("f1") or 0) - float(right.get("f1") or 0)
    if metric in ("input_tokens", "output_tokens", "api_cost_usd", "total_latency_seconds", "llm_calls"):
        return lambda left, right: float(left.get(metric) or 0) - float(right.get(metric) or 0)
    raise ValueError(f"unsupported metric {metric!r}")


def compare_pair(
    left: Sequence[dict],
    right: Sequence[dict],
    *,
    metric: str = "f1",
    label: str = "",
    dataset: str = "",
) -> dict[str, Any]:
    """Paired comparison of two aligned record lists (same order, same questions)."""
    if len(left) != len(right):
        raise ValueError("paired comparison requires equal-length record lists")
    diff = _diff(metric)
    differences = [diff(a, b) for a, b in zip(left, right)]
    interval = paired_bootstrap_ci(differences)
    em_pairs = [
        (bool(a.get("exact_match")), bool(b.get("exact_match"))) for a, b in zip(left, right)
    ]
    mcnemar = mcnemar_from_pairs(em_pairs)
    mean_sd = _sd(differences)
    effect = (interval["mean"] / mean_sd) if mean_sd and interval["mean"] is not None else None
    return {
        "label": label,
        "dataset": dataset,
        "metric": metric,
        "n": len(left),
        "mean_difference": interval["mean"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "effect_size_dz": round(effect, 4) if effect is not None else None,
        "mcnemar": mcnemar,
        "p_value": mcnemar["p_value"],
        "resamples": interval.get("resamples"),
    }


def _sd(values: Sequence[float]) -> float | None:
    numbers = [float(v) for v in values]
    if len(numbers) < 2:
        return None
    mean = sum(numbers) / len(numbers)
    variance = sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1)
    return math.sqrt(variance) if variance > 0 else None


def significance_flag(row: dict[str, Any]) -> str:
    """A compact, honest label for a table cell."""
    low, high = row.get("ci_low"), row.get("ci_high")
    if low is None or high is None:
        return "n/a"
    if low > 0:
        return "better"
    if high < 0:
        return "worse"
    return "no difference"
