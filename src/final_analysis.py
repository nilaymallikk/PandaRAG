"""Final paper-readiness analysis over the frozen experiment (read-only).

Recomputes and verifies every headline number from the frozen result files,
runs the paired/statistical battery, builds an error taxonomy, and emits
publication tables plus dependency-free SVG figures.

It never writes to any experiment input or output: only new files under
``docs/``.

    python -m src.final_analysis
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from src import svg_charts as sc

PIPELINES = ("no_rag", "standard_rag", "hybrid_rag", "adaptive_rag")
LABELS = {
    "no_rag": "No-RAG",
    "standard_rag": "Standard RAG",
    "hybrid_rag": "Hybrid RAG",
    "adaptive_rag": "Adaptive RAG",
}
RETRIEVERS = ("dense_retrieval", "bm25_retrieval", "hybrid_retrieval")
N_QUESTIONS = 500
SEED = 42
BOOT = 10_000
DATASET_SHA256 = "956405ceb43d73385d60898e1bfc7f45a8184c9a4e1eff45b9a8ec239acfd048"
FIG_DIR = Path("docs/figures")
REPORT = Path("docs/final_analysis.md")
CLASSES = ("correct", "verbosity/EM artifact", "context truncation", "retrieval failure", "model error", "no retrieval (parametric only)")


# ---------------------------------------------------------------------------
# Loading and small helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_results() -> dict[str, dict[str, dict[str, Any]]]:
    return {p: {r["question_id"]: r for r in _read_jsonl(Path("results") / f"{p}.jsonl")} for p in PIPELINES}


def load_retrieval() -> dict[str, dict[str, dict[str, Any]]]:
    return {n: {r["question_id"]: r for r in _read_jsonl(Path("results/retrieval") / f"{n}.jsonl")} for n in RETRIEVERS}


def load_faithfulness() -> dict[tuple[str, str], dict[str, Any]]:
    return {(e["pipeline"], e["question_id"]): e for e in _read_jsonl(Path("results/analysis/faithfulness.jsonl"))}


def load_metas() -> dict[str, dict[str, Any]]:
    return {p: json.loads((Path("results") / f"{p}.meta.json").read_text(encoding="utf-8")) for p in PIPELINES}


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def contained(prediction: str | None, gold: str) -> bool:
    g, p = tokens(gold), tokens(prediction)
    return bool(g) and all(t in p for t in g)


def starts_with_gold(prediction: str | None, gold: str) -> bool:
    g, p = " ".join(tokens(gold)), " ".join(tokens(prediction))
    return bool(g) and (p == g or p.startswith(g + " "))


def mean(values: Sequence[float]) -> float:
    values = [v for v in values if v is not None]
    return st.mean(values) if values else float("nan")


def boot_ci(delta: Sequence[float], *, seed: int = SEED, n: int = BOOT) -> tuple[float, float, float]:
    d = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def mcnemar(a: Mapping[str, Mapping[str, Any]], b: Mapping[str, Mapping[str, Any]], ids: Sequence[str]) -> tuple[int, int, float]:
    wins = sum(1 for i in ids if a[i]["exact_match"] == 1 and b[i]["exact_match"] == 0)
    losses = sum(1 for i in ids if a[i]["exact_match"] == 0 and b[i]["exact_match"] == 1)
    p = float(stats.binomtest(min(wins, losses), wins + losses, 0.5).pvalue) if wins + losses else 1.0
    return wins, losses, p


def f4(x: float, digits: int = 4) -> str:
    return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def fmoney(x: float) -> str:
    return f"${x:.8f}"


def fingerprint(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

def classify(pipeline: str, qid: str, results: Mapping[str, Mapping[str, Any]], retrieval: Mapping[str, Mapping[str, Any]]) -> str:
    """Deterministic error taxonomy for one answer (precedence order documented in the report)."""
    if pipeline == "no_rag":
        return "correct" if results[qid]["exact_match"] == 1 else (
            "verbosity/EM artifact" if contained(results[qid]["prediction"], results[qid]["gold_answer"]) else "no retrieval (parametric only)"
        )
    record = results[qid]
    if record["exact_match"] == 1:
        return "correct"
    if contained(record["prediction"], record["gold_answer"]):
        return "verbosity/EM artifact"
    if pipeline == "adaptive_rag" and record["final_k"] == 3:
        hybrid = retrieval["hybrid_retrieval"][qid]["metrics"]
        if not hybrid["3"]["document_complete"] and hybrid["5"]["document_complete"]:
            return "context truncation"
    if not record.get("retrieval_complete"):
        return "retrieval failure"
    return "model error"


def taxonomy(results, retrieval, ids) -> dict[str, Counter]:
    return {p: Counter(classify(p, i, results[p], retrieval) for i in ids) for p in PIPELINES}
