"""Independent integrity audit (read-only).

Recomputes every reported aggregate from the raw per-question JSONL, checks the
statistical machinery (pairing, bootstrap, McNemar), the cost ratio, the ablation
sample, the manifests and the frozen-artifact boundary.

This module never writes to ``results/`` and never mutates a record. It emits a JSON
report; the human-readable audit document is written by hand from it.

Run::

    python -m src.analysis.integrity_audit
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from src import config
from src.analysis import loaders, stats, tables

AUDIT_OUT = config.RESULTS_DIR / "analysis_scope_b" / "integrity_audit.json"
K_VALUES = (1, 3, 5, 10)
TOL = 5e-4  # aggregates in the report are rounded to 3 decimals
BOOTSTRAP = 2000
SEED = 42


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _mean(values: Iterable[float | None]) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _recall(record: Mapping[str, Any], k: int) -> float | None:
    gold = set(record.get("gold_document_ids") or [])
    if not gold:
        return None
    retrieved = list(record.get("retrieved_doc_ids") or [])[:k]
    return len(gold & set(retrieved)) / len(gold)


def _all_gold(record: Mapping[str, Any], k: int) -> float | None:
    gold = set(record.get("gold_document_ids") or [])
    if not gold:
        return None
    return 1.0 if gold.issubset(set(list(record.get("retrieved_doc_ids") or [])[:k])) else 0.0


# ---------------------------------------------------------------------------
# A. retrieval-depth / R@10 labelling
# ---------------------------------------------------------------------------


def check_retrieval_labelling() -> dict[str, Any]:
    rows = []
    for system in loaders.SYSTEMS:
        for dataset in loaders.DATASETS:
            records = loaders.load(system, dataset)
            if not records:
                continue
            depths = [len(r.get("retrieved_doc_ids") or []) for r in records]
            stored = _mean(_recall(r, 10) for r in records)
            at_depth = _mean(_recall(r, min(depths)) if False else _recall(r, d) for r, d in zip(records, depths))
            rows.append(
                {
                    "system": system,
                    "dataset": dataset,
                    "mean_depth": round(mean(depths), 3),
                    "min_depth": min(depths),
                    "max_depth": max(depths),
                    "stored_recall_at_10": round(stored, 6),
                    "recall_at_actual_depth": round(at_depth, 6),
                    "depths_below_10": sum(1 for d in depths if d < 10),
                }
            )

    # Independent Phase-1 benchmark (top-10 retrieval, no pipeline truncation).
    benchmark = {}
    for dataset, split in (("hotpotqa", "validation"), ("2wiki", "dev"), ("musique", "validation")):
        path = config.RESULTS_DIR / "sac_rag" / "retrieval" / f"corpus_retrieval_{dataset}_{split}.meta.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        benchmark[dataset] = {
            method: {
                f"recall@{k}": payload["aggregate"][method][f"recall@{k}"] for k in K_VALUES
            }
            for method in ("dense", "bm25", "hybrid")
        }

    macro = {
        method: {
            f"recall@{k}": round(mean(benchmark[d][method][f"recall@{k}"] for d in benchmark), 4)
            for k in K_VALUES
        }
        for method in ("dense", "bm25", "hybrid")
    }
    return {"depth_rows": rows, "phase1_benchmark": benchmark, "phase1_macro": macro}


# ---------------------------------------------------------------------------
# B. aggregate recomputation
# ---------------------------------------------------------------------------


def check_aggregates() -> dict[str, Any]:
    disagreements = []
    checked = 0
    table_rows = {(r["system"], r["dataset"]): r for r in tables.answer_quality()}
    for system, dataset in loaders.available():
        records = loaders.load(system, dataset)
        meta = loaders.load_meta(system, dataset) or {}
        aggregate = meta.get("aggregate", {})
        recomputed = {
            "n": len(records),
            "n_success": sum(1 for r in records if r.get("success")),
            "em": _mean(r.get("exact_match") for r in records),
            "f1": _mean(r.get("f1") for r in records),
            "recall@10": _mean(r.get("recall@10") for r in records),
            "cost_live": round(
                sum(float(r.get("api_cost_usd") or 0) for r in records if (r.get("live_llm_calls") or 0) > 0), 6
            ),
            "input_tokens": _mean(r.get("input_tokens") for r in records),
        }
        checked += 1
        for key, report_key in (("em", "exact_match"), ("f1", "f1"), ("recall@10", "recall@10")):
            reported = aggregate.get(report_key)
            if reported is not None and recomputed[key] is not None and abs(reported - recomputed[key]) > TOL:
                disagreements.append(
                    {"system": system, "dataset": dataset, "field": key, "meta": reported, "records": recomputed[key]}
                )
        table_row = table_rows.get((system, dataset))
        if table_row:
            for key in ("em", "f1", "recall@10"):
                if table_row.get(key) is not None and abs(table_row[key] - recomputed[key]) > 1e-4:
                    disagreements.append(
                        {"system": system, "dataset": dataset, "field": f"table.{key}",
                         "table": table_row[key], "records": recomputed[key]}
                    )
        if recomputed["cost_live"] and abs(recomputed["cost_live"] - float(aggregate.get("total_api_cost_usd_live") or 0)) > 1e-5:
            disagreements.append(
                {"system": system, "dataset": dataset, "field": "cost_live",
                 "meta": aggregate.get("total_api_cost_usd_live"), "records": recomputed["cost_live"]}
            )
    return {"n_system_dataset_pairs_checked": checked, "disagreements": disagreements}


def check_faithfulness() -> dict[str, Any]:
    detail = config.RESULTS_DIR / "analysis_scope_b" / "faithfulness.jsonl"
    summary_path = config.RESULTS_DIR / "analysis_scope_b" / "faithfulness_summary.json"
    if not detail.exists() or not summary_path.exists():
        return {"status": "missing"}
    entries = _jsonl(detail)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recomputed = {}
    for system, payload in summary.items():
        rows = [e for e in entries if e.get("system") == system and e.get("faithfulness") is not None]
        recomputed[system] = {
            "n_scored": len(rows),
            "faithfulness": round(mean(e["faithfulness"] for e in rows), 4) if rows else None,
            "hallucination": round(mean(e["hallucination_score"] for e in rows), 4) if rows else None,
        }
    disagreements = []
    for system, values in recomputed.items():
        reported = summary[system]
        for key in ("n_scored", "faithfulness", "hallucination"):
            if reported.get(key) != values[key]:
                disagreements.append({"system": system, "field": key, "summary": reported.get(key), "detail": values[key]})
    return {"recomputed": recomputed, "disagreements": disagreements}


# ---------------------------------------------------------------------------
# C. pairing, bootstrap, McNemar
# ---------------------------------------------------------------------------


def _paired_bootstrap(differences: Sequence[float], resamples: int = BOOTSTRAP, seed: int = SEED) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(differences)
    means = []
    for _ in range(resamples):
        means.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    # Same percentile convention as src.analysis.stats (so a recomputation is exact,
    # not merely close: an off-by-one order statistic moves the bound by ~3e-5).
    return means[int(0.025 * resamples)], means[int(0.975 * resamples)]


def _unpaired_bootstrap(left: Sequence[float], right: Sequence[float], resamples: int = BOOTSTRAP, seed: int = SEED) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(left)
    means = []
    for _ in range(resamples):
        left_mean = sum(left[rng.randrange(n)] for _ in range(n)) / n
        right_mean = sum(right[rng.randrange(n)] for _ in range(n)) / n
        means.append(left_mean - right_mean)
    means.sort()
    # Same percentile convention as src.analysis.stats (so a recomputation is exact,
    # not merely close: an off-by-one order statistic moves the bound by ~3e-5).
    return means[int(0.025 * resamples)], means[int(0.975 * resamples)]


def check_pairing_and_tests() -> dict[str, Any]:
    findings = []
    for left, right in loaders.PRIMARY_COMPARISONS:
        for dataset in loaders.DATASETS:
            left_rows = {r["question_id"]: r for r in loaders.load(left, dataset)}
            right_rows = {r["question_id"]: r for r in loaders.load(right, dataset)}
            shared = sorted(set(left_rows) & set(right_rows))
            identical = len(shared) == len(left_rows) == len(right_rows)
            diffs = [left_rows[q]["f1"] - right_rows[q]["f1"] for q in shared]
            paired_ci = _paired_bootstrap(diffs)
            unpaired_ci = _unpaired_bootstrap(
                [left_rows[q]["f1"] for q in shared], [right_rows[q]["f1"] for q in shared]
            )
            em_pairs = [(bool(left_rows[q]["exact_match"]), bool(right_rows[q]["exact_match"])) for q in shared]
            b = sum(1 for a, c in em_pairs if a and not c)
            c = sum(1 for a, cc in em_pairs if cc and not a)
            mcnemar_p = stats.mcnemar_exact(b, c)
            # Contrast: what McNemar on an F1>0.5 binarisation would have given.
            f1_pairs = [((left_rows[q]["f1"] or 0) > 0.5, (right_rows[q]["f1"] or 0) > 0.5) for q in shared]
            fb = sum(1 for a, c2 in f1_pairs if a and not c2)
            fc = sum(1 for a, c2 in f1_pairs if c2 and not a)
            findings.append(
                {
                    "comparison": f"{left} vs {right}",
                    "dataset": dataset,
                    "n_left": len(left_rows),
                    "n_right": len(right_rows),
                    "n_shared": len(shared),
                    "identical_id_sets": identical,
                    "duplicate_ids": {"left": len(loaders.load(left, dataset)) - len(left_rows),
                                      "right": len(loaders.load(right, dataset)) - len(right_rows)},
                    "mean_f1_difference": round(mean(diffs), 6),
                    "paired_ci": [round(v, 6) for v in paired_ci],
                    "unpaired_ci": [round(v, 6) for v in unpaired_ci],
                    "mcnemar_only_left": b,
                    "mcnemar_only_right": c,
                    "mcnemar_p_em": round(mcnemar_p, 6),
                    "mcnemar_p_if_f1_gt_0.5": round(stats.mcnemar_exact(fb, fc), 6),
                }
            )
    return {"rows": findings}


# ---------------------------------------------------------------------------
# D. equivalence framing
# ---------------------------------------------------------------------------


def check_equivalence(margin: float = 0.02) -> dict[str, Any]:
    pairs = loaders.paired("P6", "P4", "hotpotqa") + loaders.paired("P6", "P4", "2wiki") + loaders.paired("P6", "P4", "musique")
    diffs = [a["f1"] - b["f1"] for a, b in pairs]
    ci = _paired_bootstrap(diffs)
    low, high = min(ci[0], ci[1]), max(ci[0], ci[1])
    return {
        "n_pairs": len(pairs),
        "mean_difference": round(mean(diffs), 6),
        "ci95": [round(low, 6), round(high, 6)],
        "margin": margin,
        "ci_within_margin": bool(low > -margin and high < margin),
        "note": (
            "A confidence interval contained in (-margin, +margin) supports equivalence at that "
            "margin; it does not prove it. The margin here is chosen by the auditor after seeing the "
            "data, so it is indicative only — a pre-registered margin and a TOST test are required "
            "before any equivalence claim."
        ),
    }


# ---------------------------------------------------------------------------
# E. cost ratio
# ---------------------------------------------------------------------------


def check_cost_ratio() -> dict[str, Any]:
    rows = []
    for variant in ("A0", "A4"):
        metas = sorted((config.RESULTS_DIR / "ablations" / variant).glob("*.meta.json"))
        records = []
        for path in sorted((config.RESULTS_DIR / "ablations" / variant).glob("*.jsonl")):
            records.extend(_jsonl(path))
        records = [r for r in records if r.get("success")]
        components: dict[str, float] = {}
        for record in records:
            for name, values in (record.get("usage_by_component") or {}).items():
                components[name] = components.get(name, 0.0) + float(values.get("cost_usd") or 0)
        rows.append(
            {
                "variant": variant,
                "n": len(records),
                "cost_live_total": round(sum(float(r.get("api_cost_usd") or 0) for r in records if (r.get("live_llm_calls") or 0) > 0), 6),
                "cost_per_question_replay": round(mean(float(r.get("api_cost_usd") or 0) for r in records), 8),
                "mean_llm_calls": round(mean(float(r.get("llm_calls") or 0) for r in records), 3),
                "cost_by_component": {k: round(v, 6) for k, v in sorted(components.items())},
                "metas": [str(p.relative_to(config.REPO_ROOT)) for p in metas],
            }
        )
    a0, a4 = rows
    ratio = a4["cost_per_question_replay"] / a0["cost_per_question_replay"] if a0["cost_per_question_replay"] else None
    return {"rows": rows, "ratio_replay_cost": round(ratio, 3) if ratio else None}


# ---------------------------------------------------------------------------
# F. ablations
# ---------------------------------------------------------------------------


def check_ablations() -> dict[str, Any]:
    root = config.RESULTS_DIR / "ablations"
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    variants = sorted(p.name for p in root.glob("A*") if p.is_dir())
    per_dataset: dict[str, dict[str, set[str]]] = {}
    for variant in variants:
        for path in sorted((root / variant).glob("*.jsonl")):
            ids = {json.loads(line)["question_id"] for line in path.open() if line.strip()}
            per_dataset.setdefault(variant, {})[path.stem] = ids

    sample_identical = {}
    for dataset in loaders.DATASETS:
        sets = [per_dataset.get(v, {}).get(dataset, set()) for v in variants]
        sets = [s for s in sets if s]
        sample_identical[dataset] = bool(sets) and all(s == sets[0] for s in sets)

    paired_rows = []
    a0 = {p.stem: {json.loads(line)["question_id"]: json.loads(line) for line in p.open() if line.strip()}
          for p in (root / "A0").glob("*.jsonl")}
    a4 = {p.stem: {json.loads(line)["question_id"]: json.loads(line) for line in p.open() if line.strip()}
          for p in (root / "A4").glob("*.jsonl")}
    diffs_all: list[float] = []
    em_pairs: list[tuple[bool, bool]] = []
    for dataset in sorted(set(a0) & set(a4)):
        shared = sorted(set(a0[dataset]) & set(a4[dataset]))
        diffs = [a4[dataset][q]["f1"] - a0[dataset][q]["f1"] for q in shared]
        diffs_all.extend(diffs)
        em_pairs.extend((bool(a4[dataset][q]["exact_match"]), bool(a0[dataset][q]["exact_match"])) for q in shared)
        ci = _paired_bootstrap(diffs)
        b = sum(1 for a, c in [(a4[dataset][q]["exact_match"], a0[dataset][q]["exact_match"]) for q in shared] if a and not c)
        c_count = sum(1 for a, c in [(a4[dataset][q]["exact_match"], a0[dataset][q]["exact_match"]) for q in shared] if c and not a)
        paired_rows.append(
            {
                "dataset": dataset,
                "n_pairs": len(shared),
                "mean_f1_a4_minus_a0": round(mean(diffs), 6),
                "paired_ci95": [round(ci[0], 6), round(ci[1], 6)],
                "mcnemar_p": round(stats.mcnemar_exact(b, c_count), 6),
                "only_a4_correct": b,
                "only_a0_correct": c_count,
            }
        )
    pooled_ci = _paired_bootstrap(diffs_all)
    return {
        "variants_present": variants,
        "sample_identical_across_variants": sample_identical,
        "summary_rows": summary.get("rows", []),
        "paired_a4_vs_a0": paired_rows,
        "pooled_a4_vs_a0": {
            "n_pairs": len(diffs_all),
            "mean_f1_difference": round(mean(diffs_all), 6),
            "paired_ci95": [round(pooled_ci[0], 6), round(pooled_ci[1], 6)],
            "mcnemar_p": round(stats.mcnemar_from_pairs(em_pairs)["p_value"], 6),
        },
    }


# ---------------------------------------------------------------------------
# G. per-dataset visibility
# ---------------------------------------------------------------------------


def check_per_dataset() -> dict[str, Any]:
    rows = []
    for left, right in loaders.PRIMARY_COMPARISONS:
        for dataset in loaders.DATASETS:
            pairs = loaders.paired(left, right, dataset)
            if not pairs:
                continue
            rows.append(
                {
                    "comparison": f"{left} vs {right}",
                    "dataset": dataset,
                    "n": len(pairs),
                    "delta_f1": round(mean(a["f1"] - b["f1"] for a, b in pairs), 6),
                    "delta_em": round(mean(a["exact_match"] - b["exact_match"] for a, b in pairs), 6),
                }
            )
    return {"rows": rows}


# ---------------------------------------------------------------------------
# H. frozen boundary and manifests
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen_boundary() -> dict[str, Any]:
    legacy = ["no_rag", "standard_rag", "hybrid_rag", "adaptive_rag"]
    legacy_rows = {}
    for name in legacy:
        path = config.RESULTS_DIR / f"{name}.jsonl"
        legacy_rows[name] = {
            "exists": path.exists(),
            "n_rows": sum(1 for line in path.open() if line.strip()) if path.exists() else 0,
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
            if path.exists()
            else None,
            "sha256": _sha256(path) if path.exists() else None,
        }
    git_diff = subprocess.run(
        ["git", "status", "--porcelain", "results/"], cwd=config.REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()

    preserved = {}
    for directory in ("history", "history2", "history3", "history4"):
        root = config.RESULTS_DIR / "pandarag" / directory
        preserved[directory] = sorted(p.name for p in root.glob("*.jsonl")) if root.exists() else []

    adaptive = config.REPO_ROOT / "src" / "pipelines" / "adaptive_rag.py"
    adaptive_diff = subprocess.run(
        ["git", "diff", "--quiet", "--", str(adaptive)], cwd=config.REPO_ROOT
    ).returncode

    return {
        "legacy_study1": legacy_rows,
        "git_status_results": git_diff or "(clean)",
        "pandarag_pre_fix_preserved": preserved,
        "adaptive_rag_unchanged_vs_head": adaptive_diff == 0,
    }


def check_manifests() -> dict[str, Any]:
    manifest_doc = config.REPO_ROOT / "docs" / "pandarag" / "experiment_manifest.md"
    corpus = {}
    for dataset, split in (("hotpotqa", "validation"), ("2wiki", "dev"), ("musique", "validation")):
        path = config.CACHE_DIR / "corpus" / dataset / split / "manifest.json"
        corpus[dataset] = json.loads(path.read_text(encoding="utf-8"))["corpus_sha256"] if path.exists() else None

    dataset_sha = json.loads((config.DATA_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))["datasets"]

    mismatches = []
    for system, dataset in loaders.available():
        meta = loaders.load_meta(system, dataset) or {}
        expected_corpus = corpus.get(dataset)
        expected_dataset = dataset_sha["hotpotqa" if dataset == "hotpotqa" else dataset]["sha256"]
        got_corpus = (meta.get("corpus") or {}).get("sha256")
        got_dataset = (meta.get("dataset_fingerprint") or {}).get("sha256")
        if expected_corpus and got_corpus != expected_corpus:
            mismatches.append({"system": system, "dataset": dataset, "field": "corpus_sha256",
                               "expected": expected_corpus, "meta": got_corpus})
        if got_dataset != expected_dataset:
            mismatches.append({"system": system, "dataset": dataset, "field": "dataset_sha256",
                               "expected": expected_dataset, "meta": got_dataset})
    return {
        "manifest_document_exists": manifest_doc.exists(),
        "corpus_fingerprints": corpus,
        "mismatches": mismatches,
    }


def check_tests_and_git() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"], cwd=config.REPO_ROOT, capture_output=True, text=True
    )
    tail = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=config.REPO_ROOT, capture_output=True, text=True
    ).stdout.strip().splitlines()
    return {
        "tests_returncode": completed.returncode,
        "tests_summary": tail[-1] if tail else "",
        "git_modified": [line for line in status if line.startswith(" M")],
        "git_untracked_count": sum(1 for line in status if line.startswith("??")),
        "git_all": status[:40],
    }


# ---------------------------------------------------------------------------


def run_audit() -> dict[str, Any]:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "retrieval_labelling": check_retrieval_labelling(),
        "aggregates": check_aggregates(),
        "faithfulness": check_faithfulness(),
        "pairing_and_tests": check_pairing_and_tests(),
        "equivalence": check_equivalence(),
        "cost_ratio": check_cost_ratio(),
        "ablations": check_ablations(),
        "per_dataset": check_per_dataset(),
        "frozen_boundary": check_frozen_boundary(),
        "manifests": check_manifests(),
        "tests_and_git": check_tests_and_git(),
    }
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:  # pragma: no cover
    payload = run_audit()
    print("wrote", AUDIT_OUT.relative_to(config.REPO_ROOT))
    print("aggregate disagreements:", len(payload["aggregates"]["disagreements"]))
    print("manifest mismatches:", len(payload["manifests"]["mismatches"]))
    print("tests:", payload["tests_and_git"]["tests_summary"])
    print("cost ratio A4/A0:", payload["cost_ratio"]["ratio_replay_cost"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
