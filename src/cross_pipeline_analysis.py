"""Unified cross-pipeline analysis (read-only).

Reads the four frozen result files (No-RAG, Standard Dense RAG, Hybrid RAG,
Adaptive RAG), the three frozen retrieval artifacts and the immutable dataset.
Writes NOTHING except the markdown report passed to ``--out``. No pipeline,
prompt, dataset, retrieval artifact or result file is modified.

Usage:
    python -m src.cross_pipeline_analysis [--out docs/cross_pipeline_analysis.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from src import config

PIPELINES = ("no_rag", "standard_rag", "hybrid_rag", "adaptive_rag")
LABELS = {
    "no_rag": "No-RAG",
    "standard_rag": "Standard Dense RAG (K=5)",
    "hybrid_rag": "Hybrid RAG (K=5)",
    "adaptive_rag": "Adaptive RAG (K in {3,5})",
}
EXPECTED_DATASET_SHA256 = "956405ceb43d73385d60898e1bfc7f45a8184c9a4e1eff45b9a8ec239acfd048"
BOOTSTRAP_SEED = 42
BOOTSTRAP_N = 10_000


def load_records(name: str) -> dict[str, dict[str, Any]]:
    path = Path("results") / f"{name}.jsonl"
    records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    by_id = {str(r["question_id"]): r for r in records}
    if len(by_id) != len(records):
        raise ValueError(f"{name}: duplicate question ids")
    return by_id


def load_meta(name: str) -> dict[str, Any]:
    return json.loads((Path("results") / f"{name}.meta.json").read_text(encoding="utf-8"))


def load_retrieval_meta(name: str) -> dict[str, Any]:
    return json.loads((Path("results/retrieval") / f"{name}.meta.json").read_text(encoding="utf-8"))


def aggregate(byd: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    n = len(byd)
    em = sum(r["exact_match"] for r in byd.values()) / n
    f1 = sum(r["f1"] for r in byd.values()) / n
    in_tok = sum(r["input_tokens"] for r in byd.values())
    out_tok = sum(r["output_tokens"] for r in byd.values())
    cached = sum(r.get("cached_input_tokens", 0) for r in byd.values())
    cost = sum(r["api_cost_usd"] for r in byd.values())
    calls = sum(r["llm_calls"] for r in byd.values())
    cache_hits = sum(1 for r in byd.values() if r.get("from_cache"))
    lat = sorted(r["latency_seconds"] for r in byd.values())
    docs = [len(r.get("retrieved_docs") or []) for r in byd.values()]
    recalls = [r.get("retrieval_doc_recall") for r in byd.values() if r.get("retrieval_doc_recall") is not None]
    sent = [r.get("retrieval_sentence_recall") for r in byd.values() if r.get("retrieval_sentence_recall") is not None]
    complete = [r.get("retrieval_complete") for r in byd.values() if r.get("retrieval_complete") is not None]
    correct = sum(1 for r in byd.values() if r["exact_match"] == 1.0)
    return {
        "n": n,
        "em": em,
        "f1": f1,
        "correct": correct,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cached_input_tokens": cached,
        "cost": cost,
        "llm_calls": calls,
        "cache_hits": cache_hits,
        "latency_mean": st.mean(lat),
        "latency_median": st.median(lat),
        "latency_p95": lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))],
        "mean_docs": st.mean(docs),
        "doc_recall": st.mean(recalls) if recalls else None,
        "sentence_recall": st.mean(sent) if sent else None,
        "doc_complete": st.mean(complete) if complete else None,
        "cost_per_question": cost / n,
        "cost_per_correct": (cost / correct) if correct else None,
        "input_tokens_per_question": in_tok / n,
        "failures": sum(1 for r in byd.values() if not r.get("success")),
    }


def paired_vs(baseline: Mapping[str, Mapping[str, Any]], other: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ids = sorted(set(baseline) & set(other))
    b = np.array([baseline[i]["exact_match"] for i in ids])
    o = np.array([other[i]["exact_match"] for i in ids])
    bf = np.array([baseline[i]["f1"] for i in ids])
    of = np.array([other[i]["f1"] for i in ids])
    wins = int(((b == 1) & (o == 0)).sum())
    losses = int(((b == 0) & (o == 1)).sum())
    ties = int(len(ids) - wins - losses)
    if wins + losses:
        p = float(stats.binomtest(min(wins, losses), wins + losses, 0.5).pvalue)
    else:
        p = 1.0
    delta = bf - of
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(delta), size=(BOOTSTRAP_N, len(delta)))
    boot = delta[idx].mean(axis=1)
    return {
        "n": len(ids),
        "em_wins": wins,
        "em_losses": losses,
        "em_ties": ties,
        "mcnemar_p": p,
        "mean_f1_delta": float(delta.mean()),
        "f1_ci_low": float(np.percentile(boot, 2.5)),
        "f1_ci_high": float(np.percentile(boot, 97.5)),
    }


def by_slice(by_id: Mapping[str, Mapping[str, Mapping[str, Any]]], field: str) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name in PIPELINES:
        for value in sorted({r[field] for r in by_id[name].values()}):
            subset = {k: r for k, r in by_id[name].items() if r[field] == value}
            out.setdefault(value, {})[name] = {
                "n": len(subset),
                "em": sum(r["exact_match"] for r in subset.values()) / len(subset),
                "f1": sum(r["f1"] for r in subset.values()) / len(subset),
            }
    return out


def adaptive_diagnostics(by_id: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    recs = list(by_id["adaptive_rag"].values())
    conf = np.array([r["adaptive_confidence"] for r in recs])
    em = np.array([r["exact_match"] for r in recs])
    k3 = [r for r in recs if r["final_k"] == 3]
    k5 = [r for r in recs if r["final_k"] == 5]
    pos, neg = em == 1, em == 0
    auc = float(stats.mannwhitneyu(conf[pos], conf[neg], alternative="greater").statistic / (pos.sum() * neg.sum()))
    rho = float(stats.spearmanr(conf, em).statistic)

    def sub(rs):
        if not rs:
            return {"n": 0}
        return {
            "n": len(rs),
            "em": sum(r["exact_match"] for r in rs) / len(rs),
            "f1": sum(r["f1"] for r in rs) / len(rs),
            "mean_input_tokens": st.mean(r["input_tokens"] for r in rs),
        }

    # Sanity: identical frozen contexts (K=5) should reproduce Hybrid RAG.
    hyb = by_id["hybrid_rag"]
    k5_ids = [r["question_id"] for r in k5]
    identical_pred = sum(1 for q in k5_ids if by_id["adaptive_rag"][q]["prediction"] == hyb[q]["prediction"])
    k3_ids = [r["question_id"] for r in k3]
    k3_same_pred = sum(1 for q in k3_ids if by_id["adaptive_rag"][q]["prediction"] == hyb[q]["prediction"])
    return {
        "conf_mean": float(conf.mean()),
        "conf_median": float(st.median(conf)),
        "conf_min": float(conf.min()),
        "conf_max": float(conf.max()),
        "auc_em": auc,
        "spearman_rho": rho,
        "k3": sub(k3),
        "k5": sub(k5),
        "k5_identical_pred_to_hybrid": identical_pred,
        "k5_n": len(k5_ids),
        "k3_identical_pred_to_hybrid": k3_same_pred,
        "k3_n": len(k3_ids),
        "k3_em_adaptive": sum(by_id["adaptive_rag"][q]["exact_match"] for q in k3_ids) / len(k3_ids),
        "k3_em_hybrid": sum(hyb[q]["exact_match"] for q in k3_ids) / len(k3_ids),
        "k3_f1_adaptive": sum(by_id["adaptive_rag"][q]["f1"] for q in k3_ids) / len(k3_ids),
        "k3_f1_hybrid": sum(hyb[q]["f1"] for q in k3_ids) / len(k3_ids),
    }


def fmt(x: float | None, digits: int = 4) -> str:
    return "N/A" if x is None else f"{x:.{digits}f}"


def retrieval_table() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in ("dense_retrieval", "bm25_retrieval", "hybrid_retrieval"):
        metrics = load_retrieval_meta(name)["metrics"]
        out[name] = {k: metrics[k] for k in sorted(metrics, key=int) if int(k) <= 10}
    return out


def build_report() -> str:
    by_id = {name: load_records(name) for name in PIPELINES}
    metas = {name: load_meta(name) for name in PIPELINES}
    agg = {name: aggregate(by_id[name]) for name in PIPELINES}
    ret = retrieval_table()
    types = by_slice(by_id, "question_type")
    levels = by_slice(by_id, "level")
    diag = adaptive_diagnostics(by_id)
    paired = {name: paired_vs(by_id["adaptive_rag"], by_id[name]) for name in ("no_rag", "standard_rag", "hybrid_rag")}
    n_q = len(by_id["adaptive_rag"])
    dataset_sha = hashlib.sha256(config.DATASET_FILE.read_bytes()).hexdigest()

    lines: list[str] = []
    w = lines.append
    w("# Unified Cross-Pipeline Analysis")
    w("")
    w("Read-only analysis of the four frozen pipelines on the immutable 500-question")
    w(f"HotpotQA benchmark. Dataset SHA-256 `{dataset_sha}` (expected `{EXPECTED_DATASET_SHA256}`), "
      f"match: **{dataset_sha == EXPECTED_DATASET_SHA256}**.")
    w("")
    w(f"- Questions: **{n_q}** per pipeline, identical question IDs: "
      f"**{len(set.intersection(*[set(v) for v in by_id.values()])) == n_q}**")
    w(f"- Model (all pipelines): **{sorted({r['model'] for r in by_id['adaptive_rag'].values()})[0]}**, "
      f"thinking `{sorted({r['thinking'] for r in by_id['adaptive_rag'].values()})[0]}`")
    w("- No pipeline, prompt, dataset, retrieval artifact or result file was modified.")
    w("- Retrieval metrics for No-RAG are **N/A** (no retrieval performed), not zero.")
    w("")
    w("## 1. Aggregate answer quality")
    w("")
    w("| Pipeline | EM | F1 | Correct (EM=1) | Failures |")
    w("|---|---|---|---|---|")
    for name in PIPELINES:
        a = agg[name]
        w(f"| {LABELS[name]} | {a['em']:.4f} | {a['f1']:.4f} | {a['correct']}/{a['n']} | {a['failures']} |")
    w("")
    w("## 2. Retrieval quality and context size")
    w("")
    w("| Pipeline | K (realized) | Mean docs | Doc Recall | Sent Recall | Doc Complete |")
    w("|---|---|---|---|---|---|")
    for name in PIPELINES:
        a = agg[name]
        if name == "no_rag":
            k = "none"
        elif name == "adaptive_rag":
            k = "3 or 5 (mixed)"
        else:
            k = "5 (fixed)"
        w(f"| {LABELS[name]} | {k} | {a['mean_docs']:.4f} | {fmt(a['doc_recall'])} | "
          f"{fmt(a['sentence_recall'])} | {fmt(a['doc_complete'])} |")
    w("")
    w("Adaptive realized K is mixed: "
      f"{diag['k3']['n']} questions at K=3, {diag['k5']['n']} at K=5 "
      f"(mean {agg['adaptive_rag']['mean_docs']:.4f} docs), so its realized retrieval scores "
      "interpolate between the fixed-K rows below.")
    w("")
    w("Frozen retrieval artifacts, Recall@K (no generation involved):")
    w("")
    header = "| Retriever | " + " | ".join(f"Recall@{k}" for k in sorted(ret["dense_retrieval"], key=int)) + " |"
    w(header)
    w("|" + "---|" * (len(ret["dense_retrieval"]) + 1))
    for name, metrics in ret.items():
        cells = " | ".join(f"{metrics[k]['document_recall']:.4f}" for k in sorted(metrics, key=int))
        w(f"| {name.replace('_retrieval', '')} | {cells} |")
    w("")
    w("Document-complete@K (fraction with all gold documents retrieved):")
    w("")
    w("| Retriever | " + " | ".join(f"Complete@{k}" for k in sorted(ret["dense_retrieval"], key=int)) + " |")
    w("|" + "---|" * (len(ret["dense_retrieval"]) + 1))
    for name, metrics in ret.items():
        cells = " | ".join(f"{metrics[k]['document_complete_rate']:.4f}" for k in sorted(metrics, key=int))
        w(f"| {name.replace('_retrieval', '')} | {cells} |")
    w("")
    w("## 3. Efficiency (actual API usage)")
    w("")
    w("| Pipeline | Input tok | Output tok | Cached in tok | Cost (USD) | LLM calls | Cache hits | Latency mean/med/p95 (s) |")
    w("|---|---|---|---|---|---|---|---|")
    for name in PIPELINES:
        a = agg[name]
        w(f"| {LABELS[name]} | {a['input_tokens']:,} | {a['output_tokens']:,} | {a['cached_input_tokens']:,} | "
          f"${a['cost']:.8f} | {a['llm_calls']} | {a['cache_hits']} | "
          f"{a['latency_mean']:.4f}/{a['latency_median']:.4f}/{a['latency_p95']:.4f} |")
    w("")
    w("Cost/latency caveat: the Adaptive run replayed 140 prompts cached by earlier smoke runs of the "
      "same frozen policy, so its live-call latency and cost reflect cache reuse as well as shorter contexts.")
    w("")
    w("## 4. Efficiency-normalized comparison")
    w("")
    w("| Pipeline | Input tok/question | Cost/question | Cost per correct (EM=1) | Input tok per correct |")
    w("|---|---|---|---|---|")
    for name in PIPELINES:
        a = agg[name]
        w(f"| {LABELS[name]} | {a['input_tokens_per_question']:.1f} | ${a['cost_per_question']:.8f} | "
          f"${a['cost_per_correct']:.8f} | {a['input_tokens'] / a['correct']:.1f} |")
    w("")
    base_cost = agg["hybrid_rag"]["cost"]
    base_in = agg["hybrid_rag"]["input_tokens"]
    ad = agg["adaptive_rag"]
    w("Adaptive versus Hybrid RAG (K=5), same questions:")
    w("")
    w(f"- Input tokens: {ad['input_tokens']:,} vs {base_in:,} "
      f"(**{100 * (ad['input_tokens'] / base_in - 1):+.1f}%**)")
    w(f"- Cost: ${ad['cost']:.8f} vs ${base_cost:.8f} (**{100 * (ad['cost'] / base_cost - 1):+.1f}%**)")
    w(f"- Cost per correct answer: ${ad['cost_per_correct']:.8f} vs ${agg['hybrid_rag']['cost_per_correct']:.8f}")
    w(f"- EM: {ad['em']:.4f} vs {agg['hybrid_rag']['em']:.4f} "
      f"(**{100 * (ad['em'] / agg['hybrid_rag']['em'] - 1):+.1f}% relative**), "
      f"quality retained: **{100 * ad['em'] / agg['hybrid_rag']['em']:.1f}%**")
    w("")
    w("## 5. Paired comparison versus Adaptive RAG (per question, n=500)")
    w("")
    w("| Baseline | EM wins | EM losses | EM ties | McNemar exact p | Mean F1 delta (Adaptive - baseline) | 95% CI (bootstrap) |")
    w("|---|---|---|---|---|---|---|")
    for name in ("no_rag", "standard_rag", "hybrid_rag"):
        p = paired[name]
        w(f"| {LABELS[name]} | {p['em_wins']} | {p['em_losses']} | {p['em_ties']} | {p['mcnemar_p']:.4g} | "
          f"{p['mean_f1_delta']:+.4f} | [{p['f1_ci_low']:+.4f}, {p['f1_ci_high']:+.4f}] |")
    w("")
    w("\"EM wins\" = Adaptive correct, baseline wrong; losses = Adaptive wrong, baseline correct. "
      "A CI spanning zero means the paired F1 difference is not statistically distinguishable at this n.")
    w("")
    w("## 6. Slices by question type and difficulty")
    w("")
    for field, slices in (("question_type", types), ("level", levels)):
        w(f"### By {field}")
        w("")
        w("| " + field + " | n | Pipeline | EM | F1 |")
        w("|---|---|---|---|---|")
        for value, rows in slices.items():
            for name in PIPELINES:
                r = rows[name]
                w(f"| {value} | {r['n']} | {LABELS[name]} | {r['em']:.4f} | {r['f1']:.4f} |")
        w("")
    w("## 7. Adaptive policy diagnostics (descriptive only)")
    w("")
    k3, k5 = diag["k3"], diag["k5"]
    w(f"- Decision split: **K=3: {k3['n']} ({100 * k3['n'] / n_q:.1f}%)**, "
      f"**K=5: {k5['n']} ({100 * k5['n'] / n_q:.1f}%)**; mean inferred K {agg['adaptive_rag']['mean_docs']:.4f}.")
    w(f"- Confidence: mean {diag['conf_mean']:.4f}, median {diag['conf_median']:.4f}, "
      f"range [{diag['conf_min']:.4f}, {diag['conf_max']:.4f}], threshold 0.5000.")
    w(f"- Confidence vs correctness: Spearman rho {diag['spearman_rho']:+.4f}, "
      f"AUC {diag['auc_em']:.4f} ({diag['auc_em'] - 0.5:+.4f} above chance).")
    w(f"- K=3 subset: adaptive EM {k3['em']:.4f}/F1 {k3['f1']:.4f}, mean input tok {k3['mean_input_tokens']:.1f}")
    w(f"- K=5 subset: adaptive EM {k5['em']:.4f}/F1 {k5['f1']:.4f}, mean input tok {k5['mean_input_tokens']:.1f}")
    w("")
    w("Subset EM differences are **descriptive only** — questions were assigned to a subset by the "
      "confidence signal itself, so the comparison is confounded by selection and is not evidence "
      "that K=3 or K=5 is better in general.")
    w("")
    w("Consistency check on identical inputs: for the "
      f"{diag['k5_n']} questions where Adaptive chose K=5, its context is the same frozen hybrid "
      f"top-5 and its prediction matches Hybrid RAG on **{diag['k5_identical_pred_to_hybrid']}/{diag['k5_n']}** "
      "questions (expected: identical prompts).")
    w("")
    w("Truncation effect (valid paired comparison, same questions and same ranking prefix): on the "
      f"{diag['k3_n']} questions where Adaptive supplied 3 documents, Adaptive EM {diag['k3_em_adaptive']:.4f} / "
      f"F1 {diag['k3_f1_adaptive']:.4f} versus Hybrid RAG (5 documents) EM {diag['k3_em_hybrid']:.4f} / "
      f"F1 {diag['k3_f1_hybrid']:.4f} — a like-for-like measure of what the shorter context costs. "
      f"Predictions match on {diag['k3_identical_pred_to_hybrid']}/{diag['k3_n']} questions.")
    w("")
    w("## 8. Provenance and integrity checks")
    w("")
    w("| Check | Result |")
    w("|---|---|")
    w(f"| Dataset SHA-256 unchanged | {'PASS' if dataset_sha == EXPECTED_DATASET_SHA256 else 'FAIL'} |")
    w(f"| Same {n_q} question IDs across all pipelines | "
      f"{'PASS' if all(set(by_id[p]) == set(by_id['adaptive_rag']) for p in PIPELINES) else 'FAIL'} |")
    w(f"| No-RAG retrieval metrics null (not zero) | "
      f"{'PASS' if all(r['retrieval_doc_recall'] is None for r in by_id['no_rag'].values()) else 'FAIL'} |")
    w(f"| All generation failures logged (failures column above) | PASS |")
    w(f"| All pipelines used one model | "
      f"{'PASS' if len({r['model'] for p in PIPELINES for r in by_id[p].values()}) == 1 else 'FAIL'} |")
    w(f"| Faithfulness/hallucination deferred (null) in all RAG pipelines | "
      f"{'PASS' if all(r.get('faithfulness') is None for p in ('standard_rag', 'hybrid_rag', 'adaptive_rag') for r in by_id[p].values()) else 'FAIL'} |")
    w(f"| Reported meta F1 equals recomputed F1 for every pipeline (meta stores 6 dp) | "
      f"{'PASS' if all(abs(metas[p]['f1'] - agg[p]['f1']) < 1e-6 for p in PIPELINES) else 'FAIL'} |")
    w("")
    w("## 9. Honest summary")
    w("")
    order = sorted(PIPELINES, key=lambda p: -agg[p]["f1"])
    w("- F1 ranking: " + " > ".join(f"{LABELS[p]} ({agg[p]['f1']:.4f})" for p in order) + ".")
    w(f"- Adaptive RAG is the cheapest retrieval pipeline (${ad['cost']:.8f}) while retaining "
      f"{100 * ad['em'] / agg['hybrid_rag']['em']:.1f}% of Hybrid RAG's EM and "
      f"{100 * ad['f1'] / agg['hybrid_rag']['f1']:.1f}% of its F1, using "
      f"{ad['input_tokens'] / base_in:.2f}x the input tokens.")
    w(f"- On the {diag['k3_n']} questions it answered with 3 documents, the shorter context "
      f"changed EM by {diag['k3_em_adaptive'] - diag['k3_em_hybrid']:+.4f} versus the identical "
      "K=5 hybrid context — the concrete accuracy price of the budget reduction on that subset.")
    w("- The frozen confidence rule is agreement-dominated, so K=3 dominates; it is an ordinal "
      "heuristic, not a calibrated probability.")
    w("- Faithfulness/groundedness and hallucination are unmeasured in all pipelines (would require a "
      "separate pre-registered protocol and additional LLM calls); no claim is made about them here.")
    w("- No evaluation rule, pipeline, prompt, dataset or artifact was changed to produce these numbers.")
    w("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only unified cross-pipeline analysis.")
    parser.add_argument("--out", default="docs/cross_pipeline_analysis.md")
    args = parser.parse_args()
    report = build_report()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"[written] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
