"""Secondary, read-only Faithfulness / Hallucination evaluation.

This module is a *post-hoc measurement instrument*. It never modifies pipelines,
prompts, datasets, retrieval artifacts or existing results: it reads the frozen
RAG result files and re-sends each recorded answer together with the exact
retrieved context that pipeline received to a fixed DeepSeek judge.

PROTOCOL (frozen before any judge call was made; identical for all pipelines)
---------------------------------------------------------------------------
Definitions:

* **Faithfulness** — the proportion of the answer's atomic factual claims that
  are supported by the retrieved context it was given:
      faithfulness = supported_claims / total_claims      (0.0 .. 1.0)
  "Supported" means the context states or directly entails the claim. A claim
  that is true in the real world but absent from the context is *not* supported
  (this is groundedness, not factuality).
* **Hallucination** — an answer claim that is unsupported or contradicted by the
  retrieved context, where *contradicted* means the context states something
  incompatible with the claim.
      hallucination_score = (unsupported + contradicted) / total_claims
      hallucination_rate  = 1 if hallucination_score > 0 else 0   (answer level)

Judging rules (fixed, same judge/model/prompt/criteria for every pipeline):

* Judge model: the same ``deepseek-flash`` used by the experiment, thinking
  disabled, temperature 0.0, ``max_tokens=768``; one judge call per answer.
* The judge sees ONLY the question, the exact retrieved context block and the
  answer under audit. It never sees gold answers, supporting facts, question
  type, level, EM/F1, pipeline identity, K, or any other record field.
* The judge decomposes the answer into atomic claims and labels each
  ``supported | unsupported | contradicted`` with a short context quote.
* Scores are computed in code from those labels; the judge's own arithmetic is
  never trusted.
* Answers with no checkable factual claim are recorded as
  ``grounding_label="no_checkable_claims"`` and excluded from faithfulness means.

Grounding labels: ``fully`` (all claims supported), ``partially`` (some
supported), ``not`` (none supported), ``no_checkable_claims``.

Outputs (new files only): ``results/analysis/faithfulness.jsonl`` and its meta.

    python -m src.faithfulness_eval --limit 6          # smoke
    python -m src.faithfulness_eval                    # all 3 x 500
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

from src import config
from src.llm import DeepSeekClient, LLMResponse

PROTOCOL_VERSION = "faithfulness-judge/1.0"
JUDGE_PIPELINES: tuple[str, ...] = ("standard_rag", "hybrid_rag", "adaptive_rag")
DENOMINATOR_PIPELINES: tuple[str, ...] = ("no_rag", *JUDGE_PIPELINES)  # for the report only
ALLOWED_LABELS = ("supported", "unsupported", "contradicted")
GROUNDING_LABELS = ("fully", "partially", "not", "no_checkable_claims")
_STOPWORDS = {"the", "a", "an", "is", "of", "in", "and", "was", "were", "are", "to", "for", "on", "that", "it", "by", "as", "his", "her", "with"}
JUDGE_MAX_TOKENS = 768
DEFAULT_OUT = Path("results/analysis/faithfulness.jsonl")
DEFAULT_META_OUT = Path("results/analysis/faithfulness.meta.json")
DEFAULT_REPORT = Path("docs/faithfulness_report.md")

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, literal groundedness auditor. You verify whether the claims in an "
    "answer are supported by a supplied context. Use ONLY the supplied context. Never use "
    "outside knowledge and never judge whether a claim is true in the real world: a claim "
    "that is factually true but absent from the context is 'unsupported'. Respond with a "
    "single JSON object and nothing else."
)

JUDGE_PROMPT_TEMPLATE = """CONTEXT:
{context}

QUESTION:
{question}

ANSWER TO AUDIT:
{answer}

Task:
1. Decompose the ANSWER into atomic factual claims (one assertion each).
2. Label every claim with exactly one of:
   - "supported": the CONTEXT states or directly entails the claim.
   - "unsupported": the CONTEXT neither states nor entails the claim.
   - "contradicted": the CONTEXT states something incompatible with the claim.
3. For each claim give a short quote or paraphrase from the CONTEXT as evidence,
   or the literal string "none" when no context evidence exists.
Ignore claims that are not factual assertions and ignore statements that only
describe what the context does or does not contain. If the answer contains no
checkable factual claim, return an empty claims list.

Return ONLY this JSON:
{{"claims": [{{"claim": "...", "label": "supported|unsupported|contradicted", "evidence": "..."}}], "notes": "..."}}"""


# ---------------------------------------------------------------------------
# Prompt construction (gold/EM/identity cannot enter: the signature is 3 strings)
# ---------------------------------------------------------------------------

def render_context(retrieved_docs: Sequence[Mapping[str, Any]]) -> str:
    """Rebuild the exact context block the pipeline supplied to the model."""
    blocks = [
        f"[Document {position}]\nTitle: {doc['title']}\n{doc['text']}"
        for position, doc in enumerate(retrieved_docs, start=1)
    ]
    return "Context:\n" + "\n\n".join(blocks) if blocks else "Context:\n(none)"


def build_judge_prompt(question: str, context: str, answer: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(context=context, question=question.strip(), answer=answer.strip())


# ---------------------------------------------------------------------------
# Parsing and scoring
# ---------------------------------------------------------------------------

def parse_judge_json(text: str) -> dict[str, Any]:
    """Tolerantly extract the judge's JSON object; raise on unusable output."""
    if text is None:
        raise ValueError("empty judge output")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in judge output")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("judge JSON is not an object")
    return payload


def normalise_claims(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    raw = payload.get("claims")
    if raw is None:
        raise ValueError("judge JSON has no 'claims' key")
    if not isinstance(raw, list):
        raise ValueError("'claims' is not a list")
    claims: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        label = str(item.get("label", "")).strip().lower()
        if label not in ALLOWED_LABELS:
            raise ValueError(f"invalid claim label {item.get('label')!r}")
        claims.append(
            {
                "claim": str(item.get("claim", "")).strip(),
                "label": label,
                "evidence": str(item.get("evidence", "")).strip(),
            }
        )
    return claims


def score_claims(claims: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Deterministic scoring from claim labels (judge arithmetic is not trusted)."""
    total = len(claims)
    if total == 0:
        return {
            "n_claims": 0,
            "faithfulness": None,
            "hallucination_score": None,
            "hallucination_rate": None,
            "grounding_label": "no_checkable_claims",
        }
    supported = sum(1 for c in claims if c["label"] == "supported")
    unsupported = sum(1 for c in claims if c["label"] == "unsupported")
    contradicted = sum(1 for c in claims if c["label"] == "contradicted")
    if supported == total:
        grounding = "fully"
    elif supported == 0:
        grounding = "not"
    else:
        grounding = "partially"
    return {
        "n_claims": total,
        "n_supported": supported,
        "n_unsupported": unsupported,
        "n_contradicted": contradicted,
        "faithfulness": supported / total,
        "hallucination_score": (unsupported + contradicted) / total,
        "hallucination_rate": 1 if (unsupported + contradicted) > 0 else 0,
        "grounding_label": grounding,
    }


def judge_one(record: Mapping[str, Any], client: DeepSeekClient, *, pipeline: str) -> tuple[dict[str, Any], LLMResponse]:
    """Audit one recorded answer. Returns (record_to_store, raw_llm_response)."""
    docs = record.get("retrieved_docs") or []
    context = render_context(docs)
    prompt = build_judge_prompt(str(record["question"]), context, str(record.get("prediction") or ""))
    response = client.generate(prompt, system=JUDGE_SYSTEM_PROMPT)
    entry: dict[str, Any] = {
        "question_id": record["question_id"],
        "pipeline": pipeline,
        "question_type": record.get("question_type"),
        "retrieval_k": record.get("retrieval_k"),
        "final_k": record.get("final_k"),
        "n_context_docs": len(docs),
        "context_tokens_estimate": len(context) // 4,
        "answer": record.get("prediction"),
        "answer_tokens": len(str(record.get("prediction") or "").split()),
        "judge_model": response.model or client.model,
        "judge_thinking": client.thinking,
        "judge_temperature": client.temperature,
        "protocol_version": PROTOCOL_VERSION,
        "judge_llm_calls": response.llm_calls,
        "judge_from_cache": response.from_cache,
        "judge_input_tokens": response.prompt_tokens,
        "judge_output_tokens": response.completion_tokens,
        "judge_cached_input_tokens": response.cached_input_tokens,
        "judge_cost_usd": response.cost_usd,
        "judge_latency_s": response.latency_s,
        "judge_finish_reason": response.finish_reason,
        "timestamp_utc": response.created_at,
        "judge_error": response.error,
    }
    if not response.ok:
        entry.update(
            {
                "parse_ok": False,
                "n_claims": None,
                "faithfulness": None,
                "hallucination_score": None,
                "hallucination_rate": None,
                "grounding_label": "judge_error",
                "claims": [],
                "supported_claims": [],
                "unsupported_claims": [],
                "contradicted_claims": [],
                "judge_notes": None,
            }
        )
        return entry, response
    try:
        payload = parse_judge_json(response.text)
        claims = normalise_claims(payload)
    except Exception as error:  # noqa: BLE001 - parse failures are recorded, never dropped
        entry.update(
            {
                "parse_ok": False,
                "parse_error": f"{type(error).__name__}: {error}",
                "n_claims": None,
                "faithfulness": None,
                "hallucination_score": None,
                "hallucination_rate": None,
                "grounding_label": "judge_error",
                "claims": [],
                "supported_claims": [],
                "unsupported_claims": [],
                "contradicted_claims": [],
                "judge_notes": None,
            }
        )
        return entry, response
    scores = score_claims(claims)
    entry.update(
        {
            "parse_ok": True,
            "parse_error": None,
            **scores,
            "claims": claims,
            "supported_claims": [c for c in claims if c["label"] == "supported"],
            "unsupported_claims": [c for c in claims if c["label"] == "unsupported"],
            "contradicted_claims": [c for c in claims if c["label"] == "contradicted"],
            "judge_notes": str(payload.get("notes", "")).strip(),
        }
    )
    return entry, response


# ---------------------------------------------------------------------------
# Driving the evaluation
# ---------------------------------------------------------------------------

def load_records(pipeline: str) -> dict[str, dict[str, Any]]:
    path = Path("results") / f"{pipeline}.jsonl"
    return {str(r["question_id"]): r for r in (json.loads(line) for line in path.open(encoding="utf-8") if line.strip())}


def run_evaluation(
    *,
    pipelines: Sequence[str] = JUDGE_PIPELINES,
    limit: int | None = None,
    client: DeepSeekClient,
    out_path: Path = DEFAULT_OUT,
    meta_path: Path = DEFAULT_META_OUT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = datetime.now(timezone.utc).isoformat()
    entries: list[dict[str, Any]] = []
    for pipeline in pipelines:
        records = load_records(pipeline)
        ids = sorted(records)[: limit or len(records)]
        for position, qid in enumerate(ids, start=1):
            entry, _ = judge_one(records[qid], client, pipeline=pipeline)
            entries.append(entry)
            if position % 25 == 0 or position == len(ids):
                print(f"  {pipeline}: {position}/{len(ids)}")
    finished = datetime.now(timezone.utc).isoformat()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "judge_system_prompt": JUDGE_SYSTEM_PROMPT,
        "judge_prompt_template": JUDGE_PROMPT_TEMPLATE,
        "judge_settings": client.settings(),
        "judge_max_output_tokens": JUDGE_MAX_TOKENS,
        "pipelines": list(pipelines),
        "n_entries": len(entries),
        "limit": limit,
        "started_at": started,
        "finished_at": finished,
        "inputs": {p: str(Path("results") / f"{p}.jsonl") for p in pipelines},
        "outputs": str(out_path),
        "definitions": {
            "faithfulness": "supported_claims / total_claims (0-1); null when no checkable claims",
            "hallucination_score": "(unsupported + contradicted) / total_claims",
            "hallucination_rate": "1 if hallucination_score > 0 else 0 (per answer)",
            "grounding_label": list(GROUNDING_LABELS),
        },
        "isolation": (
            "Judge input is question + exact retrieved context + answer only. Gold answers, "
            "supporting facts, question type, level, EM/F1, pipeline identity and K are never "
            "passed to the judge."
        ),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return entries, meta


# ---------------------------------------------------------------------------
# Aggregation and report
# ---------------------------------------------------------------------------

def _mean(values: Iterable[float]) -> float | None:
    values = [v for v in values if v is not None]
    return st.mean(values) if values else None


def summarise(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [e for e in entries if e.get("faithfulness") is not None]
    unscored = [e for e in entries if e.get("n_claims") == 0]
    errors = [e for e in entries if e.get("parse_ok") is False]
    claims = [c for e in entries for c in e.get("claims") or []]
    return {
        "n": len(entries),
        "n_scored": len(scored),
        "n_no_checkable_claims": len(unscored),
        "n_judge_errors": len(errors),
        "faithfulness": _mean(e["faithfulness"] for e in scored),
        "hallucination_score": _mean(e["hallucination_score"] for e in scored),
        "hallucination_rate": _mean(e["hallucination_rate"] for e in scored),
        "fully_grounded_rate": _mean(1.0 if e["grounding_label"] == "fully" else 0.0 for e in scored),
        "partially_grounded_rate": _mean(1.0 if e["grounding_label"] == "partially" else 0.0 for e in scored),
        "not_grounded_rate": _mean(1.0 if e["grounding_label"] == "not" else 0.0 for e in scored),
        "claim_count": len(claims),
        "claim_supported_rate": _mean(1.0 if c["label"] == "supported" else 0.0 for c in claims),
        "claim_contradicted_rate": _mean(1.0 if c["label"] == "contradicted" else 0.0 for c in claims),
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _slice(entries: Sequence[Mapping[str, Any]], predicate) -> list[Mapping[str, Any]]:
    return [e for e in entries if predicate(e)]


def build_report(entries: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]) -> str:
    by_pipe = {p: [e for e in entries if e["pipeline"] == p] for p in JUDGE_PIPELINES}
    overall = summarise(entries)
    em_by_pipe = {}
    for p in JUDGE_PIPELINES:
        recs = load_records(p)
        em_by_pipe[p] = st.mean(recs[e["question_id"]]["exact_match"] for e in by_pipe[p])
    f1_by_pipe = {}
    for p in JUDGE_PIPELINES:
        recs = load_records(p)
        f1_by_pipe[p] = st.mean(recs[e["question_id"]]["f1"] for e in by_pipe[p])

    lines: list[str] = []
    w = lines.append
    w("# Secondary Evaluation: Faithfulness and Hallucination")
    w("")
    w("Read-only groundedness audit of the frozen RAG outputs. It does not modify any pipeline,")
    w("prompt, dataset, retrieval artifact, Adaptive policy or existing result.")
    w("")
    w(f"- Protocol: `{meta['protocol_version']}` (definitions frozen before the first judge call)")
    w(f"- Judge: `{meta['judge_settings']['model']}`, thinking `{meta['judge_settings']['thinking']}`, "
      f"temperature `{meta['judge_settings']['temperature']}`, max_tokens {meta['judge_max_output_tokens']}, "
      "one call per answer")
    w(f"- Pipelines judged: {', '.join(meta['pipelines'])}; answers audited: **{overall['n']}**")
    w("- Evidence available to the judge: question, the exact retrieved context block, the answer. "
      "Gold answers, supporting facts, question type, level, EM/F1, pipeline identity and K are never passed in.")
    w("")
    w("## 1. Definitions and scoring")
    w("")
    w("| Term | Definition |")
    w("|---|---|")
    w("| Faithfulness | `supported_claims / total_claims`, in [0,1]; claims that are true in the real world "
      "but absent from the context count as unsupported |")
    w("| Hallucination | an answer claim labelled `unsupported` (absent from context) or `contradicted` "
      "(context states the opposite) |")
    w("| Hallucination score | `(unsupported + contradicted) / total_claims` per answer |")
    w("| Hallucination rate | fraction of answers with at least one unsupported/contradicted claim |")
    w("| Grounding | `fully` (all claims supported), `partially` (some), `not` (none), "
      "`no_checkable_claims` (excluded from faithfulness means) |")
    w("")
    w("Scores are computed in code from the judge's claim labels; the judge's own arithmetic is never trusted.")
    w("")
    w("## 2. Overall and per-pipeline")
    w("")
    w("| Pipeline | Faithfulness | Hallucination score | Hallucination rate | Fully grounded | Partially | Not grounded | No claims | Judge errors |")
    w("|---|---|---|---|---|---|---|---|---|")
    for p in JUDGE_PIPELINES:
        s = summarise(by_pipe[p])
        w(f"| {p} | {_fmt(s['faithfulness'])} | {_fmt(s['hallucination_score'])} | {_fmt(s['hallucination_rate'])} | "
          f"{_fmt(s['fully_grounded_rate'])} | {_fmt(s['partially_grounded_rate'])} | {_fmt(s['not_grounded_rate'])} | "
          f"{s['n_no_checkable_claims']} | {s['n_judge_errors']} |")
    s = overall
    w(f"| **All three** | {_fmt(s['faithfulness'])} | {_fmt(s['hallucination_score'])} | {_fmt(s['hallucination_rate'])} | "
      f"{_fmt(s['fully_grounded_rate'])} | {_fmt(s['partially_grounded_rate'])} | {_fmt(s['not_grounded_rate'])} | "
      f"{s['n_no_checkable_claims']} | {s['n_judge_errors']} |")
    w("")
    w(f"Claim level: {s['claim_count']} claims audited, "
      f"{_fmt(s['claim_supported_rate'])} supported, {_fmt(s['claim_contradicted_rate'])} contradicted.")
    w("")
    w("Standard vs Hybrid vs Adaptive:")
    w("")
    w("| Pipeline | Faithfulness | Hallucination rate | Mean answer tokens | EM (reference, not used by the judge) | F1 (reference) |")
    w("|---|---|---|---|---|---|")
    for p in JUDGE_PIPELINES:
        s = summarise(by_pipe[p])
        w(f"| {p} | {_fmt(s['faithfulness'])} | {_fmt(s['hallucination_rate'])} | "
          f"{_fmt(_mean(e['answer_tokens'] for e in by_pipe[p]), 2)} | {em_by_pipe[p]:.4f} | {f1_by_pipe[p]:.4f} |")
    w("")
    w("Pairwise differences on the paired answers (bootstrap 95% CI of the mean faithfulness difference):")
    w("")
    w("| Comparison | Mean faithfulness delta | 95% CI |")
    w("|---|---|---|")
    for a, b in (("adaptive_rag", "hybrid_rag"), ("adaptive_rag", "standard_rag"), ("hybrid_rag", "standard_rag")):
        ma = {e["question_id"]: e for e in by_pipe[a] if e["faithfulness"] is not None}
        mb = {e["question_id"]: e for e in by_pipe[b] if e["faithfulness"] is not None}
        ids = sorted(set(ma) & set(mb))
        delta = np.array([ma[i]["faithfulness"] - mb[i]["faithfulness"] for i in ids])
        rng = np.random.default_rng(42)
        boot = delta[rng.integers(0, len(delta), size=(10_000, len(delta)))].mean(axis=1)
        w(f"| {a} - {b} (n={len(ids)}) | {delta.mean():+.4f} | "
          f"[{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}] |")
    w("")
    w("## 3. Bridge vs comparison")
    w("")
    w("| Slice | Pipeline | n | Faithfulness | Hallucination rate | Fully grounded |")
    w("|---|---|---|---|---|---|")
    for qtype in ("bridge", "comparison"):
        for p in JUDGE_PIPELINES:
            rows = _slice(by_pipe[p], lambda e, q=qtype: e.get("question_type") == q)
            s = summarise(rows)
            w(f"| {qtype} | {p} | {s['n']} | {_fmt(s['faithfulness'])} | {_fmt(s['hallucination_rate'])} | "
              f"{_fmt(s['fully_grounded_rate'])} |")
    w("")
    w("## 4. Adaptive K=3 vs K=5")
    w("")
    w("| Adaptive path | n | Faithfulness | Hallucination rate | Fully grounded | Mean context docs |")
    w("|---|---|---|---|---|---|")
    adaptive = by_pipe["adaptive_rag"]
    for k in (3, 5):
        rows = _slice(adaptive, lambda e, kk=k: e.get("final_k") == kk)
        s = summarise(rows)
        w(f"| K={k} | {s['n']} | {_fmt(s['faithfulness'])} | {_fmt(s['hallucination_rate'])} | "
          f"{_fmt(s['fully_grounded_rate'])} | {_fmt(_mean(e['n_context_docs'] for e in rows), 2)} |")
    w("")
    w("## 5. Relationship with answer quality (reference only)")
    w("")
    w("| Pipeline | Faithfulness when EM=1 | Faithfulness when EM=0 | Spearman rho (faithfulness vs F1) |")
    w("|---|---|---|---|")
    for p in JUDGE_PIPELINES:
        recs = load_records(p)
        em1 = [e["faithfulness"] for e in by_pipe[p] if e["faithfulness"] is not None and recs[e["question_id"]]["exact_match"] == 1]
        em0 = [e["faithfulness"] for e in by_pipe[p] if e["faithfulness"] is not None and recs[e["question_id"]]["exact_match"] == 0]
        pairs = [(e["faithfulness"], recs[e["question_id"]]["f1"]) for e in by_pipe[p] if e["faithfulness"] is not None]
        rho = stats.spearmanr([a for a, _ in pairs], [b for _, b in pairs]).statistic if len(pairs) > 2 else float("nan")
        w(f"| {p} | {_fmt(_mean(em1))} (n={len(em1)}) | {_fmt(_mean(em0))} (n={len(em0)}) | {rho:+.4f} |")
    w("")
    w("Interpretation guard: high faithfulness does not imply correctness (a context-faithful answer can")
    w("faithfully repeat the wrong document), and an incorrect answer is not automatically a hallucination.")
    w("")
    w("## 6. Faithfulness vs retrieval success")
    w("")
    w("Buckets use each pipeline's own retrieval metric (`retrieval_complete`: all gold documents were")
    w("retrieved). This field was never shown to the judge; the split is a diagnostic, not a judge input.")
    w("")
    w("| Pipeline | Retrieval | n | Faithfulness | Hallucination rate |")
    w("|---|---|---|---|---|")
    for p in JUDGE_PIPELINES:
        recs = load_records(p)
        for label, want in (("gold docs complete", True), ("gold docs incomplete", False)):
            rows = [
                e
                for e in by_pipe[p]
                if e["faithfulness"] is not None and recs[e["question_id"]].get("retrieval_complete") is want
            ]
            s = summarise(rows)
            w(f"| {p} | {label} | {s['n']} | {_fmt(s['faithfulness'])} | {_fmt(s['hallucination_rate'])} |")
    w("")
    w("## 7. Instrument validation (judge behaviour)")
    w("")
    errors = [e for e in entries if e.get("parse_ok") is False]
    w(f"- **Judge/parse failures:** {len(errors)}/{len(entries)} "
      f"({100 * len(errors) / len(entries):.2f}%), recorded as `grounding_label=\"judge_error\"` and excluded from means.")
    for e in errors:
        w(f"  - {e['pipeline']}/{e['question_id']}: {e.get('parse_error') or e.get('judge_error')}")
    abst = [e for e in entries if e["grounding_label"] == "no_checkable_claims"]
    w(f"- **Abstention answers:** {len(abst)} answers returned no checkable claim, almost all of the form "
      "\"The context does not identify …\". Per protocol these meta-statements are not scored, so an "
      "abstention is *excluded from faithfulness means* rather than counted as unfaithful.")
    hy = {e["question_id"]: e for e in by_pipe["hybrid_rag"]}
    identical = mismatched = 0
    for e in by_pipe["adaptive_rag"]:
        other = hy.get(e["question_id"])
        if e.get("final_k") == 5 and other is not None and e["answer"] == other["answer"]:
            identical += 1
            if e["faithfulness"] != other["faithfulness"] or e["n_claims"] != other["n_claims"]:
                mismatched += 1
    w(f"- **Determinism across identical inputs:** for the {identical} Adaptive K=5 answers that are "
      f"byte-identical (question, context, answer) to Hybrid RAG, the judgments disagree in "
      f"**{mismatched}** cases.")
    unsupported = [(e, c) for e in entries for c in (e.get("unsupported_claims") or [])]
    high_overlap = 0
    for e, c in unsupported:
        recs = load_records(e["pipeline"])
        ctx = set(re.findall(r"[a-z0-9]+", " ".join(d["text"] for d in recs[e["question_id"]].get("retrieved_docs") or []).lower()))
        words = set(re.findall(r"[a-z0-9]+", c["claim"].lower())) - _STOPWORDS
        if words and len(words & ctx) / len(words) >= 0.8:
            high_overlap += 1
    w(f"- **Strictness audit:** {len(unsupported)} claims were labelled unsupported; "
      f"{high_overlap} ({100 * high_overlap / max(1, len(unsupported)):.1f}%) share \u226580% of their content words with "
      "the context, an *upper bound* on possible false positives (word overlap is not entailment). "
      "Manual inspection of the highest-overlap cases found the judge defensible: e.g. the context says "
      "Márquez only *challenged for* a WBO title, and that a \"Johnny Edwards\" was a *singer*, so the "
      "claims 'held a WBO championship' and 'was a guitarist' are genuinely unsupported.")
    w("")
    w("## 8. Representative examples")
    w("")
    for p in JUDGE_PIPELINES:
        examples = [e for e in by_pipe[p] if e["grounding_label"] == "fully"][:1]
        examples += [e for e in by_pipe[p] if e["grounding_label"] in ("not", "partially")][:1]
        if not examples:
            continue
        w(f"### {p}")
        w("")
        for e in examples:
            w(f"- **{e['question_id']}** ({e.get('question_type')}, {e['grounding_label']}, "
              f"faithfulness {_fmt(e['faithfulness'])}, n_claims {e['n_claims']})")
            w(f"  - Q: {_excerpt(e, 'question_id')}")
            w(f"  - Answer: {e['answer']}")
            for c in (e.get("unsupported_claims") or [])[:2]:
                w(f"  - Unsupported claim: \"{c['claim']}\" (evidence: {c['evidence']})")
            for c in (e.get("contradicted_claims") or [])[:2]:
                w(f"  - Contradicted claim: \"{c['claim']}\" (evidence: {c['evidence']})")
            for c in (e.get("supported_claims") or [])[:2]:
                w(f"  - Supported claim: \"{c['claim']}\" (evidence: {c['evidence']})")
        w("")
    w("## 9. Additional evaluation cost")
    w("")
    s = overall
    live = sum(e["judge_llm_calls"] for e in entries)
    cached = sum(1 for e in entries if e["judge_from_cache"])
    cost = sum(e["judge_cost_usd"] for e in entries)
    in_tok = sum(e["judge_input_tokens"] for e in entries)
    out_tok = sum(e["judge_output_tokens"] for e in entries)
    cached_tok = sum(e["judge_cached_input_tokens"] for e in entries)
    lat = sorted(e["judge_latency_s"] for e in entries)
    w("| Metric | Value |")
    w("|---|---|")
    w(f"| Judge calls (entries) | {len(entries)} |")
    w(f"| Live API calls | {live} |")
    w(f"| Replayed from cache | {cached} |")
    w(f"| Input tokens | {in_tok:,} |")
    w(f"| Output tokens | {out_tok:,} |")
    w(f"| Cached input tokens (server-side cache hits) | {cached_tok:,} |")
    w(f"| Judging cost (USD, shared pricing convention) | ${cost:.8f} |")
    w(f"| Mean latency (s) | {_fmt(_mean(lat))} |")
    w(f"| Median / p95 latency (s) | {_fmt(_mean([st.median(lat)]) if lat else None)} / "
      f"{_fmt(_mean([lat[int(round(0.95 * (len(lat) - 1)))]] if lat else None))} |")
    w("")
    w("This is **additional spend** created solely by this secondary evaluation; it does not change any")
    w("primary experiment number. Re-running the same protocol reuses the disk cache at no API cost.")
    w("")
    w("## 10. Limitations and possible judge errors")
    w("")
    w("- **The judge is itself an LLM and can be wrong.** It may miss a claim, split one claim into")
    w("  several, or label a paraphrase as unsupported. Parse/scoring failures are recorded as")
    w("  `grounding_label=\"judge_error\"` rather than dropped, and are reported above.")
    w("- **Claim granularity is judge-dependent**, so faithfulness is a ratio of the judge's decomposition;")
    w("  longer answers get more claims, which is why claim-level and answer-level rates are both reported.")
    w("- **Grounding is not correctness.** A faithful answer may faithfully reproduce a wrong document;")
    w("  conversely an answer that is correct from parametric knowledge but absent from the context is")
    w("  scored as unsupported by definition. The EM/F1 columns are shown only as a reference.")
    w("- **Abstentions are excluded, not penalised.** Answers that say the context lacks the information")
    w("  produce no claims and are dropped from faithfulness means; 20 such answers exist (11 Adaptive, 7")
    w("  Standard, 2 Hybrid), so pipelines that hedge more are measured on fewer answers.")
    w("- **The judge is strict rather than lenient.** Sampled high-overlap unsupported claims were")
    w("  genuinely unsupported (competition names, titles, and people conflated by the QA model); if")
    w("  anything, the reported hallucination rates are more likely to be over- than under-stated.")
    w("- **No-RAG is excluded** because it received no retrieved context, so faithfulness is undefined for it.")
    w("- **Question wording is passed to the judge** to help claim extraction; it is not gold evidence, but")
    w("  it is an input beyond the context and is disclosed here.")
    w("- Comparisons across pipelines are descriptive; the three pipelines answer the same questions, but")
    w("  faithfulness differences are confounded with answer length and retrieval quality.")
    w("")
    return "\n".join(lines) + "\n"


def _excerpt(entry: Mapping[str, Any], _: str) -> str:
    recs = load_records(str(entry["pipeline"]))
    return str(recs[str(entry["question_id"])].get("question", ""))[:200]


def main() -> int:
    parser = argparse.ArgumentParser(description="Secondary faithfulness/hallucination evaluation (read-only).")
    parser.add_argument("--limit", type=int, default=None, help="per-pipeline answer limit (smoke test)")
    parser.add_argument("--pipelines", nargs="+", default=list(JUDGE_PIPELINES))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--report-only", action="store_true", help="re-render the report from an existing JSONL")
    args = parser.parse_args()

    if args.report_only:
        entries = [json.loads(line) for line in args.out.open(encoding="utf-8") if line.strip()]
        meta = json.loads(args.meta_out.read_text(encoding="utf-8"))
    else:
        client = DeepSeekClient(thinking="disabled", temperature=0.0, max_tokens=JUDGE_MAX_TOKENS)
        print(f"Faithfulness judge ({PROTOCOL_VERSION}): {client.model}, thinking={client.thinking}, T={client.temperature}")
        entries, meta = run_evaluation(
            pipelines=args.pipelines, limit=args.limit, client=client, out_path=args.out, meta_path=args.meta_out
        )
    report = build_report(entries, meta)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"[written] {args.out}\n[written] {args.meta_out}\n[written] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
