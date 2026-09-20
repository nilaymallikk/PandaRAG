# Query-Complexity-Aware Adaptive RAG for Multi-Hop Question Answering

A controlled, fully reproducible experiment comparing four QA systems on a frozen
HotpotQA benchmark, to test whether **adaptive retrieval can cut retrieval and
computational cost while preserving answer quality**.

All four pipelines use the same dataset, retrieval corpus, embedding model, LLM,
prompt family and deterministic evaluator. The experiment is **complete**; every
number below is recomputed and verified from the frozen result files in
[`docs/final_analysis.md`](docs/final_analysis.md).

**Status:** experiment complete · 500 questions × 4 pipelines · 166 offline tests passing ·
model `deepseek-flash`, thinking disabled, temperature 0.0.

---

## Headline results

500 HotpotQA distractor-validation questions (400 bridge / 100 comparison).
Cost is computed from actual API token usage.

| Pipeline | Retrieval | EM | F1 | Input tokens | Cost (USD) | LLM calls |
|---|---|---|---|---|---|---|
| No-RAG | — | 0.3520 | 0.4756 | 31,120 | 0.00600120 | 490 |
| Standard Dense RAG | dense, K=5 | **0.5100** | **0.6925** | 357,810 | 0.05547390 | 490 |
| Hybrid RAG | dense + BM25 (RRF), K=5 | 0.4900 | 0.6764 | 367,325 | 0.05096271 | 474 |
| **Adaptive RAG** | hybrid, K∈{3,5} | 0.4820 | 0.6598 | **263,640** | **0.02709842** | 360 |

Frozen retriever quality (document Recall@K, no LLM involved):

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| Dense (BGE-small + FAISS) | 0.4500 | 0.8500 | **0.9310** | 1.0000 |
| BM25 | 0.3720 | 0.6850 | 0.7920 | 1.0000 |
| Hybrid (RRF) | 0.4390 | 0.7940 | 0.8920 | 1.0000 |

### What the results support

- **Retrieval is essential for bridge questions.** Adaptive beats No-RAG 106–41 overall
  (McNemar p ≈ 8e-08) and 100–19 on bridge questions (p ≈ 2e-14).
- **Adaptive matches Hybrid quality at roughly half the cost**: EM 0.4820 vs 0.4900
  (p = 0.585; F1 95% CI includes zero), input tokens −28.2%, cost −46.8%, EM retention 98.4%.
- **Standard dense RAG is the most accurate pipeline** (EM 0.5100); its edge over Adaptive is
  small but real: 22–36 on EM (p = 0.087) and F1 Δ −0.0327 [−0.0589, −0.0075], concentrated in
  bridge questions (15–32, p = 0.019).
- **Faithfulness differences are a retrieval effect, not a policy effect.** Overall Adaptive is
  slightly less faithful than Hybrid (Δ −0.027, CI excludes 0), but when both retrieved all gold
  documents the difference vanishes (Δ +0.008, CI includes 0).

### What the results do **not** support

- That K=3 is better than K=5 — the between-subset EM gap is not significant (χ² p = 0.34).
- That the adaptive confidence signal predicts difficulty — Spearman ρ = +0.039 (p = 0.38), AUC 0.52.
- That Adaptive improves answer quality over fixed-K retrieval, or that retrieval hurts
  comparison questions (that inversion is largely an exact-match verbosity artifact).
- Any claim about groundedness of No-RAG, which receives no context.

---

## Research question

> Can query-complexity-aware adaptive retrieval maintain answer quality while reducing
> retrieval and computational cost in multi-hop question answering?

## The four systems

```
A. No-RAG        question ─▶ DeepSeek ─▶ answer
B. Standard RAG  question ─▶ dense retriever ─▶ top-K=5 ─▶ DeepSeek ─▶ answer
C. Hybrid RAG    question ─▶ dense + BM25 ─▶ RRF fusion ─▶ top-K=5 ─▶ DeepSeek ─▶ answer
D. Adaptive RAG  question ─▶ retrieve top-3 ─▶ confidence check ─▶ maybe top-5 ─▶ DeepSeek ─▶ answer
```

### Adaptive policy (frozen before results were seen)

```
confidence = 0.5·A + 0.5·M
A = 1 − ((d1 + b1) − 2) / 18          # top-1 rank agreement of dense/BM25, range 2..20
M = (s3 − s4) / W   if pool ≥ 4 else 0 # fused-score boundary margin, W = 2/61 − 2/70
final K = 3 if confidence ≥ 0.5 else 5  # ties → K=3
actual documents = min(final_k, pool_size)
```

Read only retrieval fields (`rank`, `rrf_score`, `dense_rank`, `bm25_rank`, pool size);
exactly one LLM call per question; no LLM classifier, judge, or iterative retrieval.
The policy constants live in [`src/config.py`](src/config.py) and are validated by
`check_frozen_policy()`.

---

## Repository layout

```
data/                      immutable HotpotQA-500 input (never modified)
src/
  config.py                paths, dataset constants, model resolution, pricing, frozen policy
  dataset.py               immutable dataset loader + fingerprint validation
  llm.py                   shared DeepSeek client (retries, JSONL error log, disk cache)
  retrieval.py             DenseRetriever (BGE-small + FAISS), BM25Retriever, HybridRetriever (RRF)
  retrieval_benchmark.py   retrieval-only benchmark → frozen ranking artifacts (no LLM)
  evaluation.py            retrieval metrics vs. supporting facts (Recall@K, complete@K)
  answer_eval.py           deterministic HotpotQA Exact Match / token F1
  pipelines/
    no_rag.py              A
    standard_rag.py        B  (consumes the frozen dense artifact)
    hybrid_rag.py          C  (consumes the frozen hybrid artifact)
    adaptive_rag.py        D  (consumes the frozen hybrid artifact + frozen policy)
  faithfulness_eval.py     secondary LLM-judge groundedness evaluation (read-only)
  cross_pipeline_analysis.py / research_analysis / final_analysis / final_report.py
                           read-only analyses + report generation
  svg_charts.py            dependency-free SVG figures
tests/                     166 offline unit tests (zero network calls)
docs/                      analyses and reports (see index below)
results/                   generated artifacts (git-ignored)
```

## Setup

Requires Python 3.11+ (developed on 3.14), CPU only.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your key in DEEPSEEK_API_KEY
```

`.env` is git-ignored and the API key is never logged, printed or stored in a result record
(`config.masked_api_key()` is used for diagnostics).

## Reproducing the experiment

Run from the repository root, in order. Retrieval artifacts are produced first; the three RAG
pipelines consume them instead of recomputing embeddings.

```bash
# 1. Frozen retrieval rankings (no LLM calls)
python -m src.retrieval_benchmark --method dense
python -m src.retrieval_benchmark --method bm25
python -m src.retrieval_benchmark --method hybrid

# 2. The four pipelines (one DeepSeek call per question, on-disk cache)
python -m src.pipelines.no_rag
python -m src.pipelines.standard_rag
python -m src.pipelines.hybrid_rag
python -m src.pipelines.adaptive_rag

# 3. Secondary groundedness evaluation (~1,500 judge calls, ~$0.29)
python -m src.faithfulness_eval

# 4. Analyses and reports (read-only, no API calls)
python -m src.final_report
python -m src.cross_pipeline_analysis
```

Every stage supports a smoke run, e.g.
`python -m src.pipelines.adaptive_rag --limit 10 --results-file /tmp/smoke.jsonl --meta-file /tmp/smoke.meta.json`.
Failed API requests are retried and appended to `results/logs/llm_errors.jsonl`, never silently dropped.

## Evaluation

- **Answer quality** — deterministic HotpotQA Exact Match and token F1
  (lowercase, punctuation- and article-stripped). No LLM decides correctness.
- **Retrieval quality** — document- and sentence-level Recall@K and strict "all gold retrieved"
  against `supporting_facts`. Reported as **N/A** for No-RAG, not zero.
- **Faithfulness / hallucination** — a secondary, pre-registered LLM-judge protocol
  (`faithfulness-judge/1.0`) over the exact retrieved context each pipeline received:
  faithfulness = supported claims / total claims; hallucination = any unsupported or contradicted
  claim. The judge never sees gold answers, supporting facts, EM/F1, question type or pipeline
  identity. See [`docs/faithfulness_report.md`](docs/faithfulness_report.md).
- **Cost** — from actual API usage (input, output, cached-input tokens) at the model's published
  per-token prices; never estimated from character counts.

## Output artifacts

| Artifact | Content |
|---|---|
| `results/retrieval/{dense,bm25,hybrid}_retrieval.jsonl` | raw rankings + per-K retrieval metrics |
| `results/{no_rag,standard_rag,hybrid_rag,adaptive_rag}.jsonl` | one record per question (answer, context, scores, tokens, cost, latency, errors) |
| `results/*.meta.json` | run configuration, aggregates, dataset fingerprint |
| `results/analysis/faithfulness.jsonl` | per-answer groundedness verdicts |

`results/` is git-ignored. The frozen inputs the reports cite are pinned by hash
(dataset SHA-256 `956405ce…fd048`; artifact MD5s in `docs/final_analysis.md`).

## Documentation index

| Document | Purpose |
|---|---|
| [`docs/final_analysis.md`](docs/final_analysis.md) | **Start here.** Verified numbers, statistics, error taxonomy, publication figures |
| [`docs/figures/`](docs/figures) | Six publication-ready SVG figures |
| [`docs/research_analysis.md`](docs/research_analysis.md) | Deep-dive: per-question wins/losses, bridge vs comparison, truncation math |
| [`docs/cross_pipeline_analysis.md`](docs/cross_pipeline_analysis.md) | Unified read-only comparison across the four pipelines |
| [`docs/faithfulness_report.md`](docs/faithfulness_report.md) | Secondary groundedness / hallucination evaluation |
| [`experiment.md`](experiment.md) | Original experiment specification |
| [`AGENTS.md`](AGENTS.md) | Contributor/agent conventions |

## Reproducibility and integrity guarantees

- `data/hotpotqa_500.json` is immutable input; its SHA-256 is validated on load.
- Retrieval artifacts are frozen once produced and reused verbatim by every RAG pipeline
  (stored scores are never recomputed).
- Prompts, models, K values and the adaptive policy were fixed before results were inspected;
  the analyses are strictly read-only and never tune or rerun anything.
- Analysis scripts assert their own invariants (`tests/test_final_analysis.py`), and the final
  report is generated from the raw result files rather than from hand-copied numbers.

## Testing

```bash
python -m unittest discover -s tests        # 166 tests, no network/API calls
```

## Limitations

Single run per pipeline (hosted-model nondeterminism not measured); EM/F1 are sensitive to answer
verbosity (roughly a fifth to a third of RAG errors contain the gold answer); faithfulness is
judged by an LLM (1/1,500 parse failure) and is not correctness; retrieval compute is reused, so
the measured savings are context-length savings; the adaptive confidence is an ordinal heuristic,
not a calibrated probability. Full discussion in
[`docs/final_analysis.md` §9](docs/final_analysis.md).

## License

See [`LICENSE`](LICENSE).
