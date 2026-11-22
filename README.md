# Adaptive retrieval for multi-hop QA — two studies

This repository holds **two experiments** on the same question: *can retrieval be made
cheaper or better without losing answer quality?*

| | Study 1 — frozen | Study 2 — current |
|---|---|---|
| Question | Does adaptive **K** cut cost while keeping quality? | Does **type-routed retrieval + evidence-gap control** beat uniform RAG? |
| Data | HotpotQA distractor, 500 questions | HotpotQA + 2WikiMultiHopQA + MuSiQue, 500 each |
| Corpus | each question's 10 distractor paragraphs | **pooled candidate corpus** per dataset (3.2k–7.2k documents) |
| Systems | 4 (No-RAG, dense, hybrid, adaptive-K) | 9 (P0 … P6, incl. both studies' baselines) |
| Status | complete, untouched | complete, audited, documented |
| Tests | 172 | 300 total |

Study 1's artifacts and code are frozen: `src/pipelines/{no_rag,standard_rag,hybrid_rag,adaptive_rag}.py`,
`results/*.jsonl`, `results/retrieval/` and `data/hotpotqa_500.json` are treated as immutable
inputs by everything added later.

---

# Study 2 — SAC-RAG reimplementation and PandaRAG (current)

A best-effort architectural reimplementation of **SAC-RAG** (question-type-adaptive retrieval +
context compression) on a shared corpus, plus **PandaRAG**, which replaces the single global
retrieval decision with requirement-level evidence tracking.

> **This is not an exact reproduction of the base paper.** Its corpus, classifier prompt,
> decomposition prompt and compression prompt are unpublished, and the paper contradicts itself on
> the compressor model. Every deviation is recorded in
> [`docs/pandarag/sac_rag_reproduction_protocol.md`](docs/pandarag/sac_rag_reproduction_protocol.md)
> (DV1–DV12).

## Headline result

Nine systems, three datasets, 500 questions each, one frozen generator and prompt, paired statistics
on identical question ids (pooled, n = 1,500 per comparison).

| ID | System | EM | F1 | Candidate R@10 | R@ctx | Cost / question |
|---|---|---:|---:|---:|---:|---:|
| P0 | No-RAG | 0.222 | 0.308 | — | — | $0.3e-4 |
| P1c | Dense RAG (corpus) | 0.433 | 0.548 | — (depth 5) | 0.716 | $2.0e-4 |
| P2c | Hybrid, fusion depth 5 | 0.416 | 0.532 | — (depth 5) | 0.712 | $1.9e-4 |
| P2c10 | Hybrid, fusion depth 10 | 0.422 | 0.537 | 0.775 | 0.705 | $1.6e-4 |
| P3c | Adaptive RAG (transferred policy) | 0.392 | 0.512 | 0.775 | 0.654 | $1.3e-4 |
| P1n | Naive RAG (paper's 10→5) | 0.433 | 0.548 | 0.771 | 0.716 | $2.0e-4 |
| P4 | **SAC-RAG Adaptive** | **0.459** | **0.583** | 0.788 | 0.787 | $3.9e-4 |
| P5 | Full SAC-RAG (+ compression) | 0.453 | 0.575 | 0.788 | 0.760 | $4.7e-4 |
| P6 | **PandaRAG** | **0.459** | **0.583** | 0.787 | 0.788 | $9.9e-4 |

### What the results support

- **Retrieval is essential**: F1 0.548 with dense retrieval vs 0.308 without (EM 0.433 vs 0.222).
- **Type-specific retrieval + BGE reranking is the real gain**: P4 vs P1c is ΔF1 **+0.035**
  [0.020, 0.049], McNemar p = 0.0006 (n = 1,500 paired), EM +0.026.
- **Compression is a grounding tool, not a cost saving**: the generation prompt shrinks from
  1,576 to 94 tokens (−94%) but the extra call raises total tokens 14% and cost 21%; quality is
  unchanged (ΔF1 −0.008, p = 0.43) and faithfulness improves 0.63 → 0.67.
- **Candidate recall is not consumed evidence**: pipelines that retrieve 10 and feed 5 lose 5–12
  points (P2c10 0.775 → 0.705; P3c 0.775 → 0.654). The `R@ctx` column is what the generator saw.

### What the results do **not** support — the negative result

- **PandaRAG does not beat SAC-RAG Adaptive.** ΔF1 = **−0.000** (95% CI [−0.003, +0.003]),
  McNemar p = 1.0, at **2.53× the cost** per question. That is "no detectable difference", not
  proven equivalence: no margin was pre-registered.
- **The requirement-level controller did not convert gaps**: targeted retrieval fired on
  41% / 59% / 78% of questions (HotpotQA / 2Wiki / MuSiQue) and a recovery round converted an
  uncovered requirement to supported in **1.0% / 0.0% / 0.8%** of rounds. This rate is
  *evaluator-measured*; the evaluator has no human-labelled validation, so it bounds this
  controller's observed effect rather than proving the corpus is exhausted.
- **The ablation is flat**: A0 → A4 moves EM 0.430 → 0.433 and F1 0.563 → 0.570 while cost rises
  2.60× ($0.00038 → $0.00099 per question). A3 and A4 are identical because the evaluator only sees
  the same 10 documents the context window holds (299/300 byte-identical contexts).
- **The observed bottleneck is retrieval completeness, not budget allocation.** Best all-gold@10 is
  0.588 (P4/P5); on MuSiQue only ~0.24 of questions have every gold document in the top 10, and the
  missing documents never enter the candidate pool.

## The nine systems

```
P0    question ─▶ generator                                   (closed book)
P1c   question ─▶ dense top-5 ─▶ generator                    (corpus baseline)
P2c   question ─▶ dense-5 + BM25-5 ─▶ RRF ─▶ top-5 ─▶ gen      (fusion depth 5)
P2c10 question ─▶ dense-10 + BM25-10 ─▶ RRF ─▶ feed 5 ─▶ gen   (fusion depth 10, corrected)
P3c   question ─▶ hybrid-10 ─▶ frozen confidence ─▶ K=3|5 ─▶ gen  (transferred policy)
P1n   question ─▶ dense-10 ─▶ feed 5 ─▶ gen                   (paper's "10→5")
P4    question ─▶ semantic type ─▶ Fact/Definition/Reasoning retrieval ─▶ BGE rerank ─▶ gen
P5    P4 ─▶ LLM extractive compression ─▶ gen
P6    question ─▶ evidence plan ─▶ P4 retrieval ─▶ per-requirement evaluation ─▶ coverage
              ─▶ gap controller (targeted recovery, budgets) ─▶ evidence-preserving context ─▶ gen
```

P4's routing: **Fact** = BM25 top-10 + dense top-10 → union → rerank → top-10; **Definition** =
dense top-8; **Reasoning** = LLM decomposition into ≤4 sub-questions → dense top-10 each → union →
rerank → top-10. P6 adds a planner (1 requirement for Fact/Definition, up to 4 for Reasoning), an
evidence evaluator (`supported / partial / insufficient / contradicted / unknown`), per-requirement
coverage, and a rule-based controller under fixed budgets (2 extra rounds, 20 unique documents,
8 retriever calls, 4,000 context tokens).

Full architecture diagrams and per-figure explanations:
[`docs/pandarag/final_methodology.md`](docs/pandarag/final_methodology.md) and
[`docs/pandarag/final_research_report.md`](docs/pandarag/final_research_report.md).

## Datasets and corpus

| Dataset | Questions | Pooled documents | Gold documents | Evidence |
|---|---:|---:|---:|---|
| HotpotQA (validation) | 500 | 4,928 | 994 | sentence-level |
| 2WikiMultiHopQA (dev) | 500 | 3,246 | 990 | sentence-level |
| MuSiQue (validation) | 500 | 7,192 | 912 | paragraph-level (`sent_id = -1`) |

The corpus is the union of every paragraph shipped with each question, deduplicated by
`sha256(dataset, split, title, text)`. Document ids are **content-derived**, never positional, so
MuSiQue's 421 titles that carry different text stay distinct (6,356 titles → 7,192 documents).

**Recall here is not comparable with the base paper's**: this is a closed-book pool of 3–7k
documents, not Wikipedia at scale.

## Repository layout

```
Study 1 (frozen)                    Study 2 (this work)
────────────────                    ───────────────────
src/config.py                       src/datasets/          canonical QAExample / RuntimeExample
src/dataset.py                      src/corpus_retrieval/  corpus, dense, BM25, RRF, indexes, benchmark
src/retrieval.py                    src/experiment_config.py  JSON configs + schema + fingerprint
src/retrieval_benchmark.py          configs/                dataset, retrieval, sac_rag, pandarag
src/evaluation.py                   src/pipelines/shared/   retriever, LLM-JSON, generator, runner
src/answer_eval.py                  src/pipelines/sac_rag/  P4, P5
src/pipelines/no_rag.py             src/pipelines/pandarag/ P6: planner, evaluator, coverage,
src/pipelines/standard_rag.py                               controller, context, pipeline
src/pipelines/hybrid_rag.py         src/pipelines/corpus_baselines.py  P0, P1c, P2c, P2c10, P3c, P1n
src/pipelines/adaptive_rag.py       src/experiments/        matrix + ablation drivers
src/faithfulness_eval.py            src/analysis/           loaders, stats, metrics, figures,
src/cross_pipeline_analysis.py                              faithfulness, error analysis, audit, report
src/final_analysis.py               docs/pandarag/          audit, protocol, plan, phase-1 report,
src/final_report.py                                         final report, limitations, manifest, audit
src/svg_charts.py                   docs/figures/fig1-10    Study 2 figures
docs/{final_analysis,…}.md          tests/test_{datasets,corpus,retrieval_corpus,sac_rag,
                                           pandarag,baselines,experiment_integrity}.py
```

## Reproducing Study 2

Local retrieval needs no API key; only the pipelines do. `.env` with `DEEPSEEK_API_KEY` is
git-ignored and never logged.

```bash
# 0. environment (Python 3.14 is the supported interpreter for this venv)
ln -sf /usr/bin/python3.14 .venv/bin/python      # only if the venv symlink breaks
python -m pytest tests/ -q                       # 300 tests, no network

# 1. corpora + indexes (local, ~15 min for the embeddings)
python -m src.corpus_retrieval.corpus_cli
python -m src.corpus_retrieval.index_cli

# 2. retrieval sanity metrics (no LLM)
python -m src.corpus_retrieval.benchmark --dataset hotpotqa
python -m src.corpus_retrieval.benchmark --dataset 2wiki
python -m src.corpus_retrieval.benchmark --dataset musique

# 3. experiment matrix (resumable; safe to re-run after interruption)
python -m src.experiments.run_matrix --concurrency 3

# 4. reduced-sample ablations (100 stratified questions per dataset)
python -m src.experiments.run_ablations --sample-size 100

# 5. faithfulness judging (frozen judge protocol, stratified sample)
python -m src.analysis.faithfulness_scope_b --sample 60

# 6. analysis, figures, documents, integrity audit
python -m src.analysis.figures
python -m src.analysis.report
python -m src.analysis.integrity_audit
```

Every run resumes: re-running a command fills only missing or failed questions. Rate-limited
attempts stay in the JSONL as `success: false` rows and are never turned into zeros; analysis folds
each file to one row per question, preferring the success.

## Study 2 artifacts and integrity

| Artifact | Content |
|---|---|
| `results/{baseline_*,sac_rag_adaptive,sac_rag_full,pandarag}/<dataset>.jsonl` | one record per question: prediction, retrieved ids, context, per-component tokens/cost, coverage trace, errors |
| `results/ablations/<A0–A5>/<dataset>.jsonl` | reduced-sample ablations on one fixed stratified subset |
| `results/analysis_scope_b/` | faithfulness judgements, error taxonomy, depth metrics, integrity audit |
| `docs/pandarag/results_manifest.json` | SHA-256, row counts and fingerprints for all 27 runs |
| `docs/pandarag/phase1_artifacts.json` | dataset, corpus and index fingerprints |
| `docs/figures/fig1-10.svg` | the ten figures |

- **Frozen inputs**: datasets are validated against `data/dataset_manifest.json`; corpora and
  indexes are bound to a corpus fingerprint and refuse a stale pair.
- **No gold at runtime**: evaluation-side records (`QAExample`) carry answer and gold evidence;
  runtime components receive `RuntimeExample`, which physically has neither.
- **Cost honesty**: generation cost is separated from auxiliary LLM cost (planner, evaluator,
  compressor); a replayed run is labelled `from_cache` and never reported as live spend.
- **Failure accounting**: 27 runs, 13,500 canonical records, 381 superseded HTTP-429 attempts,
  99 duplicate rows in one cell, **0 terminal failures**, denominators 500 everywhere.
- **Audit**: [`docs/pandarag/final_integrity_audit.md`](docs/pandarag/final_integrity_audit.md)
  re-derives every aggregate from the raw records, checks pairing, bootstrap and McNemar, the cost
  ratio, the ablation sample, the frozen boundary and the manifests.

## Study 2 documentation

| Document | Purpose |
|---|---|
| [`docs/pandarag/final_research_report.md`](docs/pandarag/final_research_report.md) | **Start here.** 16-section report: method, results, statistics, ablations, grounding, limitations |
| [`docs/pandarag/final_methodology.md`](docs/pandarag/final_methodology.md) | systems, retrieval protocol, PandaRAG details, budgets, statistics |
| [`docs/pandarag/final_results.md`](docs/pandarag/final_results.md) | tables and paired comparisons |
| [`docs/pandarag/final_ablation_report.md`](docs/pandarag/final_ablation_report.md) | A0–A5 with paired CIs |
| [`docs/pandarag/final_error_analysis.md`](docs/pandarag/final_error_analysis.md) | error taxonomy and PandaRAG traces |
| [`docs/pandarag/limitations.md`](docs/pandarag/limitations.md) | 22 stated limitations |
| [`docs/pandarag/sac_rag_reproduction_protocol.md`](docs/pandarag/sac_rag_reproduction_protocol.md) | paper-parameter ledger, DV1–DV12 |
| [`docs/pandarag/final_integrity_audit.md`](docs/pandarag/final_integrity_audit.md) | independent verification of every headline number |
| [`docs/pandarag/repository_audit.md`](docs/pandarag/repository_audit.md) · [`implementation_plan.md`](docs/pandarag/implementation_plan.md) · [`phase1_report.md`](docs/pandarag/phase1_report.md) · [`engineering_log.md`](docs/pandarag/engineering_log.md) · [`reproduction_instructions.md`](docs/pandarag/reproduction_instructions.md) · [`experiment_manifest.md`](docs/pandarag/experiment_manifest.md) | audits, plan, phase 1, defect log, reproduction, manifest |

## Study 2 limitations (short form)

Pooled corpus, not open-domain Wikipedia — recall is not comparable with the paper. 500 questions per
dataset (paper: 1,000). Generator and compressor are `deepseek-v4.1-flash` (paper: `gpt-5-mini` and,
inconsistently, `qwen-flash`). Unpublished prompts are implementation choices. The evidence evaluator
is unvalidated against human labels. Ablations are reduced-sample. Latency is single-machine.
Full list: [`docs/pandarag/limitations.md`](docs/pandarag/limitations.md).

---

# Study 1 — Adaptive RAG on HotpotQA (frozen)

A controlled experiment comparing four QA systems on a frozen HotpotQA benchmark, testing whether
adaptive retrieval can cut cost while preserving quality. **This study is complete and its artifacts
are not modified by Study 2.** Every number is recomputed from the frozen result files in
[`docs/final_analysis.md`](docs/final_analysis.md).

**Status:** complete · 500 questions × 4 pipelines · 172 tests · `deepseek-flash`, temperature 0.

## Headline results

500 HotpotQA distractor-validation questions (400 bridge / 100 comparison). Cost from actual API usage.

| Pipeline | Retrieval | EM | F1 | Input tokens | Cost (USD) |
|---|---|---:|---:|---:|---:|
| No-RAG | — | 0.3520 | 0.4756 | 31,120 | 0.00600120 |
| Standard Dense RAG | dense, K=5 | **0.5100** | **0.6925** | 357,810 | 0.05547390 |
| Hybrid RAG | dense + BM25 (RRF), K=5 | 0.4900 | 0.6764 | 367,325 | 0.05096271 |
| **Adaptive RAG** | hybrid, K∈{3,5} | 0.4820 | 0.6598 | **263,640** | **0.02709842** |

Frozen retriever quality (document Recall@K, no LLM):

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| Dense (BGE-small + FAISS) | 0.4500 | 0.8500 | **0.9310** | 1.0000 |
| BM25 | 0.3720 | 0.6850 | 0.7920 | 1.0000 |
| Hybrid (RRF) | 0.4390 | 0.7940 | 0.8920 | 1.0000 |

> Recall@10 ≡ 1.0 because retrieval runs over each question's ten distractor paragraphs. This is
> exactly why Study 2 introduces a pooled corpus.

### What the results support

- **Retrieval is essential for bridge questions.** Adaptive beats No-RAG 106–41 (McNemar p ≈ 8e-08)
  and 100–19 on bridge questions (p ≈ 2e-14).
- **Adaptive matches Hybrid quality at roughly half the cost**: EM 0.4820 vs 0.4900 (p = 0.585;
  F1 CI includes zero), tokens −28.2%, cost −46.8%.
- **Standard dense RAG is the most accurate single pipeline** (EM 0.5100); its edge over Adaptive is
  small but real (F1 Δ −0.0327 [−0.0589, −0.0075]), concentrated in bridge questions.
- **Faithfulness differences are a retrieval effect, not a policy effect**: when both retrieved all
  gold documents the difference vanishes (Δ +0.008, CI includes 0).

### What the results do **not** support

- That K=3 is better than K=5 (χ² p = 0.34), or that the confidence signal predicts difficulty
  (Spearman ρ = +0.039, AUC 0.52).
- That Adaptive improves answer quality over fixed-K retrieval.
- Any claim about groundedness of No-RAG, which receives no context.

## The four systems

```
A. No-RAG        question ─▶ DeepSeek ─▶ answer
B. Standard RAG  question ─▶ dense retriever ─▶ top-K=5 ─▶ DeepSeek ─▶ answer
C. Hybrid RAG    question ─▶ dense + BM25 ─▶ RRF fusion ─▶ top-K=5 ─▶ DeepSeek ─▶ answer
D. Adaptive RAG  question ─▶ retrieve top-3 ─▶ confidence check ─▶ maybe top-5 ─▶ DeepSeek ─▶ answer
```

Frozen policy (validated by `check_frozen_policy()`):

```
confidence = 0.5·A + 0.5·M
A = 1 − ((d1 + b1) − 2) / 18            # top-1 rank agreement of dense/BM25, range 2..20
M = (s3 − s4) / W   if pool ≥ 4 else 0  # fused-score boundary margin, W = 2/61 − 2/70
final K = 3 if confidence ≥ 0.5 else 5
```

## Reproducing Study 1

```bash
# 1. frozen retrieval rankings (no LLM)
python -m src.retrieval_benchmark --method dense
python -m src.retrieval_benchmark --method bm25
python -m src.retrieval_benchmark --method hybrid

# 2. the four pipelines (one call per question, on-disk cache)
python -m src.pipelines.no_rag
python -m src.pipelines.standard_rag
python -m src.pipelines.hybrid_rag
python -m src.pipelines.adaptive_rag

# 3. secondary groundedness evaluation
python -m src.faithfulness_eval

# 4. analyses and reports (read-only)
python -m src.final_report && python -m src.cross_pipeline_analysis
```

## Study 1 documentation

| Document | Purpose |
|---|---|
| [`docs/final_analysis.md`](docs/final_analysis.md) | verified numbers, statistics, error taxonomy, figures |
| [`docs/research_analysis.md`](docs/research_analysis.md) | per-question wins/losses, bridge vs comparison, truncation math |
| [`docs/cross_pipeline_analysis.md`](docs/cross_pipeline_analysis.md) | unified comparison across the four pipelines |
| [`docs/faithfulness_report.md`](docs/faithfulness_report.md) | groundedness / hallucination evaluation |
| [`experiment.md`](experiment.md) | original experiment specification |

---

# Shared conventions

## Setup

Requires Python 3.11+ (developed on 3.14), CPU only.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your key in DEEPSEEK_API_KEY
```

`.env` is git-ignored; the API key is never logged, printed or stored in a result record.

## Evaluation

- **Answer quality** — deterministic Exact Match and token F1 (lowercase, punctuation- and
  article-stripped). No LLM decides correctness.
- **Retrieval quality** — document Recall@K, plus `R@stop` (the list the pipeline produced) and
  `R@ctx` (the documents the generator actually received). Undefined values are reported as —,
  never as zero.
- **Faithfulness / hallucination** — frozen LLM-judge protocol `faithfulness-judge/1.0` over the
  exact context each pipeline received; the judge never sees gold answers, supporting facts, EM/F1,
  question type or pipeline identity.
- **Cost** — from actual API usage at published prices, never estimated from characters.

## Testing

```bash
python -m pytest tests/ -q          # 300 tests (172 Study 1 + 128 Study 2), no network
```

Study 1's own suite can also be run with `python -m unittest discover -s tests`.

## Output artifacts and git

`results/` and `cache/` are git-ignored (they are large: hundreds of MB of per-question JSONL and
embedding indexes). The committed `docs/pandarag/*manifest*.json` files pin their identities
(SHA-256, row counts, fingerprints) so the code can be tied to the data without the bulk.

## Limitations

Study 1: single run per pipeline; EM/F1 are sensitive to verbosity; faithfulness is judged by an LLM;
retrieval compute is reused; the adaptive confidence is an ordinal heuristic, not a probability.
Study 2: pooled corpus rather than open-domain; 500 questions per dataset; model substitution;
unpublished prompts; unvalidated evidence evaluator; reduced-sample ablations; single-machine latency.
Details in [`docs/final_analysis.md`](docs/final_analysis.md) §9 and
[`docs/pandarag/limitations.md`](docs/pandarag/limitations.md).

## License

See [`LICENSE`](LICENSE).
