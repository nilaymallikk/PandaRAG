# Gap Analysis: Repo vs. SAC-RAG Base Paper (IEEE Access 2026, DOI 10.1109/ACCESS.2026.3718025)

Purpose: the professor asked for a base paper we can follow with "little tweaking". The base paper is
**SAC-RAG: Semantic Adaptive Context Compression for Retrieval-Augmented Generation**
(Zhang, Yu, Huo, Su — Shenyang Ligong University, IEEE Access).

Our target paper adds one more pipeline on top: **PandaRAG** (our own), compared against
**SAC-RAG** (re-implemented) and **AirRAG / ReSearch** (cited from the AirRAG paper, exactly as the
base paper does).

This document: what the base paper does, what we already have, what must change.

---

## 1. Base paper protocol (must be reproduced "the same way")

### 1.1 Datasets (Table 1)

| | HotpotQA | 2WikiMultiHopQA | MuSiQue |
|---|---|---|---|
| Questions | 1,000 | 1,000 | 1,000 |
| Type dist. (Fact/Def/Reasoning) | 958/20/22 | 796/5/199 | 732/28/240 |
| Hops | 1–2 | 2 | 2–4 |

The type distribution is *produced by their own LLM classifier*, not shipped with the datasets.

### 1.2 Metrics ("the same way")

| Metric | Definition in paper |
|---|---|
| EM | Normalized exact match: Unicode **NFKD**, lowercasing, punctuation + article removal (DPR `normalize_answer`) |
| F1 | Token-level harmonic mean of precision/recall on normalized tokens |
| Recall@10 | `\|D_ret ∩ D_GT\| / \|D_GT\|` — retrieved top-10 **document** set vs gold supporting documents |
| Avg Tokens | Mean token consumption **per query, generation only** (LLMLingua-style efficiency metric) |

### 1.3 Four progressive ablation configs (Table 3)

| | Naïve | Compress | Adaptive | SAC-RAG (full) |
|---|---|---|---|---|
| Classifier | ✗ | ✗ | ✓ | ✓ |
| Type-specific retrieval | ✗ | ✗ | ✓ | ✓ |
| Compression | ✗ | ✓ | ✗ | ✓ |

Hypotheses tested: **H1** compression vs naïve (tokens down, quality held), **H2** adaptive vs
naïve (quality up), **H3** SAC-RAG vs Compress + SAC-RAG vs Adaptive (synergy).

### 1.4 Retrieval stack

- Corpus: **shared knowledge base** (Wikipedia chunks), Milvus collection, 1536-d, cosine — *not*
  the per-question HotpotQA distractor paragraphs.
- Fact path: BM25 top-10 (rank_bm25 + spaCy tokens) + dense top-10 → merge/dedupe → **bge-reranker-base**
  cross-encoder → top-10.
- Definition path: dense only, top-8.
- Reasoning path: LLM sub-question decomposition → each sub-question dense top-10 → merge → BGE rerank → top-10.
- Fusion visible in paper eq. (2): `S_Fact = λ·BM25 + (1−λ)·cos(e_q, e_d)`, k1=1.5, b=0.75.

### 1.5 LLMs and generation

- Classifier: LLM, structured few-shot prompt → {Fact, Definition, Reasoning}.
- Compression: unified LLM prompt extracting "evidential snippets" from top-k docs (compressor **qwen-flash**).
- Generator: **gpt-5-mini**. Temperature **0**, all groups share the same LLM configuration.

### 1.6 Analyses reported

- Table 2: system-level comparison vs Vanilla RAG, IterDRAG, ReSearch, AirRAG — **official numbers
  cited from the AirRAG paper, not re-run** (footnote: different backbones → explicitly "not a
  strictly controlled variable experiment"). SAC-RAG/AirRAG only methods reporting tokens + R@10.
- Table 4: H1/H2/H3 ΔEM and ΔRecall@10 per dataset.
- Table 5: per-question-type EM / F1 / Recall@10 / Avg Tokens (Adaptive vs SAC-RAG) — 3 datasets.
- Table 6: compression degradation analysis — per type: total, degraded, ratio, avg F1 drop.
- Fig. 2/3: grouped bars EM, F1, Accuracy, Recall@10 per dataset.
- Fig. 4: Avg tokens per config per dataset. Fig. 5: Pareto Recall@10 vs tokens.
- Section V-E: time efficiency (s/instance): Naïve ~9.6, Adaptive ~21.5, SAC-RAG ~27.8.
- Section VI-B: worked failure examples per question type (definition entity confusion, fact detail
  loss, reasoning noise filtering).
- Limitations section: classifier granularity, imbalanced subtypes (Definition n=20/5), model
  dependence, English-only data.

### 1.7 Headline numbers to beat / match

- SAC-RAG EM: HotpotQA 0.595, 2Wiki 0.590, MuSiQue 0.271; F1: 0.740 / 0.664 / 0.403.
- Tokens: 1,087 / 1,264 / 1,918 (−55%, −58%, −38% vs Adaptive RAG).
- Recall@10: 0.881 / 0.704 / 0.479.
- On **Reasoning** questions compression *improves* EM (HotpotQA 0.455 → 0.577, +26.8%).

---

## 2. Where the repo stands today

### 2.1 What exists (and is good)

- Fully reproducible, deterministic harness: frozen dataset hash, frozen retrieval artifacts,
  on-disk LLM cache, retry + error log, 166 offline tests, read-only analyses.
- 4 pipelines on HotpotQA-500 distractor/validation: No-RAG, Standard Dense RAG (K=5),
  Hybrid RRF RAG (K=5), Adaptive RAG (RRF hybrid + pre-registered A/M confidence → K∈{3,5}).
- Retrieval: BGE-small-en-v1.5 + FAISS (dense, paragraph + sentence), rank_bm25, RRF fusion.
- Evaluation: deterministic EM/F1 (HotpotQA-style), doc/sentence Recall@K, per-question
  input/output/reasoning tokens + USD cost from real API usage, latency.
- Extra beyond base paper: LLM-judge faithfulness/hallucination on the exact retrieved context.

### 2.2 Headline results so far (HotpotQA-500, DeepSeek flash, temp 0)

| Pipeline | EM | F1 | Input tok | Cost (USD) |
|---|---|---|---|---|
| No-RAG | 0.3520 | 0.4756 | 31,120 | 0.00600 |
| Standard Dense RAG | **0.5100** | **0.6925** | 357,810 | 0.05547 |
| Hybrid RRF RAG | 0.4900 | 0.6764 | 367,325 | 0.05096 |
| Adaptive (conf-K) | 0.4820 | 0.6598 | **263,640** | **0.02710** |

### 2.3 The three blocking gaps

1. **Recall@10 is degenerate.** Retrieval ran on the per-question distractor corpus (10 paragraphs),
   so every retriever scores Recall@10 = 1.000 (README). The base paper's Recall@10 is the key
   discriminative metric (0.704–0.881 on a shared corpus). **Until we index a shared corpus, we
   cannot produce the paper's central retrieval-quality column.**
2. **Name collision.** Our current "Adaptive RAG" (confidence-based K∈{3,5} over RRF) is *not* the
   base paper's "Adaptive RAG" (type-adaptive retrieval without compression), nor is it Adaptive-RAG [3].
   It is actually the seed of **PandaRAG stage 3** (evidence confidence → dynamic budget).
   Left as-is, every table will be misread.
3. **Missing components:** nothing of the classification → type-specific retrieval → compression
   spine exists yet — no question classifier, no sub-question decomposition, no cross-encoder
   reranker, no compression module, no 2Wiki/MuSiQue.

---

## 3. Component-by-component comparison

| Base-paper component | Repo now | Action |
|---|---|---|
| LLM classifier Fact/Def/Reasoning | absent | build (`src/classifier.py`), cached, JSON few-shot prompt |
| Fact path: hybrid + BGE rerank top-10 | hybrid RRF only, no rerank | add `bge-reranker-base` CrossEncoder (already-installed `sentence-transformers`) |
| Definition path: dense top-8 | absent | trivial once classifier exists |
| Reasoning path: decompose → parallel dense → rerank | absent | build decompose+merge; needs shared corpus to be meaningful |
| Unified LLM compression | absent | build `src/compression.py` with documented prompt (paper's exact prompt is unpublished) |
| Shared knowledge base (Milvus + chunks) | per-question distractor corpus | pool all unique paragraphs of each dataset into one shared index (FAISS + BM25); state as reduced-scale deviation |
| EM/F1 normalization | lowercase + punct/article strip, **no NFKD** | add `unicodedata.normalize("NFKD", …)` to `answer_eval.normalize_answer` |
| Recall@10 | degenerate =1.0 | recompute on shared corpus |
| Avg Tokens (generation only) | input/output tokens recorded, not reported this way | report as paper does (and keep cost as extra) |
| 1,000 q × 3 datasets | 500 q × 1 dataset | extend loaders + rerun |
| Table 2: Vanilla/IterDRAG/ReSearch/AirRAG | absent | **cite official numbers from AirRAG paper** exactly like the base paper — do NOT reimplement |
| Tables 4/5/6, Fig. 2–5, time efficiency | partial (cost-quality, error taxonomy, faithfulness) | extend analysis scripts to paper's exact tables |
| Temperature 0, same LLM for all groups | already true (DeepSeek flash, thinking off) | keep; document deviation from gpt-5-mini + qwen-flash |

---

## 4. Change plan (ordered)

### Phase 0 — freeze what is done
- Tag the current HotpotQA-500 confidence-K result as **Stage 1 pilot**. It stays as the
  pre-registered evidence for PandaRAG's confidence signal. Rename its label to
  **"Confidence-K RAG (pilot)"** everywhere (README, docs, figures) so "Adaptive RAG" means the
  base paper's group from now on.

### Phase 1 — protocol parity (cheap, no new science)
1. `answer_eval.normalize_answer`: add NFKD. Re-run the offline tests; EM/F1 shifts are tiny but
   the metric definition now matches the paper.
2. Dataset scaling: loaders for **2WikiMultiHopQA** and **MuSiQue** (HF `datasets` is already a
   dependency); freeze 1,000-question files with SHA-256 like `hotpotqa_500.json` (rename to
   `hotpotqa_1000.json` or keep both; paper uses 1,000).
3. Shared-corpus index per dataset: dedupe paragraphs by title across the dataset → one FAISS +
   one BM25 index per dataset → recompute Recall@1/3/5/10. Expect Recall@10 to drop into a
   realistic range; if it does not, the corpus is still too small and should be widened.
4. Token metric: add `avg_generation_tokens_per_query` to every pipeline's meta so Table 5 maps 1:1.

### Phase 2 — build the SAC-RAG spine
5. `src/classifier.py`: LLM few-shot Fact/Definition/Reasoning classifier, disk-cached, one call
   per question. Because it is cached and frozen, the class counts will be identical across all
   pipelines (the base paper's counts drift between groups — Table 1 vs Table 5, e.g. Fact 958 vs
   954 — ours will be cleaner; note this as a small improvement).
6. `src/reranker.py`: `BAAI/bge-reranker-base` CrossEncoder over merged candidates.
7. `src/pipelines/` new groups, following Table 3 exactly:
   - **Naive RAG** = existing Standard Dense RAG (top-k dense).
   - **Compress RAG** = Naive + compression module.
   - **Adaptive RAG** = classifier + type-specific retrieval (Fact hybrid+rerank / Def dense-8 /
     Reasoning decompose) with *no* compression.
   - **SAC-RAG** = Adaptive + compression.
8. `src/compression.py`: unified evidence-snippet extraction prompt. The paper does not publish the
   prompt — write ours, document it verbatim in the paper appendix and call it a deviation.

### Phase 3 — PandaRAG (our contribution)
9. Extend the frozen confidence policy from Stage 1 to the diagram's three bands:
   high confidence → K=3, medium → K=5, low → K=8/10, using the PandaRAG stage-3 signals
   (retrieval score distribution, dense–BM25 agreement, top-doc relevance, evidence coverage).
   Apply it *after* SAC-RAG's type-specific retrieval candidate pool (top-15/20), not after RRF.
10. Run PandaRAG twice: with and without the compression stage (diagram marks compression
    "optional") → this is a free extra ablation the base paper cannot show.
11. Primary claim to test: PandaRAG sits on/below the SAC-RAG Pareto point — same EM band, fewer
    tokens (and one fewer LLM call than SAC-RAG's fixed-k compression, because confidence is
    computed from retrieval signals only).

### Phase 4 — analysis parity + system table
12. Table 4 equivalent: H1/H2/H3 Δ per dataset for our 4 groups (script already has the
    hypothesis machinery in `final_analysis.py` / `final_report.py`).
13. Table 5 equivalent: per-type EM/F1/R@10/AvgTokens for Adaptive vs SAC-RAG vs PandaRAG.
14. Table 6 equivalent: compression degradation per type (degraded n, ratio, avg F1 drop) — the
    repo's error taxonomy is the seed.
15. Fig. 2–5 equivalents incl. **Pareto (Recall@10 vs Avg Tokens)**; add time efficiency (s/query).
16. System table: Vanilla RAG / IterDRAG / ReSearch / AirRAG rows cited from AirRAG [36] +
    SAC-RAG and PandaRAG from our runs, with the same footnote the base paper uses
    (different backbones → system-level, not controlled).

### Phase 5 — cost control (do not skip)
- LLM calls/question go from 1 (today) to ~3–5 (classifier + decomposition + compressor + generator).
  3 datasets × 1,000 q × 5 configs ≈ 15,000 generator calls plus extras.
- Mitigations already in repo: disk cache (replays free), per-stage `--limit` smoke runs, frozen
  retrieval artifacts, error log. Add a per-run token/cost budget guard before the full sweep and
  abort if projected cost exceeds the allowance.
- Suggested order: HotpotQA-1000 complete → verify all tables → then 2Wiki → then MuSiQue.

---

## 5. Deviations to declare in the paper (keep this list)

1. Shared corpus = pooled unique paragraphs of the dataset (reduced-scale stand-in for full
   Wikipedia + Milvus). Must be stated; Recall@10 is comparable only in spirit, not in absolute value.
2. FAISS instead of Milvus; BGE-small-en-v1.5 instead of text-embedding-3-small (1536-d).
3. One LLM (DeepSeek flash) for classifier, compressor and generator, vs. gpt-5-mini + qwen-flash.
4. Compression prompt is ours (base paper's is unpublished).
5. AirRAG / ReSearch / IterDRAG / Vanilla numbers are cited, not reproduced — identical to the
   base paper's own Table 2 method.
6. Classifier output is cached once per question, so class counts are stable across pipelines
   (base paper's drift).
7. Extra metrics retained from Stage 1: USD cost, faithfulness/hallucination, McNemar/bootstrap CIs —
   strengthens the paper beyond the base paper's four metrics.

## 6. Bottom line

The repo is a strong, rigorous *stage-1 pilot* of exactly one idea (confidence-based K), with
infrastructure the base paper lacks. It currently **cannot** produce SAC-RAG's tables because the
classification → type-retrieval → compression spine and the shared-corpus Recall@10 do not exist
yet, and because the name "Adaptive RAG" is taken by the wrong thing. Phases 1–2 make us
comparable to the base paper; Phases 3–4 turn PandaRAG into the incremental contribution the
professor asked for; the system table (AirRAG/ReSearch cited) completes the comparison story.
