# AGENTS.md

## Scope: two projects live in this repository

This file now governs **two** scopes. Read the one that applies before acting.

| Scope | Lives in | Status | Governed by |
|---|---|---|---|
| **A. Legacy experiment** (4 pipelines, HotpotQA-500) | `src/pipelines/{no_rag,standard_rag,hybrid_rag,adaptive_rag}.py`, `results/*.jsonl`, `results/retrieval/`, `data/hotpotqa_500.json` | **FROZEN — complete, published internally** | The legacy sections below (unchanged) |
| **B. PandaRAG project** (SAC-RAG reimplementation + proposed method) | `src/datasets/`, `src/retrieval/`, `src/pipelines/sac_rag/`, `src/pipelines/pandarag/`, `configs/`, `docs/pandarag/`, `results/sac_rag/`, `results/pandarag/` | **Active** | §"PandaRAG project" below + `docs/pandarag/experiment_protocol.md` |

**The legacy sections below were written for Scope A only.** Where they conflict with Scope B
(most importantly "do NOT download another dataset" and "exactly these 500 questions"), they apply
to Scope A and do **not** constrain Scope B. Scope A artifacts must never be modified to serve
Scope B.

---

## Project (Scope A — legacy, frozen)

This is a research project investigating:

Query-Complexity-Aware Adaptive Retrieval-Augmented Generation
for Multi-Hop Question Answering.

The project compares four independent QA systems:

1. No RAG
2. Standard Dense RAG
3. Hybrid RAG
4. Adaptive RAG

The goal is to determine whether adaptive retrieval can maintain
answer quality while reducing retrieval and computational cost.

---

## Dataset

Dataset:

HotpotQA

Split:

Distractor validation

Fixed dataset:

data/hotpotqa_500.json

The dataset has already been manually selected.

IMPORTANT:

- Do NOT download another dataset.
- Do NOT resample the dataset.
- Do NOT shuffle the dataset.
- Do NOT modify gold answers.
- Do NOT add or remove questions.
- Treat data/hotpotqa_500.json as immutable experiment input.

The experiment must use exactly these 500 questions.

---

## LLM

Provider:

DeepSeek API

Model:

DeepSeek-V4.1-Flash

Base URL:

https://api.deepseek.com

The API key must come from the environment:

DEEPSEEK_API_KEY

Never hard-code or print the API key.

Do not use OpenRouter.

Do not use another LLM unless explicitly instructed.

All four pipelines must use the same LLM model.

---

## Local Retrieval

Retrieval must run locally.

Dense retrieval:

- sentence-transformers
- BGE-small-en-v1.5
- FAISS

Lexical retrieval:

- BM25

Hybrid retrieval:

- Dense retrieval + BM25
- simple rank fusion

Do not introduce a hosted vector database.

Do not introduce LangChain or LlamaIndex unless explicitly requested.

---

# Four Pipelines

## 1. No RAG

Question
→ DeepSeek
→ Answer

No retrieval.

No context from the HotpotQA documents is supplied.

---

## 2. Standard RAG

Question
→ Dense Retriever
→ Fixed K
→ Context
→ DeepSeek
→ Answer

Initial experimental K:

K = 5

K must remain fixed for this pipeline.

---

## 3. Hybrid RAG

Question
→ Dense Retrieval + BM25
→ Rank Fusion
→ Fixed K
→ DeepSeek
→ Answer

Initial experimental K:

K = 5

K must remain fixed for this pipeline.

---

## 4. Adaptive RAG

Question
→ Query Complexity Analysis
→ Initial Retrieval Budget
→ Retrieval
→ Evidence/Confidence Check
→ Possibly retrieve more
→ DeepSeek
→ Answer

The adaptive pipeline is the proposed research method.

Do not make it unnecessarily complicated.

Start with a transparent rule-based policy.

The policy may use:

- query complexity
- retrieval score
- evidence coverage
- supporting evidence
- confidence

The adaptive system must record why additional retrieval was performed.

---

# Important Experimental Rules

1. Keep the four pipelines separate.

2. Use the same 500 questions for every pipeline.

3. Use the same DeepSeek model for every pipeline.

4. Keep prompts as consistent as possible across pipelines.

5. Do not tune one pipeline unfairly after seeing its results.

6. Do not modify evaluation rules after seeing results.

7. Do not fabricate or manually edit results.

8. Failed API requests must be logged, not silently removed.

9. Preserve raw results.

10. Cache API responses where practical.

11. Never expose API keys in logs.

12. Keep the implementation simple.

13. Avoid unnecessary abstractions and frameworks.

14. Do not create databases or web services.

15. Do not introduce Docker unless explicitly requested.

---

# Evaluation

Every pipeline must produce a compatible result record.

Record at minimum:

- question_id
- question
- gold_answer
- predicted_answer
- retrieval_k
- retrieved_documents
- retrieval scores
- retrieval recall
- exact match
- F1
- faithfulness/groundedness
- hallucination indicator
- input tokens
- output tokens
- LLM calls
- latency
- API cost
- error status

---

# Retrieval Evaluation

HotpotQA supporting facts should be used to evaluate retrieval.

Measure:

- Recall@1
- Recall@3
- Recall@5
- Recall@10 where applicable
- supporting-fact retrieval

For No RAG, retrieval metrics are not applicable.

Do not report them as zero.

Use null/NA.

---

# Answer Evaluation

Use the HotpotQA gold answer.

Calculate:

- Exact Match
- F1

Do not use an LLM to decide basic answer correctness when deterministic
evaluation is possible.

---

# Faithfulness / Hallucination

The generated answer should be evaluated against the retrieved context.

Distinguish:

1. Answer correctness
2. Evidence retrieval
3. Groundedness/faithfulness

Do not automatically equate an incorrect answer with hallucination.

A hallucination/unsupported claim means the answer contains information
that is not supported by the supplied retrieval context.

---

# Cost

Calculate API cost from actual token usage returned by DeepSeek.

Record:

- input tokens
- output tokens
- cached input tokens if available
- total cost

Do not estimate cost from character count.

Use actual API usage whenever available.

---

# Reproducibility

Record:

- model
- model parameters
- prompts
- retrieval method
- K
- embedding model
- dataset filename
- dataset size
- random seeds if used
- timestamp
- token usage
- latency
- cost

Do not change the 500-question dataset.

---

# Development Order

Implement in this exact order:

1. Verify dataset structure
2. Verify DeepSeek API connection
3. Implement shared LLM client
4. Implement dense retrieval
5. Implement BM25
6. Implement retrieval evaluation
7. Implement No-RAG
8. Implement Standard RAG
9. Implement Hybrid RAG
10. Implement Adaptive RAG
11. Implement unified evaluation
12. Run a small smoke test
13. Run the full 500-question experiment
14. Analyze results

Do not implement the entire project in one step.

After each major stage, run a small test.

---

# Research Integrity

The purpose is to conduct a reproducible experiment.

Do not optimize the system specifically to obtain a desired result.

If Adaptive RAG performs worse, record the result honestly.

Do not change the evaluation methodology after seeing results.

When uncertain about an experimental decision, explain the issue
before making a major change.

---

# PandaRAG project (Scope B — active)

Full specification and rationale: `docs/pandarag/implementation_plan.md`.
Parameter provenance and deviations: `docs/pandarag/sac_rag_reproduction_protocol.md`.
Repository state: `docs/pandarag/repository_audit.md`.

## Systems under study

`P0` No-RAG · `P1` Standard RAG · `P2` Hybrid RAG · `P3` Adaptive RAG (all four legacy, frozen)
`P1c`/`P2c`/`P3c` corpus-protocol twins · `P1n` paper-specified naïve baseline ·
`P4` SAC-RAG Adaptive · `P5` Full SAC-RAG · `P6` PandaRAG · `P7` PandaRAG + compression.

## Dataset scope

Three datasets, each with its own isolated corpus and split:

- HotpotQA — `data/hotpotqa_500.json` (validation/distractor, hard-only, 2-hop)
- 2WikiMultiHopQA — `data/2wiki_500.json` (dev, 2-hop filtered)
- MuSiQue — `data/musique_500.json` (validation, answerable only)

These files are already present, committed and fingerprinted in `data/dataset_manifest.json`.
**No further dataset downloads without explicit authorization.** Never resample, reshuffle or edit
these files, and never mix corpora or splits between datasets.

## Corpus protocol (decision D2)

Pooled candidate corpus: the union of every paragraph shipped with each question in the dataset
file, deduplicated by content. Document ids are **content-derived** (`sha1(title + text)`), never
positional. Each corpus artifact carries the dataset fingerprint, the split, the document count and
its own sha256. This corpus is described as a *pooled candidate corpus*; it is **not** open-domain
Wikipedia and its Recall@10 must never be compared with the base paper's.

## Non-negotiable integrity rules (Scope B)

1. Gold answers, gold supporting documents and gold sentences must never reach runtime retrieval,
   planning, evaluation-of-support or control decisions. Gold is used offline for metrics only.
2. Every result record carries dataset + split + pipeline + corpus fingerprint + resolved config
   fingerprint.
3. Thresholds, budgets and prompts are frozen before final evaluation and never tuned on the data
   being reported.
4. Auxiliary LLM cost (classifier, planner, evidence evaluator, compressor, judge) is recorded
   separately from generation cost. Headline "Avg Tokens" stays generation-only, with the auxiliary
   figure always reported adjacent to it.
5. Outputs are additive: `results/sac_rag/`, `results/pandarag/`. Never overwrite Scope A artifacts.
6. A cached/replayed run is never reported as a live run.
7. Failed API calls are logged, never silently dropped, never replaced with estimates.
8. No gold-label tuning, no post-hoc metric redefinition, no fabricated numbers.
9. No framework, hosted vector store, web service or Docker introduced for Scope B.

## Terminological rule

The base paper's corpus, classifier prompt, decomposition prompt and compression prompt are **not
published**. Therefore this work is a **best-effort architectural reimplementation**, never an
"exact reproduction". Every deviation is recorded in
`docs/pandarag/sac_rag_reproduction_protocol.md` (deviation ledger DV1–DV12) and must be repeated in
any write-up. `P1n` is called a paper-specified baseline **only** where the paper states that
configuration explicitly.

## Development order (Scope B)

Phase 0 audit (done) → Phase 1 datasets + corpus + indexes (local only, no LLM calls) → Phase 2
SAC-RAG Adaptive → Phase 3 Full SAC-RAG (compression) → Phase 4 planner + evidence evaluator →
Phase 5 gap controller → Phase 6 context builder + generator → Phase 7 pilots (≈50 questions,
engineering only) → Phase 8 freeze + full evaluation → Phase 9 paper-ready analysis.

A phase is not complete until its tests are written and passing. Completed phases must be reviewed
before the next phase starts.