# Repository Audit — PandaRAG Phase 0

**Audit date:** 2026-10-26
**Repository:** `github.com/nilaymallikk/PandaRAG` (`/home/nilaymallik/adaptive-rag-research`)
**Commit audited:** `fe8ffd0` ("Support the OpenCode Go gateway model id and session header")
**Purpose:** establish, before any new code is written, what exists, what is frozen, what is
reusable, what is missing, and which hidden assumptions must break to implement SAC-RAG Adaptive,
Full SAC-RAG and PandaRAG.

No API calls were made during this audit. No tracked file was modified. This document is the only
deliverable of Phase 0 alongside `sac_rag_reproduction_protocol.md` and `implementation_plan.md`.

---

## 1. Summary verdict

The repository is a **complete, honest, single-dataset (HotpotQA-500) four-pipeline study** with
unusually strong reproducibility hygiene: frozen policy guards, artifact fingerprint verification,
disk-cached LLM responses, actual-usage cost accounting, and 172 passing tests.

It is **not** a base for SAC-RAG reproduction as it stands, for three structural reasons:

1. **Retrieval is per-question.** Every index is built over the 10 distractor paragraphs shipped
   with each question. Recall@10 is therefore structurally 1.000, and no corpus-level retrieval
   metric can be produced. SAC-RAG reports Recall@10 as its headline retrieval metric, computed
   over a shared document collection.
2. **No retrieval-layer abstraction.** `src/retrieval.py` exposes per-question FAISS/BM25 ranking
   helpers, not a corpus index. There is no chunking, no persistent index, no reranker, no
   query-level cache.
3. **The dataset layer is HotpotQA-shaped.** `HotpotExample` is the single canonical type; the
   loader, the validation rules, the result schema and `config.DATASET_FILE` all assume HotpotQA.

The good news: the **generation, cost, cache and evaluation layers are directly reusable**, and the
two new datasets plus a semantic-type classifier already exist in the tree (committed in `ff78730`).
Roughly 40% of the SAC-RAG spine can be built on existing primitives.

---

## 2. Repository inventory

### 2.1 Tracked modules

| Path | LOC | Role | Verdict for the new work |
|---|---|---|---|
| `src/config.py` | 351 | All constants, env-driven, pricing table, `describe()` metadata | **Reuse + extend.** Single source of truth; keep the pattern. |
| `src/llm.py` | 497 | DeepSeek client: disk cache, retries, error log, actual-usage cost | **Reuse as-is.** Also serves classifier/planner/compressor/judge calls. |
| `src/dataset.py` | 330 | HotpotQA loader, validation, fingerprint, sentence ids | **Extend.** Needs generalization (§5.1). |
| `src/retrieval.py` | 522 | Per-question dense (FAISS) + BM25 + RRF | **Partially reuse.** Ranking math yes, corpus model no. |
| `src/retrieval_benchmark.py` | 247 | Runs the three retrievers over 500 questions → artifacts | Reuse shape; must be re-pointed at a corpus index. |
| `src/evaluation.py` | 128 | `recall_at_k`, supporting-fact/sentence metrics | **Reuse as-is.** Doc-level Recall@K already matches the paper's formula. |
| `src/answer_eval.py` | 89 | EM / token-F1 with normalization | **Reuse; verify on new datasets** (§7.2). |
| `src/faithfulness_eval.py` | 666 | Frozen DeepSeek judge → claim-level grounding | **Reuse as-is.** Extends grounding metrics for free. |
| `src/classifier.py` | 272 | Fact/Definition/Reasoning classifier v1.3 + validation | **Reuse, re-home** under `src/pipelines/sac_rag/` (§7.3). |
| `src/download_datasets.py` | 419 | 2Wiki + MuSiQue downloaders, provenance manifest | **Reuse as-is.** |
| `src/pipelines/no_rag.py` | 326 | P0 closed-book | Frozen, historical. |
| `src/pipelines/standard_rag.py` | 418 | P1 dense K=5 | Frozen, historical. |
| `src/pipelines/hybrid_rag.py` | 436 | P2 dense+BM25 RRF K=5 | Frozen, historical. |
| `src/pipelines/adaptive_rag.py` | 537 | P3 frozen confidence policy | **Frozen. Do not touch.** |
| `src/cross_pipeline_analysis.py` | 406 | Cross-pipeline comparison tables | Reuse patterns. |
| `src/final_analysis.py` | 139 | Aggregation → report tables | Reuse patterns. |
| `src/final_report.py` | 548 | Report + figures driver | Reuse patterns. |
| `src/svg_charts.py` | 199 | Hand-rolled SVG charts | Reuse; add Pareto plot. |
| `tests/*.py` (10 files) | 2,000 | 172 tests, all passing | **Must stay green** (§2.4). |

### 2.2 Artifacts and their fingerprints

| Artifact | Rows/Size | Fingerprint |
|---|---|---|
| `data/hotpotqa_500.json` | 500 | sha256 `956405ce…acfd048` (asserted in code) |
| `data/2wiki_500.json` | 500 | sha256 `658de4da…974b4e0` (manifest) |
| `data/musique_500.json` | 500 | sha256 `79f5aefa…8eb5a5` (manifest) |
| `data/dataset_manifest.json` | — | per-file sha, filters, hop/type histograms |
| `data/semantic_types_{hotpotqa,2wiki,musique}.json` | 500 each | `sac-rag-classifier/1.3`, model, cost |
| `results/retrieval/{dense,bm25,hybrid}_retrieval.jsonl` | 500 each | dataset sha + RRF k=60/depth=10 asserted |
| `results/{no_rag,standard_rag,hybrid_rag,adaptive_rag}.jsonl` | 500 each, 0 failed | run meta alongside each |
| `results/analysis/faithfulness.jsonl` | 500 | judge protocol `faithfulness-judge/1.0` |

`adaptive_rag.py` **verifies** these at runtime (`verify_retrieval_artifact`) and **refuses to
import** retrieval/other-pipeline modules (`_PROTOCOL_FORBIDDEN_MODULES`). Any new pipeline must be
a separate module — never an edit to that file.

### 2.3 Existing pipeline definitions (as built, not as described)

| ID | Pipeline | Retrieval | K | LLM calls/q |
|---|---|---|---|---|
| P0 | No RAG | none | — | 1 |
| P1 | Standard RAG | dense over 10 per-question paragraphs | 5 fixed | 1 |
| P2 | Hybrid RAG | dense + BM25 → RRF | 5 fixed | 1 |
| P3 | Adaptive RAG | frozen hybrid artifact, **no live retrieval** | 3 or 5 | 1 |

**P3's frozen policy** (asserted by `check_frozen_policy()`, all values raise on drift):

```
A = 1 − ((d1 + b1) − 2) / 18            # source agreement, d1/b1 = top-1 dense/BM25 ranks
M = (s3 − s4) / W,  W = 2/61 − 2/70     # RRF boundary margin, 0.0 if pool < 4
confidence = 0.5·A + 0.5·M              # ordinal heuristic, explicitly NOT a probability
K = 3 if confidence ≥ 0.5 else K = 5
```

P3 is a **post-hoc decision over a stored artifact**. It never retrieves, never classifies with an
LLM, and records `adaptive_signals` / `adaptive_reason` for every question.

### 2.4 Test coverage

172 tests, 10 files, ~38 s. Notable: `test_retrieval.py` (535 lines), `test_adaptive_rag.py` (245,
includes protocol guards), `test_faithfulness_eval.py` (218), `test_no_rag.py` (292).
There are **no** tests for: dataset adapters beyond HotpotQA, corpus construction, reranking,
compression, or cross-dataset evaluation. Everything the new work adds needs new tests, and all 172
existing tests must keep passing (they encode the frozen experiment's invariants).

### 2.5 Environment (currently broken — must fix before any run)

```
.venv/pyvenv.cfg       → version_info = 3.14.7, uv-managed
.venv/bin/python       → /usr/bin/python   (now resolves to 3.15.2)
.venv/lib/python3.14/site-packages   → 191 packages installed here
```

`python -m src.dataset` fails with `ModuleNotFoundError: No module named 'dotenv'` because the
interpreter moved. Verified workaround:
`PYTHONPATH=.venv/lib/python3.14/site-packages python3.14 -s -m pytest tests/ -q` → **172 passed**.
Permanent fix: `ln -sf /usr/bin/python3.14 .venv/bin/python`.

Installed and relevant: torch 2.14.0, transformers 5.17.0, sentence-transformers 6.1.0, faiss-cpu,
rank_bm25 0.2.2, numpy 2.5.3, openai 3.16.2, pytest 9.1.1.
**Not installed: PyYAML.** The target layout's `configs/*.yaml` therefore implies a new dependency
(decision D9).

---

## 3. What is frozen (and the mechanism that protects it)

| Frozen thing | Protection | Consequence for new work |
|---|---|---|
| `data/hotpotqa_500.json` | sha asserted in `dataset.py`, `adaptive_rag.py`, retrieval metas | Reuse as-is; never rewrite |
| P3 policy constants | `check_frozen_policy()` raises on any drift | New pipelines live in `src/pipelines/sac_rag/`, `src/pipelines/pandarag/` |
| P3's inputs | `_PROTOCOL_FORBIDDEN_MODULES` import guard + artifact verification | P3 stays an artifact replay; it cannot host corpus retrieval |
| `results/*.jsonl` (P0–P3) | AGENTS.md rules 6–9; tests | New runs write to `results/sac_rag/`, `results/pandarag/` |
| Prompts | `config.GENERATION_SYSTEM_PROMPT` + per-pipeline frozen templates | New pipelines may add prompts; **they may not alter P0–P3's** |
| Evaluation rules | `answer_eval.py`, `evaluation.py` | Extend, don't redefine; new metrics are additive |

AGENTS.md also forbids, project-wide: downloading/resampling dataset inputs, hosted vector DBs,
LangChain/LlamaIndex, using a different LLM per pipeline within a comparison, tuning after seeing
results, fabricating/editing results.

**Tension to resolve explicitly:** AGENTS.md says "Do NOT download another dataset", while the new
specification requires 2Wiki and MuSiQue. The datasets were downloaded and committed on 2026-10-07
under `src/download_datasets.py` with a provenance manifest. The audit treats the 2Wiki/MuSiQue
addition as **already authorized and closed**, and records that `AGENTS.md` must be amended (or
superseded by `docs/pandarag/experiment_protocol.md`) for the new scope. See decision **D1**.

---

## 4. Existing generation and accounting primitives (reusable, with exact behaviour)

**Prompt (frozen for P0–P3).** System: `config.GENERATION_SYSTEM_PROMPT` — "answer directly, use
supplied context, return only the final answer". User template, from `standard_rag.build_prompt`:

```
Question: {question}

Context:
[Document 1]
Title: {title}
{paragraph text}

[Document 2]
...

Answer the question using the supplied context.
Answer:
```

This is **not** the base paper's prompt (which is unpublished). It must not be changed for P0–P3.
For P4–P7 the prompt must be identical *within* a comparison and is recorded per run.

**LLM client (`src/llm.py`).** `DeepSeekClient.generate(prompt, system=…)` → `LLMResponse` with
`prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `cached_input_tokens`, `cost_usd`,
`from_cache`, `finish_reason`. Responses cached on disk keyed by request; `from_cache=True` replays
cost nothing. Peak/off-peak pricing table in `config.PRICING_USD_PER_1M_TOKENS`. Errors are logged
to `results/logs/llm_errors.jsonl`, never dropped. **This client is exactly what the classifier,
decomposer, compressor, planner, evidence judge and generator all need** — one shared surface, one
cost ledger.

**Faithfulness judge (`src/faithfulness_eval.py`).** Frozen protocol `faithfulness-judge/1.0`:
DeepSeek judge, thinking disabled, temperature 0, `max_tokens=768`, sees only question + exact
retrieved context block + answer; decomposes into atomic claims, labels each; scores computed in
code. Gives P4–P7 grounding metrics with zero new design, and its claim-level labels are a natural
second signal for the evidence evaluator (decision D8).

---

## 5. Structural gaps (what must break)

### 5.1 Dataset layer is HotpotQA-shaped

```python
HotpotExample.from_dict(record)      # the only canonical type
config.DATASET_FILE = DATA_DIR / "hotpotqa_500.json"
config.EXPECTED_DATASET_SIZE = 500
validate_records()                   # HotpotQA required/optional fields
```

No dataset registry, no `split` concept, no per-dataset normalization, no paragraph-vs-sentence
evidence abstraction. MuSiQue's paragraph-level gold (`sent_id = -1`) is handled by a special case
added in `ff78730`; that special case would be duplicated in every new consumer unless evidence is
abstracted (decision D5).

### 5.2 Retrieval layer is per-question (the central gap)

`src/retrieval.py` docstring, verbatim: *"candidate corpus = the distractor paragraphs shipped with
each question … one index per question and granularity"*. Consequences:

- Recall@10 = 1.000 for every retriever by construction (`docs/base_paper_gap_analysis.md:112`).
- SAC-RAG's headline retrieval metric is unreproducible on this protocol.
- No corpus, no chunking, no persistent index, no `doc_id` space shared across questions.
- BM25 is refit per question; no IDF across a real collection.
- K is capped at 10 (`RETRIEVAL_MAX_K`), which happens to match SAC-RAG's top-10, but only by luck.

### 5.3 No reranking, no decomposition, no compression, no coverage, no controller

`bge-reranker-base`, question decomposition, compression, evidence coverage and any gap controller
are all absent (zero references in `src/`). The spec's PandaRAG components are 100% new code.

### 5.4 No configuration system beyond constants

Everything is a Python constant or env var (`os.environ`). There is no `configs/` directory, no
resolved-config artifact per run beyond `meta.json`'s hand-built dict, and no fingerprinting of a
config as a whole. `meta.json` records `dataset.sha256` and pricing/model/branch details, but there
is no single "configuration fingerprint" as the spec's §8/§9 requires.

### 5.5 Result schema is complete but flat

`results/adaptive_rag.jsonl` keys (verified):

```
question_id, question, question_type(bridge|comparison), level, gold_answer, prediction,
exact_match, f1, retrieval_k, retrieval_method, retrieved_docs, retrieved_doc_count,
retrieval_doc_recall, retrieval_doc_complete, retrieval_sentence_recall,
retrieval_sentence_complete, retrieval_complete, pool_size, adaptive_policy, adaptive_signals,
adaptive_decision, adaptive_reason, adaptive_confidence, adaptive_threshold, initial_k, final_k,
actual_retrieved_docs, input_tokens, output_tokens, cached_input_tokens, reasoning_tokens,
llm_calls, api_cost_usd, model, thinking, finish_reason, truncated, from_cache,
latency_seconds, faithfulness, hallucination, faithfulness_note, success, error, timestamp_utc
```

Good news: nearly all of the spec's §9 fields already exist under these names. Missing for the new
work: `dataset`, `pipeline`, `requirements`, `requirement_coverage`, `retrieval_rounds`,
`retrieval_calls`, `retrieval_latency_seconds`, `generation_latency_seconds`,
`total_latency_seconds`, `selected_context`, `source_document_ids`. Field naming must stay
**backward compatible** — `final_analysis.py`, `cross_pipeline_analysis.py` and the tests read these
keys (decision D10).

### 5.6 Metric normalizer is tuned for HotpotQA only

`answer_eval.normalize_answer` = lowercase → strip punctuation → drop articles → collapse spaces.
This mirrors the paper (DPR-style NFKD/lowercase/punct/article removal) *except* NFKD is not applied.
2Wiki answers are short entities (should be fine); MuSiQue answers are longer spans and often contain
enumerations. **Unverified on either dataset** — the pilot must check equivalent answers that differ
only by Unicode form or article placement (task P7.1).

### 5.7 Good: `evaluation.recall_at_k` already matches the paper

Paper: `Recall@K = |D_ret^K ∩ D_GT| / |D_GT|`. Repo: `len(gold & set(retrieved[:k])) / len(gold)`,
`NaN` when gold is empty. Identical definition — it only needs a corpus-level document set to be
meaningful. No metric redefinition required; the *retrieval protocol* is what changes.

---

## 6. Spec-to-repo coverage matrix

| Spec item | State | Note |
|---|---|---|
| P0 No RAG | ✅ exists | frozen |
| P1 Standard RAG | ✅ exists | K=5 over per-question corpus |
| P2 Hybrid RAG | ✅ exists | RRF k=60 depth=10 |
| P3 Original Adaptive RAG | ✅ exists | frozen artifact replay |
| P4 SAC-RAG Adaptive | ❌ | needs corpus retrieval, classifier wiring, 3 strategies, reranker |
| P5 Full SAC-RAG | ❌ | needs compression module (prompt unpublished) |
| P6 PandaRAG | ❌ | needs planner, evaluator, coverage, controller, context builder |
| P7 PandaRAG+Compression | ❌ | reuse P5 compressor |
| 3 dataset adapters | 🟡 partial | files + manifest exist; loader is HotpotQA-only |
| Unified `QAExample` | ❌ | `HotpotExample` is HotpotQA-shaped |
| Corpus/index manager | ❌ | per-question indices only |
| Chunking | ❌ | absent (paper does not publish chunk size either) |
| BGE reranker | ❌ | deps present (torch 2.14, transformers 5.17) |
| Query-level retrieval cache | ❌ | only LLM responses are cached |
| Config files + fingerprint | ❌ | constants + hand-built meta |
| Result schema v2 | 🟡 | 90% present; needs new keys |
| Faithfulness / grounding | ✅ | judge v1.0 exists |
| Paired statistics | ❌ | no significance tests anywhere |
| `results/legacy/` split | ❌ | flat `results/` (decision D10) |
| `docs/pandarag/*` | ❌ | this audit is the first file |

---

## 7. Reuse decisions (concrete)

### 7.1 Reuse unchanged
`src/llm.py`, `src/answer_eval.py`, `src/evaluation.py`, `src/faithfulness_eval.py`,
`src/download_datasets.py`, `src/svg_charts.py`, and all four existing pipelines plus their tests.

### 7.2 Reuse with verification
`answer_eval.normalize_answer` — run cross-dataset equivalence checks in the pilot before trusting
EM/F1 on 2Wiki/MuSiQue. If MuSiQue needs span normalization, that is a documented deviation, not a
silent change.

### 7.3 Reuse with relocation
`src/classifier.py` (v1.3, 272 lines, 6 tests) is already a working LLM question classifier with
JSON-schema validation, keyword-heuristic diagnostics and a documented fallback. The spec asks for
`src/pipelines/sac_rag/classifier.py`. Relocating breaks `tests/test_classifier.py` imports and the
`data/semantic_types_*.json` provenance path — so either import it across (thin re-export) or move
it with an explicit test update. Recommend a **re-export shim**, keeping the file where it is
(decision D4).

### 7.4 Reuse the ranking primitives, not the corpus model
`RankedHit`, `rrf_score`, `fuse_rankings`, `tokenize` and the FAISS/BM25 call sites are reusable per
query. `DenseRetriever.rank_example` / `BM25Retriever.rank_example` (which take a `HotpotExample`)
are not — they encode the per-question corpus. New `src/retrieval/` indexes over a built corpus and
calls into these primitives.

---

## 8. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Venv interpreter moved (3.14 → 3.15); nothing imports | **blocks all work** | relink symlink; add `PYTHON`/`UV_PYTHON` pin to README; check at start of every run |
| R2 | Base paper's corpus protocol is not published, so Recall@10 is not comparable | **high** | state the deviation, publish our corpus recipe + fingerprint, never claim paper-equivalent recall (protocol §6) |
| R3 | Paper's classifier and compression prompts are not published | high | label our prompts as implementation choices; keep them versioned; no claim of reproduction |
| R4 | gpt-5-mini vs DeepSeek; compressor is a different model again in the paper | high | keep DeepSeek everywhere; declare internal-validity-only comparison (paper's own limitation 3 does the same) |
| R5 | 500 vs 1,000 questions per dataset; paper's subtype counts (Fact 958/20/22) can't be matched | medium | state the deviation; our classifier's distribution is already frozen and documented |
| R6 | BGE cross-encoder on CPU: 20–40 candidates × 500–1500 questions | medium (time) | batch scoring, cache reranker scores keyed by (query, doc_id), measure before scaling |
| R7 | New pipelines could accidentally leak gold into runtime decisions | **high (integrity)** | typed boundary: controller/evaluator receive `QAExample` without `answer`/`gold_evidence`; add integrity tests (spec §12) |
| R8 | LLM calls multiply (classifier + planner + evaluator + compressor + judge) and cost per question is no longer 1 call | medium | per-call accounting through one client; budget caps; cache; pilot before grid |
| R9 | Moving `results/` or `data/` breaks `config.py`, tests and old meta paths | medium | additive paths only (`results/sac_rag/`), leave old files in place (D10) |
| R10 | MuSiQue paragraph-level evidence makes sentence-level recall undefined | low | report paragraph recall; `evaluation.py` already returns NaN for empty gold sets |
| R11 | 2Wiki/MuSiQue contexts contain 10–20 paragraphs per question; a corpus built from *only these* is still small and will inflate recall | **high** | corpus protocol must be explicit about what is indexed and its size; a 20-paragraph corpus must never be described as open-domain retrieval |

R11 deserves emphasis: if the new corpus is "the union of every question's 10–20 distractor
paragraphs", Recall@10 will look respectable while being meaningless, and it re-commits the exact
mistake the current repository already documents. This is decision **D2**, the single most
consequential open question in Phase 0.

---

## 9. What Phase 0 did not verify

- Whether EM/F1 normalization is correct for MuSiQue's long-form answers (pilot task P7.1).
- Whether the paper's per-type counts can be approached by our frozen classifier (they cannot be
  matched exactly; it publishes neither prompt nor accuracy).
- Actual corpus size implied by the paper's Recall@10 (0.42–0.48 on MuSiQue, 0.88 on HotpotQA).
- Reranker throughput on this CPU.
- That `docs/base_paper_gap_analysis.md` (2026-10-07, from a prior session) matches this audit
  everywhere; the two were produced from the same paper and agree on the key gaps, but this audit
  supersedes it where they differ.
