# SAC-RAG Reproduction Protocol

**Purpose:** a parameter-by-parameter record of what the base paper specifies, what this repository
will implement, and where the two diverge. Written **before** any SAC-RAG code exists, so that
deviations are declared in advance rather than discovered afterwards.

**Base paper:** Zhang, Yu, Huo, Su. *SAC-RAG: Semantic Adaptive Context Compression for
Retrieval-Augmented Generation.* IEEE Access, 2026. DOI `10.1109/ACCESS.2026.3718025`.
Author's accepted version, 10 pages + references.

**Rule applied throughout:** a value is marked **REPRODUCED** only if the paper states it. A value
the paper does not state is marked **IMPLEMENTATION CHOICE** and is never described as SAC-RAG's.

---

## 1. Paper structure as understood

Five stages: (1) question classification → (2) type-specific retrieval → (3) context compression →
(4) answer generation → (5) evaluation. Four progressive ablation groups: Naïve, +Compression,
+Adaptive, Full SAC-RAG (paper Table 3). Three hypotheses, H1/H2/H3, verified in Table 4.

---

## 2. Question classifier

| Parameter | Paper | Repo plan | Status | Reason if deviating |
|---|---|---|---|---|
| Method | "LLM with a structured prompt and few-shot examples" (§III-B) | LLM, JSON-schema-validated, **zero-shot + frozen keyword diagnostics** | **DEVIATION** | Prompt and few-shot examples are **not published**. Our v1.3 classifier exists, is tested, and is frozen; changing it now would invalidate the committed `semantic_types_*.json`. |
| Categories | `{Fact, Definition, Reasoning}` | identical | **REPRODUCED** | — |
| Fact definition | facts, values, entity attributes; cue words *When, Where, How many, Which* | identical rule in v1.3 | **REPRODUCED (definition)** | — |
| Definition definition | concepts/entities; cue words *What is, Who is, Define* | identical rule in v1.3 | **REPRODUCED (definition)** | — |
| Reasoning definition | multi-step + causal; cue words *Why, How, What caused* | identical rule in v1.3, extended per `docs/base_paper_gap_analysis.md` (comparison ⇒ Reasoning) | **PARTIAL** | The paper's cue lists cannot cover HotpotQA comparison questions (its own limitation 1). Our extension is documented and frozen. |
| Classifier LLM | not stated in §III-B; implied gpt-5-mini family | `deepseek-flash` via `src/llm.py` | **IMPLEMENTATION CHOICE** | One model for the whole system; DeepSeek is the project's frozen generator. |
| Classifier output schema | not published (only "label c") | `{question_type, confidence, reason}` + validation | **IMPLEMENTATION CHOICE** | Spec §3.1 requires structure; paper's exact JSON is unknown. |
| Confidence calibration | not claimed | explicitly declared **non-calibrated** | **REPRODUCED (caveat)** | Paper makes no calibration claim either. |
| Failure fallback | not stated | documented fallback: keyword heuristic → else `Reasoning` (most retrieval) | **IMPLEMENTATION CHOICE** | Robustness requirement; logged per question. |
| Reported type distribution | Table 1: HotpotQA 958/20/22, 2Wiki 796/5/199, MuSiQue 732/28/240 | ours: HotpotQA 423/23/54, 2Wiki 358/0/142, MuSiQue 452/14/34 | **NOT COMPARABLE** | Different prompts; paper publishes no accuracy; its own counts drift between Table 1 and Table 5 (HP Fact 958 vs 954, Reasoning 22 vs 26). |

**Consequence to state in the paper:** per-type tables (paper Table 5) cannot be compared cell to
cell with ours. Compare *within* our system across configurations, and report our type distribution
explicitly.

---

## 3. Fact retrieval strategy

Paper §III-C-1: `S_Fact(q,d) = λ·BM25(q,d) + (1−λ)·cos(e_q,e_d)`, then
BM25 top-10 → dense top-10 → merge+dedupe by `doc_id` → BGE rerank → top-10.

| Parameter | Paper | Repo plan | Status | Reason |
|---|---|---|---|---|
| BM25 tokenizer | spaCy | `src/retrieval.tokenize` (regex, lowercase) | **DEVIATION** | spaCy model/version unspecified; adding spaCy is a new dependency. Our tokenizer is already frozen in P0–P3. Declare it. |
| BM25 impl | `rank_bm25.BM25Okapi` | same | **REPRODUCED** | — |
| BM25 k1 / b | 1.5 / 0.75 (§III-C-1) | `config.BM25_K1=1.5`, `BM25_B=0.75`, `epsilon=0.25` | **REPRODUCED** | epsilon is rank_bm25's own param, unreported by the paper. |
| BM25 depth | top-10 | top-10 | **REPRODUCED** | — |
| Dense model | `text-embedding-3-small`, 1536-d | `BAAI/bge-small-en-v1.5` (384-d) | **DEVIATION** | AGENTS.md mandates local embeddings; OpenAI embeddings are a hosted call with per-token cost the project does not account for. Recorded as a comparability limit. |
| Dense index | Milvus, collection `en_doc_chunk`, cosine | FAISS `IndexFlatIP` on L2-normalized vectors (= cosine), persisted per corpus | **DEVIATION (infrastructure)** | AGENTS.md forbids hosted vector DBs. Exact vs ANN search: flat is exact, so retrieval quality is not degraded — only speed differs. |
| Fusion of the two lists | "merge and deduplication by doc_id", then rerank | identical (union by `doc_id`, dedupe, keep both source ranks) | **REPRODUCED (procedure)** | — |
| **λ** | **never given a value** | **λ = 0.5** (equal weight, documented) | **IMPLEMENTATION CHOICE** | Paper defines the score but omits λ. Also: the paper merges the two *lists* and reranks, so λ may not even be used at runtime. We implement union-and-rerank as the primary path and keep λ available only for a documented ablation. |
| Reranker | `bge-reranker-base` cross-encoder | same (`BAAI/bge-reranker-base` via sentence-transformers `CrossEncoder`) | **REPRODUCED (model)** | — |
| Reranker candidate count | all merged candidates (≤20) | same | **REPRODUCED (derived)** | Paper says top-10 from each source, so ≤20 unique. |
| Final k (Fact) | 10 (Table 3) | 10 | **REPRODUCED** | — |
| Reranker score threshold | none stated | none; deterministic sort by score, ties broken by `doc_id` | **IMPLEMENTATION CHOICE** | Determinism requirement. |

---

## 4. Definition retrieval strategy

| Parameter | Paper | Repo plan | Status |
|---|---|---|---|
| Retrieval | dense only, cosine | dense only | **REPRODUCED** |
| Depth | top-8 directly into compression/generation | top-8 | **REPRODUCED** |
| BM25 | not used | not used | **REPRODUCED** |
| Reranker | **not applied** for Definition (Table 3: BGE rerank is "Fact/Reasoning" only) | not applied | **REPRODUCED** |
| Final k | 8 (Table 3) | 8 | **REPRODUCED** |

This row matters for implementation: Definition must **skip** the reranker. Applying it everywhere
would silently depart from Table 3 and change the ablation's meaning.

---

## 5. Reasoning retrieval strategy

| Parameter | Paper | Repo plan | Status | Reason |
|---|---|---|---|---|
| Decomposition | LLM produces a set of "independent sub-questions `{q1..qk}`" | LLM, JSON output, `k ≤ 4` default | **PARTIAL** | Paper does not state `k`, the model, or the prompt. `k ≤ 4` is an **IMPLEMENTATION CHOICE**; the spec's evidence planner has its own cap (D6). |
| Per-sub-question retrieval | dense, top-10 each, independent | identical | **REPRODUCED** | — |
| Merge | union, dedupe | union, dedupe by `doc_id`, retain `(sub_question → doc)` provenance | **REPRODUCED (procedure)** | Paper's equation shows merge then rerank; per-sub-question provenance is an addition for analysis (spec §3.1 requirement). |
| Rerank | BGE reranker over merged set | same | **REPRODUCED** | — |
| Final k | 10 (Table 3) | 10 | **REPRODUCED** | — |
| Decomposition prompt | not published | our prompt, versioned (`sac-rag-decompose/1.0`) | **IMPLEMENTATION CHOICE** | — |
| Sub-question independence | "independent sub-questions" in §III-C-3; the PandaRAG planner needs dependencies | SAC-RAG path: independent; PandaRAG planner: dependency DAG | **DEVIATION for PandaRAG only** | The spec's planner explicitly adds `depends_on`; SAC-RAG's own decomposition must stay independent to remain faithful. |

---

## 6. Context compression (Full SAC-RAG)

| Parameter | Paper | Repo plan | Status | Reason |
|---|---|---|---|---|
| Module | "unified LLM compression module … custom prompt" | LLM compressor, versioned prompt, per-document evidence extraction | **PARTIAL** | **Prompt is not published.** Ours must be labeled an implementation choice. |
| Input | `(q, R_t)` only | identical | **REPRODUCED** | — |
| Scope | unified across all three types; only `R_t` differs | identical | **REPRODUCED** | — |
| Output | "evidential snippets" | extractive-verbatim snippets + doc attribution | **IMPLEMENTATION CHOICE** | Paper does not say extractive vs abstractive, nor whether attribution is retained. Spec §3.2 requires attribution; extractive keeps faithfulness auditable. |
| Compressor model | **`qwen-flash`** (§VI-C limitation 3) | `deepseek-flash` | **DEVIATION** | Paper's limitation 3 is the only place it names the compressor, contradicting "all experimental groups use the same LLM configuration". We use one model for generator + compressor + classifier and declare it. |
| Compression ratio target | none stated | none imposed; measured, not targeted | **IMPLEMENTATION CHOICE** | Imposing a target ratio would make our token numbers non-comparable in an unpredictable direction. |
| Max output tokens | not stated | 768 (same ceiling as the faithfulness judge) | **IMPLEMENTATION CHOICE** | — |
| Gold visibility | must not see answer/evidence | enforced by type boundary + test | **REPRODUCED (by construction of §14 rules)** | — |

---

## 7. Answer generation

| Parameter | Paper | Repo plan | Status | Reason |
|---|---|---|---|---|
| Model | `gpt-5-mini` (§IV-B, Table 2 note) | `deepseek-flash` (DeepSeek-V4.1-Flash) | **DEVIATION** | Project-wide constraint (AGENTS.md); paper's limitation 3 makes the same class of concession. |
| Temperature | 0 | `config.TEMPERATURE = 0.0` | **REPRODUCED** | — |
| Input | compressed context `C_t` + original question | identical | **REPRODUCED** | — |
| Prompt | not published | frozen house template (`GENERATION_SYSTEM_PROMPT` + context block) | **IMPLEMENTATION CHOICE** | Must be identical across P0–P7 within a comparison. |
| Same model across groups | "All experimental groups use the same LLM configuration" | enforced (one client, one model id, recorded in meta) | **REPRODUCED** | — |

---

## 8. Data and evaluation

| Parameter | Paper | Repo plan | Status | Reason |
|---|---|---|---|---|
| Datasets | HotpotQA, 2WikiMultiHopQA, MuSiQue | same three | **REPRODUCED (names)** | — |
| Questions per dataset | 1,000 each | **500 each** | **DEVIATION** | 500 is the frozen, committed sample; re-sampling would break the historical HotpotQA comparison and the manifest. |
| Sampling rule / seed | not stated | seeded (`Random.Random(42)`), documented in `dataset_manifest.json` | **IMPLEMENTATION CHOICE** | — |
| Hops | HP 1–2, 2Wiki 2, MuSiQue 2–4 (Table 1) | HP all 2 (level=hard), 2Wiki 2 (filtered), MuSiQue 2/3/4 | **PARTIAL** | Our HotpotQA sample is hard-only, 2-hop; the paper allows 1–2 hops. Documented. |
| **Retrieval corpus** | **not described anywhere.** Only "Milvus collection `en_doc_chunk`, 1536-dim, cosine" and "metadata in MongoDB". No source, size, chunk length, or overlap. | **must be defined by us** (decision D2) | **NOT REPRODUCIBLE** | The single largest reproduction gap. Recall@10 depends entirely on it. |
| Recall@K definition | `|D_ret^K ∩ D_GT| / |D_GT|`, document level | `evaluation.recall_at_k` — identical | **REPRODUCED** | — |
| Gold document granularity | not stated (documents) | HotpotQA/2Wiki: paragraph title; MuSiQue: paragraph (`sent_id = -1`) | **PARTIAL** | MuSiQue has no sentence-level gold; report paragraph recall there and say so. |
| EM | normalized exact match, Unicode NFKD, lowercase, punctuation + article removal (DPR) | `answer_eval` (lowercase, punctuation, articles, **no NFKD**) | **PARTIAL** | Add NFKD or declare; verify in pilot. |
| F1 | token-level harmonic mean (DPR) | identical | **REPRODUCED** | — |
| Avg Tokens | **generation only**, mean per query (LLMLingua convention) | same meaning; we will additionally report classifier/decomposer/compressor tokens **separately** | **REPRODUCED (headline) + addition** | Paper hides auxiliary LLM cost; we report both, with the paper-comparable number first. |
| Significance testing | none reported | paired bootstrap / Wilcoxon on identical question IDs | **ADDITION** | Spec §10.3 requires it; the paper does not do it. |

---

## 9. The paper's four ablation groups and our mapping

Paper Table 3, verbatim structure:

| Module / parameter | Naïve | +Compression | +Adaptive | Full SAC-RAG |
|---|---|---|---|---|
| Classifier | ✗ | ✗ | ✓ | ✓ |
| Adaptive retrieval | ✗ | ✗ | ✓ | ✓ |
| Compression | ✗ | ✓ | ✗ | ✓ |
| Vector top-k | 10→5 | 10 | strategy-dep. | strategy-dep. |
| BM25 (Fact) | – | – | top-10 | top-10 |
| BGE rerank (Fact/Reasoning) | ✗ | ✗ | top-10 | top-10 |
| Sub-Q decomp (Reasoning) | ✗ | ✗ | ✓ | ✓ |
| Final k (Definition) | – | – | 8 | 8 |

Mapping to the specification's eight systems (§10.1):

| Spec ID | System | Relation to the paper |
|---|---|---|
| P0 | No-RAG | our own baseline; paper has none |
| P1 | Standard Dense RAG | ≈ paper "Naïve" (paper's Naïve takes dense top-10 then 5; ours is K=5 dense) — **not identical**, both retained and labeled |
| P2 | Hybrid RAG | our own baseline (RRF), not in the paper |
| P3 | Original Adaptive RAG | our frozen prior experiment; not in the paper |
| **P4** | **SAC-RAG Adaptive** | **paper's "+Adaptive" column** (Table 3) |
| **P5** | **Full SAC-RAG** | **paper's "Full SAC-RAG" column** |
| P6 | PandaRAG | our contribution |
| P7 | PandaRAG + Compression | our contribution + paper's compressor |

Note: the paper's `Compress` column is not one of the specification's eight systems; it is needed only
to reproduce H1. Add it internally as **P4b** if full H1/H2/H3 replication is desired.

---

## 10. Published results to compare against (paper's own numbers)

Recorded here so later comparisons are quoted correctly. **None of these may be presented as
directly comparable to our runs** — different corpus, embedding model, generator, question count.

Paper §V-A, Table 4 deltas, and §V-B text:

| Metric | Naïve | Compress | Adaptive | SAC-RAG |
|---|---|---|---|---|
| HotpotQA EM | 0.560* | 0.572* | **0.611** | 0.595 |
| HotpotQA tokens | — | — | 2,407 | **1,087** |
| HotpotQA R@10 | 0.787 | 0.788* | 0.880* | **0.881** |
| 2Wiki EM | 0.562* | 0.586* | **0.595** | 0.590 |
| 2Wiki tokens | — | — | 3,002 | **1,264** |
| 2Wiki R@10 | 0.684 | 0.685* | 0.709* | **0.704** |
| MuSiQue EM | 0.242* | 0.271 | 0.284 | 0.271 |
| MuSiQue tokens | — | — | 3,111 | **1,918** |
| MuSiQue R@10 | 0.421 | 0.421* | **0.479** | **0.479** |

`*` = derived from the paper's reported Δ (Table 4) rather than printed directly; all printed
values are exact. Baseline table (paper Table 2, F1/Acc with different backbones — **cited, never
re-run**): Vanilla RAG 61.3/56.9 · 13.5/8.3 · 45.9/42.8; IterDRAG 65.3/60.7 · 18.3/13.0 · 51.8/47.0;
ReSearch 70.3/68.1 · 30.9/27.4 · 62.5/61.0; AirRAG 74.9/70.0 · 30.3/23.6 · 65.7/63.2;
SAC-RAG 74.0/65.2 · 40.3/30.8 · 66.4/65.7 (HotpotQA / MuSiQue / 2Wiki).

---

## 11. Internal inconsistencies found in the paper (affect what we can claim)

1. **Compressor model.** Abstract/§III-E imply one LLM for everything; §VI-C limitation 3 says
   compression uses `qwen-flash` while generation uses `gpt-5-mini`. → We use one model and declare it.
2. **Type counts drift.** Table 1 (958/20/22) vs Table 5 (954/20/26) for HotpotQA; 2Wiki 199 vs 242
   Reasoning; MuSiQue 732/28/240 vs 724/31/245. → Per-type comparisons are unstable in the paper itself.
3. **H1 sign.** §V-B text says "at the cost of only a 1–2pp EM drop", but Table 4 H1 shows compression
   *improving* EM (+1.2/+2.4/+2.9pp on Naïve). Both cannot be about the same pair. → We report our own
   H1 as measured, with the pair named explicitly.
4. **Naïve "10→5".** Table 3 lists vector top-k "10→5" for Naïve: retrieve 10, feed 5. Unclear whether
   Recall@10 for Naïve is computed on the retrieved 10 or the fed 5. → We report both `retrieved@10`
   and `fed@k` recall, separately labeled.
5. **λ** is defined in Eq. 2 but never instantiated, and the procedure (list merge + rerank) does not
   obviously use it.

---

## 12. Not published — complete list

Everything below is an **implementation choice** in our reproduction. The paper gives no value, and
none of these may be described as "SAC-RAG's setting":

`λ` · classifier prompt · classifier few-shot examples · classifier model · decomposition prompt ·
decomposition model · max sub-questions `k` · compression prompt · compression output format ·
compression max tokens · compression ratio target · BGE reranker score threshold · Milvus index type
and build parameters · spaCy model · **corpus source / size / chunk length / overlap** ·
question sampling rule and seed · 2Wiki and MuSiQue preprocessing details · prompt used for
generation · whether retrieval provenance survives compression · significance tests.

---

## 13. Deviation ledger (to carry into the final paper's limitations)

| # | Deviation | Class | Effect on comparability |
|---|---|---|---|
| DV1 | Generator `deepseek-flash` instead of `gpt-5-mini` | model | Absolute EM/F1 differ; internal comparisons valid |
| DV2 | Compressor = generator instead of `qwen-flash` | model | Compression behaviour differs; token savings may differ |
| DV3 | Local `bge-small-en-v1.5` (384-d) instead of `text-embedding-3-small` (1536-d) | retrieval | Recall may be lower; FAISS flat is exact vs Milvus ANN |
| DV4 | FAISS instead of Milvus | infrastructure | No quality effect (exact search) |
| DV5 | Regex tokenizer instead of spaCy | retrieval | BM25 scores differ slightly |
| DV6 | 500 questions/dataset instead of 1,000 | sample | Wider CIs; subtype cells unstable |
| DV7 | HotpotQA sample is hard-only, 2-hop | sample | Harder than the paper's 1–2 hop mix |
| DV8 | Corpus protocol defined by us | **protocol** | Recall@10 is **not comparable**; must be reported as our own protocol |
| DV9 | Classifier prompt/model zero-shot rather than few-shot (unpublished) | classifier | Type distribution differs; per-type cells not comparable |
| DV10 | Compression is extractive with attribution | method | Token reduction and faithfulness may differ |
| DV11 | MuSiQue reported at paragraph-level recall | metric | Not directly comparable to sentence-level datasets |
| DV12 | λ = 0.5 used only in an ablation; primary path is union + rerank | method | Documented either way |

**Standing methodological consequence:** our reproduction is a *protocol-level* reproduction of
SAC-RAG's architecture, not a numeric replication of its table. Absolute numbers will differ; the
paper's own limitation 3 (model dependence) pre-authorizes this framing. Every claim in the final
write-up must be *internal*: "within our harness, P4 > P1 by X" — never "we reproduce 0.595 EM".
