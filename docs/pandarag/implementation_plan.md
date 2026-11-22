# Implementation Plan — PandaRAG

**Companion to:** `repository_audit.md` (what exists) and `sac_rag_reproduction_protocol.md` (what
the paper says).
**Status:** Phase 0 deliverable. **No pipeline code has been written.** Approval on §2 is required
before Phase 1 starts.

---

## 1. Plan in one paragraph

Reuse the generation/cost/cache/evaluation layer as-is. Replace the retrieval *protocol* (per-question
indices → persistent corpus indices with fingerprints). Generalize the dataset layer into three
adapters behind one canonical record, with gold fields structurally unavailable to runtime
components. Build SAC-RAG Adaptive on that foundation, add the compressor, then build PandaRAG's five
components on top. Re-run the historical baselines on the **same** corpus so comparisons are not
confounded, then freeze and run the grid.

**Estimated effort:** 9 phases. Phases 1–3 are the heavy ones (~60% of the code). Phases 4–6 are the
research contribution. Phases 7–9 are experiments and analysis.

---

## 2. Decisions requiring sign-off

Each needs an answer before the phase that depends on it. My recommendation is stated, with the
evidence behind it.

### D1 — `AGENTS.md` conflicts with the new scope *(blocks nothing, but must be resolved on paper)*

`AGENTS.md` §Dataset says "Do NOT download another dataset", "The experiment must use exactly these
500 questions", and defines the project as four pipelines. The new specification requires three
datasets and eight systems. The 2Wiki/MuSiQue files were already downloaded and committed
(`ff78730`) with a provenance manifest.

**Options:** (a) amend `AGENTS.md` to a scoped version + add `docs/pandarag/experiment_protocol.md`
as the authority for the new work; (b) leave `AGENTS.md` untouched and rely on the new protocol doc.
**Recommend (a)** — the audit exists to prevent silent scope drift, and two conflicting rulebooks is
exactly that.

### D2 — **Corpus protocol** (the decisive decision; everything downstream depends on it)

The paper never describes its corpus: only "Milvus collection `en_doc_chunk`, 1536-dim, cosine",
metadata in MongoDB, no source, size, chunk length or overlap. Recall@10 is therefore
**not reproducible**, and whatever we choose defines our retrieval numbers.

Corpus sizes we can build *today* from committed data (union of every question's shipped paragraphs,
deduplicated by title — measured, not estimated):

| Dataset | Unique docs | Repeated paragraph instances | Title collisions with **different text** |
|---|---|---|---|
| HotpotQA | 4,928 | 28 | 0 |
| 2Wiki | 3,246 | 1,754 | 0 |
| MuSiQue | 6,356 | 3,640 | **1,116** |

| Option | What is indexed | Pros | Cons |
|---|---|---|---|
| **A. Pooled candidate corpus** (recommended) | union of the shipped paragraphs per dataset, `doc_id = title` (HotpotQA/2Wiki) and `title#sha1(text)[:8]` for MuSiQue's 1,116 conflicting titles | no downloads; every gold document is present; recall becomes a real (non-trivial) metric at ~3–6k docs; zero licensing surface | small vs Wikipedia, so recall will be optimistic in absolute terms; must never be called open-domain |
| B. Full-wiki corpora | HotpotQA fullwiki Wikipedia abstracts (~5M docs, multi-GB download), 2Wiki full wiki contexts, MuSiQue full paragraphs | closest to the paper's intent; recall magnitudes become plausible | contradicts `AGENTS.md`; multi-GB artifacts in git/none; index build hours; licensing; a different research project |
| C. Per-question distractor corpus (status quo) | 10 paragraphs per question | zero work | Recall@10 ≡ 1.000 — the number the paper headlines is meaningless; audit §5.2 |

**Recommend A**, documented as "pooled candidate corpus (closed-book over each dataset's shipped
paragraph pool)", with B explicitly listed as future work and R@10 never compared to the paper's
0.88/0.70/0.48.

### D3 — Baselines must be re-run on the new corpus *(methodological, not optional)*

If P1/P2/P3 keep their per-question retrieval (R@10 = 1.0) while P4–P7 use the corpus, every
comparison mixes two retrieval problems and the paper's central claim becomes unfalsifiable. Any
P4-vs-P1 gain could be pure retrieval-protocol difference.

**Recommendation:** keep the historical P0–P3 artifacts frozen and untouched, and add
**corpus-protocol twins** in the new results namespace:

| New ID | System | Retrieval |
|---|---|---|
| P1c | Standard RAG (corpus) | dense, corpus index, K=5 |
| P2c | Hybrid RAG (corpus) | dense+BM25 RRF, corpus index, K=5 |
| P3c | Adaptive RAG (corpus) | same frozen confidence rule, applied to corpus retrieval |

The paper's own Naïve RAG (= dense over corpus, top-10→5) is P1c with K=10 retrieved / 5 fed — add
that as **P1n** to be faithful to paper Table 3. Then {P1c, P2c, P3c, P1n} vs {P4, P5, P6, P7} is a
controlled comparison; {P0–P3} remains the historical study.

### D4 — Where the classifier lives

`src/classifier.py` (v1.3) is frozen and referenced by `data/semantic_types_*.json` and
`tests/test_classifier.py`. The spec's layout wants it under `src/pipelines/sac_rag/classifier.py`.

**Recommend:** leave the file in place; add `src/pipelines/sac_rag/classifier.py` as a 5-line
re-export shim with a docstring pointing at the original. No test churn, no provenance break, target
layout satisfied.

### D5 — Evidence abstraction (paragraph vs sentence)

HotpotQA and 2Wiki label sentence-level gold; MuSiQue labels paragraphs (`sent_id = -1`). The new
adapters need one evidence type.

**Recommend:** `GoldEvidence(kind: "paragraph"|"sentence", doc_id, sent_id: int|None, text)`.
`evaluation` computes sentence-level recall only where `kind == "sentence"` and paragraph-level
everywhere; the result record carries `gold_evidence_kind` so no table mixes the two silently.

### D6 — Planner caps

`PANDARAG_MAX_REQUIREMENTS` (recommend **4**), `PANDARAG_MAX_ROUNDS = 2`,
`PANDARAG_MAX_DOCUMENTS = 20`, `PANDARAG_MAX_CONTEXT_TOKENS = 4000`,
`PANDARAG_MAX_RETRIEVAL_CALLS = 8`. The paper's decomposition says "k independent sub-questions"
without a cap; these are engineering limits, to be tuned on the validation pilot and **frozen**
before the final runs.

### D7 — Evidence evaluator: LLM judge vs local NLI

| Option | Cost/q (off-peak) | Determinism | Quality |
|---|---|---|---|
| LLM judge (structured JSON, 1 call per requirement batch) | ≈ $0.001 (dominant term: ~16 req×doc pairs) | temperature 0, cached | high, same family as generator |
| Local NLI (`bge-reranker-base` score as entailment proxy) | ≈ $0 | high | **wrong semantics** — the audit and spec both forbid treating relevance as entailment |
| Local NLI cross-encoder (`cross-encoder/nli-deberta-v3-base`) | ≈ $0, CPU latency | high | plausible, unvalidated on our data |

**Recommend:** LLM judge as the primary (it is the only option with defensible support semantics),
with the reranker score retained as a *feature*, not as the support decision; optionally add the local
NLI variant later as an ablation. Cost is acceptable (§6).

### D8 — Reuse the frozen faithfulness judge as a second evaluator signal?

`faithfulness-judge/1.0` already produces claim-level labels for the *answer* against the context.
It does not evaluate requirement↔document support, so it cannot replace the evidence evaluator — but
its claim decomposition is a useful validation signal.

**Recommend:** no reuse in the runtime path; use the judge only for grounding metrics (as now). Keep
the two concerns separate — mixing them would violate spec §14 rule 10.

### D9 — Config format

PyYAML is not installed. The spec's layout shows `configs/*.yaml`.

**Recommend:** **JSON** (`configs/*.json`) with a dataclass loader and a `config_fingerprint()`
(sha256 over the resolved dict). Zero new dependencies, satisfies "explicit configuration files" and
"save resolved configuration". Add a `.schema.json` per config for validation. If you prefer YAML,
that is one `pip install pyyaml` — say so and I'll use it.

### D10 — Results layout and schema versioning

Old files (`results/*.jsonl`, `results/retrieval/*`) are read by `config.py`, `final_analysis.py`,
`cross_pipeline_analysis.py` and several tests.

**Recommend:** do **not** move legacy files. Create `results/sac_rag/`, `results/pandarag/`,
`results/legacy/` (the last only as a *symlink-free pointer doc*, or leave legacy where it is and
document it). New records carry `schema_version: 2`, `dataset`, `pipeline`, and the new
timing/coverage fields, while keeping every legacy key name unchanged so old analysis code keeps
working.

### D11 — Retrieval budget accounting for retrieval cost

Embedding and reranking are local and un-priced. The spec (§9) says to track them separately rather
than fold them into API cost.

**Recommend:** every record carries `retrieval_latency_seconds`, `reranking_latency_seconds`,
`embedded_document_count`, `reranked_pair_count`, and a clearly separated
`auxiliary_llm_cost_usd` (classifier + planner + evaluator + compressor) alongside
`generation_cost_usd`. Headline "Avg Tokens" stays generation-only to match the paper, with the
auxiliary number always reported adjacent to it.

---

## 3. Target layout vs existing files

Legend: **[R]** reuse unchanged · **[E]** extend · **[M]** move/shim · **[N]** new.

```
src/
  config.py                      [E]  + corpus/reranker/pandarag constants, config_fingerprint()
  llm.py                         [R]  as-is (classifier, planner, evaluator, compressor all call it)
  answer_eval.py                 [E]  + NFKD (D-check in pilot), no behaviour change for legacy
  evaluation.py                  [R]  recall_at_k already matches the paper
  faithfulness_eval.py           [R]  as-is
  dataset.py                     [E]  keep HotpotExample; delegate to datasets/, handle kind=paragraph
  classifier.py                  [R]  frozen v1.3
  download_datasets.py           [R]  as-is

  datasets/
    base.py                      [N]  QAExample, GoldEvidence, Document, DatasetAdapter protocol
    hotpotqa.py                  [N]  wraps dataset.py
    twowiki.py                   [N]
    musique.py                   [N]
    registry.py                  [N]  name -> adapter, split, fingerprint, expected size

  retrieval/
    corpus.py                    [N]  build corpus (D2), stable doc_id, conflict policy (MuSiQue)
    chunking.py                  [N]  paragraph-level primary; sentence spans for span extraction
    dense.py                     [E]  from retrieval.DenseRetriever, corpus-level + persisted FAISS
    bm25.py                      [E]  from retrieval.BM25Retriever, one index per corpus
    fusion.py                    [E]  from retrieval.fuse_rankings / rrf_score
    reranker.py                  [N]  bge-reranker-base CrossEncoder, batched, score cache
    retrieval_cache.py           [N]  key = corpus_fp + query + retriever cfg + fusion cfg + reranker cfg
    index_manager.py             [N]  build/load/verify per (dataset, split), manifest with sha256
    schemas.py                   [N]  RetrievedDoc, RankedList, CorpusManifest

  pipelines/
    no_rag.py / standard_rag.py / hybrid_rag.py / adaptive_rag.py   [R]  FROZEN, untouched
    corpus_baselines.py          [N]  P1c / P2c / P3c / P1n (D3) — new module, reuses their logic
    sac_rag/
      classifier.py              [M]  shim -> src.classifier (D4)
      fact_retrieval.py          [N]  BM25-10 + dense-10 -> union -> rerank -> 10
      definition_retrieval.py    [N]  dense-8, no rerank
      reasoning_retrieval.py     [N]  decompose -> dense-10 each -> union -> rerank -> 10
      decomposition.py           [N]  prompt `sac-rag-decompose/1.0`
      reranking.py               [N]  thin wrapper over retrieval/reranker
      context_compression.py     [N]  prompt `sac-rag-compress/1.0`, extractive + attribution
      pipeline.py                [N]  CLI, per-question orchestration, meta, schema v2
    pandarag/
      evidence_planner.py        [N]  prompt `pandarag-plan/1.0` + DAG validation + fallback
      evidence_evaluator.py      [N]  prompt `pandarag-support/1.0` (D7)
      coverage.py                [N]  per-requirement status, all-covered flag
      controller.py              [N]  gap-directed retrieval, budget, stop conditions
      context_builder.py         [N]  evidence-preserving selection + token budget
      compression.py             [N]  optional; delegates to sac_rag/context_compression
      generator.py               [N]  shared generator surface, evidence trace
      pipeline.py                [N]  CLI, orchestration, meta, schema v2

configs/                         [N]  JSON (D9): datasets/*.json, retrieval/*.json, sac_rag.json, pandarag.json
tests/                           [N]  8 new files (§5)
docs/pandarag/                   [N]  this plan + audit + protocol + architecture/ablations/limitations
```

**Not doing:** no framework, no new abstraction layer over `llm.py`, no restructuring of the four
frozen pipelines, no `data/` reorganization (moving `data/*.json` into `data/hotpotqa/` would break
`config.DATASET_FILE`, six tests and every stored artifact path, for zero research value).

---

## 4. Phases, tasks, exit criteria

Each phase ends with: tests written *in that phase*, all previous tests green, artifacts committed.
A phase is not complete if it is implemented but untested.

### Phase 1 — Datasets + retrieval foundation
1. Fix the venv (`ln -sf /usr/bin/python3.14 .venv/bin/python`), record the pin in README.
2. `datasets/base.py`: `QAExample`, `GoldEvidence` (D5), `Document`; adapters must expose
   `fingerprint()`, `split`, `expected_size`.
3. Adapters: hotpotqa (wrap `dataset.py`, do not fork), twowiki, musique. Validation: unique ids,
   evidence resolvable, no shuffling, fingerprint match against `dataset_manifest.json`.
4. `retrieval/corpus.py` + `index_manager.py`: build the pooled corpus (D2), stable `doc_id`,
   MuSiQue conflict policy (content-hash ids), persist `corpus.jsonl` + `manifest.json` with sha256.
5. Build dense (FAISS flat, L2-normalized) and BM25 indexes per dataset; record model, dim, k1/b/eps,
   tokenizer, build time.
6. `evaluation` smoke: Recall@{1,3,5,10} on 50 questions per dataset.
   **Expect R@10 well below 1.0 — if it is ~1.0, the corpus is wrong.**
   *Exit criteria:* three corpora built and fingerprinted; recall is a real number; 0 unresolvable
   gold docs; legacy 172 tests still pass.

### Phase 2 — SAC-RAG Adaptive (P4)
1. `fact_retrieval` / `definition_retrieval` / `reasoning_retrieval` per protocol §3–5.
2. `decomposition` with JSON validation, fallback to the original question, `k ≤ 4`.
3. `reranker` integration (batched, deterministic tie-break by `doc_id`, latency recorded).
4. `pipeline.py` CLI (`--dataset --split --limit --config --out`), meta with resolved config +
   fingerprints + retrieval stats.
5. Tests: strategy routing, dedup, deterministic rerank order, decomposition merge, Definition
   skipping the reranker, no gold leakage.
   *Exit criteria:* 50-question pilot on HotpotQA; retrieval provenance inspectable per question;
   every strategy exercised; cost per question recorded.

### Phase 3 — Full SAC-RAG (P5)
1. `context_compression`: extractive snippets, doc attribution preserved, versioned prompt,
   compression ratio + token deltas recorded, source ids mapped.
2. Same pipeline, `--compression on|off`; only the compressor changes (protocol §6).
3. Tests: ratio measurement, attribution preserved, empty-compression fallback, no gold in prompt.
   *Exit criteria:* P4 and P5 differ only by the compressor; both run 50 questions; evidence
   retention measured.

### Phase 4 — PandaRAG planner + evaluator (no generation)
1. `evidence_planner` with DAG validation (unique ids, valid refs, no cycles, non-empty queries,
   caps from D6) and safe fallback recorded per question.
2. `evidence_evaluator` per requirement-document pair (D7): `relevance_score`, `support_score`,
   `support_status ∈ {supported, partial, insufficient, contradicted, unknown}`, spans, reason.
3. `coverage.py` aggregation: supported/partial/uncovered/contradicted counts, all-covered flag.
4. Tests (offline, mocked): single-hop ⇒ one requirement; multi-hop ⇒ multiple; invalid planner JSON
   ⇒ fallback; relevant-but-unsupported doc ≠ supported; dependency ordering respected; no gold
   access (assert by type).
   *Exit criteria:* planner+evaluator+coverage run on 50 questions with **zero** generator calls;
   traces saved for manual inspection.

### Phase 5 — Gap controller
1. `controller.py`: actions {stop, targeted_retrieval, reformulate, expand_pool, switch_source,
   budget_exhausted}; explicit budget counters; no-improvement stop.
2. Tests on constructed cases: missing E2 ⇒ targeted retrieval for E2 only; covered requirements are
   not re-retrieved; budget exhaustion stops cleanly; no infinite loop (hard iteration cap); every
   decision carries a machine-readable `reason`.
   *Exit criteria:* synthetic-construction tests pass; controller decisions traceable per question.

### Phase 6 — Context builder + generator
1. `context_builder.py`: smallest context preserving ≥1 strong passage per covered requirement;
   bridge evidence for dependencies retained; dedupe docs and sentences; token budget enforced.
2. `generator.py`: shared generator surface (same client, same prompt family as P0–P7), evidence
   trace stored.
3. Optional compression path (P7).
   *Exit criteria:* end-to-end on 50 HotpotQA questions; context builder never drops a covered
   requirement's evidence (asserted in tests); tokens measured.

### Phase 7 — Pilots (≈50 questions × 3 datasets)
Engineering only: schema validation, prompt sanity, metric validation (**D-check: does EM/F1 treat
equivalent MuSiQue/2Wiki answers correctly?**), cost estimation, error handling. **Pilot metrics are
never reported as results.** Manual inspection of ~20 failures per dataset.

### Phase 8 — Freeze + full evaluation
Freeze code, configs, splits, corpus, protocol. Then the matrix: {P1c, P1n, P2c, P3c, P4, P5, P6, P7}
× 3 datasets × 500 questions, plus ablations A0–A5, plus faithfulness judging. Raw per-question
records retained. Paired tests on identical question ids; bootstrap CIs; no post-hoc tuning.

### Phase 9 — Paper-ready analysis
Tables: overall, per-dataset, per-type, retrieval, cost-quality Pareto, context-token distribution,
evidence coverage before/after retrieval, faithfulness, ablations, error taxonomy, statistics,
limitations. Figures via existing `svg_charts.py` patterns.

---

## 5. Test plan (maps spec §12 to files)

| File | Covers |
|---|---|
| `tests/test_datasets.py` | three adapters, stable ids, no duplicate ids, evidence mapping, fingerprints, split isolation |
| `tests/test_corpus.py` | corpus build determinism, MuSiQue title-conflict policy, doc_id stability, manifest sha |
| `tests/test_retrieval_corpus.py` | dense/bm25 over corpus, R@K monotonicity, cache key invalidation on corpus change |
| `tests/test_sac_rag.py` | routing, dedup, reranker determinism, decomposition merge, Definition skips rerank, compression attribution |
| `tests/test_pandarag_planner.py` | requirement atomicity, DAG validity, caps, fallback |
| `tests/test_pandarag_evaluator.py` | status taxonomy, relevance≠support, dependency handling |
| `tests/test_pandarag_controller.py` | targeted retrieval of the gap only, budget, stop, no loops |
| `tests/test_pandarag_context.py` | evidence preservation, dedupe, token budget, bridge retention |
| `tests/test_experiment_integrity.py` | legacy results unchanged (sha), frozen dataset sha, no gold in runtime inputs, same ids compared, cached vs live distinguished, resolved config saved |

Offline/mocked tests first; live API only after unit tests pass and only on `--limit 50` pilots.

---

## 6. Cost and runtime budget (off-peak DeepSeek rates: $0.15/M in-miss, $0.60/M out)

Measured baseline from existing runs: 500 questions ≈ $0.027–0.055 (1 call/q). Classifier: $0.032
for 500 (≈$0.00006/q).

Per-question estimate for P6 (classifier + planner + evaluator + generation, no compression):

| Component | est. tokens (in/out) | est. cost |
|---|---|---|
| classifier | 400 / 60 | $0.00010 |
| planner | 400 / 200 | $0.00018 |
| evidence evaluator (dominant, ~2 batches) | 4,000 / 400 | $0.00084 |
| generation | 1,500 / 30 | $0.00024 |
| **total** | | **≈ $0.0014** |

P7 adds ≈$0.0007 (compressor). So one 500-question dataset run ≈ **$0.7–1.0**; the full Phase 8
matrix (8 systems × 3 datasets) ≈ **$15–30**; ablations A0–A5 on HotpotQA ≈ **$5**. Pilots (Phase 7)
≈ **$1**. All cached on disk, so re-analysis is free.

Runtime: dominated by CPU reranking (≈20 pairs/query) and sequential LLM calls; pilot will measure
before the grid. Expect Phase 8 to be hours, not days, and to be rate-limit-bound rather than
compute-bound.

---

## 7. Risk register (carried from the audit)

| Risk | Mitigation in this plan |
|---|---|
| R1 venv broken | Phase 1 task 1, before anything else |
| R2 corpus unpublished | D2 = declared protocol, never compared to paper R@10 |
| R11 corpus too small (recall inflated) | publish corpus size + build recipe; call it "pooled", never "open-domain" |
| R7 gold leakage | type boundary in `datasets/base.py` (runtime objects have no gold fields) + integrity tests |
| R6 CPU reranker latency | batch + score cache + measured in the pilot before the grid |
| R8 LLM call multiplication | one client, one ledger, `auxiliary_llm_cost_usd` reported separately, disk cache |
| D3-confounding | corpus-protocol baselines (P1c/P2c/P3c/P1n) in the same grid |
| R9 legacy breakage | additive namespaces only; old keys and files untouched; `schema_version: 2` |

---

## 8. Scope guard — what I will not do without a new instruction

- Not touch `src/pipelines/{no_rag,standard_rag,hybrid_rag,adaptive_rag}.py` or their artifacts.
- Not rewrite `data/*.json`, not resample, not re-run the classifier.
- Not tune anything on test questions.
- Not implement MMR or extra routing strategies (the audit lists them as optional variants only).
- Not claim PandaRAG superiority from anything but paired experiments.
- Not add a framework, a hosted DB, or a web service.

---

## 9. What I need from you to start Phase 1

1. **D2** — corpus protocol: pooled candidate corpus (A) — yes/no.
2. **D3** — add corpus-protocol baselines P1c/P2c/P3c/P1n to the grid — yes/no.
3. **D6** — accept the cap values (max requirements 4, rounds 2, docs 20, context 4000 tokens,
   retrieval calls 8) as pre-freeze defaults?
4. **D7** — LLM evidence judge as primary evaluator — yes/no.
5. **D9** — JSON configs (no new dependency) vs YAML (one new dependency).
6. **D1** — amend `AGENTS.md` for the new scope, or leave it and let the protocol doc govern.

Defaults if you just say "go": D1(a) amend, D2=A, D3=yes, D6=as stated, D7=LLM judge, D9=JSON.
