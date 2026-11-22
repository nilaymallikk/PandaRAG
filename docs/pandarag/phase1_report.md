# Phase 1 Report — Datasets, Corpus and Retrieval Foundation

**Generated:** 2026-11-19T06:00:09+00:00  
**Commit:** `fe8ffd08c93d35d59c002aec1e89a2edca99a38e`  
**Scope:** Phase 1 only (local, no LLM calls, no API spend).  
**Regenerate with:** `python -m src.phase1_report`

Every number below is read from a measured artifact on disk (`data/dataset_manifest.json`, `cache/corpus/*/manifest.json`, `cache/indexes/*/manifest.json`, `results/sac_rag/retrieval/*.meta.json`) — nothing is transcribed by hand.

## 1. Python environment verification

| Item | Value |
|---|---|
| Interpreter | CPython 3.14.10 |
| `.venv/bin/python` resolves to | `/usr/bin/python3.14` |
| Platform | Linux-7.3.3-300.fc45.x86_64-x86_64-with-glibc2.44 |
| torch | 2.14.0 |
| transformers | 5.17.0 |
| sentence-transformers | 6.1.0 |
| faiss-cpu | 1.15.1 |
| rank-bm25 | 0.2.2 |
| numpy | 2.5.3 |
| openai | 3.16.2 |
| pytest | 9.1.1 |

The environment had been broken: `.venv/bin/python` pointed at `/usr/bin/python`, which had moved to 3.15.2 while the 191 installed packages live under `python3.14`. Nothing was deleted — the three interpreter symlinks were repointed to `/usr/bin/python3.14` (3.14.10, ABI-compatible with the 3.14.7 the venv was built with). The broken state is preserved in `.venv-backup-*/` with a restore instruction.

## 2. Test suite

```
$ python -m pytest tests/ -q
246 passed in 20.14s
returncode 0
```

172 pre-existing Scope-A tests plus 74 new Phase-1 tests (`test_datasets.py`, `test_corpus.py`, `test_retrieval_corpus.py`, `test_experiment_integrity.py`).

## 3. Dataset counts and fingerprints

| Dataset | Split | Questions | Evidence kind | SHA-256 (first 16) |
|---|---|---|---|---|
| hotpotqa | validation | 500 | sentence | `956405ceb43d7338` |
| 2wiki | dev | 500 | sentence | `658de4da7eb7e8db` |
| musique | validation | 500 | paragraph (sent_id = -1; dataset marks is_supporting per paragraph) | `79f5aefa5a2a286f` |

All three files are unchanged and verified against `data/dataset_manifest.json` at load time; a mismatch aborts the load instead of warning.

Semantic-type sidecars (`sac-rag-classifier/1.3`, already frozen before this phase):

| Dataset | Fact | Definition | Reasoning | Sidecar SHA-256 (first 12) |
|---|---|---|---|---|
| hotpotqa | 423 | 23 | 54 | `de27e0becd1a` |
| 2wiki | 358 | 0 | 142 | `bed073708347` |
| musique | 452 | 14 | 34 | `51729449b71a` |

## 4. Corpus construction and fingerprints

Protocol: pooled candidate corpus — the union of every paragraph shipped with every question in the dataset file, deduplicated by content hash of `(dataset, split, title, paragraph text)`. Gold annotations are used only for the offline provenance file; the runtime record is `{doc_id, title, text, dataset, split}`.

| Dataset | Docs | Titles | Instances seen | Duplicates collapsed | Used as gold | Never gold | Corpus SHA-256 (first 16) |
|---|---|---|---|---|---|---|---|
| hotpotqa | 4928 | 4928 | 4956 | 28 | 994 | 3934 | `2da92dd60bdbfbe1` |
| 2wiki | 3246 | 3246 | 5000 | 1754 | 990 | 2256 | `ad500ff361d7f699` |
| musique | 7192 | 6356 | 9996 | 2804 | 912 | 6280 | `7c485109ff42ded8` |

Corpora are written to `cache/corpus/<dataset>/<split>/` (gitignored) with `corpus.jsonl`, `provenance.json` and `manifest.json`; `phase1_artifacts.json` records the fingerprints so they survive in version control.

## 5. Evidence mapping validation

Every gold evidence item is mapped to a corpus document id at load time and re-resolved against the written corpus. Asserted in `tests/test_corpus.py::test_every_gold_document_is_present`.

| Dataset | Gold evidence items | Resolved | Unresolved |
|---|---|---|---|
| hotpotqa | 1212 | 1212 | 0 |
| 2wiki | 1000 | 1000 | 0 |
| musique | 1322 | 1322 | 0 |

MuSiQue is the only dataset whose gold is paragraph-level (`sent_id = -1`); it is normalised to `GoldEvidence(kind='paragraph', sent_id=None, text=<whole paragraph>)` so no consumer has to special-case it.

## 6. MuSiQue title-collision handling

MuSiQue reuses article titles across questions while shipping different paragraph text for them. `title` is therefore **not** a usable document key:

- titles in the corpus: **6356**
- documents after content-hash splitting: **7192**
- titles carrying more than one text: **421**
- repeated paragraph instances collapsed as exact duplicates: **2804**

Policy: `doc_id = sha1(dataset, split, title, text)`.[:15] — identical text always collapses to one document; different text under the same title always yields distinct documents. No paragraph is discarded. HotpotQA and 2Wiki have zero ambiguous titles, so the policy is a no-op there (28 and 1,754 exact-duplicate instances are collapsed respectively).

## 7. Retrieval sanity metrics (pooled corpus)

Dense: `BAAI/bge-small-en-v1.5` (384-d, L2-normalised, `faiss.IndexFlatIP`, exact search). BM25: `rank_bm25.BM25Okapi` (k1=1.5, b=0.75, epsilon=0.25, regex tokenizer). Hybrid: RRF (k=60, depth=10). Gold is used for scoring only. No LLM is involved.

| Dataset | Method | R@1 | R@3 | R@5 | R@10 | All-gold@10 | Median gold rank |
|---|---|---|---|---|---|---|---|
| hotpotqa | dense | 0.447 | 0.822 | 0.885 | 0.922 | 0.854 | 2 |
| hotpotqa | bm25 | 0.389 | 0.622 | 0.729 | 0.849 | 0.704 | 2 |
| hotpotqa | hybrid | 0.431 | 0.751 | 0.855 | 0.927 | 0.858 | 2 |
| 2wiki | dense | 0.492 | 0.738 | 0.785 | 0.823 | 0.646 | 1 |
| 2wiki | bm25 | 0.434 | 0.640 | 0.699 | 0.764 | 0.542 | 1 |
| 2wiki | hybrid | 0.480 | 0.723 | 0.772 | 0.813 | 0.626 | 1 |
| musique | dense | 0.266 | 0.407 | 0.476 | 0.569 | 0.212 | 2 |
| musique | bm25 | 0.227 | 0.342 | 0.393 | 0.467 | 0.134 | 2 |
| musique | hybrid | 0.273 | 0.415 | 0.486 | 0.586 | 0.232 | 2 |

`All-gold@10` is the fraction of questions for which **every** required document is inside the top 10. It is the number that matters for this project: macro recall can look healthy while one bridge paragraph is always missing.

### Diagnostic flags

| Dataset | R@10 (hybrid) | ≥ 0.999 flag | Questions with no gold retrieved (dense/bm25/hybrid) |
|---|---|---|---|
| hotpotqa | 0.927 | False | 5/3/2 |
| 2wiki | 0.813 | False | 0/7/0 |
| musique | 0.586 | False | 33/77/21 |

No run reached 1.000, so the corpus is behaving as a real retrieval problem rather than the degenerate per-question protocol (which is structurally 1.000 and is still what the four legacy pipelines use).

## 8. Index artifacts

| Dataset | Docs | Dense dim | Vectors SHA-256 (first 12) | Built (UTC) |
|---|---|---|---|---|
| hotpotqa | 4928 | 384 | `f6be49e7faba` | 2026-11-19T05:42:15+00:00 |
| 2wiki | 3246 | 384 | `86fbee92bc1a` | 2026-11-19T05:44:59+00:00 |
| musique | 7192 | 384 | `2f02066c8f1e` | 2026-11-19T05:51:03+00:00 |

Loads verify the index's corpus fingerprint against the corpus on disk and refuse a stale pair (`test_index_manager_rejects_stale_corpus`).

## 9. Deviations recorded in this phase

| ID | Deviation | Reason |
|---|---|---|
| D13 | New package is `src/corpus_retrieval/`, not the planned `src/retrieval/` | `src/retrieval.py` is a frozen Scope-A module imported by the legacy pipelines and tests; a package of that name would shadow it. Guarded by `test_experiment_integrity.py::test_legacy_retrieval_module_is_not_shadowed`. |
| DV8 | Pooled candidate corpus is defined by us | The paper never describes its corpus. Ours is closed-book over each dataset's shipped paragraph pool (3.2k–7.2k documents), so recall is **not comparable** with the paper's R@10 and is higher than it. |
| DV3 | `bge-small-en-v1.5` (384-d) instead of `text-embedding-3-small` (1536-d) | `AGENTS.md` requires local embeddings; a hosted embedding API would add unaccounted cost. |
| DV4 | FAISS `IndexFlatIP` instead of Milvus | Hosted vector DBs are forbidden; flat search is exact, so no retrieval-quality difference is introduced. |
| DV5 | Regex tokenizer instead of spaCy | spaCy model/version is unspecified in the paper and would be a new dependency; the tokenizer is the frozen Scope-A one. |
| DV6 | 500 questions per dataset instead of 1,000 | Files are frozen, committed and manifest-verified; resampling would break the historical HotpotQA comparison. |
| DV11 | MuSiQue recall is paragraph-level | MuSiQue ships no sentence-level gold. |

## 10. Unresolved issues and implications for Phase 2

1. **HotpotQA and 2Wiki leave little retrieval headroom.** Hybrid R@10 is 0.927 and 0.813, above the paper's 0.881 / 0.704, because the pooled corpus is smaller than a real Wikipedia index. A PandaRAG retrieval gain will be hard to demonstrate there and easy to dismiss.
2. **MuSiQue is where the contribution is measurable.** Hybrid R@10 is 0.586 while all-gold coverage@10 is only 0.232 — i.e. on roughly 77% of MuSiQue questions at least one required paragraph is missing even at k=10. This is both the motivating signal and the risk: if requirement-level control cannot close that gap, PandaRAG shows nothing.
3. **Hybrid does not dominate dense everywhere.** On 2Wiki, dense R@10 0.823 > hybrid 0.813. The SAC-RAG Fact strategy assumes BM25+dense+rerank helps; the BGE reranker (Phase 2) is untested and may be what makes the difference.
4. **No reranker exists yet.** Every number above is pre-rerank. SAC-RAG's Fact and Reasoning strategies both end in `bge-reranker-base`, so Phase 2 must measure the reranker on its own before any strategy comparison is meaningful.
5. **Local compute is the real budget.** Building 15366 document embeddings took 14m20s of wall clock on CPU (≈0.056 s/document): hotpotqa 4928 docs finished 05:42:15Z — 2wiki 3246 docs finished 05:44:59Z — musique 7192 docs finished 05:51:03Z. The cross-encoder reranker scores query–document pairs and will be slower per pair; the Phase 7 pilot must measure it before the Phase 8 grid is planned.
6. **`Avg Tokens` remains generation-only.** No LLM was called in this phase, so auxiliary cost accounting is still unexercised; the schema for it (D11) is defined but untested end-to-end until Phase 2.

## 11. How to reproduce this report

```bash
ln -sf /usr/bin/python3.14 .venv/bin/python      # only needed if the venv breaks again
python -m pytest tests/ -q                        # 246 tests

# corpora (pure Python, seconds)
python -m src.corpus_retrieval.corpus_cli

# indexes (CPU embedding, ~15 minutes first time)
python -m src.corpus_retrieval.index_cli --force

# sanity metrics (seconds; queries only)
python -m src.corpus_retrieval.benchmark --dataset hotpotqa
python -m src.corpus_retrieval.benchmark --dataset 2wiki
python -m src.corpus_retrieval.benchmark --dataset musique

python -m src.phase1_report                       # regenerate this file
```

