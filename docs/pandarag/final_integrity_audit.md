# Final integrity audit

**Scope:** independent, read-only verification of the Scope-B results, aggregates, statistics,
costs, ablations, manifests and the frozen Scope-A boundary.
**Commit audited:** `fe8ffd0` (working tree uncommitted by design).
**Auditor artefacts:** `results/analysis_scope_b/integrity_audit.json`, produced by
`python -m src.analysis.integrity_audit` (read-only; it never writes to `results/`).

**Method.** Every aggregate was recomputed from the raw per-question JSONL, not from the report or
from `meta.json`. Where the audit needed a baseline the report does not contain, it was recomputed
from records (paired confidence intervals, McNemar counts, ablation CIs). No experiment code and no
result file was modified during the audit. Report-text corrections listed in §11 were applied only
after the checks were recorded.

---

## 1. Verdict summary

| # | Check | Verdict |
|---|---|---|
| 1 | R@10 discrepancy (0.771/0.775 vs 0.716/0.712) | **FAIL** — metric mislabelled (not a computation error); corrected in report |
| 2 | Aggregates vs per-question records (EM, F1, R@10, cost, tokens, faithfulness) | **PASS** — 24/24 pairs and 7/7 faithfulness summaries agree |
| 3 | Pairing, paired bootstrap, McNemar on EM | **PASS** — identical id sets, paired CIs reproduce, McNemar is EM-based |
| 4 | "At parity" wording (equivalence claim) | **FAIL** — equivalence asserted without a pre-registered margin or test; corrected |
| 5 | 2.6× cost comparison | **PASS** (with a labelling warning) — ratio 2.597 reproduced; cost column is cache-normalised |
| 6 | Ablation definitions, sample, paired CIs | **WARNING** — sample and flags verified; report lacked CIs and cost labelling; CIs now supplied |
| 7 | Per-dataset results and hidden failures | **PASS** — all 24 cells present, 500/500 successful, deltas consistent by dataset |
| 8 | Frozen Study-1 artifacts, pre-fix preservation, manifests | **PASS** (with one caveat) — legacy files untouched, `adaptive_rag.py` unchanged, 0 fingerprint mismatches; `results/` is gitignored, so content evidence is mtime/row-count only |
| 9 | Test suite and git state | **PASS** — 300 passed; 2 modified files, 30 untracked, nothing committed |
| 10 | P2c re-run at fusion depth 10 (follow-up) | **PASS** — new `P2c10`, all 500×3 answered; P6−P2c10 ΔF1 +0.045 vs +0.050 before; original archived byte-for-byte |
| 11 | Standardised Recall@5/@8/@10 + stopping depth (follow-up) | **PASS** — undefined cells are `–`, not zero; `R@stop` reported per system |
| 12 | HTTP-429 failure accounting (follow-up) | **PASS** — 13,980 rows, 13,599 successes, 381 superseded attempts, 0 terminal failures; denominators 500 |
| 13 | A3 vs A4 identity (follow-up) | **PASS (explained) / WARNING (design)** — components active (920 verdicts, $0.186) but the evaluation pool equals the context window, so 299/300 contexts are byte-identical |

Two additional findings not in the original request:

| # | Finding | Verdict |
|---|---|---|
| A | P2c "Hybrid RAG" fuses **truncated top-5** source lists, unlike P3c/P4/P5/P6 and the Phase-1 benchmark which fuse top-10 then truncate | **WARNING** — definitional inconsistency in a baseline |
| B | P4/P5/P6 retrieval depth is 9.9/9.9/10.0 because the Definition strategy stops at 8 documents | **WARNING** — their "R@10" is R@8 for those cells (23/14/7 questions) |

---

## 2. Check 1 — the R@10 discrepancy

**Reported:** Phase-1 benchmark macro R@10 = 0.7712 (dense) / 0.7754 (hybrid). Final table: P1c 0.716,
P2c 0.712.

**Recomputation from the raw records** (`results/*/*.jsonl`, field `retrieved_doc_ids`, gold from
`gold_document_ids`):

| System | mean retrieved docs | stored `recall@10` macro | recall at actual depth | questions with depth < 10 |
|---|---:|---:|---:|---:|
| P1c | 5.00 | 0.7155 | 0.7155 | 500 / 500 |
| P2c | 5.00 | 0.7124 | 0.7124 | 500 / 500 |
| P3c | 10.00 | 0.7754 | 0.7754 | 0 |
| P1n | 10.00 | 0.7712 | 0.7712 | 0 |
| P4 | 9.91 | 0.9390 / 0.8190 / 0.6027 | same | 23 / 0 / 14 |
| P5 | 9.91 | same as P4 | same | 23 / 0 / 14 |
| P6 | 9.93 | 0.9390 / 0.8190 / 0.6038 | same | 20 / 0 / 7 |

Phase-1 benchmark macro values (independent, top-10 retrieval, `results/sac_rag/retrieval/*.meta.json`):

```
dense  R@1 0.4017  R@3 0.6558  R@5 0.7155  R@10 0.7712
bm25   R@1 0.3499  R@3 0.5348  R@5 0.6071  R@10 0.6934
hybrid R@1 0.3946  R@3 0.6298  R@5 0.7045  R@10 0.7754
```

**Explanation (exact).**

1. P1c and P2c retrieve and store **five** documents. `recall_at_k(..., k=10)` therefore evaluates a
   five-document list, i.e. it returns Recall@5. P1c's 0.7155 is *identically* Phase-1 dense R@5
   (0.7155). The number is correct; the **label** is wrong.
2. P1n (dense top-10) reproduces Phase-1 dense R@10 exactly: 0.7712. P3c (hybrid top-10) reproduces
   Phase-1 hybrid R@10 exactly: 0.7754. That is the independent confirmation that the retrievers
   themselves are unchanged between Phase 1 and the final matrix.
3. P2c's 0.7124 is **not** Phase-1 hybrid R@5 (0.7045). Cause: P2c calls the fusion with source lists
   already truncated to 5, so the candidate union is the top-5 of each source rather than the top-10.
   Per-question comparison against the Phase-1 benchmark for HotpotQA: **162/500 identical, 338/500
   different** top-5 lists. P1c, by contrast, is identical to the benchmark dense top-5 in **500/500**
   questions. See finding A in §9.
4. P4/P5/P6 cells are a mixture of R@8 (23 HotpotQA Definition questions) and R@10 (the rest), which is
   why their stored macro (0.7869) differs from a strict R@10 over a full ten-document list (0.7817).

**Root cause.** The record builder computes `recall@k` over whatever the pipeline retrieved; the report
then printed a single column headed `R@10` for systems that retrieved 5, 8 or 10 documents. No
aggregate arithmetic is wrong — the mismatch is entirely explained by retrieval depth and a mislabelled
column.

**Correction applied:** the results tables now carry the mean retrieval depth and are retitled
`R@k (depth)`, with a footnote stating that `R@10` equals `R@5`/`R@8` when a system retrieves fewer
than ten documents, and that P2c's fusion depth differs from the other hybrid systems. The frozen
numbers were not changed.

---

## 3. Check 2 — aggregates versus records

`check_aggregates` recomputed, for all 24 `(system, dataset)` pairs, from JSONL only:

- `n`, `n_success`, mean `exact_match`, mean `f1`, mean `recall@10`, mean `input_tokens`, and
  `total_api_cost_usd_live` (sum over records with `live_llm_calls > 0`).

Compared against both `meta.json` and the report's tables: **0 disagreements** at tolerance 5e-4
(report tables are rounded to 3 dp).

Faithfulness: means recomputed from the 1,165 per-answer judge entries in
`results/analysis_scope_b/faithfulness.jsonl` versus `faithfulness_summary.json` for 7 systems:
**0 disagreements** (n_scored, faithfulness, hallucination).

Coverage completeness: for every one of the 24 runs, **all 500 question ids have a successful
record**. 374 failed attempts (HTTP 429) remain in the files as `success: false` rows; they are
superseded, not deleted, and the analysis folds to one row per question preferring the success.

**Verdict: PASS.**

---

## 4. Check 3 — pairing, bootstrap, McNemar

**Pairing.** For all 15 primary comparisons × 3 datasets (`P6` against `P4`, `P5`, `P1c`, `P2c`, `P3c`):
left and right id sets are **identical** (`n_left = n_right = n_shared = 500`), with **zero duplicate
ids** after folding. Comparisons therefore use identical question ids.

**Bootstrap is paired.** `stats.compare_pair` builds a per-question difference vector and resamples
question indices, applying each draw to both systems. Evidence that this matters and that it is what
was run — P6 vs P4, HotpotQA:

```
paired   CI95 = [-0.0017, +0.0080]
unpaired CI95 = [-0.0448, +0.0501]   (each system resampled independently)
```

The reported CI matches the paired recomputation to rounding. The unpaired interval is ~6× wider,
so the reported intervals cannot have come from an unpaired procedure. The audit was then extended to
**all 15 primary comparisons**: every reported mean difference, CI bound and McNemar p-value is
reproduced exactly (0 mismatches at 1e-6). One caveat discovered while doing this: the audit's first
pass used the 1949th bootstrap order statistic where the implementation uses the 1950th, which moved a
CI bound by 3e-5 — the correct reading is that the report is reproducible to the order statistic, not
that the report was wrong.

Pooled intervals use `stratified_bootstrap_ci`, which resamples within each dataset and pools with
fixed per-dataset weights — pairing preserved inside each stratum.

**McNemar is applied to Exact Match.** In `compare_pair`, `em_pairs` is built from
`exact_match` booleans and the McNemar counts come from those pairs, while the F1 difference is handled
by the bootstrap. Contrast that proves the distinction: for P6 vs P1c on HotpotQA, McNemar on EM gives
p = 0.0166, whereas the same test on an `F1 > 0.5` binarisation gives p = 0.0095 — different numbers,
and the report quotes the EM one. Counts (`only_left_correct`, `only_right_correct`) are stored in the
audit JSON.

**Verdict: PASS.** One presentation caveat: the F1 rows in the statistics table carry the EM-based
McNemar p-value, because it is labelled `McNemar p` in its own column; the metric and the test are not
the same quantity and the table now says so in a footnote.

---

## 5. Check 4 — the "at parity" claim

Reported sentence: *"answer quality is at parity, at higher cost"* with ΔF1 = −0.000,
95% CI [−0.003, +0.003], McNemar p = 1.0.

**Problem.** A non-significant difference is not equivalence. The claim as written asserts
equivalence without a pre-registered margin or an equivalence test, which violates the project's own
rule against claiming more than the statistics support.

**Audit-side equivalence analysis** (post hoc, pooled 1,500 paired questions):

```
mean ΔF1 = -0.000224
CI95     = [-0.003305, +0.002747]
margin   = ±0.02 F1 (chosen by the auditor AFTER seeing the data)
CI inside margin: yes
```

At a ±0.02 margin the interval supports equivalence; but the margin was not pre-registered and no TOST
test was pre-specified, so this is **indicative, not a valid equivalence claim**.

**Correction applied:** the wording now reads *"no detectable difference (ΔF1 −0.000, 95% CI
[−0.003, +0.003]); failing to reject the null is not evidence of equivalence — at a post-hoc ±0.02 F1
margin the interval is consistent with equivalence, but the margin was not pre-registered"*. The
abstract, discussion and results text carry the same qualification.

**Verdict: FAIL (wording), corrected.**

---

## 6. Check 5 — the 2.6× cost comparison

Recomputed from `results/ablations/A{0,4}/*.jsonl` (300 questions each, identical ids):

| Variant | mean LLM calls / q | cost / q (USD) | components (total USD over 300 q) |
|---|---:|---:|---|
| A0 (SAC-RAG Adaptive) | 1.153 | 0.00038096 | generation 0.111418, decomposition 0.002871 |
| A4 (PandaRAG) | 2.853 | 0.00098924 | evidence_evaluator 0.186331, generation 0.104372, planner 0.006070 |

Ratio: **2.597×** — the reported 2.6× is reproduced.

**What is included.** All LLM calls, using the provider's reported token usage and the frozen pricing
table: generator, decomposition (A0 only, for Reasoning questions), planner (A4 only, Reasoning
questions), and the evidence evaluator (A4 only). Local retrieval, embedding and reranking cost nothing
in USD and are reported separately as latency/execution counts.

**Warning.** The cost columns in the ablation table are **cache-normalised replay values**
(`total_api_cost_usd_live = 0` for both variants because every call was served from the disk cache on
the run that produced the final artifacts). They represent what the run would cost live and are
comparable between variants, but they are not money spent. The report now labels the column
`$/q (replay value)`.

A4's *generation* cost is slightly **lower** than A0's ($0.104 vs $0.111) because its context is
smaller; the entire ratio comes from the evaluator call, which is 63% of A4's cost. That is worth
stating explicitly, and the report now does.

**Verdict: PASS with a labelling correction.**

---

## 7. Check 6 — ablations

**Definitions.** The six variants match `ABLATIONS` in `src/pipelines/pandarag/pipeline.py`:
A1 planner only; A2 +evaluator; A3 +controller with standard top-K context; A4 +evidence-preserving
context (= PandaRAG); A5 +compression. A0 is SAC-RAG Adaptive run in-process through the same runner.

**Sample.** 100 questions per dataset, stratified by frozen semantic type, seed 42. The question-id sets
are **identical across all six variants on all three datasets** (verified set equality), and each
variant has n = 100 per dataset (300 total), all successful.

**Paired comparison A4 vs A0** (computed by the audit; not previously in the report):

| Dataset | n | mean ΔF1 (A4−A0) | paired CI95 | McNemar p (EM) | only A4 correct | only A0 correct |
|---|---:|---:|---|---:|---:|---:|
| HotpotQA | 100 | +0.0167 | [0.0000, +0.0433] | 1.000 | 1 | 0 |
| 2Wiki | 100 | 0.0000 | [0.0000, 0.0000] | 1.000 | 0 | 0 |
| MuSiQue | 100 | +0.0025 | [−0.0034, +0.0107] | 1.000 | 0 | 0 |
| **Pooled** | **300** | **+0.0064** | **[−0.0003, +0.0162]** | **1.000** | 1 | 0 |

The pooled interval includes zero; Exact Match is identical between A0 and A4 (a single discordant
question). So the ablation supports "no detectable quality difference", not "improvement", on this
sample.

**Warnings.** (a) The report's ablation table previously reported point estimates only — no paired CI
and no test; the audit's table above is now included in the report. (b) A3 and A4 produce identical
results (0.433 / 0.570), which is itself informative: with a bounded context window the
evidence-preserving builder rarely changes the final prompt. (c) The ablation sample (300 questions) is
labelled reduced-sample everywhere and is never pooled with the 500-question results.

**Verdict: WARNING (now corrected by adding CIs and tests).**

---

## 8. Check 7 — per-dataset visibility

All 24 cells (8 systems × 3 datasets) exist and are reported individually; there is no system for which
only a pooled number is shown.

The two claims that matter, per dataset (paired, n = 500 each):

| Comparison | HotpotQA ΔF1 | 2Wiki ΔF1 | MuSiQue ΔF1 |
|---|---:|---:|---:|
| P6 − P4 | +0.0025 | −0.0020 | −0.0011 |
| P6 − P1c | +0.0329 | +0.0306 | +0.0402 |

Neither effect is produced by one dataset: P6-vs-P4 is within ±0.003 everywhere, and P6-vs-P1c is
consistently positive. Dataset-specific behaviour that the pooled numbers *would* hide is reported in
the per-dataset table rather than only pooled — notably P5 (compression) is −0.024 F1 on 2Wiki while
being +0.008 on HotpotQA, and MuSiQue is where all systems collapse (P4 F1 0.372 vs HotpotQA 0.756).

**Verdict: PASS.**

---

## 9. Additional findings

### A. P2c is a different hybrid from every other hybrid (WARNING)

`P2c` calls `service.search(question, "hybrid", 5)`, which fuses source lists already truncated to
five documents. `P3c`, `P4`, `P5`, `P6` and the Phase-1 benchmark retrieve ten per source, fuse, then
truncate. Consequence: 338/500 HotpotQA questions have a different top-5 list from the benchmark
hybrid. P2c is therefore *not* the same system as "Hybrid RAG (dense+BM25 RRF)" used elsewhere.

Direction of bias is not obvious (the smaller union both adds and removes candidates), so it may
flatter or penalise P6's comparison against P2c. **Recommended correction:** re-run P2c as
fusion-at-depth-10-then-truncate (a new results file; not an overwrite), or report it as
"hybrid, fusion depth 5". The report now carries the disclosure; the re-run was **not** performed
during this audit.

### B. Retrieval-depth mixture inside P4–P6 (WARNING)

The Definition strategy stops at 8 documents, so those cells contribute R@8 rather than R@10. Magnitude
is small (macro effect 0.005) but it means the column is not a uniform metric. Covered by the depth
column added in the correction.

### C. Legacy-artifact evidence is weaker than "unchanged" implies (WARNING)

`results/` is listed in `.gitignore`, so git cannot prove the Study-1 files were not modified. The
audit verified: all four files still contain exactly 500 rows; modification times are 2026-09-20
(before Scope B began on 2026-11-19); and `src/pipelines/adaptive_rag.py` is byte-identical to
`HEAD` (`git diff --quiet` exits 0). SHA-256 of each legacy file is now recorded in
`results/analysis_scope_b/integrity_audit.json` so future audits have a content baseline:

```
no_rag       df960364bd1f…   500 rows   2026-09-20T00:11:55Z
standard_rag 295052d4cbce…   500 rows   2026-09-20T00:34:52Z
hybrid_rag   ac8d76b62a3e…   500 rows   2026-09-20T01:00:29Z
adaptive_rag 78a421782f75…   500 rows   2026-09-20T01:44:37Z
```

**Verdict: PASS, with the evidence caveat recorded.**

### D. Pre-fix PandaRAG retained (PASS)

Four archives exist and are non-empty: `results/pandarag/history` (3 JSONL), `history2` (3),
`history3` (1), `history4` (3) — covering the pre-fix, ordering-only, minimal-context and
note-v2 variants described in `docs/pandarag/engineering_log.md`.

### E. Manifests (PASS)

All 24 metas carry the correct corpus fingerprint (HotpotQA `2da92dd60bdb…`, 2Wiki `ad500ff361d7…`,
MuSiQue `7c485109ff42…`) and the correct dataset SHA-256 from `data/dataset_manifest.json`.
**0 mismatches.** `docs/pandarag/experiment_manifest.md` lists every result file with its config,
prompt and corpus fingerprints. Config fingerprints in the metas match the current files in `configs/`.

---

## 10. Check 9 — tests and working tree

```
$ python -m pytest tests/ -q
300 passed in 20.15s        (returncode 0)
```

Git state: `M .gitignore`, `M AGENTS.md`, 30 untracked paths (`configs/`, `docs/figures/fig1…fig10`,
`docs/pandarag/`, `src/analysis/`, `src/corpus_retrieval/`, `src/datasets/`, `src/experiments/`,
`src/pipelines/{shared,sac_rag,pandarag}`, `src/experiment_config.py`, `src/phase1_report.py`, five
test files). **Nothing was committed, deleted or overwritten during this audit.**

---

## 11. Report corrections applied

| Correction | Where | Why |
|---|---|---|
| Retrieval column renamed to `R@k (depth)` and a mean-depth column added | results tables, `final_results.md` | 0.771 vs 0.716 is a depth difference, not a retriever difference (§2) |
| Footnote: `R@10` = `R@5`/`R@8` when fewer than ten documents are retrieved; P2c fuses at depth 5 | results section | same |
| "At parity" replaced by "no detectable difference" + explicit post-hoc-margin caveat | abstract, discussion, results | equivalence needs a pre-registered margin and TOST (§5) |
| Ablation table gains paired CI and McNemar columns; `$/q` labelled replay value | ablation section | §7 |
| P2c fusion-depth disclosure added to methodology and limitations | `final_methodology.md`, `limitations.md` | finding A |
| Cost section states that the evaluator is 63% of A4's cost and that A4's generation cost is *lower* than A0's | discussion | §6 |
| Faithfulness table states the judged sample size per system (n ≈ 170, stratified 60/dataset) | grounding section | avoids implying full-sample judging |

Corrections touch **report text and reporting code only** (`src/analysis/report.py`,
`src/analysis/tables.py` labels). No result file, experiment module, pipeline, config or dataset was
modified.

---

## 12. What this audit did not do

- It did not re-run any experiment, and it did not re-run P2c with corrected fusion depth.
- It did not verify the base paper's unpublished corpus or prompts (impossible; documented as
  deviations DV1–DV12).
- It did not human-label evidence-support judgements, so the evaluator's accuracy remains
  unmeasured at scale (limitation 8).
- It did not test generator stochasticity (single run per question, temperature 0).
- It did not establish a pre-audit content baseline for `results/` because none existed in version
  control; it created one for future audits (§9C).

---

## 13. Follow-up round: P2c correction, depth metrics, failures, ablation inertness

Requested after the first audit. Nothing here overwrites a result: the depth-5 P2c artifacts stay in
place and were additionally snapshotted byte-for-byte to
`results/archive/baseline_hybrid_corpus_depth5/` (`ARCHIVE_MANIFEST.json` carries the SHA-256s).

### 13.1 P2c re-run at fusion depth 10 — **PASS**

New system `P2c10` (`results/baseline_hybrid_corpus_depth10/`): retrieve top-10 per source, fuse at
RRF depth 10, feed the top 5. All 500 questions per dataset, 0 terminal failures.

| Dataset | n | EM | F1 | R@10 | Phase-1 hybrid R@10 |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 500 | 0.558 | 0.713 | 0.9270 | 0.9270 |
| 2Wiki | 500 | 0.490 | 0.559 | 0.8130 | 0.8130 |
| MuSiQue | 500 | 0.218 | 0.341 | 0.5862 | 0.5862 |

The retrieval numbers now match the Phase-1 benchmark exactly on all three datasets, which is the
expected check: `P2c10` and `P3c` share identical retrieval and differ only in how many documents they
feed. The corrected baseline is **stronger** than the depth-5 one (pooled EM 0.422 vs 0.416, F1 0.537
vs 0.532), so the original comparison was marginally favourable to PandaRAG:

| Comparison | Pooled ΔF1 | 95% CI | Holm p | n |
|---|---:|---|---:|---:|
| P6 vs P2c (depth 5, original) | +0.0501 | [+0.035, +0.065] | 0.0000 | 1500 |
| P6 vs P2c10 (depth 10, corrected) | +0.0451 | [+0.030, +0.060] | 0.0000 | 1500 |

Direction confirmed: the original P2c slightly overstated P6's advantage. The conclusion (P6 beats
both hybrid baselines, and P6 is not distinguishable from SAC-RAG Adaptive) is unchanged.

### 13.2 Standardised Recall@5 / @8 / @10 — **PASS (implemented in the report)**

`src/analysis/final_checks.py::depth_metrics` computes recall only where the depth is defined, and
separately reports recall at each pipeline's actual stopping depth. Macro over the three datasets:

| System | mean depth | R@5 | R@8 | R@10 | all-gold@10 | R@stop |
|---|---:|---:|---:|---:|---:|---:|
| P1c | 5.00 | 0.7155 | – | – | – | 0.7155 |
| P2c | 5.00 | 0.7124 | – | – | – | 0.7124 |
| P2c10 | 10.00 | 0.7045 | 0.7587 | 0.7754 | 0.5720 | 0.7754 |
| P3c | 10.00 | 0.7045 | 0.7587 | 0.7754 | 0.5720 | 0.7754 |
| P1n | 10.00 | 0.7155 | 0.7555 | 0.7712 | 0.5707 | 0.7712 |
| P4 | 9.95 | 0.7265 | 0.7762 | 0.7876 | 0.5878 | 0.7869 |
| P5 | 9.95 | 0.7265 | 0.7762 | 0.7876 | 0.5878 | 0.7869 |
| P6 | 9.97 | 0.7267 | 0.7766 | 0.7872 | 0.5862 | 0.7873 |

`–` means undefined (the system never retrieved that many documents), not zero. R@5 is the only depth
at which every retrieving system is defined; R@stop is the fair same-depth comparison. The report now
prints this table with that caveat, and figures 5–6 plot fixed-depth curves with gaps where undefined.

### 13.3 HTTP-429 accounting — **PASS**, apparent contradiction resolved

Across the 27 system–dataset runs (24 from the matrix plus 3 for `P2c10`):

```
rows written            13,980
successful rows         13,599   (= 13,500 canonical + 99 duplicate successes, see §15.1)
superseded failures        381   (all HTTP-429, plus a handful of timeouts)
canonical records       13,500   (27 cells × 500 unique question ids)
questions with a success   500 in every run
terminal failures            0
EM/F1 denominator          500 per cell
retrieval denominator      500 per cell
```

The contradiction is terminological: `success: false` rows are **superseded attempts**, not unanswered
questions. A rate-limited attempt is written to the JSONL, the runner retries the question on the next
pass, and the successful row is appended. Analysis folds the file to one row per question id, preferring
the success (`loaders.load`), so a failure can never enter an aggregate as a zero. One torn line from a
killed process existed at `results/baseline_hybrid_corpus_depth10/2wiki.jsonl:501`; it was moved
(as-is) to `TORN_LINE_2wiki.txt` next to the file and the question was re-run, leaving 500/500.

Retry accounting from the process-global log (`results/logs/llm_errors.jsonl`):

```
entries 7,284 · status 429 = 7,253 · 400 = 1 · 502 = 1 · none = 29
attempt histogram: 1 → 2,095 · 2 → 1,848 · 3 → 1,564 · 4 → 1,244 · 5 → 533
entries flagged will_retry: 6,751
```

The log carries model, base URL, request hash, attempt number and status — **no pipeline or question
id** — so retries cannot be attributed per system from it. That is a reporting limitation, not a
correctness issue: per-run superseded counts come from the result files themselves.

Faithfulness denominators are separate and smaller by design (stratified sample):

| System | sampled | scored | no checkable claims | parse failures |
|---|---:|---:|---:|---:|
| P1c / P1n | 180 | 167 | 13 | 0 |
| P2c | 180 | 169 | 11 | 0 |
| P3c | 180 | 168 | 12 | 0 |
| P4 / P6 | 180 | 172 | 8 | 0 |
| P5 | 180 | 172 | 8 | 0 |

### 13.4 Why A3 and A4 are identical — **PASS (explained), WARNING (design)**

Context-text comparison across the ablation artifacts:

| Pair | Questions | identical context | identical prediction |
|---|---:|---:|---:|
| A1 vs A2 | 300 | 300 | 300 |
| A2 vs A3 | 300 | 300 | 300 |
| A3 vs A4 | 300 | 299 | 300 |

The components are **active**: A2/A3/A4 each produce 920 evaluator verdicts in the first round and
$0.186 of evaluator calls over 300 questions. The inertness is structural, not a wiring bug:

1. The evaluator only sees `admitted[:10]` (`EVALUATION_POOL = 10`).
2. The context builder fills a window of `base_top_k = admission` = 10 and then re-sorts by reranker
   rank.
3. Therefore every passage the builder "protects" is already inside the window, and the final order
   is the reranker's order — which is exactly A3's context.

Measured: of 468 protected requirement–document links in A4, only **1** fell outside the top-10
window. The single context difference in A3 vs A4 did not change the prediction.

**Consequence for the paper:** the A3-vs-A4 ablation does not test evidence-preserving selection as
intended, because the experiment never gives it a candidate beyond the window. To exercise it, the
evaluator would have to see a wider pool than the generator (e.g. evaluate 20, build 10). That is a
design change, not a bug fix, and was **not** implemented in this round.

---

## 14. Freeze recommendation

**Recommendation: freeze the artifact set as a negative-result study, not as evidence that
PandaRAG improves multi-hop QA.** Everything needed to defend that framing is present, reproducible
and now internally consistent; what is missing is a positive effect, and no further tuning within this
design will produce one honestly.

### 14.1 What is safe to freeze

| Item | Evidence | Status |
|---|---|---|
| 27 runs (9 systems × 3 datasets × 500 questions) | `results/*/`; 13,599 successful rows, 0 terminal failures | complete |
| Reviewable per-question records | every record carries dataset/corpus/config/prompt fingerprints | complete |
| Frozen Phase-1 corpus + indexes | `docs/pandarag/phase1_artifacts.json`; corpus SHAs match all 27 metas | complete |
| Statistical layer | 15 paired comparisons reproduce exactly; paired bootstrap; exact McNemar on EM; Holm | complete |
| Ablations A0–A5 | identical 100-question stratified sample; paired CIs supplied | complete |
| Grounding | 1,165 judged answers, frozen `faithfulness-judge/1.0` | complete (stratified sample) |
| Error taxonomy + traces | `results/analysis_scope_b/error_analysis.json` | complete |
| Frozen Scope-A experiment | 4 files, 500 rows each, mtime 2026-09-20, `adaptive_rag.py` identical to HEAD | preserved |
| Pre-fix PandaRAG variants | `results/pandarag/history{,2,3,4}/` | preserved |
| Corrected + original P2c | `results/baseline_hybrid_corpus_depth10/` and archived depth-5 snapshot | preserved |

### 14.2 What the write-up must say (non-negotiable)

1. **No positive quality effect.** P6 vs SAC-RAG Adaptive: pooled ΔF1 −0.000 (95% CI [−0.003, +0.003]),
   McNemar p = 1.0. "No detectable difference", never "equivalent" without a pre-registered margin.
2. **Higher cost.** 2.60× per question (ablation), 63% of it the evidence evaluator.
3. **The proposed mechanism did not convert gaps, as measured by the evaluator.** Targeted retrieval
   ran on 41–78% of questions and converted an uncovered requirement to supported in 0–1% of rounds.
   This is evaluator-measured; without human validation of the evaluator it bounds the controller's
   observed effect and is not proof that no further evidence existed.
4. **A3 ≡ A4 is a design artefact**, not a result: evaluation pool = context window, so the
   evidence-preserving builder has nothing to displace (299/300 identical contexts, 1/468 protected
   links outside the window).
5. **Retrieval metrics are pooled-corpus and depth-dependent.** Recall is not comparable with the
   base paper, and `R@5` is the only depth defined for every system.
6. **The 381 `success:false` rows are superseded attempts**, not unanswered questions.
7. **SAC-RAG is a best-effort reimplementation** (DV1–DV12): its corpus, classifier prompt,
   decomposition prompt and compression prompt are unpublished; the paper contradicts itself on the
   compressor model.

### 14.3 Changes that would be needed to test the hypothesis properly (not implemented)

These are research-design options, recorded so the next iteration does not repeat the same blind
alleys. None is a bug fix, and none was implemented here:

- **Widen the evaluator's pool beyond the generator's window** (e.g. evaluate 20 candidates, build 10).
  Until that changes, evidence-preserving selection cannot alter a prompt, and A3/A4 cannot differ.
- **Retrieve for a gap from a different source or with an expanded query**, rather than re-running the
  same query on the other retriever — measured recovery success is ~1%, so the current action set is
  effectively a no-op.
- **Test on a corpus where completeness is actually reachable**; on MuSiQue only 23% of questions have
  all gold documents in the top 10, and the missing ones are not in the candidate pool at all.
- **Pre-register an equivalence margin** if an "at least as good, cheaper" claim is the goal, and
  pre-register the non-inferiority test accordingly.

### 14.4 Open items that do not block freezing

- The evaluator has no human-labelled support audit at scale (limitation 8); its accuracy is
  unmeasured, which is one more reason not to build claims on it.
- Latency numbers are single-machine and API-dependent.
- Generator stochasticity is unquantified (temperature 0, one sample per question).
- Retry counts cannot be attributed per system because the error log lacks a pipeline id.

---

## 15. Final reporting checks

Three clarifications requested before freezing. No experiment configuration, result file or
fingerprint was changed.

### 15.1 Record reconciliation — **PASS** (99 duplicates explained and confined)

27 runs × 500 questions = **13,500 canonical records**. The audit's 13,599 "successful rows" is a row
count, not a record count. The arithmetic closes exactly:

```
13,500  canonical records  (one per question id per cell)
+   99  duplicate successful rows   (P2c10 / 2Wiki only)
+  381  superseded HTTP-429 attempts
= 13,980 rows written
```

Per-cell verification: **all 27 cells have exactly 500 unique question ids, and 500 of them have a
successful record.** Terminal failures: 0. Every metric denominator is 500 after deduplication; the
first version of this table counted successful *rows* for the denominator and therefore displayed 599
for `P2c10/2Wiki`. That was a reporting bug in the audit table, not in any result — the aggregates
were always computed over deduplicated records (verified independently: 0 disagreements). It is fixed
and all 27 cells now read 500.

The 99 duplicates are confined to `results/baseline_hybrid_corpus_depth10/2wiki.jsonl`: 500 unique
successful ids, 599 successful rows. Timestamps of the duplicated ids are ~20 s apart (e.g.
`00:01:05` and `00:01:28`), i.e. **two processes appended to the same file concurrently** after an
apparent process death during the re-run. This is an operational artefact of that one cell, not a
runner bug under single-writer use, and it has no effect on any result: the analysis folds each file
to one row per question id, preferring a success, so every aggregate is computed over exactly 500
questions. The duplicate rows are **preserved, not deleted**; the report labels them explicitly.

**Recommendation for the reproduction instructions:** never run two processes against the same
system/dataset output file. The runner resumes correctly with one writer.

### 15.2 Candidate recall vs context recall — **PASS** (three distinct columns now reported)

The report's retrieval table previously implied that "R@10" was evidence the generator consumed. For
two systems it was not:

| System | mean depth | candidate R@5 | candidate R@10 | R@stop | **R@ctx** |
|---|---:|---:|---:|---:|---:|
| P1c | 5.00 | 0.7155 | – | 0.7155 | 0.7155 |
| P2c | 5.00 | 0.7124 | – | 0.7124 | 0.7124 |
| **P2c10** | 10.00 | 0.7045 | **0.7754** | 0.7754 | **0.7045** |
| P3c | 10.00 | 0.7045 | **0.7754** | 0.7754 | **0.6543** |
| **P1n** | 10.00 | 0.7155 | **0.7712** | 0.7712 | **0.7155** |
| P4 | 9.95 | 0.7265 | 0.7876 | 0.7869 | 0.7869 |
| P5 | 9.95 | 0.7265 | 0.7876 | 0.7869 | **0.7600** |
| P6 | 9.97 | 0.7267 | 0.7872 | 0.7873 | 0.7877 |

Definitions now stated in the report:

- **candidate R@k** — what retrieval returned; undefined (`–`) below depth k.
- **R@stop** — recall over the ranked list the retriever produced.
- **R@ctx** — recall over the documents actually placed in the generator prompt. Only this bounds what
  the model could use.

Consequences that were previously under-stated and are now explicit:

- **P2c10** finds 0.775 at depth 10 but feeds **five** documents, so its generator saw 0.705. Its
  candidate R@10 must never be cited as evidence consumed.
- **P1n** has the same shape: 0.771 found, 0.716 fed.
- **P3c** retrieves 10 (0.775) but adaptively feeds 3–5, so context recall is **0.654** — its
  transferred K policy is where evidence is lost, not its retrieval.
- **P5** drops from 0.787 retrieved to **0.760** after compression.
- **P4/P6** feed every document they rank, so `R@stop = R@ctx`; P6 is marginally higher (0.7877 vs
  0.7869) because protected passages are added.

The figure that plots Recall@K is retitled *"Candidate document Recall@K at fixed depth (not context
recall)"*, and the report table carries all three columns with the definitions above.

### 15.3 Mechanism conclusion tempered — **PASS**

The 0–1% recovery rate was measured by the LLM evidence evaluator, which has **no human-labelled
validation at scale** (limitation 8). The report, discussion and audit now say:

- the rate is **evaluator-measured**;
- it bounds the controller's observed effect *under this evaluator*, and does not establish that the
  corpus contains no further evidence;
- the retrieval-side claim is stated separately and does not depend on the evaluator: on MuSiQue only
  23% of questions have all gold documents inside the top 10, and the missing documents never entered
  the candidate pool (gold-label measurement, limitation 7 applies to nothing here).

The corresponding limitation is now item 21 in `limitations.md`, and the audit's freeze recommendation
(§14.2.3) carries the same wording.

---

## 16. Figure re-layout (reporting only)

The ten Scope-B figures were re-rendered because several had overlapping text: axis labels ran into
each other, the nine-entry legends overflowed the canvas, and the scatter labels collided.

**What changed.** Layout only; every plotted value is unchanged and still comes from the same tables:

- Per-system comparisons are now **horizontal** charts, so full system names sit on the y-axis
  (`fig1`, `fig3`, `fig4`, `fig9`, `fig10`). No rotated or truncated labels remain.
- Multi-series charts keep dataset names on the x-axis and place full system names in a **wrapped
  legend** that is sized to the canvas (`fig2`, `fig5`, `fig6`).
- The cost–quality figure became a ranked horizontal log-scale chart with `F1` annotated per system,
  because nine long labels on a scatter cannot be placed without collisions (`fig3`).
- Axis bounds, tick steps and label gutters are now computed from the data and the longest label, so
  nothing is clipped or over-padded.
- Ablation labels spell out the component (`A4 + evidence-preserving context`) instead of a bare code
  (`fig10`).

**Verification.** All ten SVGs parse as valid XML; each was rasterised and inspected for overlap.
The report's Figures section lists the files and captions and no longer carries an ID-to-name mapping,
because the figures now use names.

**Unchanged:** scores, denominators, sample sizes, statistical results, result files, fingerprints.
