# Engineering log (Scope B)

Chronological record of defects found during implementation, how they were diagnosed,
and what was changed. Results are reported from the **fixed** system; the pre-fix
numbers are kept here so the change is auditable and not silently absorbed.

## 1. Package name collision (`src/retrieval`)

**Symptom.** The target layout asked for `src/retrieval/`, but `src/retrieval.py` is a
frozen Scope-A module imported by the legacy pipelines and their tests. Creating the
package would shadow it.

**Diagnosis.** `import src.retrieval` resolves to the package once `__init__.py` exists.

**Fix.** New package named `src/corpus_retrieval/`; legacy module untouched. A
regression test asserts `src.retrieval.__file__` still points at the `.py` file and
that no `src/retrieval/` directory exists. Recorded as deviation D13.

## 2. Broken virtual environment

**Symptom.** `python -m src.dataset` → `ModuleNotFoundError: No module named 'dotenv'`.

**Diagnosis.** `.venv/bin/python` pointed at `/usr/bin/python`, which had moved to
3.15.2; the 191 packages live under `lib/python3.14/site-packages`.

**Fix.** Repointed the three interpreter symlinks to `/usr/bin/python3.14` (3.14.10,
ABI-compatible with the 3.14.7 the venv was built with). Nothing deleted; the broken
state is preserved in `.venv-backup-*/` with a restore instruction. 172 pre-existing
tests pass again.

## 3. Comparison confound: baselines on a different retrieval problem

**Symptom.** Legacy P1–P3 retrieve from each question's ten distractor paragraphs
(Recall@10 ≡ 1.000). Comparing them with corpus-retrieving P4–P6 would confound the
architecture with the retrieval protocol.

**Fix.** Corpus-matched baselines P1c/P2c/P3c (+ P1n for the paper's exact 10→5
configuration), all on the same pooled corpus. P3c imports the frozen confidence
formula, weights, threshold and normaliser unchanged; only its input changed, hence
"transferred policy".

## 4. Derivation-only requirements (first PandaRAG trace)

**Symptom.** For "which battle was fought first", the planner emitted a third
requirement "compare the two dates", which no document can support. The controller
then spent a retrieval round chasing it and stopped on `no_improvement`.

**Fix.** Planner prompt forbids synthesis-only requirements; comparison questions
produce one requirement per entity. An intermediate "derived requirement" credit
mechanism was implemented and then **removed** — crediting a requirement because its
parent was satisfied would have inflated coverage without evidence. Coverage is now
strictly evidence-based.

## 5. Targeted retrieval repeating the initial query

**Symptom.** The controller re-ran the same query on the same retriever the initial
round had just used, so no new documents could appear and the round always failed.

**Fix.** `ControllerState.initial_modes_by_query` records which retrievers the initial
round used per query; the controller switches to the unused retriever (dense→BM25) and
only expands the pool beyond the initial depth when both are exhausted.

## 6. Cumulative reranker latency in per-question records

**Symptom.** `rerank_latency_seconds` grew monotonically across a run because it read
the reranker's process-wide cumulative timer.

**Fix.** Record the delta around each question.

## 7. Rate limits (HTTP 429)

**Symptom.** 1,200+ questions failed across runs at concurrency 6.

**Diagnosis.** All failures were `RateLimitError`; each was written to the JSONL with
`success: false` and to `results/logs/llm_errors.jsonl` — nothing was dropped.

**Fix.** Re-ran at concurrency 3 with `DEEPSEEK_MAX_RETRIES=6` and a 4 s backoff. The
runner resumes: only failed ids re-run, and the failed rows stay in the file. Analysis
folds the file to one row per question, preferring the successful attempt.

## 8. Cached runs reported as live spend

**Symptom.** P1n replayed P1c's identical prompts (dense top-10, top-5 fed) at zero
cost, yet the meta reported the full cost.

**Fix.** Aggregates now separate `total_api_cost_usd_live` (questions with at least one
live call) from `total_api_cost_usd` (replay value), and every record carries
`from_cache`. Tables quote live cost.

## 9. PandaRAG abstention and context ordering (found in the first full run)

**Symptom.** On the first full run PandaRAG answered `insufficient evidence` on 7.8%
(HotpotQA), 31.4% (2Wiki) and **52.4%** (MuSiQue) of questions, all scored 0, and its
F1 was 0.070 below SAC-RAG Adaptive pooled. Its context also differed from SAC-RAG's
in ordering: only 78–89% of questions had the same first document.

**Diagnosis.** Two independent defects:

1. The insufficiency instruction fired whenever *any* requirement was uncovered.
   Requirement coverage is a conservative, noisy heuristic (`partial` does not count
   as covered), so this vetoed answers the context actually supported.
2. The context builder ordered documents by requirement attachment, demoting
   better-ranked passages below bridge passages.

**Fix.** (a) the insufficiency note now fires only when **no** requirement is
supported — i.e. when there is no usable evidence at all; uncovered requirements are
still recorded in the trace. (b) Selection stays coverage-driven, but the final order
is the reranker's order.

**Disclosure.** These fixes were made after observing the abstention rate, i.e. after
seeing aggregate behaviour. They are structural corrections (a system that refuses
half the questions is not a working system), not threshold tuning, and the affected
artifacts (PandaRAG only) were re-run in full. Baselines P0–P5 were not touched. The
pre-fix numbers are in this log and in `results/pandarag/history/`.

## 10. Missing `context_text` in records

**Symptom.** Faithfulness judging needs the exact context the generator saw; the first
records stored only document ids and titles.

**Fix.** All Scope-B pipelines now store `context_text`. The matrix was restarted from
the beginning rather than mixing schemas.

## 11. P1c hotpotqa missing

**Symptom.** `KeyError: 'P1c'` in the matrix log.

**Diagnosis.** Adding the P0 entry to the `PIPELINES` map accidentally replaced the
P1c entry.

**Fix.** Both entries restored; P1c hotpotqa re-run in full.
