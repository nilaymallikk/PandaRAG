# Methodology



## Systems

| ID | System | Retrieval | Generator calls | Notes |
|---|---|---|---|---|
| P0 | No-RAG | none | 1 | closed-book baseline |
| P1c | Dense RAG (corpus) | dense top-5 over the pooled corpus | 1 | corpus-matched baseline |
| P2c | Hybrid RAG (corpus) | dense+BM25 RRF, sources truncated to 5 | 1 | corpus-matched baseline; **fusion depth differs from P3c/P4/P5/P6** (see limitations) |
| P3c | Adaptive RAG (transferred) | frozen K=3/K=5 policy applied to corpus retrieval | 1 | formula imported unmodified |
| P1n | Naive RAG (paper 10→5) | dense top-10 retrieved, top-5 fed | 1 | SAC-RAG Table 3 configuration |
| P4 | SAC-RAG Adaptive | type-specific retrieval + BGE rerank | 1 (+1 decompose for Reasoning) | best-effort reimplementation |
| P5 | Full SAC-RAG | P4 + LLM extractive compression | 2 | same pipeline, compressor enabled |
| P6 | PandaRAG | requirement plan → type-specific retrieval → per-requirement evidence evaluation → gap-directed retrieval → evidence-preserving context | 2–4 | proposed system |

## Retrieval protocol

Pooled candidate corpus per dataset split: the union of every paragraph shipped with
every question, deduplicated by `sha1(dataset, split, title, text)`. Documents used
as gold: 994 (HotpotQA), 990 (2Wiki), 912 (MuSiQue). Recall@10 is computed against
this corpus and is therefore **not comparable** with the base paper's
Wikipedia-scale figures.

## PandaRAG details

- **Planner**: 1 requirement for Fact/Definition without an LLM call; up to 4
  requirements for Reasoning via a structured prompt with dependency validation
  (unknown ids, self-dependency, cycles, cap overflow all handled explicitly).
- **Evidence evaluator**: one batched LLM call per round over ≤10 candidate
  documents, statuses `supported | partial | insufficient | contradicted | unknown`,
  strict id whitelisting, unknown ids dropped, failures degrade to `unknown`.
- **Coverage**: per requirement; `partial` counts 0.5; nothing is credited across
  requirements.
- **Controller**: rule-based. Targets the uncovered requirement whose status is
  closest to supported, prefers a retriever the initial round did not use, and stops
  on coverage, round, document or retriever-execution budget, or when a round fails
  to improve coverage.
- **Context builder**: one covering passage per supported requirement (dependency
  order first), then best candidates for uncovered requirements, then the rest by
  rerank order; 4,000-token budget; drops are recorded, requirements are never
  truncated.

## Budgets (frozen before the final runs)

| Parameter | Value | Meaning |
|---|---|---|
| requirements | 4 | planner cap |
| additional rounds | 2 | after the initial retrieval |
| unique documents | 20 | cumulative admitted candidates |
| generator context | 4,000 tokens | final prompt budget |
| retriever executions | 8 | individual dense/BM25 calls, initial round included |

## Statistics

- McNemar exact test for paired Exact Match.
- Paired bootstrap (2,000 resamples, seed 42) for F1, token and cost differences;
  stratified by dataset when pooling.
- Holm–Bonferroni correction over each primary family.
- Comparisons use identical question ids; no question is ever removed.
