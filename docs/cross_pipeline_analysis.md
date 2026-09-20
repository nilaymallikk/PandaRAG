# Unified Cross-Pipeline Analysis

Read-only analysis of the four frozen pipelines on the immutable 500-question
HotpotQA benchmark. Dataset SHA-256 `956405ceb43d73385d60898e1bfc7f45a8184c9a4e1eff45b9a8ec239acfd048` (expected `956405ceb43d73385d60898e1bfc7f45a8184c9a4e1eff45b9a8ec239acfd048`), match: **True**.

- Questions: **500** per pipeline, identical question IDs: **True**
- Model (all pipelines): **deepseek-flash**, thinking `disabled`
- No pipeline, prompt, dataset, retrieval artifact or result file was modified.
- Retrieval metrics for No-RAG are **N/A** (no retrieval performed), not zero.

## 1. Aggregate answer quality

| Pipeline | EM | F1 | Correct (EM=1) | Failures |
|---|---|---|---|---|
| No-RAG | 0.3520 | 0.4756 | 176/500 | 0 |
| Standard Dense RAG (K=5) | 0.5100 | 0.6925 | 255/500 | 0 |
| Hybrid RAG (K=5) | 0.4900 | 0.6764 | 245/500 | 0 |
| Adaptive RAG (K in {3,5}) | 0.4820 | 0.6598 | 241/500 | 0 |

## 2. Retrieval quality and context size

| Pipeline | K (realized) | Mean docs | Doc Recall | Sent Recall | Doc Complete |
|---|---|---|---|---|---|
| No-RAG | none | 0.0000 | N/A | N/A | N/A |
| Standard Dense RAG (K=5) | 5 (fixed) | 4.9760 | 0.9310 | 0.6718 | 0.8640 |
| Hybrid RAG (K=5) | 5 (fixed) | 4.9760 | 0.8920 | 0.6627 | 0.7880 |
| Adaptive RAG (K in {3,5}) | 3 or 5 (mixed) | 3.5060 | 0.8340 | 0.5914 | 0.6740 |

Adaptive realized K is mixed: 371 questions at K=3, 129 at K=5 (mean 3.5060 docs), so its realized retrieval scores interpolate between the fixed-K rows below.

Frozen retrieval artifacts, Recall@K (no generation involved):

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| dense | 0.4500 | 0.8500 | 0.9310 | 1.0000 |
| bm25 | 0.3720 | 0.6850 | 0.7920 | 1.0000 |
| hybrid | 0.4390 | 0.7940 | 0.8920 | 1.0000 |

Document-complete@K (fraction with all gold documents retrieved):

| Retriever | Complete@1 | Complete@3 | Complete@5 | Complete@10 |
|---|---|---|---|---|
| dense | 0.0000 | 0.7220 | 0.8640 | 1.0000 |
| bm25 | 0.0000 | 0.4360 | 0.6120 | 1.0000 |
| hybrid | 0.0000 | 0.6080 | 0.7880 | 1.0000 |

## 3. Efficiency (actual API usage)

| Pipeline | Input tok | Output tok | Cached in tok | Cost (USD) | LLM calls | Cache hits | Latency mean/med/p95 (s) |
|---|---|---|---|---|---|---|---|
| No-RAG | 31,120 | 2,222 | 0 | $0.00600120 | 490 | 10 | 0.5763/0.5897/0.8184 |
| Standard Dense RAG (K=5) | 357,810 | 3,004 | 0 | $0.05547390 | 490 | 10 | 0.6440/0.6258/0.9190 |
| Hybrid RAG (K=5) | 367,325 | 2,985 | 40,320 | $0.05096271 | 474 | 26 | 0.7064/0.7258/0.9726 |
| Adaptive RAG (K in {3,5}) | 263,640 | 3,119 | 97,408 | $0.02709842 | 360 | 140 | 0.5113/0.6128/0.9206 |

Cost/latency caveat: the Adaptive run replayed 140 prompts cached by earlier smoke runs of the same frozen policy, so its live-call latency and cost reflect cache reuse as well as shorter contexts.

## 4. Efficiency-normalized comparison

| Pipeline | Input tok/question | Cost/question | Cost per correct (EM=1) | Input tok per correct |
|---|---|---|---|---|
| No-RAG | 62.2 | $0.00001200 | $0.00003410 | 176.8 |
| Standard Dense RAG (K=5) | 715.6 | $0.00011095 | $0.00021754 | 1403.2 |
| Hybrid RAG (K=5) | 734.6 | $0.00010193 | $0.00020801 | 1499.3 |
| Adaptive RAG (K in {3,5}) | 527.3 | $0.00005420 | $0.00011244 | 1093.9 |

Adaptive versus Hybrid RAG (K=5), same questions:

- Input tokens: 263,640 vs 367,325 (**-28.2%**)
- Cost: $0.02709842 vs $0.05096271 (**-46.8%**)
- Cost per correct answer: $0.00011244 vs $0.00020801
- EM: 0.4820 vs 0.4900 (**-1.6% relative**), quality retained: **98.4%**

## 5. Paired comparison versus Adaptive RAG (per question, n=500)

| Baseline | EM wins | EM losses | EM ties | McNemar exact p | Mean F1 delta (Adaptive - baseline) | 95% CI (bootstrap) |
|---|---|---|---|---|---|---|
| No-RAG | 106 | 41 | 353 | 8.089e-08 | +0.1843 | [+0.1424, +0.2250] |
| Standard Dense RAG (K=5) | 22 | 36 | 442 | 0.08695 | -0.0327 | [-0.0589, -0.0075] |
| Hybrid RAG (K=5) | 13 | 17 | 470 | 0.5847 | -0.0166 | [-0.0367, +0.0029] |

"EM wins" = Adaptive correct, baseline wrong; losses = Adaptive wrong, baseline correct. A CI spanning zero means the paired F1 difference is not statistically distinguishable at this n.

## 6. Slices by question type and difficulty

### By question_type

| question_type | n | Pipeline | EM | F1 |
|---|---|---|---|---|
| bridge | 400 | No-RAG | 0.2950 | 0.4242 |
| bridge | 400 | Standard Dense RAG (K=5) | 0.5400 | 0.7306 |
| bridge | 400 | Hybrid RAG (K=5) | 0.5025 | 0.7009 |
| bridge | 400 | Adaptive RAG (K in {3,5}) | 0.4975 | 0.6869 |
| comparison | 100 | No-RAG | 0.5800 | 0.6811 |
| comparison | 100 | Standard Dense RAG (K=5) | 0.3900 | 0.5400 |
| comparison | 100 | Hybrid RAG (K=5) | 0.4400 | 0.5785 |
| comparison | 100 | Adaptive RAG (K in {3,5}) | 0.4200 | 0.5515 |

### By level

| level | n | Pipeline | EM | F1 |
|---|---|---|---|---|
| hard | 500 | No-RAG | 0.3520 | 0.4756 |
| hard | 500 | Standard Dense RAG (K=5) | 0.5100 | 0.6925 |
| hard | 500 | Hybrid RAG (K=5) | 0.4900 | 0.6764 |
| hard | 500 | Adaptive RAG (K in {3,5}) | 0.4820 | 0.6598 |

## 7. Adaptive policy diagnostics (descriptive only)

- Decision split: **K=3: 371 (74.2%)**, **K=5: 129 (25.8%)**; mean inferred K 3.5060.
- Confidence: mean 0.5268, median 0.5258, range [0.3463, 0.6755], threshold 0.5000.
- Confidence vs correctness: Spearman rho +0.0394, AUC 0.5228 (+0.0228 above chance).
- K=3 subset: adaptive EM 0.4960/F1 0.6684, mean input tok 455.1
- K=5 subset: adaptive EM 0.4419/F1 0.6351, mean input tok 734.8

Subset EM differences are **descriptive only** — questions were assigned to a subset by the confidence signal itself, so the comparison is confounded by selection and is not evidence that K=3 or K=5 is better in general.

Consistency check on identical inputs: for the 129 questions where Adaptive chose K=5, its context is the same frozen hybrid top-5 and its prediction matches Hybrid RAG on **129/129** questions (expected: identical prompts).

Truncation effect (valid paired comparison, same questions and same ranking prefix): on the 371 questions where Adaptive supplied 3 documents, Adaptive EM 0.4960 / F1 0.6684 versus Hybrid RAG (5 documents) EM 0.5067 / F1 0.6908 — a like-for-like measure of what the shorter context costs. Predictions match on 273/371 questions.

## 8. Provenance and integrity checks

| Check | Result |
|---|---|
| Dataset SHA-256 unchanged | PASS |
| Same 500 question IDs across all pipelines | PASS |
| No-RAG retrieval metrics null (not zero) | PASS |
| All generation failures logged (failures column above) | PASS |
| All pipelines used one model | PASS |
| Faithfulness/hallucination deferred (null) in all RAG pipelines | PASS |
| Reported meta F1 equals recomputed F1 for every pipeline (meta stores 6 dp) | PASS |

## 9. Honest summary

- F1 ranking: Standard Dense RAG (K=5) (0.6925) > Hybrid RAG (K=5) (0.6764) > Adaptive RAG (K in {3,5}) (0.6598) > No-RAG (0.4756).
- Adaptive RAG is the cheapest retrieval pipeline ($0.02709842) while retaining 98.4% of Hybrid RAG's EM and 97.6% of its F1, using 0.72x the input tokens.
- On the 371 questions it answered with 3 documents, the shorter context changed EM by -0.0108 versus the identical K=5 hybrid context — the concrete accuracy price of the budget reduction on that subset.
- The frozen confidence rule is agreement-dominated, so K=3 dominates; it is an ordinal heuristic, not a calibrated probability.
- Faithfulness/groundedness and hallucination are unmeasured in all pipelines (would require a separate pre-registered protocol and additional LLM calls); no claim is made about them here.
- No evaluation rule, pipeline, prompt, dataset or artifact was changed to produce these numbers.

