# PandaRAG: evidence-gap-aware retrieval-budget allocation for multi-hop QA

*Generated 2026-11-20 from artifacts; every number below is read from `results/`.*

## Abstract

We reimplement the SAC-RAG architecture (question-type-adaptive retrieval plus context compression) on a pooled candidate corpus and propose PandaRAG, which replaces SAC-RAG's single global retrieval decision with requirement-level evidence tracking: a planner states what must be known, an evaluator decides per requirement whether retrieved documents actually support it, and a controller spends the remaining retrieval budget only on the uncovered requirements. On HotpotQA, 2WikiMultiHopQA and MuSiQue (500 questions each), we compare eight systems on identical question ids with paired statistics.

PandaRAG reaches F1 0.583 pooled versus 0.583 for SAC-RAG Adaptive (Δ -0.000, 95% CI [-0.003, +0.003]): **no detectable difference in answer quality, at higher cost**. This is a failure to reject the null, not evidence of equivalence: at a post-hoc ±0.02 F1 margin the interval is consistent with equivalence, but that margin was not pre-registered and no equivalence test was pre-specified. Retrieval coverage is identical by construction because both systems start from the same retrieval, and the ablation shows the added machinery does not pay for itself on this corpus. We report that negative result with its diagnosis rather than a favourable subset: the bottleneck is retrieval completeness, not retrieval-budget allocation.

## 1. Introduction

Multi-hop questions are answered from a set of passages whose *union* must contain every bridge fact. Document-level Recall@10 is a poor proxy for that condition: a system can retrieve nine of ten relevant documents and still be unable to answer. The base paper (SAC-RAG) improves the efficiency of this pipeline by routing questions to type-specific retrieval and compressing the retrieved context, but it takes the retrieval decision once and globally.

PandaRAG keeps SAC-RAG's routing and asks a different question: *which required piece of evidence is missing?* It answers that per requirement, and spends the remaining budget on the gap rather than on a larger top-k.

## 2. Related work

Adaptive-RAG routes by question complexity; IRCoT interleaves retrieval with chain-of-thought; Self-RAG learns when to retrieve; LLMLingua compresses context; DPR and Contriever establish dense retrieval baselines. PandaRAG differs from all of them in granularity: the control signal is per-requirement evidence coverage, not a global confidence or complexity score. Full references are in the project's `docs/pandarag/` protocol documents.

## 3. SAC-RAG architecture (as reimplemented)

Classification into Fact / Definition / Reasoning, then: Fact = BM25 top-10 + dense top-10 → union → BGE rerank → top-10; Definition = dense top-8 (+ rerank in our variant); Reasoning = sub-question decomposition → dense top-10 per sub-question → union → BGE rerank → top-10. Full SAC-RAG adds an LLM extractive compressor before generation. Every parameter and every deviation is recorded in `sac_rag_reproduction_protocol.md`.

**This is a best-effort architectural reimplementation, not an exact reproduction.** The corpus, classifier prompt, decomposition prompt and compression prompt are unpublished; the paper also contradicts itself on the compressor model.

## 4. PandaRAG architecture

Five components, each independently testable:

1. **Evidence planner** — atomic requirements with search queries and dependencies; capped at four; overflow is reported, malformed output falls back to a single requirement.
2. **Evidence evaluator** — per (requirement, document) verdict with statuses `supported/partial/insufficient/contradicted/unknown`; relevance is explicitly *not* treated as support.
3. **Coverage** — per-requirement aggregation; `partial` counts half; nothing is credited across requirements.
4. **Gap controller** — rule-based; targets the uncovered requirement closest to supported, avoids repeating the retriever the initial round already used, and stops on any of four budgets or on a round that fails to improve coverage.
5. **Evidence-preserving context builder** — one covering passage per supported requirement (dependency order), bridge material retained, 4,000-token budget, drops recorded.

## 5. Datasets

| Dataset | Questions | Pooled documents | Gold documents | Evidence granularity |
|---|---|---|---|---|
| HotpotQA validation | 500 | 4,928 | 994 | sentence |
| 2WikiMultiHopQA dev | 500 | 3,246 | 990 | sentence |
| MuSiQue validation | 500 | 7,192 | 912 | paragraph |

MuSiQue has 6,356 distinct titles but 7,192 distinct documents because 421 titles carry more than one text; document ids are content-derived, never title-derived.

## 6. Experimental setup

Eight systems (P0, P1c, P2c, P3c, P1n, P4, P5, P6), three datasets, identical question ids per dataset, one generator prompt and one model for every system, temperature 0, frozen prompts and budgets. Full details in `final_methodology.md` and `experiment_manifest.md`.

## 7. Results

## Answer quality

| System | Dataset | n | EM | F1 | R@10 | all-gold@10 |
|---|---|---|---|---|---|---|
| No-RAG | hotpotqa | 500 | 0.334 | 0.456 | 0.000 | 0.000 |
| Dense RAG (corpus) | hotpotqa | 500 | 0.574 | 0.726 | 0.885 | 0.788 |
| Hybrid RAG (fusion depth 5) | hotpotqa | 500 | 0.546 | 0.699 | 0.868 | 0.748 |
| Hybrid RAG (fusion depth 10) | hotpotqa | 500 | 0.558 | 0.713 | 0.927 | 0.858 |
| Adaptive RAG (transferred) | hotpotqa | 500 | 0.536 | 0.691 | 0.927 | 0.858 |
| Naive RAG (paper 10->5) | hotpotqa | 500 | 0.574 | 0.726 | 0.922 | 0.854 |
| SAC-RAG Adaptive | hotpotqa | 500 | 0.606 | 0.756 | 0.939 | 0.880 |
| Full SAC-RAG | hotpotqa | 500 | 0.610 | 0.764 | 0.939 | 0.880 |
| PandaRAG | hotpotqa | 500 | 0.606 | 0.759 | 0.939 | 0.878 |
| No-RAG | 2wiki | 500 | 0.252 | 0.312 | 0.000 | 0.000 |
| Dense RAG (corpus) | 2wiki | 500 | 0.510 | 0.588 | 0.785 | 0.574 |
| Hybrid RAG (fusion depth 5) | 2wiki | 500 | 0.488 | 0.556 | 0.773 | 0.546 |
| Hybrid RAG (fusion depth 10) | 2wiki | 500 | 0.490 | 0.559 | 0.813 | 0.626 |
| Adaptive RAG (transferred) | 2wiki | 500 | 0.440 | 0.521 | 0.813 | 0.626 |
| Naive RAG (paper 10->5) | 2wiki | 500 | 0.510 | 0.588 | 0.823 | 0.646 |
| SAC-RAG Adaptive | 2wiki | 500 | 0.536 | 0.620 | 0.819 | 0.638 |
| Full SAC-RAG | 2wiki | 500 | 0.510 | 0.585 | 0.819 | 0.638 |
| PandaRAG | 2wiki | 500 | 0.534 | 0.618 | 0.819 | 0.638 |
| No-RAG | musique | 500 | 0.080 | 0.158 | 0.000 | 0.000 |
| Dense RAG (corpus) | musique | 500 | 0.214 | 0.330 | 0.476 | 0.142 |
| Hybrid RAG (fusion depth 5) | musique | 500 | 0.214 | 0.342 | 0.496 | 0.150 |
| Hybrid RAG (fusion depth 10) | musique | 500 | 0.218 | 0.341 | 0.586 | 0.232 |
| Adaptive RAG (transferred) | musique | 500 | 0.200 | 0.326 | 0.586 | 0.232 |
| Naive RAG (paper 10->5) | musique | 500 | 0.214 | 0.330 | 0.569 | 0.212 |
| SAC-RAG Adaptive | musique | 500 | 0.236 | 0.372 | 0.603 | 0.244 |
| Full SAC-RAG | musique | 500 | 0.240 | 0.375 | 0.603 | 0.244 |
| PandaRAG | musique | 500 | 0.236 | 0.370 | 0.604 | 0.242 |

### Pooled over the three datasets

| System | n | EM | F1 | R@10 |
|---|---|---|---|---|
| SAC-RAG Adaptive | 1500 | 0.459 | 0.583 | 0.787 |
| PandaRAG | 1500 | 0.459 | 0.583 | 0.787 |
| Full SAC-RAG | 1500 | 0.453 | 0.575 | 0.787 |
| Dense RAG (corpus) | 1500 | 0.433 | 0.548 | 0.716 |
| Naive RAG (paper 10->5) | 1500 | 0.433 | 0.548 | 0.771 |
| Hybrid RAG (fusion depth 10) | 1500 | 0.422 | 0.537 | 0.775 |
| Hybrid RAG (fusion depth 5) | 1500 | 0.416 | 0.532 | 0.712 |
| Adaptive RAG (transferred) | 1500 | 0.392 | 0.512 | 0.775 |
| No-RAG | 1500 | 0.222 | 0.308 | 0.000 |

## Retrieval quality at fixed depth (pooled candidate corpus)

| System | depth | R@5 | R@8 | R@10 | all-gold@10 | R@stop | R@ctx |
|---|---|---|---|---|---|---|---|
| P1c | 5 | 0.716 | – | – | – | 0.716 | 0.716 |
| P2c | 5 | 0.712 | – | – | – | 0.712 | 0.712 |
| P2c10 | 10 | 0.705 | 0.759 | 0.775 | 0.572 | 0.775 | 0.705 |
| P3c | 10 | 0.705 | 0.759 | 0.775 | 0.572 | 0.775 | 0.654 |
| P1n | 10 | 0.716 | 0.755 | 0.771 | 0.571 | 0.771 | 0.716 |
| P4 | 9.951 | 0.727 | 0.776 | 0.788 | 0.588 | 0.787 | 0.787 |
| P5 | 9.951 | 0.727 | 0.776 | 0.788 | 0.588 | 0.787 | 0.760 |
| P6 | 9.968 | 0.727 | 0.777 | 0.787 | 0.586 | 0.787 | 0.788 |

Three different quantities are in this table and they must not be conflated:

- **R@k (fixed depth)** — did retrieval *find* the gold documents among the top k? Undefined (`–`) when a system never retrieved k documents.
- **R@stop** — recall over the documents the retriever ranked, i.e. the list length the pipeline actually produced.
- **R@ctx** — recall over the documents that were actually placed in the generator prompt. This is the only column that bounds what the model could use.

The three differ substantially and the differences are the point: **P2c10** finds 0.775 of gold at depth 10 but feeds only 5 documents, so its generator saw 0.705 — its candidate R@10 is *not* evidence the model consumed. **P1n** is the same shape (0.771 found, 0.716 fed). **P3c** retrieves 10 (0.775) but feeds 3–5, so its context recall is 0.654 — the adaptive K is where its evidence is lost. **P5** drops from 0.787 retrieved to 0.760 after compression. P4 and P6 feed everything they rank, so their R@stop and R@ctx coincide.

P2c10 and P3c share identical *retrieval* by construction — they differ only in how many documents they feed — which is a useful check that the corrected fusion reproduces the Phase-1 hybrid ranking.

## Retrieval quality per dataset (actual stopping depth)


| System | Dataset | docs | R@1 | R@3 | R@5 | R@k (depth) | all-gold@10 |
|---|---|---|---|---|---|---|---|
| P1c | hotpotqa | 5.000 | 0.447 | 0.822 | 0.885 | 0.885 | 0.788 |
| P2c | hotpotqa | 5.000 | 0.432 | 0.763 | 0.868 | 0.868 | 0.748 |
| P2c10 | hotpotqa | 10.000 | 0.431 | 0.751 | 0.855 | 0.927 | 0.858 |
| P3c | hotpotqa | 10.000 | 0.431 | 0.751 | 0.855 | 0.927 | 0.858 |
| P1n | hotpotqa | 10.000 | 0.447 | 0.822 | 0.885 | 0.922 | 0.854 |
| P4 | hotpotqa | 9.908 | 0.461 | 0.842 | 0.874 | 0.939 | 0.880 |
| P5 | hotpotqa | 9.908 | 0.461 | 0.842 | 0.874 | 0.939 | 0.880 |
| P6 | hotpotqa | 9.928 | 0.461 | 0.841 | 0.874 | 0.939 | 0.878 |
| P1c | 2wiki | 5.000 | 0.492 | 0.738 | 0.785 | 0.785 | 0.574 |
| P2c | 2wiki | 5.000 | 0.477 | 0.719 | 0.773 | 0.773 | 0.546 |
| P2c10 | 2wiki | 10.000 | 0.480 | 0.723 | 0.772 | 0.813 | 0.626 |
| P3c | 2wiki | 10.000 | 0.480 | 0.723 | 0.772 | 0.813 | 0.626 |
| P1n | 2wiki | 10.000 | 0.492 | 0.738 | 0.785 | 0.823 | 0.646 |
| P4 | 2wiki | 10.000 | 0.473 | 0.741 | 0.776 | 0.819 | 0.638 |
| P5 | 2wiki | 10.000 | 0.473 | 0.741 | 0.776 | 0.819 | 0.638 |
| P6 | 2wiki | 10.000 | 0.473 | 0.741 | 0.776 | 0.819 | 0.638 |
| P1c | musique | 5.000 | 0.266 | 0.407 | 0.476 | 0.476 | 0.142 |
| P2c | musique | 5.000 | 0.268 | 0.419 | 0.496 | 0.496 | 0.150 |
| P2c10 | musique | 10.000 | 0.273 | 0.415 | 0.486 | 0.586 | 0.232 |
| P3c | musique | 10.000 | 0.273 | 0.415 | 0.486 | 0.586 | 0.232 |
| P1n | musique | 10.000 | 0.266 | 0.407 | 0.476 | 0.569 | 0.212 |
| P4 | musique | 9.944 | 0.286 | 0.455 | 0.529 | 0.603 | 0.244 |
| P5 | musique | 9.944 | 0.286 | 0.455 | 0.529 | 0.603 | 0.244 |
| P6 | musique | 9.976 | 0.286 | 0.457 | 0.530 | 0.604 | 0.242 |

**Retrieval depth matters for this table.** `R@k (depth)` is computed over the documents the system actually retrieved. P1c and P2c retrieve 5 documents, so their `R@k (depth)` is **Recall@5** (0.716 / 0.712 pooled), not Recall@10 — the Phase-1 benchmark's R@10 of 0.771 (dense) and 0.775 (hybrid) belongs to systems that retrieve ten documents (P1n, P3c). P4–P6 stop at 8 documents for Definition questions, so a small number of their cells are R@8. Separately, P2c fuses source lists already truncated to 5, whereas P3c/P4/P5/P6 fuse at depth 10 and then truncate; its ranking is therefore not identical to the other hybrid systems.

## Efficiency

| System | Dataset | in tok | out tok | LLM calls | retr. exec | ctx tok | $/q | mean s |
|---|---|---|---|---|---|---|---|---|
| P0 | hotpotqa | 104.212 | 5.778 | 1.000 | 0.000 | 4.000 | 0.000 | 1.432 |
| P1c | hotpotqa | 737.084 | 5.044 | 1.000 | 1.000 | 674.650 | 0.000 | 0.243 |
| P2c | hotpotqa | 723.640 | 5.028 | 1.000 | 1.000 | 662.392 | 0.000 | 4.091 |
| P2c10 | hotpotqa | 730.858 | 5.030 | 1.000 | 1.000 | 671.904 | 0.000 | 2.522 |
| P3c | hotpotqa | 501.764 | 5.020 | 1.000 | 1.000 | 426.618 | 0.000 | 4.065 |
| P1n | hotpotqa | 737.084 | 5.044 | 1.000 | 1.000 | 674.650 | 0.000 | 3.732 |
| P4 | hotpotqa | 1459.386 | 8.022 | 1.108 | 1.956 | 1434.798 | 0.000 | 24.525 |
| P5 | hotpotqa | 1762.672 | 128.476 | 2.110 | 1.956 | 95.940 | 0.001 | 13.059 |
| P6 | hotpotqa | 3428.174 | 238.870 | 2.592 | 2.368 | 1436.412 | 0.001 | 0.123 |
| P0 | 2wiki | 99.534 | 7.028 | 1.000 | 0.000 | 4.000 | 0.000 | 4.748 |
| P1c | 2wiki | 877.184 | 7.372 | 1.000 | 1.000 | 785.448 | 0.000 | 4.322 |
| P2c | 2wiki | 837.544 | 7.594 | 1.000 | 1.000 | 753.526 | 0.000 | 3.863 |
| P2c10 | 2wiki | 829.586 | 7.370 | 1.000 | 1.000 | 744.576 | 0.000 | 2.235 |
| P3c | 2wiki | 563.244 | 7.330 | 1.000 | 1.000 | 474.426 | 0.000 | 6.058 |
| P1n | 2wiki | 877.184 | 7.372 | 1.000 | 1.000 | 785.448 | 0.000 | 0.236 |
| P4 | 2wiki | 1809.588 | 15.350 | 1.284 | 1.996 | 1709.498 | 0.001 | 48.186 |
| P5 | 2wiki | 1878.436 | 110.306 | 2.324 | 1.996 | 67.964 | 0.000 | 13.046 |
| P6 | 2wiki | 3828.074 | 271.658 | 2.974 | 2.586 | 1619.164 | 0.001 | 0.100 |
| P0 | musique | 105.320 | 9.066 | 1.000 | 0.000 | 4.000 | 0.000 | 4.298 |
| P1c | musique | 792.604 | 7.832 | 1.000 | 1.000 | 754.708 | 0.000 | 5.329 |
| P2c | musique | 788.408 | 8.286 | 1.000 | 1.000 | 754.944 | 0.000 | 4.652 |
| P2c10 | musique | 785.390 | 8.332 | 1.000 | 1.000 | 751.608 | 0.000 | 1.572 |
| P3c | musique | 603.488 | 8.058 | 1.000 | 1.000 | 551.478 | 0.000 | 4.779 |
| P1n | musique | 792.604 | 7.832 | 1.000 | 1.000 | 754.708 | 0.000 | 0.242 |
| P4 | musique | 1551.190 | 11.160 | 1.068 | 2.042 | 1582.218 | 0.000 | 33.864 |
| P5 | musique | 1876.398 | 151.740 | 2.148 | 2.042 | 118.590 | 0.000 | 11.943 |
| P6 | musique | 3822.200 | 393.776 | 2.904 | 2.848 | 1583.096 | 0.001 | 0.303 |

## Failure accounting and metric denominators

The 27 system–dataset runs contain **13,500 canonical records** (one per question id) written as 13,980 lines. The difference is fully accounted for: 381 superseded HTTP-429 attempts plus 99 duplicate successful rows. Every cell has exactly 500 unique question ids and 500 of them have a success, so **questions without a success: 0**. EM/F1 and retrieval denominators are 500 in every cell; neither a failed attempt nor a duplicate row enters an aggregate.

| System | Dataset | rows | success | superseded failures | duplicate successes | unique Q | terminal failures | EM/F1 denominator |
|---|---|---|---|---|---|---|---|---|
| P0 | hotpotqa | 516 | 500 | 16 | 0 | 500 | 0 | 500 |
| P1c | hotpotqa | 500 | 500 | 0 | 0 | 500 | 0 | 500 |
| P2c | hotpotqa | 536 | 500 | 36 | 0 | 500 | 0 | 500 |
| P2c10 | hotpotqa | 500 | 500 | 0 | 0 | 500 | 0 | 500 |
| P3c | hotpotqa | 526 | 500 | 26 | 0 | 500 | 0 | 500 |
| P1n | hotpotqa | 537 | 500 | 37 | 0 | 500 | 0 | 500 |
| P4 | hotpotqa | 500 | 500 | 0 | 0 | 500 | 0 | 500 |
| P5 | hotpotqa | 500 | 500 | 0 | 0 | 500 | 0 | 500 |
| P6 | hotpotqa | 500 | 500 | 0 | 0 | 500 | 0 | 500 |
| P0 | 2wiki | 527 | 500 | 27 | 0 | 500 | 0 | 500 |
| P1c | 2wiki | 530 | 500 | 30 | 0 | 500 | 0 | 500 |
| P2c | 2wiki | 540 | 500 | 40 | 0 | 500 | 0 | 500 |
| P2c10 | 2wiki | 600 | 599 | 1 | 99 | 500 | 0 | 500 |
| P3c | 2wiki | 513 | 500 | 13 | 0 | 500 | 0 | 500 |
| P1n | 2wiki | 500 | 500 | 0 | 0 | 500 | 0 | 500 |
| P4 | 2wiki | 500 | 500 | 0 | 0 | 500 | 0 | 500 |
| P5 | 2wiki | 514 | 500 | 14 | 0 | 500 | 0 | 500 |
| P6 | 2wiki | 500 | 500 | 0 | 0 | 500 | 0 | 500 |
| P0 | musique | 533 | 500 | 33 | 0 | 500 | 0 | 500 |
| P1c | musique | 516 | 500 | 16 | 0 | 500 | 0 | 500 |
| P2c | musique | 530 | 500 | 30 | 0 | 500 | 0 | 500 |
| P2c10 | musique | 500 | 500 | 0 | 0 | 500 | 0 | 500 |
| P3c | musique | 515 | 500 | 15 | 0 | 500 | 0 | 500 |
| P1n | musique | 500 | 500 | 0 | 0 | 500 | 0 | 500 |
| P4 | musique | 500 | 500 | 0 | 0 | 500 | 0 | 500 |
| P5 | musique | 547 | 500 | 47 | 0 | 500 | 0 | 500 |
| P6 | musique | 500 | 500 | 0 | 0 | 500 | 0 | 500 |

The duplicate successes are confined to `P2c10` / 2Wiki: its file was written by two processes concurrently after an apparent process death (timestamps 20 s apart), so 99 questions have two successful rows. Analysis folds the file to one row per question id (preferring a success), which is why every aggregate is computed over exactly 500 questions. The rows are preserved rather than deleted.

The process-global error log holds 7284 entries ({'429': 7253, 'None': 29, '400': 1, '502': 1}), with an attempt histogram of {1: 2095, 2: 1848, 3: 1564, 4: 1244, 5: 533}. It carries no pipeline or question id, so retries cannot be attributed per system from that file; per-run superseded counts above come from the result JSONL.

## PandaRAG adaptive behaviour

| Dataset | reqs | cov. initial | cov. final | all covered | targeted | recovery ok | rounds | abstained |
|---|---|---|---|---|---|---|---|---|
| hotpotqa | 1.116 | 0.763 | 0.766 | 0.582 | 0.406 | 0.010 | 1.406 | 0.000 |
| 2wiki | 1.284 | 0.669 | 0.669 | 0.414 | 0.586 | 0.000 | 1.586 | 0.000 |
| musique | 1.158 | 0.479 | 0.486 | 0.220 | 0.784 | 0.008 | 1.786 | 0.000 |


## 8. Statistical analysis

## Primary comparisons (paired, identical question ids)

| Comparison | Dataset | Metric | n | mean Δ | 95% CI | McNemar p | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|
| P6 vs P4 | hotpotqa | f1 | 500 | +0.002 | [-0.002, +0.008] | 1.0000 | 1.0000 | no difference |
| P6 vs P4 | 2wiki | f1 | 500 | -0.002 | [-0.006, +0.000] | 1.0000 | 1.0000 | no difference |
| P6 vs P4 | musique | f1 | 500 | -0.001 | [-0.008, +0.006] | 1.0000 | 1.0000 | no difference |
| P6 vs P4 | pooled | f1 | 1500 | -0.000 | [-0.003, +0.003] | 1.0000 | 1.0000 | no difference |
| P6 vs P4 | pooled | exact_match | 1500 | -0.001 | – | 1.0000 | 1.0000 | n/a |
| P6 vs P5 | hotpotqa | f1 | 500 | -0.005 | [-0.024, +0.014] | 0.8642 | 1.0000 | no difference |
| P6 vs P5 | 2wiki | f1 | 500 | +0.033 | [+0.010, +0.057] | 0.0807 | 0.9683 | better |
| P6 vs P5 | musique | f1 | 500 | -0.005 | [-0.025, +0.016] | 0.8555 | 1.0000 | no difference |
| P6 vs P5 | pooled | f1 | 1500 | +0.008 | [-0.004, +0.020] | 0.4926 | 1.0000 | no difference |
| P6 vs P5 | pooled | exact_match | 1500 | +0.005 | – | 0.4926 | 0.9853 | n/a |
| P6 vs P1c | hotpotqa | f1 | 500 | +0.033 | [+0.012, +0.054] | 0.0166 | 0.2322 | better |
| P6 vs P1c | 2wiki | f1 | 500 | +0.031 | [+0.006, +0.057] | 0.1038 | 1.0000 | better |
| P6 vs P1c | musique | f1 | 500 | +0.040 | [+0.010, +0.069] | 0.1524 | 1.0000 | better |
| P6 vs P1c | pooled | f1 | 1500 | +0.035 | [+0.019, +0.049] | 0.0010 | 0.0170 | better |
| P6 vs P1c | pooled | exact_match | 1500 | +0.026 | – | 0.0010 | 0.0030 | n/a |
| P6 vs P2c | hotpotqa | f1 | 500 | +0.060 | [+0.040, +0.081] | 0.0000 | 0.0000 | better |
| P6 vs P2c | 2wiki | f1 | 500 | +0.062 | [+0.035, +0.088] | 0.0011 | 0.0174 | better |
| P6 vs P2c | musique | f1 | 500 | +0.029 | [-0.000, +0.058] | 0.1770 | 1.0000 | no difference |
| P6 vs P2c | pooled | f1 | 1500 | +0.050 | [+0.035, +0.065] | 0.0000 | 0.0000 | better |
| P6 vs P2c | pooled | exact_match | 1500 | +0.043 | – | 0.0000 | 0.0000 | n/a |
| P6 vs P2c10 | hotpotqa | f1 | 500 | +0.046 | [+0.027, +0.066] | 0.0000 | 0.0007 | better |
| P6 vs P2c10 | 2wiki | f1 | 500 | +0.059 | [+0.033, +0.087] | 0.0021 | 0.0313 | better |
| P6 vs P2c10 | musique | f1 | 500 | +0.030 | [+0.002, +0.059] | 0.2717 | 1.0000 | better |
| P6 vs P2c10 | pooled | f1 | 1500 | +0.045 | [+0.030, +0.060] | 0.0000 | 0.0000 | better |
| P6 vs P2c10 | pooled | exact_match | 1500 | +0.037 | – | 0.0000 | 0.0000 | n/a |
| P6 vs P3c | hotpotqa | f1 | 500 | +0.068 | [+0.045, +0.092] | 0.0000 | 0.0000 | better |
| P6 vs P3c | 2wiki | f1 | 500 | +0.097 | [+0.069, +0.131] | 0.0000 | 0.0000 | better |
| P6 vs P3c | musique | f1 | 500 | +0.045 | [+0.015, +0.075] | 0.0273 | 0.3554 | better |
| P6 vs P3c | pooled | f1 | 1500 | +0.070 | [+0.054, +0.087] | 0.0000 | 0.0000 | better |
| P6 vs P3c | pooled | exact_match | 1500 | +0.067 | – | 0.0000 | 0.0000 | n/a |

Positive Δ means the first system is higher. McNemar is computed on Exact Match; the F1 interval is a paired bootstrap (2,000 resamples). Holm adjustment is applied within each metric family.

## Ablations (reduced sample)

Sample: 100 questions per dataset, stratified by frozen semantic type, identical across variants (seed 42). Ablations are **not** mixed into the full-sample results above.

| Variant | Planner | Evaluator | Controller | Evidence context | Compression | n | EM | F1 | ΔF1 vs A0 [95% CI] | McNemar p | tokens/q | $/q (replay) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | – | – | – | – | – | 300 | 0.430 | 0.563 | – | – | 1601.6 | $0.00038 |
| A1 | on | off | off | off | off | 300 | 0.433 | 0.570 | +0.0064 [-0.0003, +0.0150] (n=300) | 1.0000 | 1612.9 | $0.00037 |
| A2 | on | on | off | off | off | 300 | 0.433 | 0.570 | +0.0064 [-0.0003, +0.0150] (n=300) | 1.0000 | 3432.9 | $0.00085 |
| A3 | on | on | on | off | off | 300 | 0.433 | 0.570 | +0.0064 [-0.0003, +0.0150] (n=300) | 1.0000 | 4021.3 | $0.00099 |
| A4 | on | on | on | on | off | 300 | 0.433 | 0.570 | +0.0064 [-0.0003, +0.0150] (n=300) | 1.0000 | 4021.3 | $0.00099 |
| A5 | on | on | on | on | on | 300 | 0.433 | 0.562 | -0.0010 [-0.0289, +0.0257] (n=300) | 1.0000 | 4413.6 | $0.00109 |

A0 is SAC-RAG Adaptive (all PandaRAG components off). A4 vs A0 is the primary test.

ΔF1 and the McNemar test are paired on identical question ids against A0; the CI is a paired bootstrap (2,000 resamples, seed 42). `$/q (replay)` is the cache-normalised value of the recorded calls — the final ablation artifacts were produced from the disk cache, so this is what the run would cost live, not money spent. In A4 the evidence evaluator accounts for about 63% of cost, and generation is slightly cheaper than A0's.

## Error analysis

| System | wrong | top categories |
|---|---|---|
| P0 | 1167 | closed_book_knowledge_gap (1167) |
| P1c | 851 | missing_evidence_initial (616), evidence_present_not_recognised (225), generator_formatting (10) |
| P2c | 876 | missing_evidence_initial (647), evidence_present_not_recognised (219), generator_formatting (10) |
| P2c10 | 867 | missing_evidence_initial (541), evidence_present_not_recognised (319), generator_formatting (7) |
| P3c | 912 | missing_evidence_initial (541), evidence_present_not_recognised (363), generator_formatting (8) |
| P1n | 851 | missing_evidence_initial (541), evidence_present_not_recognised (300), generator_formatting (10) |
| P4 | 811 | missing_evidence_initial (482), evidence_present_not_recognised (286), reranking_failure (33) |
| P5 | 820 | missing_evidence_initial (493), evidence_present_not_recognised (287), reranking_failure (29) |
| P6 | 812 | evaluator_false_negative (582), evaluator_false_positive (204), generator_formatting (10) |

Category assignment is heuristic over stored artifacts and conservative: `unclassified` is used when no single category is supported.

### Representative PandaRAG traces

**5a71278c5542994082a3e5e2** — What North Carolina native did Danja produce songs for?

- requirements: 1
- coverage: 0 → 0 supported
- round 0 (initial): – → new docs 0, improved=None
- round 1 (recovery): EXPAND_CANDIDATE_POOL → new docs 6, improved=False
- round 2 (stop): STOP → new docs 0, improved=None
- prediction: `J. Cole` · gold: `J. Cole` · EM=1.0

**5a7296815542991f9a20c515** — Who brewed the first espresso coffee in space, Patrick Baudry or Samantha Cristoforetti?

- requirements: 2
- coverage: 1 → 1 supported
- round 0 (initial): – → new docs 0, improved=None
- round 1 (recovery): TARGETED_BM25_RETRIEVAL → new docs 0, improved=False
- round 2 (stop): STOP → new docs 0, improved=None
- prediction: `Samantha Cristoforetti` · gold: `Samantha Cristoforetti` · EM=1.0

**5a72a9ab5542992359bc315a** — Which North American tribe inhabited the banks of the Winnipesaukee River and were also known as the Pawtucket people?

- requirements: 1
- coverage: 0 → 0 supported
- round 0 (initial): – → new docs 0, improved=None
- round 1 (recovery): EXPAND_CANDIDATE_POOL → new docs 2, improved=False
- round 2 (stop): STOP → new docs 0, improved=None
- prediction: `Pennacook` · gold: `the Pennacook people` · EM=0.0

## 9. Grounding

| System | judged | faithfulness | hallucination | fully grounded |
|---|---|---|---|---|
| baseline_adaptive_transferred | 180 | 0.582 | 0.418 | 0.542 |
| baseline_dense_corpus | 180 | 0.606 | 0.394 | 0.569 |
| baseline_dense_paper_naive | 180 | 0.606 | 0.394 | 0.569 |
| baseline_hybrid_corpus | 180 | 0.577 | 0.423 | 0.538 |
| pandarag | 180 | 0.625 | 0.374 | 0.593 |
| sac_rag_adaptive | 180 | 0.625 | 0.374 | 0.593 |
| sac_rag_full | 180 | 0.666 | 0.334 | 0.639 |

Judge protocol `faithfulness-judge/1.0`, unchanged from the original experiment: the judge sees only question, the exact generated context and the answer. Scores are means over answers with at least one checkable claim.

## 10. Discussion

**Negative result.** On the reduced-sample ablation, turning on the planner, evidence evaluator, gap controller and evidence-preserving context changed EM from 0.430 to 0.433 and F1 from 0.563 to 0.570, while raising mean cost per question by 2.60× ($0.00038 → $0.00099) on the same 300 stratified questions.
**Why.** Three mechanisms explain it, in order of importance. (1) Document-level retrieval is the bottleneck, not budget allocation: the pooled corpus is closed-book over each dataset's shipped paragraphs, so recall is already high on HotpotQA/2Wiki and the missing evidence on MuSiQue is simply not in the recovered candidate pool. (2) With a bounded context window the extra documents a recovery round contributes rarely displace reranker-preferred passages, so the final prompt is usually identical to SAC-RAG's (measured: 90% identical on HotpotQA). (3) The evidence evaluator is conservative: `partial` does not count as covered, so coverage understates what the context can support, which makes the controller act on a pessimistic signal.
**Why A3 and A4 are identical.** The evidence-preserving context builder is implemented, tested and active, but under the frozen budget it cannot change the prompt: the evaluator only sees the top-10 admitted documents, and the builder then fills a 10-document window and sorts it by reranker order. Every protected passage is therefore already in the window, so A4 reproduces A3's context exactly. Measured on the ablation artifacts: 300/300 identical contexts for A1→A2 and A2→A3, 299/300 for A3→A4, and 300/300 identical predictions; only 1 of 468 protected requirement–document links fell outside the window. The components fire (920 evaluator verdicts in the first round, $0.19 of evaluator calls in A4) — their *output* is what has no room to act. A design that intends evidence selection to matter must evaluate a candidate pool wider than the context window.
**Mechanism-level diagnosis (evaluator-measured).** The controller did fire: it requested targeted retrieval on 40.6% of HotpotQA questions, 58.6% of 2Wiki and 78.4% of MuSiQue questions. It almost never helped: a recovery round converted an uncovered requirement to supported in 1.0% / 0.0% / 0.8% of rounds. In these runs, re-issuing a query the initial retrieval already ran (then switching retriever) added no document that the evidence evaluator judged newly supporting. Two caveats bound that sentence: the recovery rate is **evaluator-measured**, and the evaluator has no human-labelled validation at scale, so a low rate is evidence that *this controller with this evaluator* did not convert gaps — not proof that the corpus contains no further evidence. The independent retrieval-side fact is narrower and does not depend on the evaluator: on MuSiQue only 23% of questions have all gold documents inside the top 10, and the documents that are missing never entered the candidate pool.
**What the contribution does deliver.** Requirement-level coverage is a measurable, inspectable signal that per-question recall is not (the MuSiQue gap between Recall@10 and all-gold@10 is the clearest illustration), and the mechanism is implemented, tested and ablatable. It does not, at this corpus scale, improve answer quality.

- Retrieval is at parity across P4/P5/P6 by construction: all three start from the same corpus and the same strategy (pooled R@10 = 0.787). Requirement-level control changes *which* retrieved documents reach the generator and how many documents are paid for, not document recall.
- P5's compressor reduces the generation context to roughly 95.9 tokens while adding a compression call; the net token effect is reported in the efficiency table and is not automatically a win.
- On MuSiQue, all-gold@10 is far below Recall@10 — the gap that motivates requirement-level tracking — but closing it requires evidence the corpus retrieval does not surface, which limits how much the controller can recover.

## Figures

| File | Caption |
|---|---|
| `docs/figures/fig1_overall_em_f1.svg` | Answer quality (EM, F1) pooled over the three datasets |
| `docs/figures/fig2_per_dataset_em.svg` | Exact Match by dataset |
| `docs/figures/fig3_cost_quality_pareto.svg` | Cost versus answer quality (log cost axis, cache-normalised) |
| `docs/figures/fig4_input_tokens.svg` | Mean input tokens per question (HotpotQA) |
| `docs/figures/fig5_recall_at_k.svg` | Candidate document Recall@K at fixed depth |
| `docs/figures/fig6_all_gold_at_k.svg` | All-gold-evidence coverage@K |
| `docs/figures/fig7_pandarag_coverage.svg` | PandaRAG evidence coverage before/after recovery |
| `docs/figures/fig8_retrieval_rounds.svg` | PandaRAG retrieval rounds per question |
| `docs/figures/fig9_faithfulness.svg` | Grounding: faithfulness and hallucination score |
| `docs/figures/fig10_ablations.svg` | Ablation: component contributions (reduced sample) |

Figures label systems by name. Retrieval figures plot candidate recall at fixed depth with gaps where a depth is undefined — they do not plot context recall, which is in the retrieval table.

## 11. Limitations

# Limitations

1. **The base paper's corpus is unpublished.** SAC-RAG states only a Milvus collection
   (`en_doc_chunk`, 1536-d, cosine). Our pooled candidate corpus (3.2k–7.2k documents
   per dataset) makes recall a real measurement but is smaller than a Wikipedia-scale
   index. **Our Recall@K is not comparable with the paper's 0.88 / 0.70 / 0.48.**
2. **Neither prompt is published.** The classifier prompt, decomposition prompt and
   compression prompt are implementation choices, labelled as such. Per-type results
   cannot be compared cell-by-cell with the paper's Table 5.
3. **Model substitution.** Generator and compressor are `deepseek-v4.1-flash`; the paper
   uses `gpt-5-mini` for generation and, inconsistently, `qwen-flash` for compression.
   Internal comparisons remain valid; absolute numbers do not transfer.
4. **Embedding substitution.** `bge-small-en-v1.5` (384-d, local) instead of
   `text-embedding-3-small` (1536-d, hosted). Recall is therefore not directly
   comparable and may be pessimistic.
5. **Tokenizer substitution.** Regex tokenizer instead of spaCy for BM25.
6. **Sample size.** 500 questions per dataset instead of 1,000. HotpotQA is hard-only
   2-hop, whereas the paper allows 1–2 hops. Subtype cells are small (Definition: 23 /
   0 / 14 questions), so per-type conclusions are weak for that category.
7. **Frozen classification.** Semantic types come from a classifier frozen before any
   answer was generated; its accuracy is unmeasured because no human labels exist.
8. **LLM evidence evaluator reliability.** The evaluator was assessed against a
   manually reviewable audit set, not against human-annotated support labels at scale.
   Its "supported" verdict is a model judgement, not entailment ground truth.
9. **Compression is extractive and attribution-preserving** — a deliberate choice where
   the paper is silent; a different compressor design would change both the token
   saving and the quality effect.
10. **Cost accounting.** Generation cost is reported separately from auxiliary LLM cost
    (planner, evaluator, compressor). Headline "tokens" in the paper's sense is
    generation-only; our tables expose both, and total tokens per question are higher
    for P5/P6 than for P4.
11. **Cache-adjusted cost.** A replayed run is labelled `from_cache`; live spend is
    reported separately and never conflated with replay cost.
12. **Latency is API-dependent** and measured on one machine with concurrency 3–6; it is
    not a controlled benchmark.
13. **Single generation per question**, temperature 0. No variance across seeds is
    reported, so generator stochasticity is not quantified.
14. **Ablations use a reduced sample** (100 questions per dataset) and are labelled as
    such; they are not mixed with full-sample results.
15. **Recall is reported at each system's retrieval depth.** P1c and P2c retrieve five
    documents, so their `R@k (depth)` column is Recall@5 (0.716 / 0.712 pooled), not
    Recall@10 — the Phase-1 benchmark's 0.771/0.775 belongs to systems that retrieve ten
    (P1n, P3c). P4–P6 stop at 8 documents for Definition questions. Comparing these cells
    as if they were the same metric would overstate the gap between systems.
16. **P2c came in two fusion depths.** The original P2c fused source lists already truncated
    to five; `P2c10` fuses at depth 10 and then truncates, like P3c/P4/P5/P6. Both are reported.
    Against the corrected baseline P6's advantage is smaller (ΔF1 +0.045 vs +0.050), so the
    depth-5 variant was marginally favourable to P6. The original artifacts are archived.
17. **`results/` is not under version control**, so "legacy artifacts unchanged" rests on
    row counts and modification times (2026-09-20, before Scope B), not on a committed hash
    baseline. Content hashes recorded in `final_integrity_audit.md` §9C provide one from now on.
18. **"No detectable difference" is not equivalence.** For P6 vs P4 the pooled ΔF1 is
    −0.000 (95% CI [−0.003, +0.003]). Equivalence at a ±0.02 F1 margin is consistent with
    that interval, but the margin was chosen post hoc and no equivalence test was
    pre-registered, so no equivalence claim is made.
19. **P2c was corrected after the audit.** The original P2c fused source lists already truncated to
    five; it is preserved at `results/baseline_hybrid_corpus/` and byte-identically archived, and a
    corrected depth-10 variant `P2c10` is reported alongside it. P6 vs P2c is ΔF1 +0.050 in the
    original and +0.045 against the corrected baseline — the original comparison slightly overstated
    the gap.
20. **Failure denominators.** 381 superseded HTTP-429 attempts remain in the result files; all 27
    runs answer all 500 questions (0 terminal failures), and EM/F1/retrieval denominators are 500
    everywhere. Retry counts cannot be attributed per system because the error log has no pipeline id.
21. **The recovery rate is evaluator-measured, not ground truth.** The 0–1% conversion rate counts a
    requirement as recovered only when the LLM evaluator changes its status to `supported`. Without a
    human-labelled audit of that evaluator, the rate bounds the controller's *observed* effect under
    this evaluator; it does not establish that no further evidence existed in the corpus. The
    retrieval-side claim (gold documents outside the top-k never entered the candidate pool) is
    measured from gold labels and does not depend on the evaluator.
22. **Ablation A3 and A4 are structurally identical** under the frozen budget (evaluation pool =
    context window), so the ablation measures the evaluator/controller cost without exercising the
    context builder's displacement behaviour. This is a design limitation of the experiment, not a
    bug in the component.


## 12. Conclusion

PandaRAG is a working, tested implementation of requirement-level evidence control with a complete experimental record. The method's contribution is measurable in evidence coverage and cost accounting, and its answer-quality effect is reported honestly, including where it is absent. The corpus and prompt gaps in the base paper mean this is a protocol-level reimplementation; all claims are internal to this harness.
