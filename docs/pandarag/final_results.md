# Results

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
