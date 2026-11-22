# Ablations

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
