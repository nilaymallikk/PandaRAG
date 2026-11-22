# Experiment manifest

Generated: 2026-11-20T01:02:33+00:00  
Commit: `fe8ffd08c93d35d59c002aec1e89a2edca99a38e`

| System | Dataset | records | raw rows | config fp | prompt fp | corpus sha | file |
|---|---|---|---|---|---|---|---|
| P0 | hotpotqa | 500 | 516 | `8f9d3c69418e` | `2ea32dfa90d9` | `2da92dd60bdb` | `results/baseline_no_rag/hotpotqa.jsonl` |
| P1c | hotpotqa | 500 | 500 | `8f9d3c69418e` | `2ea32dfa90d9` | `2da92dd60bdb` | `results/baseline_dense_corpus/hotpotqa.jsonl` |
| P2c | hotpotqa | 500 | 536 | `8f9d3c69418e` | `2ea32dfa90d9` | `2da92dd60bdb` | `results/baseline_hybrid_corpus/hotpotqa.jsonl` |
| P2c10 | hotpotqa | 500 | 500 | `8f9d3c69418e` | `2ea32dfa90d9` | `2da92dd60bdb` | `results/baseline_hybrid_corpus_depth10/hotpotqa.jsonl` |
| P3c | hotpotqa | 500 | 526 | `8f9d3c69418e` | `2ea32dfa90d9` | `2da92dd60bdb` | `results/baseline_adaptive_transferred/hotpotqa.jsonl` |
| P1n | hotpotqa | 500 | 537 | `8f9d3c69418e` | `2ea32dfa90d9` | `2da92dd60bdb` | `results/baseline_dense_paper_naive/hotpotqa.jsonl` |
| P4 | hotpotqa | 500 | 500 | `058eec94db66` | `2ea32dfa90d9` | `2da92dd60bdb` | `results/sac_rag_adaptive/hotpotqa.jsonl` |
| P5 | hotpotqa | 500 | 500 | `a24af91bf0e6` | `2ea32dfa90d9` | `2da92dd60bdb` | `results/sac_rag_full/hotpotqa.jsonl` |
| P6 | hotpotqa | 500 | 500 | `42698c5d0256` | `2ea32dfa90d9` | `2da92dd60bdb` | `results/pandarag/hotpotqa.jsonl` |
| P0 | 2wiki | 500 | 527 | `8f9d3c69418e` | `2ea32dfa90d9` | `ad500ff361d7` | `results/baseline_no_rag/2wiki.jsonl` |
| P1c | 2wiki | 500 | 530 | `8f9d3c69418e` | `2ea32dfa90d9` | `ad500ff361d7` | `results/baseline_dense_corpus/2wiki.jsonl` |
| P2c | 2wiki | 500 | 540 | `8f9d3c69418e` | `2ea32dfa90d9` | `ad500ff361d7` | `results/baseline_hybrid_corpus/2wiki.jsonl` |
| P2c10 | 2wiki | 500 | 600 | `8f9d3c69418e` | `2ea32dfa90d9` | `ad500ff361d7` | `results/baseline_hybrid_corpus_depth10/2wiki.jsonl` |
| P3c | 2wiki | 500 | 513 | `8f9d3c69418e` | `2ea32dfa90d9` | `ad500ff361d7` | `results/baseline_adaptive_transferred/2wiki.jsonl` |
| P1n | 2wiki | 500 | 500 | `8f9d3c69418e` | `2ea32dfa90d9` | `ad500ff361d7` | `results/baseline_dense_paper_naive/2wiki.jsonl` |
| P4 | 2wiki | 500 | 500 | `058eec94db66` | `2ea32dfa90d9` | `ad500ff361d7` | `results/sac_rag_adaptive/2wiki.jsonl` |
| P5 | 2wiki | 500 | 514 | `a24af91bf0e6` | `2ea32dfa90d9` | `ad500ff361d7` | `results/sac_rag_full/2wiki.jsonl` |
| P6 | 2wiki | 500 | 500 | `42698c5d0256` | `2ea32dfa90d9` | `ad500ff361d7` | `results/pandarag/2wiki.jsonl` |
| P0 | musique | 500 | 533 | `8f9d3c69418e` | `2ea32dfa90d9` | `7c485109ff42` | `results/baseline_no_rag/musique.jsonl` |
| P1c | musique | 500 | 516 | `8f9d3c69418e` | `2ea32dfa90d9` | `7c485109ff42` | `results/baseline_dense_corpus/musique.jsonl` |
| P2c | musique | 500 | 530 | `8f9d3c69418e` | `2ea32dfa90d9` | `7c485109ff42` | `results/baseline_hybrid_corpus/musique.jsonl` |
| P2c10 | musique | 500 | 500 | `8f9d3c69418e` | `2ea32dfa90d9` | `7c485109ff42` | `results/baseline_hybrid_corpus_depth10/musique.jsonl` |
| P3c | musique | 500 | 515 | `8f9d3c69418e` | `2ea32dfa90d9` | `7c485109ff42` | `results/baseline_adaptive_transferred/musique.jsonl` |
| P1n | musique | 500 | 500 | `8f9d3c69418e` | `2ea32dfa90d9` | `7c485109ff42` | `results/baseline_dense_paper_naive/musique.jsonl` |
| P4 | musique | 500 | 500 | `058eec94db66` | `2ea32dfa90d9` | `7c485109ff42` | `results/sac_rag_adaptive/musique.jsonl` |
| P5 | musique | 500 | 547 | `a24af91bf0e6` | `2ea32dfa90d9` | `7c485109ff42` | `results/sac_rag_full/musique.jsonl` |
| P6 | musique | 500 | 500 | `42698c5d0256` | `2ea32dfa90d9` | `7c485109ff42` | `results/pandarag/musique.jsonl` |

`n_records` counts unique question ids; `raw rows` counts every line written (a resumed question appends a new row and keeps the failed one for the record).
