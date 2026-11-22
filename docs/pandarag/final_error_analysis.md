# Error analysis

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
