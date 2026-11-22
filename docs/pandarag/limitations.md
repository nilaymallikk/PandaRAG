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
