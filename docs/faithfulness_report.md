# Secondary Evaluation: Faithfulness and Hallucination

Read-only groundedness audit of the frozen RAG outputs. It does not modify any pipeline,
prompt, dataset, retrieval artifact, Adaptive policy or existing result.

- Protocol: `faithfulness-judge/1.0` (definitions frozen before the first judge call)
- Judge: `deepseek-flash`, thinking `disabled`, temperature `0.0`, max_tokens 768, one call per answer
- Pipelines judged: standard_rag, hybrid_rag, adaptive_rag; answers audited: **1500**
- Evidence available to the judge: question, the exact retrieved context block, the answer. Gold answers, supporting facts, question type, level, EM/F1, pipeline identity and K are never passed in.

## 1. Definitions and scoring

| Term | Definition |
|---|---|
| Faithfulness | `supported_claims / total_claims`, in [0,1]; claims that are true in the real world but absent from the context count as unsupported |
| Hallucination | an answer claim labelled `unsupported` (absent from context) or `contradicted` (context states the opposite) |
| Hallucination score | `(unsupported + contradicted) / total_claims` per answer |
| Hallucination rate | fraction of answers with at least one unsupported/contradicted claim |
| Grounding | `fully` (all claims supported), `partially` (some), `not` (none), `no_checkable_claims` (excluded from faithfulness means) |

Scores are computed in code from the judge's claim labels; the judge's own arithmetic is never trusted.

## 2. Overall and per-pipeline

| Pipeline | Faithfulness | Hallucination score | Hallucination rate | Fully grounded | Partially | Not grounded | No claims | Judge errors |
|---|---|---|---|---|---|---|---|---|
| standard_rag | 0.8511 | 0.1489 | 0.1826 | 0.8174 | 0.0588 | 0.1237 | 7 | 0 |
| hybrid_rag | 0.8078 | 0.1922 | 0.2334 | 0.7666 | 0.0785 | 0.1549 | 2 | 1 |
| adaptive_rag | 0.7870 | 0.2130 | 0.2618 | 0.7382 | 0.0920 | 0.1697 | 11 | 0 |
| **All three** | 0.8154 | 0.1846 | 0.2258 | 0.7742 | 0.0764 | 0.1494 | 20 | 1 |

Claim level: 2229 claims audited, 0.8398 supported, 0.0094 contradicted.

Standard vs Hybrid vs Adaptive:

| Pipeline | Faithfulness | Hallucination rate | Mean answer tokens | EM (reference, not used by the judge) | F1 (reference) |
|---|---|---|---|---|---|
| standard_rag | 0.8511 | 0.1826 | 3.88 | 0.5100 | 0.6925 |
| hybrid_rag | 0.8078 | 0.2334 | 3.84 | 0.4900 | 0.6764 |
| adaptive_rag | 0.7870 | 0.2618 | 4.12 | 0.4820 | 0.6598 |

Pairwise differences on the paired answers (bootstrap 95% CI of the mean faithfulness difference):

| Comparison | Mean faithfulness delta | 95% CI |
|---|---|---|
| adaptive_rag - hybrid_rag (n=488) | -0.0270 | [-0.0478, -0.0065] |
| adaptive_rag - standard_rag (n=486) | -0.0674 | [-0.1000, -0.0350] |
| hybrid_rag - standard_rag (n=492) | -0.0388 | [-0.0686, -0.0097] |

## 3. Bridge vs comparison

| Slice | Pipeline | n | Faithfulness | Hallucination rate | Fully grounded |
|---|---|---|---|---|---|
| bridge | standard_rag | 400 | 0.8694 | 0.1552 | 0.8448 |
| bridge | hybrid_rag | 400 | 0.8098 | 0.2217 | 0.7783 |
| bridge | adaptive_rag | 400 | 0.7922 | 0.2416 | 0.7584 |
| comparison | standard_rag | 100 | 0.7792 | 0.2900 | 0.7100 |
| comparison | hybrid_rag | 100 | 0.8000 | 0.2800 | 0.7200 |
| comparison | adaptive_rag | 100 | 0.7667 | 0.3400 | 0.6600 |

## 4. Adaptive K=3 vs K=5

| Adaptive path | n | Faithfulness | Hallucination rate | Fully grounded | Mean context docs |
|---|---|---|---|---|---|
| K=3 | 371 | 0.7971 | 0.2465 | 0.7535 | 2.99 |
| K=5 | 129 | 0.7585 | 0.3047 | 0.6953 | 4.98 |

## 5. Relationship with answer quality (reference only)

| Pipeline | Faithfulness when EM=1 | Faithfulness when EM=0 | Spearman rho (faithfulness vs F1) |
|---|---|---|---|
| standard_rag | 0.8827 (n=255) | 0.8172 (n=238) | +0.1732 |
| hybrid_rag | 0.8337 (n=245) | 0.7827 (n=252) | +0.1170 |
| adaptive_rag | 0.8254 (n=241) | 0.7497 (n=248) | +0.1520 |

Interpretation guard: high faithfulness does not imply correctness (a context-faithful answer can
faithfully repeat the wrong document), and an incorrect answer is not automatically a hallucination.

## 6. Faithfulness vs retrieval success

Buckets use each pipeline's own retrieval metric (`retrieval_complete`: all gold documents were
retrieved). This field was never shown to the judge; the split is a diagnostic, not a judge input.

| Pipeline | Retrieval | n | Faithfulness | Hallucination rate |
|---|---|---|---|---|
| standard_rag | gold docs complete | 431 | 0.8975 | 0.1276 |
| standard_rag | gold docs incomplete | 62 | 0.5282 | 0.5645 |
| hybrid_rag | gold docs complete | 392 | 0.8999 | 0.1301 |
| hybrid_rag | gold docs incomplete | 105 | 0.4643 | 0.6190 |
| adaptive_rag | gold docs complete | 336 | 0.9107 | 0.1161 |
| adaptive_rag | gold docs incomplete | 153 | 0.5153 | 0.5817 |

## 7. Instrument validation (judge behaviour)

- **Judge/parse failures:** 1/1500 (0.07%), recorded as `grounding_label="judge_error"` and excluded from means.
  - hybrid_rag/5a72a9ab5542992359bc315a: JSONDecodeError: Expecting ',' delimiter: line 1 column 482 (char 481)
- **Abstention answers:** 20 answers returned no checkable claim, almost all of the form "The context does not identify …". Per protocol these meta-statements are not scored, so an abstention is *excluded from faithfulness means* rather than counted as unfaithful.
- **Determinism across identical inputs:** for the 129 Adaptive K=5 answers that are byte-identical (question, context, answer) to Hybrid RAG, the judgments disagree in **0** cases.
- **Strictness audit:** 336 claims were labelled unsupported; 181 (53.9%) share ≥80% of their content words with the context, an *upper bound* on possible false positives (word overlap is not entailment). Manual inspection of the highest-overlap cases found the judge defensible: e.g. the context says Márquez only *challenged for* a WBO title, and that a "Johnny Edwards" was a *singer*, so the claims 'held a WBO championship' and 'was a guitarist' are genuinely unsupported.

## 8. Representative examples

### standard_rag

- **5a7108fa5542994082a3e4ef** (comparison, fully, faithfulness 1.0000, n_claims 1)
  - Q: Which battle, the Battle of Cold Harbor, or the Second Battle of Bull Run, was fought first?
  - Answer: The Second Battle of Bull Run was fought first.
  - Supported claim: "The Second Battle of Bull Run was fought before the Battle of Cold Harbor." (evidence: Second Battle of Bull Run ... fought August 28–30, 1862; Battle of Cold Harbor was fought from May 31 to June 12, 1864)
- **5a716dc15542994082a3e82b** (bridge, not, faithfulness 0.0000, n_claims 1)
  - Q: Who is younger Kirk Humphreys or Tom Coburn?
  - Answer: Kirk Humphreys
  - Contradicted claim: "Kirk Humphreys is younger than Tom Coburn." (evidence: Kirk Humphreys (born September 13, 1950) ... Thomas Allen "Tom" Coburn (born March 14, 1948))

### hybrid_rag

- **5a7108fa5542994082a3e4ef** (comparison, fully, faithfulness 1.0000, n_claims 1)
  - Q: Which battle, the Battle of Cold Harbor, or the Second Battle of Bull Run, was fought first?
  - Answer: The Second Battle of Bull Run was fought first.
  - Supported claim: "The Second Battle of Bull Run was fought before the Battle of Cold Harbor." (evidence: Second Battle of Bull Run ... fought August 28–30, 1862; Battle of Cold Harbor was fought from May 31 to June 12, 1864)
- **5a7129685542994082a3e5fa** (bridge, partially, faithfulness 0.5000, n_claims 2)
  - Q: Which "Blackzilians" fighter is currently competing in the Middleweight division of Ultimate Fighting Championship?
  - Answer: Rashad Evans
  - Unsupported claim: "Rashad Evans is a "Blackzilians" fighter." (evidence: none)
  - Supported claim: "Rashad Evans is currently competing in the Middleweight division of the Ultimate Fighting Championship." (evidence: Rashad Anton Evans ... is an American mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting Championship.)

### adaptive_rag

- **5a7108fa5542994082a3e4ef** (comparison, fully, faithfulness 1.0000, n_claims 1)
  - Q: Which battle, the Battle of Cold Harbor, or the Second Battle of Bull Run, was fought first?
  - Answer: The Second Battle of Bull Run was fought first.
  - Supported claim: "The Second Battle of Bull Run was fought before the Battle of Cold Harbor." (evidence: Second Battle of Bull Run ... fought August 28–30, 1862; Battle of Cold Harbor was fought from May 31 to June 12, 1864)
- **5a71166d5542994082a3e576** (comparison, not, faithfulness 0.0000, n_claims 1)
  - Q: Which battle occurred first, the Battle of Manila or the Battle of Guam?
  - Answer: The Battle of Guam (1944) occurred first.
  - Unsupported claim: "The Battle of Guam (1944) occurred before the Battle of Manila." (evidence: none)

## 9. Additional evaluation cost

| Metric | Value |
|---|---|
| Judge calls (entries) | 1500 |
| Live API calls | 1346 |
| Replayed from cache | 154 |
| Input tokens | 1,324,575 |
| Output tokens | 185,990 |
| Cached input tokens (server-side cache hits) | 145,664 |
| Judging cost (USD, shared pricing convention) | $0.28886764 |
| Mean latency (s) | 0.9878 |
| Median / p95 latency (s) | 1.0339 / 1.4317 |

This is **additional spend** created solely by this secondary evaluation; it does not change any
primary experiment number. Re-running the same protocol reuses the disk cache at no API cost.

## 10. Limitations and possible judge errors

- **The judge is itself an LLM and can be wrong.** It may miss a claim, split one claim into
  several, or label a paraphrase as unsupported. Parse/scoring failures are recorded as
  `grounding_label="judge_error"` rather than dropped, and are reported above.
- **Claim granularity is judge-dependent**, so faithfulness is a ratio of the judge's decomposition;
  longer answers get more claims, which is why claim-level and answer-level rates are both reported.
- **Grounding is not correctness.** A faithful answer may faithfully reproduce a wrong document;
  conversely an answer that is correct from parametric knowledge but absent from the context is
  scored as unsupported by definition. The EM/F1 columns are shown only as a reference.
- **Abstentions are excluded, not penalised.** Answers that say the context lacks the information
  produce no claims and are dropped from faithfulness means; 20 such answers exist (11 Adaptive, 7
  Standard, 2 Hybrid), so pipelines that hedge more are measured on fewer answers.
- **The judge is strict rather than lenient.** Sampled high-overlap unsupported claims were
  genuinely unsupported (competition names, titles, and people conflated by the QA model); if
  anything, the reported hallucination rates are more likely to be over- than under-stated.
- **No-RAG is excluded** because it received no retrieved context, so faithfulness is undefined for it.
- **Question wording is passed to the judge** to help claim extraction; it is not gold evidence, but
  it is an input beyond the context and is disclosed here.
- Comparisons across pipelines are descriptive; the three pipelines answer the same questions, but
  faithfulness differences are confounded with answer length and retrieval quality.

