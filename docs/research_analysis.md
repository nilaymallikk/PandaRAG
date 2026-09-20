# Research Analysis: Four Frozen Pipelines on HotpotQA-500

**Scope.** Read-only analysis of the four frozen pipelines (`results/{no_rag,standard_rag,hybrid_rag,adaptive_rag}.jsonl`,
`results/retrieval/*.jsonl`) on the immutable 500-question benchmark
(SHA-256 `956405ce…fd048`, verified unchanged). No code, prompt, dataset, retrieval artifact,
policy or result was modified. The Adaptive policy constants are those pre-registered in
`src/config.py`; nothing was tuned after seeing these numbers.

Derivation recipes (all recomputable from the raw JSONL): EM/F1 are the stored per-record
scores; per-question win/loss = adaptive `exact_match` vs baseline `exact_match`; the
"containment" metric = all tokens of the gold answer appear in the normalized prediction
(lowercased `[a-z0-9]+`), a verbosity-robust cross-check on exact match.

---

## 1. Headline results

| Pipeline | EM | F1 | Input tok | Cost (USD) | Calls | Mean docs |
|---|---|---|---|---|---|---|
| No-RAG | 0.3520 | 0.4756 | 31,120 | 0.00600120 | 490 | 0 |
| Standard Dense RAG (K=5) | **0.5100** | **0.6925** | 357,810 | 0.05547390 | 490 | 4.98 |
| Hybrid RAG (K=5) | 0.4900 | 0.6764 | 367,325 | 0.05096271 | 474 | 4.98 |
| Adaptive RAG (K∈{3,5}) | 0.4820 | 0.6598 | **263,640** | **0.02709842** | 360 | 3.51 |

Adaptive vs Hybrid: **−28.2% input tokens, −46.8% cost, EM retention 98.4%, F1 retention 97.6%**.
Adaptive vs Standard: −26.3% tokens, −51.2% cost, EM 0.4820 vs 0.5100.
Adaptive vs No-RAG: +747% tokens, +4.5× cost, **EM +0.1300 absolute (+36.9% relative)**.

---

## 2. Per-question wins / losses / ties (EM)

| Comparison | Adaptive-only correct (win) | Baseline-only correct (loss) | Both correct | Both wrong | McNemar p |
|---|---|---|---|---|---|
| Adaptive vs No-RAG | 106 | 41 | 135 | 218 | 8.1e‑08 |
| Adaptive vs Standard RAG | 22 | 36 | 219 | 223 | 0.087 (F1 95% CI excludes 0) |
| Adaptive vs Hybrid RAG | 13 | 17 | 228 | 242 | 0.585 (F1 95% CI [−0.037, +0.003]) |
| Hybrid vs Standard RAG | 21 | 31 | 224 | 224 | — |

Reading: the Adaptive–Hybrid difference is **statistically indistinguishable at n=500**; the
Adaptive–Standard gap is borderline on EM and small on F1; the retrieval gain over No-RAG is
unambiguous. **Every Adaptive–Hybrid difference occurs inside the 371 K=3 questions** — on the
129 K=5 questions the two pipelines saw identical contexts and produced **129/129 identical
predictions** (a strong internal-consistency check: Adaptive is a context-budget variant, not a
different system).

---

## 3. Where Adaptive improves vs Hybrid, and where it fails

On the 371 K=3 questions: Adaptive EM 0.4960 / F1 0.6684 vs Hybrid-on-the-same-questions
0.5067 / 0.6908 (ΔEM −0.0108, ΔF1 −0.0223). The 30 discordant questions split:

**The 17 Hybrid-only wins are mostly real information loss.** In **11 of 17**, the K=3 truncation
dropped a gold document that the K=5 context contained (`retrieval_complete` False at 3, True at 5).
Adaptive's own answer was usually an explicit abstention or a distractor. This is the concrete,
attributable price of the smaller budget.

**The 13 Adaptive-only wins are mostly verbosity effects, not accuracy gains.** In **7 of 13** the
Hybrid prediction already *contained* the gold answer but failed strict EM due to extra wording;
Adaptive's mean prediction was 2.2 tokens vs Hybrid's 5.6 on those questions. Only 2 of 13
coincided with a dropped gold document. EM here is measuring answer *style* as much as answer
*content*.

---

## 4. K=3 vs K=5 behavior

| | n | Adaptive EM | Hybrid-on-same EM | Mean input tok | Cost | Cost/question |
|---|---|---|---|---|---|---|
| K=3 | 371 (74.2%) | 0.4960 | 0.5067 | 455.1 | $0.012938 (47.7%) | $0.00003487 |
| K=5 | 129 (25.8%) | 0.4419 | 0.4419 | 734.8 | $0.014160 (52.3%) | $0.00010977 |

Two distinct facts must not be conflated:

1. **The budget reduction is cheap in aggregate** (ΔEM −0.0108 on the K=3 subset) but concentrated:
   a handful of questions lose the gold document entirely.
2. **The K=3 subset scores higher in absolute EM than the K=5 subset (+0.054)** — this is
   **selection, not a benefit of fewer documents**. Within the K=3 subset, truncation still *lowers*
   EM relative to the identical K=5 context. So the policy is selecting a slightly easier subset
   (retrievers agree on the right top-1), not making questions easier by retrieving less.

The escalation rule is agreement-dominated:

| Fused top-1 (dense, bm25) | →K=3 | →K=5 | Share at K=3 |
|---|---|---|---|
| (1,1) | 264 | 0 | 100% |
| (1,2) | 42 | 25 | 63% |
| (2,1) | 33 | 21 | 61% |
| (3,1) | 13 | 11 | 54% |
| (2,2) | 4 | 11 | 27% |
| (4,1) | 0 | 9 | 0% |

Escalation is triggered by *retriever disagreement*; the boundary-margin term only breaks ties
(margin is 0 whenever the pool < 4, and RRF scores are quantized to ~13 levels). Cost-wise,
the 26% of questions escalated to K=5 consume **52.3%** of the total Adaptive spend
(3.15× the per-question cost of the K=3 path).

---

## 5. Bridge vs comparison questions

| Slice | No-RAG | Standard | Hybrid | Adaptive |
|---|---|---|---|---|
| Bridge (n=400) EM | 0.2950 | 0.5400 | 0.5025 | 0.4975 |
| Comparison (n=100) EM | **0.5800** | 0.3900 | 0.4400 | 0.4200 |
| Comparison F1 | 0.6811 | 0.5400 | 0.5785 | 0.5515 |
| Comparison containment | 0.6600 | 0.7400 | 0.7500 | 0.7100 |

- **Bridge:** retrieval is essential — Adaptive 100 wins vs 19 losses over No-RAG (p≈1.8e‑14).
- **Comparison:** under strict EM, No-RAG beats *every* RAG pipeline, and Adaptive loses to No-RAG
  22–6 (p=0.0037). **This inversion is substantially an evaluation artifact.** Comparison gold
  answers are very short (mean 1.79 tokens: "no", "yes", a single name), while context-conditioned
  answers tend to be explanatory sentences. Of the 19 comparison questions where No-RAG was "correct"
  and Hybrid "wrong", **18 of 19 Hybrid predictions actually contain the gold answer**
  (e.g. gold `no`, prediction "No. Lucy Saroyan was an actress and photographer, …"). Under the
  verbosity-robust containment metric, all RAG pipelines (0.71–0.75) beat No-RAG (0.66) on
  comparison questions too.

**Consequence:** the "retrieval hurts comparison questions" result should be reported as
**conditional on exact-match scoring**, with the containment analysis as the counter-evidence.

**Prediction verbosity (mean predicted tokens):** No-RAG 2.9, Standard 4.0, Hybrid 3.9, Adaptive 4.2 —
i.e. all RAG pipelines answer more verbosely, and Adaptive is the most verbose, which penalizes it
under EM and inflates its output-token count relative to its shorter context.

---

## 6. Representative examples

**Adaptive-only win (verbosity of the 5-doc answer):** *"Which of the 2017-18 Cheshire League
divisions is Ashton Town AFC currently a member?"* — gold `Premier Division`; Hybrid@5 returned
"Cheshire League Premier Division" (contains the answer, fails EM), Adaptive@3 returned
"Premier Division". Gold documents present at both K=3 and K=5. **No information advantage.**

**Hybrid-only win (genuine truncation loss):** *"What North Carolina native did Danja produce songs
for?"* — gold `J. Cole`; Hybrid@5 "J. Cole", Adaptive@3 "Colonel Alexander Boyd Andrews".
Doc recall 0.5 at K=3 vs 1.0 at K=5; doc-complete lost. **The third document held the evidence.**

**Adaptive wins where No-RAG also wins (comparison):** *"Are Trent Edwards and Lucy Saroyan both
football players?"* — gold `no`; No-RAG "No"; all RAG pipelines "No. Lucy Saroyan was an actress…".
Same answer, different length; EM splits them.

**Escalation did not rescue (K=5, both wrong):** *"Which 'Blackzilians' fighter is currently
competing in the Middleweight division of the UFC?"* — gold `Vitor Belfort`; No-RAG and Standard
were correct, Hybrid and Adaptive returned "Rashad Evans". Doc recall 0.5 at both K=3 and K=5 and
doc-complete False: the gold document was never retrieved, so escalating K could not help. This is
the policy's blind spot — it spends the larger budget on questions whose evidence is *outside* the
fused pool, while the cheap path could have sufficed (No-RAG here).

---

## 7. Quality vs token/cost trade-off

| | vs Hybrid K=5 | vs Standard K=5 | vs No-RAG |
|---|---|---|---|
| Cost | **−46.8%** | −51.2% | +4.5× |
| Input tokens | −28.2% | −26.3% | +8.5× |
| EM | −0.0080 (−1.6% rel.) | −0.0280 (−5.5% rel.) | **+0.1300** |
| F1 | −0.0166 | −0.0327 | +0.1843 |
| Cost per correct answer | $0.00011244 vs $0.00020801 | $0.00021754 | $0.00003410 |

Adaptive is the **most cost-efficient retrieval pipeline** on both cost-per-question and
cost-per-correct-answer. The efficiency gain comes from two additive sources: (a) 74% of questions
use 3 documents; (b) the same frozen context prompts were partly served from the response cache
(140/500 replays, 97,408 cached input tokens), so part of the dollar saving is cache reuse rather
than the retrieval policy. Retrieval computation itself is sunk (the frozen hybrid artifact is
reused), so the measured saving is a *context length* saving, not an end-to-end retrieval saving.

---

## 8. What the results actually support

**Supported:**
1. Retrieval, not model knowledge, drives bridge-question accuracy: every RAG pipeline roughly
   doubles No-RAG on bridge questions (0.50–0.54 vs 0.295; +0.13 EM overall, p≈1e‑7).
2. A confidence-gated budget that uses the full 5-document context only when the retrievers
   disagree retains **98.4% of Hybrid RAG's EM and 97.6% of its F1 at 53% of the cost**. The
   Adaptive–Hybrid difference is not statistically distinguishable at this sample size.
3. The pre-registered agreement signal does carry *weak* information about answerability:
   the K=3 subset is intrinsically easier (EM 0.496 vs 0.442), consistent with retriever agreement
   being mildly diagnostic — but its AUC for correctness is only **0.52 (bridge 0.516,
   comparison 0.524)**, i.e. barely above chance.
4. Reducing context from 5 to 3 documents has a **small mean cost (ΔEM −0.0108)** that is
   **concentrated**: in 11/17 discordant losses a gold document was removed by truncation.

**Not supported:**
1. That K=3 is *better* than K=5 — the higher absolute score of the K=3 subset is selection, and
   within the subset truncation lowers EM.
2. That adaptive retrieval improves answer quality over fixed-K RAG. On this benchmark it does not;
   it trades a small, partly evaluation-artifact-driven quality difference for a large cost saving.
3. That the confidence score is a calibrated probability of answering correctly; it is an ordinal
   agreement/margin heuristic.
4. That retrieval hurts comparison questions — that conclusion is an exact-match artifact; under
   containment, RAG helps there too.
5. Anything about faithfulness/groundedness: those fields are `null` in all pipelines by design
   and no claim is made.

---

## 9. Limitations

- **Single run per pipeline** (temperature 0.0, but hosted-model nondeterminism is possible);
  no seed-variance estimate, no multiplicity correction across the many slices reported here.
- **Metric sensitivity.** A material fraction of per-question wins/losses, and the entire
  comparison-question inversion, flip under a verbosity-robust containment check. All pipeline
  conclusions are therefore stated as *conditional on the exact-match/F1 evaluator*, with
  containment reported alongside.
- **No counterfactual for the confidence rule itself.** We can compare Adaptive@3 vs Hybrid@5 on
  the questions the policy assigned to K=3 (done above), but we cannot know what K=3 would have
  done on the questions it escalated to K=5 — those were answered at K=5 only.
- **Efficiency is context-length and cache, not end-to-end retrieval cost.** Dense/BM25/FAISS
  indexing and the hybrid artifact are reused; retrieval latency and compute are excluded.
- **Cache provenance.** 140 Adaptive prompts were replayed from earlier smoke runs of the identical
  frozen policy; the answers are unaffected, but cost/latency for those questions reflect cache
  replay.
- **No qualitative error taxonomy was audited beyond the examples shown**; the 11/17 truncation
  attribution and the 7/13 verbosity finding are counts over those 30 discordant questions, not a
  full manual review of all 500.
