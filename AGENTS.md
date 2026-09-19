# AGENTS.md

## Project

This is a research project investigating:

Query-Complexity-Aware Adaptive Retrieval-Augmented Generation
for Multi-Hop Question Answering.

The project compares four independent QA systems:

1. No RAG
2. Standard Dense RAG
3. Hybrid RAG
4. Adaptive RAG

The goal is to determine whether adaptive retrieval can maintain
answer quality while reducing retrieval and computational cost.

---

## Dataset

Dataset:

HotpotQA

Split:

Distractor validation

Fixed dataset:

data/hotpotqa_500.json

The dataset has already been manually selected.

IMPORTANT:

- Do NOT download another dataset.
- Do NOT resample the dataset.
- Do NOT shuffle the dataset.
- Do NOT modify gold answers.
- Do NOT add or remove questions.
- Treat data/hotpotqa_500.json as immutable experiment input.

The experiment must use exactly these 500 questions.

---

## LLM

Provider:

DeepSeek API

Model:

DeepSeek-V4.1-Flash

Base URL:

https://api.deepseek.com

The API key must come from the environment:

DEEPSEEK_API_KEY

Never hard-code or print the API key.

Do not use OpenRouter.

Do not use another LLM unless explicitly instructed.

All four pipelines must use the same LLM model.

---

## Local Retrieval

Retrieval must run locally.

Dense retrieval:

- sentence-transformers
- BGE-small-en-v1.5
- FAISS

Lexical retrieval:

- BM25

Hybrid retrieval:

- Dense retrieval + BM25
- simple rank fusion

Do not introduce a hosted vector database.

Do not introduce LangChain or LlamaIndex unless explicitly requested.

---

# Four Pipelines

## 1. No RAG

Question
→ DeepSeek
→ Answer

No retrieval.

No context from the HotpotQA documents is supplied.

---

## 2. Standard RAG

Question
→ Dense Retriever
→ Fixed K
→ Context
→ DeepSeek
→ Answer

Initial experimental K:

K = 5

K must remain fixed for this pipeline.

---

## 3. Hybrid RAG

Question
→ Dense Retrieval + BM25
→ Rank Fusion
→ Fixed K
→ DeepSeek
→ Answer

Initial experimental K:

K = 5

K must remain fixed for this pipeline.

---

## 4. Adaptive RAG

Question
→ Query Complexity Analysis
→ Initial Retrieval Budget
→ Retrieval
→ Evidence/Confidence Check
→ Possibly retrieve more
→ DeepSeek
→ Answer

The adaptive pipeline is the proposed research method.

Do not make it unnecessarily complicated.

Start with a transparent rule-based policy.

The policy may use:

- query complexity
- retrieval score
- evidence coverage
- supporting evidence
- confidence

The adaptive system must record why additional retrieval was performed.

---

# Important Experimental Rules

1. Keep the four pipelines separate.

2. Use the same 500 questions for every pipeline.

3. Use the same DeepSeek model for every pipeline.

4. Keep prompts as consistent as possible across pipelines.

5. Do not tune one pipeline unfairly after seeing its results.

6. Do not modify evaluation rules after seeing results.

7. Do not fabricate or manually edit results.

8. Failed API requests must be logged, not silently removed.

9. Preserve raw results.

10. Cache API responses where practical.

11. Never expose API keys in logs.

12. Keep the implementation simple.

13. Avoid unnecessary abstractions and frameworks.

14. Do not create databases or web services.

15. Do not introduce Docker unless explicitly requested.

---

# Evaluation

Every pipeline must produce a compatible result record.

Record at minimum:

- question_id
- question
- gold_answer
- predicted_answer
- retrieval_k
- retrieved_documents
- retrieval scores
- retrieval recall
- exact match
- F1
- faithfulness/groundedness
- hallucination indicator
- input tokens
- output tokens
- LLM calls
- latency
- API cost
- error status

---

# Retrieval Evaluation

HotpotQA supporting facts should be used to evaluate retrieval.

Measure:

- Recall@1
- Recall@3
- Recall@5
- Recall@10 where applicable
- supporting-fact retrieval

For No RAG, retrieval metrics are not applicable.

Do not report them as zero.

Use null/NA.

---

# Answer Evaluation

Use the HotpotQA gold answer.

Calculate:

- Exact Match
- F1

Do not use an LLM to decide basic answer correctness when deterministic
evaluation is possible.

---

# Faithfulness / Hallucination

The generated answer should be evaluated against the retrieved context.

Distinguish:

1. Answer correctness
2. Evidence retrieval
3. Groundedness/faithfulness

Do not automatically equate an incorrect answer with hallucination.

A hallucination/unsupported claim means the answer contains information
that is not supported by the supplied retrieval context.

---

# Cost

Calculate API cost from actual token usage returned by DeepSeek.

Record:

- input tokens
- output tokens
- cached input tokens if available
- total cost

Do not estimate cost from character count.

Use actual API usage whenever available.

---

# Reproducibility

Record:

- model
- model parameters
- prompts
- retrieval method
- K
- embedding model
- dataset filename
- dataset size
- random seeds if used
- timestamp
- token usage
- latency
- cost

Do not change the 500-question dataset.

---

# Development Order

Implement in this exact order:

1. Verify dataset structure
2. Verify DeepSeek API connection
3. Implement shared LLM client
4. Implement dense retrieval
5. Implement BM25
6. Implement retrieval evaluation
7. Implement No-RAG
8. Implement Standard RAG
9. Implement Hybrid RAG
10. Implement Adaptive RAG
11. Implement unified evaluation
12. Run a small smoke test
13. Run the full 500-question experiment
14. Analyze results

Do not implement the entire project in one step.

After each major stage, run a small test.

---

# Research Integrity

The purpose is to conduct a reproducible experiment.

Do not optimize the system specifically to obtain a desired result.

If Adaptive RAG performs worse, record the result honestly.

Do not change the evaluation methodology after seeing results.

When uncertain about an experimental decision, explain the issue
before making a major change.