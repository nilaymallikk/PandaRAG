# Experiment Specification

## Research Question

Can query-complexity-aware adaptive retrieval maintain answer quality
while reducing retrieval and computational cost in multi-hop question answering?

## Dataset

HotpotQA

Split:
distractor / validation

File:

data/hotpotqa_500.json

Questions:

500

The dataset is fixed and must not be modified.

---

# Systems

## A. No RAG

Question → DeepSeek → Answer

No retrieved context.

## B. Standard RAG

Question → Dense Retrieval → Top 5 → DeepSeek → Answer

## C. Hybrid RAG

Question → Dense + BM25 → Fusion → Top 5 → DeepSeek → Answer

## D. Adaptive RAG

Question
→ Complexity estimation
→ Dynamic retrieval
→ Evidence check
→ Additional retrieval when necessary
→ DeepSeek
→ Answer

---

# Embedding

Model:

BAAI/bge-small-en-v1.5

Run locally.

---

# Retrieval

Dense:

FAISS

Lexical:

BM25

Hybrid:

Dense + BM25 with rank fusion.

---

# Primary Metrics

### Answer Quality

- Exact Match
- F1

### Retrieval

- Recall@1
- Recall@3
- Recall@5
- Recall@10
- Supporting-fact retrieval

### Groundedness

- Faithfulness
- Hallucination / unsupported claim rate

### Efficiency

- Average retrieved documents
- Input tokens
- Output tokens
- LLM calls
- Latency
- API cost

---

# Main Comparison

The primary comparison is:

Answer quality
vs.
retrieval/computational cost.

The goal is to determine whether Adaptive RAG can achieve competitive
answer quality with less retrieval and API cost.

---

# Initial Parameters

Standard RAG:

K = 5

Hybrid RAG:

K = 5

Adaptive RAG:

Dynamic K

Initial adaptive policy should remain simple and interpretable.

---

# LLM

Provider:

DeepSeek

Model:

deepseek-flash

All systems use the same model.

---

# Experimental Stages

1. Retrieval-only evaluation
2. Small smoke test
3. Full 500-question experiment
4. Metric aggregation
5. Error analysis
6. Ablation study