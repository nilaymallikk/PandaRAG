# Adaptive RAG Research

Research project studying query-complexity-aware adaptive retrieval
for multi-hop question answering.

## Systems

- No RAG
- Standard Dense RAG
- Hybrid RAG
- Adaptive RAG

## Dataset

HotpotQA distractor validation.

Fixed 500-question subset.

## LLM

DeepSeek V4.1-Flash through the DeepSeek API.

## Retrieval

- BGE-small-en-v1.5
- FAISS
- BM25
- Rank fusion

## Evaluation

- Exact Match
- F1
- Recall@K
- Supporting-fact retrieval
- Faithfulness
- Hallucination
- Retrieved documents
- Tokens
- Latency
- API cost