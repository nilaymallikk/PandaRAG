"""SAC-RAG Adaptive (P4) and Full SAC-RAG (P5) pipeline.

One implementation, one flag: ``compression=False`` is P4, ``compression=True`` is
P5. That is deliberate — the only difference between the two systems is the
compressor, so P4-vs-P5 measures compression and nothing else.

Flow::

    question -> frozen semantic-type label -> type-specific retrieval
             -> BGE rerank -> [optional compression]
             -> frozen answer prompt -> DeepSeek -> answer

The classifier label comes from the frozen sidecar (see
``sac_rag.classifier.load_frozen_labels``), so P4/P5 cannot tune classification
against results.

Usage::

    python -m src.pipelines.sac_rag.pipeline --dataset hotpotqa --limit 30
    python -m src.pipelines.sac_rag.pipeline --dataset hotpotqa --compression
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from src import config
from src.datasets import QAExample
from src.experiment_config import fingerprint, load_config
from src.llm import DeepSeekClient
from src.pipelines.sac_rag import classifier, decomposition
from src.pipelines.sac_rag.context_compression import compress
from src.pipelines.sac_rag.retrieval_strategies import retrieve_for_type
from src.pipelines.shared.generator import ContextBlock, generate_answer
from src.pipelines.shared.llm_json import Usage
from src.pipelines.shared.retriever import RetrievalService
from src.pipelines.shared.runner import RunSpec, run_pipeline

PIPELINE_ADAPTIVE = "sac_rag_adaptive"
PIPELINE_FULL = "sac_rag_full"


def load_sac_config() -> dict[str, Any]:
    return load_config(config.REPO_ROOT / "configs" / "sac_rag.json")


def execute_question(
    example: QAExample,
    service: RetrievalService,
    client: DeepSeekClient,
    *,
    semantic_type: str,
    sac_config: dict[str, Any],
    compression_enabled: bool,
) -> dict[str, Any]:
    usage = Usage()
    decomposition_info: dict[str, Any] | None = None

    def _decompose(question: str) -> list[str]:
        nonlocal decomposition_info
        result = decomposition.decompose(client, question, usage=usage)
        decomposition_info = result.as_dict()
        return result.subquestions

    strategy = retrieve_for_type(
        service,
        example.question,
        semantic_type,
        config=sac_config,
        decompose_fn=_decompose,
    )

    context_documents = [
        {"doc_id": document.doc_id, "title": document.title, "text": document.text}
        for document in strategy.selected
    ]
    compression_info: dict[str, Any] | None = None
    if compression_enabled:
        compressed = compress(client, example.question, strategy.selected, usage=usage)
        context_documents = compressed.documents
        compression_info = compressed.as_dict()

    block = ContextBlock(documents=context_documents, max_chars=None)
    generation = generate_answer(client, example.question, block, usage=usage)

    payload: dict[str, Any] = {
        "prediction": generation.prediction,
        "retrieved_documents": [document.as_record() for document in strategy.selected],
        "candidate_document_ids": [document.doc_id for document in strategy.ranked],
        "context_documents": [
            {"doc_id": document["doc_id"], "title": document["title"]}
            for document in context_documents
        ],
        "context_text": block.text,
        "context_tokens_estimate": block.tokens_estimate,
        "retrieval_rounds": 1,
        "retrieval_executions": strategy.retrieval_executions,
        "rerank_latency_seconds": strategy.rerank_latency_seconds,
        "generation_latency_seconds": generation.latency_seconds,
        "usage": usage,
        "error": generation.error,
        "sac_rag": {
            "semantic_type": semantic_type,
            "classifier": classifier.load_frozen_labels.__name__,
            "strategy": strategy.as_dict(),
            "decomposition": decomposition_info,
            "compression": compression_info,
        },
    }
    return payload


def build(
    *,
    dataset: str,
    split: str | None = None,
    limit: int | None = None,
    compression_enabled: bool = False,
    concurrency: int = 4,
    incremental_retrieval: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    sac_config = load_sac_config()
    if compression_enabled:
        sac_config["compression"]["enabled"] = True
    pipeline_name = PIPELINE_FULL if compression_enabled else PIPELINE_ADAPTIVE

    labels = classifier.load_frozen_labels(dataset)

    def question_fn(example: QAExample, service: RetrievalService, client: DeepSeekClient) -> dict[str, Any]:
        label = labels.get(example.example_id, {"question_type": classifier.FALLBACK_LABEL})
        return execute_question(
            example,
            service,
            client,
            semantic_type=label["question_type"],
            sac_config=sac_config,
            compression_enabled=compression_enabled,
        )

    spec = RunSpec(
        pipeline=pipeline_name,
        dataset=dataset,
        split=split,
        limit=limit,
        concurrency=concurrency,
        config_fingerprint=fingerprint(sac_config),
        extra_meta={
            "pipeline_family": "sac_rag",
            "compression": bool(compression_enabled),
            "retrieval_incremental": bool(incremental_retrieval),
            "resolved_sac_rag_config": sac_config,
        },
    )
    results_file = config.RESULTS_DIR / pipeline_name / f"{dataset}.jsonl"
    spec.results_file = results_file
    spec.meta_file = results_file.with_suffix(".meta.json")
    return run_pipeline(
        spec,
        question_fn,
        semantic_types={key: value["question_type"] for key, value in labels.items()},
        quiet=quiet,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SAC-RAG Adaptive / Full SAC-RAG")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compression", action="store_true", help="enable P5 compression")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    build(
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        compression_enabled=args.compression,
        concurrency=args.concurrency,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
