"""PandaRAG pipeline (P6): evidence-gap-aware retrieval-budget allocation.

Flow::

    question -> frozen semantic-type label
             -> evidence requirement plan (1 LLM call for Reasoning; 1 requirement
                without a call for Fact/Definition)
             -> SAC-RAG type-specific initial retrieval (planner queries drive
                Reasoning retrieval, so decomposition is not paid for twice)
             -> BGE rerank -> admit candidates (budgeted)
             -> evidence evaluation per requirement  <----+
             -> coverage summary                            |
             -> gap controller ---------------------------+
                (targeted retrieval for the uncovered requirement only)
             -> evidence-preserving context build (<=4,000 tokens)
             -> generation (with insufficiency note when coverage is incomplete)

The pipeline never reads gold. It records the full trace: plan, per-round decision,
queries, retrieved ids, coverage before/after each round, and what the context
builder kept or dropped.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src import config
from src.datasets import QAExample
from src.experiment_config import fingerprint, load_config
from src.llm import DeepSeekClient
from src.pipelines.pandarag import controller as controller_module
from src.pipelines.pandarag.context_builder import build_context
from src.pipelines.pandarag.coverage import Coverage, summarise
from src.pipelines.pandarag.evidence_evaluator import evaluate
from src.pipelines.pandarag.evidence_planner import plan_requirements
from src.pipelines.sac_rag import classifier
from src.pipelines.sac_rag.context_compression import compress
from src.pipelines.sac_rag.retrieval_strategies import retrieve_for_type
from src.pipelines.shared.generator import ContextBlock, estimate_tokens, generate_answer
from src.pipelines.shared.llm_json import Usage
from src.pipelines.shared.retriever import RetrievalService, RetrievedDoc
from src.pipelines.shared.runner import RunSpec, run_pipeline

PIPELINE = "pandarag"


@dataclass(frozen=True)
class AblationFlags:
    """Which PandaRAG components are enabled (A0 is SAC-RAG itself, not this file)."""

    planner: bool = True
    evaluator: bool = True
    controller: bool = True
    evidence_context: bool = True
    compression: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "planner": self.planner,
            "evaluator": self.evaluator,
            "controller": self.controller,
            "evidence_context": self.evidence_context,
            "compression": self.compression,
        }


ABLATIONS: dict[str, AblationFlags] = {
    "A1": AblationFlags(planner=True, evaluator=False, controller=False, evidence_context=False),
    "A2": AblationFlags(planner=True, evaluator=True, controller=False, evidence_context=False),
    "A3": AblationFlags(planner=True, evaluator=True, controller=True, evidence_context=False),
    "A4": AblationFlags(),
    "A5": AblationFlags(compression=True),
}
FULL = ABLATIONS["A4"]
INITIAL_ADMISSION = {"Fact": 10, "Definition": 8, "Reasoning": 10}
EVALUATION_POOL = 10


def load_pandarag_config() -> dict[str, Any]:
    return load_config(config.REPO_ROOT / "configs" / "pandarag.json")


def _budget(config_payload: dict[str, Any]) -> controller_module.Budget:
    budget = config_payload["budget"]
    return controller_module.Budget(
        max_additional_rounds=int(budget["max_additional_rounds"]),
        max_cumulative_unique_documents=int(budget["max_cumulative_unique_documents"]),
        max_total_retriever_executions=int(budget["max_total_retriever_executions"]),
        max_generator_context_tokens=int(budget["max_generator_context_tokens"]),
        max_requirements=int(budget["max_requirements"]),
    )


def execute_question(
    example: QAExample,
    service: RetrievalService,
    client: DeepSeekClient,
    *,
    semantic_type: str,
    pandarag_config: dict[str, Any],
    sac_config: dict[str, Any],
    ablation: AblationFlags = FULL,
) -> dict[str, Any]:
    usage = Usage()
    budget = _budget(pandarag_config)
    state = controller_module.ControllerState()
    rerank_latency_before = (
        service._reranker.latency_seconds if service._reranker is not None else 0.0
    )

    plan = plan_requirements(
        client,
        example.question,
        semantic_type,
        usage=usage,
        max_requirements=budget.max_requirements,
    )

    # -- initial retrieval: SAC-RAG strategy driven by the plan's queries -----
    initial = retrieve_for_type(
        service,
        example.question,
        semantic_type,
        config=sac_config,
        decompose_fn=lambda question: plan.search_queries(),
    )
    # Remember what the initial round already did, per query: repeating the same
    # query on the same retriever can only return the same documents.
    for query in initial.queries:
        state.initial_modes_by_query.setdefault(query, [])
    if semantic_type == "Fact":
        for query in initial.queries:
            state.initial_modes_by_query[query] = ["dense", "bm25"]
    elif semantic_type == "Reasoning":
        for query in initial.queries:
            state.initial_modes_by_query[query] = ["dense"]
    else:
        for query in initial.queries:
            state.initial_modes_by_query[query] = ["dense"]

    admission = INITIAL_ADMISSION.get(semantic_type, 10)
    admitted: list[RetrievedDoc] = list(initial.ranked[:admission])
    admitted_ids = {document.doc_id for document in admitted}
    state.unique_documents = len(admitted)
    state.retriever_executions = initial.retrieval_executions

    rounds: list[dict[str, Any]] = []
    verdicts: list = []
    if ablation.evaluator:
        evaluation = evaluate(
            client,
            example.question,
            plan.requirements,
            admitted[:EVALUATION_POOL],
            usage=usage,
        )
        verdicts.extend(evaluation.verdicts)
    else:
        evaluation = None
    coverage = summarise(plan.requirements, verdicts)
    rounds.append(
        {
            "round": 0,
            "kind": "initial",
            "strategy": initial.as_dict(),
            "admitted_doc_ids": [document.doc_id for document in admitted],
            "n_admitted": len(admitted),
            "coverage": coverage.as_dict(),
            "evaluation": evaluation.as_dict() if evaluation else None,
        }
    )
    initial_coverage = coverage

    # -- gap-directed rounds --------------------------------------------------
    while ablation.controller:
        decision = controller_module.decide(
            plan.requirements, coverage, state, budget
        )
        if decision.action == controller_module.STOP:
            rounds.append({"round": state.rounds_used + 1, "kind": "stop", "decision": decision.as_dict()})
            break

        target = decision.target_requirement
        query = decision.query or example.question
        before_supported = coverage.n_supported

        retrieved = service.search(query, decision.mode or "dense", decision.top_k)
        state.retriever_executions += 1
        fresh = [document for document in retrieved if document.doc_id not in admitted_ids]
        if fresh:
            reranked = service.rerank(query, fresh)
            new_documents = reranked[: decision.top_k]
            admitted.extend(new_documents)
            admitted_ids.update(document.doc_id for document in new_documents)
        else:
            new_documents = []

        state.unique_documents = len(admitted_ids)
        state.rounds_used += 1
        state.record_attempt(target or "E1", decision.mode or "dense")

        # Evaluate only the newly admitted documents: pairs already judged in an
        # earlier round keep their verdict, so a recovery round costs one small
        # call instead of re-scoring the whole pool.
        evaluation = evaluate(
            client,
            example.question,
            plan.requirements,
            new_documents[:EVALUATION_POOL],
            usage=usage,
        )
        verdicts.extend(evaluation.verdicts)
        previous = coverage
        coverage = summarise(plan.requirements, verdicts)
        state.last_improved = coverage.n_supported > before_supported
        rounds.append(
            {
                "round": state.rounds_used,
                "kind": "recovery",
                "decision": decision.as_dict(),
                "retrieved_doc_ids": [document.doc_id for document in retrieved],
                "new_doc_ids": [document.doc_id for document in new_documents],
                "n_new": len(new_documents),
                "coverage_before": previous.as_dict(),
                "coverage": coverage.as_dict(),
                "improved": state.last_improved,
                "evaluation": evaluation.as_dict(),
            }
        )

    # -- evidence-preserving context -----------------------------------------
    if ablation.evidence_context:
        selection = build_context(
            plan.requirements,
            coverage,
            admitted,
            max_tokens=budget.max_generator_context_tokens,
            base_top_k=admission,
            min_passages_per_requirement=int(
                pandarag_config["context_builder"].get("min_passages_per_covered_requirement", 1)
            ),
        )
        context_documents = selection.documents
        context_tokens = selection.tokens_estimate
        selection_trace = selection.as_dict()
    else:
        # A1–A3: standard top-K context, same budget, no evidence-based ordering.
        context_documents = [
            {"doc_id": document.doc_id, "title": document.title, "text": document.text}
            for document in admitted[:INITIAL_ADMISSION.get(semantic_type, 10)]
        ]
        context_tokens = estimate_tokens(
            "\n\n".join(f"{item['title']}\n{item['text']}" for item in context_documents)
        )
        selection_trace = {
            "strategy": "standard_top_k",
            "n_documents": len(context_documents),
            "document_ids": [item["doc_id"] for item in context_documents],
            "tokens_estimate": context_tokens,
        }

    if ablation.compression:
        compressed = compress(client, example.question, admitted[: len(context_documents)], usage=usage)
        context_documents = compressed.documents
        context_tokens = compressed.compressed_tokens
        selection_trace["compression"] = compressed.as_dict()

    # No refusal instruction. Measured on the first full runs, an insufficiency note
    # gated on requirement coverage made the system decline 33% (2Wiki) and 50%
    # (MuSiQue) of questions: requirement coverage is a conservative, noisy heuristic
    # and is not a claim that the context is unusable. Coverage now drives retrieval
    # and context selection only, which also removes a prompt-level confound against
    # the baselines. The status stays in the trace for analysis.
    note = False
    block = ContextBlock(documents=context_documents)
    generation = generate_answer(
        client, example.question, block, usage=usage, insufficiency_note=note
    )

    payload: dict[str, Any] = {
        "prediction": generation.prediction,
        "retrieved_documents": [document.as_record() for document in admitted],
        "candidate_document_ids": [document.doc_id for document in admitted],
        "context_documents": [
            {"doc_id": document["doc_id"], "title": document["title"]}
            for document in context_documents
        ],
        "context_text": block.text,
        "context_tokens_estimate": context_tokens,
        "retrieval_rounds": state.rounds_used + 1,
        "retrieval_executions": state.retriever_executions,
        "rerank_latency_seconds": (
            (service._reranker.latency_seconds if service._reranker is not None else 0.0)
            - rerank_latency_before
        ),
        "generation_latency_seconds": generation.latency_seconds,
        "usage": usage,
        "error": generation.error,
        "pandarag": {
            "semantic_type": semantic_type,
            "plan": plan.as_dict(),
            "requirements": [requirement.as_dict() for requirement in plan.requirements],
            "initial_coverage": initial_coverage.as_dict(),
            "final_coverage": coverage.as_dict(),
            "coverage_delta_supported": coverage.n_supported - initial_coverage.n_supported,
            "rounds": rounds,
            "budget": {
                "max_additional_rounds": budget.max_additional_rounds,
                "max_cumulative_unique_documents": budget.max_cumulative_unique_documents,
                "max_total_retriever_executions": budget.max_total_retriever_executions,
                "max_generator_context_tokens": budget.max_generator_context_tokens,
                "used_additional_rounds": state.rounds_used,
                "used_unique_documents": state.unique_documents,
                "used_retriever_executions": state.retriever_executions,
            },
            "context_selection": selection_trace,
            "ablation": ablation.as_dict(),
            "insufficiency_note": note,
            "targeted_retrieval_requested": any(
                item.get("kind") == "recovery" for item in rounds
            ),
        },
    }
    return payload


def build(
    *,
    dataset: str,
    split: str | None = None,
    limit: int | None = None,
    concurrency: int = 4,
    ablation_name: str = "A4",
    sample_ids: set[str] | None = None,
    sample_size: int | None = None,
    results_dir: Path | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    if ablation_name not in ABLATIONS:
        raise SystemExit(f"unknown ablation {ablation_name!r}; choose {sorted(ABLATIONS)}")
    ablation = ABLATIONS[ablation_name]
    if sample_size and sample_ids is None:
        from src.pipelines.shared.sampling import stratified_sample

        sample_ids = set(stratified_sample(dataset, sample_size))
    pandarag_config = load_pandarag_config()
    sac_config = load_config(config.REPO_ROOT / "configs" / "sac_rag.json")
    labels = classifier.load_frozen_labels(dataset)

    def question_fn(example: QAExample, service: RetrievalService, client: DeepSeekClient) -> dict[str, Any]:
        label = labels.get(example.example_id, {"question_type": classifier.FALLBACK_LABEL})
        return execute_question(
            example,
            service,
            client,
            semantic_type=label["question_type"],
            pandarag_config=pandarag_config,
            sac_config=sac_config,
            ablation=ablation,
        )

    spec = RunSpec(
        pipeline=PIPELINE if ablation_name == "A4" else f"{PIPELINE}_{ablation_name}",
        dataset=dataset,
        split=split,
        limit=limit,
        concurrency=concurrency,
        config_fingerprint=fingerprint(pandarag_config),
        extra_meta={
            "pipeline_family": "pandarag",
            "resolved_pandarag_config": pandarag_config,
            "resolved_sac_rag_config": sac_config,
            "budget": pandarag_config["budget"],
            "ablation": ablation_name,
            "ablation_flags": ablation.as_dict(),
            "n_sample_ids": len(sample_ids) if sample_ids is not None else None,
        },
    )
    spec.sample_ids = sample_ids
    base_dir = Path(results_dir) if results_dir else (config.RESULTS_DIR / PIPELINE)
    results_file = base_dir / f"{dataset}.jsonl"
    spec.results_file = results_file
    spec.meta_file = results_file.with_suffix(".meta.json")
    return run_pipeline(
        spec,
        question_fn,
        semantic_types={key: value["question_type"] for key, value in labels.items()},
        quiet=quiet,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PandaRAG (P6)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--ablation", default="A4", choices=sorted(ABLATIONS))
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    build(
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        concurrency=args.concurrency,
        ablation_name=args.ablation,
        sample_size=args.sample_size,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
