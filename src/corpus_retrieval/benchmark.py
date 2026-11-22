"""Corpus-level retrieval benchmark (Phase 1 sanity check).

For every question it retrieves ``max(k_values)`` documents from the pooled corpus
with each method and reports, at every K:

* document **Recall@K** — ``|retrieved ∩ gold| / |gold|`` (the base paper's
  definition, reused from :mod:`src.evaluation`);
* **all-gold coverage@K** — the fraction of questions whose *entire* gold document
  set is inside the top K. Recall can look healthy while one bridge paragraph is
  always missing, which is exactly the failure mode this project cares about, so
  both numbers are always reported together.

Diagnostics are deliberately verbose: if a corpus-level Recall@10 came back at
1.000 the run is treated as suspicious, and the per-question rank distribution of
gold documents is what distinguishes "genuinely easy corpus" from "corpus is
wrong / evidence mapping is wrong / index is wrong / ranking is wrong".

No LLM calls, no API cost. Gold is used here for scoring only.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from src import config
from src.corpus_retrieval import corpus as corpus_module
from src.corpus_retrieval import fusion as fusion_module
from src.corpus_retrieval.dense import Encoder
from src.corpus_retrieval.index_manager import load_indexes
from src.corpus_retrieval.schemas import RankedList
from src.datasets import get_adapter
from src.evaluation import recall_at_k

RESULTS_DIR = config.RESULTS_DIR / "sac_rag" / "retrieval"
BENCHMARK_VERSION = "corpus-benchmark/1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def benchmark_paths(dataset: str, split: str) -> dict[str, Path]:
    stem = f"corpus_retrieval_{dataset}_{split}"
    return {
        "results": RESULTS_DIR / f"{stem}.jsonl",
        "meta": RESULTS_DIR / f"{stem}.meta.json",
    }


def _rankings(
    *,
    question: str,
    dense,
    bm25,
    encoder: Encoder,
    top_k: int,
    depth: int,
    constant: float,
) -> dict[str, RankedList]:
    dense_ids, dense_scores = dense.search(encoder, question, top_k)
    bm25_ids, bm25_scores = bm25.search(question, top_k)
    dense_list = RankedList(query=question, method="dense", doc_ids=dense_ids, scores=dense_scores)
    bm25_list = RankedList(query=question, method="bm25", doc_ids=bm25_ids, scores=bm25_scores)
    fused = fusion_module.fuse(
        {"dense": dense_list, "bm25": bm25_list}, depth=depth, constant=constant, k=top_k
    )
    hybrid_list = RankedList(
        query=question,
        method="hybrid",
        doc_ids=tuple(hit.doc_id for hit in fused),
        scores=tuple(hit.rrf_score for hit in fused),
    )
    return {"dense": dense_list, "bm25": bm25_list, "hybrid": hybrid_list}


def run_benchmark(
    dataset: str,
    split: str | None = None,
    *,
    limit: int | None = None,
    k_values: Sequence[int] = config.RETRIEVAL_K_VALUES,
    encoder: Encoder | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Run the corpus retrieval benchmark and return the aggregate report."""
    adapter = get_adapter(dataset, split)
    split = adapter.split
    examples = sorted(adapter.load(), key=lambda example: example.example_id)
    if limit:
        examples = examples[: int(limit)]

    artifact = corpus_module.load_corpus(dataset, split)
    dense, bm25, manifest = load_indexes(dataset, split)

    if encoder is None:
        from src.corpus_retrieval.dense import SentenceTransformerEncoder

        encoder = SentenceTransformerEncoder()

    top_k = max(k_values)
    started = _utc_now()
    timer = time.perf_counter()

    per_question: list[dict[str, Any]] = []
    for example in examples:
        rankings = _rankings(
            question=example.question,
            dense=dense,
            bm25=bm25,
            encoder=encoder,
            top_k=top_k,
            depth=config.RRF_DEPTH,
            constant=config.RRF_CONSTANT,
        )
        gold = set(example.gold_doc_ids)
        row: dict[str, Any] = {
            "schema_version": 2,
            "dataset": dataset,
            "split": split,
            "pipeline": "corpus_retrieval",
            "question_id": example.example_id,
            "question": example.question,
            "gold_document_ids": sorted(gold),
            "n_gold_documents": len(gold),
            "evidence_kind": artifact.stats.get("granularity"),
            "corpus_sha256": artifact.corpus_sha256,
        }
        for method, ranking in rankings.items():
            retrieved = list(ranking.doc_ids)
            hits = [doc_id for doc_id in retrieved if doc_id in gold]
            ranks = [retrieved.index(doc_id) + 1 for doc_id in gold if doc_id in retrieved]
            row[method] = {
                "top_k": retrieved,
                "scores": list(ranking.scores),
                "recall": {f"recall@{k}": recall_at_k(retrieved, gold, k) for k in k_values},
                "complete": {f"complete@{k}": float(gold.issubset(set(retrieved[:k]))) for k in k_values},
                "n_gold_retrieved": len(hits),
                "gold_ranks": sorted(ranks),
                "max_gold_rank": max(ranks) if ranks else None,
            }
        per_question.append(row)

    elapsed = round(time.perf_counter() - timer, 2)
    aggregate = _aggregate(per_question, k_values, artifact, manifest, elapsed)

    payload = {
        "benchmark": BENCHMARK_VERSION,
        "dataset": dataset,
        "split": split,
        "started_utc": started,
        "runtime_s": elapsed,
        "corpus": {
            "n_documents": len(artifact.documents),
            "corpus_sha256": artifact.corpus_sha256,
            "dataset_sha256": artifact.dataset_sha256,
            "protocol": artifact.stats.get("protocol"),
        },
        "index": manifest.as_dict()["dense"] | {"bm25": manifest.as_dict()["bm25"]},
        "encoder": getattr(encoder, "name", type(encoder).__name__),
        "k_values": list(k_values),
        "n_questions": len(per_question),
        "aggregate": aggregate,
    }

    if write:
        paths = benchmark_paths(dataset, split)
        paths["results"].parent.mkdir(parents=True, exist_ok=True)
        with paths["results"].open("w", encoding="utf-8") as handle:
            for row in per_question:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        paths["meta"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        payload["paths"] = {name: str(path) for name, path in paths.items()}

    payload["per_question"] = per_question
    return payload


def _aggregate(
    per_question: Sequence[dict[str, Any]],
    k_values: Iterable[int],
    artifact,
    manifest,
    elapsed: float,
) -> dict[str, Any]:
    methods = ("dense", "bm25", "hybrid")
    summary: dict[str, Any] = {
        "n_questions": len(per_question),
        "mean_gold_documents": round(
            mean(row["n_gold_documents"] for row in per_question), 4
        )
        if per_question
        else 0.0,
        "corpus_documents": len(artifact.documents),
        "runtime_s": elapsed,
    }
    for method in methods:
        block: dict[str, Any] = {}
        for k in k_values:
            block[f"recall@{k}"] = round(
                mean(row[method]["recall"][f"recall@{k}"] for row in per_question), 4
            ) if per_question else 0.0
            block[f"all_gold_coverage@{k}"] = round(
                mean(row[method]["complete"][f"complete@{k}"] for row in per_question), 4
            ) if per_question else 0.0
        block["questions_with_no_gold_retrieved"] = sum(
            1 for row in per_question if row[method]["n_gold_retrieved"] == 0
        )
        block["mean_gold_docs_retrieved"] = round(
            mean(row[method]["n_gold_retrieved"] for row in per_question), 4
        ) if per_question else 0.0
        ranks = [
            rank
            for row in per_question
            for rank in row[method]["gold_ranks"]
        ]
        if ranks:
            ranks.sort()
            block["gold_rank_median"] = ranks[len(ranks) // 2]
            block["gold_rank_p90"] = ranks[min(len(ranks) - 1, int(len(ranks) * 0.9))]
            block["gold_rank_max"] = ranks[-1]
        block["suspicious_recall_at_10"] = bool(
            per_question and block[f"recall@{max(k_values)}"] >= 0.999 and len(artifact.documents) > max(k_values)
        )
        summary[method] = block
    summary["index_n_documents"] = manifest.n_documents
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Corpus-level retrieval sanity benchmark")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    payload = run_benchmark(args.dataset, args.split, limit=args.limit)
    aggregate = payload["aggregate"]
    print(f"{args.dataset}/{payload['split']}: {payload['corpus']['n_documents']} documents, "
          f"{payload['n_questions']} questions, {payload['runtime_s']}s")
    for method in ("dense", "bm25", "hybrid"):
        block = aggregate[method]
        print(f"  {method:6s} " + "  ".join(
            f"R@{k}={block[f'recall@{k}']:.3f}" for k in payload["k_values"]
        ) + f"  all-gold@10={block['all_gold_coverage@10']:.3f}"
          + f"  median_gold_rank={block.get('gold_rank_median')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
