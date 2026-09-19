"""Offline tests for the hybrid (RRF) retriever and its benchmark artifacts.

Run with::

    .venv/bin/python -m unittest discover -s tests -v

No model is downloaded, no network is used and no LLM/API call is made: the
dense encoder is replaced by the deterministic stub encoder from
``test_retrieval`` and BM25 is pure rank_bm25.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_retrieval import (  # noqa: E402  (shared offline helpers)
    EXPECTED_DATASET_SHA256,
    StubEncoder,
    make_example,
)

from src import config, dataset, evaluation  # noqa: E402
from src.retrieval import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RankedHit,
    fuse_rankings,
    rrf_score,
)
from src.retrieval_benchmark import (  # noqa: E402
    METHOD_PATHS,
    RETRIEVER_CLASSES,
    build_record,
    load_records,
    run_benchmark,
    save_records,
)


def dense_ranking(ranks_titles: list[tuple[int, str]], scores: float = 0.9) -> list[RankedHit]:
    """Build a plain dense ranking for fusion tests."""
    return [
        RankedHit(rank=rank, title=title, score=scores - 0.01 * rank)
        for rank, title in ranks_titles
    ]


def bm25_ranking(ranks_titles: list[tuple[int, str]], scores: float = 10.0) -> list[RankedHit]:
    """Build a plain BM25 ranking for fusion tests."""
    return [
        RankedHit(rank=rank, title=title, score=scores - 0.1 * rank)
        for rank, title in ranks_titles
    ]


class RRFScoreTests(unittest.TestCase):
    """The frozen RRF formula: score(d) = sum(1 / (60 + rank))."""

    def test_single_rank_uses_the_documented_formula(self) -> None:
        self.assertAlmostEqual(rrf_score([1]), 1 / 61)
        self.assertAlmostEqual(rrf_score([3]), 1 / 63)
        self.assertAlmostEqual(rrf_score([10]), 1 / 70)

    def test_both_sources_contribute(self) -> None:
        self.assertAlmostEqual(rrf_score([1, 4]), 1 / 61 + 1 / 64)

    def test_constant_is_configurable(self) -> None:
        self.assertAlmostEqual(rrf_score([1], constant=0), 1.0)
        self.assertAlmostEqual(rrf_score([2, 3], constant=1), 1 / 3 + 1 / 4)

    def test_invalid_ranks_are_ignored(self) -> None:
        self.assertEqual(rrf_score([]), 0.0)
        self.assertEqual(rrf_score([None, 0, -2, 5]), 1 / 65)  # only rank 5 counts

    def test_frozen_constant_matches_config(self) -> None:
        self.assertEqual(config.RRF_CONSTANT, 60.0)
        self.assertEqual(config.RRF_DEPTH, config.RETRIEVAL_MAX_K)


class FuseRankingsTests(unittest.TestCase):
    """Union, provenance, tie-breaking and truncation of the rank fusion."""

    def test_document_in_both_rankings_sums_contributions(self) -> None:
        dense = dense_ranking([(1, "A"), (2, "B"), (3, "C")])
        bm25 = bm25_ranking([(2, "C"), (1, "A"), (3, "B")])
        fused = fuse_rankings(dense_hits=dense, bm25_hits=bm25, k=3)
        # A: 1/61+1/61; B and C tie on RRF score (1/62+1/63) with the same best
        # rank 2, so the title breaks the tie ("B" before "C").
        self.assertEqual([hit.title for hit in fused], ["A", "B", "C"])
        first = fused[0]
        self.assertAlmostEqual(first.score, 1 / 61 + 1 / 61)
        self.assertEqual(first.dense_rank, 1)
        self.assertEqual(first.bm25_rank, 1)
        self.assertAlmostEqual(first.dense_score, dense[0].score)
        self.assertAlmostEqual(first.bm25_score, bm25[1].score)
        self.assertTrue(first.has_provenance)

    def test_document_in_only_one_ranking_keeps_provenance(self) -> None:
        dense = dense_ranking([(1, "A"), (2, "B")])
        bm25 = bm25_ranking([(1, "C")])
        fused = fuse_rankings(dense_hits=dense, bm25_hits=bm25, k=3)
        self.assertEqual(set(hit.title for hit in fused), {"A", "B", "C"})
        by_title = {hit.title: hit for hit in fused}
        c = by_title["C"]
        self.assertAlmostEqual(c.score, 1 / 61)  # BM25 contribution only
        self.assertIsNone(c.dense_rank)
        self.assertIsNone(c.dense_score)
        self.assertEqual(c.bm25_rank, 1)
        b = by_title["B"]
        self.assertAlmostEqual(b.score, 1 / 62)  # dense contribution only
        self.assertEqual(b.dense_rank, 2)
        self.assertIsNone(b.bm25_rank)

    def test_ties_are_broken_deterministically(self) -> None:
        # A full score tie (each title is rank 1 with exactly one source) is
        # broken by the title, so input order never matters.
        first = fuse_rankings(
            dense_hits=dense_ranking([(1, "B")]),
            bm25_hits=bm25_ranking([(1, "A")]),
            k=2,
        )
        second = fuse_rankings(
            dense_hits=dense_ranking([(1, "A")]),
            bm25_hits=bm25_ranking([(1, "B")]),
            k=2,
        )
        self.assertEqual([hit.title for hit in first], ["A", "B"])
        self.assertEqual([hit.title for hit in second], ["A", "B"])  # tie -> title order

    def test_k_truncates_and_renumbers_ranks(self) -> None:
        dense = dense_ranking([(1, "A"), (2, "B"), (3, "C")])
        bm25 = bm25_ranking([(1, "C"), (2, "B"), (3, "A")])
        fused = fuse_rankings(dense_hits=dense, bm25_hits=bm25, k=2)
        self.assertEqual(len(fused), 2)
        self.assertEqual([hit.rank for hit in fused], [1, 2])

    def test_depth_limits_each_source(self) -> None:
        dense = dense_ranking([(r, f"D{r}") for r in range(1, 6)])
        bm25 = bm25_ranking([(r, f"L{r}") for r in range(1, 6)])
        fused = fuse_rankings(dense_hits=dense, bm25_hits=bm25, k=10, depth=3)
        # With depth 3 the candidate set is {D1,D2,D3,L1,L2,L3}; D4/D5/L4/L5 excluded.
        self.assertEqual(len(fused), 6)
        self.assertFalse(any(hit.title in {"D4", "D5", "L4", "L5"} for hit in fused))

    def test_empty_rankings_yield_no_hits(self) -> None:
        self.assertEqual(fuse_rankings(dense_hits=[], bm25_hits=[], k=5), [])
        only_dense = fuse_rankings(dense_hits=dense_ranking([(1, "A")]), bm25_hits=[], k=5)
        self.assertEqual([hit.title for hit in only_dense], ["A"])
        self.assertIsNone(only_dense[0].bm25_rank)


class HybridRetrieverTests(unittest.TestCase):
    """End-to-end fusion behaviour with a stub encoder (fully offline)."""

    def setUp(self) -> None:
        self.encoder = StubEncoder()
        self.retriever = HybridRetriever(
            dense=DenseRetriever(encoder=self.encoder), bm25=BM25Retriever()
        )
        self.example = make_example()

    def test_supporting_paragraph_ranks_first_and_provenance_is_kept(self) -> None:
        result = self.retriever.retrieve(self.example, k=config.RETRIEVAL_MAX_K)
        self.assertEqual(result.paragraphs[0].title, "Ada Lovelace")
        self.assertEqual(result.example_id, "test-question")
        for hit in result.paragraphs:
            self.assertTrue(hit.has_provenance)
            self.assertIn("rrf_score", hit.to_dict())
            self.assertTrue(hit.dense_rank is not None or hit.bm25_rank is not None)

    def test_scores_are_descending_and_k_truncates(self) -> None:
        result = self.retriever.retrieve(self.example, k=2)
        scores = [hit.score for hit in result.paragraphs]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual([hit.rank for hit in result.paragraphs], [1, 2])
        self.assertEqual(len(result.paragraphs), 2)

    def test_smaller_k_is_a_prefix_of_the_deeper_ranking(self) -> None:
        deep = self.retriever.retrieve(self.example, k=config.RETRIEVAL_MAX_K)
        shallow = self.retriever.retrieve(self.example, k=3)
        self.assertEqual(deep.paragraph_titles(3), shallow.paragraph_titles(3))
        self.assertEqual(deep.sentence_ids(3), shallow.sentence_ids(3))

    def test_result_is_compatible_with_the_shared_evaluation(self) -> None:
        result = self.retriever.retrieve(self.example, k=config.RETRIEVAL_MAX_K)
        metrics = evaluation.evaluate_retrieval(result, self.example, config.RETRIEVAL_K_VALUES)
        self.assertEqual(sorted(metrics, key=int), ["1", "3", "5", "10"])
        # Both supporting titles are strong for this synthetic question.
        self.assertEqual(metrics["5"]["document_recall"], 1.0)
        self.assertIn("Ada Lovelace::0", result.sentence_ids(config.RETRIEVAL_MAX_K))

    def test_settings_document_both_sources_and_the_fusion(self) -> None:
        settings = self.retriever.settings()
        self.assertEqual(settings["retrieval_method"], "hybrid")
        self.assertEqual(settings["fusion"]["rrf_constant"], 60.0)
        self.assertFalse(settings["fusion"]["raw_scores_used_in_fusion"])
        self.assertEqual(settings["dense"]["retrieval_method"], "dense")
        self.assertEqual(settings["bm25"]["retrieval_method"], "bm25")
        self.assertEqual(settings["max_k"], config.RETRIEVAL_MAX_K)

    def test_load_delegates_to_the_dense_encoder(self) -> None:
        self.assertEqual(self.retriever.load(), 0.0)  # stub encoder: nothing to load

    def test_retriever_is_registered_in_the_benchmark(self) -> None:
        self.assertIn("hybrid", RETRIEVER_CLASSES)
        self.assertIn("hybrid", METHOD_PATHS)
        results_path, meta_path = METHOD_PATHS["hybrid"]
        self.assertEqual(results_path.name, "hybrid_retrieval.jsonl")
        self.assertEqual(meta_path.name, "hybrid_retrieval.meta.json")


class HybridArtifactTests(unittest.TestCase):
    """Record serialisation, artifact round-trip and dataset integrity."""

    def test_build_record_serialises_provenance(self) -> None:
        example = make_example()
        retriever = HybridRetriever(
            dense=DenseRetriever(encoder=StubEncoder()), bm25=BM25Retriever()
        )
        result = retriever.retrieve(example, k=config.RETRIEVAL_MAX_K)
        metrics = evaluation.evaluate_retrieval(result, example, config.RETRIEVAL_K_VALUES)
        record = build_record(example, result, metrics, config.RETRIEVAL_MAX_K, "hybrid")
        self.assertEqual(record["retrieval_method"], "hybrid")
        top = record["retrieved_documents"][0]
        for key in ("rank", "title", "score", "rrf_score", "dense_rank", "dense_score"):
            self.assertIn(key, top)
        # Pure dense/BM25 records must not gain provenance keys.
        dense_record = build_record(
            example,
            DenseRetriever(encoder=StubEncoder()).retrieve(example, k=3),
            metrics,
            3,
            "dense",
        )
        self.assertNotIn("rrf_score", dense_record["retrieved_documents"][0])

    def test_artifact_round_trip_preserves_hybrid_records(self) -> None:
        example = make_example()
        retriever = HybridRetriever(
            dense=DenseRetriever(encoder=StubEncoder()), bm25=BM25Retriever()
        )
        result = retriever.retrieve(example, k=config.RETRIEVAL_MAX_K)
        metrics = evaluation.evaluate_retrieval(result, example, config.RETRIEVAL_K_VALUES)
        record = build_record(example, result, metrics, config.RETRIEVAL_MAX_K, "hybrid")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hybrid_retrieval.jsonl"
            save_records([record], path)
            loaded = load_records(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0], record)  # round-trip is lossless

    def test_hybrid_benchmark_smoke_run_makes_no_llm_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = run_benchmark(
                method="hybrid",
                limit=2,
                results_file=Path(tmp) / "hybrid.jsonl",
                meta_file=Path(tmp) / "hybrid.meta.json",
                quiet=True,
            )
            self.assertEqual(meta["llm_calls"], 0)
            self.assertFalse(meta["llm_used"])
            self.assertEqual(meta["n_questions"], 2)
            self.assertEqual(meta["retriever"]["retrieval_method"], "hybrid")
            self.assertEqual(meta["dataset"]["sha256"], EXPECTED_DATASET_SHA256)
            records = load_records(Path(tmp) / "hybrid.jsonl")
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r["retrieval_method"] == "hybrid" for r in records))
            for line in (Path(tmp) / "hybrid.jsonl").read_text().splitlines():
                self.assertNotIn("NaN", line)
                self.assertNotIn("Infinity", line)
                json.loads(line)

    def test_dataset_fingerprint_is_unchanged(self) -> None:
        self.assertEqual(dataset.dataset_fingerprint()["sha256"], EXPECTED_DATASET_SHA256)
        self.assertEqual(len(dataset.load_dataset()), config.EXPECTED_DATASET_SIZE)


if __name__ == "__main__":
    unittest.main()