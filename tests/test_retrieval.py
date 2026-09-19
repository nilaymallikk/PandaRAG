"""Tests for the dense retriever, the retrieval metrics and the benchmark artifacts.

Run with::

    .venv/bin/python -m unittest discover -s tests -v

The tests never download a model, never touch the network and never call the
DeepSeek API: the embedding model is replaced by a deterministic stub encoder.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from src import config, dataset, evaluation
from src.retrieval import DenseRetriever, RankedHit, RetrievalResult
from src.retrieval_benchmark import (
    build_record,
    load_records,
    load_records_by_id,
    run_benchmark,
    save_records,
)

EXPECTED_DATASET_SHA256 = "956405ceb43d73385d60898e1bfc7f45a8184c9a4e1eff45b9a8ec239acfd048"

QUESTION = "Which city is the birthplace of Ada Lovelace?"

PARAGRAPHS: dict[str, list[str]] = {
    "Ada Lovelace": [
        "Ada Lovelace was born in the city of London.",
        "She is a city birthplace of computing and wrote notes on the Analytical Engine.",
    ],
    "Charles Babbage": [
        "Charles Babbage designed the Analytical Engine in 1837.",
        "He was an English polymath.",
    ],
    "Eiffel Tower": ["The Eiffel Tower is a tower in Paris.", "It was completed in 1889."],
    "Analytical Engine": ["The Analytical Engine was a proposed mechanical computer."],
}
SUPPORTING_FACTS = (("Ada Lovelace", 0), ("Analytical Engine", 0))


class StubEncoder:
    """Deterministic hashed bag-of-words encoder used instead of BGE."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def get_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self,
        texts,
        batch_size=None,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ) -> np.ndarray:
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in text.lower().translate(str.maketrans("", "", "?.,")).split():
                vectors[row, zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (vectors / norms).astype("float32")


def make_example(
    example_id: str = "test-question",
    question: str = QUESTION,
    paragraphs: dict[str, list[str]] | None = None,
    supporting_facts: tuple[tuple[str, int], ...] = SUPPORTING_FACTS,
) -> dataset.HotpotExample:
    """Build a synthetic HotpotExample in the exact dataset encoding."""
    paragraphs = paragraphs or PARAGRAPHS
    titles = tuple(paragraphs)
    return dataset.HotpotExample(
        example_id=example_id,
        question=question,
        answer="London",
        question_type="bridge",
        level="hard",
        context_titles=titles,
        context_sentences=tuple(tuple(paragraphs[title]) for title in titles),
        supporting_facts=supporting_facts,
    )


class DenseRetrieverTests(unittest.TestCase):
    """Ranking behaviour of the dense retriever (stub encoder, no model)."""

    def setUp(self) -> None:
        self.encoder = StubEncoder()
        self.retriever = DenseRetriever(encoder=self.encoder)
        self.example = make_example()

    def test_supporting_paragraph_is_ranked_first(self) -> None:
        result = self.retriever.retrieve(self.example, k=config.RETRIEVAL_MAX_K)
        self.assertEqual(result.paragraphs[0].title, "Ada Lovelace")
        self.assertEqual(result.example_id, "test-question")
        scores = [hit.score for hit in result.paragraphs]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(result.k, len(PARAGRAPHS))

    def test_hits_are_truncated_to_k_with_document_and_sentence_metadata(self) -> None:
        result = self.retriever.retrieve(self.example, k=2)
        self.assertEqual([hit.rank for hit in result.top_paragraphs(2)], [1, 2])
        self.assertTrue(all(hit.sent_id is None for hit in result.top_paragraphs(2)))
        self.assertGreaterEqual(len(result.sentences), 2)
        sentence_hits = result.top_sentences(2)
        self.assertTrue(all(hit.is_sentence and isinstance(hit.sent_id, int) for hit in sentence_hits))
        self.assertIn("::", sentence_hits[0].identifier())

    def test_smaller_k_is_a_prefix_of_the_deeper_ranking(self) -> None:
        deep = self.retriever.retrieve(self.example, k=config.RETRIEVAL_MAX_K)
        shallow = self.retriever.retrieve(self.example, k=3)
        self.assertEqual(deep.paragraph_titles(3), shallow.paragraph_titles(3))
        self.assertEqual(deep.sentence_ids(3), shallow.sentence_ids(3))

    def test_query_instruction_is_applied_to_the_query_only(self) -> None:
        self.retriever.retrieve(self.example, k=3)
        paragraph_call, sentence_call, query_call = self.encoder.calls
        self.assertEqual(query_call, [f"{config.EMBEDDING_QUERY_INSTRUCTION}{QUESTION}"])
        self.assertEqual(
            paragraph_call,
            [self.example.paragraph_text(title) for title in self.example.context_titles],
        )
        self.assertEqual(len(sentence_call), sum(len(p) for p in PARAGRAPHS.values()))
        candidates = paragraph_call + sentence_call
        self.assertTrue(
            all(not text.startswith(config.EMBEDDING_QUERY_INSTRUCTION) for text in candidates)
        )

    def test_gold_supporting_sentence_ranks_highly(self) -> None:
        result = self.retriever.retrieve(self.example, k=5)
        gold_ids = set(self.example.supporting_fact_ids)
        self.assertIn("Ada Lovelace::0", gold_ids)
        self.assertIn("Ada Lovelace::0", result.sentence_ids(5))

    def test_empty_candidate_pool_yields_no_hits(self) -> None:
        empty = np.zeros((0, 4), dtype="float32")
        query = np.zeros((1, 4), dtype="float32")
        self.assertEqual(DenseRetriever._search(empty, query, 5, []), [])
        self.assertEqual(DenseRetriever._search(empty, query, 5, [("Title", None)]), [])

    def test_settings_document_the_retrieval_configuration(self) -> None:
        settings = self.retriever.settings()
        self.assertEqual(settings["retrieval_method"], "dense")
        self.assertEqual(settings["embedding_model"], config.EMBEDDING_MODEL_NAME)
        self.assertEqual(settings["index"], "faiss.IndexFlatIP")
        self.assertEqual(settings["max_k"], config.RETRIEVAL_MAX_K)


class RankingHitTests(unittest.TestCase):
    def test_identifier_uses_dataset_sentence_convention(self) -> None:
        paragraph_hit = RankedHit(rank=1, title="Ada Lovelace", score=0.5)
        sentence_hit = RankedHit(rank=1, title="Ada Lovelace", score=0.5, sent_id=3)
        self.assertEqual(paragraph_hit.identifier(), "Ada Lovelace")
        self.assertEqual(sentence_hit.identifier(), dataset.sentence_id("Ada Lovelace", 3))
        self.assertEqual(sentence_hit.to_dict()["sent_id"], 3)
        self.assertNotIn("sent_id", paragraph_hit.to_dict())


class RetrievalMetricTests(unittest.TestCase):
    """Recall@K logic for both granularities."""

    def setUp(self) -> None:
        self.example = make_example()

    def test_document_recall_at_k_uses_supporting_titles(self) -> None:
        supporting = self.example.supporting_titles
        retrieved = ["Analytical Engine", "Charles Babbage", "Ada Lovelace"]
        self.assertEqual(evaluation.recall_at_k(retrieved, supporting, 1), 0.5)
        self.assertEqual(evaluation.recall_at_k(retrieved, supporting, 2), 0.5)
        self.assertEqual(evaluation.recall_at_k(retrieved, supporting, 3), 1.0)

    def test_sentence_recall_at_k_uses_supporting_fact_ids(self) -> None:
        supporting = self.example.supporting_fact_ids
        self.assertEqual(supporting, ("Ada Lovelace::0", "Analytical Engine::0"))
        retrieved = ["Analytical Engine::0", "Eiffel Tower::1", "Ada Lovelace::0"]
        self.assertEqual(evaluation.recall_at_k(retrieved, supporting, 1), 0.5)
        self.assertEqual(evaluation.recall_at_k(retrieved, supporting, 2), 0.5)
        self.assertEqual(evaluation.recall_at_k(retrieved, supporting, 3), 1.0)

    def test_recall_at_k_edge_cases(self) -> None:
        supporting = ["a", "b"]
        self.assertEqual(evaluation.recall_at_k(["a"], supporting, 0), 0.0)
        self.assertEqual(evaluation.recall_at_k([], supporting, 5), 0.0)
        self.assertEqual(evaluation.recall_at_k(["a", "b", "c"], supporting, 99), 1.0)
        self.assertTrue(math.isnan(evaluation.recall_at_k(["a"], [], 5)))
        self.assertEqual(evaluation.recall_at_k(["a", "a", "a"], supporting, 3), 0.5)

    def test_complete_at_k_is_strict(self) -> None:
        self.assertTrue(evaluation.complete_at_k(["a", "b"], ["a", "b"], 2))
        self.assertFalse(evaluation.complete_at_k(["a", "b"], ["a", "b"], 1))
        self.assertTrue(evaluation.complete_at_k([], [], 5))

    def test_evaluate_retrieval_reports_both_granularities(self) -> None:
        result = RetrievalResult(
            example_id=self.example.example_id,
            question=self.example.question,
            k=config.RETRIEVAL_MAX_K,
            paragraphs=[
                RankedHit(rank=1, title="Ada Lovelace", score=0.9),
                RankedHit(rank=2, title="Eiffel Tower", score=0.2),
            ],
            sentences=[
                RankedHit(rank=1, title="Ada Lovelace", score=0.8, sent_id=0),
                RankedHit(rank=2, title="Eiffel Tower", score=0.1, sent_id=1),
            ],
        )
        metrics = evaluation.evaluate_retrieval(result, self.example, (1, 5))
        self.assertEqual(sorted(metrics), ["1", "5"])
        self.assertEqual(metrics["1"]["k"], 1)
        self.assertEqual(metrics["1"]["document_recall"], 0.5)
        self.assertEqual(metrics["1"]["document_complete"], 0.0)
        self.assertEqual(metrics["1"]["sentence_recall"], 0.5)
        self.assertEqual(metrics["1"]["sentence_complete"], 0.0)
        self.assertEqual(metrics["5"]["document_recall"], 0.5)

    def test_aggregate_retrieval_metrics_averages_and_skips_nan(self) -> None:
        records = [
            {
                "1": {
                    "k": 1,
                    "document_recall": 1.0,
                    "document_complete": 1.0,
                    "sentence_recall": 0.5,
                    "sentence_complete": 0.0,
                }
            },
            {
                "1": {
                    "k": 1,
                    "document_recall": 0.0,
                    "document_complete": 0.0,
                    "sentence_recall": math.nan,
                    "sentence_complete": 0.0,
                }
            },
        ]
        aggregate = evaluation.aggregate_retrieval_metrics(records, (1,))
        self.assertEqual(aggregate["1"]["n_questions"], 2)
        self.assertEqual(aggregate["1"]["document_recall"], 0.5)
        self.assertEqual(aggregate["1"]["document_complete_rate"], 0.5)
        self.assertEqual(aggregate["1"]["sentence_recall"], 0.5)

    def test_aggregate_latency(self) -> None:
        stats = evaluation.aggregate_latency([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["mean"], 2.5)
        self.assertEqual(stats["median"], 2.5)
        self.assertEqual(stats["total"], 10.0)
        self.assertEqual(stats["p95"], 4.0)
        empty = evaluation.aggregate_latency([])
        self.assertTrue(math.isnan(empty["mean"]))
        self.assertEqual(empty["total"], 0.0)

    def test_format_retrieval_summary_lists_every_k(self) -> None:
        aggregate = evaluation.aggregate_retrieval_metrics(
            [
                {
                    str(k): {
                        "k": k,
                        "document_recall": 0.5,
                        "document_complete": 0.25,
                        "sentence_recall": 0.4,
                        "sentence_complete": 0.1,
                    }
                    for k in config.RETRIEVAL_K_VALUES
                }
            ]
        )
        summary = evaluation.format_retrieval_summary(aggregate)
        self.assertIn("Doc Recall@K", summary)
        for k in config.RETRIEVAL_K_VALUES:
            self.assertIn(f"{k:>3}", summary)


class BenchmarkArtifactTests(unittest.TestCase):
    """Raw retrieval artifacts must be reusable by the later RAG pipelines."""

    def setUp(self) -> None:
        self.retriever = DenseRetriever(encoder=StubEncoder())
        self.example = make_example()
        self.result = self.retriever.retrieve(self.example, k=config.RETRIEVAL_MAX_K)
        self.metrics = evaluation.evaluate_retrieval(self.result, self.example)
        self.record = build_record(
            self.example, self.result, self.metrics, config.RETRIEVAL_MAX_K
        )

    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "records.jsonl"
            save_records([self.record], path)
            loaded = load_records(path)
            by_id = load_records_by_id(path)
        self.assertEqual(len(loaded), 1)
        record = loaded[0]
        self.assertEqual(record["question_id"], self.example.example_id)
        self.assertEqual(record["supporting_facts"], ["Ada Lovelace::0", "Analytical Engine::0"])
        self.assertEqual(record["supporting_titles"], ["Ada Lovelace", "Analytical Engine"])
        self.assertEqual(record["retrieved_documents"][0]["title"], "Ada Lovelace")
        self.assertGreater(record["retrieval_latency_s"], 0.0)
        self.assertEqual(by_id[self.example.example_id]["metrics"]["10"]["k"], 10)

    def test_metrics_are_recomputable_from_the_saved_ranking(self) -> None:
        titles = [hit["title"] for hit in self.record["retrieved_documents"]]
        fact_ids = [
            dataset.sentence_id(hit["title"], hit["sent_id"])
            for hit in self.record["retrieved_sentences"]
        ]
        for k in config.RETRIEVAL_K_VALUES:
            stored = self.record["metrics"][str(k)]
            self.assertEqual(
                stored["document_recall"],
                evaluation.recall_at_k(titles, self.example.supporting_titles, k),
            )
            self.assertEqual(
                stored["sentence_recall"],
                evaluation.recall_at_k(fact_ids, self.example.supporting_fact_ids, k),
            )

    def test_run_benchmark_smoke_on_the_real_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_path = Path(tmp) / "dense_retrieval.jsonl"
            meta_path = Path(tmp) / "dense_retrieval.meta.json"
            meta = run_benchmark(
                limit=2,
                retriever=DenseRetriever(encoder=StubEncoder()),
                results_file=results_path,
                meta_file=meta_path,
                quiet=True,
            )
            records = load_records(results_path)
            meta_on_disk = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["n_questions"], 2)
        self.assertEqual(meta["llm_calls"], 0)
        self.assertFalse(meta["llm_used"])
        self.assertEqual(len(records), 2)
        self.assertEqual(meta_on_disk["dataset"]["sha256"], EXPECTED_DATASET_SHA256)
        for record in records:
            self.assertGreaterEqual(len(record["supporting_titles"]), 2)
            self.assertTrue(all("::" in fact for fact in record["supporting_facts"]))
            self.assertLessEqual(len(record["retrieved_documents"]), config.RETRIEVAL_MAX_K)
            self.assertGreater(record["candidate_paragraphs"], 0)

    def test_dataset_file_is_untouched(self) -> None:
        fingerprint = dataset.dataset_fingerprint()
        self.assertEqual(fingerprint["sha256"], EXPECTED_DATASET_SHA256)
        self.assertEqual(len(dataset.load_dataset()), config.EXPECTED_DATASET_SIZE)


if __name__ == "__main__":
    unittest.main()
