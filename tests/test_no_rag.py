"""Tests for No-RAG answer evaluation and the No-RAG baseline pipeline.

Run with::

    .venv/bin/python -m unittest discover -s tests -v

Everything here is offline: the DeepSeek API is replaced by a stub client
returning :class:`src.llm.LLMResponse` objects, and the dataset file is only
read for its fingerprint (never modified).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src import answer_eval, config, dataset
from src.llm import LLMResponse
from src.pipelines import no_rag

EXPECTED_DATASET_SHA256 = "956405ceb43d73385d60898e1bfc7f45a8184c9a4e1eff45b9a8ec239acfd048"


def _example(
    example_id: str = "q1",
    question: str = "What is the capital of France?",
    answer: str = "Paris",
) -> dataset.HotpotExample:
    return dataset.HotpotExample(
        example_id=example_id,
        question=question,
        answer=answer,
        question_type="bridge",
        level="hard",
        context_titles=("France",),
        context_sentences=(("Paris is the capital.",),),
        supporting_facts=(("France", 0),),
    )


class StubClient:
    """Offline stand-in for DeepSeekClient (records prompts, no network)."""

    def __init__(self, texts: list[str] | None = None) -> None:
        self.texts = list(texts or ["Paris"])
        self.prompts: list[str] = []
        self.systems: list[str | None] = []
        self.model = config.DEEPSEEK_MODEL
        self.thinking = "disabled"
        self.calls = 0

    def generate(self, prompt: str, *, system: str | None = None, **_: object) -> LLMResponse:
        self.prompts.append(prompt)
        self.systems.append(system)
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=10,
            completion_tokens=2,
            cached_input_tokens=0,
            total_tokens=12,
            latency_s=0.5,
            llm_calls=1,
            cost_usd=0.000001,
            finish_reason="stop",
            from_cache=False,
            created_at="2026-01-01T00:00:00+00:00",
        )

    def settings(self) -> dict[str, object]:
        return {
            "provider": "DeepSeek",
            "model": self.model,
            "thinking": self.thinking,
            "temperature": 0.0,
            "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        }


class NormalizeTests(unittest.TestCase):
    def test_lowercase_and_punctuation(self) -> None:
        self.assertEqual(answer_eval.normalize_answer("  The Eiffel Tower! "), "eiffel tower")

    def test_articles_removed(self) -> None:
        self.assertEqual(answer_eval.normalize_answer("a cat"), "cat")
        self.assertEqual(answer_eval.normalize_answer("An Apple"), "apple")
        self.assertEqual(answer_eval.normalize_answer("the dog"), "dog")

    def test_none_and_empty(self) -> None:
        self.assertEqual(answer_eval.normalize_answer(None), "")
        self.assertEqual(answer_eval.normalize_answer("   "), "")
        self.assertEqual(answer_eval.normalize_answer(""), "")

    def test_whitespace_collapsed(self) -> None:
        self.assertEqual(answer_eval.normalize_answer("New   York\tCity\n"), "new york city")


class ExactMatchTests(unittest.TestCase):
    def test_match_ignores_case_punct_articles(self) -> None:
        self.assertEqual(answer_eval.exact_match("The Eiffel Tower!", "eiffel tower"), 1.0)

    def test_mismatch(self) -> None:
        self.assertEqual(answer_eval.exact_match("London", "Paris"), 0.0)

    def test_both_empty_counts_as_match(self) -> None:
        self.assertEqual(answer_eval.exact_match("", ""), 1.0)
        self.assertEqual(answer_eval.exact_match(None, ""), 1.0)


class F1Tests(unittest.TestCase):
    def test_identical_is_one(self) -> None:
        self.assertAlmostEqual(answer_eval.token_f1("New York City", "New York City"), 1.0)

    def test_partial_overlap(self) -> None:
        self.assertAlmostEqual(answer_eval.token_f1("New York", "York New City"), 0.8)

    def test_no_overlap_is_zero(self) -> None:
        self.assertEqual(answer_eval.token_f1("London", "Paris"), 0.0)

    def test_empty_cases(self) -> None:
        self.assertEqual(answer_eval.token_f1("", ""), 1.0)
        self.assertEqual(answer_eval.token_f1("Paris", ""), 0.0)
        self.assertEqual(answer_eval.token_f1("", "Paris"), 0.0)

    def test_duplicate_tokens_counted_once(self) -> None:
        self.assertAlmostEqual(answer_eval.token_f1("yes yes", "yes"), 2 / 3)


class PromptTests(unittest.TestCase):
    def test_prompt_contains_question_only(self) -> None:
        prompt = no_rag.build_prompt("  What is Paris?  ")
        self.assertEqual(prompt, "Question: What is Paris?\nAnswer:")

    def test_prompt_has_no_context(self) -> None:
        prompt = no_rag.build_prompt("Who wrote Hamlet?")
        self.assertNotIn("Context", prompt)
        self.assertIn("Who wrote Hamlet?", prompt)

    def test_system_prompt_is_shared_config(self) -> None:
        self.assertTrue(config.GENERATION_SYSTEM_PROMPT.startswith(
            "You are a question-answering assistant."))


class RecordSchemaTests(unittest.TestCase):
    REQUIRED_KEYS = {
        "question_id", "question", "gold_answer", "prediction",
        "retrieval_method", "retrieval_k", "retrieved_docs",
        "retrieval_doc_recall", "retrieval_sentence_recall",
        "retrieval_complete", "exact_match", "f1", "input_tokens",
        "output_tokens", "cached_input_tokens", "llm_calls",
        "latency_seconds", "api_cost_usd", "error", "success",
    }

    def test_success_record_schema(self) -> None:
        client = StubClient(["Paris"])
        response = client.generate("Question: x\nAnswer:", system="sys")
        record = no_rag.build_record(
            _example(), response, client=client)  # type: ignore[arg-type]
        self.assertTrue(self.REQUIRED_KEYS.issubset(record.keys()))
        self.assertEqual(record["retrieval_method"], "none")
        self.assertIsNone(record["retrieval_k"])
        self.assertEqual(record["retrieved_docs"], [])
        self.assertIsNone(record["retrieval_doc_recall"])
        self.assertIsNone(record["retrieval_sentence_recall"])
        self.assertIsNone(record["retrieval_complete"])
        self.assertEqual(record["prediction"], "Paris")
        self.assertEqual(record["exact_match"], 1.0)
        self.assertEqual(record["f1"], 1.0)
        self.assertEqual(record["input_tokens"], 10)
        self.assertEqual(record["output_tokens"], 2)
        self.assertEqual(record["cached_input_tokens"], 0)
        self.assertEqual(record["llm_calls"], 1)
        self.assertAlmostEqual(record["api_cost_usd"], 0.000001)
        self.assertEqual(record["latency_seconds"], 0.5)
        self.assertIsNone(record["error"])
        self.assertTrue(record["success"])

    def test_no_retrieved_documents_ever(self) -> None:
        client = StubClient(["London"])
        example = _example(answer="Paris")
        record = no_rag.build_record(
            example, client.generate("p"), client=client)  # type: ignore[arg-type]
        self.assertEqual(record["retrieved_docs"], [])
        self.assertEqual(record["exact_match"], 0.0)
        self.assertEqual(record["f1"], 0.0)
        self.assertNotIn("hallucination", record)

    def test_failure_record(self) -> None:
        client = StubClient()
        failed = LLMResponse(
            text="", model=client.model, llm_calls=3,
            error="boom", created_at="2026-01-01T00:00:00+00:00")
        record = no_rag.build_record(
            _example(), failed, client=client)  # type: ignore[arg-type]
        self.assertIsNone(record["prediction"])
        self.assertEqual(record["exact_match"], 0.0)
        self.assertEqual(record["f1"], 0.0)
        self.assertEqual(record["error"], "boom")
        self.assertFalse(record["success"])
        self.assertEqual(record["llm_calls"], 3)

    def test_identical_is_one(self) -> None:
        self.assertAlmostEqual(answer_eval.token_f1("New York City", "New York City"), 1.0)

    def test_partial_overlap(self) -> None:
        # pred={new,york}, gold={york,new,city}: P=1.0, R=2/3 -> F1=0.8
        self.assertAlmostEqual(answer_eval.token_f1("New York", "York New City"), 0.8)

    def test_no_overlap_is_zero(self) -> None:
        self.assertEqual(answer_eval.token_f1("London", "Paris"), 0.0)

    def test_empty_cases(self) -> None:
        self.assertEqual(answer_eval.token_f1("", ""), 1.0)
        self.assertEqual(answer_eval.token_f1("Paris", ""), 0.0)
        self.assertEqual(answer_eval.token_f1("", "Paris"), 0.0)

    def test_duplicate_tokens_counted_once(self) -> None:
        self.assertAlmostEqual(answer_eval.token_f1("yes yes", "yes"), 2 / 3)

class CacheAndMappingTests(unittest.TestCase):
    def test_cached_response_mapping(self) -> None:
        client = StubClient()
        cached = LLMResponse(
            text="Paris", model=client.model, prompt_tokens=10,
            completion_tokens=2, llm_calls=0, from_cache=True,
            cost_usd=0.000001, created_at="2026-01-01T00:00:00+00:00")
        record = no_rag.build_record(
            _example(), cached, client=client)  # type: ignore[arg-type]
        self.assertTrue(record["from_cache"])
        self.assertEqual(record["llm_calls"], 0)
        self.assertTrue(record["success"])

    def test_run_pipeline_no_retrieval(self) -> None:
        client = StubClient(["Paris"] * 3)
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "no_rag.jsonl"
            meta = Path(tmp) / "no_rag.meta.json"
            import src.pipelines.no_rag as module
            real_loader = module.dataset.load_dataset
            module.dataset.load_dataset = lambda: [  # type: ignore[assignment]
                _example(f"q{i}") for i in range(3)]
            try:
                metadata = module.run_pipeline(
                    client=client, results_file=results,  # type: ignore[arg-type]
                    meta_file=meta, quiet=True)
            finally:
                module.dataset.load_dataset = real_loader
            self.assertEqual(metadata["n_questions"], 3)
            self.assertEqual(metadata["successful"], 3)
            self.assertEqual(metadata["total_llm_calls"], 3)
            records = [json.loads(line)
                       for line in results.read_text().splitlines()]
            self.assertEqual(len(records), 3)
            self.assertTrue(all(r["retrieval_method"] == "none" for r in records))
            self.assertTrue(all(r["retrieved_docs"] == [] for r in records))
            self.assertEqual(client.systems[0], config.GENERATION_SYSTEM_PROMPT)

    def test_metadata_generation(self) -> None:
        client = StubClient(["Paris", "London"])
        responses = [client.generate("a"), client.generate("b")]
        examples = [_example("q1", answer="Paris"),
                    _example("q2", answer="Paris")]
        records = [no_rag.build_record(ex, resp, client=client)  # type: ignore[arg-type]
                   for ex, resp in zip(examples, responses)]
        metadata = no_rag.build_metadata(
            records, client=client, started_at="s",  # type: ignore[arg-type]
            finished_at="f", runtime_s=1.0)
        self.assertEqual(metadata["n_questions"], 2)
        self.assertEqual(metadata["successful"], 2)
        self.assertEqual(metadata["failed"], 0)
        self.assertAlmostEqual(metadata["exact_match"], 0.5)
        self.assertAlmostEqual(metadata["f1"], 0.5)
        self.assertEqual(metadata["total_llm_calls"], 2)
        self.assertEqual(metadata["total_input_tokens"], 20)
        self.assertEqual(metadata["total_output_tokens"], 4)
        self.assertEqual(metadata["llm"]["model"], config.DEEPSEEK_MODEL)
        self.assertEqual(metadata["dataset"]["sha256"], EXPECTED_DATASET_SHA256)


class FingerprintTests(unittest.TestCase):
    def test_dataset_fingerprint_unchanged(self) -> None:
        fingerprint = dataset.dataset_fingerprint()
        self.assertEqual(fingerprint["sha256"], EXPECTED_DATASET_SHA256)


if __name__ == "__main__":
    unittest.main()

