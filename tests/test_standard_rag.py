"""Offline tests for the Standard Dense RAG baseline.

Everything here is offline: the DeepSeek API is replaced by a stub client,
and the frozen dense retrieval artifact is read but never modified.
Run with::

    .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from src import config, dataset
from src.llm import LLMResponse
from src.pipelines import standard_rag

EXPECTED_DATASET_SHA256 = "956405ceb43d73385d60898e1bfc7f45a8184c9a4e1eff45b9a8ec239acfd048"


def _retrieval_record(question_id="q1", n_docs=7):
    docs = [{"rank": i + 1, "title": f"Doc{i + 1}", "score": 1.0 / (i + 1)} for i in range(n_docs)]
    return {
        "question_id": question_id,
        "retrieved_documents": docs,
        "metrics": {"5": {"k": 5, "document_recall": 0.5, "document_complete": 0.0, "sentence_recall": 0.25, "sentence_complete": 0.0}},
    }


def _example(example_id="q1"):
    titles = tuple(f"Doc{i + 1}" for i in range(7))
    sentences = tuple((f"Sentence about {t}.",) for t in titles)
    return dataset.HotpotExample(
        example_id=example_id, question="Which doc?", answer="Doc2",
        question_type="bridge", level="hard", context_titles=titles,
        context_sentences=sentences, supporting_facts=(("Doc2", 0), ("Doc4", 0)),
    )


class StubClient:
    def __init__(self, texts=None):
        self.texts = list(texts or ["Doc2"])
        self.prompts = []
        self.systems = []
        self.model = config.DEEPSEEK_MODEL
        self.thinking = "disabled"
        self.calls = 0

    def generate(self, prompt, *, system=None, **_):
        self.prompts.append(prompt)
        self.systems.append(system)
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return LLMResponse(text=text, model=self.model, prompt_tokens=100, completion_tokens=3, latency_s=0.6, llm_calls=1, cost_usd=0.000002, finish_reason="stop", created_at="2026-01-01T00:00:00+00:00")

    def settings(self):
        return {"provider": "DeepSeek", "model": self.model, "thinking": self.thinking, "temperature": 0.0, "max_output_tokens": config.MAX_OUTPUT_TOKENS}


class FrozenArtifactTests(unittest.TestCase):
    def test_top_k_is_five(self):
        self.assertEqual(standard_rag.TOP_K, 5)
        self.assertEqual(config.STANDARD_RAG_TOP_K, 5)

    def test_load_and_verify_real_artifact(self):
        raw = standard_rag.load_dense_records()
        self.assertEqual(len(raw), 500)
        by_id = standard_rag.verify_retrieval_artifact(raw)
        self.assertEqual(len(by_id), 500)

    def test_verify_rejects_wrong_fingerprint(self):
        with self.assertRaises(ValueError):
            standard_rag.verify_retrieval_artifact([_retrieval_record()], expected_sha256="0" * 64)

    def test_verify_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            standard_rag.verify_retrieval_artifact([_retrieval_record(), _retrieval_record()])

    def test_verify_rejects_missing_k5(self):
        bad = _retrieval_record()
        bad["metrics"] = {}
        with self.assertRaises(ValueError):
            standard_rag.verify_retrieval_artifact([bad])

    def test_top_k_enforced(self):
        hits = standard_rag.top_k_hits(_retrieval_record(n_docs=10))
        self.assertEqual(len(hits), 5)
        self.assertEqual([h["rank"] for h in hits], [1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            standard_rag.top_k_hits(_retrieval_record(), 3)


class ContextPromptTests(unittest.TestCase):
    def test_format_context_deterministic_order(self):
        hits = standard_rag.top_k_hits(_retrieval_record(n_docs=6))
        paras = {f"Doc{i + 1}": f"Text {i + 1}." for i in range(6)}
        first = standard_rag.format_context(hits, paras)
        second = standard_rag.format_context(hits, paras)
        self.assertEqual(first, second)
        for i in range(1, 6):
            self.assertIn(f"[Document {i}]", first)
            self.assertIn(f"Title: Doc{i}", first)
        self.assertLess(first.index("Doc1"), first.index("Doc2"))
        self.assertNotIn("Doc6", first)

    def test_format_context_rejects_unknown_title(self):
        with self.assertRaises(KeyError):
            standard_rag.format_context([{"rank": 1, "title": "Missing", "score": 1.0}], {})

    def test_build_prompt_contains_question_and_context(self):
        prompt = standard_rag.build_prompt("Which doc?", "Context:\n[Document 1]\nTitle: Doc1\nText.")
        self.assertTrue(prompt.startswith("Question: Which doc?"))
        self.assertIn("[Document 1]", prompt)
        self.assertIn("using the supplied context", prompt)
        self.assertTrue(prompt.rstrip().endswith("Answer:"))

    def test_system_prompt_is_shared_config(self):
        self.assertIsNotNone(config.GENERATION_SYSTEM_PROMPT)
        self.assertIn("supplied", config.GENERATION_SYSTEM_PROMPT)

    def test_no_forbidden_imports(self):
        # NOTE: other test modules (test_retrieval) legitimately import the
        # retrieval stack, so sys.modules is polluted process-wide. The
        # protocol guarantee is about THIS module's own import surface,
        # verified statically below via AST.
        import ast
        tree = ast.parse(Path(standard_rag.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module)
        for mod in ("rank_bm25", "faiss", "sentence_transformers"):
            self.assertFalse(any(m == mod or m.startswith(mod + ".") for m in imported), mod)
        self.assertFalse(any(m == "src.retrieval" or m.startswith("src.retrieval.") for m in imported), "src.retrieval")


class RecordSchemaTests(unittest.TestCase):
    REQUIRED_KEYS = {"question_id", "question", "gold_answer", "prediction", "retrieval_method", "retrieval_k", "retrieved_docs", "retrieval_doc_recall", "retrieval_sentence_recall", "retrieval_complete", "exact_match", "f1", "input_tokens", "output_tokens", "cached_input_tokens", "llm_calls", "latency_seconds", "api_cost_usd", "error", "success"}

    def test_success_record_schema(self):
        client = StubClient(["Doc2"])
        record = standard_rag.build_record(_example(), _retrieval_record(), client.generate("p"), client=client)
        self.assertTrue(self.REQUIRED_KEYS.issubset(record.keys()))
        self.assertEqual(record["retrieval_method"], "dense")
        self.assertEqual(record["retrieval_k"], 5)
        self.assertEqual(len(record["retrieved_docs"]), 5)
        for i, doc in enumerate(record["retrieved_docs"], start=1):
            self.assertEqual(doc["rank"], i)
            self.assertIn("title", doc)
            self.assertIn("score", doc)
            self.assertIn("text", doc)
        self.assertEqual([d["title"] for d in record["retrieved_docs"]], [f"Doc{i}" for i in range(1, 6)])
        self.assertEqual(record["retrieval_doc_recall"], 0.5)
        self.assertEqual(record["retrieval_sentence_recall"], 0.25)
        self.assertFalse(record["retrieval_complete"])
        self.assertIsNone(record["faithfulness"])
        self.assertIsNone(record["hallucination"])
        self.assertEqual(record["prediction"], "Doc2")
        self.assertEqual(record["exact_match"], 1.0)
        self.assertTrue(record["success"])
        self.assertEqual(record["input_tokens"], 100)
        self.assertEqual(record["llm_calls"], 1)

    def test_failure_record(self):
        client = StubClient()
        failed = LLMResponse(text="", model=client.model, llm_calls=2, error="boom", created_at="2026-01-01T00:00:00+00:00")
        record = standard_rag.build_record(_example(), _retrieval_record(), failed, client=client)
        self.assertIsNone(record["prediction"])
        self.assertEqual(record["exact_match"], 0.0)
        self.assertEqual(record["f1"], 0.0)
        self.assertEqual(record["error"], "boom")
        self.assertFalse(record["success"])
        self.assertEqual(len(record["retrieved_docs"]), 5)

    def test_cached_response_mapping(self):
        client = StubClient()
        cached = LLMResponse(text="Doc2", model=client.model, prompt_tokens=100, completion_tokens=3, llm_calls=0, from_cache=True, cost_usd=0.1, created_at="2026-01-01T00:00:00+00:00")
        record = standard_rag.build_record(_example(), _retrieval_record(), cached, client=client)
        self.assertTrue(record["from_cache"])
        self.assertEqual(record["llm_calls"], 0)
        self.assertTrue(record["success"])

    def test_run_pipeline_offline(self):
        client = StubClient(["Doc2"] * 2)
        retrieval = [_retrieval_record("q1"), _retrieval_record("q2")]
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "s.jsonl"
            meta = Path(tmp) / "s.meta.json"
            import src.pipelines.standard_rag as module
            real_loader = module.dataset.load_dataset
            examples = [_example("q1"), _example("q2")]
            module.dataset.load_dataset = lambda: examples
            try:
                metadata = module.run_pipeline(client=client, retrieval_records=retrieval, results_file=results, meta_file=meta, quiet=True)
            finally:
                module.dataset.load_dataset = real_loader
            self.assertEqual(metadata["n_questions"], 2)
            self.assertEqual(metadata["successful"], 2)
            self.assertEqual(metadata["retrieval_k"], 5)
            rows = [json.loads(line) for line in results.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(len(r["retrieved_docs"]) == 5 for r in rows))
            self.assertTrue(all(r["retrieval_method"] == "dense" for r in rows))
            self.assertEqual(client.systems[0], config.GENERATION_SYSTEM_PROMPT)
            self.assertIn("Context:", client.prompts[0])
            self.assertIn("using the supplied context", client.prompts[0])

    def test_metadata_generation(self):
        client = StubClient(["Doc2", "Other"])
        records = [standard_rag.build_record(_example(f"q{i}"), _retrieval_record(f"q{i}"), client.generate("p"), client=client) for i in (1, 2)]
        metadata = standard_rag.build_metadata(records, client=client, started_at="s", finished_at="f", runtime_s=1.0, retrieval_meta={"dataset": {"sha256": EXPECTED_DATASET_SHA256}, "retriever": {"retrieval_method": "dense"}})
        self.assertEqual(metadata["pipeline"], "standard_rag")
        self.assertEqual(metadata["retrieval_k"], 5)
        self.assertEqual(metadata["n_questions"], 2)
        self.assertEqual(metadata["total_llm_calls"], 2)
        self.assertEqual(metadata["dataset_sha256"], EXPECTED_DATASET_SHA256)
        self.assertEqual(metadata["faithfulness"], "deferred (no LLM judge; no extra API call)")
        self.assertEqual(metadata["total_retrieved_docs"], 10)
        self.assertEqual(metadata["mean_retrieved_docs"], 5.0)
        self.assertIn("mean_input_tokens", metadata)
        self.assertIn("mean_output_tokens", metadata)


class FingerprintTests(unittest.TestCase):
    def test_dataset_fingerprint_unchanged(self):
        self.assertEqual(dataset.dataset_fingerprint()["sha256"], EXPECTED_DATASET_SHA256)


if __name__ == "__main__":
    unittest.main()
