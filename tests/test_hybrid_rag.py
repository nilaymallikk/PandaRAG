"""Offline tests for the Hybrid RAG pipeline (frozen RRF artifact, no network)."""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from src import config, dataset
from src.llm import LLMResponse
from src.pipelines import hybrid_rag

EXPECTED_DATASET_SHA256 = "956405ceb43d73385d60898e1bfc7f45a8184c9a4e1eff45b9a8ec239acfd048"

def _retrieval_record(question_id="q1", n_docs=7):
    docs = [{"rank": i+1, "title": f"Doc{i+1}", "score": 0.03/(i+1), "rrf_score": 0.03/(i+1), "dense_rank": i+1, "bm25_rank": 7-i, "dense_score": 0.9-0.05*i, "bm25_score": 5.0+float(i)} for i in range(n_docs)]
    return {"question_id": question_id, "retrieval_method": "hybrid", "retrieved_documents": docs, "metrics": {"5": {"k": 5, "document_recall": 0.5, "document_complete": 0.0, "sentence_recall": 0.25, "sentence_complete": 0.0}}}

def _example(example_id="q1"):
    titles = tuple(f"Doc{i+1}" for i in range(7))
    sentences = tuple((f"Sentence about {t}.",) for t in titles)
    return dataset.HotpotExample(example_id=example_id, question="Which doc?", answer="Doc2", question_type="bridge", level="hard", context_titles=titles, context_sentences=sentences, supporting_facts=(("Doc2", 0), ("Doc4", 0)))

class StubClient:
    def __init__(self, texts=None):
        self.texts = list(texts or ["Doc2"])
        self.prompts = []; self.systems = []
        self.model = config.DEEPSEEK_MODEL; self.thinking = "disabled"; self.calls = 0
    def generate(self, prompt, *, system=None, **_):
        self.prompts.append(prompt); self.systems.append(system)
        text = self.texts[min(self.calls, len(self.texts)-1)]; self.calls += 1
        return LLMResponse(text=text, model=self.model, prompt_tokens=100, completion_tokens=3, latency_s=0.6, llm_calls=1, cost_usd=0.000002, finish_reason="stop", created_at="2026-01-01T00:00:00+00:00")
    def settings(self):
        return {"provider": "DeepSeek", "model": self.model, "thinking": self.thinking, "temperature": 0.0, "max_output_tokens": config.MAX_OUTPUT_TOKENS}

class FrozenArtifactTests(unittest.TestCase):
    def test_top_k_is_five(self):
        self.assertEqual(hybrid_rag.TOP_K, 5); self.assertEqual(config.HYBRID_RAG_TOP_K, 5)
    def test_load_and_verify_real_artifact(self):
        raw = hybrid_rag.load_hybrid_records()
        self.assertEqual(len(raw), 500)
        by_id = hybrid_rag.verify_retrieval_artifact(raw)
        self.assertEqual(len(by_id), 500)
        self.assertTrue(all(r.get("retrieval_method","hybrid")=="hybrid" for r in raw))
    def test_rrf_config_frozen(self):
        rrf = hybrid_rag.rrf_config()
        self.assertEqual(rrf["rrf_constant"], 60.0); self.assertEqual(rrf["rrf_depth"], 10)
    def test_verify_rejects_wrong_fingerprint(self):
        with self.assertRaises(ValueError):
            hybrid_rag.verify_retrieval_artifact([_retrieval_record()], expected_sha256="0"*64)
    def test_verify_rejects_wrong_method(self):
        bad = _retrieval_record(); bad["retrieval_method"] = "dense"
        with self.assertRaises(ValueError):
            hybrid_rag.verify_retrieval_artifact([bad])
    def test_verify_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            hybrid_rag.verify_retrieval_artifact([_retrieval_record(), _retrieval_record()])
    def test_verify_rejects_missing_k5(self):
        bad = _retrieval_record(); bad["metrics"] = {}
        with self.assertRaises(ValueError):
            hybrid_rag.verify_retrieval_artifact([bad])
    def test_top_k_enforced(self):
        hits = hybrid_rag.top_k_hits(_retrieval_record(n_docs=10))
        self.assertEqual(len(hits), 5)
        self.assertEqual([h["rank"] for h in hits], [1,2,3,4,5])
        with self.assertRaises(ValueError):
            hybrid_rag.top_k_hits(_retrieval_record(), 3)

class ContextPromptTests(unittest.TestCase):
    def test_format_context_matches_standard_layout(self):
        from src.pipelines import standard_rag
        hits = hybrid_rag.top_k_hits(_retrieval_record(n_docs=6))
        paras = {f"Doc{i+1}": f"Text {i+1}." for i in range(6)}
        h = hybrid_rag.format_context(hits, paras)
        s = standard_rag.format_context(hits, paras)
        self.assertEqual(h, s)
        self.assertLess(h.index("Doc1"), h.index("Doc2"))
        self.assertNotIn("Doc6", h)
    def test_format_context_rejects_unknown_title(self):
        with self.assertRaises(KeyError):
            hybrid_rag.format_context([{"rank":1,"title":"Missing","score":1.0}], {})
    def test_build_prompt_same_structure_as_standard(self):
        from src.pipelines import standard_rag
        ctx = "Context:\n[Document 1]\nTitle: Doc1\nText."
        self.assertEqual(hybrid_rag.build_prompt("Which doc?", ctx), standard_rag.build_prompt("Which doc?", ctx))
    def test_system_prompt_is_shared_config(self):
        self.assertIn("supplied", config.GENERATION_SYSTEM_PROMPT)
    def test_no_forbidden_imports(self):
        import ast
        tree = ast.parse(Path(hybrid_rag.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module: imported.add(node.module)
        for mod in ("rank_bm25", "faiss", "sentence_transformers"):
            self.assertFalse(any(m==mod or m.startswith(mod+".") for m in imported), mod)
        self.assertFalse(any(m=="src.retrieval" or m.startswith("src.retrieval.") for m in imported))
        self.assertFalse(any(m in ("src.pipelines.standard_rag","src.pipelines.adaptive_rag") for m in imported))

class RecordSchemaTests(unittest.TestCase):
    REQUIRED_KEYS = {"question_id","question","gold_answer","prediction","retrieval_method","retrieval_k","retrieved_docs","retrieval_doc_recall","retrieval_sentence_recall","retrieval_complete","exact_match","f1","input_tokens","output_tokens","cached_input_tokens","llm_calls","latency_seconds","api_cost_usd","error","success"}
    def test_success_record_schema_and_provenance(self):
        client = StubClient(["Doc2"])
        record = hybrid_rag.build_record(_example(), _retrieval_record(), client.generate("p"), client=client)
        self.assertTrue(self.REQUIRED_KEYS.issubset(record.keys()))
        self.assertEqual(record["retrieval_method"], "hybrid")
        self.assertEqual(record["retrieval_k"], 5)
        self.assertEqual(len(record["retrieved_docs"]), 5)
        d0 = record["retrieved_docs"][0]
        for k in ("title","rank","rrf_score","dense_rank","bm25_rank","dense_score","bm25_score","text"):
            self.assertIn(k, d0)
        self.assertEqual(d0["title"], "Doc1")
        self.assertIn("Sentence about Doc1.", d0["text"])
        self.assertIsNone(record["faithfulness"]); self.assertIsNone(record["hallucination"])
        self.assertEqual(record["model"], config.DEEPSEEK_MODEL)
        self.assertEqual(record["thinking"], "disabled")
    def test_retrieval_metric_mapping(self):
        client = StubClient(["Doc2"])
        record = hybrid_rag.build_record(_example(), _retrieval_record(), client.generate("p"), client=client)
        self.assertEqual(record["retrieval_doc_recall"], 0.5)
        self.assertEqual(record["retrieval_sentence_recall"], 0.25)
        self.assertFalse(record["retrieval_complete"])
    def test_failure_record(self):
        client = StubClient()
        failed = LLMResponse(text="", model=client.model, error="boom", llm_calls=1, created_at="2026-01-01T00:00:00+00:00")
        record = hybrid_rag.build_record(_example(), _retrieval_record(), failed, client=client)
        self.assertIsNone(record["prediction"])
        self.assertEqual(record["exact_match"], 0.0); self.assertEqual(record["f1"], 0.0)
        self.assertFalse(record["success"]); self.assertEqual(len(record["retrieved_docs"]), 5)
    def test_cached_response_mapping(self):
        client = StubClient()
        cached = LLMResponse(text="Doc2", model=client.model, prompt_tokens=100, completion_tokens=3, llm_calls=0, from_cache=True, cost_usd=0.1, created_at="2026-01-01T00:00:00+00:00")
        record = hybrid_rag.build_record(_example(), _retrieval_record(), cached, client=client)
        self.assertTrue(record["from_cache"]); self.assertEqual(record["llm_calls"], 0)
    def test_ordering_preserved(self):
        client = StubClient(["Doc2"])
        rec = _retrieval_record(); rec["retrieved_documents"] = list(reversed(rec["retrieved_documents"]))
        record = hybrid_rag.build_record(_example(), rec, client.generate("p"), client=client)
        self.assertEqual([d["rank"] for d in record["retrieved_docs"]], [7,6,5,4,3])
    def test_run_pipeline_offline(self):
        client = StubClient(["Doc2"]*2)
        retrieval = [_retrieval_record("q1"), _retrieval_record("q2")]
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)/"h.jsonl"; meta = Path(tmp)/"h.meta.json"
            import src.pipelines.hybrid_rag as module
            real_loader = module.dataset.load_dataset
            examples = [_example("q1"), _example("q2")]
            module.dataset.load_dataset = lambda: examples
            try:
                metadata = module.run_pipeline(client=client, retrieval_records=retrieval, results_file=results, meta_file=meta, quiet=True)
            finally:
                module.dataset.load_dataset = real_loader
            self.assertEqual(metadata["n_questions"], 2)
            self.assertEqual(metadata["rrf_constant"], 60.0); self.assertEqual(metadata["rrf_depth"], 10)
            rows = [json.loads(line) for line in results.read_text().splitlines()]
            self.assertTrue(all(r["retrieval_method"]=="hybrid" for r in rows))
            self.assertEqual(client.systems[0], config.GENERATION_SYSTEM_PROMPT)
    def test_metadata_generation(self):
        client = StubClient(["Doc2","Other"])
        records = [hybrid_rag.build_record(_example(f"q{i}"), _retrieval_record(f"q{i}"), client.generate("p"), client=client) for i in (1,2)]
        rmeta = {"dataset": {"sha256": EXPECTED_DATASET_SHA256}, "retriever": {"retrieval_method": "hybrid", "fusion": {"rrf_constant": 60.0, "rrf_depth": 10}}}
        metadata = hybrid_rag.build_metadata(records, client=client, started_at="s", finished_at="f", runtime_s=1.0, retrieval_meta=rmeta)
        self.assertEqual(metadata["pipeline"], "hybrid_rag")
        self.assertEqual(metadata["rrf_constant"], 60.0); self.assertEqual(metadata["rrf_depth"], 10)
        self.assertEqual(metadata["dataset_sha256"], EXPECTED_DATASET_SHA256)

class FingerprintTests(unittest.TestCase):
    def test_dataset_fingerprint_unchanged(self):
        self.assertEqual(dataset.dataset_fingerprint()["sha256"], EXPECTED_DATASET_SHA256)

if __name__ == "__main__":
    unittest.main()
