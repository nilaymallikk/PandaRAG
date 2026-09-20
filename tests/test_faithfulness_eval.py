"""Offline tests for the secondary faithfulness/hallucination evaluation.

No network: the DeepSeek judge is replaced by a stub. The tests pin the frozen
protocol rules -- judge isolation from gold/EM/identity, deterministic scoring
from claim labels, error paths -- so the instrument cannot silently drift.
"""

from __future__ import annotations

import json
import unittest

from src.llm import LLMResponse
from src.pipelines import adaptive_rag, hybrid_rag
from src import faithfulness_eval as fe


def _docs(titles=("France", "Paris"), texts=("Paris is the capital of France.", "Paris is in Europe.")):
    return [{"title": t, "text": x, "rank": i + 1} for i, (t, x) in enumerate(zip(titles, texts))]


def _record(qid="q1", prediction="Paris", question="What is the capital of France?"):
    return {
        "question_id": qid,
        "question": question,
        "prediction": prediction,
        "gold_answer": "SECRET_GOLD",
        "question_type": "bridge",
        "retrieval_k": 3,
        "final_k": 3,
        "exact_match": 1.0,
        "f1": 1.0,
        "retrieved_docs": _docs(),
    }


class StubJudge:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts = []
        self.calls = 0
        self.model = "deepseek-flash"
        self.thinking = "disabled"
        self.temperature = 0.0

    def generate(self, prompt, *, system=None, **_):
        self.prompts.append((prompt, system))
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=500,
            completion_tokens=60,
            total_tokens=560,
            latency_s=0.9,
            llm_calls=1,
            cost_usd=0.0001,
            finish_reason="stop",
            created_at="2026-01-01T00:00:00+00:00",
        )


class FakeClient(StubJudge):
    def settings(self):
        return {"model": self.model, "thinking": self.thinking, "temperature": self.temperature}


class PromptIsolationTests(unittest.TestCase):
    def test_render_context_matches_pipeline_format(self):
        docs = _docs()
        self.assertEqual(fe.render_context(docs), hybrid_rag.format_context(docs, {"France": "Paris is the capital of France.", "Paris": "Paris is in Europe."}))
        self.assertEqual(fe.render_context(docs), adaptive_rag.format_context(docs, {"France": "Paris is the capital of France.", "Paris": "Paris is in Europe."}))

    def test_judge_prompt_excludes_gold_em_and_identity(self):
        stub = StubJudge([{"claims": []}])
        entry, _ = fe.judge_one(_record(), stub, pipeline="hybrid_rag")
        prompt, system = stub.prompts[0]
        self.assertNotIn("SECRET_GOLD", prompt)
        self.assertNotIn("exact_match", prompt)
        self.assertNotIn("hybrid_rag", prompt)
        self.assertNotIn("standard_rag", prompt)
        self.assertNotIn("adaptive", prompt)
        self.assertIn("Paris is the capital of France.", prompt)
        self.assertIn("What is the capital of France?", prompt)
        self.assertEqual(system, fe.JUDGE_SYSTEM_PROMPT)

    def test_build_judge_prompt_signature_takes_only_question_context_answer(self):
        import inspect

        self.assertEqual(list(inspect.signature(fe.build_judge_prompt).parameters), ["question", "context", "answer"])


class ParsingTests(unittest.TestCase):
    def test_parses_fenced_json(self):
        payload = fe.parse_judge_json('```json\n{"claims": []}\n```')
        self.assertEqual(payload, {"claims": []})

    def test_parses_surrounding_prose(self):
        payload = fe.parse_judge_json('Here is the result: {"claims": [], "notes": "ok"} done.')
        self.assertEqual(payload["notes"], "ok")

    def test_rejects_unusable_output(self):
        with self.assertRaises(ValueError):
            fe.parse_judge_json("no json here")
        with self.assertRaises(ValueError):
            fe.parse_judge_json(None)

    def test_rejects_invalid_label(self):
        with self.assertRaises(ValueError):
            fe.normalise_claims({"claims": [{"claim": "x", "label": "maybe"}]})


class ScoringTests(unittest.TestCase):
    def test_faithfulness_is_supported_over_total(self):
        s = fe.score_claims([{"label": "supported"}, {"label": "supported"}, {"label": "unsupported"}, {"label": "contradicted"}])
        self.assertEqual(s["n_claims"], 4)
        self.assertAlmostEqual(s["faithfulness"], 0.5)
        self.assertAlmostEqual(s["hallucination_score"], 0.5)
        self.assertEqual(s["hallucination_rate"], 1)
        self.assertEqual(s["grounding_label"], "partially")

    def test_fully_and_not_grounded_labels(self):
        all_sup = fe.score_claims([{"label": "supported"}] * 3)
        self.assertEqual(all_sup["grounding_label"], "fully")
        self.assertEqual(all_sup["hallucination_rate"], 0)
        none_sup = fe.score_claims([{"label": "unsupported"}, {"label": "contradicted"}])
        self.assertEqual(none_sup["grounding_label"], "not")
        self.assertEqual(none_sup["hallucination_rate"], 1)

    def test_no_claims_is_unscored_not_perfect(self):
        s = fe.score_claims([])
        self.assertIsNone(s["faithfulness"])
        self.assertIsNone(s["hallucination_score"])
        self.assertEqual(s["grounding_label"], "no_checkable_claims")


class JudgementTests(unittest.TestCase):
    def test_judge_one_records_claims_and_scores(self):
        payload = {
            "claims": [
                {"claim": "Paris is the capital of France", "label": "supported", "evidence": "Paris is the capital of France."},
                {"claim": "Paris has 3 million people", "label": "unsupported", "evidence": "none"},
            ],
            "notes": "one claim unsupported",
        }
        entry, response = fe.judge_one(_record(), StubJudge([payload]), pipeline="hybrid_rag")
        self.assertTrue(entry["parse_ok"])
        self.assertAlmostEqual(entry["faithfulness"], 0.5)
        self.assertEqual(entry["hallucination_rate"], 1)
        self.assertEqual(len(entry["unsupported_claims"]), 1)
        self.assertEqual(len(entry["supported_claims"]), 1)
        self.assertEqual(entry["grounding_label"], "partially")
        self.assertEqual(entry["judge_input_tokens"], 500)
        self.assertEqual(entry["judge_llm_calls"], 1)
        self.assertTrue(response.ok)

    def test_parse_failure_is_recorded_not_dropped(self):
        entry, _ = fe.judge_one(_record(), StubJudge(["not json at all"]), pipeline="standard_rag")
        self.assertFalse(entry["parse_ok"])
        self.assertEqual(entry["grounding_label"], "judge_error")
        self.assertIn("parse_error", entry)

    def test_api_failure_is_recorded_not_dropped(self):
        class Failing(StubJudge):
            def generate(self, prompt, *, system=None, **_):
                self.calls += 1
                return LLMResponse(text="", model=self.model, error="api down", llm_calls=0)

        entry, _ = fe.judge_one(_record(), Failing([{}]), pipeline="adaptive_rag")
        self.assertFalse(entry["parse_ok"])
        self.assertEqual(entry["grounding_label"], "judge_error")
        self.assertEqual(entry["judge_error"], "api down")

    def test_identical_protocol_across_pipelines(self):
        prompts = {}
        for p in fe.JUDGE_PIPELINES:
            entries, stub = None, StubJudge([{"claims": []}])
            fe.judge_one(_record(), stub, pipeline=p)
            prompts[p] = stub.prompts[0][0]
        self.assertEqual(len(set(prompts.values())), 1)


class AggregationTests(unittest.TestCase):
    def test_summarise_excludes_unscored_and_reports_rates(self):
        entries = [
            {"faithfulness": 1.0, "hallucination_score": 0.0, "hallucination_rate": 0, "grounding_label": "fully", "n_claims": 1, "parse_ok": True, "claims": [{"label": "supported"}]},
            {"faithfulness": 0.0, "hallucination_score": 1.0, "hallucination_rate": 1, "grounding_label": "not", "n_claims": 1, "parse_ok": True, "claims": [{"label": "unsupported"}]},
            {"faithfulness": None, "hallucination_score": None, "hallucination_rate": None, "grounding_label": "no_checkable_claims", "n_claims": 0, "parse_ok": True, "claims": []},
            {"faithfulness": None, "hallucination_score": None, "hallucination_rate": None, "grounding_label": "judge_error", "n_claims": None, "parse_ok": False, "claims": []},
        ]
        s = fe.summarise(entries)
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["n_scored"], 2)
        self.assertEqual(s["n_no_checkable_claims"], 1)
        self.assertEqual(s["n_judge_errors"], 1)
        self.assertAlmostEqual(s["faithfulness"], 0.5)
        self.assertAlmostEqual(s["hallucination_rate"], 0.5)
        self.assertAlmostEqual(s["claim_supported_rate"], 0.5)


class ProtocolFreezeTests(unittest.TestCase):
    def test_prompt_is_frozen_and_carries_no_gold_slots(self):
        for token in ("{context}", "{question}", "{answer}"):
            self.assertIn(token, fe.JUDGE_PROMPT_TEMPLATE)
        self.assertNotIn("gold", fe.JUDGE_PROMPT_TEMPLATE.lower())
        self.assertNotIn("{question_type}", fe.JUDGE_PROMPT_TEMPLATE)
        self.assertNotIn("{final_k}", fe.JUDGE_PROMPT_TEMPLATE)

    def test_judge_uses_same_model_as_experiment(self):
        self.assertEqual(fe.JUDGE_MAX_TOKENS, 768)
        self.assertEqual(fe.PROTOCOL_VERSION, "faithfulness-judge/1.0")
        self.assertEqual(fe.JUDGE_PIPELINES, ("standard_rag", "hybrid_rag", "adaptive_rag"))


if __name__ == "__main__":
    unittest.main()
