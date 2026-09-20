"""Offline tests for the frozen Adaptive RAG policy and pipeline.

Zero network/API calls: DeepSeek is replaced by a stub client. The decision
function is exercised with retrieval-only input to demonstrate that gold
information cannot reach the controller.
"""

from __future__ import annotations

import ast
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

from src import answer_eval, config, dataset
from src.llm import LLMResponse
from src.pipelines import adaptive_rag

EXPECTED_DATASET_SHA256 = "956405ceb43d73385d60898e1bfc7f45a8184c9a4e1eff45b9a8ec239acfd048"
W_NORM = (2.0 / 61.0) - (2.0 / 70.0)


def _hit(rank, rrf, dense_rank, bm25_rank, title=None):
    return {
        "rank": rank,
        "title": title or f"Doc{rank}",
        "score": rrf,
        "rrf_score": rrf,
        "dense_rank": dense_rank,
        "bm25_rank": bm25_rank,
        "dense_score": 0.8 - 0.01 * dense_rank,
        "bm25_score": 5.0 - 0.1 * bm25_rank,
    }


def _hits(rrf=(0.0327, 0.0325, 0.0323, 0.0321, 0.0319), pairs=((1, 1), (1, 2), (2, 1), (2, 2), (3, 3))):
    return [_hit(i + 1, rrf[i], pairs[i][0], pairs[i][1]) for i in range(len(rrf))]


def _example():
    return dataset.HotpotExample(
        example_id="q1",
        question="Who is the capital of France named after?",
        answer="Paris",
        question_type="bridge",
        level="hard",
        context_titles=("France", "Paris", "Doc3", "Doc4", "Doc5"),
        context_sentences=(
            ("Paris is the capital of France.", "France is in Europe."),
            ("Paris was named by the Romans.", "Paris has two million people."),
            ("Doc3 sentence one.",),
            ("Doc4 sentence one.",),
            ("Doc5 sentence one.",),
        ),
        supporting_facts=(("France", 0), ("Paris", 0)),
    )


def _retrieval_record(hits=None):
    hits = hits if hits is not None else _hits()
    pool = len(hits)
    return {
        "question_id": "q1",
        "retrieval_method": "hybrid",
        "retrieved_documents": hits,
        "metrics": {
            "3": {"k": 3, "document_recall": 0.5, "document_complete": 0.0, "sentence_recall": 0.5, "sentence_complete": 0.0},
            "5": {"k": 5, "document_recall": 1.0, "document_complete": 1.0, "sentence_recall": 1.0, "sentence_complete": 1.0},
        },
    }


def _response(ok=True, text="Paris", from_cache=False):
    return LLMResponse(
        text=text if ok else None,
        model="deepseek-flash",
        prompt_tokens=100,
        completion_tokens=2,
        cached_input_tokens=10 if from_cache else 0,
        total_tokens=102,
        latency_s=0.4,
        llm_calls=0 if from_cache else 1,
        cost_usd=0.000002,
        finish_reason="stop",
        from_cache=from_cache,
        created_at="2026-01-01T00:00:00+00:00",
        error=None if ok else "stub failure",
    )


class StubClient:
    """Offline stand-in for DeepSeekClient (records prompts, no network)."""

    def __init__(self, texts=None):
        self.texts = list(texts or ["Paris"])
        self.prompts = []
        self.systems = []
        self.model = "deepseek-flash"
        self.thinking = "disabled"
        self.calls = 0

    def generate(self, prompt, *, system=None, **_):
        self.prompts.append(prompt)
        self.systems.append(system)
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return _response(text=text)

    def settings(self):
        return {
            "provider": "DeepSeek",
            "model": self.model,
            "thinking": self.thinking,
            "temperature": 0.0,
            "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        }


class PolicyFreezeTests(unittest.TestCase):
    def test_check_frozen_policy_passes(self):
        policy = adaptive_rag.check_frozen_policy()
        self.assertEqual(policy["initial_k"], 3)
        self.assertEqual(policy["allowed_final_k"], [3, 5])
        self.assertEqual(policy["agreement_weight"], 0.5)
        self.assertEqual(policy["margin_weight"], 0.5)
        self.assertEqual(policy["threshold"], 0.5)
        self.assertEqual(policy["agreement_rank_range"], [2, 20])
        self.assertAlmostEqual(policy["margin_normalizer_W"], W_NORM)

    def test_config_constants_frozen(self):
        self.assertEqual(config.ADAPTIVE_RAG_INITIAL_K, 3)
        self.assertEqual(config.ADAPTIVE_RAG_THRESHOLD, 0.5)
        self.assertEqual(config.ADAPTIVE_RAG_AGREEMENT_WEIGHT, 0.5)
        self.assertEqual(config.ADAPTIVE_RAG_MARGIN_WEIGHT, 0.5)
        self.assertAlmostEqual(config.ADAPTIVE_RAG_MARGIN_NORMALIZER_W, 2 / 61 - 2 / 70, places=15)
        self.assertEqual(config.ADAPTIVE_RAG_ALLOWED_FINAL_K, (3, 5))


class AgreementTests(unittest.TestCase):
    def test_formula_top_agreement(self):
        self.assertAlmostEqual(adaptive_rag.agreement(1, 1), 1.0)

    def test_formula_general(self):
        self.assertAlmostEqual(adaptive_rag.agreement(1, 2), 1.0 - (3 - 2) / 18)
        self.assertAlmostEqual(adaptive_rag.agreement(10, 10), 1.0 - 18 / 18)

    def test_theoretical_normalization_bounds(self):
        self.assertGreaterEqual(adaptive_rag.agreement(1, 1), 1.0)
        self.assertLessEqual(adaptive_rag.agreement(10, 10), 0.0 + 1e-12)
        self.assertGreater(adaptive_rag.agreement(2, 1), adaptive_rag.agreement(2, 2))


class MarginTests(unittest.TestCase):
    def test_formula_raw_and_normalized(self):
        raw, m = adaptive_rag.boundary_margin(5, 0.0323, 0.0321)
        self.assertAlmostEqual(raw, 0.0002)
        self.assertAlmostEqual(m, 0.0002 / W_NORM)

    def test_pool_below_four_is_zero(self):
        for pool in (2, 3):
            raw, m = adaptive_rag.boundary_margin(pool, 0.0323, 0.0321)
            self.assertEqual(raw, 0.0)
            self.assertEqual(m, 0.0)

    def test_missing_scores_are_zero(self):
        raw, m = adaptive_rag.boundary_margin(5, None, 0.0321)
        self.assertEqual((raw, m), (0.0, 0.0))

    def test_margin_non_negative_for_valid_ranks(self):
        # RRF scores strictly decrease with rank, so s3 >= s4.
        raw, m = adaptive_rag.boundary_margin(10, 1 / 63, 1 / 64)
        self.assertGreaterEqual(raw, 0.0)
        self.assertGreaterEqual(m, 0.0)


class ConfidenceTests(unittest.TestCase):
    def test_formula(self):
        self.assertAlmostEqual(adaptive_rag.confidence_score(1.0, 0.0), 0.5)
        self.assertAlmostEqual(adaptive_rag.confidence_score(0.0, 1.0), 0.5)
        self.assertAlmostEqual(adaptive_rag.confidence_score(1.0, 1.0), 1.0)

    def test_is_not_probability_claim(self):
        # Weights 0.5/0.5 with signals in [0,1] keep confidence in [0,1].
        self.assertTrue(0.0 <= adaptive_rag.confidence_score(0.3, 0.9) <= 1.0)


class DecisionTests(unittest.TestCase):
    def test_confidence_above_threshold_uses_k3(self):
        decision = adaptive_rag.decide_adaptive_k(_hits())
        self.assertEqual(decision["final_k"], 3)

    def test_low_margin_escalates_to_k5(self):
        # (d1,b1)=(10,10) -> A=0; s3 == s4 -> margin 0 -> confidence 0 -> K=5.
        hits = _hits(
            rrf=(0.0327, 0.0325, 0.0323, 0.0323, 0.0319),
            pairs=((10, 10), (1, 2), (2, 1), (2, 2), (3, 3)),
        )
        decision = adaptive_rag.decide_adaptive_k(hits)
        self.assertEqual(decision["final_k"], 5)
        self.assertAlmostEqual(decision["confidence"], 0.0)

    def test_exact_threshold_equals_k3(self):
        # Construct A=1 (1,1) and M such that 0.5*A+0.5*M == 0.5 exactly -> M=0.
        hits = _hits(rrf=(1 / 62, 1 / 62, 1 / 62, 1 / 62, 1 / 62), pairs=((1, 1),) * 5)
        decision = adaptive_rag.decide_adaptive_k(hits)
        self.assertAlmostEqual(decision["confidence"], 0.5)
        self.assertEqual(decision["final_k"], 3)

    def test_final_k_always_in_allowed_set(self):
        for d1 in (1, 3, 10):
            for s3s4 in ((1 / 63, 1 / 64), (1 / 63, 1 / 63)):
                hits = _hits(pairs=((d1, 1), (1, 2), (2, 1), (2, 2), (3, 3)))
                hits[2]["rrf_score"] = s3s4[0]
                hits[2]["score"] = s3s4[0]
                hits[3]["rrf_score"] = s3s4[1]
                hits[3]["score"] = s3s4[1]
                self.assertIn(adaptive_rag.decide_adaptive_k(hits)["final_k"], (3, 5))

    def test_short_pool_decision(self):
        # Pool 3: margin zero; decision driven by agreement only.
        hits3 = _hits(rrf=(0.0327, 0.0325, 0.0323), pairs=((1, 1), (1, 2), (2, 1)))
        d3 = adaptive_rag.decide_adaptive_k(hits3)
        self.assertEqual(d3["pool_size"], 3)
        self.assertEqual(d3["rrf_score_rank4"], None)
        self.assertEqual(d3["boundary_margin_normalized"], 0.0)
        self.assertEqual(d3["final_k"], 3)

    def test_decision_reproducible(self):
        hits = _hits()
        first = adaptive_rag.decide_adaptive_k(hits)
        second = adaptive_rag.decide_adaptive_k([dict(h) for h in hits])
        self.assertEqual(first, second)

    def test_empty_retrieved_documents_rejected(self):
        with self.assertRaises(ValueError):
            adaptive_rag.decide_adaptive_k([])

    def test_missing_provenance_rejected(self):
        hit = _hit(1, 0.0327, 1, 1)
        del hit["dense_rank"]
        with self.assertRaises(ValueError):
            adaptive_rag.decide_adaptive_k([hit])
