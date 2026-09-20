"""Guard tests for the final paper-readiness analysis (offline, read-only)."""

from __future__ import annotations

import unittest

from src import final_analysis as fa
from src import final_report as fr


class FinalAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = fr.build_report()

    def test_no_failed_verification(self):
        self.assertNotIn("FAIL", self.report)
        self.assertIn("| Check |", self.report)

    def test_error_taxonomy_partitions_all_questions(self):
        results = fa.load_results()
        retrieval = fa.load_retrieval()
        ids = sorted(results["adaptive_rag"])
        tax = fa.taxonomy(results, retrieval, ids)
        for pipeline, counter in tax.items():
            total = sum(counter.values())
            self.assertEqual(total, 500, f"{pipeline} taxonomy covers {total} questions")
            self.assertEqual(counter["correct"], sum(1 for i in ids if results[pipeline][i]["exact_match"] == 1))

    def test_adaptive_truncation_only_on_k3(self):
        results = fa.load_results()
        retrieval = fa.load_retrieval()
        ids = sorted(results["adaptive_rag"])
        truncated = [i for i in ids if fa.classify("adaptive_rag", i, results["adaptive_rag"], retrieval) == "context truncation"]
        self.assertTrue(truncated)
        self.assertTrue(all(results["adaptive_rag"][i]["final_k"] == 3 for i in truncated))

    def test_headline_numbers_match_meta(self):
        results = fa.load_results()
        metas = fa.load_metas()
        for p in fa.PIPELINES:
            em = fa.mean([r["exact_match"] for r in results[p].values()])
            self.assertAlmostEqual(em, metas[p]["exact_match"], places=6)


if __name__ == "__main__":
    unittest.main()
