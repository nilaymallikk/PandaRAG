"""Guard test for the read-only cross-pipeline analysis.

Asserts the report the script generates is consistent with the raw result
files (which are read, never written). Offline: no API calls.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from src import cross_pipeline_analysis as xpa


class CrossPipelineAnalysisTests(unittest.TestCase):
    def test_report_aggregates_match_raw_results(self):
        report = xpa.build_report()
        for name in xpa.PIPELINES:
            recs = [json.loads(line) for line in (Path("results") / f"{name}.jsonl").open() if line.strip()]
            em = sum(r["exact_match"] for r in recs) / len(recs)
            row = re.search(rf"\| {re.escape(xpa.LABELS[name])} \| ([\d.]+) \| ([\d.]+) \|", report)
            self.assertIsNotNone(row, f"missing aggregate row for {name}")
            self.assertAlmostEqual(float(row.group(1)), em, places=4)

    def test_integrity_checks_pass(self):
        report = xpa.build_report()
        checks = report.split("## 8. Provenance and integrity checks")[1].split("## 9.")[0]
        self.assertNotIn("FAIL", checks)
        self.assertGreaterEqual(checks.count("PASS"), 6)

    def test_no_rag_retrieval_is_na_not_zero(self):
        report = xpa.build_report()
        self.assertIn("| No-RAG | none | 0.0000 | N/A | N/A | N/A |", report)

    def test_all_pipelines_cover_same_questions(self):
        by_id = {n: xpa.load_records(n) for n in xpa.PIPELINES}
        ids = set(by_id[xpa.PIPELINES[0]])
        self.assertEqual(len(ids), 500)
        for name in xpa.PIPELINES:
            self.assertEqual(set(by_id[name]), ids)


if __name__ == "__main__":
    unittest.main()
