"""Tests for the Fact/Definition/Reasoning classifier (offline, no API calls).

Covers the label parser, the keyword sanity heuristic, and the frozen sidecar
files: every question in all three datasets must have exactly one label.
"""

from __future__ import annotations

import json
import unittest

from src import classifier, config, dataset


class ParseLabelTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(classifier.parse_label('{"type": "Fact"}'), "Fact")
        self.assertEqual(classifier.parse_label('{"type": "reasoning"}'), "Reasoning")

    def test_code_fence_and_surrounding_text(self):
        self.assertEqual(classifier.parse_label('```json\n{"type": "Definition"}\n```'), "Definition")
        self.assertEqual(classifier.parse_label('The answer is {"type": "Reasoning"}'), "Reasoning")

    def test_unparseable_is_none(self):
        self.assertIsNone(classifier.parse_label(""))
        self.assertIsNone(classifier.parse_label("I cannot classify this."))


class KeywordGuessTests(unittest.TestCase):
    def test_heuristic_matches_paper_keyword_lists(self):
        self.assertEqual(classifier.keyword_guess("Why did the Roman Empire fall?"), "Reasoning")
        self.assertEqual(classifier.keyword_guess("How many people live there?"), "Fact")
        self.assertEqual(classifier.keyword_guess("What is quantum entanglement?"), "Definition")
        self.assertEqual(classifier.keyword_guess("Which battle occurred first?"), "Fact")


class SidecarFilesTests(unittest.TestCase):
    """The frozen classifier output must cover every question exactly once."""

    def test_all_datasets_fully_labelled(self):
        for name in classifier.DATASET_FILES:
            examples = dataset.load_dataset(config.REPO_ROOT / classifier.DATASET_FILES[name])
            types = classifier.load_semantic_types(name)
            self.assertEqual(len(types), len(examples), name)
            self.assertEqual({e.example_id for e in examples}, set(types), name)
            self.assertTrue(set(types.values()) <= set(classifier.LABELS), name)

    def test_sidecar_records_provenance(self):
        for name in classifier.DATASET_FILES:
            payload = json.loads(
                (config.DATA_DIR / f"semantic_types_{name}.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["prompt_version"], classifier.PROMPT_VERSION)
            self.assertEqual(payload["decoding"]["temperature"], 0.0)
            self.assertEqual(payload["dataset_sha256"],
                             dataset.dataset_fingerprint(config.REPO_ROOT / classifier.DATASET_FILES[name])["sha256"])


if __name__ == "__main__":
    unittest.main()
