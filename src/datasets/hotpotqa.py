"""HotpotQA adapter (Scope B).

Wraps the frozen, legacy HotpotQA-500 file. The legacy validator
(:mod:`src.dataset`) is still applied as an extra guard: this adapter adds the
canonical :class:`~src.datasets.base.QAExample` view and content-derived document
ids, but it does not relax or replace the original schema checks.

Source: HuggingFace ``hotpotqa/hotpot_qa``, config ``distractor``, split
``validation``; the committed file is a fixed, hard-only, 2-hop sample.
"""

from __future__ import annotations

from src import dataset as legacy_dataset
from src.datasets.base import DatasetError
from src.datasets.common import JsonFileAdapter


class HotpotQAAdapter(JsonFileAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="hotpotqa",
            manifest_key="hotpotqa",
            filename="hotpotqa_500.json",
            split="validation",
            evidence_kind="sentence",
            source="hotpotqa/hotpot_qa distractor/validation, hard-only 2-hop sample",
        )

    def load(self):
        # Legacy guard: the frozen experiment's own validator must also pass.
        report = legacy_dataset.validate_records(self.records())
        if not report["ok"]:
            raise DatasetError(
                "hotpotqa: legacy validator rejected the file: "
                + " | ".join(report["errors"][:5])
            )
        return super().load()
