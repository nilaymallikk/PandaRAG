"""Shared JSON-file dataset adapter.

All three datasets used by this project are stored in the same dict-ified shape
(``context`` / ``supporting_facts`` as parallel-list mappings), so the mechanics
live here once and each dataset module only declares what actually differs:
its file, split, evidence granularity and provenance note.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import config
from src.datasets.base import (
    DatasetError,
    DatasetFingerprint,
    Document,
    EvidenceKind,
    QAExample,
    build_examples,
    summarise,
)

MANIFEST_FILE = config.DATA_DIR / "dataset_manifest.json"


@dataclass(frozen=True)
class JsonFileAdapter:
    """Load, validate and fingerprint one dataset file."""

    name: str
    manifest_key: str
    filename: str
    split: str
    evidence_kind: EvidenceKind
    source: str

    # -- file access --------------------------------------------------------
    def path(self) -> Path:
        return config.DATA_DIR / self.filename

    def records(self) -> list[dict[str, Any]]:
        path = self.path()
        if not path.exists():
            raise DatasetError(f"{self.name}: dataset file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise DatasetError(f"{self.name}: expected a JSON list, got {type(payload).__name__}")
        return payload

    # -- loading ------------------------------------------------------------
    def load(self) -> list[QAExample]:
        return build_examples(
            self.records(),
            dataset=self.name,
            split=self.split,
            evidence_kind=self.evidence_kind,
        )

    def documents(self) -> list[Document]:
        """Every paragraph shipped with every question, deduplicated by content."""
        unique: dict[str, Document] = {}
        for example in self.load():
            for document in example.documents():
                unique.setdefault(document.doc_id, document)
        return list(unique.values())

    # -- provenance ---------------------------------------------------------
    def manifest_entry(self) -> dict[str, Any]:
        if not MANIFEST_FILE.exists():
            return {}
        payload = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        return dict(payload.get("datasets", {}).get(self.manifest_key, {}))

    def fingerprint(self) -> DatasetFingerprint:
        """Identity of the dataset file, verified against the committed manifest."""
        path = self.path()
        payload = path.read_bytes()
        actual_sha = hashlib.sha256(payload).hexdigest()
        entry = self.manifest_entry()
        expected_sha = entry.get("sha256")

        source = self.source
        if expected_sha and expected_sha != actual_sha:
            raise DatasetError(
                f"{self.name}: file fingerprint does not match dataset_manifest.json "
                f"(expected {expected_sha[:12]}…, got {actual_sha[:12]}…). "
                "The dataset is immutable experiment input; refusing to continue."
            )
        if expected_sha:
            source = f"{source} (sha256 verified against data/dataset_manifest.json)"
        else:
            source = f"{source} (manifest entry missing; sha256 computed locally)"

        records = self.records()
        expected_n = entry.get("n")
        if isinstance(expected_n, int) and expected_n != len(records):
            raise DatasetError(
                f"{self.name}: expected {expected_n} questions per manifest, found {len(records)}"
            )

        return DatasetFingerprint(
            dataset=self.name,
            split=self.split,
            path=str(path.relative_to(config.REPO_ROOT)),
            sha256=actual_sha,
            bytes=len(payload),
            n_questions=len(records),
            evidence_kind=self.evidence_kind,
            source=source,
        )

    def summary(self) -> dict[str, Any]:
        """Fingerprint + statistics, suitable for a JSON manifest."""
        return {"fingerprint": self.fingerprint().as_dict(), "statistics": summarise(self.load())}
