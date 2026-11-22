"""MuSiQue adapter (Scope B).

Source: HuggingFace ``dgslibisey/MuSiQue``, config ``default``, split
``validation``, answerable-only, seeded-sampled to 500.

Evidence is **paragraph-level**: MuSiQue marks ``is_supporting`` per paragraph and
encodes that as ``sent_id = -1``. The adapter therefore normalises gold evidence
to ``kind="paragraph"`` instead of inventing sentence indices, and duplicate
paragraph titles inside one question are disambiguated at download time
(``" (n)"`` suffix) — across questions, identical titles may carry different
paragraph text, which is why corpus document ids are content-derived.
"""

from __future__ import annotations

from src.datasets.common import JsonFileAdapter


class MuSiQueAdapter(JsonFileAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="musique",
            manifest_key="musique",
            filename="musique_500.json",
            split="validation",
            evidence_kind="paragraph",
            source="dgslibisey/MuSiQue validation, answerable-only sample",
        )
