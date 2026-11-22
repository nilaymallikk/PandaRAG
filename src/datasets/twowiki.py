"""2WikiMultiHopQA adapter (Scope B).

Source: official 2WikiMultiHopQA v1.1 ``dev.json`` (Alab-NII/2wikimultihop),
filtered to exactly 2 distinct supporting paragraphs with resolvable gold
evidence, then seeded-sampled to 500 (see ``src/download_datasets.py`` and
``data/dataset_manifest.json``).

Evidence is sentence-level (``sent_id`` is a real sentence index).
"""

from __future__ import annotations

from src.datasets.common import JsonFileAdapter


class TwoWikiAdapter(JsonFileAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="2wiki",
            manifest_key="2wiki",
            filename="2wiki_500.json",
            split="dev",
            evidence_kind="sentence",
            source="Alab-NII/2wikimultihop v1.1 dev.json, 2-hop + resolvable-gold filtered",
        )
