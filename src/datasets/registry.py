"""Dataset registry for the PandaRAG project (Scope B).

One place that knows which datasets exist, which split each one uses, and how to
load the semantic-type sidecars produced by ``src/classifier.py`` v1.3.

Datasets and splits are isolated by construction: an adapter is keyed by
``(name, split)`` and its documents are hashed with both values, so a corpus
built for one dataset can never be silently reused for another.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from src import config
from src.datasets.base import DatasetError, DatasetFingerprint, Document, QAExample
from src.datasets.hotpotqa import HotpotQAAdapter
from src.datasets.musique import MuSiQueAdapter
from src.datasets.twowiki import TwoWikiAdapter

_ADAPTERS = {
    ("hotpotqa", "validation"): HotpotQAAdapter,
    ("2wiki", "dev"): TwoWikiAdapter,
    ("musique", "validation"): MuSiQueAdapter,
}

DEFAULT_DATASETS = ("hotpotqa", "2wiki", "musique")


def available() -> list[tuple[str, str]]:
    """Registered ``(dataset, split)`` pairs, in a stable order."""
    return list(_ADAPTERS)


def get_adapter(dataset: str, split: str | None = None):
    """Return the adapter for ``dataset``; ``split`` is validated, not guessed."""
    candidates = [key for key in _ADAPTERS if key[0] == dataset]
    if not candidates:
        raise DatasetError(f"unknown dataset {dataset!r}; known: {sorted({k[0] for k in _ADAPTERS})}")
    if split is None:
        if len(candidates) != 1:
            raise DatasetError(f"{dataset!r} has several splits: {[c[1] for c in candidates]}")
        key = candidates[0]
    else:
        key = (dataset, split)
        if key not in _ADAPTERS:
            raise DatasetError(
                f"unknown split {split!r} for {dataset!r}; known: {[c[1] for c in candidates]}"
            )
    return _ADAPTERS[key]()


def load_all(split: str | None = None) -> list[tuple[str, str, list[QAExample]]]:
    """Load every registered dataset (optionally filtered to one split name)."""
    loaded = []
    for dataset, dataset_split in available():
        if split is not None and dataset_split != split:
            continue
        adapter = get_adapter(dataset, dataset_split)
        loaded.append((dataset, dataset_split, adapter.load()))
    return loaded


def fingerprints(datasets: Iterable[str] = DEFAULT_DATASETS) -> list[DatasetFingerprint]:
    return [get_adapter(name).fingerprint() for name in datasets]


def documents(dataset: str, split: str | None = None) -> list[Document]:
    return get_adapter(dataset, split).documents()


# ---------------------------------------------------------------------------
# Semantic-type sidecars (produced by src/classifier.py v1.3)
# ---------------------------------------------------------------------------


def semantic_types_path(dataset: str):
    return config.DATA_DIR / f"semantic_types_{dataset}.json"


def load_semantic_types(
    dataset: str,
    *,
    examples: Iterable[QAExample] | None = None,
    require_full_coverage: bool = True,
) -> dict[str, Any]:
    """Load ``data/semantic_types_<dataset>.json`` and check id coverage.

    The sidecar is dataset-level metadata: it is *not* part of the runtime record
    and must never be confused with gold evidence. Coverage is verified against
    the loaded examples so a stale sidecar cannot silently mislabel a run.

    Values are the frozen classifier's labels — plain strings in v1.3, but a
    mapping is accepted so a future sidecar may carry a confidence and a reason
    without changing every caller.
    """
    path = semantic_types_path(dataset)
    if not path.exists():
        raise DatasetError(f"semantic-type sidecar not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    types = payload.get("types")
    if not isinstance(types, dict) or not types:
        raise DatasetError(f"{path.name}: missing 'types' mapping")

    if examples is not None:
        ids = {example.example_id for example in examples}
        missing = sorted(ids - set(types))
        extra = sorted(set(types) - ids)
        if require_full_coverage and (missing or extra):
            raise DatasetError(
                f"{path.name}: coverage mismatch (missing {len(missing)}, extra {len(extra)}); "
                f"first missing={missing[:3]}, first extra={extra[:3]}"
            )
    return dict(types)


def semantic_label(value: Any) -> str:
    """Normalise one sidecar value to its label."""
    if isinstance(value, Mapping):
        return str(value.get("type", value.get("label", "unknown")))
    return str(value)


def semantic_type_histogram(dataset: str) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for value in load_semantic_types(dataset).values():
        label = semantic_label(value)
        histogram[label] = histogram.get(label, 0) + 1
    return dict(sorted(histogram.items()))
