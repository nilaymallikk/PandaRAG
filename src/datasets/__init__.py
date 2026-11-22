"""Canonical dataset layer for Scope B (PandaRAG).

Import from here rather than from the individual modules::

    from src.datasets import get_adapter, QAExample, RuntimeExample

The split between :class:`QAExample` (gold-bearing, evaluation-side) and
:class:`RuntimeExample` (gold-free, runtime-side) is a load-bearing invariant of
this project, not a convenience.
"""

from src.datasets.base import (
    DatasetError,
    DatasetFingerprint,
    Document,
    EvidenceKind,
    GoldEvidence,
    QAExample,
    RuntimeExample,
    build_examples,
    content_doc_id,
    summarise,
)
from src.datasets.registry import (
    DEFAULT_DATASETS,
    available,
    documents,
    fingerprints,
    get_adapter,
    load_all,
    load_semantic_types,
    semantic_type_histogram,
    semantic_types_path,
)

__all__ = [
    "DEFAULT_DATASETS",
    "DatasetError",
    "DatasetFingerprint",
    "Document",
    "EvidenceKind",
    "GoldEvidence",
    "QAExample",
    "RuntimeExample",
    "available",
    "build_examples",
    "content_doc_id",
    "documents",
    "fingerprints",
    "get_adapter",
    "load_all",
    "load_semantic_types",
    "semantic_type_histogram",
    "semantic_types_path",
    "summarise",
]
