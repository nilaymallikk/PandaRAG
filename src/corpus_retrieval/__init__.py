"""Corpus-level retrieval for the PandaRAG project (Scope B).

Named ``corpus_retrieval`` rather than ``retrieval`` on purpose: ``src/retrieval.py``
is a frozen Scope-A module (imported by the legacy pipelines and their tests), and a
package called ``src/retrieval/`` would shadow it. See deviation D13 in
``docs/pandarag/phase1_report.md``.

Layers:

* :mod:`.corpus`         — pooled candidate corpus construction + fingerprints
* :mod:`.dense`          — exact FAISS dense index over a corpus
* :mod:`.bm25`           — corpus-level BM25
* :mod:`.fusion`         — RRF fusion (shares the legacy scoring function)
* :mod:`.index_manager`  — build/load/verify persisted indexes
* :mod:`.benchmark`      — Recall@K / all-gold-coverage sanity metrics
* :mod:`.schemas`        — typed records
"""

from src.corpus_retrieval.schemas import CorpusArtifact, FusedHit, IndexManifest, RankedList

__all__ = ["CorpusArtifact", "FusedHit", "IndexManifest", "RankedList"]
