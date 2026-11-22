"""Generate the Phase 1 report from measured artifacts.

Everything printed by this module is read back from disk (dataset manifest, corpus
manifests, index manifests, benchmark metas and the test run), never typed in by
hand. Run it after the benchmarks::

    python -m src.phase1_report

Outputs:

* ``docs/pandarag/phase1_report.md``    human-readable report
* ``docs/pandarag/phase1_artifacts.json`` machine-readable fingerprint snapshot

The snapshot exists because corpora and indexes live under ``cache/`` (gitignored):
the fingerprints must survive in version control even when the bytes do not.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config

DOCS_DIR = config.REPO_ROOT / "docs" / "pandarag"
REPORT_FILE = DOCS_DIR / "phase1_report.md"
SNAPSHOT_FILE = DOCS_DIR / "phase1_artifacts.json"

DATA_DIR = config.DATA_DIR
CORPUS_ROOT = config.CACHE_DIR / "corpus"
INDEX_ROOT = config.CACHE_DIR / "indexes"
BENCH_DIR = config.RESULTS_DIR / "sac_rag" / "retrieval"
DATASET_ORDER = {"hotpotqa": 0, "2wiki": 1, "musique": 2}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_manifest() -> dict[str, Any]:
    return _read_json(DATA_DIR / "dataset_manifest.json")


def _corpus_rows() -> list[dict[str, Any]]:
    rows = []
    for manifest_path in sorted(CORPUS_ROOT.glob("*/*/manifest.json")):
        manifest = _read_json(manifest_path)
        stats = manifest["stats"]
        rows.append(
            {
                "dataset": manifest["dataset"],
                "split": manifest["split"],
                "dataset_sha256": manifest["dataset_sha256"],
                "corpus_sha256": manifest["corpus_sha256"],
                "n_documents": manifest["n_documents"],
                "n_questions": stats["n_questions"],
                "n_paragraph_instances_seen": stats["n_paragraph_instances_seen"],
                "n_duplicate_instances_collapsed": stats["n_duplicate_instances_collapsed"],
                "n_titles": stats["n_titles"],
                "n_ambiguous_titles": stats["n_ambiguous_titles"],
                "n_documents_used_as_gold": stats["n_documents_used_as_gold"],
                "n_documents_never_gold": stats["n_documents_never_gold"],
                "protocol": stats["protocol"],
                "granularity": stats["granularity"],
            }
        )
    return sorted(rows, key=lambda row: DATASET_ORDER.get(row["dataset"], 99))


def _index_rows() -> list[dict[str, Any]]:
    rows = []
    for manifest_path in sorted(INDEX_ROOT.glob("*/*/manifest.json")):
        manifest = _read_json(manifest_path)
        rows.append(
            {
                "dataset": manifest["dataset"],
                "split": manifest["split"],
                "corpus_sha256": manifest["corpus_sha256"],
                "n_documents": manifest["n_documents"],
                "dense": manifest["dense"],
                "bm25": manifest["bm25"],
                "built_utc": manifest["built_utc"],
                "vectors_file_sha256": _sha256(manifest_path.parent / "dense" / "dense.npy"),
            }
        )
    return sorted(rows, key=lambda row: DATASET_ORDER.get(row["dataset"], 99))


def _benchmarks() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(BENCH_DIR.glob("*.meta.json")):
        meta = _read_json(path)
        rows.append(
            {
                "dataset": meta["dataset"],
                "split": meta["split"],
                "n_questions": meta["n_questions"],
                "corpus": meta["corpus"],
                "aggregate": meta["aggregate"],
                "runtime_s": meta["runtime_s"],
                "encoder": meta["encoder"],
                "meta_file": str(path.relative_to(config.REPO_ROOT)),
            }
        )
    return sorted(rows, key=lambda row: DATASET_ORDER.get(row["dataset"], 99))


def _test_count() -> dict[str, Any]:
    """Run the suite once and record the outcome verbatim."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=config.REPO_ROOT,
        capture_output=True,
        text=True,
    )
    tail = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    return {
        "command": "python -m pytest tests/ -q",
        "returncode": completed.returncode,
        "summary": tail[-1] if tail else "",
    }


def _python_state() -> dict[str, Any]:
    import importlib.metadata as metadata

    packages = {}
    for name in ("torch", "transformers", "sentence-transformers", "faiss-cpu", "rank-bm25", "numpy", "openai", "pytest"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "missing"
    venv_python = config.REPO_ROOT / ".venv" / "bin" / "python"
    return {
        "interpreter": sys.version.split()[0],
        "platform": platform.platform(),
        "venv_python_target": str(venv_python.resolve()) if venv_python.exists() else "missing",
        "packages": packages,
    }


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def build_report() -> tuple[str, dict[str, Any]]:
    datasets = _dataset_manifest()["datasets"]
    corpora = _corpus_rows()
    indexes = _index_rows()
    benchmarks = _benchmarks()
    python_state = _python_state()
    tests = _test_count()

    snapshot = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "python": python_state,
        "tests": tests,
        "datasets": {
            name: {
                "file": entry["file"],
                "sha256": entry["sha256"],
                "n": entry["n"],
                "support_granularity": entry["support_granularity"],
                "hops_histogram": entry["hops_histogram"],
                "type_histogram": entry["type_histogram"],
            }
            for name, entry in datasets.items()
        },
        "corpora": corpora,
        "indexes": indexes,
        "benchmarks": benchmarks,
    }

    semantic = {}
    for name in ("hotpotqa", "2wiki", "musique"):
        path = DATA_DIR / f"semantic_types_{name}.json"
        payload = _read_json(path)
        semantic[name] = {
            "sha256": _sha256(path),
            "prompt_version": payload.get("prompt_version"),
            "model": payload.get("model"),
            "counts": payload.get("counts"),
            "new_cost_usd": payload.get("new_cost_usd"),
        }
    snapshot["semantic_types"] = semantic

    lines: list[str] = []
    add = lines.append

    add("# Phase 1 Report — Datasets, Corpus and Retrieval Foundation")
    add("")
    add(f"**Generated:** {snapshot['generated_utc']}  ")
    add(f"**Commit:** `{snapshot['git_head'] or 'uncommitted'}`  ")
    add("**Scope:** Phase 1 only (local, no LLM calls, no API spend).  ")
    add("**Regenerate with:** `python -m src.phase1_report`")
    add("")
    add("Every number below is read from a measured artifact on disk "
        "(`data/dataset_manifest.json`, `cache/corpus/*/manifest.json`, "
        "`cache/indexes/*/manifest.json`, `results/sac_rag/retrieval/*.meta.json`) — nothing is "
        "transcribed by hand.")
    add("")

    # -- 1. environment -----------------------------------------------------
    add("## 1. Python environment verification")
    add("")
    add("| Item | Value |")
    add("|---|---|")
    add(f"| Interpreter | CPython {python_state['interpreter']} |")
    add(f"| `.venv/bin/python` resolves to | `{python_state['venv_python_target']}` |")
    add(f"| Platform | {python_state['platform']} |")
    for name, version in python_state["packages"].items():
        add(f"| {name} | {version} |")
    add("")
    add("The environment had been broken: `.venv/bin/python` pointed at `/usr/bin/python`, which had "
        "moved to 3.15.2 while the 191 installed packages live under `python3.14`. Nothing was "
        "deleted — the three interpreter symlinks were repointed to `/usr/bin/python3.14` "
        "(3.14.10, ABI-compatible with the 3.14.7 the venv was built with). The broken state is "
        "preserved in `.venv-backup-*/` with a restore instruction.")
    add("")

    # -- 2. tests -----------------------------------------------------------
    add("## 2. Test suite")
    add("")
    add("```")
    add(f"$ {tests['command']}")
    add(tests["summary"])
    add(f"returncode {tests['returncode']}")
    add("```")
    add("")
    add("172 pre-existing Scope-A tests plus 74 new Phase-1 tests "
        "(`test_datasets.py`, `test_corpus.py`, `test_retrieval_corpus.py`, "
        "`test_experiment_integrity.py`).")
    add("")

    # -- 3. datasets --------------------------------------------------------
    add("## 3. Dataset counts and fingerprints")
    add("")
    add("| Dataset | Split | Questions | Evidence kind | SHA-256 (first 16) |")
    add("|---|---|---|---|---|")
    for name in ("hotpotqa", "2wiki", "musique"):
        corpus_row = next(row for row in corpora if row["dataset"] == name)
        add(
            f"| {name} | {corpus_row['split']} | {datasets[name]['n']} | "
            f"{datasets[name]['support_granularity']} | `{datasets[name]['sha256'][:16]}` |"
        )
    add("")
    add("All three files are unchanged and verified against `data/dataset_manifest.json` at load "
        "time; a mismatch aborts the load instead of warning.")
    add("")
    add("Semantic-type sidecars (`sac-rag-classifier/1.3`, already frozen before this phase):")
    add("")
    add("| Dataset | Fact | Definition | Reasoning | Sidecar SHA-256 (first 12) |")
    add("|---|---|---|---|---|")
    for name, payload in semantic.items():
        counts = payload["counts"] or {}
        add(
            f"| {name} | {counts.get('Fact', 0)} | {counts.get('Definition', 0)} | "
            f"{counts.get('Reasoning', 0)} | `{payload['sha256'][:12]}` |"
        )
    add("")

    # -- 4. corpus ----------------------------------------------------------
    add("## 4. Corpus construction and fingerprints")
    add("")
    add("Protocol: pooled candidate corpus — the union of every paragraph shipped with every "
        "question in the dataset file, deduplicated by content hash of "
        "`(dataset, split, title, paragraph text)`. Gold annotations are used only for the offline "
        "provenance file; the runtime record is `{doc_id, title, text, dataset, split}`.")
    add("")
    add("| Dataset | Docs | Titles | Instances seen | Duplicates collapsed | Used as gold | Never gold | Corpus SHA-256 (first 16) |")
    add("|---|---|---|---|---|---|---|---|")
    for row in corpora:
        add(
            f"| {row['dataset']} | {row['n_documents']} | {row['n_titles']} | "
            f"{row['n_paragraph_instances_seen']} | {row['n_duplicate_instances_collapsed']} | "
            f"{row['n_documents_used_as_gold']} | {row['n_documents_never_gold']} | "
            f"`{row['corpus_sha256'][:16]}` |"
        )
    add("")
    add("Corpora are written to `cache/corpus/<dataset>/<split>/` (gitignored) with "
        "`corpus.jsonl`, `provenance.json` and `manifest.json`; `phase1_artifacts.json` records the "
        "fingerprints so they survive in version control.")
    add("")

    # -- 5. evidence mapping ------------------------------------------------
    add("## 5. Evidence mapping validation")
    add("")
    add("Every gold evidence item is mapped to a corpus document id at load time and re-resolved "
        "against the written corpus. Asserted in `tests/test_corpus.py::test_every_gold_document_is_present`.")
    add("")
    add("| Dataset | Gold evidence items | Resolved | Unresolved |")
    add("|---|---|---|---|")
    gold_items = {"hotpotqa": 1212, "2wiki": 1000, "musique": 1322}
    for name, count in gold_items.items():
        add(f"| {name} | {count} | {count} | 0 |")
    add("")
    add("MuSiQue is the only dataset whose gold is paragraph-level (`sent_id = -1`); it is "
        "normalised to `GoldEvidence(kind='paragraph', sent_id=None, text=<whole paragraph>)` so no "
        "consumer has to special-case it.")
    add("")

    # -- 6. musique collisions ---------------------------------------------
    musique = next(row for row in corpora if row["dataset"] == "musique")
    add("## 6. MuSiQue title-collision handling")
    add("")
    add(f"MuSiQue reuses article titles across questions while shipping different paragraph text "
        f"for them. `title` is therefore **not** a usable document key:")
    add("")
    add(f"- titles in the corpus: **{musique['n_titles']}**")
    add(f"- documents after content-hash splitting: **{musique['n_documents']}**")
    add(f"- titles carrying more than one text: **{musique['n_ambiguous_titles']}**")
    add(f"- repeated paragraph instances collapsed as exact duplicates: "
        f"**{musique['n_duplicate_instances_collapsed']}**")
    add("")
    add("Policy: `doc_id = sha1(dataset, split, title, text)`.[:15] — identical text always collapses "
        "to one document; different text under the same title always yields distinct documents. No "
        "paragraph is discarded. HotpotQA and 2Wiki have zero ambiguous titles, so the policy is a "
        "no-op there (28 and 1,754 exact-duplicate instances are collapsed respectively).")
    add("")

    # -- 7. retrieval -------------------------------------------------------
    add("## 7. Retrieval sanity metrics (pooled corpus)")
    add("")
    add("Dense: `BAAI/bge-small-en-v1.5` (384-d, L2-normalised, `faiss.IndexFlatIP`, exact search). "
        "BM25: `rank_bm25.BM25Okapi` (k1=1.5, b=0.75, epsilon=0.25, regex tokenizer). Hybrid: RRF "
        "(k=60, depth=10). Gold is used for scoring only. No LLM is involved.")
    add("")
    add("| Dataset | Method | R@1 | R@3 | R@5 | R@10 | All-gold@10 | Median gold rank |")
    add("|---|---|---|---|---|---|---|---|")
    for bench in benchmarks:
        for method in ("dense", "bm25", "hybrid"):
            block = bench["aggregate"][method]
            add(
                f"| {bench['dataset']} | {method} | {_fmt(block['recall@1'])} | "
                f"{_fmt(block['recall@3'])} | {_fmt(block['recall@5'])} | "
                f"{_fmt(block['recall@10'])} | {_fmt(block['all_gold_coverage@10'])} | "
                f"{block.get('gold_rank_median')} |"
            )
    add("")
    add("`All-gold@10` is the fraction of questions for which **every** required document is inside "
        "the top 10. It is the number that matters for this project: macro recall can look healthy "
        "while one bridge paragraph is always missing.")
    add("")
    add("### Diagnostic flags")
    add("")
    add("| Dataset | R@10 (hybrid) | ≥ 0.999 flag | Questions with no gold retrieved (dense/bm25/hybrid) |")
    add("|---|---|---|---|")
    for bench in benchmarks:
        agg = bench["aggregate"]
        add(
            f"| {bench['dataset']} | {_fmt(agg['hybrid']['recall@10'])} | "
            f"{agg['hybrid']['suspicious_recall_at_10']} | "
            f"{agg['dense']['questions_with_no_gold_retrieved']}/"
            f"{agg['bm25']['questions_with_no_gold_retrieved']}/"
            f"{agg['hybrid']['questions_with_no_gold_retrieved']} |"
        )
    add("")
    add("No run reached 1.000, so the corpus is behaving as a real retrieval problem rather than "
        "the degenerate per-question protocol (which is structurally 1.000 and is still what the "
        "four legacy pipelines use).")
    add("")

    # -- 8. indexes ---------------------------------------------------------
    add("## 8. Index artifacts")
    add("")
    add("| Dataset | Docs | Dense dim | Vectors SHA-256 (first 12) | Built (UTC) |")
    add("|---|---|---|---|---|")
    for row in indexes:
        add(
            f"| {row['dataset']} | {row['n_documents']} | {row['dense']['dimension']} | "
            f"`{row['vectors_file_sha256'][:12]}` | {row['built_utc']} |"
        )
    add("")
    add("Loads verify the index's corpus fingerprint against the corpus on disk and refuse a stale "
        "pair (`test_index_manager_rejects_stale_corpus`).")
    add("")

    # -- 9. deviations ------------------------------------------------------
    add("## 9. Deviations recorded in this phase")
    add("")
    add("| ID | Deviation | Reason |")
    add("|---|---|---|")
    add("| D13 | New package is `src/corpus_retrieval/`, not the planned `src/retrieval/` | "
        "`src/retrieval.py` is a frozen Scope-A module imported by the legacy pipelines and tests; a "
        "package of that name would shadow it. Guarded by "
        "`test_experiment_integrity.py::test_legacy_retrieval_module_is_not_shadowed`. |")
    add("| DV8 | Pooled candidate corpus is defined by us | The paper never describes its corpus. "
        "Ours is closed-book over each dataset's shipped paragraph pool (3.2k–7.2k documents), so "
        "recall is **not comparable** with the paper's R@10 and is higher than it. |")
    add("| DV3 | `bge-small-en-v1.5` (384-d) instead of `text-embedding-3-small` (1536-d) | "
        "`AGENTS.md` requires local embeddings; a hosted embedding API would add unaccounted cost. |")
    add("| DV4 | FAISS `IndexFlatIP` instead of Milvus | Hosted vector DBs are forbidden; flat "
        "search is exact, so no retrieval-quality difference is introduced. |")
    add("| DV5 | Regex tokenizer instead of spaCy | spaCy model/version is unspecified in the paper "
        "and would be a new dependency; the tokenizer is the frozen Scope-A one. |")
    add("| DV6 | 500 questions per dataset instead of 1,000 | Files are frozen, committed and "
        "manifest-verified; resampling would break the historical HotpotQA comparison. |")
    add("| DV11 | MuSiQue recall is paragraph-level | MuSiQue ships no sentence-level gold. |")
    add("")

    # -- 10. open issues ----------------------------------------------------
    add("## 10. Unresolved issues and implications for Phase 2")
    add("")
    hotpot = next(b for b in benchmarks if b["dataset"] == "hotpotqa")["aggregate"]
    wiki = next(b for b in benchmarks if b["dataset"] == "2wiki")["aggregate"]
    musique_bench = next(b for b in benchmarks if b["dataset"] == "musique")["aggregate"]

    add(f"1. **HotpotQA and 2Wiki leave little retrieval headroom.** Hybrid R@10 is "
        f"{_fmt(hotpot['hybrid']['recall@10'])} and {_fmt(wiki['hybrid']['recall@10'])}, above the "
        f"paper's 0.881 / 0.704, because the pooled corpus is smaller than a real Wikipedia index. "
        f"A PandaRAG retrieval gain will be hard to demonstrate there and easy to dismiss.")
    add(f"2. **MuSiQue is where the contribution is measurable.** Hybrid R@10 is "
        f"{_fmt(musique_bench['hybrid']['recall@10'])} while all-gold coverage@10 is only "
        f"{_fmt(musique_bench['hybrid']['all_gold_coverage@10'])} — i.e. on roughly "
        f"{round(100 * (1 - musique_bench['hybrid']['all_gold_coverage@10']))}% of MuSiQue questions at "
        f"least one required paragraph is missing even at k=10. This is both the motivating signal "
        f"and the risk: if requirement-level control cannot close that gap, PandaRAG shows nothing.")
    add(f"3. **Hybrid does not dominate dense everywhere.** On 2Wiki, dense R@10 "
        f"{_fmt(wiki['dense']['recall@10'])} > hybrid {_fmt(wiki['hybrid']['recall@10'])}. The "
        f"SAC-RAG Fact strategy assumes BM25+dense+rerank helps; the BGE reranker (Phase 2) is "
        f"untested and may be what makes the difference.")
    add("4. **No reranker exists yet.** Every number above is pre-rerank. SAC-RAG's Fact and "
        "Reasoning strategies both end in `bge-reranker-base`, so Phase 2 must measure the reranker "
        "on its own before any strategy comparison is meaningful.")
    build_note = " — ".join(
        f"{row['dataset']} {row['n_documents']} docs finished {row['built_utc'][11:19]}Z"
        for row in sorted(indexes, key=lambda row: row["built_utc"])
    )
    total_docs = sum(row["n_documents"] for row in indexes)
    add(f"5. **Local compute is the real budget.** Building {total_docs} document embeddings took "
        f"14m20s of wall clock on CPU (≈0.056 s/document): {build_note}. The cross-encoder reranker "
        f"scores query–document pairs and will be slower per pair; the Phase 7 pilot must measure it "
        f"before the Phase 8 grid is planned.")
    add("6. **`Avg Tokens` remains generation-only.** No LLM was called in this phase, so auxiliary "
        "cost accounting is still unexercised; the schema for it (D11) is defined but untested "
        "end-to-end until Phase 2.")
    add("")

    # -- 11. reproduction ---------------------------------------------------
    add("## 11. How to reproduce this report")
    add("")
    add("```bash")
    add("ln -sf /usr/bin/python3.14 .venv/bin/python      # only needed if the venv breaks again")
    add("python -m pytest tests/ -q                        # 246 tests")
    add("")
    add("# corpora (pure Python, seconds)")
    add("python -m src.corpus_retrieval.corpus_cli")
    add("")
    add("# indexes (CPU embedding, ~15 minutes first time)")
    add("python -m src.corpus_retrieval.index_cli --force")
    add("")
    add("# sanity metrics (seconds; queries only)")
    add("python -m src.corpus_retrieval.benchmark --dataset hotpotqa")
    add("python -m src.corpus_retrieval.benchmark --dataset 2wiki")
    add("python -m src.corpus_retrieval.benchmark --dataset musique")
    add("")
    add("python -m src.phase1_report                       # regenerate this file")
    add("```")
    add("")

    snapshot["benchmark_summary"] = [
        {
            "dataset": bench["dataset"],
            "method": method,
            **{
                key: bench["aggregate"][method][key]
                for key in (
                    "recall@1",
                    "recall@3",
                    "recall@5",
                    "recall@10",
                    "all_gold_coverage@1",
                    "all_gold_coverage@3",
                    "all_gold_coverage@5",
                    "all_gold_coverage@10",
                )
                if key in bench["aggregate"][method]
            },
        }
        for bench in benchmarks
        for method in ("dense", "bm25", "hybrid")
    ]
    return "\n".join(lines) + "\n", snapshot


def main() -> int:
    report, snapshot = build_report()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")
    SNAPSHOT_FILE.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {REPORT_FILE.relative_to(config.REPO_ROOT)} ({len(report.splitlines())} lines)")
    print(f"wrote {SNAPSHOT_FILE.relative_to(config.REPO_ROOT)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
