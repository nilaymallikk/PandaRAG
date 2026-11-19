"""Freeze the three multi-hop QA datasets as 500-question JSON files.

Outputs (committed once, then immutable):

    data/hotpotqa_500.json         pre-existing frozen file -- verified, never rebuilt
    data/2wiki_500.json            official 2WikiMultiHopQA v1.1 dev.json, seeded sample
    data/musique_500.json          HF dgslibisey/MuSiQue validation, seeded sample
    data/dataset_manifest.json     provenance: sources, hashes, selection rule, stats

Record schema (matches the frozen HotpotQA file, plus ``hops``)::

    {
      "id": "...", "question": "...", "answer": "...",
      "type": "bridge" | "compositional" | "multihop" | ...,
      "level": "hard" | "2hop" | "3hop" | "4hop",
      "hops": 2,
      "supporting_facts": {"title": [...], "sent_id": [...]},
      "context": {"title": [...], "sentences": [[...], ...]}
    }

What is NOT downloaded here:

* The paper's ``Fact / Definition / Reasoning`` taxonomy is **classifier output** --
  it does not exist in any of the three datasets. It is produced by the LLM
  classifier stage and cached in ``data/semantic_types_<dataset>.json``
  (qid -> type), never merged into these frozen files.
* ``hops`` is derived, not a native field: 2Wiki = distinct supporting paragraph
  titles (filtered to 2-hop questions to match the base paper's Table 1);
  MuSiQue = number of ``question_decomposition`` steps (2-4);
  HotpotQA = 2 for every question in the frozen file (derived at load time
  because the frozen file must not be modified).

Usage::

    python -m src.download_datasets                      # all three, 500 each
    python -m src.download_datasets --datasets 2wiki musique
    python -m src.download_datasets --limit 20           # smoke run
    python -m src.download_datasets --force-download     # re-fetch 2Wiki zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src import config

# ---------------------------------------------------------------------------
# Constants (frozen provenance)
# ---------------------------------------------------------------------------

SEED = 42
DEFAULT_LIMIT = 500

# Pre-existing frozen HotpotQA file: verified by hash, never rebuilt by this script.
HOTPOTQA_FILE = config.DATA_DIR / "hotpotqa_500.json"
HOTPOTQA_EXPECTED_SHA256 = "956405ceb43d73385d60898e1bfc7f45a8184c9a4e1eff45b9a8ec239acfd048"
HOTPOTQA_SOURCE = {
    "origin": "HuggingFace hotpotqa/hotpot_qa, config=distractor, split=validation",
    "selection": "pre-existing frozen file (all 500 level=hard, all 2-hop); initial commit 0e50e12",
}

# 2WikiMultiHopQA v1.1 (fixed sentence segmentation), official distribution.
TWO_WIKI_ZIP_URL = "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1"
TWO_WIKI_MEMBER = "dev.json"
TWO_WIKI_FILE = config.DATA_DIR / "2wiki_500.json"
TWO_WIKI_SOURCE = {
    "origin": "official 2WikiMultiHopQA v1.1 (Alab-NII/2wikimultihop), dev.json",
    "url": TWO_WIKI_ZIP_URL,
    "filter": ("exactly 2 distinct supporting paragraphs (paper Table 1: 2Wiki hops = 2) "
               "AND resolvable gold evidence (unique context titles, sent_id in range)"),
    "license": "see github.com/Alab-NII/2wikimultihop LICENSE; HF mirror xanhho/2WikiMultihopQA is cc-by-sa-4.0",
}

MUSIQUE_HF_ID = "dgslibisey/MuSiQue"
MUSIQUE_FILE = config.DATA_DIR / "musique_500.json"
MUSIQUE_SOURCE = {
    "origin": f"HuggingFace {MUSIQUE_HF_ID}, config=default, split=validation",
    "filter": "answerable == True (all 2,417 validation questions are answerable)",
    "license": "CC-BY-4.0",
}

MANIFEST_FILE = config.DATA_DIR / "dataset_manifest.json"
DOWNLOAD_DIR = config.CACHE_DIR / "downloads"

# MuSiQue paragraphs are raw text; the repo needs a sentence list. Deterministic
# and dependency-free: split after . ! ? followed by whitespace.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split((text or "").strip()) if s]


def download(url: str, dest: Path, *, force: bool = False) -> Path:
    """Download ``url`` to ``dest`` unless the cached copy exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"  cached: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:  # noqa: S310
        total = int(response.headers.get("content-length") or 0)
        seen = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            seen += len(chunk)
            if total and seen % (25 << 20) < (1 << 20):
                print(f"    {seen / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
    tmp.replace(dest)
    print(f"  saved: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def extract_member(zip_path: Path, member: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_mtime >= zip_path.stat().st_mtime:
        print(f"  cached: {dest}")
        return dest
    print(f"  extracting {member} from {zip_path.name}")
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as src, dest.open("wb") as out:
            while chunk := src.read(1 << 20):
                out.write(chunk)
    return dest


def seeded_sample(records: list[dict], limit: int, seed: int) -> list[dict]:
    """Deterministic sample: seeded shuffle of the full filtered split, first N."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


def write_dataset(path: Path, records: list[dict]) -> str:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return sha256_file(path)


def check_records(records: list[dict], name: str, limit: int) -> None:
    """Fail loudly if a frozen file violates the schema it claims to satisfy."""
    assert len(records) == limit, f"{name}: expected {limit} questions, got {len(records)}"
    for i, r in enumerate(records):
        for field in ("id", "question", "answer", "type", "level", "hops",
                      "supporting_facts", "context"):
            assert field in r, f"{name}[{i}]: missing {field}"
        assert r["question"].strip() and r["answer"].strip(), f"{name}[{i}]: empty q/a"
        assert int(r["hops"]) >= 1, f"{name}[{i}]: bad hops"
        titles = r["context"]["title"]
        assert len(titles) == len(r["context"]["sentences"]), f"{name}[{i}]: context shape"
        assert len(r["supporting_facts"]["title"]) == len(r["supporting_facts"]["sent_id"]), \
            f"{name}[{i}]: support shape"
        assert len(titles) == len(set(titles)), f"{name}[{i}]: duplicate context titles"
        lookup = {t: s for t, s in zip(titles, r["context"]["sentences"])}
        for title, sent in zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"]):
            assert title in lookup, f"{name}[{i}]: gold title {title!r} not in context"
            # sent_id == -1 means paragraph-level support (MuSiQue).
            assert sent == -1 or 0 <= int(sent) < len(lookup[title]), \
                f"{name}[{i}]: sent_id {sent} out of range for {title!r}"


def dataset_stats(records: list[dict]) -> dict:
    return {
        "n": len(records),
        "hops_histogram": dict(sorted(Counter(int(r["hops"]) for r in records).items())),
        "type_histogram": dict(sorted(Counter(str(r["type"]) for r in records).items())),
        "mean_gold_supports": round(
            sum(len(r["supporting_facts"]["title"]) for r in records) / len(records), 3),
        "mean_context_paragraphs": round(
            sum(len(r["context"]["title"]) for r in records) / len(records), 3),
    }


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


def convert_2wiki(raw: dict) -> dict:
    support_titles = [title for title, _ in raw["supporting_facts"]]
    context = raw["context"]  # [[title, [sentence, ...]], ...]
    hops = len(set(support_titles))
    return {
        "id": raw["_id"],
        "question": raw["question"],
        "answer": raw["answer"],
        "type": raw["type"],
        "level": f"{hops}hop",
        "hops": hops,
        "supporting_facts": {
            "title": support_titles,
            "sent_id": [int(sent) for _, sent in raw["supporting_facts"]],
        },
        "context": {
            "title": [title for title, _ in context],
            "sentences": [list(sentences) for _, sentences in context],
        },
    }


def convert_musique(raw: dict) -> dict:
    paragraphs = raw["paragraphs"]
    # MuSiQue's distractor pool repeats article titles (same name, different article,
    # or several chunks of one article). Gold support is paragraph-level, so make
    # titles unique in paragraph order: "Green", "Green (2)", ... Lossless.
    seen: Counter = Counter()
    titles = []
    for paragraph in paragraphs:
        seen[paragraph["title"]] += 1
        suffix = "" if seen[paragraph["title"]] == 1 else f" ({seen[paragraph['title']]})"
        titles.append(paragraph["title"] + suffix)
    support_titles = [t for t, p in zip(titles, paragraphs) if p["is_supporting"]]
    hops = len(raw["question_decomposition"])
    return {
        "id": raw["id"],
        "question": raw["question"],
        "answer": raw["answer"],
        # MuSiQue ships no native type label; the paper's Fact/Definition/Reasoning
        # classification replaces this placeholder downstream.
        "type": "multihop",
        "level": f"{hops}hop",
        "hops": hops,
        "supporting_facts": {
            "title": support_titles,
            # Paragraph-level support only (dataset provides is_supporting, not sent ids).
            "sent_id": [-1] * len(support_titles),
        },
        "context": {
            "title": titles,
            "sentences": [split_sentences(p["paragraph_text"]) for p in paragraphs],
        },
    }


# ---------------------------------------------------------------------------
# Per-dataset builders
# ---------------------------------------------------------------------------


def build_hotpotqa() -> dict:
    """Verify the pre-existing frozen HotpotQA file; never rewrite it."""
    if not HOTPOTQA_FILE.exists():
        sys.exit(f"FATAL: {HOTPOTQA_FILE} is missing. It is frozen experiment input; "
                 "restore it with 'git checkout -- data/hotpotqa_500.json'.")
    sha = sha256_file(HOTPOTQA_FILE)
    if sha != HOTPOTQA_EXPECTED_SHA256:
        sys.exit(f"FATAL: {HOTPOTQA_FILE} hash {sha} != frozen {HOTPOTQA_EXPECTED_SHA256}. "
                 "Stage-1 results (docs/final_analysis.md) depend on this file; do not edit it.")
    records = json.loads(HOTPOTQA_FILE.read_text(encoding="utf-8"))
    # Derive hops on the fly: the frozen file has no hops field.
    derived = [{"hops": len(set(r["supporting_facts"]["title"]))} for r in records]
    stats = dataset_stats([{**r, **d} for r, d in zip(records, derived)])
    print(f"  hotpotqa: verified frozen file, sha256={sha[:12]}… n={len(records)}")
    print(f"    hops={stats['hops_histogram']} types={stats['type_histogram']}")
    return {
        "file": str(HOTPOTQA_FILE.relative_to(config.REPO_ROOT)),
        "sha256": sha,
        "status": "frozen (verified, not modified by this script)",
        "source": HOTPOTQA_SOURCE,
        "support_granularity": "sentence",
        "hops_field": "derived at load time (len of distinct supporting_facts titles)",
        "semantic_type": "see data/semantic_types_hotpotqa.json (produced by src/classifier.py)",
        **stats,
    }


def two_wiki_quality_ok(raw: dict) -> bool:
    """True only when the gold evidence resolves against the context.

    The official dev set contains a few annotation quirks: ~4% of questions
    repeat a context title and ~0.7% carry a sent_id outside the paragraph.
    Those questions are dropped before sampling (selection stays deterministic).
    """
    titles = [t for t, _ in raw["context"]]
    if len(titles) != len(set(titles)):
        return False
    lookup = dict(raw["context"])
    for title, sent in raw["supporting_facts"]:
        if title not in lookup or not (0 <= int(sent) < len(lookup[title])):
            return False
    return True


def build_2wiki(limit: int, seed: int, *, force: bool = False) -> dict:
    zip_path = download(TWO_WIKI_ZIP_URL, DOWNLOAD_DIR / "2wiki_data_ids_april7.zip", force=force)
    dev_path = extract_member(zip_path, TWO_WIKI_MEMBER, DOWNLOAD_DIR / "2wiki_dev.json")
    raw = json.loads(dev_path.read_text(encoding="utf-8"))
    two_hop = [r for r in raw if len(set(t for t, _ in r["supporting_facts"])) == 2]
    clean = [r for r in two_hop if two_wiki_quality_ok(r)]
    sample = [convert_2wiki(r) for r in seeded_sample(clean, limit, seed)]
    check_records(sample, "2wiki", limit)
    sha = write_dataset(TWO_WIKI_FILE, sample)
    stats = dataset_stats(sample)
    print(f"  {TWO_WIKI_FILE.name}: {len(raw)} dev questions -> {len(two_hop)} 2-hop -> "
          f"{len(clean)} resolvable -> {len(sample)} sampled, sha256={sha[:12]}…")
    print(f"    hops={stats['hops_histogram']} types={stats['type_histogram']}")
    return {
        "file": str(TWO_WIKI_FILE.relative_to(config.REPO_ROOT)),
        "sha256": sha,
        "status": "generated",
        "source": {
            **TWO_WIKI_SOURCE,
            "source_questions": len(raw),
            "after_hops_filter": len(two_hop),
            "after_quality_filter": len(clean),
            "dropped_unresolvable_gold": len(two_hop) - len(clean),
            "dev_json_sha256": sha256_file(dev_path),
        },
        "support_granularity": "sentence",
        "semantic_type": "see data/semantic_types_2wiki.json (produced by src/classifier.py)",
        **stats,
    }


def build_musique(limit: int, seed: int) -> dict:
    from datasets import load_dataset  # lazy: only MuSiQue needs the HF client

    ds = load_dataset(MUSIQUE_HF_ID, split="validation")
    raw = [dict(r) for r in ds if r["answerable"]]
    sample = [convert_musique(r) for r in seeded_sample(raw, limit, seed)]
    check_records(sample, "musique", limit)
    sha = write_dataset(MUSIQUE_FILE, sample)
    stats = dataset_stats(sample)
    print(f"  {MUSIQUE_FILE.name}: {len(raw)} answerable validation questions -> "
          f"{len(sample)} sampled, sha256={sha[:12]}…")
    print(f"    hops={stats['hops_histogram']} types={stats['type_histogram']}")
    return {
        "file": str(MUSIQUE_FILE.relative_to(config.REPO_ROOT)),
        "sha256": sha,
        "status": "generated",
        "source": {**MUSIQUE_SOURCE, "source_questions": len(raw)},
        "support_granularity": "paragraph (sent_id = -1; dataset marks is_supporting per paragraph)",
        "title_disambiguation": ("duplicate paragraph titles get ' (n)' suffix in paragraph order "
                                 "(lossless; gold support is paragraph-level)"),
        "sentence_splitter": r"regex (?<=[.!?])\s+",
        "semantic_type": "see data/semantic_types_musique.json (produced by src/classifier.py)",
        **stats,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the three multi-hop QA datasets.")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wiki", "musique"],
                        choices=["hotpotqa", "2wiki", "musique"])
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args(argv)

    print(f"datasets={args.datasets} limit={args.limit} seed={args.seed}")
    datasets: dict[str, dict] = {}
    if "hotpotqa" in args.datasets:
        datasets["hotpotqa"] = build_hotpotqa()
    if "2wiki" in args.datasets:
        datasets["2wiki"] = build_2wiki(args.limit, args.seed, force=args.force_download)
    if "musique" in args.datasets:
        datasets["musique"] = build_musique(args.limit, args.seed)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "python -m src.download_datasets",
        "selection": {
            "rule": "seeded shuffle (random.Random(seed)) over the full split, then first N",
            "seed": args.seed,
            "limit": args.limit,
        },
        "note": ("type = dataset-native label only. Fact/Definition/Reasoning is classifier "
                 "output -> data/semantic_types_<dataset>.json (never merged into these files)."),
        "datasets": datasets,
    }
    existing = {}
    if MANIFEST_FILE.exists():
        existing = json.loads(MANIFEST_FILE.read_text(encoding="utf-8")).get("datasets", {})
    manifest["datasets"] = {**existing, **datasets}  # partial runs keep other entries
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"\nmanifest: {MANIFEST_FILE}")
    print("Summary")
    for name, info in sorted(manifest["datasets"].items()):
        print(f"  {name:9s} n={info['n']:>4}  hops={info['hops_histogram']}  "
              f"sha256={info['sha256'][:12]}…  ({info['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
