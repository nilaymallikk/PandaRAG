"""Loading and validation of the fixed HotpotQA dataset.

``data/hotpotqa_500.json`` is immutable experiment input: this module only ever
reads it.

File layout (a dict-ified variant of the official HotpotQA dev-distractor
records -- note that ``context`` and ``supporting_facts`` are dicts here, not
the raw list-of-pairs encoding)::

    [
      {
        "id": "5a85b2d95542997b5ce40028",
        "question": "...",
        "answer": "...",
        "type": "bridge" | "comparison",
        "level": "hard",
        "supporting_facts": {"title": ["...", "..."], "sent_id": [0, 0]},
        "context": {"title": ["...", "..."], "sentences": [["...", "..."], ...]}
      },
      ...
    ]

Run ``python -m src.dataset`` from the repository root to load, validate and
summarise the dataset.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src import config

REQUIRED_FIELDS = (
    "id",
    "question",
    "answer",
    "type",
    "level",
    "supporting_facts",
    "context",
)


@dataclass(frozen=True)
class HotpotExample:
    """One HotpotQA question with its distractor paragraphs and gold facts."""

    example_id: str
    question: str
    answer: str
    question_type: str
    level: str
    context_titles: tuple[str, ...]
    context_sentences: tuple[tuple[str, ...], ...]
    supporting_facts: tuple[tuple[str, int], ...]

    @classmethod
    def from_dict(cls, record: dict) -> "HotpotExample":
        supporting = record["supporting_facts"]
        context = record["context"]
        return cls(
            example_id=record["id"],
            question=record["question"],
            answer=record["answer"],
            question_type=record["type"],
            level=record["level"],
            context_titles=tuple(context["title"]),
            context_sentences=tuple(tuple(s) for s in context["sentences"]),
            supporting_facts=tuple(zip(supporting["title"], supporting["sent_id"])),
        )

    @property
    def supporting_titles(self) -> tuple[str, ...]:
        """Distinct paragraph titles that contain a gold supporting fact."""
        return tuple(dict.fromkeys(title for title, _ in self.supporting_facts))

    @property
    def supporting_fact_ids(self) -> tuple[str, ...]:
        """Gold supporting facts as ``title::sent_id`` identifiers."""
        return tuple(sentence_id(title, sent_id) for title, sent_id in self.supporting_facts)

    def paragraph(self, title: str) -> tuple[str, ...]:
        """Return the sentences of the paragraph with ``title``."""
        index = self.context_titles.index(title)
        return self.context_sentences[index]

    def paragraph_text(self, title: str) -> str:
        """Return the paragraph with ``title`` as a single whitespace-joined string."""
        return " ".join(sentence.strip() for sentence in self.paragraph(title))


def sentence_id(title: str, sent_id: int) -> str:
    """Stable identifier for a single context sentence (used by retrieval eval)."""
    return f"{title}::{sent_id}"


def iter_sentences(example: HotpotExample):
    """Yield ``(sentence_id, title, sent_id, text)`` for every context sentence."""
    for title, sentences in zip(example.context_titles, example.context_sentences):
        for sent_id, text in enumerate(sentences):
            yield sentence_id(title, sent_id), title, sent_id, text


def dataset_fingerprint(path: Path | None = None) -> dict[str, object]:
    """Return size/hash/mtime of the dataset file for reproducibility records."""
    dataset_path = Path(path or config.DATASET_FILE)
    payload = dataset_path.read_bytes()
    stat = dataset_path.stat()
    return {
        "path": str(dataset_path),
        "bytes": stat.st_size,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_records(path: Path | None = None) -> list[dict]:
    dataset_path = Path(path or config.DATASET_FILE)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"Dataset must be a JSON list of questions, got {type(records).__name__}.")
    return records


def load_dataset(path: Path | None = None, *, check: bool = True) -> list[HotpotExample]:
    """Load the fixed dataset, validating it first unless ``check=False``."""
    records = _read_records(path)
    if check:
        report = validate_records(records)
        if not report["ok"]:
            raise ValueError("Dataset validation failed: " + " | ".join(report["errors"][:5]))
    return [HotpotExample.from_dict(record) for record in records]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _as_list(value) -> list:
    return value if isinstance(value, list) else []


def validate_records(
    records: list[dict], expected_size: int | None = None
) -> dict[str, object]:
    """Structurally validate raw dataset records and collect basic statistics.

    Returns a report with ``ok``, ``errors``, ``warnings`` and ``stats`` keys.
    The dataset is never modified.
    """
    errors: list[str] = []
    warnings: list[str] = []
    size = config.EXPECTED_DATASET_SIZE if expected_size is None else expected_size

    if not isinstance(records, list):
        return {"ok": False, "errors": ["dataset is not a list"], "warnings": [], "stats": {}}

    if len(records) != size:
        errors.append(f"expected {size} questions, found {len(records)}")

    field_names: Counter = Counter()
    types: Counter = Counter()
    levels: Counter = Counter()
    support_pair_counts: Counter = Counter()
    support_title_counts: Counter = Counter()
    paragraph_counts: Counter = Counter()
    ids: list[str] = []
    questions: list[str] = []
    total_paragraphs = 0
    total_sentences = 0

    for position, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"record {position} is not an object")
            continue

        field_names.update(record.keys())
        missing = [name for name in REQUIRED_FIELDS if name not in record]
        if missing:
            errors.append(f"record {position}: missing fields {missing}")
            continue
        extra = sorted(set(record) - set(REQUIRED_FIELDS))
        if extra:
            warnings.append(f"record {position}: unexpected fields {extra}")

        ids.append(str(record["id"]))
        questions.append(str(record["question"]))
        types[str(record["type"])] += 1
        levels[str(record["level"])] += 1

        if not str(record["question"]).strip():
            errors.append(f"record {position}: empty question")
        if not str(record["answer"]).strip():
            errors.append(f"record {position}: empty answer")

        context = record["context"]
        if not isinstance(context, dict) or not {"title", "sentences"} <= set(context):
            errors.append(f"record {position}: context must be an object with title/sentences")
            continue

        titles = _as_list(context["title"])
        sentences = _as_list(context["sentences"])
        if len(titles) != len(sentences):
            errors.append(
                f"record {position}: {len(titles)} context titles vs {len(sentences)} paragraphs"
            )
            continue
        if len(titles) != len(set(titles)):
            errors.append(f"record {position}: duplicate context titles")
        if not all(isinstance(title, str) and title.strip() for title in titles):
            errors.append(f"record {position}: context titles must be non-empty strings")
        if not all(isinstance(paragraph, list) and paragraph for paragraph in sentences):
            errors.append(f"record {position}: every context paragraph must be a non-empty list")
        if not all(
            isinstance(sentence, str)
            for paragraph in sentences
            if isinstance(paragraph, list)
            for sentence in paragraph
        ):
            errors.append(f"record {position}: context sentences must be strings")

        paragraph_counts[len(titles)] += 1
        total_paragraphs += len(titles)
        total_sentences += sum(len(p) for p in sentences if isinstance(p, list))

        supporting = record["supporting_facts"]
        if not isinstance(supporting, dict) or not {"title", "sent_id"} <= set(supporting):
            errors.append(
                f"record {position}: supporting_facts must be an object with title/sent_id"
            )
            continue

        fact_titles = _as_list(supporting["title"])
        fact_sent_ids = _as_list(supporting["sent_id"])
        if len(fact_titles) != len(fact_sent_ids):
            errors.append(f"record {position}: supporting title/sent_id length mismatch")
            continue
        if not fact_titles:
            errors.append(f"record {position}: no supporting facts")

        pairs = list(zip(fact_titles, fact_sent_ids))
        if len(set(pairs)) != len(pairs):
            warnings.append(f"record {position}: duplicate supporting (title, sent_id) pairs")
        support_pair_counts[len(pairs)] += 1
        support_title_counts[len(set(fact_titles))] += 1

        for title, sent_id in pairs:
            if title not in titles:
                errors.append(f"record {position}: supporting title {title!r} not in context")
                continue
            paragraph = sentences[titles.index(title)]
            if not isinstance(sent_id, int) or not 0 <= sent_id < len(paragraph):
                errors.append(f"record {position}: sent_id {sent_id!r} out of range for {title!r}")

    if len(set(ids)) != len(ids):
        errors.append("duplicate question ids")
    if len(set(questions)) != len(questions):
        warnings.append("duplicate question texts")

    stats = {
        "n_questions": len(records),
        "n_unique_ids": len(set(ids)),
        "n_unique_questions": len(set(questions)),
        "question_types": dict(sorted(types.items())),
        "levels": dict(sorted(levels.items())),
        "supporting_facts_per_question": dict(sorted(support_pair_counts.items())),
        "supporting_titles_per_question": dict(sorted(support_title_counts.items())),
        "context_paragraphs_per_question": dict(sorted(paragraph_counts.items())),
        "mean_context_paragraphs": round(total_paragraphs / len(records), 3) if records else 0.0,
        "mean_sentences_per_paragraph": round(total_sentences / total_paragraphs, 3)
        if total_paragraphs
        else 0.0,
        "mean_sentences_per_question": round(total_sentences / len(records), 3) if records else 0.0,
        "fields": sorted(field_names),
        "fingerprint": dataset_fingerprint(),
    }
    return {"ok": not errors, "errors": errors, "warnings": warnings, "stats": stats}


def main() -> int:
    """Validate and summarise the fixed dataset."""
    report = validate_records(_read_records())
    stats = report["stats"]
    print("HotpotQA dataset validation")
    print(f"  file            {stats['fingerprint']['path']}")
    print(f"  sha256          {stats['fingerprint']['sha256']}")
    print(f"  size (bytes)    {stats['fingerprint']['bytes']}")
    print(f"  questions       {stats['n_questions']}")
    print(f"  fields          {', '.join(stats['fields'])}")
    print(f"  question types  {stats['question_types']}")
    print(f"  levels          {stats['levels']}")
    print(f"  support pairs   {stats['supporting_facts_per_question']}")
    print(f"  support titles  {stats['supporting_titles_per_question']}")
    print(f"  paragraphs/q    {stats['context_paragraphs_per_question']}")
    print(f"  mean sent/para  {stats['mean_sentences_per_paragraph']}")
    print(f"  mean sent/q     {stats['mean_sentences_per_question']}")
    for warning in report["warnings"]:
        print(f"  warning         {warning}")
    if report["errors"]:
        print("  ERRORS:")
        for error in report["errors"][:20]:
            print(f"    - {error}")
        print(f"dataset INVALID ({len(report['errors'])} errors)")
        return 1
    print("dataset OK (immutable input, unmodified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
