"""Canonical dataset layer for the PandaRAG project (Scope B).

Two object kinds, deliberately separated by type:

* :class:`QAExample` — the *evaluation-side* record. Holds the question, the gold
  answer and the gold evidence. This object must never be passed to runtime
  retrieval, planning, support-evaluation or control code.
* :class:`RuntimeExample` — the same question with ``answer`` and every gold
  field physically removed. This is what runtime components receive, so a gold
  leak is a ``TypeError``/``AttributeError`` rather than a silent bug.

Document identifiers are **content-derived**, so they are stable across runs,
questions and corpus rebuilds::

    doc_id = "d" + sha1(f"{dataset}\\x00{split}\\x00{title}\\x00{text}")[:15]

Including the dataset and split in the hash keeps corpora isolated: the same
paragraph appearing in two datasets can never share an id.

Gold evidence granularity differs per dataset (HotpotQA and 2Wiki label
sentences, MuSiQue labels paragraphs via ``sent_id = -1``), so evidence is
normalised through :class:`GoldEvidence.kind` instead of being special-cased in
every consumer.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence

EvidenceKind = Literal["paragraph", "sentence"]

REQUIRED_RECORD_FIELDS = (
    "id",
    "question",
    "answer",
    "type",
    "level",
    "supporting_facts",
    "context",
)

DOC_ID_PREFIX = "d"
DOC_ID_HEX_CHARS = 15


def content_doc_id(dataset: str, split: str, title: str, text: str) -> str:
    """Stable document id derived only from content and corpus identity."""
    payload = "\x00".join((dataset, split, title, text))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return DOC_ID_PREFIX + digest[:DOC_ID_HEX_CHARS]


@dataclass(frozen=True)
class Document:
    """One corpus document (a paragraph, in every dataset we use)."""

    doc_id: str
    title: str
    text: str
    dataset: str
    split: str

    @property
    def text_for_embedding(self) -> str:
        """Paragraph text with its title, matching the legacy retrieval input."""
        return f"{self.title}\n{self.text}"


@dataclass(frozen=True)
class GoldEvidence:
    """One gold evidence item, normalised across datasets.

    ``kind == "sentence"``: ``sent_id`` is the sentence index inside the
    paragraph and ``text`` is that sentence.
    ``kind == "paragraph"``: ``sent_id`` is ``None`` and ``text`` is the whole
    paragraph (MuSiQue labels paragraphs, not sentences).
    """

    kind: EvidenceKind
    doc_id: str
    title: str
    sent_id: int | None
    text: str


@dataclass(frozen=True)
class RuntimeExample:
    """Gold-free view of a question. The only object runtime code may receive."""

    example_id: str
    question: str
    dataset: str
    split: str
    question_type: str | None
    context_titles: tuple[str, ...]
    context_sentences: tuple[tuple[str, ...], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def paragraph_text(self, title: str) -> str:
        index = self.context_titles.index(title)
        return " ".join(sentence.strip() for sentence in self.context_sentences[index])

    def documents(self) -> tuple[Document, ...]:
        return tuple(
            Document(
                doc_id=content_doc_id(self.dataset, self.split, title, text),
                title=title,
                text=text,
                dataset=self.dataset,
                split=self.split,
            )
            for title, text in self.paragraphs()
        )

    def paragraphs(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (title, " ".join(sentence.strip() for sentence in sentences))
            for title, sentences in zip(self.context_titles, self.context_sentences)
        )


@dataclass(frozen=True)
class QAExample:
    """Evaluation-side question record: question + answer + gold evidence."""

    example_id: str
    question: str
    answer: str
    dataset: str
    split: str
    question_type: str | None
    context_titles: tuple[str, ...]
    context_sentences: tuple[tuple[str, ...], ...]
    gold_evidence: tuple[GoldEvidence, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # -- corpus / evidence helpers -----------------------------------------
    def paragraph_text(self, title: str) -> str:
        index = self.context_titles.index(title)
        return " ".join(sentence.strip() for sentence in self.context_sentences[index])

    def paragraphs(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (title, " ".join(sentence.strip() for sentence in sentences))
            for title, sentences in zip(self.context_titles, self.context_sentences)
        )

    def documents(self) -> tuple[Document, ...]:
        return tuple(
            Document(
                doc_id=content_doc_id(self.dataset, self.split, title, text),
                title=title,
                text=text,
                dataset=self.dataset,
                split=self.split,
            )
            for title, text in self.paragraphs()
        )

    @property
    def gold_doc_ids(self) -> tuple[str, ...]:
        """Distinct gold document ids, in first-mention order."""
        return tuple(dict.fromkeys(item.doc_id for item in self.gold_evidence))

    @property
    def gold_sentences(self) -> tuple[GoldEvidence, ...]:
        return tuple(item for item in self.gold_evidence if item.kind == "sentence")

    def hops(self) -> int:
        """Hop count as distinct gold paragraphs (dataset field wins if present)."""
        declared = self.metadata.get("hops")
        if isinstance(declared, int) and declared > 0:
            return declared
        return len(self.gold_doc_ids)

    def runtime(self) -> RuntimeExample:
        """Project to the gold-free runtime view."""
        return RuntimeExample(
            example_id=self.example_id,
            question=self.question,
            dataset=self.dataset,
            split=self.split,
            question_type=self.question_type,
            context_titles=self.context_titles,
            context_sentences=self.context_sentences,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class DatasetFingerprint:
    """Identity of one dataset file plus its corpus-relevant statistics."""

    dataset: str
    split: str
    path: str
    sha256: str
    bytes: int
    n_questions: int
    evidence_kind: EvidenceKind
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "n_questions": self.n_questions,
            "evidence_kind": self.evidence_kind,
            "source": self.source,
        }


class DatasetAdapter(Protocol):
    """What every dataset adapter must provide."""

    name: str
    split: str
    evidence_kind: EvidenceKind
    source: str

    def path(self) -> Any: ...

    def records(self) -> list[dict[str, Any]]: ...

    def load(self) -> list[QAExample]: ...

    def fingerprint(self) -> DatasetFingerprint: ...

    def documents(self) -> list[Document]: ...


class DatasetError(ValueError):
    """Raised when a dataset file does not match the expected schema."""


# ---------------------------------------------------------------------------
# Shared record -> QAExample conversion
# ---------------------------------------------------------------------------


def _as_parallel_lists(payload: Mapping[str, Any], field_name: str, position: int) -> list:
    if not isinstance(payload, Mapping):
        raise DatasetError(f"record {position}: {field_name} must be a mapping")
    titles = payload.get("title")
    if not isinstance(titles, list):
        raise DatasetError(f"record {position}: {field_name}.title must be a list")
    other_key = "sent_id" if field_name == "supporting_facts" else "sentences"
    other = payload.get(other_key)
    if not isinstance(other, list):
        raise DatasetError(f"record {position}: {field_name}.{other_key} must be a list")
    if len(titles) != len(other):
        raise DatasetError(
            f"record {position}: {field_name} has {len(titles)} titles but "
            f"{len(other)} {other_key} entries"
        )
    return [titles, other]


def build_examples(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    split: str,
    evidence_kind: EvidenceKind,
) -> list[QAExample]:
    """Validate raw records and convert them to :class:`QAExample` objects.

    Validation is strict on purpose: a schema drift must fail loudly here rather
    than produce a silently empty gold set (which would make recall undefined).
    """
    if not isinstance(records, list):
        raise DatasetError(f"{dataset}: dataset must be a JSON list, got {type(records).__name__}")

    examples: list[QAExample] = []
    seen_ids: Counter = Counter()

    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise DatasetError(f"record {position} is not an object")
        missing = [name for name in REQUIRED_RECORD_FIELDS if name not in record]
        if missing:
            raise DatasetError(f"record {position}: missing fields {missing}")

        example_id = str(record["id"]).strip()
        if not example_id:
            raise DatasetError(f"record {position}: empty id")
        seen_ids[example_id] += 1

        question = str(record["question"]).strip()
        if not question:
            raise DatasetError(f"record {position}: empty question")

        titles, sentences = _as_parallel_lists(record["context"], "context", position)
        if len(set(titles)) != len(titles):
            raise DatasetError(f"record {position}: duplicate paragraph titles inside context")
        normalised_sentences: list[tuple[str, ...]] = []
        for index, paragraph in enumerate(sentences):
            if not isinstance(paragraph, list) or not paragraph:
                raise DatasetError(f"record {position}: context.sentences[{index}] must be a non-empty list")
            normalised_sentences.append(tuple(str(s) for s in paragraph))

        evidence_titles, evidence_sent_ids = _as_parallel_lists(
            record["supporting_facts"], "supporting_facts", position
        )
        if len(evidence_titles) != len(evidence_sent_ids):
            raise DatasetError(f"record {position}: supporting_facts is not a parallel list pair")

        title_positions = {title: index for index, title in enumerate(titles)}
        seen_evidence: set[tuple[str, int | None]] = set()
        gold: list[GoldEvidence] = []
        for title, sent_id in zip(evidence_titles, evidence_sent_ids):
            title = str(title)
            if title not in title_positions:
                raise DatasetError(
                    f"record {position}: supporting title {title!r} is not in the context"
                )
            paragraph_sentences = normalised_sentences[title_positions[title]]
            if evidence_kind == "sentence":
                if not isinstance(sent_id, int) or isinstance(sent_id, bool):
                    raise DatasetError(f"record {position}: sent_id {sent_id!r} is not an int")
                if not 0 <= sent_id < len(paragraph_sentences):
                    raise DatasetError(
                        f"record {position}: sent_id {sent_id} outside paragraph {title!r}"
                    )
                key: tuple[str, int | None] = (title, sent_id)
                text = paragraph_sentences[sent_id]
            else:
                if sent_id != -1:
                    raise DatasetError(
                        f"record {position}: {dataset} labels whole paragraphs, "
                        f"expected sent_id -1, got {sent_id!r}"
                    )
                key = (title, None)
                text = " ".join(sentence.strip() for sentence in paragraph_sentences)
            if key in seen_evidence:
                continue
            seen_evidence.add(key)
            gold.append(
                GoldEvidence(
                    kind=evidence_kind,
                    doc_id=content_doc_id(
                        dataset,
                        split,
                        title,
                        " ".join(s.strip() for s in paragraph_sentences),
                    ),
                    title=title,
                    sent_id=sent_id if evidence_kind == "sentence" else None,
                    text=str(text),
                )
            )

        if not gold:
            raise DatasetError(f"record {position} ({example_id}): empty gold evidence set")

        metadata: dict[str, Any] = {
            "native_type": record.get("type"),
            "level": record.get("level"),
        }
        if "hops" in record:
            metadata["hops"] = record["hops"]

        examples.append(
            QAExample(
                example_id=example_id,
                question=question,
                answer=str(record["answer"]),
                dataset=dataset,
                split=split,
                question_type=str(record["type"]) if record.get("type") is not None else None,
                context_titles=tuple(str(t) for t in titles),
                context_sentences=tuple(normalised_sentences),
                gold_evidence=tuple(gold),
                metadata=metadata,
            )
        )

    duplicates = [example_id for example_id, count in seen_ids.items() if count > 1]
    if duplicates:
        raise DatasetError(f"{dataset}: duplicate question ids: {duplicates[:5]}")

    return examples


def summarise(examples: Iterable[QAExample]) -> dict[str, Any]:
    """Deterministic statistics for a loaded split (used by the phase report)."""
    examples = list(examples)
    evidence_counts = [len(example.gold_evidence) for example in examples]
    doc_counts = [len(example.gold_doc_ids) for example in examples]
    hops: Counter = Counter()
    native_types: Counter = Counter()
    levels: Counter = Counter()
    for example in examples:
        hops[example.hops()] += 1
        native_types[example.question_type] += 1
        levels[str(example.metadata.get("level"))] += 1
    return {
        "n_questions": len(examples),
        "n_gold_evidence_items": int(sum(evidence_counts)),
        "mean_gold_evidence_per_question": round(sum(evidence_counts) / max(1, len(examples)), 4),
        "mean_gold_documents_per_question": round(sum(doc_counts) / max(1, len(examples)), 4),
        "hops_histogram": {str(key): value for key, value in sorted(hops.items())},
        "native_type_histogram": {str(key): value for key, value in sorted(native_types.items())},
        "level_histogram": {str(key): value for key, value in sorted(levels.items())},
    }
