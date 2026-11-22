"""Full SAC-RAG context compression (P5).

The paper describes a "unified LLM context compression module … custom prompt" that
"extracts evidential snippets". Neither the prompt nor the output format is
published, and the compressor model is stated inconsistently (§VI-C names
``qwen-flash`` while the abstract implies one model). This implementation is
therefore an **implementation choice**, recorded as such:

* extractive snippets (verbatim spans where possible), one list per document, so
  source attribution survives compression;
* the compressor sees only the question and the retrieved passages — never the
  answer, the gold evidence, or any score;
* failure falls back to the uncompressed context and is recorded, because a
  compressor crash must not lose the question.

Compression is measured, not assumed: original/compressed token estimates and the
ratio are stored per question so P4-vs-P5 is an empirical comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.llm import DeepSeekClient
from src.pipelines.shared.generator import estimate_tokens
from src.pipelines.shared.llm_json import Usage, call_json
from src.pipelines.shared.retriever import RetrievedDoc

PROMPT_VERSION = "sac-rag-compress/1.0"
MAX_OUTPUT_TOKENS = 900
DOC_CHAR_LIMIT = 1200

SYSTEM_PROMPT = (
    "You compress retrieved passages for a question-answering system. "
    "You keep evidence, you never add knowledge. Return JSON only."
)

USER_PROMPT = """Compress the passages for the question. Keep only what the question needs.

Rules:
- Prefer verbatim spans from the passage; do not paraphrase facts, names, dates or numbers.
- Keep entities, relationships, bridge facts and the chain needed to connect them.
- Drop boilerplate, repeated statements and unrelated sentences.
- Never add information that is not in the passages. Never answer the question.
- One entry per passage that contains anything useful. Omit passages with nothing useful.

Question: {question}

Passages:
{documents}

Return JSON exactly like: {{"snippets": [{{"doc_id": "<id>", "text": "<verbatim span>"}}]}}"""


@dataclass
class CompressionResult:
    documents: list[dict[str, Any]]
    original_tokens: int
    compressed_tokens: int
    ratio: float
    fallback: bool
    error: str | None
    doc_ids_retained: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_version": PROMPT_VERSION,
            "original_tokens_estimate": self.original_tokens,
            "compressed_tokens_estimate": self.compressed_tokens,
            "compression_ratio": round(self.ratio, 4),
            "fallback": self.fallback,
            "error": self.error,
            "doc_ids_retained": list(self.doc_ids_retained),
            "n_documents": len(self.documents),
        }


def _render_documents(documents: Sequence[RetrievedDoc]) -> str:
    blocks = []
    for document in documents:
        blocks.append(
            f"[{document.doc_id}] {document.title}\n{document.text[:DOC_CHAR_LIMIT]}"
        )
    return "\n\n".join(blocks)


def compress(
    client: DeepSeekClient,
    question: str,
    documents: Sequence[RetrievedDoc],
    *,
    usage: Usage,
) -> CompressionResult:
    original_tokens = estimate_tokens(
        "\n\n".join(f"{document.title}\n{document.text}" for document in documents)
    )
    if not documents:
        return CompressionResult([], 0, 0, 0.0, fallback=True, error="no documents to compress")

    known = {document.doc_id: document for document in documents}

    def _validate(payload: Any) -> Mapping[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
        snippets = payload.get("snippets")
        if not isinstance(snippets, list) or not snippets:
            raise ValueError("'snippets' must be a non-empty list")
        cleaned = []
        for item in snippets:
            if not isinstance(item, dict):
                raise ValueError("each snippet must be an object")
            doc_id = str(item.get("doc_id", "")).strip()
            text = str(item.get("text", "")).strip()
            if doc_id not in known:
                raise ValueError(f"unknown doc_id {doc_id!r}")
            if not text:
                raise ValueError(f"empty snippet for {doc_id!r}")
            cleaned.append({"doc_id": doc_id, "title": known[doc_id].title, "text": text})
        return {"snippets": cleaned}

    result = call_json(
        client,
        prompt=USER_PROMPT.format(
            question=question.strip(), documents=_render_documents(documents)
        ),
        system=SYSTEM_PROMPT,
        validator=_validate,
        component="compression",
        usage=usage,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    if not result.ok or not result.data:
        fallback_docs = [
            {"doc_id": document.doc_id, "title": document.title, "text": document.text}
            for document in documents
        ]
        return CompressionResult(
            documents=fallback_docs,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            ratio=1.0,
            fallback=True,
            error=result.error,
            doc_ids_retained=[document.doc_id for document in documents],
        )

    compressed = list(result.data["snippets"])
    compressed_tokens = estimate_tokens(
        "\n\n".join(f"{item['title']}\n{item['text']}" for item in compressed)
    )
    ratio = (compressed_tokens / original_tokens) if original_tokens else 1.0
    return CompressionResult(
        documents=compressed,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        ratio=ratio,
        fallback=False,
        error=None,
        doc_ids_retained=[item["doc_id"] for item in compressed],
    )
