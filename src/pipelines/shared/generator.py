"""The single frozen answer-generation surface shared by every Scope-B pipeline.

All of P1c, P2c, P3c, P4, P5 and P6 call :func:`generate_answer`. The only
permitted variation is the context block itself, plus one documented affordance
for PandaRAG: when evidence requirements are provably uncovered, the prompt may
carry an explicit insufficiency instruction (see ``PANDARAG_INSUFFICIENCY_NOTE``).

Why one function: "same generator for every pipeline" is only true if it is
enforced in one place. The prompt fingerprint is recorded with every run so a
silent drift is visible.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Sequence

from src import config
from src.llm import DeepSeekClient, LLMResponse
from src.pipelines.shared.llm_json import Usage

VERSION = "generation/1.1"
PANDARAG_INSUFFICIENCY_NOTE = (
    "If the context does not contain the information needed to answer, reply exactly: "
    "insufficient evidence"
)


@dataclass
class ContextBlock:
    """Ordered context handed to the generator, plus its provenance."""

    documents: list[dict]  # {doc_id, title, text}
    max_chars: int | None = None

    @property
    def text(self) -> str:
        if not self.documents:
            return "Context:\n(none)"
        blocks = []
        for position, document in enumerate(self.documents, start=1):
            text = document["text"]
            if self.max_chars is not None:
                text = text[: self.max_chars]
            blocks.append(f"[Document {position}]\nTitle: {document['title']}\n{text}")
        return "Context:\n" + "\n\n".join(blocks)

    @property
    def tokens_estimate(self) -> int:
        return estimate_tokens(self.text)

    def source_ids(self) -> list[str]:
        return [document["doc_id"] for document in self.documents]


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate (~4 characters per token).

    Used for the 4,000-token context budget before generation, where the exact
    tokenizer is unavailable locally. The generator's real input token count is
    recorded from the API response and is the number reported in results.
    """
    return max(1, (len(text) + 3) // 4)


def build_prompt(question: str, context: ContextBlock, *, insufficiency_note: bool = False) -> str:
    tail = (
        "Answer the question using the supplied context.\n"
        "Reply with only the answer itself: a short phrase, no explanation, no restatement.\n"
    )
    if insufficiency_note:
        tail += PANDARAG_INSUFFICIENCY_NOTE + ".\n"
    return f"Question: {question.strip()}\n\n{context.text}\n\n{tail}Answer:"


def prompt_fingerprint() -> str:
    payload = "|".join(
        (
            VERSION,
            config.GENERATION_SYSTEM_PROMPT,
            "Answer the question using the supplied context.",
            "Reply with only the answer itself: a short phrase, no explanation, no restatement.",
            PANDARAG_INSUFFICIENCY_NOTE,
            config.DEEPSEEK_MODEL,
            str(config.TEMPERATURE),
        )
    )
    return hashlib.sha1(payload.encode()).hexdigest()


@dataclass
class GenerationResult:
    prediction: str
    raw_text: str
    response: LLMResponse
    latency_seconds: float
    prompt: str
    prompt_tokens_estimate: int

    @property
    def error(self) -> str | None:
        return self.response.error


def generate_answer(
    client: DeepSeekClient,
    question: str,
    context: ContextBlock,
    *,
    usage: Usage,
    insufficiency_note: bool = False,
    max_tokens: int | None = None,
) -> GenerationResult:
    prompt = build_prompt(question, context, insufficiency_note=insufficiency_note)
    started = time.perf_counter()
    response = client.generate(
        prompt,
        system=config.GENERATION_SYSTEM_PROMPT,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - started
    usage.add(response, "generation", latency)
    prediction = clean_prediction(response.text)
    return GenerationResult(
        prediction=prediction,
        raw_text=response.text or "",
        response=response,
        latency_seconds=latency,
        prompt=prompt,
        prompt_tokens_estimate=estimate_tokens(prompt),
    )


def clean_prediction(text: str | None) -> str:
    """First non-empty line, stripped of common answer prefixes."""
    if not text:
        return ""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return ""
    answer = lines[0]
    lowered = answer.lower()
    for prefix in ("answer:", "final answer:", "a:"):
        if lowered.startswith(prefix):
            answer = answer[len(prefix) :].strip()
            break
    return answer.strip().strip('"').strip()
