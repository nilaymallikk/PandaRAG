"""Structured LLM calls with validation, one repair retry and full accounting.

Every pipeline component that needs JSON (planner, evidence evaluator, compressor)
goes through :func:`call_json`. Parsing is deliberately forgiving about fences and
surrounding prose, and strict about the payload: a call that cannot be parsed and
validated is returned with ``error`` set, never silently coerced.

Accounting: each call returns its own token/cost figures and accumulates into a
:class:`Usage` object, so a pipeline record can separate generation cost from
auxiliary cost (classifier, planner, evaluator, compressor) as required.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from src.llm import DeepSeekClient, LLMResponse

_JSON_BLOCK = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class Usage:
    """Accumulated LLM usage for one question (or one run)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    live_calls: int = 0
    latency_seconds: float = 0.0
    by_component: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, response: LLMResponse, component: str, latency: float) -> None:
        self.prompt_tokens += int(response.prompt_tokens)
        self.completion_tokens += int(response.completion_tokens)
        self.cached_input_tokens += int(response.cached_input_tokens)
        self.reasoning_tokens += int(getattr(response, "reasoning_tokens", 0) or 0)
        self.cost_usd += float(response.cost_usd)
        self.llm_calls += 1
        if not response.from_cache:
            self.live_calls += 1
        self.latency_seconds += latency
        bucket = self.by_component.setdefault(
            component,
            {"prompt_tokens": 0.0, "completion_tokens": 0.0, "cost_usd": 0.0, "llm_calls": 0.0},
        )
        bucket["prompt_tokens"] += int(response.prompt_tokens)
        bucket["completion_tokens"] += int(response.completion_tokens)
        bucket["cost_usd"] += float(response.cost_usd)
        bucket["llm_calls"] += 1

    def merge(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cost_usd += other.cost_usd
        self.llm_calls += other.llm_calls
        self.live_calls += other.live_calls
        self.latency_seconds += other.latency_seconds
        for component, values in other.by_component.items():
            bucket = self.by_component.setdefault(
                component,
                {"prompt_tokens": 0.0, "completion_tokens": 0.0, "cost_usd": 0.0, "llm_calls": 0.0},
            )
            for key, value in values.items():
                bucket[key] += value

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.prompt_tokens,
            "output_tokens": self.completion_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "api_cost_usd": round(self.cost_usd, 10),
            "llm_calls": self.llm_calls,
            "live_llm_calls": self.live_calls,
            "llm_latency_seconds": round(self.latency_seconds, 3),
            "usage_by_component": {
                component: {key: round(value, 8) for key, value in values.items()}
                for component, values in sorted(self.by_component.items())
            },
        }


@dataclass
class JSONResult:
    data: dict[str, Any] | None
    raw_text: str
    error: str | None
    attempts: int
    usage: Usage

    @property
    def ok(self) -> bool:
        return self.data is not None and self.error is None


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from a model response."""
    if not text:
        raise ValueError("empty response")
    candidates: list[str] = []
    for match in _FENCE.finditer(text):
        candidates.append(match.group(1).strip())
    candidates.append(text.strip())
    block = _JSON_BLOCK.search(text)
    if block:
        candidates.append(block.group(1).strip())

    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as error:
            last_error = error
    raise ValueError(f"no parseable JSON in response ({last_error})")


def call_json(
    client: DeepSeekClient,
    *,
    prompt: str,
    system: str,
    validator: Callable[[Any], Mapping[str, Any]],
    component: str,
    usage: Usage | None = None,
    max_tokens: int | None = None,
    retries: int = 1,
) -> JSONResult:
    """Call the LLM, parse JSON, validate it, retry once on failure."""
    usage = usage if usage is not None else Usage()
    attempts = 0
    last_raw = ""
    last_error = "unknown error"
    prompt_text = prompt

    for attempt in range(retries + 1):
        attempts += 1
        started = time.perf_counter()
        try:
            response = client.generate(
                prompt_text,
                system=system,
                max_tokens=max_tokens if max_tokens is not None else None,
            )
        except Exception as error:  # network/API failure: report, do not hide
            last_error = f"llm_error: {type(error).__name__}: {error}"
            usage.latency_seconds += time.perf_counter() - started
            break
        latency = time.perf_counter() - started
        usage.add(response, component, latency)
        last_raw = response.text or ""

        try:
            payload = extract_json(last_raw)
            validated = validator(payload)
            return JSONResult(data=dict(validated), raw_text=last_raw, error=None, attempts=attempts, usage=usage)
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            # One repair attempt: restate the format requirement, keep the request.
            prompt_text = (
                prompt
                + "\n\nYour previous reply could not be used because: "
                + last_error
                + "\nReturn ONLY valid JSON matching the required schema, no prose, no code fences."
            )

    return JSONResult(data=None, raw_text=last_raw, error=last_error, attempts=attempts, usage=usage)


def as_float(value: Any, default: float = 0.0, *, lo: float | None = None, hi: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if lo is not None:
        number = max(lo, number)
    if hi is not None:
        number = min(hi, number)
    return number
