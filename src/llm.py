"""Shared DeepSeek LLM client used by all four pipelines.

The client wraps the OpenAI-compatible DeepSeek chat completions endpoint and
returns a single :class:`LLMResponse` object per request so that every pipeline
records the same accounting fields (tokens, reasoning tokens, cached input
tokens, latency, LLM calls, cost, error status).

Behaviour rules taken from the project specification:

* the API key is read from the environment via :mod:`src.config` and is never
  logged, printed or stored in a result record;
* failed requests are retried with exponential backoff and always logged to
  ``results/logs/llm_errors.jsonl`` (never silently dropped);
* responses are cached on disk and replayed with ``from_cache=True``;
* thinking-mode output (``reasoning_content``) is kept separate from the final
  answer text, so chains of thought can never leak into ``predicted_answer``.

Run ``python -m src.llm`` from the repository root for a one-request smoke test.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from src import config

logger = logging.getLogger(__name__)

# Sentinel so that "not specified" can be distinguished from an explicit None.
_UNSET: Any = object()

RETRYABLE_EXCEPTIONS = (APIConnectionError, APITimeoutError, RateLimitError)
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


@dataclass
class LLMResponse:
    """Result of one chat completion (or of one replayed cache entry).

    ``cost_usd`` describes the price of the tokens in this response. When
    ``from_cache`` is True no API call was made and the response was replayed
    from disk, so cost aggregations can exclude it from actual spend.
    """

    text: str
    model: str = ""
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    llm_calls: int = 0
    cost_usd: float = 0.0
    finish_reason: str | None = None
    from_cache: bool = False
    error: str | None = None
    request_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def ok(self) -> bool:
        """True when the request completed without an API or transport error."""
        return self.error is None

    @property
    def truncated(self) -> bool:
        """True when the output budget was exhausted (thinking tokens included)."""
        return self.finish_reason == "length"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_cost_usd(
    *,
    model: str,
    prompt_tokens: int,
    cached_input_tokens: int,
    completion_tokens: int,
    timestamp: datetime | None = None,
) -> float:
    """Cost in USD for actual token usage, using the published DeepSeek rates."""
    prices = config.pricing_for(model, timestamp)
    cache_hit = max(0, min(int(cached_input_tokens or 0), int(prompt_tokens or 0)))
    cache_miss = max(0, int(prompt_tokens or 0) - cache_hit)
    output = max(0, int(completion_tokens or 0))
    return (
        cache_hit * prices["input_cache_hit"]
        + cache_miss * prices["input_cache_miss"]
        + output * prices["output"]
    ) / 1_000_000


# ---------------------------------------------------------------------------
# Disk cache and error logging
# ---------------------------------------------------------------------------


def _request_fingerprint(params: Mapping[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return config.LLM_CACHE_DIR / f"{key}.json"


def _cache_read(key: str) -> dict[str, Any] | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Ignoring unreadable LLM cache entry %s: %s", path, error)
        return None


def _cache_write(key: str, payload: dict[str, Any]) -> None:
    config.LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(key)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    tmp_path.replace(path)


def _log_api_error(record: Mapping[str, Any]) -> None:
    """Append a failed request to the JSONL error log (never the API key)."""
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        with config.LLM_ERROR_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as error:  # logging must never break the experiment
        logger.error("Could not write LLM error log: %s", error)


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in RETRYABLE_STATUS_CODES
    return False


def _error_message(error: Exception) -> str:
    """Short, secret-free description of a failed request."""
    message = f"{type(error).__name__}: {error}"
    return message[:500]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _to_response(
    completion: Any, *, latency_s: float, llm_calls: int, request_hash: str
) -> LLMResponse:
    """Convert an SDK completion object into an :class:`LLMResponse`."""
    choice = completion.choices[0]
    message = choice.message
    usage = getattr(completion, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    cached_input_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
    reasoning_tokens = int(getattr(completion_details, "reasoning_tokens", 0) or 0)
    model = str(getattr(completion, "model", "") or "")
    return LLMResponse(
        text=(message.content or "").strip(),
        model=model,
        reasoning=(getattr(message, "reasoning_content", "") or "").strip(),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_input_tokens=cached_input_tokens,
        total_tokens=total_tokens or (prompt_tokens + completion_tokens),
        latency_s=round(latency_s, 4),
        llm_calls=llm_calls,
        cost_usd=compute_cost_usd(
            model=model,
            prompt_tokens=prompt_tokens,
            cached_input_tokens=cached_input_tokens,
            completion_tokens=completion_tokens,
        ),
        finish_reason=getattr(choice, "finish_reason", None),
        request_hash=request_hash,
    )


class DeepSeekClient:
    """Single-model DeepSeek chat client shared by all four pipelines."""

    def __init__(
        self,
        *,
        model: str | None = None,
        thinking: str | None = None,
        reasoning_effort: Any = _UNSET,
        max_tokens: int | None = None,
        temperature: float | None = None,
        use_cache: bool | None = None,
        max_retries: int | None = None,
    ) -> None:
        problems = config.environment_problems()
        if problems:
            raise RuntimeError("DeepSeekClient not usable: " + " | ".join(problems))

        self.model = model or config.DEEPSEEK_MODEL
        if self.model not in config.VALID_MODEL_IDS:
            raise ValueError(
                f"Unsupported model {self.model!r}; expected one of {config.VALID_MODEL_IDS}."
            )
        self.thinking = (thinking or config.DEEPSEEK_THINKING).lower()
        self.reasoning_effort = (
            config.DEEPSEEK_REASONING_EFFORT if reasoning_effort is _UNSET else reasoning_effort
        )
        self.max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS
        self.temperature = config.TEMPERATURE if temperature is None else temperature
        self.use_cache = config.LLM_CACHE_ENABLED if use_cache is None else use_cache
        self.max_retries = config.MAX_RETRIES if max_retries is None else max_retries
        self._client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            timeout=config.REQUEST_TIMEOUT_SECONDS,
            max_retries=0,  # retries happen here so they can be logged and counted
        )

    def settings(self) -> dict[str, Any]:
        """Decoding parameters to store with every pipeline result."""
        return {
            "provider": "DeepSeek",
            "base_url": config.DEEPSEEK_BASE_URL,
            "model": self.model,
            "model_version": config.DEEPSEEK_MODEL_VERSION,
            "thinking": self.thinking,
            "reasoning_effort": self.reasoning_effort or "api_default",
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature if self.thinking == "disabled" else None,
            "cache_enabled": self.use_cache,
            "max_retries": self.max_retries,
        }

    def _build_params(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int | None,
        temperature: float | None,
        thinking: str | None,
        reasoning_effort: Any,
    ) -> dict[str, Any]:
        mode = (thinking or self.thinking).lower()
        effort = self.reasoning_effort if reasoning_effort is _UNSET else reasoning_effort
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "thinking": mode,
        }
        if effort:
            params["reasoning_effort"] = effort
        if mode == "disabled":
            # The API ignores sampling parameters while thinking mode is on.
            params["temperature"] = self.temperature if temperature is None else temperature
        return params

    @staticmethod
    def _api_payload(params: Mapping[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": params["model"],
            "messages": params["messages"],
            "max_tokens": params["max_tokens"],
            "extra_body": {"thinking": {"type": params["thinking"]}},
        }
        if "reasoning_effort" in params:
            payload["reasoning_effort"] = params["reasoning_effort"]
        if "temperature" in params:
            payload["temperature"] = params["temperature"]
        return payload

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        thinking: str | None = None,
        reasoning_effort: Any = _UNSET,
        use_cache: bool | None = None,
    ) -> LLMResponse:
        """Run one chat completion and return its accounting record."""
        params = self._build_params(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        )
        key = _request_fingerprint(params)
        cache_enabled = self.use_cache if use_cache is None else use_cache
        if cache_enabled:
            entry = _cache_read(key)
            if entry is not None:
                logger.info("LLM cache hit (%s)", key[:12])
                return _cached_response(entry, key)
        return self._request(params, key, cache_enabled=cache_enabled)

    def generate(
        self, prompt: str, *, system: str | None = None, **kwargs: Any
    ) -> LLMResponse:
        """Convenience wrapper around :meth:`chat` for a single-turn prompt."""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def _request(
        self, params: Mapping[str, Any], key: str, *, cache_enabled: bool
    ) -> LLMResponse:
        """Send the request, retrying transient failures with backoff."""
        payload = self._api_payload(params)
        attempts = self.max_retries + 1
        started = time.perf_counter()
        failure: str | None = None

        for attempt in range(1, attempts + 1):
            try:
                completion = self._client.chat.completions.create(**payload)
            except Exception as error:  # noqa: BLE001 - failures are logged, never hidden
                failure = _error_message(error)
                latency_s = round(time.perf_counter() - started, 4)
                will_retry = _is_retryable(error) and attempt < attempts
                _log_api_error(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "model": self.model,
                        "base_url": config.DEEPSEEK_BASE_URL,
                        "request_hash": key,
                        "attempt": attempt,
                        "max_attempts": attempts,
                        "will_retry": will_retry,
                        "status_code": getattr(error, "status_code", None),
                        "error": failure,
                    }
                )
                logger.warning(
                    "DeepSeek request failed (attempt %d/%d, model=%s): %s",
                    attempt,
                    attempts,
                    self.model,
                    failure,
                )
                if will_retry:
                    time.sleep(config.RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
                    continue
                return LLMResponse(
                    text="",
                    model=self.model,
                    latency_s=latency_s,
                    llm_calls=attempt,
                    error=failure,
                    request_hash=key,
                )

            latency_s = round(time.perf_counter() - started, 4)
            response = _to_response(
                completion, latency_s=latency_s, llm_calls=attempt, request_hash=key
            )
            if cache_enabled:
                _cache_write(
                    key,
                    {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "request": params,
                        "response": response.to_dict(),
                    },
                )
            return response

        # Defensive: the loop above returns on every path.
        return LLMResponse(
            text="",
            model=self.model,
            latency_s=round(time.perf_counter() - started, 4),
            llm_calls=attempts,
            error=failure or "request failed",
            request_hash=key,
        )


def _cached_response(entry: Mapping[str, Any], key: str) -> LLMResponse:
    """Rebuild an :class:`LLMResponse` from a cache entry."""
    data = dict(entry.get("response", {}))
    data["from_cache"] = True
    data["request_hash"] = key
    data["latency_s"] = 0.0
    data["llm_calls"] = 0
    known = {field_name for field_name in LLMResponse.__dataclass_fields__}
    return LLMResponse(**{name: value for name, value in data.items() if name in known})


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------

_default_client: DeepSeekClient | None = None


def get_client() -> DeepSeekClient:
    """Return the process-wide client shared by the pipelines."""
    global _default_client
    if _default_client is None:
        _default_client = DeepSeekClient()
    return _default_client


def generate(
    prompt: str,
    *,
    system: str | None = None,
    client: DeepSeekClient | None = None,
    **kwargs: Any,
) -> LLMResponse:
    """Generate one answer with the shared default client (or ``client``)."""
    return (client or get_client()).generate(prompt, system=system, **kwargs)


def main() -> int:
    """One-request smoke test: credentials, model id and usage accounting.

    The cache is bypassed so that the API connection is really exercised.
    """
    _configure_logging()
    print("DeepSeek smoke test")
    client = DeepSeekClient(use_cache=False)
    for key, value in client.settings().items():
        print(f"  {key:20s} {value}")
    print(f"  {'api_key':20s} {config.masked_api_key()}")

    response = client.generate("Reply with the single word: ok", max_tokens=256)

    print("response")
    print(f"  text                {response.text!r}")
    print(f"  reasoning_chars     {len(response.reasoning)}")
    print(f"  finish_reason       {response.finish_reason}")
    print(f"  prompt_tokens       {response.prompt_tokens} (cached {response.cached_input_tokens})")
    print(f"  completion_tokens   {response.completion_tokens} (reasoning {response.reasoning_tokens})")
    print(f"  total_tokens        {response.total_tokens}")
    print(f"  latency_s           {response.latency_s}")
    print(f"  llm_calls           {response.llm_calls}")
    print(f"  cost_usd            {response.cost_usd:.8f}")
    print(f"  error               {response.error}")

    if not response.ok:
        print("DeepSeek connection FAILED")
        return 1
    if not response.text:
        print("DeepSeek connection OK, but no answer text was returned (check max_tokens)")
        return 1
    print("DeepSeek connection OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
