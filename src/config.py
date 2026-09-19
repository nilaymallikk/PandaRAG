"""Shared configuration for the Adaptive RAG experiment.

This module is the single source of truth for paths, dataset constants,
retrieval constants, DeepSeek LLM settings and API pricing.

Rules:

* Import values from here instead of hard-coding them elsewhere.
* Environment variables are read only in this module.
* The API key is never printed or logged; use :func:`masked_api_key` if a
  human-readable identifier is needed.

Run ``python -m src.config`` from the repository root to print a
secret-free summary of the effective configuration.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = REPO_ROOT / ".env"
DATA_DIR = REPO_ROOT / "data"
DATASET_FILE = DATA_DIR / "hotpotqa_500.json"
RESULTS_DIR = REPO_ROOT / "results"
LOG_DIR = RESULTS_DIR / "logs"
LLM_ERROR_LOG = LOG_DIR / "llm_errors.jsonl"
CACHE_DIR = REPO_ROOT / "cache"
LLM_CACHE_DIR = CACHE_DIR / "llm"

# Real environment variables take precedence over the .env file.
load_dotenv(ENV_FILE)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATASET_NAME = "HotpotQA"
DATASET_SPLIT = "distractor/validation"
DATASET_FILENAME = DATASET_FILE.name
EXPECTED_DATASET_SIZE = 500

# ---------------------------------------------------------------------------
# Retrieval (fixed by experiment.md; consumed by later stages)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE's documented "short query -> long passage" usage: the same fixed
# instruction is prepended to every query (documents are embedded as-is).
# Source: BAAI/bge-small-en-v1.5 model card (answer by the model authors).
EMBEDDING_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")
# Small batches keep padding waste low: sentence-transformers sorts texts by
# length, but a batch is still padded to its longest member.
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))

# Recall@K values reported for retrieval (K=10 is the deepest ranking produced,
# smaller K are prefixes of it).
RETRIEVAL_K_VALUES = (1, 3, 5, 10)
RETRIEVAL_MAX_K = max(RETRIEVAL_K_VALUES)

DENSE_RESULTS_FILE = RESULTS_DIR / "retrieval" / "dense_retrieval.jsonl"
DENSE_RESULTS_META_FILE = RESULTS_DIR / "retrieval" / "dense_retrieval.meta.json"
BM25_RESULTS_FILE = RESULTS_DIR / "retrieval" / "bm25_retrieval.jsonl"
BM25_RESULTS_META_FILE = RESULTS_DIR / "retrieval" / "bm25_retrieval.meta.json"

# Lexical retrieval (rank_bm25 defaults, kept explicit for the record).
BM25_K1 = float(os.environ.get("BM25_K1", "1.5"))
BM25_B = float(os.environ.get("BM25_B", "0.75"))
BM25_EPSILON = float(os.environ.get("BM25_EPSILON", "0.25"))

STANDARD_RAG_TOP_K = 5
HYBRID_RAG_TOP_K = 5


# ---------------------------------------------------------------------------
# DeepSeek LLM
# ---------------------------------------------------------------------------

# The API is OpenAI-compatible; the official SDK is used (see src/llm.py).
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")

# Human-readable model version used in AGENTS.md / README.md.
DEEPSEEK_MODEL_VERSION = "DeepSeek-V4.1-Flash"

# The DeepSeek API accepts model *ids*, not version names. As of the official
# docs, ``deepseek-flash`` serves the DeepSeek-V4.1-Flash model version, so the
# version string that appears in .env / AGENTS.md is mapped to its id here.
VALID_MODEL_IDS = ("deepseek-flash", "deepseek-v4-pro")
MODEL_ID_ALIASES = {
    "deepseek-flash": "deepseek-flash",
    "deepseek-v4.1-flash": "deepseek-flash",
    "deepseek_v4.1-flash": "deepseek-flash",
    "deepseek-v4-flash": "deepseek-flash",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "deepseek-v4.1-pro": "deepseek-v4-pro",
}


def resolve_model_id(name: str) -> str:
    """Map a model name/version string to a valid DeepSeek API model id."""
    candidate = (name or "").strip()
    resolved = MODEL_ID_ALIASES.get(candidate.lower())
    if resolved is None:
        raise ValueError(
            f"Unsupported DEEPSEEK_MODEL {candidate!r}. "
            f"Supported API model ids: {', '.join(VALID_MODEL_IDS)}."
        )
    return resolved


DEEPSEEK_MODEL = resolve_model_id(os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"))

# Thinking mode is enabled by default by the API. It is kept configurable so the
# setting is identical across the four pipelines and recorded in every result.
DEEPSEEK_THINKING = os.environ.get("DEEPSEEK_THINKING", "enabled").strip().lower()
if DEEPSEEK_THINKING not in {"enabled", "disabled"}:
    raise ValueError("DEEPSEEK_THINKING must be 'enabled' or 'disabled'.")

# None means "use the API default effort" (high). Otherwise: low | high | max.
DEEPSEEK_REASONING_EFFORT = (
    os.environ.get("DEEPSEEK_REASONING_EFFORT", "").strip().lower() or None
)
if DEEPSEEK_REASONING_EFFORT not in {None, "low", "high", "max"}:
    raise ValueError("DEEPSEEK_REASONING_EFFORT must be low, high or max.")

# In thinking mode the chain of thought is billed as output tokens, so the
# output budget must leave room for the reasoning trace plus the final answer.
MAX_OUTPUT_TOKENS = int(os.environ.get("DEEPSEEK_MAX_OUTPUT_TOKENS", "2048"))

# Temperature only has an effect in non-thinking mode (the API ignores it while
# thinking mode is enabled). Kept at 0 for reproducibility.
TEMPERATURE = float(os.environ.get("DEEPSEEK_TEMPERATURE", "0"))

MAX_RETRIES = int(os.environ.get("DEEPSEEK_MAX_RETRIES", "4"))
RETRY_BACKOFF_SECONDS = float(os.environ.get("DEEPSEEK_RETRY_BACKOFF", "2"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("DEEPSEEK_TIMEOUT", "300"))

# API responses are cached on disk (AGENTS.md rule 10). Cache hits are recorded
# with ``cached=True`` so replayed calls are never billed twice.
LLM_CACHE_ENABLED = os.environ.get("LLM_CACHE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}

# ---------------------------------------------------------------------------
# Pricing (converts the actual token usage returned by the API into cost)
# ---------------------------------------------------------------------------

# Source: https://api-docs.deepseek.com/quick_start/pricing, USD per 1M tokens.
PRICING_USD_PER_1M_TOKENS = {
    "deepseek-flash": {
        "off_peak": {
            "input_cache_hit": 0.003,
            "input_cache_miss": 0.15,
            "output": 0.60,
        },
        "peak": {
            "input_cache_hit": 0.006,
            "input_cache_miss": 0.30,
            "output": 1.20,
        },
    },
    "deepseek-v4-pro": {
        "off_peak": {
            "input_cache_hit": 0.022,
            "input_cache_miss": 0.66,
            "output": 1.98,
        },
        "peak": {
            "input_cache_hit": 0.044,
            "input_cache_miss": 1.32,
            "output": 3.96,
        },
    },
}

# Peak hours are 01:00-04:00 and 06:00-10:00 UTC, Monday-Friday, excluding
# Chinese public holidays. Every other hour is off-peak at half the peak rate.
PEAK_HOURS_UTC = ((1, 4), (6, 10))


def is_peak_time(timestamp: datetime | None = None) -> bool:
    """Return True if ``timestamp`` (UTC) falls inside DeepSeek peak hours.

    Chinese public holidays are not modelled: holiday hours are treated as
    peak, which makes the reported cost a (small) upper bound.
    """
    ts = timestamp or datetime.now(timezone.utc)
    ts = ts.astimezone(timezone.utc)
    if ts.weekday() >= 5:  # Saturday / Sunday
        return False
    return any(start <= ts.hour < end for start, end in PEAK_HOURS_UTC)


def pricing_for(
    model_id: str | None = None, timestamp: datetime | None = None
) -> dict[str, float]:
    """Return the USD-per-1M-token price table that applies to a request."""
    prices = PRICING_USD_PER_1M_TOKENS.get(model_id or DEEPSEEK_MODEL)
    if prices is None:
        raise KeyError(f"No pricing data for model {model_id or DEEPSEEK_MODEL!r}.")
    return prices["peak" if is_peak_time(timestamp) else "off_peak"]


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def masked_api_key() -> str:
    """Return a non-reversible-enough identifier for logs (never the key)."""
    if not DEEPSEEK_API_KEY:
        return "<missing>"
    return f"{DEEPSEEK_API_KEY[:6]}...<redacted:{len(DEEPSEEK_API_KEY)} chars>"


def environment_problems() -> list[str]:
    """Return a list of human-readable configuration problems (empty = OK)."""
    problems: list[str] = []
    if not DEEPSEEK_API_KEY:
        problems.append(f"DEEPSEEK_API_KEY is not set (expected in {ENV_FILE}).")
    if not ENV_FILE.exists():
        problems.append(f"Environment file not found: {ENV_FILE}")
    return problems


def describe(timestamp: datetime | None = None) -> dict[str, object]:
    """Return a secret-free snapshot of the effective configuration."""
    ts = timestamp or datetime.now(timezone.utc)
    return {
        "dataset_name": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "dataset_file": str(DATASET_FILE),
        "dataset_size": EXPECTED_DATASET_SIZE,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_query_instruction": EMBEDDING_QUERY_INSTRUCTION,
        "embedding_device": EMBEDDING_DEVICE,
        "retrieval_k_values": list(RETRIEVAL_K_VALUES),
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "bm25_epsilon": BM25_EPSILON,
        "standard_rag_k": STANDARD_RAG_TOP_K,
        "hybrid_rag_k": HYBRID_RAG_TOP_K,
        "llm_provider": "DeepSeek",
        "llm_base_url": DEEPSEEK_BASE_URL,
        "llm_model_id": DEEPSEEK_MODEL,
        "llm_model_version": DEEPSEEK_MODEL_VERSION,
        "llm_thinking": DEEPSEEK_THINKING,
        "llm_reasoning_effort": DEEPSEEK_REASONING_EFFORT or "api_default",
        "llm_max_output_tokens": MAX_OUTPUT_TOKENS,
        "llm_temperature": TEMPERATURE,
        "llm_max_retries": MAX_RETRIES,
        "llm_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "llm_cache_enabled": LLM_CACHE_ENABLED,
        "llm_cache_dir": str(LLM_CACHE_DIR),
        "api_key": masked_api_key(),
        "pricing_period": "peak" if is_peak_time(ts) else "off_peak",
        "timestamp_utc": ts.isoformat(),
    }


def main() -> int:
    """Print the effective configuration without exposing secrets."""
    problems = environment_problems()
    for key, value in describe().items():
        print(f"{key:24s} {value}")
    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nconfiguration OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
