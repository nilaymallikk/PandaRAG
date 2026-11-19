"""LLM question-type classifier: Fact / Definition / Reasoning (SAC-RAG stage 1).

The base paper classifies every query with an LLM before retrieval. The label is
**not** a dataset field: no multi-hop QA dataset ships it. This module produces
it once per question and caches it in a sidecar file, so the frozen dataset files
(`data/*_500.json`) stay untouched and their hashes stay valid.

Output (one per dataset)::

    data/semantic_types_hotpotqa.json
    {
      "dataset": "hotpotqa", "dataset_sha256": "...",
      "prompt_version": "sac-rag-classifier/1.0", "model": "deepseek-flash",
      "counts": {"Fact": 480, "Definition": 8, "Reasoning": 12},
      "types": {"<question id>": "Fact", ...}
    }

Design choices (recorded in the output):

* temperature 0, thinking mode disabled -- one deterministic label per question,
  identical across every pipeline, unlike the base paper whose type counts drift
  between runs (Table 1 vs Table 5: Fact 958 vs 954).
* few-shot examples are hand-written, never taken from the three sampled
  datasets, so the prompt leaks no evaluation questions.
* the classifier is LLM-based like the base paper; a keyword heuristic is NOT
  used for labelling (only as a printed sanity check against the paper's own
  keyword lists).

Usage::

    python -m src.classifier                        # all three datasets
    python -m src.classifier --datasets hotpotqa    # one dataset
    python -m src.classifier --limit 10             # smoke run
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from src import config, dataset
from src.llm import DeepSeekClient, LLMResponse

PROMPT_VERSION = "sac-rag-classifier/1.3"
LABELS = ("Fact", "Definition", "Reasoning")

DATASET_FILES = {
    "hotpotqa": "data/hotpotqa_500.json",
    "2wiki": "data/2wiki_500.json",
    "musique": "data/musique_500.json",
}

SYSTEM_PROMPT = (
    "You are a question-type classifier for a multi-hop question answering system. "
    "You assign exactly one label to a question: Fact, Definition, or Reasoning. "
    "You reply with a single JSON object and nothing else."
)

# Few-shot examples are hand-written (not dataset questions) so no evaluation
# question ever appears in the prompt.
FEWSHOT = (
    'EXAMPLE 1\n'
    'Question: In which year was the University of Warsaw established?\n'
    'Answer: {"type": "Fact"}\n\n'
    'EXAMPLE 2\n'
    'Question: Who won the 1998 FIFA World Cup?\n'
    'Answer: {"type": "Fact"}\n\n'
    'EXAMPLE 3\n'
    'Question: What is quantum entanglement?\n'
    'Answer: {"type": "Definition"}\n\n'
    'EXAMPLE 4\n'
    'Question: Who is Ada Lovelace?\n'
    'Answer: {"type": "Definition"}\n\n'
    'EXAMPLE 5\n'
    'Question: Why did the Roman Empire fall?\n'
    'Answer: {"type": "Reasoning"}\n\n'
    'EXAMPLE 6\n'
    'Question: What caused the decline of the Ottoman Empire?\n'
    'Answer: {"type": "Reasoning"}\n\n'
    'EXAMPLE 7\n'
    'Question: Which film was released first, Alien or Blade Runner?\n'
    'Answer: {"type": "Reasoning"}'
)

# Definitions and keyword lists follow the base paper's Section III-B verbatim. The
# "where does the answer live" rule resolves the only ambiguity in their text:
# a multi-hop chain (bridge/compositional) whose answer IS stated in some passage stays
# Fact or Definition -- the search is hard, not the answer; an answer that is stated
# nowhere and must be derived (comparison) or explained (cause/process) is Reasoning.
# The base paper publishes neither its prompt nor its classifier accuracy, so its exact
# per-type counts are not reproducible; our distribution is reported as-is, never tuned.
TASK_PROMPT = """Classify the QUESTION into exactly one type.

Type definitions:
- Fact: asks about a specific fact, value, or entity attribute. Typical question words: When, Where, How many, Which.
- Definition: asks for the definition or identity of a concept or entity. Typical question words: What is, Who is, Define.
- Reasoning: requires multi-step reasoning and causal analysis. Typical question words: Why, How, What caused.

Decide by where the answer lives:
- Fact or Definition: the answer is stated in some passage. A multi-hop chain that ends in a stated fact is still Fact (the search is hard, not the answer).
- Reasoning: the answer is stated nowhere and must be derived by comparing evidence (e.g. "Which came first, X or Y?") or by explaining a cause or process.

{examples}

Now classify this question:
Question: {question}

Reply with only a JSON object: {{"type": "Fact"}} or {{"type": "Definition"}} or {{"type": "Reasoning"}}"""


# ---------------------------------------------------------------------------
# Label parsing and (sanity-check only) keyword heuristic
# ---------------------------------------------------------------------------


def parse_label(text: str) -> str | None:
    """Extract a valid label from the model output; None when unparseable."""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
        label = str(payload.get("type", "")).strip().capitalize()
        if label in LABELS:
            return label
    except (json.JSONDecodeError, AttributeError):
        pass
    for label in LABELS:  # fallback: first valid label mentioned anywhere
        if re.search(rf"\b{label}\b", stripped, flags=re.IGNORECASE):
            return label
    return None


def keyword_guess(question: str) -> str:
    """Paper's keyword lists as a heuristic -- sanity check, never the label."""
    text = question.strip().lower()
    if re.match(r"^(what|who)\s+(is|are|was|were)\b", text) or text.startswith("define "):
        return "Definition"
    if re.match(r"^(why|how)\b", text) and not re.match(
        r"^how\s+(many|much|long|old|tall|far|large|big|deep|high|wide|often)\b", text
    ):
        return "Reasoning"
    if "what caused" in text:
        return "Reasoning"
    return "Fact"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_question(client: DeepSeekClient, question: str) -> tuple[str, LLMResponse]:
    """Return the label for one question plus the raw response for accounting."""
    prompt = TASK_PROMPT.format(examples=FEWSHOT, question=question.strip())
    response = client.generate(
        prompt,
        system=SYSTEM_PROMPT,
        thinking="disabled",  # one deterministic label, minimal tokens
        temperature=0.0,
        max_tokens=64,
    )
    if not response.ok:
        return "unclassified", response
    return parse_label(response.text) or "unclassified", response


def load_semantic_types(name: str) -> dict[str, str]:
    """Load the ``qid -> label`` sidecar for a dataset (raises if missing)."""
    path = config.DATA_DIR / f"semantic_types_{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No semantic types for {name!r}; run `python -m src.classifier`.")
    return json.loads(path.read_text(encoding="utf-8"))["types"]


def classify_dataset(
    name: str, *, limit: int | None = None, workers: int = 8, use_cache: bool = True
) -> dict:
    path = config.REPO_ROOT / DATASET_FILES[name]
    examples = dataset.load_dataset(path)
    if limit is not None:
        examples = examples[:limit]
    client = DeepSeekClient()

    def work(example) -> tuple[str, str, LLMResponse]:
        label, response = classify_question(client, example.question)
        return example.example_id, label, response

    print(f"  {name}: classifying {len(examples)} questions with {workers} workers …", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(work, examples))

    types: dict[str, str] = {}
    calls = cost = 0.0
    for example_id, label, response in results:
        types[example_id] = label
        if not response.from_cache:
            calls += response.llm_calls
            cost += response.cost_usd

    counts = {label: sum(1 for v in types.values() if v == label) for label in LABELS}
    unclassified = sum(1 for v in types.values() if v == "unclassified")
    heuristic_agree = sum(
        1 for ex, (qid, label, _) in zip(examples, results) if keyword_guess(ex.question) == label
    )

    payload = {
        "dataset": name,
        "dataset_file": DATASET_FILES[name],
        "dataset_sha256": dataset.dataset_fingerprint(path)["sha256"],
        "generator": "src/classifier.py",
        "prompt_version": PROMPT_VERSION,
        "taxonomy_note": ("Fact/Definition/Reasoning follow the base paper's Section III-B; the "
                          "derived-vs-located-answer rule resolves its ambiguous Reasoning "
                          "definition. Base paper publishes no prompt/accuracy, so type counts "
                          "are not directly comparable"),
        "model": client.model,
        "decoding": {"thinking": "disabled", "temperature": 0.0, "max_tokens": 64},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "unclassified": unclassified,
        "keyword_heuristic_agreement": round(heuristic_agree / len(examples), 4),
        "new_api_calls": int(calls),
        "new_cost_usd": round(cost, 8),
        "types": dict(sorted(types.items())),
    }
    out = config.DATA_DIR / f"semantic_types_{name}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    n = len(examples)
    print(f"    -> {out.name}: " +
          " ".join(f"{k}={v} ({v / n:.1%})" for k, v in counts.items()) +
          (f" unclassified={unclassified}" if unclassified else "") +
          f" | heuristic agreement={heuristic_agree / n:.1%} | new calls={int(calls)} cost=${cost:.4f}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify questions as Fact/Definition/Reasoning.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_FILES), choices=list(DATASET_FILES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    if not config.DEEPSEEK_API_KEY:
        raise SystemExit("DEEPSEEK_API_KEY is not set (.env); cannot classify.")

    summaries = {}
    for name in args.datasets:
        summaries[name] = classify_dataset(
            name, limit=args.limit, workers=args.workers, use_cache=not args.no_cache
        )

    print("\nSummary vs base paper Table 1 (Fact / Definition / Reasoning)")
    paper = {"hotpotqa": (0.958, 0.020, 0.022), "2wiki": (0.796, 0.005, 0.199),
             "musique": (0.732, 0.028, 0.240)}
    for name, payload in summaries.items():
        n = sum(payload["counts"].values()) + payload["unclassified"]
        ours = tuple(payload["counts"][label] / n for label in LABELS)
        reference = paper[name]
        print(f"  {name:9s} ours={ours[0]:.3f}/{ours[1]:.3f}/{ours[2]:.3f}  "
              f"paper={reference[0]:.3f}/{reference[1]:.3f}/{reference[2]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
