"""Experiment-integrity tests for the two-scope repository.

These guard the boundary between the frozen Scope-A experiment (four pipelines,
HotpotQA-500) and the active Scope-B work (SAC-RAG / PandaRAG). If any of them
fails, something has crossed that boundary and results from the two scopes would
stop being separable.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from src import config
from src.experiment_config import (
    ConfigError,
    dataset_config,
    fingerprint,
    load_config,
    pandarag_config,
    sac_rag_config,
    validate,
)

FROZEN_HOTPOT_SHA = "956405ceb43d73385d60898e1bfc7f45a8184c9a4e1eff45b9a8ec239acfd048"


def test_frozen_hotpotqa_dataset_is_unchanged() -> None:
    payload = (config.DATA_DIR / "hotpotqa_500.json").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == FROZEN_HOTPOT_SHA


def test_legacy_retrieval_module_is_not_shadowed() -> None:
    """Scope B adds ``src/corpus_retrieval``; ``src/retrieval.py`` must stay importable.

    A package named ``src/retrieval/`` would shadow the frozen module and silently
    change what the legacy pipelines and their tests import.
    """
    import src.retrieval as legacy

    assert legacy.__file__.endswith("src/retrieval.py")
    assert hasattr(legacy, "tokenize")
    assert hasattr(legacy, "fuse_rankings")
    assert not (config.REPO_ROOT / "src" / "retrieval").is_dir()


def test_adaptive_rag_protocol_guards_are_intact() -> None:
    from src.pipelines import adaptive_rag

    assert "src.retrieval" in adaptive_rag._PROTOCOL_FORBIDDEN_MODULES
    policy = adaptive_rag.check_frozen_policy()
    assert policy["initial_k"] == 3
    assert policy["allowed_final_k"] == [3, 5]
    assert policy["threshold"] == 0.5


def test_legacy_results_are_still_readable_and_complete() -> None:
    for name in ("no_rag", "standard_rag", "hybrid_rag", "adaptive_rag"):
        path = config.RESULTS_DIR / f"{name}.jsonl"
        assert path.exists(), f"legacy artifact missing: {path}"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 500, f"{name}: expected 500 legacy records, found {len(rows)}"
        assert all("question_id" in row for row in rows)


def test_new_results_namespaces_do_not_overwrite_legacy() -> None:
    legacy = {path.name for path in config.RESULTS_DIR.glob("*.jsonl")}
    sac_rag_dir = config.RESULTS_DIR / "sac_rag"
    if sac_rag_dir.exists():
        for path in sac_rag_dir.rglob("*.jsonl"):
            assert path.name not in legacy, f"{path} would shadow a legacy artifact name"


@pytest.mark.parametrize("dataset", ["hotpotqa", "2wiki", "musique"])
def test_dataset_config_matches_live_adapter(dataset: str) -> None:
    payload = dataset_config(dataset)
    assert payload["expected_questions"] == 500
    assert payload["schema"] == "dataset"


def test_dataset_config_rejects_wrong_sha(tmp_path, monkeypatch) -> None:
    """A config whose dataset_sha256 disagrees with the file must be refused."""
    from src import experiment_config

    configs_dir = tmp_path / "configs"
    (configs_dir / "datasets").mkdir(parents=True)
    (configs_dir / "schemas").mkdir(parents=True)

    source = config.REPO_ROOT / "configs" / "datasets" / "hotpotqa.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["dataset_sha256"] = "0" * 64
    (configs_dir / "datasets" / "hotpotqa.json").write_text(json.dumps(payload), encoding="utf-8")
    schema = config.REPO_ROOT / "configs" / "schemas" / "dataset.schema.json"
    (configs_dir / "schemas" / "dataset.schema.json").write_text(
        schema.read_text(encoding="utf-8"), encoding="utf-8"
    )

    monkeypatch.setattr(experiment_config, "CONFIG_ROOT", configs_dir)
    monkeypatch.setattr(experiment_config, "SCHEMA_ROOT", configs_dir / "schemas")

    with pytest.raises(ConfigError, match="config expects dataset sha"):
        experiment_config.dataset_config("hotpotqa")


def test_config_fingerprint_is_stable_and_content_sensitive() -> None:
    payload = sac_rag_config()
    assert fingerprint(payload) == fingerprint(json.loads(json.dumps(payload)))
    changed = json.loads(json.dumps(payload))
    changed["fact"]["final_k"] = 9
    assert fingerprint(changed) != fingerprint(payload)


def test_pandarag_budget_semantics_are_recorded() -> None:
    payload = pandarag_config()
    budget = payload["budget"]
    assert budget["max_additional_rounds"] == 2
    assert budget["max_cumulative_unique_documents"] == 20
    assert budget["max_total_retriever_executions"] == 8
    assert budget["max_generator_context_tokens"] == 4000
    assert budget["max_requirements"] == 4
    semantics = payload["budget_semantics"]
    assert "AFTER" in semantics["max_additional_rounds"]
    assert "UNIQUE" in semantics["max_cumulative_unique_documents"]
    assert "individual retriever calls" in semantics["max_total_retriever_executions"]
    assert "never by truncating a requirement" in semantics["max_generator_context_tokens"]


def test_config_loader_rejects_unknown_keys() -> None:
    payload = json.loads((config.REPO_ROOT / "configs" / "pandarag.json").read_text())
    payload["unexpected_key"] = 1
    schema = json.loads((config.REPO_ROOT / "configs" / "schemas" / "pandarag.schema.json").read_text())
    errors = validate(payload, schema)
    assert any("unexpected keys" in error for error in errors)


def test_config_loader_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(ConfigError, match="config not found"):
        load_config(tmp_path / "nope.json")
