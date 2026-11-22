"""Explicit JSON configuration with schema validation and fingerprinting (Scope B).

Why JSON and not YAML: no new dependency (decision D9). Why a hand-written
validator and not ``jsonschema``: the subset of JSON Schema this project uses is
small (object/array/string/integer/number/boolean, ``required``, ``enum``,
``additionalProperties``, ``items``, ``minimum``/``maximum``), and adding a
dependency to validate ten small files is not justified.

Every experiment run records ``config_fingerprint`` — the sha256 of the resolved
config with keys sorted — so a result can always be traced to the exact settings
that produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src import config

CONFIG_ROOT = config.REPO_ROOT / "configs"
SCHEMA_ROOT = CONFIG_ROOT / "schemas"


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed or fails its schema."""


_TYPE_CHECKS = {
    "object": lambda value: isinstance(value, Mapping),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
}


def validate(instance: Any, schema: Mapping[str, Any], path: str = "$") -> list[str]:
    """Validate ``instance`` against the supported JSON-Schema subset."""
    errors: list[str] = []

    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        unknown = [name for name in types if name not in _TYPE_CHECKS]
        if unknown:
            raise ConfigError(f"schema error at {path}: unsupported type {unknown}")
        if not any(_TYPE_CHECKS[name](instance) for name in types):
            return [f"{path}: expected {' or '.join(types)}, got {type(instance).__name__}"]

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")

    if isinstance(instance, Mapping):
        for key in schema.get("required", ()):
            if key not in instance:
                errors.append(f"{path}: missing required key {key!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(instance) - set(properties))
            if extra:
                errors.append(f"{path}: unexpected keys {extra}")
        for key, sub_schema in properties.items():
            if key in instance:
                errors.extend(validate(instance[key], sub_schema, f"{path}.{key}"))

    if isinstance(instance, list) and "items" in schema:
        for index, item in enumerate(instance):
            errors.extend(validate(item, schema["items"], f"{path}[{index}]"))

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")

    return errors


def load_config(path: str | Path, *, schema_path: str | Path | None = None) -> dict[str, Any]:
    """Load one JSON config and validate it against its schema.

    The schema is taken from the config's ``schema`` key (a name under
    ``configs/schemas/``) unless ``schema_path`` is given.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigError(f"{path}: top level must be an object")

    schema_name = payload.get("schema")
    if schema_path is None:
        if not schema_name:
            raise ConfigError(f"{path}: missing 'schema' key (or pass schema_path explicitly)")
        schema_path = SCHEMA_ROOT / f"{schema_name}.schema.json"
    schema_path = Path(schema_path)
    if not schema_path.exists():
        raise ConfigError(f"{path}: schema not found: {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    errors = validate(payload, schema)
    if errors:
        raise ConfigError(f"{path}: schema validation failed:\n  " + "\n  ".join(errors))
    return payload


def load_many(paths: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    """Load several configs, keyed by file stem."""
    return {Path(path).stem: load_config(path) for path in paths}


def fingerprint(payload: Mapping[str, Any]) -> str:
    """Stable sha256 over a resolved config (08 = the 'schema' key is metadata only).

    The ``schema`` key is included deliberately: changing which schema a config
    claims to satisfy is a change of configuration.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def describe(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {"config_fingerprint": fingerprint(payload), "resolved_config": dict(payload)}


def dataset_config(dataset: str) -> dict[str, Any]:
    """Load and cross-check one dataset config against the live adapter."""
    from src.datasets import get_adapter

    payload = load_config(CONFIG_ROOT / "datasets" / f"{dataset}.json")
    adapter = get_adapter(payload["dataset"], payload["split"])
    actual = adapter.fingerprint()
    if actual.sha256 != payload["dataset_sha256"]:
        raise ConfigError(
            f"{dataset}: config expects dataset sha {payload['dataset_sha256'][:12]}…, "
            f"file hashes to {actual.sha256[:12]}…"
        )
    if payload["expected_questions"] != actual.n_questions:
        raise ConfigError(
            f"{dataset}: config expects {payload['expected_questions']} questions, file has "
            f"{actual.n_questions}"
        )
    if payload["evidence_kind"] != actual.evidence_kind:
        raise ConfigError(
            f"{dataset}: config expects evidence_kind {payload['evidence_kind']!r}, adapter "
            f"reports {actual.evidence_kind!r}"
        )
    return payload


def retrieval_config() -> dict[str, Any]:
    return load_config(CONFIG_ROOT / "retrieval" / "pooled_corpus.json")


def pandarag_config() -> dict[str, Any]:
    return load_config(CONFIG_ROOT / "pandarag.json")


def sac_rag_config() -> dict[str, Any]:
    return load_config(CONFIG_ROOT / "sac_rag.json")
