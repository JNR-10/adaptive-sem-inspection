"""Small deterministic YAML configuration loader used by every script.

The project deliberately avoids notebook-local configuration.  Includes are
resolved relative to the including file, merged left-to-right, and represented
as regular dictionaries so the resolved configuration can be hashed and sealed.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

from seminspect.utils.io import canonical_json_sha256

_ENV = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


class ConfigError(ValueError):
    """Raised when a configuration is malformed or impossible to resolve."""


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group("name")
            default = match.group("default")
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigError(f"Environment variable {name!r} is required by configuration")

        return _ENV.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand_env(item) for key, item in value.items()}
    return value


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Merge dictionaries without mutating either operand.

    Lists and scalar values are replaced rather than concatenated.  This makes
    a resolved configuration unambiguous and prevents an include from silently
    retaining a stale list item.
    """

    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigError(f"Circular config include: {chain}")
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Top-level YAML object must be a mapping: {path}")

    include = raw.pop("include", [])
    if isinstance(include, str):
        include = [include]
    if not isinstance(include, list) or not all(isinstance(item, str) for item in include):
        raise ConfigError(f"include must be a string or a list of strings: {path}")

    merged: dict[str, Any] = {}
    for item in include:
        merged = deep_merge(merged, _load(path.parent / item, (*stack, path)))
    return deep_merge(merged, raw)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load, include, environment-expand, validate, and annotate a YAML config."""

    config_path = Path(path)
    config = _expand_env(_load(config_path, ()))
    if not isinstance(config, dict):  # defensive; `_load` already ensures it.
        raise ConfigError("Resolved config must be a mapping")
    config["_resolved_from"] = str(config_path.resolve())
    validate_core_config(config)
    return config


def config_hash(config: dict[str, Any]) -> str:
    """Hash a resolved config while retaining its source path for provenance."""

    return canonical_json_sha256(config)


def require(config: dict[str, Any], *keys: str) -> Any:
    """Read a nested configuration key and raise a useful error if absent."""

    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            joined = ".".join(keys)
            raise ConfigError(f"Required configuration key is missing: {joined}")
        current = current[key]
    return current


def validate_core_config(config: dict[str, Any]) -> None:
    """Validate invariants that must hold before any feature cache is created."""

    qualities = config.get("qualities")
    if qualities is None:
        return  # Data-only config files are valid inputs to archive verification.
    if not isinstance(qualities, list) or len(qualities) != 4:
        raise ConfigError("core_v1 requires exactly four quality levels")
    expected_ids = ["Q0", "Q1", "Q2", "Q3"]
    expected_sides = [56, 112, 224, 448]
    ids = [item.get("id") for item in qualities]
    sides = [item.get("effective_side") for item in qualities]
    costs = [item.get("cumulative_cost") for item in qualities]
    if ids != expected_ids or sides != expected_sides:
        raise ConfigError("core_v1 qualities must be Q0–Q3 at effective sides 56/112/224/448")
    if any(not isinstance(cost, (int, float)) for cost in costs):
        raise ConfigError("Each quality must have a numeric cumulative_cost")
    if any(float(b) <= float(a) for a, b in zip(costs, costs[1:])):
        raise ConfigError("Quality cumulative costs must be strictly increasing")
    if float(costs[-1]) != 1.0:
        raise ConfigError("The primary quality schedule must have Q3 cumulative cost 1.0")
    router = config.get("router", {})
    if router and router.get("input_dim") != 53:
        raise ConfigError("router.input_dim must be 53 for router_feature_schema_v1")
