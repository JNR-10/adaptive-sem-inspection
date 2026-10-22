"""Shared script plumbing. This file is intentionally importable from `scripts/`."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from seminspect.utils.config import load_config
from seminspect.utils.io import atomic_write_json


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path, help="YAML configuration path")


def runtime_root() -> Path:
    return Path(os.environ.get("SEMSPECT_RUNTIME_ROOT", "runtime"))


def runtime_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = config.get("paths", {})
    root = Path(paths.get("runtime_root", runtime_root()))
    return {
        "runtime_root": root,
        "raw_dir": Path(paths.get("raw_dir", root / "data/raw")),
        "extracted_dir": Path(paths.get("extracted_dir", root / "data/extracted")),
        "manifest_dir": Path(paths.get("manifest_dir", root / "data/manifests")),
        "cache_dir": Path(paths.get("cache_dir", root / "cache")),
        "artifacts_dir": Path(paths.get("artifacts_dir", root / "artifacts")),
        "runs_dir": Path(paths.get("runs_dir", root / "runs")),
    }


def load_core_config(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = load_config(path)
    paths = runtime_paths(config)
    for item in paths.values():
        item.mkdir(parents=True, exist_ok=True)
    return config, paths


def write_failure(artifacts_dir: Path, stage: str, error: Exception) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        artifacts_dir / "data_audit_failure.json",
        {"stage": stage, "error_type": type(error).__name__, "message": str(error)},
        overwrite=True,
    )
