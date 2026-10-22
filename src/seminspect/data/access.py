"""Resolve manifest-relative image and mask paths without embedding Colab paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def dataset_root(extracted_dir: str | Path, dataset: str) -> Path:
    root = Path(extracted_dir) / dataset
    # Carinthia-S's zip holds a top-level `data` directory.
    if dataset == "carinthia_s" and (root / "data").is_dir():
        return root / "data"
    return root


def resolve_manifest_path(row: Any, extracted_dir: str | Path, column: str = "relative_path") -> Path:
    dataset = str(getattr(row, "dataset") if hasattr(row, "dataset") else row["dataset"])
    relative = getattr(row, column) if hasattr(row, column) else row[column]
    if relative is None:
        raise ValueError(f"Manifest row has no {column}")
    candidate = dataset_root(extracted_dir, dataset) / str(relative)
    if not candidate.is_file():
        raise FileNotFoundError(f"Manifest artifact is missing: {candidate}")
    return candidate
