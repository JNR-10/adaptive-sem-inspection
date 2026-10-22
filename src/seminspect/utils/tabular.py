"""Atomic Parquet tables for non-manifest run artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from seminspect.utils.io import ArtifactError


def write_parquet(frame: pd.DataFrame, path: str | Path, *, overwrite: bool = False) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise ArtifactError(f"Refusing to overwrite completed table: {destination}")
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(destination)
    return destination


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path), engine="pyarrow")
