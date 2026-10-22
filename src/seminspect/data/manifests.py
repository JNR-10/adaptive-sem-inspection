"""Canonical manifest schema and Parquet persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from seminspect.utils.io import ArtifactError, atomic_write_json, canonical_json_sha256

MANIFEST_COLUMNS = [
    "image_id",
    "dataset",
    "source_archive",
    "relative_path",
    "label",
    "pair_id",
    "mask_path",
    "class_id",
    "width",
    "height",
    "mode",
    "file_sha256",
    "pixel_sha256",
    "phash64",
    "split",
    "duplicate_group",
    "quarantined",
    "quarantine_reason",
]


class ManifestError(ValueError):
    """Raised for a manifest that cannot enter the modeling pipeline."""


def empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def validate_manifest(frame: pd.DataFrame, *, require_complete_hashes: bool = True) -> None:
    missing = [column for column in MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise ManifestError(f"Manifest is missing columns: {missing}")
    if frame["image_id"].isna().any() or frame["image_id"].duplicated().any():
        raise ManifestError("image_id must be non-null and unique")
    if not set(frame["dataset"].dropna().unique()).issubset({"miic", "carinthia_s"}):
        raise ManifestError("dataset contains an unknown value")
    labels = set(frame["label"].dropna().astype(int).unique())
    if not labels.issubset({0, 1}):
        raise ManifestError("label must be 0, 1, or null")
    if (frame["width"].dropna().astype(int) <= 0).any() or (frame["height"].dropna().astype(int) <= 0).any():
        raise ManifestError("image dimensions must be positive")
    if require_complete_hashes and (frame["pixel_sha256"].isna().any() or frame["file_sha256"].isna().any()):
        raise ManifestError("all modeled rows require raw and decoded hashes")
    if frame["quarantined"].isna().any():
        raise ManifestError("quarantined must be explicitly true or false")
    invalid_reason = frame["quarantined"] & frame["quarantine_reason"].isna()
    if invalid_reason.any():
        raise ManifestError("quarantined rows need a quarantine_reason")
    if frame.duplicated(["relative_path", "dataset"], keep=False).any():
        raise ManifestError("a dataset-relative path may appear only once")


def normalize_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an ordered, schema-complete copy with predictable nullable values."""

    result = frame.copy()
    for column in MANIFEST_COLUMNS:
        if column not in result:
            result[column] = None
    result = result[MANIFEST_COLUMNS].copy()
    result["quarantined"] = result["quarantined"].fillna(False).astype(bool)
    result["label"] = pd.array(result["label"], dtype="Int64")
    result["phash64"] = pd.array(result["phash64"], dtype="UInt64")
    result["width"] = pd.array(result["width"], dtype="Int64")
    result["height"] = pd.array(result["height"], dtype="Int64")
    result = result.sort_values("image_id", kind="mergesort").reset_index(drop=True)
    validate_manifest(result)
    return result


def write_manifest(frame: pd.DataFrame, path: str | Path, *, overwrite: bool = False) -> Path:
    """Atomically write a normalized Parquet manifest plus a JSON sidecar hash."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise ArtifactError(f"Refusing to overwrite manifest: {destination}")
    normalized = normalize_manifest(frame)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        normalized.to_parquet(temporary, index=False, engine="pyarrow")
    except ImportError as exc:
        temporary.unlink(missing_ok=True)
        raise ArtifactError("pyarrow is required to write core_v1 Parquet manifests") from exc
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(destination)
    sidecar = destination.with_suffix(destination.suffix + ".sha256.json")
    atomic_write_json(
        sidecar,
        {
            "manifest": destination.name,
            "row_count": len(normalized),
            "schema_columns": MANIFEST_COLUMNS,
            "content_hash": canonical_json_sha256(normalized.to_dict(orient="records")),
        },
        overwrite=overwrite,
    )
    return destination


def read_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(Path(path), engine="pyarrow")
    return normalize_manifest(frame)


def assert_unique_quality_rows(frame: pd.DataFrame) -> None:
    required = {"image_id", "quality_id"}
    if not required.issubset(frame.columns):
        raise ManifestError("Quality feature manifest must include image_id and quality_id")
    if frame.duplicated(["image_id", "quality_id"]).any():
        raise ManifestError("Each (image_id, quality_id) pair must be unique")


def manifest_content_hash(frame: pd.DataFrame) -> str:
    """Hash values, not Parquet metadata, in stable image-ID order."""

    normalized = normalize_manifest(frame)
    return canonical_json_sha256(normalized.to_dict(orient="records"))


def concatenate_manifests(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(list(frames), axis=0, ignore_index=True)
    return normalize_manifest(combined)
