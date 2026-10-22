"""Carinthia-S manifest construction with the release-specific mask contract."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from seminspect.data.hashing import image_hash_metadata
from seminspect.data.manifests import normalize_manifest
from seminspect.utils.io import ArtifactError


# The published release has one historically decoder-sensitive JPEG.  It is
# accepted only when it has the exact bytes shipped in the checksum-locked
# Zenodo archive; this is not a permissive decoder setting for arbitrary
# future corruptions.
_TOLERATED_TRUNCATED_JPEG = {
    "e1fd81839c094c1cb02e7c6cc4459035.jpg": "b0c347a11d4238086f711f862232292a1de8c5d0406cc540ab4fe068d3ab3e03"
}


def _allow_known_truncated_image(path: Path) -> bool:
    expected = _TOLERATED_TRUNCATED_JPEG.get(path.name)
    if expected is None:
        return False
    from seminspect.utils.io import file_sha256

    if file_sha256(path) != expected:
        raise ArtifactError(f"Known truncated Carinthia filename has unexpected content hash: {path.name}")
    return True


def decode_carinthia_mask(path: str | Path) -> tuple[np.ndarray, str]:
    """Decode foreground while intentionally ignoring an RGBA alpha channel."""

    with Image.open(path) as source:
        mode = source.mode
        if mode == "RGBA":
            # Alpha is fully opaque in two released masks and must not become foreground.
            grayscale = source.convert("RGB").convert("L")
        elif mode == "RGB":
            grayscale = source.convert("L")
        else:
            grayscale = source.convert("L")
        pixels = np.asarray(grayscale, dtype=np.uint8)
    return pixels > 0, mode


def _read_csv(path: Path) -> pd.DataFrame:
    # The published file is semicolon-delimited despite its .csv suffix.
    table = pd.read_csv(path, sep=";")
    expected = {"image_path", "mask_path", "filename", "label"}
    if set(table.columns) != expected:
        raise ArtifactError(f"Unexpected Carinthia-S CSV columns: {table.columns.tolist()}")
    return table


def validate_carinthia_extraction(root: str | Path, expected: dict[str, Any]) -> None:
    """Validate archive layout/counts before a completed extraction is reused."""

    source = Path(root)
    csv_path = source / "carinthia-s.csv"
    if not csv_path.is_file() and (source / "data" / "carinthia-s.csv").is_file():
        source, csv_path = source / "data", source / "data" / "carinthia-s.csv"
    table = _read_csv(csv_path)
    if len(table) != int(expected["image_count"]):
        raise ArtifactError(f"Carinthia-S has {len(table)} rows; expected {expected['image_count']}")
    for row in table.itertuples(index=False):
        if not (source / row.image_path).is_file() or not (source / row.mask_path).is_file():
            raise ArtifactError(f"Carinthia-S extraction is missing pair {row.filename}")


def build_carinthia_manifest(root: str | Path, expected: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build image manifest and explicit per-mask audit table.

    For classes 1–5, a nonempty decoded foreground is mandatory. Violations are
    quarantined and retained in the audit rather than silently relabeled.
    """

    source = Path(root)
    csv_path = source / "carinthia-s.csv"
    if not csv_path.is_file():
        # The Zip archive has a `data/` top-level folder; accept either root form.
        nested = source / "data" / "carinthia-s.csv"
        if nested.is_file():
            source, csv_path = source / "data", nested
        else:
            raise ArtifactError(f"Carinthia-S CSV was not found below {root}")
    table = _read_csv(csv_path)
    if len(table) != int(expected["image_count"]):
        raise ArtifactError(f"Carinthia-S has {len(table)} rows; expected {expected['image_count']}")
    records: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for row in table.sort_values("filename", kind="mergesort").itertuples(index=False):
        image_rel, mask_rel, filename, label = row
        class_id = str(int(label))
        image_path = source / image_rel
        mask_path = source / mask_rel
        if not image_path.is_file() or not mask_path.is_file():
            raise ArtifactError(f"Missing Carinthia image/mask pair for {filename}")
        foreground, mask_mode = decode_carinthia_mask(mask_path)
        allow_truncated = _allow_known_truncated_image(image_path)
        has_foreground = bool(foreground.any())
        quarantined = class_id != "6" and not has_foreground
        reason = "nonempty_defect_class_mask_required" if quarantined else None
        records.append(
            {
                "image_id": f"carinthia_s:{filename}",
                "dataset": "carinthia_s",
                "source_archive": "data.zip",
                "relative_path": str(Path(image_rel)).replace(os.sep, "/"),
                "label": 0 if class_id == "6" else 1,
                "pair_id": None,
                "mask_path": str(Path(mask_rel)).replace(os.sep, "/"),
                "class_id": class_id,
                **image_hash_metadata(image_path, allow_truncated=allow_truncated),
                "split": "carinthia_secondary",
                "duplicate_group": None,
                "quarantined": quarantined,
                "quarantine_reason": reason,
            }
        )
        audits.append(
            {
                "image_id": f"carinthia_s:{filename}",
                "class_id": class_id,
                "mask_mode": mask_mode,
                "decoded_foreground_pixels": int(foreground.sum()),
                "image_label": 0 if class_id == "6" else 1,
                "quarantined": quarantined,
                "quarantine_reason": reason,
                "decode_exception": "hash_locked_truncated_jpeg" if allow_truncated else None,
            }
        )
    manifest = normalize_manifest(pd.DataFrame.from_records(records))
    audit = pd.DataFrame.from_records(audits).sort_values("image_id", kind="mergesort").reset_index(drop=True)
    observed_counts = manifest["class_id"].value_counts().sort_index().to_dict()
    declared_counts = {str(key): int(value) for key, value in expected["class_counts"].items()}
    if observed_counts != declared_counts:
        raise ArtifactError(f"Carinthia class-count mismatch: {observed_counts} != {declared_counts}")
    if not ((manifest["width"].astype(int) == int(expected["native_side"])).all() and (manifest["height"].astype(int) == int(expected["native_side"])).all()):
        raise ArtifactError("Carinthia images must all decode to 480×480")
    return manifest, audit
