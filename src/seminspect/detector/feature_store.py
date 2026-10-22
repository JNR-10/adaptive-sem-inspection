"""Resumable CLS/anomaly-map feature shards without full patch-token persistence."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from seminspect.utils.io import ArtifactError


FEATURE_COLUMNS = [
    "image_id",
    "quality_id",
    "quality_index",
    "effective_side",
    "quality_cost",
    "simulator_hash",
    "detector_raw_score",
    "calibrated_probability",
    "patch_score_max",
    "patch_score_top5_std",
    "patch_score_q95",
    "cls_normal_distance",
    "intensity_mean",
    "intensity_std",
    "robust_dynamic_range",
    "laplacian_variance",
    "tenengrad",
    "highpass_mad",
    "embedding_shard",
    "embedding_row",
    "anomaly_map_shard",
    "anomaly_map_row",
]


@dataclass(frozen=True)
class ShardReference:
    path: str
    row: int


def _write_npz_atomic(path: Path, *, overwrite: bool, **arrays: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ArtifactError(f"Refusing to overwrite completed shard: {path}")
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def write_feature_shard(
    root: str | Path,
    *,
    split: str,
    quality_id: str,
    part: int,
    image_ids: Iterable[str],
    cls_embeddings: np.ndarray,
    anomaly_maps: np.ndarray | None,
    metadata: dict[str, str | int | float],
    overwrite: bool = False,
) -> tuple[ShardReference, ShardReference | None]:
    """Write one complete shard and return manifest-relative references for row zero.

    Callers derive each row reference by adding its local index. The shard holds
    CLS embeddings plus optional square anomaly maps; it intentionally never
    holds raw patch-token tensors for every image. The primary DINOv2 encoder
    writes 16×16 maps and the predeclared ResNet fallback writes 14×14 maps.
    """

    base = Path(root)
    ids = np.asarray(list(image_ids), dtype="U128")
    embeddings = np.asarray(cls_embeddings, dtype=np.float32)
    if embeddings.shape != (len(ids), 384):
        raise ValueError("Feature shard needs one 384-D CLS embedding per image")
    relative = Path("shards") / split / quality_id / f"part-{part:05d}.npz"
    required_metadata = {"config_hash", "source_manifest_hash", "model_commit", "dtype", "shape"}
    if required_metadata - set(metadata):
        raise ValueError(f"Feature shard metadata is missing: {required_metadata - set(metadata)}")
    arrays: dict[str, np.ndarray] = {
        "image_ids": ids,
        "cls_embeddings": embeddings,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")), dtype="U"),
    }
    map_reference: ShardReference | None = None
    if anomaly_maps is not None:
        maps = np.asarray(anomaly_maps, dtype=np.float32)
        if maps.ndim != 3 or maps.shape[0] != len(ids) or maps.shape[1] != maps.shape[2]:
            raise ValueError("Feature shard anomaly maps must have square shape [N,H,H]")
        arrays["anomaly_maps"] = maps
        map_reference = ShardReference(str(relative), 0)
    _write_npz_atomic(base / relative, overwrite=overwrite, **arrays)
    return ShardReference(str(relative), 0), map_reference


def read_embedding(root: str | Path, reference: str, row: int) -> np.ndarray:
    path = Path(root) / reference
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["cls_embeddings"][int(row)], dtype=np.float32)


def read_anomaly_map(root: str | Path, reference: str, row: int) -> np.ndarray:
    path = Path(root) / reference
    with np.load(path, allow_pickle=False) as payload:
        if "anomaly_maps" not in payload:
            raise ArtifactError(f"Feature shard does not contain anomaly maps: {path}")
        return np.asarray(payload["anomaly_maps"][int(row)], dtype=np.float32)


def validate_feature_shard(path: str | Path, *, expected_simulator_hash: str | None = None) -> None:
    """Validate array shapes; per-shard metadata is kept in the Parquet manifest."""

    with np.load(Path(path), allow_pickle=False) as payload:
        if set(payload.files) - {"image_ids", "cls_embeddings", "anomaly_maps", "metadata_json"}:
            raise ArtifactError(f"Unexpected feature shard arrays in {path}")
        ids = payload["image_ids"]
        embeddings = payload["cls_embeddings"]
        if embeddings.shape != (len(ids), 384):
            raise ArtifactError(f"Invalid CLS shape in feature shard: {path}")
        if "anomaly_maps" in payload:
            maps = payload["anomaly_maps"]
            if maps.ndim != 3 or maps.shape[0] != len(ids) or maps.shape[1] != maps.shape[2]:
                raise ArtifactError(f"Invalid anomaly-map shape in feature shard: {path}")
        if "metadata_json" not in payload:
            raise ArtifactError(f"Feature shard does not contain required metadata: {path}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        required = {"config_hash", "source_manifest_hash", "model_commit", "dtype", "shape"}
        if required - set(metadata):
            raise ArtifactError(f"Feature shard metadata is incomplete: {path}")
