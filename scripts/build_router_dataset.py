#!/usr/bin/env python3
"""Construct leakage-safe Value-of-Resolution supervision and 53-D inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from _common import add_config_argument, load_core_config
from seminspect.data.manifests import read_manifest
from seminspect.detector.feature_store import read_embedding
from seminspect.router.features import FeatureTransform, ROUTER_FEATURE_NAMES, RouterObservation, raw_router_features
from seminspect.router.targets import next_resolution_gain
from seminspect.utils.io import ArtifactError, atomic_write_json, read_json
from seminspect.utils.tabular import read_parquet, write_parquet


class _EmbeddingReader:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.cache: dict[str, np.ndarray] = {}

    def read(self, shard: str, row: int) -> np.ndarray:
        if shard not in self.cache:
            with np.load(self.root / shard, allow_pickle=False) as payload:
                self.cache[shard] = np.asarray(payload["cls_embeddings"], dtype=np.float32)
        return self.cache[shard][int(row)]


def _source_labels(paths: dict[str, Path]) -> pd.DataFrame:
    manifest = read_manifest(paths["manifest_dir"] / "miic_splits.parquet")
    return manifest[["image_id", "label", "pair_id", "split"]]


def _load_split(name: str, paths: dict[str, Path]) -> pd.DataFrame:
    path = paths["cache_dir"] / "features" / f"features_calibrated_{name}.parquet"
    if not path.is_file():
        raise ArtifactError(f"Calibrated feature table is missing: {path}")
    frame = read_parquet(path).merge(_source_labels(paths), on="image_id", how="left", validate="many_to_one")
    if frame["label"].isna().any() or not frame["split"].eq(name).all():
        raise ArtifactError(f"Feature table {name} contains non-{name} source rows")
    if frame["calibrated_probability"].isna().any():
        raise ArtifactError(f"Feature table {name} has uncalibrated detector scores")
    if frame.duplicated(["image_id", "quality_index"]).any() or not set(frame["quality_index"]) == {0, 1, 2, 3}:
        raise ArtifactError(f"Feature table {name} must have exactly Q0–Q3 per image")
    return frame.sort_values(["image_id", "quality_index"], kind="mergesort").reset_index(drop=True)


def _observation(row: pd.Series, reader: _EmbeddingReader) -> RouterObservation:
    return RouterObservation(
        calibrated_probability=float(row.calibrated_probability),
        detector_raw_score=float(row.detector_raw_score),
        patch_score_max=float(row.patch_score_max),
        patch_score_top5_std=float(row.patch_score_top5_std),
        patch_score_q95=float(row.patch_score_q95),
        cls_normal_distance=float(row.cls_normal_distance),
        cls_embedding=reader.read(str(row.embedding_shard), int(row.embedding_row)),
        intensity_mean=float(row.intensity_mean),
        intensity_std=float(row.intensity_std),
        robust_dynamic_range=float(row.robust_dynamic_range),
        laplacian_variance=float(row.laplacian_variance),
        tenengrad=float(row.tenengrad),
        highpass_mad=float(row.highpass_mad),
    )


def _raw_states(
    frame: pd.DataFrame,
    pca,
    reader: _EmbeddingReader,
    *,
    include_target: bool,
    false_negative_weight: float,
    false_positive_weight: float,
    target_clip: tuple[float, float],
    helpful_delta: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict] = []
    embeddings: list[np.ndarray] = []
    for image_id, group in frame.groupby("image_id", sort=True):
        group = group.sort_values("quality_index", kind="mergesort").reset_index(drop=True)
        if len(group) != 4:
            raise ArtifactError(f"Image {image_id} does not have Q0–Q3 feature rows")
        label = int(group.loc[0, "label"])
        observations = [_observation(group.loc[index], reader) for index in range(4)]
        for quality in range(3):
            raw = raw_router_features(
                observations[quality],
                quality_index=quality,
                pca=pca,
                previous=observations[quality - 1] if quality else None,
            )
            record = {"image_id": str(image_id), "quality_index": quality}
            record.update({name: float(value) for name, value in zip(ROUTER_FEATURE_NAMES, raw, strict=True)})
            if include_target:
                gain, helpful = next_resolution_gain(
                    label,
                    observations[quality].calibrated_probability,
                    observations[quality + 1].calibrated_probability,
                    false_negative_weight=false_negative_weight,
                    false_positive_weight=false_positive_weight,
                    clip_target=target_clip,
                    helpful_delta=helpful_delta,
                )
                record["signed_gain"] = gain
                record["helpful_acquisition"] = helpful
            rows.append(record)
            embeddings.append(observations[quality].cls_embedding)
    return pd.DataFrame.from_records(rows), np.stack(embeddings)


def _standardize(frame: pd.DataFrame, transform: FeatureTransform) -> pd.DataFrame:
    output = frame[["image_id", "quality_index"]].copy()
    standardized = transform.transform(frame.loc[:, ROUTER_FEATURE_NAMES].to_numpy(dtype=np.float64))
    output.loc[:, ROUTER_FEATURE_NAMES] = standardized
    for column in ["signed_gain", "helpful_acquisition"]:
        if column in frame:
            output[column] = frame[column].to_numpy()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--force", action="store_true", help="Replace development router-dataset outputs only")
    args = parser.parse_args()
    config, paths = load_core_config(args.config)
    gate_path = paths["artifacts_dir"] / "detector_feasibility.json"
    if not gate_path.is_file():
        raise ArtifactError("Run scripts/run_detector_feasibility.py before constructing router supervision")
    gate = read_json(gate_path)
    if gate.get("protocol_version") != config["protocol_version"] or not gate.get("passed"):
        raise ArtifactError("Detector feasibility gate is missing, belongs to another protocol, or did not pass")
    train = _load_split("router_train", paths)
    validation = _load_split("router_validation", paths)
    reader = _EmbeddingReader(paths["cache_dir"] / "features")
    train_embeddings = np.stack(
        [reader.read(str(row.embedding_shard), int(row.embedding_row)) for row in train.loc[train["quality_index"] < 3].itertuples(index=False)]
    )
    pca = FeatureTransform.fit_pca(train_embeddings)
    raw_train, _ = _raw_states(
        train,
        pca,
        reader,
        include_target=True,
        false_negative_weight=float(config["router"]["fn_weight"]),
        false_positive_weight=float(config["router"]["fp_weight"]),
        target_clip=tuple(float(item) for item in config["router"]["target_clip"]),
        helpful_delta=float(config["router"]["helpful_delta"]),
    )
    raw_validation, _ = _raw_states(
        validation,
        pca,
        reader,
        include_target=True,
        false_negative_weight=float(config["router"]["fn_weight"]),
        false_positive_weight=float(config["router"]["fp_weight"]),
        target_clip=tuple(float(item) for item in config["router"]["target_clip"]),
        helpful_delta=float(config["router"]["helpful_delta"]),
    )
    expected_train = int(config["splits"]["router_train_pairs"]) * 2 * 3
    expected_validation = int(config["splits"]["router_validation_pairs"]) * 2 * 3
    if len(raw_train) != expected_train or len(raw_validation) != expected_validation:
        raise ArtifactError(
            f"Router supervision must have {expected_train}/{expected_validation} train/validation states; "
            f"got {len(raw_train)}/{len(raw_validation)}"
        )
    scaler = StandardScaler().fit(raw_train.loc[:, ROUTER_FEATURE_NAMES].to_numpy(dtype=np.float64))
    transform = FeatureTransform(pca=pca, scaler=scaler)
    router_root = paths["cache_dir"] / "router"
    transform.save(router_root / "pca_scaler.joblib", overwrite=args.force)
    train_output = _standardize(raw_train, transform)
    validation_output = _standardize(raw_validation, transform)
    for output in (train_output, validation_output):
        forbidden = {column for column in output.columns if any(token in column.lower() for token in ("label", "mask", "future", "cost", "budget"))}
        if forbidden:
            raise ArtifactError(f"Serialized router input contains forbidden columns: {sorted(forbidden)}")
    write_parquet(train_output, router_root / "router_train.parquet", overwrite=args.force)
    write_parquet(validation_output, router_root / "router_validation.parquet", overwrite=args.force)
    atomic_write_json(
        router_root / "router_dataset_metadata.json",
        {
            "feature_schema": transform.schema,
            "schema_hash": transform.schema_hash,
            "feature_names": list(ROUTER_FEATURE_NAMES),
            "train_rows": len(train_output),
            "validation_rows": len(validation_output),
            "pca_fit_rows": len(train_embeddings),
            "no_future_or_label_input": True,
        },
        overwrite=args.force,
    )
    print(f"Wrote 53-D router datasets: {len(train_output)} train and {len(validation_output)} validation states")


if __name__ == "__main__":
    main()
