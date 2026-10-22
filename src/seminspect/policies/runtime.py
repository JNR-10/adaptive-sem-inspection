"""Assemble policy-only observations from cached detector features.

This module is the boundary that ensures allocation policies receive only
current/historical observations. Labels are accepted solely when explicitly
constructing the non-deployable oracle table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from seminspect.policies.allocator import Observation, ObservationTable
from seminspect.router.features import FeatureTransform, RouterObservation, raw_router_features
from seminspect.router.targets import next_resolution_gain


class GainPredictor(Protocol):
    def predict_gain(self, features: np.ndarray) -> np.ndarray: ...


class EmbeddingReader:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._cache: dict[str, np.ndarray] = {}

    def read(self, shard: str, row: int) -> np.ndarray:
        if shard not in self._cache:
            with np.load(self.root / shard, allow_pickle=False) as payload:
                self._cache[shard] = np.asarray(payload["cls_embeddings"], dtype=np.float32)
        return self._cache[shard][int(row)]


def _router_observation(row: pd.Series, reader: EmbeddingReader) -> RouterObservation:
    values = [
        "calibrated_probability",
        "detector_raw_score",
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
    ]
    if any(pd.isna(row[name]) for name in values):
        raise ValueError("Policy observations require calibrated, scored features at all quality levels")
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


def observation_table_from_features(
    features: pd.DataFrame,
    *,
    feature_root: str | Path,
    transform: FeatureTransform | None = None,
    predictor: GainPredictor | None = None,
    feature_keep_mask: np.ndarray | None = None,
    oracle_labels: Mapping[str, int] | None = None,
) -> ObservationTable:
    """Make all Q0–Q3 policy observations, predicting only local Q0–Q2 gains."""

    required = {"image_id", "quality_index", "calibrated_probability", "embedding_shard", "embedding_row"}
    if not required.issubset(features.columns):
        raise ValueError(f"Feature table lacks policy fields: {required - set(features.columns)}")
    if (predictor is None) != (transform is None):
        raise ValueError("Router predictor and PCA/scaler transform must be provided together")
    if feature_keep_mask is not None and predictor is None:
        raise ValueError("A feature_keep_mask is meaningful only for a router predictor")
    keep_mask = None
    if feature_keep_mask is not None:
        keep_mask = np.asarray(feature_keep_mask, dtype=bool)
        if keep_mask.shape != (53,):
            raise ValueError("feature_keep_mask must be a length-53 router schema mask")
    reader = EmbeddingReader(feature_root)
    observations: list[Observation] = []
    for image_id, group in features.groupby("image_id", sort=True):
        group = group.sort_values("quality_index", kind="mergesort").reset_index(drop=True)
        if group["quality_index"].tolist() != [0, 1, 2, 3]:
            raise ValueError(f"Policy image {image_id} lacks an exact Q0–Q3 ladder")
        router_values = [_router_observation(group.loc[index], reader) for index in range(4)]
        for quality in range(4):
            predicted_gain: float | None = None
            oracle_gain: float | None = None
            if quality < 3:
                if predictor is not None and transform is not None:
                    raw = raw_router_features(
                        router_values[quality],
                        quality_index=quality,
                        pca=transform.pca,
                        previous=router_values[quality - 1] if quality else None,
                    )
                    transformed = transform.transform(raw)
                    if keep_mask is not None:
                        # The standardized value zero is the train-set mean, so
                        # masking preserves the fixed 53-D architecture while
                        # making the removed feature group uninformative.
                        transformed[:, ~keep_mask] = 0.0
                    predicted_gain = float(predictor.predict_gain(transformed)[0])
                if oracle_labels is not None:
                    if str(image_id) not in oracle_labels:
                        raise ValueError(f"Oracle label is missing for {image_id}")
                    oracle_gain, _ = next_resolution_gain(
                        int(oracle_labels[str(image_id)]),
                        router_values[quality].calibrated_probability,
                        router_values[quality + 1].calibrated_probability,
                    )
            current = router_values[quality]
            observations.append(
                Observation(
                    image_id=str(image_id),
                    quality=quality,
                    calibrated_probability=current.calibrated_probability,
                    detector_raw_score=current.detector_raw_score,
                    predicted_gain=predicted_gain,
                    oracle_gain=oracle_gain,
                )
            )
    return ObservationTable(observations)


def virtual_lot(table: ObservationTable, source_ids: list[str]) -> tuple[ObservationTable, dict[str, str]]:
    """Clone observations under unique virtual IDs for one sampled prevalence lot.

    A final-test ID can appear in different seeded lots but not twice within one
    lot. The cloning keeps allocator state separate and still gives the policy no
    label, mask, or future observation.
    """

    if len(source_ids) != len(set(source_ids)):
        raise ValueError("A prevalence lot must sample without replacement within the lot")
    rows: list[Observation] = []
    reverse: dict[str, str] = {}
    for index, source_id in enumerate(source_ids):
        virtual_id = f"lot_{index:04d}:{source_id}"
        reverse[virtual_id] = source_id
        for quality in range(4):
            observation = table.get(source_id, quality)
            rows.append(
                Observation(
                    image_id=virtual_id,
                    quality=quality,
                    calibrated_probability=observation.calibrated_probability,
                    detector_raw_score=observation.detector_raw_score,
                    predicted_gain=observation.predicted_gain,
                    oracle_gain=observation.oracle_gain,
                )
            )
    return ObservationTable(rows), reverse
