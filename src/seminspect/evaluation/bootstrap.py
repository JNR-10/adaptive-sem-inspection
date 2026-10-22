"""Paired label-stratified bootstrap intervals from a single frozen final run."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    replicates: int


def _aligned(first: pd.DataFrame, second: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = {"image_id", "label", "prediction"}
    if not columns.issubset(first.columns) or not columns.issubset(second.columns):
        raise ValueError("Bootstrap inputs require image_id, label, and prediction")
    left = first.sort_values("image_id", kind="mergesort").reset_index(drop=True)
    right = second.sort_values("image_id", kind="mergesort").reset_index(drop=True)
    if not left["image_id"].equals(right["image_id"]) or not left["label"].equals(right["label"]):
        raise ValueError("Paired bootstrap policies must share identical image IDs and labels")
    return left, right


def paired_stratified_recall_difference(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    replicates: int = 10000,
    seed: int = 20260917,
) -> BootstrapInterval:
    """Compute percentile CI for recall(first) − recall(second).

    Normals are deliberately resampled too, even though they do not change a
    recall numerator; that preserves the exact specified label-stratified sample
    construction for later paired metrics.
    """

    left, right = _aligned(first, second)
    labels = left["label"].to_numpy(dtype=np.int8)
    first_prediction = left["prediction"].to_numpy(dtype=np.int8)
    second_prediction = right["prediction"].to_numpy(dtype=np.int8)
    normal = np.flatnonzero(labels == 0)
    anomaly = np.flatnonzero(labels == 1)
    if len(normal) != 1272 or len(anomaly) != 116:
        raise ValueError("core_v1 paired bootstrap requires exactly 1,272 normals and 116 anomalies")
    rng = np.random.Generator(np.random.PCG64(seed))
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        # Build full resample even though recall uses only anomaly elements.
        normal_draw = rng.choice(normal, size=len(normal), replace=True)
        anomaly_draw = rng.choice(anomaly, size=len(anomaly), replace=True)
        _ = normal_draw  # retained to document/guarantee the prescribed RNG stream.
        values[index] = float(first_prediction[anomaly_draw].mean() - second_prediction[anomaly_draw].mean())
    point = float(first_prediction[anomaly].mean() - second_prediction[anomaly].mean())
    lower, upper = np.percentile(values, [2.5, 97.5], method="linear")
    return BootstrapInterval(point, float(lower), float(upper), replicates)
