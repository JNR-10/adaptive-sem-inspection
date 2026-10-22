"""Quality-specific synthetic-pair score calibration and reliability metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from seminspect.utils.io import ArtifactError


def clip_probability(values: np.ndarray | float, clip: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), clip, 1.0 - clip)


def expected_calibration_error(probability: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(probability) != len(labels):
        raise ValueError("probability and labels must have equal length")
    total = len(labels)
    if total == 0:
        return float("nan")
    output = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        included = (probability >= lower) & ((probability < upper) if index < bins - 1 else (probability <= upper))
        if included.any():
            output += included.mean() * abs(float(probability[included].mean()) - float(labels[included].mean()))
    return float(output)


def reliability_bins(probability: np.ndarray, labels: np.ndarray, bins: int = 15) -> list[dict[str, float | int]]:
    """Return all fixed equal-width reliability bins, including empty ones."""

    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(probability) != len(labels):
        raise ValueError("probability and labels must have equal length")
    rows: list[dict[str, float | int]] = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        included = (probability >= lower) & ((probability < upper) if index < bins - 1 else (probability <= upper))
        count = int(included.sum())
        rows.append(
            {
                "bin_index": index,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_probability": float(probability[included].mean()) if count else float("nan"),
                "empirical_rate": float(labels[included].mean()) if count else float("nan"),
            }
        )
    return rows


@dataclass
class ScoreCalibrator:
    model: Pipeline
    probability_clip: float = 1e-6

    @classmethod
    def fit(
        cls,
        scores: np.ndarray,
        labels: np.ndarray,
        *,
        C: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-8,
        probability_clip: float = 1e-6,
    ) -> "ScoreCalibrator":
        scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(scores) != 524 or len(labels) != 524:
            raise ValueError("Each core_v1 score calibrator must be fit on exactly 524 synthetic-pair rows")
        if set(labels) != {0, 1} or int((labels == 0).sum()) != 262 or int((labels == 1).sum()) != 262:
            raise ValueError("Score calibration must contain 262 originals and 262 synthetic anomalies")
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=C,
                        solver="lbfgs",
                        max_iter=max_iter,
                        tol=tol,
                        fit_intercept=True,
                        class_weight=None,
                    ),
                ),
            ]
        )
        model.fit(scores, labels)
        return cls(model=model, probability_clip=probability_clip)

    def predict(self, scores: np.ndarray | float) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        return clip_probability(self.model.predict_proba(values)[:, 1], self.probability_clip)

    def diagnostics(self, scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        probability = self.predict(scores)
        labels = np.asarray(labels, dtype=np.int64)
        return {
            "brier": float(np.mean((probability - labels) ** 2)),
            "ece_15": expected_calibration_error(probability, labels, bins=15),
        }

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise ArtifactError(f"Refusing to overwrite calibrator: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.name}.tmp")
        temporary.unlink(missing_ok=True)
        joblib.dump(self, temporary)
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ScoreCalibrator":
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise ArtifactError(f"Unexpected calibrator artifact type: {type(value)!r}")
        return value
