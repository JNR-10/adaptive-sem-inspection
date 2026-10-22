"""Required linear Value-of-Resolution router baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from seminspect.utils.io import ArtifactError


@dataclass
class RidgeRouter:
    model: Ridge
    alpha: float
    validation_mae: float

    @classmethod
    def fit_select(
        cls,
        train_features: np.ndarray,
        train_gain: np.ndarray,
        validation_features: np.ndarray,
        validation_gain: np.ndarray,
        alphas: tuple[float, ...] | list[float] = (0.01, 0.1, 1.0, 10.0),
    ) -> "RidgeRouter":
        candidates: list[tuple[float, float, Ridge]] = []
        for alpha in sorted(float(value) for value in alphas):
            model = Ridge(alpha=alpha, fit_intercept=True, solver="svd")
            model.fit(train_features, train_gain)
            mae = float(mean_absolute_error(validation_gain, model.predict(validation_features)))
            candidates.append((mae, alpha, model))
        # Lowest MAE; an exact tie chooses larger alpha.
        mae, alpha, model = min(candidates, key=lambda item: (item[0], -item[1]))
        return cls(model=model, alpha=alpha, validation_mae=mae)

    def predict_gain(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(features), dtype=np.float64)

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise ArtifactError(f"Refusing to overwrite Ridge router: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.name}.tmp")
        temporary.unlink(missing_ok=True)
        joblib.dump(self, temporary)
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "RidgeRouter":
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise ArtifactError("Router artifact is not a RidgeRouter")
        return value
