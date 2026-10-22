"""The locked 53-column current-observation router feature schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from seminspect.utils.io import ArtifactError, canonical_json_sha256


ROUTER_FEATURE_SCHEMA = "router_feature_schema_v1"
ROUTER_FEATURE_NAMES = tuple(
    [
        "calibrated_probability",
        "probability_logit",
        "binary_entropy",
        "detector_raw_score",
        "patch_score_max",
        "patch_score_top5_std",
        "patch_score_q95",
        "cls_normal_distance",
    ]
    + [f"cls_pca_{index:02d}" for index in range(32)]
    + [
        "intensity_mean",
        "intensity_std",
        "robust_dynamic_range",
        "laplacian_variance",
        "tenengrad",
        "highpass_mad",
        "quality_is_q0",
        "quality_is_q1",
        "quality_is_q2",
        "probability_delta",
        "raw_score_delta",
        "cls_cosine_change",
        "has_history",
    ]
)
FORBIDDEN_FEATURE_TOKENS = ("label", "mask", "future", "cost", "budget", "target")

if len(ROUTER_FEATURE_NAMES) != 53:
    raise RuntimeError(f"router_feature_schema_v1 must contain 53 features, got {len(ROUTER_FEATURE_NAMES)}")


@dataclass(frozen=True)
class RouterObservation:
    calibrated_probability: float
    detector_raw_score: float
    patch_score_max: float
    patch_score_top5_std: float
    patch_score_q95: float
    cls_normal_distance: float
    cls_embedding: np.ndarray
    intensity_mean: float
    intensity_std: float
    robust_dynamic_range: float
    laplacian_variance: float
    tenengrad: float
    highpass_mad: float

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "RouterObservation":
        embedding = np.asarray(row["cls_embedding"], dtype=np.float32)
        if embedding.shape != (384,):
            raise ValueError("Router observations require a 384-dimensional CLS embedding")
        values = {name: float(row[name]) for name in cls.__dataclass_fields__ if name != "cls_embedding"}
        return cls(cls_embedding=embedding, **values)


def _clip_probability(probability: float, clip: float = 1e-6) -> float:
    return float(np.clip(probability, clip, 1.0 - clip))


def _probability_features(probability: float) -> tuple[float, float, float]:
    p = _clip_probability(probability)
    return p, float(np.log(p / (1.0 - p))), float(-p * np.log(p) - (1.0 - p) * np.log(1.0 - p))


def raw_router_features(
    current: RouterObservation,
    *,
    quality_index: int,
    pca: PCA,
    previous: RouterObservation | None = None,
) -> np.ndarray:
    """Build an unstandardized, current-and-history-only feature vector."""

    if quality_index not in {0, 1, 2}:
        raise ValueError("The router is never called at Q3")
    probability, logit, entropy = _probability_features(current.calibrated_probability)
    projection = pca.transform(current.cls_embedding.reshape(1, -1))[0]
    if projection.shape != (32,):
        raise ValueError("Router PCA transform must output 32 values")
    if previous is None:
        probability_delta, raw_score_delta, cosine_change, has_history = 0.0, 0.0, 0.0, 0.0
    else:
        previous_probability = _clip_probability(previous.calibrated_probability)
        denominator = float(np.linalg.norm(current.cls_embedding) * np.linalg.norm(previous.cls_embedding))
        if denominator <= 0:
            raise ValueError("CLS embeddings must have nonzero norm")
        cosine = float(np.clip(np.dot(current.cls_embedding, previous.cls_embedding) / denominator, -1.0, 1.0))
        probability_delta = probability - previous_probability
        raw_score_delta = current.detector_raw_score - previous.detector_raw_score
        cosine_change = 1.0 - cosine
        has_history = 1.0
    quality_one_hot = [float(quality_index == value) for value in range(3)]
    values = np.asarray(
        [
            probability,
            logit,
            entropy,
            current.detector_raw_score,
            current.patch_score_max,
            current.patch_score_top5_std,
            current.patch_score_q95,
            current.cls_normal_distance,
            *projection.tolist(),
            current.intensity_mean,
            current.intensity_std,
            current.robust_dynamic_range,
            current.laplacian_variance,
            current.tenengrad,
            current.highpass_mad,
            *quality_one_hot,
            probability_delta,
            raw_score_delta,
            cosine_change,
            has_history,
        ],
        dtype=np.float64,
    )
    if values.shape != (53,) or not np.isfinite(values).all():
        raise ValueError("Router feature vector must be finite and have exactly 53 values")
    return values


def validate_feature_schema(names: tuple[str, ...] | list[str] = ROUTER_FEATURE_NAMES) -> None:
    if tuple(names) != ROUTER_FEATURE_NAMES:
        raise ValueError("Router feature names/order differs from router_feature_schema_v1")
    for name in names:
        lowered = name.lower()
        if any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS):
            raise ValueError(f"Forbidden router feature name: {name}")


@dataclass
class FeatureTransform:
    pca: PCA
    scaler: StandardScaler
    feature_names: tuple[str, ...] = ROUTER_FEATURE_NAMES
    schema: str = ROUTER_FEATURE_SCHEMA

    @classmethod
    def fit_pca(cls, train_embeddings: np.ndarray) -> PCA:
        """Fit only the 32-D PCA projection on router-train Q0–Q2 CLS vectors."""

        embeddings = np.asarray(train_embeddings, dtype=np.float64)
        if embeddings.ndim != 2 or embeddings.shape[1] != 384:
            raise ValueError("PCA requires router-train Q0–Q2 CLS embeddings with width 384")
        pca = PCA(n_components=32, svd_solver="full", whiten=False)
        pca.fit(embeddings)
        return pca

    @classmethod
    def fit(cls, train_embeddings: np.ndarray, train_feature_matrix: np.ndarray) -> "FeatureTransform":
        """Fit PCA and scaler using router-train data only."""

        embeddings = np.asarray(train_embeddings, dtype=np.float64)
        matrix = np.asarray(train_feature_matrix, dtype=np.float64)
        if embeddings.ndim != 2 or embeddings.shape[1] != 384:
            raise ValueError("PCA requires router-train Q0–Q2 CLS embeddings with width 384")
        if matrix.ndim != 2 or matrix.shape[1] != 53:
            raise ValueError("Scaler requires router-train feature matrix with width 53")
        pca = cls.fit_pca(embeddings)
        scaler = StandardScaler()
        scaler.fit(matrix)
        return cls(pca=pca, scaler=scaler)

    @property
    def schema_hash(self) -> str:
        return canonical_json_sha256({"schema": self.schema, "feature_names": list(self.feature_names)})

    def transform(self, raw_matrix: np.ndarray, names: tuple[str, ...] | list[str] = ROUTER_FEATURE_NAMES) -> np.ndarray:
        validate_feature_schema(names)
        matrix = np.asarray(raw_matrix, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError("Router transform input has an incompatible number of columns")
        return self.scaler.transform(matrix).astype(np.float32)

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise ArtifactError(f"Refusing to overwrite PCA/scaler artifact: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.name}.tmp")
        temporary.unlink(missing_ok=True)
        joblib.dump(self, temporary)
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "FeatureTransform":
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise ArtifactError(f"Unexpected feature-transform artifact: {type(value)!r}")
        validate_feature_schema(value.feature_names)
        if value.schema != ROUTER_FEATURE_SCHEMA:
            raise ArtifactError("Feature-transform schema does not match router_feature_schema_v1")
        return value
