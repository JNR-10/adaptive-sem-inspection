"""Finite-sample FPR-controlled policy-specific decision thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Threshold:
    value: float
    alpha: float
    normal_count: int
    rank_one_based: int
    selected_image_id: str
    tie_count: int


def select_normal_fpr_threshold(normal_scores: Iterable[tuple[str, float]], alpha: float = 0.05) -> Threshold:
    """Pick `s_(ceil((n+1)(1-alpha)))`; predictions are later strictly `>` this value."""

    rows = [(str(image_id), float(score)) for image_id, score in normal_scores]
    if not rows:
        raise ValueError("At least one calibration-normal score is required")
    if not 0 < alpha < 1:
        raise ValueError("FPR alpha must be in (0, 1)")
    if not np.isfinite([score for _, score in rows]).all():
        raise ValueError("Calibration scores must be finite")
    rows.sort(key=lambda item: (item[1], item[0]))
    count = len(rows)
    rank = min(int(np.ceil((count + 1) * (1.0 - alpha))), count)
    selected_id, value = rows[rank - 1]
    tie_count = sum(score == value for _, score in rows)
    return Threshold(value, alpha, count, rank, selected_id, tie_count)


def predict_with_threshold(scores: np.ndarray | list[float], threshold: Threshold | float) -> np.ndarray:
    value = threshold.value if isinstance(threshold, Threshold) else float(threshold)
    return np.asarray(scores, dtype=np.float64) > value
