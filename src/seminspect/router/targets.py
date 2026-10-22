"""Cost-sensitive next-resolution risk targets used only during development."""

from __future__ import annotations

import numpy as np


def binary_decision_risk(label: int, probability: float, *, false_negative_weight: float = 10.0, false_positive_weight: float = 1.0, clip: float = 1e-6) -> float:
    """Natural-log weighted cross-entropy for a known development label."""

    if label not in {0, 1}:
        raise ValueError("Decision-risk label must be 0 or 1")
    probability = float(np.clip(probability, clip, 1.0 - clip))
    if label == 1:
        return float(-false_negative_weight * np.log(probability))
    return float(-false_positive_weight * np.log(1.0 - probability))


def next_resolution_gain(
    label: int,
    current_probability: float,
    next_probability: float,
    *,
    false_negative_weight: float = 10.0,
    false_positive_weight: float = 1.0,
    clip_target: tuple[float, float] = (-5.0, 5.0),
    helpful_delta: float = 0.01,
) -> tuple[float, int]:
    """Return the clipped signed gain and the auxiliary helpful-acquisition label."""

    current = binary_decision_risk(
        label,
        current_probability,
        false_negative_weight=false_negative_weight,
        false_positive_weight=false_positive_weight,
    )
    next_risk = binary_decision_risk(
        label,
        next_probability,
        false_negative_weight=false_negative_weight,
        false_positive_weight=false_positive_weight,
    )
    gain = float(np.clip(current - next_risk, clip_target[0], clip_target[1]))
    return gain, int(gain > helpful_delta)
