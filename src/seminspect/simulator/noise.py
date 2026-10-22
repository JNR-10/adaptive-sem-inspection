"""Secondary deterministic SEM-motivated stress simulator."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class StressScenario:
    name: str
    poisson_peak: float | None
    contrast: float
    blur_sigma: float
    read_noise_sigma: float

    @classmethod
    def from_config(cls, name: str, values: dict[str, Any]) -> "StressScenario":
        peak = values.get("poisson_peak")
        return cls(
            name=name,
            poisson_peak=None if peak is None else float(peak),
            contrast=float(values["contrast"]),
            blur_sigma=float(values["blur_sigma"]),
            read_noise_sigma=float(values["read_noise_sigma"]),
        )


def stress_seed(image_id: str, quality_id: str, scenario: str, global_seed: int) -> int:
    payload = f"{image_id}|{quality_id}|{scenario}|{global_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def apply_stress(
    observation: np.ndarray,
    *,
    image_id: str,
    quality_id: str,
    scenario: StressScenario,
    global_seed: int,
) -> np.ndarray:
    """Apply the predeclared CPU-float64 perturbation sequence exactly once."""

    if observation.shape != (448, 448):
        raise ValueError("Stress inputs must be 448×448 resolution observations")
    x = np.asarray(observation, dtype=np.float64).copy()
    if np.nanmin(x) < 0 or np.nanmax(x) > 1:
        raise ValueError("Stress input must be in [0, 1]")
    rng = np.random.Generator(np.random.PCG64(stress_seed(image_id, quality_id, scenario.name, global_seed)))
    mean = float(x.mean())
    x = np.clip(mean + scenario.contrast * (x - mean), 0.0, 1.0)
    if scenario.blur_sigma != 0:
        x = gaussian_filter(x, sigma=scenario.blur_sigma, mode="reflect", truncate=4.0)
    if scenario.poisson_peak is not None and np.isfinite(scenario.poisson_peak):
        x = rng.poisson(x * scenario.poisson_peak) / scenario.poisson_peak
    if scenario.read_noise_sigma != 0:
        x = x + rng.normal(0.0, scenario.read_noise_sigma, size=x.shape)
    return np.ascontiguousarray(np.clip(x, 0.0, 1.0).astype(np.float32))
