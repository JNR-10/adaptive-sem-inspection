"""The primary resolution-only Q0–Q3 acquisition simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps

from seminspect.utils.device import require_torch


@dataclass(frozen=True, order=True)
class Quality:
    index: int
    id: str
    effective_side: int
    cumulative_cost: float


class QualityLadder:
    """Validated quality IDs and cumulative acquisition costs."""

    def __init__(self, qualities: Iterable[Quality]) -> None:
        self._qualities = tuple(sorted(qualities))
        if [item.index for item in self._qualities] != [0, 1, 2, 3]:
            raise ValueError("Quality indices must be exactly 0, 1, 2, 3")
        if [item.id for item in self._qualities] != ["Q0", "Q1", "Q2", "Q3"]:
            raise ValueError("Quality IDs must be Q0, Q1, Q2, Q3")
        if [item.effective_side for item in self._qualities] != [56, 112, 224, 448]:
            raise ValueError("Quality effective sides must be 56, 112, 224, 448")
        costs = [item.cumulative_cost for item in self._qualities]
        if any(next_cost <= current for current, next_cost in zip(costs, costs[1:])):
            raise ValueError("Cumulative quality costs must strictly increase")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QualityLadder":
        return cls(
            Quality(
                index=int(item["index"]),
                id=str(item["id"]),
                effective_side=int(item["effective_side"]),
                cumulative_cost=float(item["cumulative_cost"]),
            )
            for item in config["qualities"]
        )

    def __iter__(self):
        return iter(self._qualities)

    def __len__(self) -> int:
        return len(self._qualities)

    def __getitem__(self, key: int | str) -> Quality:
        if isinstance(key, str):
            return next(item for item in self._qualities if item.id == key)
        return self._qualities[key]

    @property
    def q0_cost(self) -> float:
        return self._qualities[0].cumulative_cost

    def incremental_cost(self, index: int) -> float:
        if not 0 <= index < 3:
            raise ValueError("Only Q0→Q1, Q1→Q2, and Q2→Q3 upgrades have incremental cost")
        return self._qualities[index + 1].cumulative_cost - self._qualities[index].cumulative_cost


def canonical_q3_from_array(array: np.ndarray, canonical_size: int = 448) -> np.ndarray:
    """Make the exact benchmark Q3 observation: field-of-view resize only."""

    if array.ndim != 2:
        raise ValueError("Canonical Q3 source must be a two-dimensional grayscale image")
    if array.dtype != np.uint8:
        raise ValueError("Canonical Q3 source must be uint8 after canonical decoding")
    image = Image.fromarray(array, mode="L").resize(
        (canonical_size, canonical_size), Image.Resampling.LANCZOS
    )
    return np.ascontiguousarray(np.asarray(image, dtype=np.float32) / np.float32(255.0))


def canonical_q3_from_path(path: str | Path, canonical_size: int = 448) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    return canonical_q3_from_array(array, canonical_size=canonical_size)


def simulate_resolution(q3: np.ndarray, quality: Quality, canonical_size: int = 448) -> np.ndarray:
    """Generate a CPU-float32 448×448 observation from the single canonical Q3.

    Q0–Q2 use `torch.nn.functional.interpolate` exactly as specified: area down
    sampling followed by bicubic upsampling, no random degradation, and no
    accelerator-dependent input construction.
    """

    if q3.shape != (canonical_size, canonical_size) or q3.dtype != np.float32:
        raise ValueError("q3 must be a float32 canonical_size×canonical_size field")
    if quality.effective_side == canonical_size:
        return np.ascontiguousarray(q3.copy())
    torch = require_torch()
    import torch.nn.functional as functional

    source = torch.from_numpy(np.ascontiguousarray(q3))[None, None].to(device="cpu", dtype=torch.float32)
    down = functional.interpolate(source, size=(quality.effective_side, quality.effective_side), mode="area")
    up = functional.interpolate(
        down,
        size=(canonical_size, canonical_size),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    )
    observation = up.squeeze(0).squeeze(0).contiguous().numpy().astype(np.float32, copy=False)
    return np.ascontiguousarray(np.clip(observation, 0.0, 1.0))


def simulate_ladder(q3: np.ndarray, ladder: QualityLadder, canonical_size: int = 448) -> dict[str, np.ndarray]:
    return {quality.id: simulate_resolution(q3, quality, canonical_size) for quality in ladder}


def encoder_tensor(observation: np.ndarray, device: str = "cuda") -> Any:
    """Repeat unnormalized grayscale to RGB and apply the fixed ImageNet transform."""

    if observation.shape != (448, 448) or observation.dtype != np.float32:
        raise ValueError("Encoder observation must be a float32 448×448 quality observation")
    torch = require_torch()
    image = torch.from_numpy(np.ascontiguousarray(observation))[None, None].to(device=device, dtype=torch.float32)
    image = image.repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32)[None, :, None, None]
    return (image - mean) / std
