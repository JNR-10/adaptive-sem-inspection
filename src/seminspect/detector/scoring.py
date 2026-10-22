"""Patch-memory anomaly-map aggregation and feature summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from seminspect.detector.memory_bank import MemoryBank, closest_cls_distance


@dataclass
class DetectorScore:
    raw_score: float
    patch_score_max: float
    patch_score_top5_std: float
    patch_score_q95: float
    cls_normal_distance: float
    anomaly_map: np.ndarray


def summarize_patch_distances(distances: np.ndarray, grid: tuple[int, int], top_k: int = 5) -> tuple[float, float, float, float, np.ndarray]:
    values = np.asarray(distances, dtype=np.float32).reshape(-1)
    if len(values) != grid[0] * grid[1]:
        raise ValueError("Patch distance count does not match declared grid")
    if top_k <= 0 or top_k > len(values):
        raise ValueError("top_k must be positive and no larger than patch count")
    sorted_values = np.sort(values)
    largest = sorted_values[-top_k:]
    return (
        float(largest.mean()),
        float(values.max()),
        float(largest.std(ddof=0)),
        float(np.percentile(values, 95, method="linear")),
        values.reshape(grid),
    )


def score_encoded_image(
    patch_descriptors: Any,
    cls_embedding: Any,
    *,
    patch_bank: MemoryBank,
    cls_reference: Any,
    patch_grid: tuple[int, int] = (16, 16),
    device: str = "cuda",
    query_chunk: int = 1024,
    memory_chunk: int = 4096,
    top_k: int = 5,
) -> DetectorScore:
    """Score one image with its matching quality-specific normal memory bank."""

    distances = patch_bank.nearest_cosine_distances(
        patch_descriptors,
        device=device,
        query_chunk=query_chunk,
        memory_chunk=memory_chunk,
    ).detach().cpu().numpy()
    cls_distance = closest_cls_distance(
        cls_embedding.reshape(1, -1),
        cls_reference,
        device=device,
        query_chunk=query_chunk,
        memory_chunk=memory_chunk,
    ).detach().cpu().numpy()[0]
    raw, maximum, top_std, q95, anomaly_map = summarize_patch_distances(distances, patch_grid, top_k=top_k)
    return DetectorScore(raw, maximum, top_std, q95, float(cls_distance), anomaly_map)


def score_encoded_batch(
    patch_descriptors: Any,
    cls_embeddings: Any,
    *,
    patch_bank: MemoryBank,
    cls_reference: Any,
    patch_grid: tuple[int, int] = (16, 16),
    device: str = "cuda",
    query_chunk: int = 1024,
    memory_chunk: int = 4096,
    top_k: int = 5,
) -> list[DetectorScore]:
    """Score a batch while retaining exact search semantics and 16×16 maps."""

    if patch_descriptors.ndim != 3:
        raise ValueError("Patch descriptor batch must have shape [B,patches,dimension]")
    batch, patches, dimension = patch_descriptors.shape
    if patches != patch_grid[0] * patch_grid[1] or dimension != patch_bank.dimension:
        raise ValueError("Patch descriptor batch does not match detector grid or memory dimension")
    distances = patch_bank.nearest_cosine_distances(
        patch_descriptors.reshape(batch * patches, dimension),
        device=device,
        query_chunk=query_chunk,
        memory_chunk=memory_chunk,
    ).detach().cpu().numpy().reshape(batch, patches)
    cls = closest_cls_distance(
        cls_embeddings,
        cls_reference,
        device=device,
        query_chunk=query_chunk,
        memory_chunk=memory_chunk,
    ).detach().cpu().numpy()
    output: list[DetectorScore] = []
    for index in range(batch):
        raw, maximum, top_std, q95, anomaly_map = summarize_patch_distances(
            distances[index], patch_grid, top_k=top_k
        )
        output.append(DetectorScore(raw, maximum, top_std, q95, float(cls[index]), anomaly_map))
    return output
