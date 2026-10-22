"""Quality-specific reservoir memory banks and exact chunked cosine search."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from seminspect.utils.device import require_torch
from seminspect.utils.io import ArtifactError


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Every memory/query vector must have positive L2 norm")
    return values / norms


class ReservoirSampler:
    """Vitter Algorithm R with the exact zero-based rule locked in core_v1."""

    def __init__(self, capacity: int, embedding_dim: int, seed: int) -> None:
        if capacity <= 0 or embedding_dim <= 0:
            raise ValueError("Reservoir capacity and embedding dimension must be positive")
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.descriptors = np.empty((capacity, embedding_dim), dtype=np.float32)
        self.image_ids = np.empty(capacity, dtype="U128")
        self.coordinates = np.empty((capacity, 2), dtype=np.int16)
        self.count = 0

    def add(self, descriptors: np.ndarray, image_id: str, coordinates: np.ndarray) -> None:
        values = _normalize_rows(descriptors)
        coords = np.asarray(coordinates, dtype=np.int16)
        if values.ndim != 2 or values.shape[1] != self.embedding_dim:
            raise ValueError("Descriptor shape does not match reservoir embedding dimension")
        if coords.shape != (len(values), 2):
            raise ValueError("Coordinates must have one row/column pair per descriptor")
        for descriptor, coordinate in zip(values, coords, strict=True):
            t = self.count
            if t < self.capacity:
                location = t
            else:
                candidate = int(self.rng.integers(0, t + 1))
                location = candidate if candidate < self.capacity else None
            if location is not None:
                self.descriptors[location] = descriptor
                self.image_ids[location] = str(image_id)
                self.coordinates[location] = coordinate
            self.count += 1

    def finish(self) -> "MemoryBank":
        if self.count < self.capacity:
            raise ArtifactError(f"Reservoir stream has only {self.count} entries; needs {self.capacity}")
        return MemoryBank(
            descriptors=self.descriptors.astype(np.float16),
            image_ids=self.image_ids.copy(),
            coordinates=self.coordinates.copy(),
        )


@dataclass
class MemoryBank:
    descriptors: np.ndarray
    image_ids: np.ndarray
    coordinates: np.ndarray

    def __post_init__(self) -> None:
        self.descriptors = np.asarray(self.descriptors, dtype=np.float16)
        self.image_ids = np.asarray(self.image_ids, dtype="U128")
        self.coordinates = np.asarray(self.coordinates, dtype=np.int16)
        if self.descriptors.ndim != 2:
            raise ValueError("Memory descriptors must be a two-dimensional matrix")
        if len(self.descriptors) != len(self.image_ids) or self.coordinates.shape != (len(self.descriptors), 2):
            raise ValueError("Memory bank metadata lengths do not match descriptors")

    @property
    def size(self) -> int:
        return int(self.descriptors.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.descriptors.shape[1])

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            raise ArtifactError(f"Refusing to overwrite memory bank: {destination}")
        temporary = destination.with_name(f"{destination.name}.tmp")
        temporary.unlink(missing_ok=True)
        try:
            with temporary.open("xb") as handle:
                np.savez_compressed(
                    handle,
                    descriptors=self.descriptors,
                    image_ids=self.image_ids,
                    coordinates=self.coordinates,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "MemoryBank":
        with np.load(Path(path), allow_pickle=False) as values:
            return cls(values["descriptors"], values["image_ids"], values["coordinates"])

    def nearest_cosine_distances(
        self,
        query_descriptors: Any,
        *,
        device: str = "cuda",
        query_chunk: int = 1024,
        memory_chunk: int = 4096,
    ) -> Any:
        """Exact nearest-bank cosine distances, bounded by query/memory chunks."""

        torch = require_torch()
        import torch.nn.functional as functional

        query = query_descriptors.to(device=device, dtype=torch.float32)
        if query.ndim != 2 or query.shape[1] != self.dimension:
            raise ValueError("Query descriptor shape does not match memory bank")
        query = functional.normalize(query, dim=1)
        bank = torch.from_numpy(self.descriptors.astype(np.float32, copy=False)).to(device=device)
        bank = functional.normalize(bank, dim=1)
        result: list[Any] = []
        for start in range(0, query.shape[0], query_chunk):
            chunk = query[start : start + query_chunk]
            best = torch.full((chunk.shape[0],), -float("inf"), device=device, dtype=torch.float32)
            for memory_start in range(0, bank.shape[0], memory_chunk):
                similarities = chunk @ bank[memory_start : memory_start + memory_chunk].T
                best = torch.maximum(best, similarities.max(dim=1).values)
            result.append(1.0 - torch.clamp(best, min=-1.0, max=1.0))
        return torch.cat(result, dim=0)


def closest_cls_distance(
    query_embeddings: Any,
    normal_embeddings: Any,
    *,
    device: str = "cuda",
    query_chunk: int = 1024,
    memory_chunk: int = 4096,
) -> Any:
    """Exact per-query closest-normal CLS cosine distance with stable bank order."""

    torch = require_torch()
    import torch.nn.functional as functional

    query = functional.normalize(query_embeddings.to(device=device, dtype=torch.float32), dim=1)
    bank = functional.normalize(normal_embeddings.to(device=device, dtype=torch.float32), dim=1)
    output: list[Any] = []
    for start in range(0, query.shape[0], query_chunk):
        values = query[start : start + query_chunk]
        best = torch.full((values.shape[0],), -float("inf"), device=device, dtype=torch.float32)
        for memory_start in range(0, bank.shape[0], memory_chunk):
            similarity = values @ bank[memory_start : memory_start + memory_chunk].T
            best = torch.maximum(best, similarity.max(dim=1).values)
        output.append(1.0 - torch.clamp(best, min=-1.0, max=1.0))
    return torch.cat(output, dim=0)


def descriptor_coordinates(grid: tuple[int, int]) -> np.ndarray:
    rows, columns = grid
    return np.asarray([(row, column) for row in range(rows) for column in range(columns)], dtype=np.int16)
