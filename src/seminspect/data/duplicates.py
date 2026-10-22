"""Deterministic duplicate-audit helpers for leakage prevention."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from seminspect.data.hashing import canonical_float448, decode_grayscale


class DuplicateAuditError(RuntimeError):
    """Raised when an audit detects an unresolved leakage component."""


def hamming64(first: int, second: int) -> int:
    return (int(first) ^ int(second)).bit_count()


class BKTree:
    """Small deterministic BK-tree for 64-bit perceptual hashes."""

    def __init__(self) -> None:
        self._root: tuple[int, str, dict[int, tuple]] | None = None

    def add(self, value: int, identifier: str) -> None:
        value = int(value)
        node = self._root
        if node is None:
            self._root = (value, identifier, {})
            return
        while True:
            node_value, node_id, children = node
            distance = hamming64(value, node_value)
            if distance == 0 and identifier == node_id:
                return
            child = children.get(distance)
            if child is None:
                children[distance] = (value, identifier, {})
                return
            node = child

    def query(self, value: int, radius: int) -> list[str]:
        if self._root is None:
            return []
        hits: list[str] = []
        stack = [self._root]
        while stack:
            node_value, identifier, children = stack.pop()
            distance = hamming64(value, node_value)
            if distance <= radius:
                hits.append(identifier)
            lower, upper = distance - radius, distance + radius
            for edge in sorted(children, reverse=True):
                if lower <= edge <= upper:
                    stack.append(children[edge])
        return sorted(hits)


@dataclass(frozen=True)
class PairEvidence:
    first_id: str
    second_id: str
    source: str
    phash_distance: int | None = None
    cosine_similarity: float | None = None
    ssim: float | None = None


class UnionFind:
    def __init__(self, identifiers: Iterable[str]) -> None:
        self.parent = {item: item for item in identifiers}

    def find(self, item: str) -> str:
        root = self.parent[item]
        while root != self.parent[root]:
            root = self.parent[root]
        while item != root:
            next_item = self.parent[item]
            self.parent[item] = root
            item = next_item
        return root

    def union(self, first: str, second: str) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            # Stable root choice keeps a component independent of candidate order.
            self.parent[max(first_root, second_root)] = min(first_root, second_root)

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for identifier in sorted(self.parent):
            groups[self.find(identifier)].append(identifier)
        return {root: sorted(items) for root, items in groups.items()}


def exact_duplicate_pairs(frame: pd.DataFrame) -> list[PairEvidence]:
    """Return every decoded-pixel-identical image pair in a stable order.

    MIIC's repeated SEM textures make pHash plus SSIM an over-sensitive proxy
    for leakage: visually similar but distinct fields can form very large
    components.  Decoded-pixel identity is unambiguous and still captures the
    published MIIC archive overlaps and reused inpainting source images.
    """

    required = {"image_id", "pixel_sha256"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DuplicateAuditError(f"Exact duplicate audit is missing columns: {missing}")
    pairs: list[PairEvidence] = []
    for _, group in frame.groupby("pixel_sha256", dropna=False, sort=True):
        identifiers = sorted(group["image_id"].astype(str).tolist())
        for first, second in combinations(identifiers, 2):
            pairs.append(PairEvidence(first, second, "pixel_sha256"))
    return pairs


def phash_candidate_pairs(frame: pd.DataFrame, radius: int = 4) -> list[PairEvidence]:
    """Find pHash candidates using fixed insertion and result ordering."""

    tree = BKTree()
    values = frame[["image_id", "phash64"]].sort_values("image_id", kind="mergesort")
    for row in values.itertuples(index=False):
        tree.add(int(row.phash64), str(row.image_id))
    hashes = {str(row.image_id): int(row.phash64) for row in values.itertuples(index=False)}
    pairs: list[PairEvidence] = []
    seen: set[tuple[str, str]] = set()
    for identifier in sorted(hashes):
        for candidate in tree.query(hashes[identifier], radius):
            if candidate == identifier:
                continue
            first, second = sorted((identifier, candidate))
            if (first, second) in seen:
                continue
            seen.add((first, second))
            pairs.append(
                PairEvidence(first, second, "phash", phash_distance=hamming64(hashes[first], hashes[second]))
            )
    return sorted(pairs, key=lambda item: (item.first_id, item.second_id, item.source))


def cls_candidate_pairs(
    image_ids: list[str],
    embeddings: np.ndarray,
    *,
    top_k: int = 20,
    block_size: int = 256,
    memory_chunk: int = 4096,
    device: str | None = None,
) -> list[PairEvidence]:
    """Compute exact blockwise cosine-neighbor candidates without an all-pairs matrix.

    When a CUDA device is selected, this performs exact GPU chunked matrix
    search and merges candidates by `(similarity descending, image_id ascending)`
    using stable sorts. It never materializes an all-pairs matrix or relies on an
    unstable `topk` tie boundary. The NumPy path is a deterministic CPU fallback.
    """

    if len(image_ids) != len(embeddings):
        raise ValueError("image_ids and embeddings must have the same length")
    if not image_ids:
        return []
    ids = [str(item) for item in image_ids]
    if ids != sorted(ids):
        raise ValueError("CLS candidates require image_ids in sorted order")
    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("CLS embeddings must be nonzero")
    vectors = vectors / norms
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise DuplicateAuditError("CUDA CLS duplicate audit was requested but no CUDA device is available")
            return _cls_candidate_pairs_gpu(ids, vectors, top_k=top_k, query_chunk=block_size, memory_chunk=memory_chunk)
        except ImportError as exc:
            raise DuplicateAuditError("PyTorch is required for CUDA CLS duplicate audit") from exc
    output: dict[tuple[str, str], float] = {}
    count = len(ids)
    # Each row is processed in blocks; no NxN matrix is materialized.
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        similarities = vectors[start:stop] @ vectors.T
        for local, row in enumerate(similarities):
            index = start + local
            ranked = sorted(
                ((float(score), ids[other]) for other, score in enumerate(row) if other != index),
                key=lambda item: (-item[0], item[1]),
            )[:top_k]
            for score, other_id in ranked:
                if score < 0.995:
                    continue
                first, second = sorted((ids[index], other_id))
                output[(first, second)] = max(output.get((first, second), -np.inf), score)
    return [PairEvidence(first, second, "cls", cosine_similarity=score) for (first, second), score in sorted(output.items())]


def _cls_candidate_pairs_gpu(
    ids: list[str], vectors: np.ndarray, *, top_k: int, query_chunk: int, memory_chunk: int
) -> list[PairEvidence]:
    """Exact CUDA neighbor candidate generation with deterministic stable merge."""

    import torch

    total = len(ids)
    values = torch.from_numpy(vectors).to(device="cuda", dtype=torch.float32)
    values = torch.nn.functional.normalize(values, dim=1)
    keep = top_k + 1  # retain self plus 20 non-self candidates
    all_indices = torch.arange(total, device="cuda", dtype=torch.long)
    output: dict[tuple[str, str], float] = {}
    for start in range(0, total, query_chunk):
        query = values[start : start + query_chunk]
        rows = query.shape[0]
        candidate_scores = torch.full((rows, keep), -float("inf"), device="cuda", dtype=torch.float32)
        candidate_indices = torch.full((rows, keep), total, device="cuda", dtype=torch.long)
        for memory_start in range(0, total, memory_chunk):
            end = min(memory_start + memory_chunk, total)
            scores = query @ values[memory_start:end].T
            indices = all_indices[memory_start:end].expand(rows, -1)
            merged_scores = torch.cat([candidate_scores, scores], dim=1)
            merged_indices = torch.cat([candidate_indices, indices], dim=1)
            # First sort IDs ascending, then stable-sort scores descending; ties retain ID order.
            id_order = torch.argsort(merged_indices, dim=1, stable=True)
            ordered_scores = torch.gather(merged_scores, 1, id_order)
            ordered_indices = torch.gather(merged_indices, 1, id_order)
            score_order = torch.argsort(ordered_scores, dim=1, descending=True, stable=True)[:, :keep]
            candidate_scores = torch.gather(ordered_scores, 1, score_order)
            candidate_indices = torch.gather(ordered_indices, 1, score_order)
        candidate_scores_cpu = candidate_scores.cpu().numpy()
        candidate_indices_cpu = candidate_indices.cpu().numpy()
        for local in range(rows):
            source_index = start + local
            selected = 0
            for score, other_index in zip(candidate_scores_cpu[local], candidate_indices_cpu[local], strict=True):
                if other_index == source_index or other_index >= total:
                    continue
                if float(score) < 0.995:
                    break
                first, second = sorted((ids[source_index], ids[int(other_index)]))
                output[(first, second)] = max(output.get((first, second), -np.inf), float(score))
                selected += 1
                if selected == top_k:
                    break
    return [PairEvidence(first, second, "cls", cosine_similarity=score) for (first, second), score in sorted(output.items())]


def _path_lookup(frame: pd.DataFrame, root: Path) -> Mapping[str, Path]:
    return {str(row.image_id): root / str(row.relative_path) for row in frame.itertuples(index=False)}


def confirm_candidates(
    frame: pd.DataFrame,
    root: str | Path,
    candidates: Iterable[PairEvidence],
    *,
    ssim_threshold: float = 0.98,
) -> list[PairEvidence]:
    """Confirm candidates using protocol-fixed Q3 SSIM settings."""

    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise DuplicateAuditError("scikit-image is required to confirm near duplicates by SSIM") from exc
    paths = _path_lookup(frame, Path(root))
    cache: dict[str, np.ndarray] = {}

    def image(identifier: str) -> np.ndarray:
        if identifier not in cache:
            pixels, _ = decode_grayscale(paths[identifier])
            cache[identifier] = canonical_float448(pixels)
        return cache[identifier]

    confirmed: list[PairEvidence] = []
    for evidence in sorted(candidates, key=lambda item: (item.first_id, item.second_id, item.source)):
        score = float(
            structural_similarity(
                image(evidence.first_id),
                image(evidence.second_id),
                data_range=1.0,
                win_size=7,
                gaussian_weights=False,
            )
        )
        if score >= ssim_threshold:
            confirmed.append(
                PairEvidence(
                    evidence.first_id,
                    evidence.second_id,
                    evidence.source,
                    phash_distance=evidence.phash_distance,
                    cosine_similarity=evidence.cosine_similarity,
                    ssim=score,
                )
            )
    return confirmed


def assign_duplicate_groups(frame: pd.DataFrame, confirmed: Iterable[PairEvidence]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign lexicographic pixel-hash component IDs and return audit decisions."""

    result = frame.copy()
    ids = result["image_id"].astype(str).tolist()
    union_find = UnionFind(ids)
    confirmed_list = list(confirmed)
    for evidence in confirmed_list:
        union_find.union(evidence.first_id, evidence.second_id)
    rows_by_id = result.set_index("image_id", drop=False)
    group_for_id: dict[str, str] = {}
    for members in union_find.components().values():
        representative = min(str(rows_by_id.loc[item, "pixel_sha256"]) for item in members)
        for identifier in members:
            group_for_id[identifier] = representative
    result["duplicate_group"] = result["image_id"].map(group_for_id)
    evidence_rows = [
        {
            "first_id": item.first_id,
            "second_id": item.second_id,
            "candidate_source": item.source,
            "phash_distance": item.phash_distance,
            "cosine_similarity": item.cosine_similarity,
            "ssim": item.ssim,
            "confirmed": True,
        }
        for item in confirmed_list
    ]
    decisions = pd.DataFrame(evidence_rows)
    return result, decisions


def assert_no_cross_partition_components(frame: pd.DataFrame) -> None:
    """Hard-fail any duplicate component spanning distinct supervised partitions."""

    active = frame.loc[~frame["quarantined"].astype(bool)].copy()
    for component, group in active.groupby("duplicate_group", dropna=False, sort=True):
        splits = sorted(set(group["split"].dropna().astype(str)) - {"unassigned", "normal_reserve"})
        pair_ids = sorted(set(group["pair_id"].dropna().astype(str)))
        final = (group["split"] == "final_test").any()
        if len(splits) > 1:
            raise DuplicateAuditError(f"Duplicate component {component} crosses supervised partitions: {splits}")
        if final and any(split != "final_test" for split in splits):
            raise DuplicateAuditError(f"Duplicate component {component} crosses final test and development")


def quarantine_duplicate_reference_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove only development-side detector references, never test images.

    Call this before fixed normal splitting. Exact and confirmed duplicate
    components linking training normals to an inpainting or final-test image are
    fully quarantined on the training side. Within an all-training component,
    retain its lexicographically minimal `(pixel_sha256, image_id)` member.
    """

    result = frame.copy()
    train_mask = result["source_archive"].eq("Anomaly_train.rar") & ~result["quarantined"].astype(bool)
    for _, group in result.groupby("duplicate_group", dropna=False, sort=True):
        train = group.loc[train_mask.loc[group.index]]
        if train.empty:
            continue
        other = group.loc[~train_mask.loc[group.index]]
        if not other.empty:
            result.loc[train.index, ["quarantined", "quarantine_reason"]] = [
                True,
                "near_duplicate_with_inpainting_or_final_test",
            ]
            continue
        keep = min(
            ((str(row.pixel_sha256), str(row.image_id), int(index)) for index, row in train.iterrows()),
            key=lambda item: (item[0], item[1]),
        )[2]
        remove = [index for index in train.index if index != keep]
        if remove:
            result.loc[remove, ["quarantined", "quarantine_reason"]] = [
                True,
                "near_duplicate_detector_reference",
            ]
    return result
