"""Leakage-safe, fixed MIIC partitions and policy-calibration inspection lot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from seminspect.data.duplicates import (
    UnionFind,
    assert_no_cross_partition_components,
    quarantine_duplicate_reference_candidates,
)
from seminspect.data.manifests import normalize_manifest


class SplitError(RuntimeError):
    """Raised when a declared deterministic split cannot be constructed."""


@dataclass(frozen=True)
class SplitArtifacts:
    manifest: pd.DataFrame
    policy_calibration_lot: pd.DataFrame
    exact_overlap_count: int
    eligible_normal_count: int


def _permuted(values: list[str], seed: int) -> list[str]:
    ordered = np.asarray(sorted(values), dtype=object)
    rng = np.random.default_rng(seed)
    return ordered[rng.permutation(len(ordered))].tolist()


def _set_split(frame: pd.DataFrame, ids: list[str], split: str) -> None:
    rows = frame["image_id"].isin(ids)
    if rows.sum() != len(ids):
        raise SplitError(f"Could not assign every selected ID to split {split}")
    frame.loc[rows, "split"] = split


def _pair_components(frame: pd.DataFrame) -> list[tuple[str, ...]]:
    """Return pair-level components induced by exact image-duplicate groups."""

    paired = frame.loc[frame["pair_id"].notna()].copy()
    pair_ids = sorted(paired["pair_id"].astype(str).unique().tolist())
    if not pair_ids:
        raise SplitError("MIIC inpainting pairs are missing")
    groups = UnionFind(pair_ids)
    for _, group in paired.groupby("duplicate_group", dropna=False, sort=True):
        members = sorted(group["pair_id"].dropna().astype(str).unique().tolist())
        for member in members[1:]:
            groups.union(members[0], member)
    return [tuple(items) for _, items in sorted(groups.components().items())]


def _take_components_exact(
    components: list[tuple[str, ...]], target: int
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """Select a deterministic component subset with exactly ``target`` pairs."""

    if target < 0:
        raise SplitError("Pair-component target must be non-negative")
    predecessors: dict[int, tuple[int, int] | None] = {0: None}
    for index, component in enumerate(components):
        size = len(component)
        for subtotal in sorted(predecessors, reverse=True):
            total = subtotal + size
            if total <= target and total not in predecessors:
                predecessors[total] = (subtotal, index)
    if target not in predecessors:
        sizes = sorted(len(component) for component in components)
        raise SplitError(f"Cannot form a leakage-safe inpainting split with {target} pairs from components {sizes}")
    selected_indices: set[int] = set()
    subtotal = target
    while subtotal:
        predecessor = predecessors[subtotal]
        if predecessor is None:  # pragma: no cover - defensive invariant
            raise SplitError("Broken pair-component reconstruction")
        subtotal, index = predecessor
        selected_indices.add(index)
    selected = [component for index, component in enumerate(components) if index in selected_indices]
    remainder = [component for index, component in enumerate(components) if index not in selected_indices]
    return selected, remainder


def _assign_pair_splits(frame: pd.DataFrame, seed: int, counts: dict[str, int]) -> None:
    """Assign whole exact-duplicate components to development pair splits.

    MIIC's inpainting archives reuse source images across their published
    train/test folders.  We therefore pool the non-test-contaminated pairs and
    split by duplicate-connected component rather than trusting the archive
    folder names as independent partitions.
    """

    components = _pair_components(frame)
    final_groups = set(frame.loc[frame["source_archive"].eq("Anomaly_test.rar"), "duplicate_group"].astype(str))
    blocked_pairs: set[str] = set()
    for component in components:
        component_groups = set(
            frame.loc[frame["pair_id"].isin(component), "duplicate_group"].astype(str)
        )
        if component_groups & final_groups:
            blocked_pairs.update(component)
    if blocked_pairs:
        blocked_rows = frame["pair_id"].isin(sorted(blocked_pairs))
        frame.loc[blocked_rows, ["quarantined", "quarantine_reason"]] = [
            True,
            "exact_duplicate_component_with_final_test",
        ]
    active_components = [component for component in components if not any(pair in blocked_pairs for pair in component)]
    shuffled = [
        active_components[index]
        for index in np.random.default_rng(seed).permutation(len(active_components)).tolist()
    ]
    assignments = (
        ("policy_calibration_anomalies", int(counts["policy_calibration_anomaly_pairs"])),
        ("router_validation", int(counts["router_validation_pairs"])),
        ("score_calibration", int(counts["score_calibration_pairs"])),
    )
    remainder = shuffled
    for split, target in assignments:
        selected, remainder = _take_components_exact(remainder, target)
        pair_ids = [pair for component in selected for pair in component]
        frame.loc[frame["pair_id"].isin(pair_ids), "split"] = split
    expected_train = int(counts["router_train_pairs"])
    if sum(len(component) for component in remainder) != expected_train:
        found = sum(len(component) for component in remainder)
        raise SplitError(
            f"Leakage-safe inpainting pool leaves {found} router-train pairs; expected {expected_train}"
        )
    train_pairs = [pair for component in remainder for pair in component]
    frame.loc[frame["pair_id"].isin(train_pairs), "split"] = "router_train"
    active_pairs = frame.loc[frame["pair_id"].notna() & ~frame["quarantined"].astype(bool)]
    observed = active_pairs.groupby("split")["pair_id"].nunique().to_dict()
    expected = {
        "router_train": expected_train,
        "score_calibration": int(counts["score_calibration_pairs"]),
        "router_validation": int(counts["router_validation_pairs"]),
        "policy_calibration_anomalies": int(counts["policy_calibration_anomaly_pairs"]),
    }
    if observed != expected:
        raise SplitError(f"Leakage-safe pair split cardinality mismatch: {observed} != {expected}")
    for pair_id, group in active_pairs.groupby("pair_id", sort=True):
        if len(group) != 2 or group["split"].nunique() != 1:
            raise SplitError(f"Pair {pair_id} was not assigned atomically")


def build_miic_splits(
    manifest: pd.DataFrame,
    *,
    seed: int,
    expected_train_after_exact_test: int,
    counts: dict[str, int],
) -> SplitArtifacts:
    """Apply exact overlap removal, duplicate quarantine, and fixed partitions.

    The explicit count check happens immediately after exact final-test subtraction
    so a changed upstream archive cannot be silently treated as ordinary data.
    """

    result = normalize_manifest(manifest)
    if result["duplicate_group"].isna().any():
        raise SplitError("Exact-duplicate groups must be assigned before fixed MIIC splitting")
    train = result["source_archive"].eq("Anomaly_train.rar")
    final = result["source_archive"].eq("Anomaly_test.rar")
    final_hashes = set(result.loc[final, "pixel_sha256"].astype(str))
    exact = train & result["pixel_sha256"].astype(str).isin(final_hashes)
    exact_overlap_count = int(exact.sum())
    result.loc[exact, ["quarantined", "quarantine_reason"]] = [True, "exact_final_test_overlap"]
    remainder = int((train & ~result["quarantined"].astype(bool)).sum())
    if remainder != expected_train_after_exact_test:
        raise SplitError(
            f"MIIC train minus final-test exact overlap is {remainder}; expected {expected_train_after_exact_test}"
        )
    # The published archive relationship establishes that the remaining train
    # images are normal references after all 116 abnormal test duplicates leave.
    result.loc[train & ~result["quarantined"].astype(bool), "label"] = 0

    originals = result["pair_id"].notna() & result["image_id"].str.endswith(":original")
    original_hashes = set(result.loc[originals, "pixel_sha256"].astype(str))
    inpainting_overlap = train & ~result["quarantined"].astype(bool) & result["pixel_sha256"].astype(str).isin(original_hashes)
    result.loc[inpainting_overlap, ["quarantined", "quarantine_reason"]] = [
        True,
        "exact_inpainting_original_overlap",
    ]
    result = quarantine_duplicate_reference_candidates(result)
    result.loc[train & ~result["quarantined"].astype(bool), "label"] = 0

    normal_candidates = result.loc[train & ~result["quarantined"].astype(bool)].copy()
    if len(normal_candidates) < counts["normal_memory_train"] + counts["policy_calibration_normals"]:
        raise SplitError("Fewer than 10,000 eligible normal detector-reference images remain")
    normal_candidates = normal_candidates.sort_values("pixel_sha256", kind="mergesort")
    normal_ids = _permuted(normal_candidates["image_id"].tolist(), seed)
    memory_count = counts["normal_memory_train"]
    calibration_count = counts["policy_calibration_normals"]
    _set_split(result, normal_ids[:memory_count], "normal_memory_train")
    _set_split(result, normal_ids[memory_count : memory_count + calibration_count], "policy_calibration_normals")
    _set_split(result, normal_ids[memory_count + calibration_count :], "normal_reserve")

    _assign_pair_splits(result, seed, counts)
    # Every final-test item stays in its sealed test split, including masks' image rows.
    result.loc[final, "split"] = "final_test"
    result.loc[result["split"].eq("unassigned") & result["quarantined"].astype(bool), "split"] = "quarantined"
    if result.loc[result["pair_id"].notna()].groupby("pair_id")["split"].nunique().gt(1).any():
        raise SplitError("Members of an inpainting pair were assigned to different splits")
    assert_no_cross_partition_components(result)

    calibration_normals = result.loc[result["split"].eq("policy_calibration_normals")].copy()
    lot_ids = _permuted(calibration_normals["image_id"].tolist(), seed)[: counts["policy_calibration_normal_count"]]
    anomaly_lot = result.loc[result["split"].eq("policy_calibration_anomalies")].copy()
    # The lot has 67 anomaly-pair *images* only on their anomaly side.
    anomaly_lot = anomaly_lot.loc[anomaly_lot["label"].astype("Int64").eq(1)]
    if len(lot_ids) != counts["policy_calibration_normal_count"] or len(anomaly_lot) != counts["policy_calibration_anomaly_pairs"]:
        raise SplitError("Policy calibration lot cardinality does not match the sealed protocol")
    normal_lot = result.loc[result["image_id"].isin(lot_ids)].copy()
    normal_lot["lot_role"] = "normal"
    anomaly_lot = anomaly_lot.copy()
    anomaly_lot["lot_role"] = "anomaly"
    lot = pd.concat([normal_lot, anomaly_lot], ignore_index=True).sort_values("image_id", kind="mergesort")
    if len(lot) != 802:
        raise SplitError(f"Policy calibration lot must have 802 images; found {len(lot)}")
    return SplitArtifacts(
        manifest=normalize_manifest(result),
        policy_calibration_lot=lot.reset_index(drop=True),
        exact_overlap_count=exact_overlap_count,
        eligible_normal_count=len(normal_candidates),
    )
