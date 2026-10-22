#!/usr/bin/env python3
"""Assign exact duplicate groups, remove development leakage, and freeze fixed splits."""

from __future__ import annotations

import argparse

import pandas as pd

from _common import add_config_argument, load_core_config, write_failure
from seminspect.data.duplicates import assign_duplicate_groups, exact_duplicate_pairs
from seminspect.data.manifests import read_manifest, write_manifest
from seminspect.data.splits import build_miic_splits
from seminspect.utils.tabular import write_parquet


def _decision_table(before: pd.DataFrame, after: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    previous = before.set_index("image_id")
    working = after.copy()
    representatives = working.groupby("duplicate_group", dropna=False)["pixel_sha256"].min().rename("canonical_representative")
    working = working.join(representatives, on="duplicate_group")
    component_size = working.groupby("duplicate_group", dropna=False)["image_id"].transform("size")
    selected = working.loc[(component_size > 1) | working["quarantined"].astype(bool)].copy()
    members = pd.DataFrame(
        {
            "decision_type": "member",
            "image_id": selected["image_id"],
            "first_id": None,
            "second_id": None,
            "pixel_sha256": selected["pixel_sha256"],
            "original_split": [previous.loc[item, "split"] for item in selected["image_id"]],
            "final_split": selected["split"],
            "canonical_representative": selected["canonical_representative"],
            "reason": selected["quarantine_reason"].fillna("exact_duplicate_component"),
            "phash_distance": None,
            "cosine_similarity": None,
            "ssim": None,
        }
    )
    if evidence.empty:
        return members.sort_values(["decision_type", "image_id"], kind="mergesort").reset_index(drop=True)
    pairs = evidence.copy()
    pairs["decision_type"] = "pair_evidence"
    pairs["image_id"] = None
    pairs["pixel_sha256"] = None
    pairs["original_split"] = "unassigned"
    pairs["final_split"] = None
    pairs["canonical_representative"] = None
    pairs["reason"] = pairs["candidate_source"].map(
        {"pixel_sha256": "exact_decoded_pixel_match"}
    ).fillna("confirmed_duplicate")
    pairs = pairs.rename(columns={"candidate_source": "candidate_source"})
    for column in members.columns:
        if column not in pairs:
            pairs[column] = None
    return pd.concat([members, pairs[members.columns]], ignore_index=True).sort_values(
        ["decision_type", "image_id", "first_id", "second_id"], kind="mergesort", na_position="last"
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--force", action="store_true", help="Replace development audit outputs only")
    args = parser.parse_args()
    config, paths = load_core_config(args.config)
    try:
        raw_path = paths["manifest_dir"] / "miic_raw.parquet"
        manifest = read_manifest(raw_path)
        confirmed = exact_duplicate_pairs(manifest)
        grouped, evidence = assign_duplicate_groups(manifest, confirmed)
        artifacts = build_miic_splits(
            grouped,
            seed=int(config["seed"]),
            expected_train_after_exact_test=int(
                config["datasets"]["miic"]["expected"]["train_after_exact_test_subtraction"]
            ),
            counts=config["splits"],
        )
        split_path = paths["manifest_dir"] / "miic_splits.parquet"
        lot_path = paths["manifest_dir"] / "policy_calibration_lot.parquet"
        audit_path = paths["artifacts_dir"] / "audits" / "duplicate_decisions.parquet"
        write_manifest(artifacts.manifest, split_path, overwrite=args.force)
        write_parquet(artifacts.policy_calibration_lot, lot_path, overwrite=args.force)
        write_parquet(_decision_table(manifest, artifacts.manifest, evidence), audit_path, overwrite=args.force)
        print(
            f"Split audit complete: exact_overlap={artifacts.exact_overlap_count}, "
            f"eligible_normals={artifacts.eligible_normal_count}, confirmed_pairs={len(confirmed)}"
        )
    except Exception as exc:
        write_failure(paths["artifacts_dir"], "audit_leakage", exc)
        raise


if __name__ == "__main__":
    main()
