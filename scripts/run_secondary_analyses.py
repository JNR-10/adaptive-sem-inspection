#!/usr/bin/env python3
"""Run prespecified exploratory defect-size, prevalence, and Carinthia-S analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_core_config
from run_final_benchmark import _current_lock_values, _verify_git
from seminspect.data.manifests import read_manifest
from seminspect.evaluation.locking import verify_test_lock
from seminspect.evaluation.metrics import policy_result_frame, summarize_policy_results
from seminspect.evaluation.secondary import carinthia_mask_statistics, miic_mask_statistics, prevalence_lots
from seminspect.evaluation.thresholds import Threshold
from seminspect.policies.allocator import CostSchedule, allocate_deterministic
from seminspect.policies.runtime import observation_table_from_features, virtual_lot
from seminspect.policies.uncertainty import entropy_uncertainty_policy
from seminspect.policies.vor import predicted_gain_policy
from seminspect.router.features import FeatureTransform
from seminspect.router.mlp import MLPRouter
from seminspect.utils.config import config_hash
from seminspect.utils.io import ArtifactError, atomic_write_json, read_json
from seminspect.utils.tabular import read_parquet, write_parquet


def _load_context(lock_path: Path):
    lock = read_json(lock_path)
    config, paths = load_core_config(Path(lock["resolved_config_path"]))
    if lock["config_resolved_sha256"] != config_hash(config):
        raise ArtifactError("Current config differs from locked primary protocol")
    verify_test_lock(lock_path, _current_lock_values(config, paths))
    _verify_git(lock, Path(config["_resolved_from"]).parents[2])
    return lock, config, paths


def _feature_table(paths: dict[str, Path], name: str) -> pd.DataFrame:
    root = paths["cache_dir"] / "features"
    for candidate in (root / f"features_{name}.parquet", root / f"features_calibrated_{name}.parquet"):
        if candidate.is_file():
            frame = read_parquet(candidate)
            if frame["calibrated_probability"].notna().all():
                return frame
    raise ArtifactError(f"Fully calibrated feature table for {name} has not been built")


def _threshold(records: list[dict], policy: str, budget: float) -> Threshold:
    matches = [
        item
        for item in records
        if item["policy"] == policy
        and item["cost_model"] == "pixel_information"
        and abs(float(item["budget"]) - budget) <= 1e-12
        and item.get("seed") is None
    ]
    if len(matches) != 1:
        raise ArtifactError(f"Could not locate frozen primary threshold for {policy} at {budget}")
    item = matches[0]
    return Threshold(float(item["threshold"]), 0.05, int(item["normal_count"]), int(item["rank_one_based"]), str(item["selected_image_id"]), int(item["tie_count"]))


def _defect_size(config: dict, paths: dict[str, Path], run_dir: Path, force: bool) -> None:
    manifest = read_manifest(paths["manifest_dir"] / "miic_splits.parquet")
    stats = miic_mask_statistics(manifest.loc[manifest["split"].eq("final_test")], paths["extracted_dir"])
    per_image = read_parquet(run_dir / "per_image_results.parquet")
    primary = per_image.loc[(per_image["cost_model"] == "pixel_information") & (per_image["budget"] == float(config["evaluation"]["primary_budget"])) & per_image["seed"].isna()].copy()
    anomaly = primary.loc[primary["label"] == 1].merge(stats, on="image_id", how="inner", validate="many_to_one")
    schedule = CostSchedule.from_values("pixel_information", config["cost_models"]["pixel_information"]["cumulative_costs"])
    anomaly["per_image_cost"] = anomaly["final_quality"].map(schedule.cumulative)
    summary = anomaly.groupby(["policy", "size_quartile"], as_index=False).agg(
        recall=("prediction", "mean"), average_final_quality=("final_quality", "mean"), average_cost=("per_image_cost", "mean"), count=("image_id", "count")
    )
    output = run_dir / "secondary"
    write_parquet(stats, output / "miic_mask_statistics.parquet", overwrite=force)
    write_parquet(summary, output / "defect_size_summary.parquet", overwrite=force)


def _prevalence(config: dict, paths: dict[str, Path], run_dir: Path, force: bool) -> None:
    manifest = read_manifest(paths["manifest_dir"] / "miic_splits.parquet")
    final = manifest.loc[manifest["split"].eq("final_test")]
    labels = {str(row.image_id): int(row.label) for row in final.itertuples(index=False)}
    features = _feature_table(paths, "final_test")
    transform = FeatureTransform.load(paths["cache_dir"] / "router" / "pca_scaler.joblib")
    feature_root = paths["cache_dir"] / "features"
    base = observation_table_from_features(features, feature_root=feature_root)
    mlp = observation_table_from_features(features, feature_root=feature_root, transform=transform, predictor=MLPRouter.load(paths["cache_dir"] / "router" / "best_validation.safetensors"))
    schedule = CostSchedule.from_values("pixel_information", config["cost_models"]["pixel_information"]["cumulative_costs"])
    records = read_json(paths["artifacts_dir"] / "policy_thresholds.json")["thresholds"]
    rows: list[dict] = []
    for anomaly_count, repeat, source_ids in prevalence_lots(labels):
        base_lot, reverse = virtual_lot(base, source_ids)
        mlp_lot, _ = virtual_lot(mlp, source_ids)
        virtual_labels = {virtual: labels[source] for virtual, source in reverse.items()}
        for budget in config["evaluation"]["budgets"]:
            for name, table, policy in (
                ("uncertainty_entropy", base_lot, entropy_uncertainty_policy()),
                ("vor_mlp", mlp_lot, predicted_gain_policy("vor_mlp")),
            ):
                allocation = allocate_deterministic(table, policy, schedule, float(budget))
                threshold = _threshold(records, name, float(budget))
                result = policy_result_frame(virtual_labels, allocation.final_scores, allocation.final_qualities, threshold, policy=name, budget=float(budget), actual_cost=allocation.actual_cost)
                rows.append({"anomaly_count": anomaly_count, "repeat": repeat, "policy": name, "budget": float(budget), **summarize_policy_results(result)})
    write_parquet(pd.DataFrame(rows), run_dir / "secondary" / "prevalence_sensitivity.parquet", overwrite=force)


def _carinthia(config: dict, paths: dict[str, Path], run_dir: Path, force: bool) -> None:
    manifest = read_manifest(paths["manifest_dir"] / "carinthia_s.parquet")
    features = _feature_table(paths, "carinthia_secondary")
    transform = FeatureTransform.load(paths["cache_dir"] / "router" / "pca_scaler.joblib")
    feature_root = paths["cache_dir"] / "features"
    base = observation_table_from_features(features, feature_root=feature_root)
    mlp = observation_table_from_features(features, feature_root=feature_root, transform=transform, predictor=MLPRouter.load(paths["cache_dir"] / "router" / "best_validation.safetensors"))
    schedule = CostSchedule.from_values("pixel_information", config["cost_models"]["pixel_information"]["cumulative_costs"])
    budget = float(config["evaluation"]["primary_budget"])
    uncertainty = allocate_deterministic(base, entropy_uncertainty_policy(), schedule, budget)
    vor = allocate_deterministic(mlp, predicted_gain_policy("vor_mlp"), schedule, budget)
    p = features.pivot(index="image_id", columns="quality_index", values="calibrated_probability").rename(columns={0: "p_q0", 1: "p_q1", 2: "p_q2", 3: "p_q3"}).reset_index()
    masks = carinthia_mask_statistics(manifest, paths["extracted_dir"])
    output = manifest[["image_id", "class_id", "label"]].merge(masks, on=["image_id", "class_id"], how="left", validate="one_to_one").merge(p, on="image_id", how="left", validate="one_to_one")
    output["q3_minus_q0_probability"] = output["p_q3"] - output["p_q0"]
    output["vor_final_quality"] = output["image_id"].map(vor.final_qualities)
    output["uncertainty_final_quality"] = output["image_id"].map(uncertainty.final_qualities)
    output["vor_final_score"] = output["image_id"].map(vor.final_scores)
    output["uncertainty_final_score"] = output["image_id"].map(uncertainty.final_scores)
    # This is deliberately descriptive: no Carinthia normal corpus supports valid FPR/precision claims.
    class_summary = output.loc[output["class_id"].isin(["1", "3", "4"])].groupby("class_id", as_index=False).agg(
        count=("image_id", "count"), median_mask_fraction=("mask_fraction", "median"), median_q3_minus_q0_probability=("q3_minus_q0_probability", "median"), median_vor_final_quality=("vor_final_quality", "median")
    )
    class6_summary = output.loc[output["class_id"] == "6", ["p_q0", "p_q1", "p_q2", "p_q3", "vor_final_score"]].describe().reset_index()
    target = run_dir / "secondary"
    write_parquet(output, target / "carinthia_secondary_per_image.parquet", overwrite=force)
    write_parquet(class_summary, target / "carinthia_secondary_class_summary.parquet", overwrite=force)
    write_parquet(class6_summary, target / "carinthia_class6_score_distribution.parquet", overwrite=force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--analysis", nargs="+", choices=["defect_size", "prevalence", "carinthia", "all"], default=["all"])
    parser.add_argument("--force", action="store_true", help="Replace exploratory secondary tables only")
    args = parser.parse_args()
    _, config, paths = _load_context(args.lock)
    run_dir = args.run_dir
    if not (run_dir / "primary_endpoint.json").is_file():
        raise ArtifactError("Secondary analysis requires a completed immutable primary final run")
    selections = {"defect_size", "prevalence", "carinthia"} if "all" in args.analysis else set(args.analysis)
    if "defect_size" in selections:
        _defect_size(config, paths, run_dir, args.force)
    if "prevalence" in selections:
        _prevalence(config, paths, run_dir, args.force)
    if "carinthia" in selections:
        _carinthia(config, paths, run_dir, args.force)
    atomic_write_json(run_dir / "secondary" / "secondary_analysis_metadata.json", {"analyses": sorted(selections), "exploratory": True}, overwrite=args.force)
    print(f"Wrote exploratory secondary analyses: {', '.join(sorted(selections))}")


if __name__ == "__main__":
    main()
