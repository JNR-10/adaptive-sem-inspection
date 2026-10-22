#!/usr/bin/env python3
"""Run one locked final-test benchmark from cached features and frozen policy thresholds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_core_config
from seminspect.data.manifests import read_manifest
from seminspect.detector.feature_store import read_anomaly_map
from seminspect.evaluation.bootstrap import paired_stratified_recall_difference
from seminspect.evaluation.locking import ids_hash, verify_test_lock
from seminspect.evaluation.metrics import policy_result_frame, summarize_policy_results
from seminspect.evaluation.thresholds import Threshold
from seminspect.policies.allocator import CostSchedule, allocate_deterministic, allocate_random, full_q3, survey_only
from seminspect.policies.anomaly_score import anomaly_score_policy
from seminspect.policies.instability import instability_policy
from seminspect.policies.oracle import oracle_gain_policy
from seminspect.policies.runtime import observation_table_from_features
from seminspect.policies.uncertainty import entropy_uncertainty_policy, margin_uncertainty_policy
from seminspect.policies.vor import predicted_gain_policy
from seminspect.router.features import FeatureTransform
from seminspect.router.mlp import MLPRouter
from seminspect.router.ridge import RidgeRouter
from seminspect.utils.config import config_hash
from seminspect.utils.io import ArtifactError, atomic_copy, atomic_write_json, atomic_write_text, canonical_json_sha256, file_sha256, path_tree_sha256, read_json, utc_now
from seminspect.utils.reproducibility import write_environment
from seminspect.utils.tabular import read_parquet, write_parquet


def _environment_lock_sha256(project_root: Path) -> str:
    """Hash the dependency lock selected by the invoking pipeline."""

    configured = os.environ.get("SEMSPECT_ENVIRONMENT_LOCK")
    lock_path = Path(configured) if configured else project_root / "requirements-colab.lock"
    if not lock_path.is_file():
        raise ArtifactError(f"Environment lock does not exist: {lock_path}")
    return file_sha256(lock_path)


def _current_lock_values(config: dict, paths: dict[str, Path]) -> dict[str, str]:
    project_root = Path(config["_resolved_from"]).parents[2]
    split_path = paths["manifest_dir"] / "miic_splits.parquet"
    raw_path = paths["manifest_dir"] / "miic_raw.parquet"
    artifact = paths["artifacts_dir"]
    bank_root = paths["cache_dir"] / "memory_banks"
    calibrator_root = paths["cache_dir"] / "calibrators"
    router_root = paths["cache_dir"] / "router"
    memory_metadata = read_json(bank_root / "memory_bank_metadata.json")
    manifest = read_manifest(split_path)
    return {
        "environment_lock_sha256": _environment_lock_sha256(project_root),
        "dataset_manifest_sha256": file_sha256(raw_path),
        "split_manifest_sha256": file_sha256(split_path),
        "duplicate_decisions_sha256": file_sha256(artifact / "audits" / "duplicate_decisions.parquet"),
        "simulator_config_sha256": canonical_json_sha256({"image": config["image"], "qualities": config["qualities"], "stress_scenarios": config["stress_scenarios"]}),
        "detector_config_sha256": canonical_json_sha256({"detector": config["detector"], "calibration": config["calibration"]}),
        "encoder_weight_sha256": str(memory_metadata.get("weight_sha256")),
        "memory_bank_sha256": path_tree_sha256(
            [bank_root / f"patch_bank_Q{quality}.npz" for quality in range(4)]
            + [bank_root / f"cls_reference_Q{quality}.npy" for quality in range(4)]
            + [bank_root / f"cls_reference_ids_Q{quality}.npy" for quality in range(4)]
            + [bank_root / "memory_bank_metadata.json"],
            root=paths["cache_dir"],
        ),
        "calibrator_sha256": path_tree_sha256([calibrator_root / f"score_calibrator_Q{quality}.joblib" for quality in range(4)], root=paths["cache_dir"]),
        "pca_and_scaler_sha256": file_sha256(router_root / "pca_scaler.joblib"),
        "router_checkpoint_sha256": file_sha256(router_root / "best_validation.safetensors"),
        "policy_config_sha256": canonical_json_sha256({"cost_models": config["cost_models"], "policy": config["policy"]}),
        "policy_thresholds_sha256": file_sha256(artifact / "policy_thresholds.json"),
        "final_test_id_sha256": ids_hash(manifest.loc[manifest["split"].eq("final_test"), "image_id"].astype(str).tolist()),
        "config_resolved_sha256": config_hash(config),
    }


def _verify_git(lock: dict, project_root: Path) -> None:
    commit = subprocess.run(["git", "-C", str(project_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if commit != lock["git_commit"]:
        raise ArtifactError("Git commit differs from the sealed test lock")
    diff = subprocess.run(["git", "-C", str(project_root), "diff", "--binary"], capture_output=True, check=True).stdout
    current = None if not diff else __import__("hashlib").sha256(diff).hexdigest()
    if current != lock["git_diff_sha256"]:
        raise ArtifactError("Working-tree patch differs from the sealed test lock")


def _final_features(paths: dict[str, Path]) -> pd.DataFrame:
    root = paths["cache_dir"] / "features"
    candidates = [root / "features_final_test.parquet", root / "features_calibrated_final_test.parquet"]
    for path in candidates:
        if not path.is_file():
            continue
        frame = read_parquet(path)
        if len(frame) == 5552 and frame["calibrated_probability"].notna().all():
            return frame
    raise ArtifactError("Final-test feature table must contain 1,388 fully calibrated Q0–Q3 ladders")


def _threshold_lookup(records: list[dict], policy: str, cost_model: str, budget: float, seed: int | None) -> Threshold:
    candidates = [
        item
        for item in records
        if item["policy"] == policy
        and item["cost_model"] == cost_model
        and abs(float(item["budget"]) - budget) <= 1e-12
        and item.get("seed") == seed
    ]
    if len(candidates) != 1:
        raise ArtifactError(f"Expected one frozen threshold for {policy}/{cost_model}/{budget}/seed={seed}; found {len(candidates)}")
    value = candidates[0]
    return Threshold(
        value=float(value["threshold"]),
        alpha=0.05,
        normal_count=int(value["normal_count"]),
        rank_one_based=int(value["rank_one_based"]),
        selected_image_id=str(value["selected_image_id"]),
        tie_count=int(value["tie_count"]),
    )


def _write_trajectories(path: Path, records: list[dict]) -> None:
    text = "".join(json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in records)
    atomic_write_text(path, text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path, help="Default: $SEMSPECT_RUNTIME_ROOT/runs/core_v1_final")
    args = parser.parse_args()
    lock = read_json(args.lock)
    config_path = Path(lock["resolved_config_path"])
    config, paths = load_core_config(config_path)
    verify_test_lock(args.lock, _current_lock_values(config, paths))
    _verify_git(lock, Path(config["_resolved_from"]).parents[2])
    run_dir = args.run_dir or paths["runs_dir"] / "core_v1_final"
    if run_dir.exists():
        raise ArtifactError(f"Final run directory already exists and will not be overwritten: {run_dir}")
    reliability_path = paths["cache_dir"] / "calibrators" / "calibration_reliability.parquet"
    if not reliability_path.is_file():
        raise ArtifactError("Calibration reliability table is missing; rerun the development calibration stage before sealing")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_environment(run_dir / "environment.json", int(config["seed"]), deterministic=True)
    atomic_copy(args.lock, run_dir / "test_lock.json")
    atomic_write_json(run_dir / "config_resolved.json", config)
    atomic_write_json(run_dir / "git_state.json", {"commit": lock["git_commit"], "diff_sha256": lock["git_diff_sha256"]})
    atomic_write_json(run_dir / "dataset_manifest_hashes.json", {key: lock[key] for key in ("dataset_manifest_sha256", "split_manifest_sha256", "final_test_id_sha256")})
    atomic_copy(reliability_path, run_dir / "calibration_reliability.parquet")

    manifest = read_manifest(paths["manifest_dir"] / "miic_splits.parquet")
    test = manifest.loc[manifest["split"].eq("final_test")].sort_values("image_id", kind="mergesort")
    labels = {str(row.image_id): int(row.label) for row in test.itertuples(index=False)}
    if len(labels) != 1388 or sum(labels.values()) != 116:
        raise ArtifactError("Final test must contain 1,272 normals and 116 real anomalies")
    features = _final_features(paths)
    if set(features["image_id"].astype(str)) != set(labels):
        raise ArtifactError("Final-test feature IDs differ from sealed manifest IDs")
    feature_root = paths["cache_dir"] / "features"
    router_root = paths["cache_dir"] / "router"
    transform = FeatureTransform.load(router_root / "pca_scaler.joblib")
    base_table = observation_table_from_features(features, feature_root=feature_root)
    ridge_table = observation_table_from_features(features, feature_root=feature_root, transform=transform, predictor=RidgeRouter.load(router_root / "ridge_router.joblib"))
    mlp_table = observation_table_from_features(features, feature_root=feature_root, transform=transform, predictor=MLPRouter.load(router_root / "best_validation.safetensors"))
    oracle_table = observation_table_from_features(features, feature_root=feature_root, oracle_labels=labels)
    deterministic = [
        ("anomaly_score", base_table, anomaly_score_policy()),
        ("uncertainty_entropy", base_table, entropy_uncertainty_policy()),
        ("uncertainty_margin", base_table, margin_uncertainty_policy()),
        ("resolution_instability", base_table, instability_policy()),
        ("vor_ridge", ridge_table, predicted_gain_policy("vor_ridge")),
        ("vor_mlp", mlp_table, predicted_gain_policy("vor_mlp")),
        ("oracle_gain", oracle_table, oracle_gain_policy()),
    ]
    threshold_records = read_json(paths["artifacts_dir"] / "policy_thresholds.json")["thresholds"]
    per_image: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    trajectories: list[dict] = []
    for model_name, schedule_config in config["cost_models"].items():
        schedule = CostSchedule.from_values(model_name, schedule_config["cumulative_costs"])
        for budget in [float(item) for item in schedule_config["budgets"]]:
            runs = [("survey_only", base_table, None, None, survey_only(base_table, schedule, budget))]
            runs.extend((name, table, policy, None, allocate_deterministic(table, policy, schedule, budget)) for name, table, policy in deterministic)
            if abs(budget - schedule.q3_cost) <= 1e-12:
                runs.append(("full_q3", base_table, None, None, full_q3(base_table, schedule, budget)))
            for policy_name, table, policy, seed, allocation in runs:
                threshold = _threshold_lookup(threshold_records, policy_name, model_name, budget, seed)
                frame = policy_result_frame(labels, allocation.final_scores, allocation.final_qualities, threshold, policy=policy_name, budget=budget, actual_cost=allocation.actual_cost)
                frame["cost_model"] = model_name
                frame["seed"] = seed
                per_image.append(frame)
                metric_rows.append({"policy": policy_name, "cost_model": model_name, "budget": budget, "seed": seed, **summarize_policy_results(frame), "non_deployable": bool(policy.non_deployable) if policy else False})
                for image_id, path in allocation.trajectories.items():
                    final_quality = allocation.final_qualities[image_id]
                    final_score = allocation.final_scores[image_id]
                    trajectories.append(
                        {
                            "image_id": image_id,
                            "label": labels[image_id],
                            "policy": policy_name,
                            "cost_model": model_name,
                            "budget": budget,
                            "seed": seed,
                            "path": path,
                            "final_quality": final_quality,
                            "final_score": final_score,
                            "threshold": threshold.value,
                            "prediction": int(final_score > threshold.value),
                            "cost": schedule.cumulative(final_quality),
                        }
                    )
            for seed in config["policy"]["random_seeds"]:
                allocation = allocate_random(base_table, schedule, budget, int(seed))
                threshold = _threshold_lookup(threshold_records, "random", model_name, budget, int(seed))
                frame = policy_result_frame(labels, allocation.final_scores, allocation.final_qualities, threshold, policy="random", budget=budget, actual_cost=allocation.actual_cost)
                frame["cost_model"] = model_name
                frame["seed"] = int(seed)
                per_image.append(frame)
                metric_rows.append({"policy": "random", "cost_model": model_name, "budget": budget, "seed": int(seed), **summarize_policy_results(frame), "non_deployable": False})
                for image_id, path in allocation.trajectories.items():
                    final_quality = allocation.final_qualities[image_id]
                    final_score = allocation.final_scores[image_id]
                    trajectories.append(
                        {
                            "image_id": image_id,
                            "label": labels[image_id],
                            "policy": "random",
                            "cost_model": model_name,
                            "budget": budget,
                            "seed": int(seed),
                            "path": path,
                            "final_quality": final_quality,
                            "final_score": final_score,
                            "threshold": threshold.value,
                            "prediction": int(final_score > threshold.value),
                            "cost": schedule.cumulative(final_quality),
                        }
                    )
    per_image_frame = pd.concat(per_image, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    # Report fixed-seed random mean separately without averaging scores or predictions before thresholding.
    numeric = [column for column in metrics.columns if pd.api.types.is_numeric_dtype(metrics[column]) and column not in {"seed", "budget"}]
    random_mean = metrics.loc[metrics["policy"].eq("random")].groupby(["policy", "cost_model", "budget"], as_index=False)[numeric].mean()
    random_mean["seed"] = "mean_30_seeds"
    random_mean["non_deployable"] = False
    metrics = pd.concat([metrics, random_mean[metrics.columns]], ignore_index=True)
    primary_mlp = per_image_frame.loc[(per_image_frame["policy"] == "vor_mlp") & (per_image_frame["cost_model"] == "pixel_information") & (per_image_frame["budget"] == float(config["evaluation"]["primary_budget"]))]
    primary_uncertainty = per_image_frame.loc[(per_image_frame["policy"] == "uncertainty_entropy") & (per_image_frame["cost_model"] == "pixel_information") & (per_image_frame["budget"] == float(config["evaluation"]["primary_budget"]))]
    interval = paired_stratified_recall_difference(primary_mlp, primary_uncertainty, replicates=int(config["evaluation"]["bootstrap_replicates"]), seed=int(config["evaluation"]["bootstrap_seed"]))
    write_parquet(metrics, run_dir / "metrics.parquet")
    write_parquet(per_image_frame, run_dir / "per_image_results.parquet")
    _write_trajectories(run_dir / "decision_trajectories.jsonl", trajectories)
    atomic_write_json(run_dir / "primary_endpoint.json", {"comparison": "vor_mlp_minus_uncertainty_entropy_recall", "cost_model": "pixel_information", "budget": config["evaluation"]["primary_budget"], "estimate": interval.estimate, "ci95": [interval.lower, interval.upper], "bootstrap_replicates": interval.replicates})
    atomic_write_json(
        run_dir / "artifact_hashes.json",
        {
            "metrics_sha256": file_sha256(run_dir / "metrics.parquet"),
            "per_image_results_sha256": file_sha256(run_dir / "per_image_results.parquet"),
            "trajectories_sha256": file_sha256(run_dir / "decision_trajectories.jsonl"),
            "calibration_reliability_sha256": file_sha256(run_dir / "calibration_reliability.parquet"),
            "created_utc": utc_now(),
        },
    )
    atomic_write_text(run_dir / "warnings.log", "")
    print(f"Locked final benchmark complete: {run_dir}")


if __name__ == "__main__":
    main()
