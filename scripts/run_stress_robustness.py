#!/usr/bin/env python3
"""Run prespecified stressed-observation policy evaluations after core_v1 is sealed.

This script never refits a detector calibrator, PCA/scaler, or router.  It
selects only scenario-specific policy decision thresholds from stressed
calibration normals, then reports both controlled-FPR and clean-threshold
transfer results on the same stressed final-test observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import load_core_config
from run_final_benchmark import _current_lock_values, _verify_git
from seminspect.data.manifests import read_manifest
from seminspect.evaluation.locking import verify_test_lock
from seminspect.evaluation.metrics import policy_result_frame, summarize_policy_results
from seminspect.evaluation.thresholds import Threshold, select_normal_fpr_threshold
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
from seminspect.utils.io import ArtifactError, atomic_write_json, read_json
from seminspect.utils.tabular import read_parquet, write_parquet


def _features(paths: dict[str, Path], split: str, scenario: str) -> pd.DataFrame:
    path = paths["cache_dir"] / "features" / f"features_{split}__stress_{scenario}.parquet"
    if not path.is_file():
        raise ArtifactError(
            f"Missing stressed feature table {path}. Build it with --split {split} --stress-scenario {scenario}."
        )
    frame = read_parquet(path)
    if frame["calibrated_probability"].isna().any():
        raise ArtifactError(f"Stress table is not scored with the frozen clean calibrators: {path}")
    if frame.duplicated(["image_id", "quality_index"]).any():
        raise ArtifactError(f"Stress table contains duplicate quality rows: {path}")
    return frame


def _labels(paths: dict[str, Path], split: str) -> dict[str, int]:
    if split == "policy_calibration":
        frame = read_parquet(paths["manifest_dir"] / "policy_calibration_lot.parquet")
        expected, anomalies = 802, 67
    elif split == "final_test":
        manifest = read_manifest(paths["manifest_dir"] / "miic_splits.parquet")
        frame = manifest.loc[manifest["split"].eq("final_test")]
        expected, anomalies = 1388, 116
    else:  # pragma: no cover - internal callers only
        raise ValueError(split)
    output = {str(row.image_id): int(row.label) for row in frame.itertuples(index=False)}
    if len(output) != expected or sum(output.values()) != anomalies:
        raise ArtifactError(f"Unexpected {split} label cardinality")
    return output


def _threshold_from_record(record: dict) -> Threshold:
    return Threshold(
        value=float(record["threshold"]),
        alpha=0.05,
        normal_count=int(record["normal_count"]),
        rank_one_based=int(record["rank_one_based"]),
        selected_image_id=str(record["selected_image_id"]),
        tie_count=int(record["tie_count"]),
    )


def _clean_threshold(records: list[dict], policy: str, schedule: str, budget: float, seed: int | None) -> Threshold:
    matches = [
        item
        for item in records
        if item["policy"] == policy
        and item["cost_model"] == schedule
        and abs(float(item["budget"]) - budget) <= 1e-12
        and item.get("seed") == seed
    ]
    if len(matches) != 1:
        raise ArtifactError(f"Missing one clean threshold for {policy}/{schedule}/{budget}/seed={seed}")
    return _threshold_from_record(matches[0])


def _stress_threshold(labels: dict[str, int], allocation) -> tuple[Threshold, dict]:
    normals = [(image_id, allocation.final_scores[image_id]) for image_id in sorted(labels) if labels[image_id] == 0]
    threshold = select_normal_fpr_threshold(normals, alpha=0.05)
    record = {
        "threshold": threshold.value,
        "normal_count": threshold.normal_count,
        "rank_one_based": threshold.rank_one_based,
        "selected_image_id": threshold.selected_image_id,
        "tie_count": threshold.tie_count,
        "calibration_fpr": sum(score > threshold.value for _, score in normals) / len(normals),
    }
    return threshold, record


def _tables(features: pd.DataFrame, labels: dict[str, int], paths: dict[str, Path]):
    root = paths["cache_dir"] / "features"
    router_root = paths["cache_dir"] / "router"
    transform = FeatureTransform.load(router_root / "pca_scaler.joblib")
    base = observation_table_from_features(features, feature_root=root)
    ridge = observation_table_from_features(
        features,
        feature_root=root,
        transform=transform,
        predictor=RidgeRouter.load(router_root / "ridge_router.joblib"),
    )
    mlp = observation_table_from_features(
        features,
        feature_root=root,
        transform=transform,
        predictor=MLPRouter.load(router_root / "best_validation.safetensors"),
    )
    oracle = observation_table_from_features(features, feature_root=root, oracle_labels=labels)
    return {
        "base": base,
        "vor_ridge": ridge,
        "vor_mlp": mlp,
        "oracle": oracle,
    }


def _deterministic_runs(tables):
    return [
        ("anomaly_score", "base", anomaly_score_policy()),
        ("uncertainty_entropy", "base", entropy_uncertainty_policy()),
        ("uncertainty_margin", "base", margin_uncertainty_policy()),
        ("resolution_instability", "base", instability_policy()),
        ("vor_ridge", "vor_ridge", predicted_gain_policy("vor_ridge")),
        ("vor_mlp", "vor_mlp", predicted_gain_policy("vor_mlp")),
        ("oracle_gain", "oracle", oracle_gain_policy()),
    ]


def _add_result(
    rows: list[dict],
    labels: dict[str, int],
    allocation,
    threshold: Threshold,
    *,
    scenario: str,
    policy: str,
    schedule: str,
    budget: float,
    seed: int | None,
    threshold_mode: str,
    non_deployable: bool,
) -> None:
    frame = policy_result_frame(
        labels,
        allocation.final_scores,
        allocation.final_qualities,
        threshold,
        policy=policy,
        budget=budget,
        actual_cost=allocation.actual_cost,
    )
    rows.append(
        {
            "scenario": scenario,
            "policy": policy,
            "cost_model": schedule,
            "budget": budget,
            "seed": seed,
            "threshold_mode": threshold_mode,
            "non_deployable": non_deployable,
            **summarize_policy_results(frame),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--scenarios", nargs="+", help="Defaults to every non-clean predeclared scenario")
    parser.add_argument("--all-cost-models", action="store_true", help="Also run non-primary cost schedules")
    parser.add_argument("--force", action="store_true", help="Replace exploratory stress outputs only")
    args = parser.parse_args()
    lock = read_json(args.lock)
    config, paths = load_core_config(Path(lock["resolved_config_path"]))
    if lock.get("config_resolved_sha256") != config_hash(config):
        raise ArtifactError("Current configuration differs from the sealed primary protocol")
    verify_test_lock(args.lock, _current_lock_values(config, paths))
    _verify_git(lock, Path(config["_resolved_from"]).parents[2])
    if not (args.run_dir / "primary_endpoint.json").is_file():
        raise ArtifactError("Stress analysis requires the immutable primary final benchmark")
    scenarios = args.scenarios or [name for name in config["stress_scenarios"] if name != "clean_reference"]
    invalid = sorted(set(scenarios) - set(config["stress_scenarios"]))
    if invalid:
        raise ArtifactError(f"Unknown stress scenarios: {invalid}")
    schedules = config["cost_models"] if args.all_cost_models else {"pixel_information": config["cost_models"]["pixel_information"]}
    clean_records = read_json(paths["artifacts_dir"] / "policy_thresholds.json")["thresholds"]
    calibration_labels = _labels(paths, "policy_calibration")
    test_labels = _labels(paths, "final_test")
    threshold_rows: list[dict] = []
    calibration_rows: list[dict] = []
    test_rows: list[dict] = []
    for scenario in scenarios:
        calibration_features = _features(paths, "policy_calibration", scenario)
        test_features = _features(paths, "final_test", scenario)
        if set(calibration_features["image_id"].astype(str)) != set(calibration_labels):
            raise ArtifactError(f"Stress calibration IDs differ for {scenario}")
        if set(test_features["image_id"].astype(str)) != set(test_labels):
            raise ArtifactError(f"Stress final-test IDs differ for {scenario}")
        calibration_tables = _tables(calibration_features, calibration_labels, paths)
        test_tables = _tables(test_features, test_labels, paths)
        for schedule_name, schedule_config in schedules.items():
            schedule = CostSchedule.from_values(schedule_name, schedule_config["cumulative_costs"])
            for budget in [float(item) for item in schedule_config["budgets"]]:
                runs = [("survey_only", "base", None, survey_only(calibration_tables["base"], schedule, budget), survey_only(test_tables["base"], schedule, budget), False)]
                for policy_name, table_key, policy in _deterministic_runs(calibration_tables):
                    test_policy = next(item[2] for item in _deterministic_runs(test_tables) if item[0] == policy_name)
                    runs.append(
                        (
                            policy_name,
                            table_key,
                            policy,
                            allocate_deterministic(calibration_tables[table_key], policy, schedule, budget),
                            allocate_deterministic(test_tables[table_key], test_policy, schedule, budget),
                            bool(policy.non_deployable),
                        )
                    )
                if abs(budget - schedule.q3_cost) <= 1e-12:
                    runs.append(("full_q3", "base", None, full_q3(calibration_tables["base"], schedule, budget), full_q3(test_tables["base"], schedule, budget), False))
                for policy_name, _, _, calibration_allocation, test_allocation, non_deployable in runs:
                    threshold, record = _stress_threshold(calibration_labels, calibration_allocation)
                    metadata = {
                        "scenario": scenario,
                        "policy": policy_name,
                        "cost_model": schedule_name,
                        "budget": budget,
                        "seed": None,
                        "actual_calibration_cost": calibration_allocation.actual_cost,
                        "non_deployable": non_deployable,
                        **record,
                    }
                    threshold_rows.append(metadata)
                    _add_result(calibration_rows, calibration_labels, calibration_allocation, threshold, scenario=scenario, policy=policy_name, schedule=schedule_name, budget=budget, seed=None, threshold_mode="scenario_controlled_fpr", non_deployable=non_deployable)
                    _add_result(test_rows, test_labels, test_allocation, threshold, scenario=scenario, policy=policy_name, schedule=schedule_name, budget=budget, seed=None, threshold_mode="scenario_controlled_fpr", non_deployable=non_deployable)
                    _add_result(test_rows, test_labels, test_allocation, _clean_threshold(clean_records, policy_name, schedule_name, budget, None), scenario=scenario, policy=policy_name, schedule=schedule_name, budget=budget, seed=None, threshold_mode="clean_threshold_transfer", non_deployable=non_deployable)
                for seed in config["policy"]["random_seeds"]:
                    calibration_allocation = allocate_random(calibration_tables["base"], schedule, budget, int(seed))
                    test_allocation = allocate_random(test_tables["base"], schedule, budget, int(seed))
                    threshold, record = _stress_threshold(calibration_labels, calibration_allocation)
                    threshold_rows.append({"scenario": scenario, "policy": "random", "cost_model": schedule_name, "budget": budget, "seed": int(seed), "actual_calibration_cost": calibration_allocation.actual_cost, "non_deployable": False, **record})
                    _add_result(calibration_rows, calibration_labels, calibration_allocation, threshold, scenario=scenario, policy="random", schedule=schedule_name, budget=budget, seed=int(seed), threshold_mode="scenario_controlled_fpr", non_deployable=False)
                    _add_result(test_rows, test_labels, test_allocation, threshold, scenario=scenario, policy="random", schedule=schedule_name, budget=budget, seed=int(seed), threshold_mode="scenario_controlled_fpr", non_deployable=False)
                    _add_result(test_rows, test_labels, test_allocation, _clean_threshold(clean_records, "random", schedule_name, budget, int(seed)), scenario=scenario, policy="random", schedule=schedule_name, budget=budget, seed=int(seed), threshold_mode="clean_threshold_transfer", non_deployable=False)
    output = args.run_dir / "secondary"
    atomic_write_json(output / "stress_thresholds.json", {"exploratory": True, "thresholds": threshold_rows}, overwrite=args.force)
    write_parquet(pd.DataFrame(calibration_rows), output / "stress_policy_calibration.parquet", overwrite=args.force)
    write_parquet(pd.DataFrame(test_rows), output / "stress_results.parquet", overwrite=args.force)
    print(f"Wrote exploratory stress results for {len(scenarios)} scenario(s) to {output}")


if __name__ == "__main__":
    main()
