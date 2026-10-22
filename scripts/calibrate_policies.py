#!/usr/bin/env python3
"""Allocate the policy-calibration lot and freeze every policy/budget threshold."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import add_config_argument, load_core_config
from seminspect.policies.allocator import (
    CostSchedule,
    allocate_deterministic,
    allocate_random,
    full_q3,
    survey_only,
)
from seminspect.policies.anomaly_score import anomaly_score_policy
from seminspect.policies.instability import instability_policy
from seminspect.policies.oracle import oracle_gain_policy
from seminspect.policies.runtime import observation_table_from_features
from seminspect.policies.uncertainty import entropy_uncertainty_policy, margin_uncertainty_policy
from seminspect.policies.vor import predicted_gain_policy
from seminspect.router.features import FeatureTransform
from seminspect.router.mlp import MLPRouter
from seminspect.router.ridge import RidgeRouter
from seminspect.evaluation.metrics import policy_result_frame
from seminspect.evaluation.thresholds import select_normal_fpr_threshold
from seminspect.utils.io import ArtifactError, atomic_write_json
from seminspect.utils.tabular import read_parquet, write_parquet


def _features_and_labels(paths: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, int]]:
    feature_path = paths["cache_dir"] / "features" / "features_calibrated_policy_calibration.parquet"
    lot_path = paths["manifest_dir"] / "policy_calibration_lot.parquet"
    if not feature_path.is_file() or not lot_path.is_file():
        raise ArtifactError("Policy-calibration feature table and 802-image lot are required")
    features = read_parquet(feature_path)
    lot = read_parquet(lot_path)
    labels = {str(row.image_id): int(row.label) for row in lot.itertuples(index=False)}
    if len(labels) != 802 or sum(labels.values()) != 67:
        raise ArtifactError("Policy calibration lot must contain 735 normals and 67 anomalies")
    if set(features["image_id"].astype(str)) != set(labels) or len(features) != 4 * len(labels):
        raise ArtifactError("Policy calibration features must have exact Q0–Q3 rows for the 802-image lot")
    return features, labels


def _record_threshold(policy: str, schedule: CostSchedule, budget: float, seed: int | None, result, labels: dict[str, int]) -> dict:
    normal_scores = [(image_id, result.final_scores[image_id]) for image_id in sorted(labels) if labels[image_id] == 0]
    threshold = select_normal_fpr_threshold(normal_scores, alpha=0.05)
    calibration_fpr = sum(score > threshold.value for _, score in normal_scores) / len(normal_scores)
    return {
        "policy": policy,
        "cost_model": schedule.name,
        "budget": budget,
        "seed": seed,
        "threshold": threshold.value,
        "normal_count": threshold.normal_count,
        "rank_one_based": threshold.rank_one_based,
        "selected_image_id": threshold.selected_image_id,
        "tie_count": threshold.tie_count,
        "calibration_fpr": calibration_fpr,
        "actual_cost": result.actual_cost,
    }


def _result_rows(policy: str, schedule: CostSchedule, budget: float, seed: int | None, result, labels: dict[str, int], threshold_record: dict) -> pd.DataFrame:
    frame = policy_result_frame(
        labels,
        result.final_scores,
        result.final_qualities,
        threshold=float(threshold_record["threshold"]),
        policy=policy,
        budget=budget,
        actual_cost=result.actual_cost,
    )
    frame["cost_model"] = schedule.name
    frame["seed"] = seed
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--force", action="store_true", help="Replace development policy-calibration outputs only")
    args = parser.parse_args()
    config, paths = load_core_config(args.config)
    features, labels = _features_and_labels(paths)
    feature_root = paths["cache_dir"] / "features"
    router_root = paths["cache_dir"] / "router"
    transform = FeatureTransform.load(router_root / "pca_scaler.joblib")
    base_table = observation_table_from_features(features, feature_root=feature_root)
    ridge_table = observation_table_from_features(
        features, feature_root=feature_root, transform=transform, predictor=RidgeRouter.load(router_root / "ridge_router.joblib")
    )
    mlp_table = observation_table_from_features(
        features, feature_root=feature_root, transform=transform, predictor=MLPRouter.load(router_root / "best_validation.safetensors")
    )
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
    threshold_rows: list[dict] = []
    output_rows: list[pd.DataFrame] = []
    for name, schedule_config in config["cost_models"].items():
        schedule = CostSchedule.from_values(name, schedule_config["cumulative_costs"])
        for budget in [float(item) for item in schedule_config["budgets"]]:
            # Survey does not enter the allocator heap; it remains an important Q0 reference at every cap.
            survey = survey_only(base_table, schedule, budget)
            survey_threshold = _record_threshold("survey_only", schedule, budget, None, survey, labels)
            threshold_rows.append(survey_threshold)
            output_rows.append(_result_rows("survey_only", schedule, budget, None, survey, labels, survey_threshold))
            for policy_name, table, policy in deterministic:
                result = allocate_deterministic(table, policy, schedule, budget)
                record = _record_threshold(policy_name, schedule, budget, None, result, labels)
                record["non_deployable"] = bool(policy.non_deployable)
                threshold_rows.append(record)
                output_rows.append(_result_rows(policy_name, schedule, budget, None, result, labels, record))
            for seed in config["policy"]["random_seeds"]:
                result = allocate_random(base_table, schedule, budget, int(seed))
                record = _record_threshold("random", schedule, budget, int(seed), result, labels)
                threshold_rows.append(record)
                output_rows.append(_result_rows("random", schedule, budget, int(seed), result, labels, record))
            if abs(budget - schedule.q3_cost) <= 1e-12:
                result = full_q3(base_table, schedule, budget)
                record = _record_threshold("full_q3", schedule, budget, None, result, labels)
                threshold_rows.append(record)
                output_rows.append(_result_rows("full_q3", schedule, budget, None, result, labels, record))
    artifact_root = paths["artifacts_dir"]
    threshold_path = artifact_root / "policy_thresholds.json"
    atomic_write_json(
        threshold_path,
        {"protocol_version": config["protocol_version"], "strict_decision_rule": "final_score > threshold", "thresholds": threshold_rows},
        overwrite=args.force,
    )
    write_parquet(pd.concat(output_rows, ignore_index=True), artifact_root / "policy_calibration_results.parquet", overwrite=args.force)
    print(f"Froze {len(threshold_rows)} policy-specific calibration thresholds")


if __name__ == "__main__":
    main()
