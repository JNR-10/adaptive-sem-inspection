#!/usr/bin/env python3
"""Run post-lock router-information and risk-preference ablations at the primary cap.

Detector-capacity/backbone variants are separate complete protocol variants
because they change cached detector observations.  This script covers the
ablations that reuse the locked detector features without leaking final labels
into training: Ridge versus MLP, five feature-group removals, and w1 risk
preference values 1/5/10/20.
"""

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
from seminspect.evaluation.thresholds import select_normal_fpr_threshold
from seminspect.policies.allocator import CostSchedule, allocate_deterministic
from seminspect.policies.runtime import observation_table_from_features
from seminspect.policies.vor import predicted_gain_policy
from seminspect.router.features import FeatureTransform, ROUTER_FEATURE_NAMES
from seminspect.router.mlp import train_mlp_router
from seminspect.router.targets import next_resolution_gain
from seminspect.utils.config import config_hash
from seminspect.utils.io import ArtifactError, atomic_write_json, read_json
from seminspect.utils.tabular import read_parquet, write_parquet


FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "remove_pca_embedding": tuple(name for name in ROUTER_FEATURE_NAMES if name.startswith("cls_pca_")),
    "remove_image_quality": (
        "intensity_mean",
        "intensity_std",
        "robust_dynamic_range",
        "laplacian_variance",
        "tenengrad",
        "highpass_mad",
    ),
    "remove_history": ("probability_delta", "raw_score_delta", "cls_cosine_change", "has_history"),
    "remove_uncertainty": ("binary_entropy",),
    "remove_current_quality": ("quality_is_q0", "quality_is_q1", "quality_is_q2"),
}


def _labels(paths: dict[str, Path], split: str) -> dict[str, int]:
    if split == "policy_calibration":
        frame = read_parquet(paths["manifest_dir"] / "policy_calibration_lot.parquet")
    else:
        frame = read_manifest(paths["manifest_dir"] / "miic_splits.parquet")
        frame = frame.loc[frame["split"].eq(split)]
    return {str(row.image_id): int(row.label) for row in frame.itertuples(index=False)}


def _router_matrix(paths: dict[str, Path], split: str) -> pd.DataFrame:
    path = paths["cache_dir"] / "router" / f"router_{split}.parquet"
    frame = read_parquet(path).sort_values(["image_id", "quality_index"], kind="mergesort").reset_index(drop=True)
    if not set(ROUTER_FEATURE_NAMES).issubset(frame.columns):
        raise ArtifactError(f"Router table lacks the sealed feature schema: {path}")
    return frame


def _targets_for_weight(paths: dict[str, Path], split: str, weight: float, matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feature_path = paths["cache_dir"] / "features" / f"features_calibrated_{split}.parquet"
    features = read_parquet(feature_path)
    labels = _labels(paths, split)
    rows: list[dict] = []
    for image_id, group in features.groupby("image_id", sort=True):
        ordered = group.sort_values("quality_index", kind="mergesort")
        if ordered["quality_index"].tolist() != [0, 1, 2, 3] or str(image_id) not in labels:
            raise ArtifactError(f"Invalid Q0–Q3 target ladder for {image_id}")
        probability = ordered["calibrated_probability"].to_numpy(dtype=float)
        for quality in range(3):
            gain, helpful = next_resolution_gain(
                labels[str(image_id)],
                probability[quality],
                probability[quality + 1],
                false_negative_weight=weight,
                false_positive_weight=1.0,
            )
            rows.append({"image_id": str(image_id), "quality_index": quality, "signed_gain": gain, "helpful_acquisition": helpful})
    target = pd.DataFrame(rows)
    aligned = matrix[["image_id", "quality_index"]].merge(target, on=["image_id", "quality_index"], how="left", validate="one_to_one")
    if aligned[["signed_gain", "helpful_acquisition"]].isna().any().any():
        raise ArtifactError(f"Could not align risk-preference targets for {split}")
    return aligned["signed_gain"].to_numpy(dtype=np.float32), aligned["helpful_acquisition"].to_numpy(dtype=np.float32)


def _table(paths: dict[str, Path], split: str) -> pd.DataFrame:
    root = paths["cache_dir"] / "features"
    for path in (root / f"features_{split}.parquet", root / f"features_calibrated_{split}.parquet"):
        if path.is_file():
            frame = read_parquet(path)
            if frame["calibrated_probability"].notna().all():
                return frame
    raise ArtifactError(f"No fully calibrated feature table exists for {split}")


def _evaluate(
    *,
    router,
    keep_mask: np.ndarray,
    transform: FeatureTransform,
    calibration_features: pd.DataFrame,
    calibration_labels: dict[str, int],
    final_features: pd.DataFrame,
    final_labels: dict[str, int],
    paths: dict[str, Path],
    schedule: CostSchedule,
    budget: float,
) -> tuple[dict, float]:
    root = paths["cache_dir"] / "features"
    calibration_table = observation_table_from_features(
        calibration_features,
        feature_root=root,
        transform=transform,
        predictor=router,
        feature_keep_mask=keep_mask,
    )
    final_table = observation_table_from_features(
        final_features,
        feature_root=root,
        transform=transform,
        predictor=router,
        feature_keep_mask=keep_mask,
    )
    policy = predicted_gain_policy("vor_mlp_ablation")
    calibration = allocate_deterministic(calibration_table, policy, schedule, budget)
    normals = [(image_id, calibration.final_scores[image_id]) for image_id in sorted(calibration_labels) if calibration_labels[image_id] == 0]
    threshold = select_normal_fpr_threshold(normals, alpha=0.05)
    final = allocate_deterministic(final_table, policy, schedule, budget)
    result = policy_result_frame(final_labels, final.final_scores, final.final_qualities, threshold, policy="vor_mlp_ablation", budget=budget, actual_cost=final.actual_cost)
    return summarize_policy_results(result), threshold.value


def _train(config: dict, train_x, train_gain, train_helpful, validation_x, validation_gain, validation_helpful, seed: int):
    router = config["router"]
    return train_mlp_router(
        train_x,
        train_gain,
        train_helpful,
        validation_x,
        validation_gain,
        validation_helpful,
        seed=seed,
        input_dim=int(router["input_dim"]),
        hidden_dims=tuple(router["hidden_dims"]),
        dropout=float(router["dropout"]),
        batch_size=int(router["batch_size"]),
        learning_rate=float(router["learning_rate"]),
        weight_decay=float(router["weight_decay"]),
        max_epochs=int(router["max_epochs"]),
        patience=int(router["patience"]),
        gradient_clip_norm=float(router["gradient_clip_norm"]),
        huber_beta=float(router["huber_beta"]),
        auxiliary_weight=float(router["auxiliary_weight"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="Replace exploratory ablation outputs only")
    args = parser.parse_args()
    lock = read_json(args.lock)
    config, paths = load_core_config(Path(lock["resolved_config_path"]))
    if lock.get("config_resolved_sha256") != config_hash(config):
        raise ArtifactError("Current configuration differs from the sealed primary protocol")
    verify_test_lock(args.lock, _current_lock_values(config, paths))
    _verify_git(lock, Path(config["_resolved_from"]).parents[2])
    if not (args.run_dir / "primary_endpoint.json").is_file():
        raise ArtifactError("Ablations require a completed immutable primary final benchmark")
    train = _router_matrix(paths, "train")
    validation = _router_matrix(paths, "validation")
    transform = FeatureTransform.load(paths["cache_dir"] / "router" / "pca_scaler.joblib")
    selected_seed = int(read_json(paths["cache_dir"] / "router" / "router_selection.json")["selected_seed"])
    calibration_features = _table(paths, "policy_calibration")
    final_features = _table(paths, "final_test")
    calibration_labels = _labels(paths, "policy_calibration")
    final_labels = _labels(paths, "final_test")
    schedule = CostSchedule.from_values("pixel_information", config["cost_models"]["pixel_information"]["cumulative_costs"])
    budget = float(config["evaluation"]["primary_budget"])
    output_rows: list[dict] = []
    output_root = args.run_dir / "secondary" / "ablations"
    output_path = output_root / "router_risk_ablations.parquet"
    if output_path.is_file() and not args.force:
        print(f"[resume] Reusing complete exploratory ablation table: {output_path}")
        return
    full_mask = np.ones(53, dtype=bool)
    experiments: list[tuple[str, np.ndarray, float]] = []
    for name, removed in FEATURE_GROUPS.items():
        mask = full_mask.copy()
        mask[[ROUTER_FEATURE_NAMES.index(feature) for feature in removed]] = False
        experiments.append((name, mask, 10.0))
    # Evaluate the risk targets directly from frozen development probabilities;
    # the 10x case is retained as a reproducibility check of the primary target.
    experiments.extend((f"risk_w1_{int(weight)}", full_mask.copy(), float(weight)) for weight in config["evaluation"]["risk_weights"])
    train_x_base = train.loc[:, ROUTER_FEATURE_NAMES].to_numpy(dtype=np.float32)
    validation_x_base = validation.loc[:, ROUTER_FEATURE_NAMES].to_numpy(dtype=np.float32)
    target_cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for _, mask, weight in experiments:
        if weight not in target_cache:
            train_gain, train_helpful = _targets_for_weight(paths, "router_train", weight, train)
            val_gain, val_helpful = _targets_for_weight(paths, "router_validation", weight, validation)
            target_cache[weight] = (train_gain, train_helpful, val_gain, val_helpful)
        train_gain, train_helpful, val_gain, val_helpful = target_cache[weight]
        result = _train(
            config,
            train_x_base * mask,
            train_gain,
            train_helpful,
            validation_x_base * mask,
            val_gain,
            val_helpful,
            selected_seed,
        )
        metrics, threshold = _evaluate(
            router=result.router,
            keep_mask=mask,
            transform=transform,
            calibration_features=calibration_features,
            calibration_labels=calibration_labels,
            final_features=final_features,
            final_labels=final_labels,
            paths=paths,
            schedule=schedule,
            budget=budget,
        )
        checkpoint = output_root / "models" / f"{name}.safetensors"
        # A runtime interruption can happen after this checkpoint was written
        # but before the aggregate table was completed.  The run is
        # deterministic under the sealed lock, so retain that immutable
        # checkpoint on resume and rebuild the pending table around it.
        if checkpoint.is_file() and not args.force:
            print(f"[resume] Keeping existing ablation checkpoint: {checkpoint.name}")
        else:
            result.router.save(
                checkpoint,
                overwrite=args.force,
                metadata={"ablation": name, "seed": selected_seed, "risk_weight": weight, "kept_features": [name for name, keep in zip(ROUTER_FEATURE_NAMES, mask, strict=True) if keep]},
            )
        output_rows.append(
            {
                "ablation": name,
                "selected_seed": selected_seed,
                "risk_weight": weight,
                "removed_features": ",".join(name for name, keep in zip(ROUTER_FEATURE_NAMES, mask, strict=True) if not keep),
                "validation_mae": float(np.mean(np.abs(result.router.predict_gain(validation_x_base * mask) - val_gain))),
                "threshold": threshold,
                **metrics,
            }
        )
    primary = read_parquet(args.run_dir / "metrics.parquet")
    comparison = primary.loc[
        (primary["cost_model"] == "pixel_information")
        & (primary["budget"] == budget)
        & primary["seed"].isna()
        & primary["policy"].isin(["vor_ridge", "vor_mlp"])
    ].copy()
    comparison["ablation"] = comparison["policy"].map({"vor_ridge": "ridge_vs_mlp_ridge", "vor_mlp": "ridge_vs_mlp_mlp"})
    comparison["selected_seed"] = selected_seed
    comparison["risk_weight"] = 10.0
    output = pd.concat([pd.DataFrame(output_rows), comparison], ignore_index=True, sort=False)
    write_parquet(output, output_path, overwrite=args.force)
    atomic_write_json(
        output_root / "metadata.json",
        {
            "exploratory": True,
            "primary_budget": budget,
            "selected_seed": selected_seed,
            "feature_groups": {name: list(features) for name, features in FEATURE_GROUPS.items()},
            "risk_weights": list(config["evaluation"]["risk_weights"]),
            "detector_variant_note": "Run backbone/memory/patch-pooling/top-k variants as separately named complete protocol configurations; they alter detector observations.",
        },
        overwrite=args.force,
    )
    print(f"Wrote exploratory router/risk ablations to {output_root}")


if __name__ == "__main__":
    main()
