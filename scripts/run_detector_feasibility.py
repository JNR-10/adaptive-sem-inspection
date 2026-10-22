#!/usr/bin/env python3
"""Evaluate the prespecified resolution-feasibility gate before router training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from _common import add_config_argument, load_core_config
from seminspect.data.manifests import read_manifest
from seminspect.detector.calibration import ScoreCalibrator
from seminspect.utils.io import ArtifactError, atomic_write_json
from seminspect.utils.tabular import read_parquet, write_parquet


def _validation_rows(paths: dict[str, Path]) -> pd.DataFrame:
    feature_path = paths["cache_dir"] / "features" / "features_calibrated_router_validation.parquet"
    if not feature_path.is_file():
        raise ArtifactError("Router-validation calibrated features are required for the detector feasibility gate")
    labels = read_manifest(paths["manifest_dir"] / "miic_splits.parquet")
    labels = labels.loc[labels["split"].eq("router_validation"), ["image_id", "label"]]
    features = read_parquet(feature_path).merge(labels, on="image_id", how="inner", validate="many_to_one")
    if len(features) != 544 or features["calibrated_probability"].isna().any():
        raise ArtifactError("Feasibility gate requires 136 fully calibrated router-validation ladders")
    counts = features.groupby("quality_id")["label"].agg(["size", "sum"])
    if not ((counts["size"] == 136) & (counts["sum"] == 68)).all():
        raise ArtifactError("Each quality must contain 68 original and 68 synthetic validation examples")
    return features


def _rows(features: pd.DataFrame, paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for quality_id, group in features.groupby("quality_id", sort=True):
        labels = group["label"].to_numpy(dtype=int)
        scores = group["calibrated_probability"].to_numpy(dtype=float)
        calibrator = ScoreCalibrator.load(paths["cache_dir"] / "calibrators" / f"score_calibrator_{quality_id}.joblib")
        rows.append(
            {
                "quality_id": str(quality_id),
                "auroc": float(roc_auc_score(labels, scores)),
                "auprc": float(average_precision_score(labels, scores)),
                **calibrator.diagnostics(group["detector_raw_score"].to_numpy(dtype=float), labels),
            }
        )
    output = pd.DataFrame(rows)
    if output["quality_id"].tolist() != ["Q0", "Q1", "Q2", "Q3"]:
        raise ArtifactError("Feasibility gate requires an exact Q0–Q3 quality ladder")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--report-only", action="store_true", help="Write a failed gate report without exiting nonzero")
    parser.add_argument("--force", action="store_true", help="Replace development feasibility outputs")
    args = parser.parse_args()
    config, paths = load_core_config(args.config)
    features = _validation_rows(paths)
    table = _rows(features, paths)
    q0 = table.loc[table["quality_id"].eq("Q0")].iloc[0]
    q3 = table.loc[table["quality_id"].eq("Q3")].iloc[0]
    adjacent = [
        bool(table.iloc[index + 1]["auprc"] > table.iloc[index]["auprc"])
        for index in range(len(table) - 1)
    ]
    criteria = {
        "q3_minus_q0_auprc_at_least_0_03": bool(q3["auprc"] - q0["auprc"] >= 0.03),
        "at_least_one_adjacent_auprc_improvement": any(adjacent),
        "q3_auroc_not_more_than_0_01_below_q0": bool(q3["auroc"] >= q0["auroc"] - 0.01),
    }
    passed = all(criteria.values())
    artifact_root = paths["artifacts_dir"]
    write_parquet(table, artifact_root / "detector_feasibility.parquet", overwrite=args.force)
    atomic_write_json(
        artifact_root / "detector_feasibility.json",
        {
            "protocol_version": config["protocol_version"],
            "encoder_type": config["detector"].get("encoder_type", "dinov2"),
            "criteria": criteria,
            "passed": passed,
            "next_step": (
                "Proceed to router construction."
                if passed
                else "Do not train core_v1 router. Verify the pipeline, then run the separately named core_v1_resnet fallback or report a negative feasibility result."
            ),
        },
        overwrite=args.force,
    )
    print(f"Detector feasibility gate {'PASSED' if passed else 'FAILED'}: {artifact_root / 'detector_feasibility.json'}")
    if not passed and not args.report_only:
        raise ArtifactError("Detector feasibility gate failed; core_v1 router training is blocked")


if __name__ == "__main__":
    main()
