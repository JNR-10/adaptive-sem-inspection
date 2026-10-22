#!/usr/bin/env python3
"""Fit four score calibrators only on the sealed synthetic score-calibration split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import add_config_argument, load_core_config
from seminspect.data.manifests import read_manifest
from seminspect.detector.calibration import ScoreCalibrator, reliability_bins
from seminspect.utils.io import ArtifactError, atomic_write_json
from seminspect.utils.tabular import read_parquet, write_parquet


def _labels(paths: dict[str, Path]) -> pd.DataFrame:
    manifest = read_manifest(paths["manifest_dir"] / "miic_splits.parquet")
    return manifest[["image_id", "label", "split"]].copy()


def _calibrate_table(frame: pd.DataFrame, calibrators: dict[str, ScoreCalibrator]) -> pd.DataFrame:
    output = frame.copy()
    output["calibrated_probability"] = None
    for quality_id, indices in output.groupby("quality_id", sort=True).groups.items():
        subset = output.loc[indices, "detector_raw_score"]
        if subset.isna().any():
            # The embeddings-only normal-memory cache deliberately has no detector scores.
            continue
        output.loc[indices, "calibrated_probability"] = calibrators[str(quality_id)].predict(subset.to_numpy())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--force", action="store_true", help="Replace development calibrators/features only")
    args = parser.parse_args()
    config, paths = load_core_config(args.config)
    feature_root = paths["cache_dir"] / "features"
    calibration_path = feature_root / "features_score_calibration.parquet"
    if not calibration_path.is_file():
        raise ArtifactError("Score-calibration features are missing; run build_quality_features first")
    features = read_parquet(calibration_path).merge(_labels(paths), on="image_id", how="left", validate="many_to_one")
    if features["label"].isna().any() or not features["split"].eq("score_calibration").all():
        raise ArtifactError("Score calibrator input must contain only labeled score_calibration rows")
    calibrator_root = paths["cache_dir"] / "calibrators"
    calibrators: dict[str, ScoreCalibrator] = {}
    diagnostics: list[dict] = []
    reliability: list[dict] = []
    for quality_id, group in features.groupby("quality_id", sort=True):
        if len(group) != 524:
            raise ArtifactError(f"{quality_id} has {len(group)} calibration rows; expected 524")
        calibrator = ScoreCalibrator.fit(
            group["detector_raw_score"].to_numpy(),
            group["label"].to_numpy(),
            C=float(config["calibration"]["C"]),
            max_iter=int(config["calibration"]["max_iter"]),
            tol=float(config["calibration"]["tol"]),
            probability_clip=float(config["calibration"]["probability_clip"]),
        )
        calibrator.save(calibrator_root / f"score_calibrator_{quality_id}.joblib", overwrite=args.force)
        calibrators[str(quality_id)] = calibrator
        diagnostics.append({"split": "score_calibration", "quality_id": quality_id, **calibrator.diagnostics(group["detector_raw_score"].to_numpy(), group["label"].to_numpy())})
        reliability.extend(
            {
                "split": "score_calibration",
                "quality_id": quality_id,
                **row,
            }
            for row in reliability_bins(
                calibrator.predict(group["detector_raw_score"].to_numpy()),
                group["label"].to_numpy(),
                bins=int(config["calibration"]["ece_bins"]),
            )
        )
    # Apply frozen calibration to every already-complete feature table; no model is refit here.
    raw_tables = sorted(
        path for path in feature_root.glob("features_*.parquet") if not path.name.startswith("features_calibrated_")
    )
    for raw_path in raw_tables:
        split = raw_path.stem[len("features_") :]
        table = read_parquet(raw_path)
        if table["detector_raw_score"].isna().all():
            continue
        calibrated = _calibrate_table(table, calibrators)
        output_path = feature_root / f"features_calibrated_{split}.parquet"
        write_parquet(calibrated, output_path, overwrite=args.force)
        if split == "router_validation":
            merged = calibrated.merge(_labels(paths), on="image_id", how="left", validate="many_to_one")
            for quality_id, group in merged.groupby("quality_id", sort=True):
                diagnostics.append(
                    {
                        "split": "router_validation",
                        "quality_id": quality_id,
                        **calibrators[str(quality_id)].diagnostics(group["detector_raw_score"].to_numpy(), group["label"].to_numpy()),
                    }
                )
                reliability.extend(
                    {
                        "split": "router_validation",
                        "quality_id": quality_id,
                        **row,
                    }
                    for row in reliability_bins(
                        group["calibrated_probability"].to_numpy(dtype=float),
                        group["label"].to_numpy(),
                        bins=int(config["calibration"]["ece_bins"]),
                    )
                )
    atomic_write_json(calibrator_root / "calibration_diagnostics.json", {"rows": diagnostics}, overwrite=args.force)
    write_parquet(pd.DataFrame(reliability), calibrator_root / "calibration_reliability.parquet", overwrite=args.force)
    print(f"Fit and persisted {len(calibrators)} quality-specific score calibrators")


if __name__ == "__main__":
    main()
