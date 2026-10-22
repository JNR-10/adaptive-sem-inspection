#!/usr/bin/env python3
"""Render deterministic, data-derived final-test diagnostic galleries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from _common import load_core_config
from run_final_benchmark import _current_lock_values, _verify_git
from seminspect.data.access import resolve_manifest_path
from seminspect.data.manifests import read_manifest
from seminspect.detector.feature_store import read_anomaly_map
from seminspect.evaluation.failure_cases import select_failure_cases
from seminspect.evaluation.locking import verify_test_lock
from seminspect.router.targets import next_resolution_gain
from seminspect.simulator.resolution import QualityLadder, canonical_q3_from_path, simulate_resolution
from seminspect.utils.config import config_hash
from seminspect.utils.io import ArtifactError, read_json
from seminspect.utils.tabular import read_parquet, write_parquet


def _features(paths: dict[str, Path]) -> pd.DataFrame:
    root = paths["cache_dir"] / "features"
    for path in (root / "features_final_test.parquet", root / "features_calibrated_final_test.parquet"):
        if path.is_file():
            frame = read_parquet(path)
            if frame["calibrated_probability"].notna().all():
                return frame
    raise ArtifactError("Final-test feature table is required and fully calibrated for failure galleries")


def _trajectories(path: Path, policy: str, budget: float) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if (
                item["policy"] == policy
                and item["cost_model"] == "pixel_information"
                and abs(float(item["budget"]) - budget) <= 1e-12
                and item.get("seed") is None
            ):
                output[str(item["image_id"])] = item["path"]
    return output


def _mask(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        values = np.asarray(source.convert("L"), dtype=np.uint8)
    return values >= 128


def _quality_gain(label: int, probabilities: np.ndarray, quality: int) -> float:
    if quality >= 3:
        return float("nan")
    return next_resolution_gain(label, probabilities[quality], probabilities[quality + 1])[0]


def _diagnostic_frame(
    features: pd.DataFrame,
    results: pd.DataFrame,
    trajectories: dict[str, list[dict]],
    uncertainty: pd.DataFrame,
) -> pd.DataFrame:
    probabilities = features.pivot(index="image_id", columns="quality_index", values="calibrated_probability")
    if probabilities.columns.tolist() != [0, 1, 2, 3]:
        raise ArtifactError("Failure galleries require a complete Q0–Q3 probability ladder")
    output = results.copy().set_index("image_id")
    output = output.join(probabilities.rename(columns={0: "p_q0", 1: "p_q1", 2: "p_q2", 3: "p_q3"}), how="inner")
    quality_by_uncertainty = uncertainty.set_index("image_id")["final_quality"]
    output["vor_uncertainty_quality_difference"] = (output["final_quality"] - quality_by_uncertainty).abs()
    output["q0_prediction"] = (output["p_q0"] > output["threshold"]).astype(int)
    clipped = output["p_q0"].clip(1e-6, 1.0 - 1e-6)
    output["q0_entropy"] = -clipped * np.log(clipped) - (1.0 - clipped) * np.log(1.0 - clipped)
    output["q3_minus_q0_probability"] = output["p_q3"] - output["p_q0"]
    output["absolute_q3_q0_probability_change"] = output["q3_minus_q0_probability"].abs()
    output["score_margin"] = output["final_score"] - output["threshold"]
    realized: list[float] = []
    unacquired: list[float] = []
    stopped: list[int] = []
    predicted_initial: list[float] = []
    for image_id, row in output.iterrows():
        probability = row[["p_q0", "p_q1", "p_q2", "p_q3"]].to_numpy(dtype=float)
        chosen = trajectories.get(str(image_id), [])
        selected_qualities = [int(item["quality"]) for item in chosen[:-1]]
        gains = [_quality_gain(int(row.label), probability, quality) for quality in selected_qualities]
        realized.append(float(min(gains)) if gains else 0.0)
        initial = next((item.get("predicted_gain") for item in chosen if int(item["quality"]) == 0), None)
        predicted_initial.append(float(initial) if initial is not None else float("nan"))
        next_gain = _quality_gain(int(row.label), probability, int(row.final_quality))
        unacquired.append(next_gain)
        stopped.append(int(int(row.final_quality) < 3 and np.isfinite(next_gain) and next_gain > 0))
    output["realized_gain"] = realized
    output["unacquired_true_gain"] = unacquired
    output["stopped_with_positive_unacquired_gain"] = stopped
    output["predicted_q0_gain"] = predicted_initial
    return output.reset_index()


def _safe_name(image_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", image_id)


def _render_item(
    row: pd.Series,
    feature_rows: pd.DataFrame,
    manifest_row: pd.Series,
    trajectories: dict[str, list[dict]],
    paths: dict[str, Path],
    ladder: QualityLadder,
    destination: Path,
) -> None:
    image_id = str(row.image_id)
    q3 = canonical_q3_from_path(resolve_manifest_path(manifest_row, paths["extracted_dir"]))
    mask: np.ndarray | None = None
    if int(row.label) == 1 and pd.notna(manifest_row.mask_path):
        mask = _mask(resolve_manifest_path(manifest_row, paths["extracted_dir"], column="mask_path"))
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((448, 448), Image.Resampling.NEAREST), dtype=bool)
    rows = feature_rows.sort_values("quality_index", kind="mergesort")
    trajectory = {int(item["quality"]): item for item in trajectories.get(image_id, [])}
    figure, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for quality in range(4):
        observation = simulate_resolution(q3, ladder[f"Q{quality}"])
        image_axis, map_axis = axes[0, quality], axes[1, quality]
        image_axis.imshow(observation, cmap="gray", vmin=0, vmax=1)
        if mask is not None:
            image_axis.contour(mask, levels=[0.5], colors="red", linewidths=0.6)
        state = trajectory.get(quality, {})
        probability = float(rows.iloc[quality].calibrated_probability)
        gain = state.get("predicted_gain")
        title = f"Q{quality}  p={probability:.3f}"
        if gain is not None:
            title += f"\n$\\hat G$={float(gain):.3f}"
        image_axis.set_title(title, fontsize=9)
        image_axis.axis("off")
        map_value = read_anomaly_map(
            paths["cache_dir"] / "features",
            str(rows.iloc[quality].anomaly_map_shard),
            int(rows.iloc[quality].anomaly_map_row),
        )
        map_axis.imshow(map_value, cmap="magma", interpolation="nearest")
        map_axis.set_title(f"Q{quality} patch anomaly map", fontsize=9)
        map_axis.axis("off")
    actions = ", ".join(str(item.get("action")) for item in trajectories.get(image_id, []))
    figure.suptitle(
        f"{image_id} | label={int(row.label)} | final Q{int(row.final_quality)} | "
        f"score={float(row.final_score):.3f}, threshold={float(row.threshold):.3f}, prediction={int(row.prediction)}\n"
        f"actions: {actions}",
        fontsize=10,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--policy", default="vor_mlp", choices=["vor_mlp", "vor_ridge"])
    parser.add_argument("--force", action="store_true", help="Replace exploratory diagnostic selection tables/images")
    args = parser.parse_args()
    lock = read_json(args.lock)
    config, paths = load_core_config(Path(lock["resolved_config_path"]))
    if lock.get("config_resolved_sha256") != config_hash(config):
        raise ArtifactError("Current configuration differs from the sealed primary protocol")
    verify_test_lock(args.lock, _current_lock_values(config, paths))
    _verify_git(lock, Path(config["_resolved_from"]).parents[2])
    budget = float(config["evaluation"]["primary_budget"])
    per_image = read_parquet(args.run_dir / "per_image_results.parquet")
    selector = per_image.loc[
        (per_image["policy"] == args.policy)
        & (per_image["cost_model"] == "pixel_information")
        & np.isclose(per_image["budget"], budget)
        & per_image["seed"].isna()
    ]
    uncertainty = per_image.loc[
        (per_image["policy"] == "uncertainty_entropy")
        & (per_image["cost_model"] == "pixel_information")
        & np.isclose(per_image["budget"], budget)
        & per_image["seed"].isna()
    ]
    if len(selector) != 1388 or len(uncertainty) != 1388:
        raise ArtifactError("Failure galleries require complete primary-cap final per-image results")
    features = _features(paths)
    trajectories = _trajectories(args.run_dir / "decision_trajectories.jsonl", args.policy, budget)
    diagnostic = _diagnostic_frame(features, selector, trajectories, uncertainty)
    selected = select_failure_cases(diagnostic, cap=20)
    manifest = read_manifest(paths["manifest_dir"] / "miic_splits.parquet")
    manifest = manifest.loc[manifest["split"].eq("final_test")].set_index("image_id")
    feature_groups = {str(image_id): group for image_id, group in features.groupby("image_id", sort=False)}
    ladder = QualityLadder.from_config(config)
    output = args.run_dir / "failure_cases" / args.policy
    write_parquet(diagnostic, output / "diagnostic_frame.parquet", overwrite=args.force)
    for category, frame in selected.items():
        table_path = output / category / "selection.parquet"
        write_parquet(frame, table_path, overwrite=args.force)
        for rank, row in enumerate(frame.itertuples(index=False), start=1):
            image_id = str(row.image_id)
            _render_item(
                pd.Series(row._asdict()),
                feature_groups[image_id],
                manifest.loc[image_id],
                trajectories,
                paths,
                ladder,
                output / category / f"{rank:02d}_{_safe_name(image_id)}.png",
            )
    print(f"Wrote deterministic {args.policy} failure galleries to {output}")


if __name__ == "__main__":
    main()
