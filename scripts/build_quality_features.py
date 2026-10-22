#!/usr/bin/env python3
"""Extract Q0–Q3 CLS/scalar/detector features into resumable compact shards."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from _common import add_config_argument, load_core_config
from seminspect.data.access import resolve_manifest_path
from seminspect.data.manifests import manifest_content_hash, read_manifest
from seminspect.detector.dinov2 import build_encoder, features_to_numpy
from seminspect.detector.calibration import ScoreCalibrator
from seminspect.detector.feature_store import FEATURE_COLUMNS, write_feature_shard
from seminspect.detector.memory_bank import MemoryBank
from seminspect.detector.scoring import score_encoded_batch
from seminspect.simulator.quality_metrics import quality_metrics
from seminspect.simulator.noise import StressScenario, apply_stress
from seminspect.simulator.resolution import QualityLadder, canonical_q3_from_path, encoder_tensor, simulate_resolution
from seminspect.utils.config import config_hash
from seminspect.utils.io import ArtifactError, atomic_write_json, canonical_json_sha256, path_tree_sha256, read_json
from seminspect.utils.io import file_sha256
from seminspect.evaluation.locking import ids_hash
from seminspect.utils.tabular import read_parquet, write_parquet


def _selected_rows(split: str, paths: dict[str, Path]) -> pd.DataFrame:
    if split == "policy_calibration":
        return read_parquet(paths["manifest_dir"] / "policy_calibration_lot.parquet")
    if split == "carinthia_secondary":
        frame = read_manifest(paths["manifest_dir"] / "carinthia_s.parquet")
        return frame.loc[frame["split"].eq("carinthia_secondary") & ~frame["quarantined"].astype(bool)].copy()
    frame = read_manifest(paths["manifest_dir"] / "miic_splits.parquet")
    return frame.loc[frame["split"].eq(split) & ~frame["quarantined"].astype(bool)].copy()


def _artifact_name(split: str, stress_scenario: str | None) -> str:
    return split if stress_scenario is None else f"{split}__stress_{stress_scenario}"


def _batches(frame: pd.DataFrame, size: int):
    for start in range(0, len(frame), size):
        yield start // size, frame.iloc[start : start + size]


def _load_banks(bank_root: Path, ladder: QualityLadder) -> tuple[dict[str, MemoryBank], dict[str, np.ndarray]]:
    banks: dict[str, MemoryBank] = {}
    references: dict[str, np.ndarray] = {}
    for quality in ladder:
        bank_path = bank_root / f"patch_bank_{quality.id}.npz"
        reference_path = bank_root / f"cls_reference_{quality.id}.npy"
        if not bank_path.is_file() or not reference_path.is_file():
            raise ArtifactError("All four matching quality memory banks must exist before detector scoring")
        banks[quality.id] = MemoryBank.load(bank_path)
        references[quality.id] = np.load(reference_path, allow_pickle=False).astype(np.float32)
    return banks, references


def _check_lock(lock_path: Path, config: dict, split_manifest_path: Path, final_ids: list[str]) -> dict:
    lock = read_json(lock_path)
    if lock.get("config_resolved_sha256") != config_hash(config):
        raise ArtifactError("Feature build configuration differs from the sealed test lock")
    if lock.get("split_manifest_sha256") != file_sha256(split_manifest_path):
        raise ArtifactError("Feature build split manifest differs from the sealed test lock")
    if lock.get("final_test_id_sha256") != ids_hash(final_ids):
        raise ArtifactError("Feature build final-test IDs differ from the sealed test lock")
    return lock


def _check_locked_detector_inputs(lock: dict, paths: dict[str, Path]) -> None:
    bank_root = paths["cache_dir"] / "memory_banks"
    calibrator_root = paths["cache_dir"] / "calibrators"
    memory_paths = [bank_root / f"patch_bank_Q{quality}.npz" for quality in range(4)]
    memory_paths += [bank_root / f"cls_reference_Q{quality}.npy" for quality in range(4)]
    memory_paths += [bank_root / f"cls_reference_ids_Q{quality}.npy" for quality in range(4)]
    memory_paths.append(bank_root / "memory_bank_metadata.json")
    memory_hash = path_tree_sha256(memory_paths, root=paths["cache_dir"])
    calibrator_hash = path_tree_sha256(
        [calibrator_root / f"score_calibrator_Q{quality}.joblib" for quality in range(4)],
        root=paths["cache_dir"],
    )
    if memory_hash != lock.get("memory_bank_sha256"):
        raise ArtifactError("Final-test feature extraction memory banks differ from the sealed test lock")
    if calibrator_hash != lock.get("calibrator_sha256"):
        raise ArtifactError("Final-test feature extraction score calibrators differ from the sealed test lock")


def _run_split(
    split: str,
    frame: pd.DataFrame,
    config: dict,
    paths: dict[str, Path],
    *,
    force: bool,
    lock: Path | None,
    stress_scenario: str | None,
) -> None:
    if frame.empty:
        raise ArtifactError(f"Requested split has no active rows: {split}")
    frame = frame.sort_values("image_id", kind="mergesort").reset_index(drop=True)
    manifest_hash = manifest_content_hash(frame)
    if stress_scenario is not None and stress_scenario not in config["stress_scenarios"]:
        raise ArtifactError(f"Unknown stress scenario: {stress_scenario}")
    if stress_scenario is not None and split == "normal_memory_train":
        raise ArtifactError("Stress observations must use the frozen clean normal memory banks, not rebuild them")
    sealed_lock: dict | None = None
    if lock is not None:
        split_path = paths["manifest_dir"] / "miic_splits.parquet"
        full_manifest = read_manifest(split_path)
        final_ids = full_manifest.loc[full_manifest["split"].eq("final_test"), "image_id"].astype(str).tolist()
        sealed_lock = _check_lock(lock, config, split_path, final_ids)
        _check_locked_detector_inputs(sealed_lock, paths)
    feature_root = paths["cache_dir"] / "features"
    artifact_name = _artifact_name(split, stress_scenario)
    output_path = feature_root / f"features_{artifact_name}.parquet"
    metadata_path = feature_root / f"features_{artifact_name}.metadata.json"
    if output_path.exists() and metadata_path.exists() and not force:
        metadata = read_json(metadata_path)
        if metadata.get("source_manifest_hash") != manifest_hash or metadata.get("config_hash") != config_hash(config):
            raise ArtifactError(f"Existing feature table is incompatible with current input/config: {output_path}")
        if sealed_lock is not None and metadata.get("weight_sha256") != sealed_lock.get("encoder_weight_sha256"):
            raise ArtifactError("Existing final-test feature table was built with weights differing from the test lock")
        print(f"Reusing complete feature table for {artifact_name}: {output_path}")
        return
    if output_path.exists() and not force:
        raise ArtifactError(f"Incomplete feature artifact exists for {split}; inspect it or use --force in development")
    ladder = QualityLadder.from_config(config)
    detector = config["detector"]
    bank_root = paths["cache_dir"] / "memory_banks"
    embeddings_only = split == "normal_memory_train" and not (bank_root / "patch_bank_Q0.npz").is_file()
    banks: dict[str, MemoryBank] = {}
    references: dict[str, np.ndarray] = {}
    if not embeddings_only:
        banks, references = _load_banks(bank_root, ladder)
    calibrator_root = paths["cache_dir"] / "calibrators"
    calibrators: dict[str, ScoreCalibrator] = {}
    if not embeddings_only and all((calibrator_root / f"score_calibrator_{quality.id}.joblib").is_file() for quality in ladder):
        calibrators = {
            quality.id: ScoreCalibrator.load(calibrator_root / f"score_calibrator_{quality.id}.joblib") for quality in ladder
        }
    encoder = build_encoder(
        detector,
        device=config["runtime"]["device"],
        weight_path=os.environ.get("DINO_WEIGHT_PATH") or None,
    ).load()
    if sealed_lock is not None and encoder.weight_sha256 != sealed_lock.get("encoder_weight_sha256"):
        raise ArtifactError("Final-test feature extraction encoder weights differ from the sealed test lock")
    stress = (
        StressScenario.from_config(stress_scenario, config["stress_scenarios"][stress_scenario])
        if stress_scenario is not None
        else None
    )
    simulator_payload = {"image": config["image"], "qualities": config["qualities"]}
    if stress_scenario is not None:
        simulator_payload["stress_scenario"] = {stress_scenario: config["stress_scenarios"][stress_scenario]}
    simulator_hash = canonical_json_sha256(simulator_payload)
    records: list[dict] = []
    for quality in ladder:
        for part, batch in _batches(frame, int(config["runtime"]["feature_shard_size"])):
            observations = []
            metrics = []
            for row in batch.itertuples(index=False):
                q3 = canonical_q3_from_path(resolve_manifest_path(row, paths["extracted_dir"]))
                observation = simulate_resolution(q3, quality)
                if stress is not None:
                    observation = apply_stress(
                        observation,
                        image_id=str(row.image_id),
                        quality_id=quality.id,
                        scenario=stress,
                        global_seed=int(config["seed"]),
                    )
                observations.append(observation)
                metrics.append(quality_metrics(observation))
            import torch

            tensors = torch.cat([encoder_tensor(observation, device=config["runtime"]["device"]) for observation in observations], dim=0)
            encoded = encoder.extract(tensors)
            patch_values, cls_values = features_to_numpy(encoded)
            if embeddings_only:
                scores = [None] * len(batch)
                maps = None
            else:
                scores = score_encoded_batch(
                    encoded.patch_descriptors,
                    encoded.cls_embeddings,
                    patch_bank=banks[quality.id],
                    cls_reference=torch.from_numpy(references[quality.id]),
                    patch_grid=encoded.patch_grid,
                    device=config["runtime"]["device"],
                    query_chunk=int(detector["query_chunk"]),
                    memory_chunk=int(detector["memory_chunk"]),
                    top_k=int(detector["score_top_k"]),
                )
                maps = np.stack([score.anomaly_map for score in scores]).astype(np.float32)
            shard_metadata = {
                "config_hash": config_hash(config),
                "source_manifest_hash": manifest_hash,
                "model_commit": detector["source_commit"],
                "dtype": "float32",
                "shape": f"[{len(batch)},384]",
            }
            embedding_reference, map_reference = write_feature_shard(
                feature_root,
                split=artifact_name,
                quality_id=quality.id,
                part=part,
                image_ids=batch["image_id"].astype(str).tolist(),
                cls_embeddings=cls_values,
                anomaly_maps=maps,
                metadata=shard_metadata,
                overwrite=force,
            )
            for local, row in enumerate(batch.itertuples(index=False)):
                score = scores[local]
                values = {
                    "image_id": str(row.image_id),
                    "quality_id": quality.id,
                    "quality_index": quality.index,
                    "effective_side": quality.effective_side,
                    "quality_cost": quality.cumulative_cost,
                    "simulator_hash": simulator_hash,
                    "detector_raw_score": None if score is None else score.raw_score,
                    "calibrated_probability": None if score is None or not calibrators else float(calibrators[quality.id].predict([score.raw_score])[0]),
                    "patch_score_max": None if score is None else score.patch_score_max,
                    "patch_score_top5_std": None if score is None else score.patch_score_top5_std,
                    "patch_score_q95": None if score is None else score.patch_score_q95,
                    "cls_normal_distance": None if score is None else score.cls_normal_distance,
                    **metrics[local],
                    "embedding_shard": embedding_reference.path,
                    "embedding_row": local,
                    "anomaly_map_shard": None if map_reference is None else map_reference.path,
                    "anomaly_map_row": None if map_reference is None else local,
                    "source_split": split,
                }
                records.append(values)
    features = pd.DataFrame.from_records(records)
    if len(features) != len(frame) * 4 or features.duplicated(["image_id", "quality_id"]).any():
        raise ArtifactError("Feature extraction did not produce exactly one row per image/quality")
    ordered = [*FEATURE_COLUMNS, "source_split"]
    write_parquet(features[ordered].sort_values(["image_id", "quality_index"], kind="mergesort"), output_path, overwrite=force)
    atomic_write_json(
        metadata_path,
        {
            "split": split,
            "row_count": len(features),
            "source_manifest_hash": manifest_hash,
            "config_hash": config_hash(config),
            "simulator_hash": simulator_hash,
            "model_commit": detector["source_commit"],
            "weight_sha256": encoder.weight_sha256,
            "embeddings_only": embeddings_only,
            "calibrators_applied": bool(calibrators),
            "stress_scenario": stress_scenario,
        },
        overwrite=force,
    )
    print(f"Wrote {len(features)} quality-feature rows for {artifact_name} to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--split", required=True, help="Comma-separated manifest split(s)")
    parser.add_argument(
        "--stress-scenario",
        help="Optional predeclared secondary stress scenario; uses frozen clean memory banks/calibrators",
    )
    parser.add_argument("--lock", type=Path, help="Required when extracting sealed final-test features")
    parser.add_argument("--force", action="store_true", help="Replace development feature outputs only")
    args = parser.parse_args()
    config, paths = load_core_config(args.config)
    splits = [item.strip() for item in args.split.split(",") if item.strip()]
    if not splits:
        raise ValueError("At least one split is required")
    if "final_test" in splits and args.lock is None:
        raise ArtifactError("Final-test feature extraction requires --lock artifacts/test_lock.json")
    for split in splits:
        _run_split(
            split,
            _selected_rows(split, paths),
            config,
            paths,
            force=args.force,
            lock=args.lock,
            stress_scenario=args.stress_scenario,
        )


if __name__ == "__main__":
    main()
