#!/usr/bin/env python3
"""Write immutable core_v1 test_lock.json after all development choices are frozen."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from _common import add_config_argument, load_core_config
from seminspect.data.manifests import read_manifest
from seminspect.evaluation.locking import ids_hash
from seminspect.utils.config import config_hash
from seminspect.utils.io import ArtifactError, atomic_write_json, canonical_json_sha256, file_sha256, path_tree_sha256, read_json


def _environment_lock_sha256(project_root: Path) -> str:
    """Hash the dependency lock selected by the invoking pipeline."""

    configured = os.environ.get("SEMSPECT_ENVIRONMENT_LOCK")
    lock_path = Path(configured) if configured else project_root / "requirements-colab.lock"
    if not lock_path.is_file():
        raise ArtifactError(f"Environment lock does not exist: {lock_path}")
    return file_sha256(lock_path)


def _git_state(project_root: Path, artifacts: Path, *, force: bool) -> tuple[str, str | None]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise ArtifactError("Sealed final runs require a committed Git revision") from exc
    dirty = subprocess.run(
        ["git", "-C", str(project_root), "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    if not dirty:
        return commit, None
    patch = subprocess.run(
        ["git", "-C", str(project_root), "diff", "--binary"], capture_output=True, check=True
    ).stdout
    patch_path = artifacts / "core_v1_working_tree.patch"
    from seminspect.utils.io import atomic_write_bytes

    atomic_write_bytes(patch_path, patch, overwrite=force)
    return commit, file_sha256(patch_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--force", action="store_true", help="Replace a development lock only; never replace an analyzed final lock")
    args = parser.parse_args()
    config, paths = load_core_config(args.config)
    project_root = Path(config["_resolved_from"]).parents[2]
    artifacts = paths["artifacts_dir"]
    lock_path = artifacts / "test_lock.json"
    if lock_path.exists() and not args.force:
        raise ArtifactError(f"Refusing to overwrite existing test lock: {lock_path}")
    split_path = paths["manifest_dir"] / "miic_splits.parquet"
    raw_path = paths["manifest_dir"] / "miic_raw.parquet"
    duplicate_path = artifacts / "audits" / "duplicate_decisions.parquet"
    threshold_path = artifacts / "policy_thresholds.json"
    bank_root = paths["cache_dir"] / "memory_banks"
    calibrator_root = paths["cache_dir"] / "calibrators"
    router_root = paths["cache_dir"] / "router"
    memory_metadata_path = bank_root / "memory_bank_metadata.json"
    required = [
        split_path,
        raw_path,
        duplicate_path,
        threshold_path,
        memory_metadata_path,
        calibrator_root / "calibration_reliability.parquet",
        router_root / "pca_scaler.joblib",
        router_root / "best_validation.safetensors",
    ]
    required += [bank_root / f"patch_bank_Q{quality}.npz" for quality in range(4)]
    required += [bank_root / f"cls_reference_Q{quality}.npy" for quality in range(4)]
    required += [bank_root / f"cls_reference_ids_Q{quality}.npy" for quality in range(4)]
    required += [calibrator_root / f"score_calibrator_Q{quality}.joblib" for quality in range(4)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ArtifactError(f"Cannot freeze protocol; required artifacts are missing: {missing}")
    memory_metadata = read_json(memory_metadata_path)
    if not memory_metadata.get("weight_sha256"):
        raise ArtifactError("DINO weight SHA-256 is required before final test locking; set DINO_WEIGHT_PATH")
    manifest = read_manifest(split_path)
    final_ids = manifest.loc[manifest["split"].eq("final_test"), "image_id"].astype(str).tolist()
    if len(final_ids) != 1388:
        raise ArtifactError(f"Final-test manifest has {len(final_ids)} IDs; expected 1,388")
    commit, diff_sha = _git_state(project_root, artifacts, force=args.force)
    resolved_path = artifacts / "core_v1_resolved_config.json"
    atomic_write_json(resolved_path, config, overwrite=args.force)
    lock = {
        "protocol_version": config["protocol_version"],
        "git_commit": commit,
        "git_diff_sha256": diff_sha,
        "environment_lock_sha256": _environment_lock_sha256(project_root),
        "dataset_manifest_sha256": file_sha256(raw_path),
        "split_manifest_sha256": file_sha256(split_path),
        "duplicate_decisions_sha256": file_sha256(duplicate_path),
        "simulator_config_sha256": canonical_json_sha256({"image": config["image"], "qualities": config["qualities"], "stress_scenarios": config["stress_scenarios"]}),
        "detector_config_sha256": canonical_json_sha256({"detector": config["detector"], "calibration": config["calibration"]}),
        "encoder_weight_sha256": memory_metadata["weight_sha256"],
        "memory_bank_sha256": path_tree_sha256(
            [bank_root / f"patch_bank_Q{quality}.npz" for quality in range(4)]
            + [bank_root / f"cls_reference_Q{quality}.npy" for quality in range(4)]
            + [bank_root / f"cls_reference_ids_Q{quality}.npy" for quality in range(4)]
            + [memory_metadata_path],
            root=paths["cache_dir"],
        ),
        "calibrator_sha256": path_tree_sha256([calibrator_root / f"score_calibrator_Q{quality}.joblib" for quality in range(4)], root=paths["cache_dir"]),
        "pca_and_scaler_sha256": file_sha256(router_root / "pca_scaler.joblib"),
        "router_checkpoint_sha256": file_sha256(router_root / "best_validation.safetensors"),
        "policy_config_sha256": canonical_json_sha256({"cost_models": config["cost_models"], "policy": config["policy"]}),
        "policy_thresholds_sha256": file_sha256(threshold_path),
        "final_test_id_sha256": ids_hash(final_ids),
        "budget_list": config["evaluation"]["budgets"],
        "config_resolved_sha256": config_hash(config),
        "resolved_config_path": str(resolved_path),
        "project_root": str(project_root),
    }
    atomic_write_json(lock_path, lock, overwrite=args.force)
    print(f"Sealed {config['protocol_version']} final-test lock: {lock_path}")


if __name__ == "__main__":
    main()
