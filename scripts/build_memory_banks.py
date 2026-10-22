#!/usr/bin/env python3
"""Build the four quality-specific normal patch/CLS memory banks exactly once."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from _common import add_config_argument, load_core_config
from seminspect.data.access import resolve_manifest_path
from seminspect.data.manifests import manifest_content_hash, read_manifest
from seminspect.detector.dinov2 import build_encoder, features_to_numpy
from seminspect.detector.memory_bank import ReservoirSampler, descriptor_coordinates
from seminspect.simulator.resolution import QualityLadder, canonical_q3_from_path, encoder_tensor, simulate_resolution
from seminspect.utils.config import config_hash
from seminspect.utils.io import ArtifactError, atomic_write_json


def _write_npy_atomic(path: Path, values: np.ndarray, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ArtifactError(f"Refusing to overwrite memory reference: {path}")
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _batches(frame, size: int):
    for start in range(0, len(frame), size):
        yield frame.iloc[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--force", action="store_true", help="Replace development memory-bank artifacts only")
    args = parser.parse_args()
    config, paths = load_core_config(args.config)
    manifest = read_manifest(paths["manifest_dir"] / "miic_splits.parquet")
    memory = manifest.loc[
        manifest["split"].eq("normal_memory_train") & ~manifest["quarantined"].astype(bool)
    ].sort_values("image_id", kind="mergesort")
    expected_count = int(config["detector"]["normal_images"])
    if len(memory) != expected_count or not memory["label"].astype("Int64").eq(0).all():
        raise ArtifactError("Memory bank must contain exactly 8,000 non-quarantined normal_memory_train images")
    ladder = QualityLadder.from_config(config)
    bank_root = paths["cache_dir"] / "memory_banks"
    final_paths = [bank_root / f"patch_bank_{quality.id}.npz" for quality in ladder]
    final_paths += [bank_root / f"cls_reference_{quality.id}.npy" for quality in ladder]
    final_paths += [bank_root / f"cls_reference_ids_{quality.id}.npy" for quality in ladder]
    final_paths.append(bank_root / "memory_bank_metadata.json")
    if all(path.exists() for path in final_paths) and not args.force:
        print(f"Reusing complete four-quality memory banks in {bank_root}")
        return
    if any(path.exists() for path in final_paths) and not args.force:
        raise ArtifactError("Incomplete memory-bank set exists; inspect it or use --force for development artifacts")
    detector = config["detector"]
    encoder = build_encoder(
        detector,
        device=config["runtime"]["device"],
        weight_path=os.environ.get("DINO_WEIGHT_PATH") or None,
    ).load()
    source_hash = manifest_content_hash(memory)
    metadata = {
        "protocol_version": config["protocol_version"],
        "config_hash": config_hash(config),
        "source_manifest_hash": source_hash,
        "model_commit": detector["source_commit"],
        "encoder_type": detector.get("encoder_type", "dinov2"),
        "weight_sha256": encoder.weight_sha256,
        "memory_image_count": len(memory),
    }
    for quality in ladder:
        sampler: ReservoirSampler | None = None
        expected_grid: tuple[int, int] | None = None
        cls = np.empty((len(memory), 384), dtype=np.float32)
        for offset, batch in enumerate(_batches(memory, int(config["runtime"]["feature_batch_size"]))):
            observations = []
            for row in batch.itertuples(index=False):
                q3 = canonical_q3_from_path(resolve_manifest_path(row, paths["extracted_dir"]))
                observations.append(simulate_resolution(q3, quality))
            import torch

            tensors = torch.cat([encoder_tensor(item, device=config["runtime"]["device"]) for item in observations], dim=0)
            output = encoder.extract(tensors)
            patches, embeddings = features_to_numpy(output)
            if sampler is None:
                expected_grid = output.patch_grid
                if tuple(int(value) for value in detector["query_grid"]) != expected_grid:
                    raise ArtifactError(
                        f"Configured query grid {detector['query_grid']} differs from encoder output {expected_grid}"
                    )
                sampler = ReservoirSampler(
                    capacity=int(detector["descriptors_per_quality"]),
                    embedding_dim=int(patches.shape[-1]),
                    seed=int(config["seed"]) + quality.index,
                )
            if output.patch_grid != expected_grid:
                raise ArtifactError("Frozen encoder changed its patch grid within a memory-bank run")
            if patches.shape[1] != output.patch_grid[0] * output.patch_grid[1] or patches.shape[-1] != 384:
                raise ArtifactError("Encoder output does not match the expected patch descriptor contract")
            coordinates = descriptor_coordinates(output.patch_grid)
            for local, row in enumerate(batch.itertuples(index=False)):
                assert sampler is not None
                sampler.add(patches[local], str(row.image_id), coordinates)
            start = offset * int(config["runtime"]["feature_batch_size"])
            cls[start : start + len(batch)] = embeddings
        if sampler is None or expected_grid is None:
            raise ArtifactError("No descriptors were emitted while building a memory bank")
        metadata[f"patch_grid_{quality.id}"] = list(expected_grid)
        bank = sampler.finish()
        bank.save(bank_root / f"patch_bank_{quality.id}.npz", overwrite=args.force)
        _write_npy_atomic(bank_root / f"cls_reference_{quality.id}.npy", cls, overwrite=args.force)
        _write_npy_atomic(
            bank_root / f"cls_reference_ids_{quality.id}.npy",
            memory["image_id"].to_numpy(dtype="U128"),
            overwrite=args.force,
        )
    atomic_write_json(bank_root / "memory_bank_metadata.json", metadata, overwrite=args.force)
    print(f"Built four quality-specific banks from {len(memory)} normal images in {bank_root}")


if __name__ == "__main__":
    main()
