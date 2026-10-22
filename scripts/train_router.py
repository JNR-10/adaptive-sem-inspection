#!/usr/bin/env python3
"""Train required Ridge and five deterministic MLP Value-of-Resolution routers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from _common import add_config_argument, load_core_config
from seminspect.router.features import ROUTER_FEATURE_NAMES
from seminspect.router.mlp import train_mlp_router
from seminspect.router.ridge import RidgeRouter
from seminspect.utils.io import ArtifactError, atomic_write_json
from seminspect.utils.tabular import read_parquet


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = read_parquet(path)
    required = set(ROUTER_FEATURE_NAMES) | {"signed_gain", "helpful_acquisition"}
    if not required.issubset(frame.columns):
        raise ArtifactError(f"Router dataset misses columns: {required - set(frame.columns)}")
    return (
        frame.loc[:, ROUTER_FEATURE_NAMES].to_numpy(dtype=np.float32),
        frame["signed_gain"].to_numpy(dtype=np.float32),
        frame["helpful_acquisition"].to_numpy(dtype=np.float32),
    )


def _torch_save_atomic(value, path: Path, *, overwrite: bool) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ArtifactError(f"Refusing to overwrite router training state: {path}")
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    torch.save(value, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--force", action="store_true", help="Replace development router checkpoints only")
    args = parser.parse_args()
    config, paths = load_core_config(args.config)
    router_config = config["router"]
    expected_seeds = list(router_config["seeds"])
    if list(args.seeds) != expected_seeds:
        raise ArtifactError(f"core_v1 must train exactly the prespecified seeds {expected_seeds}")
    router_root = paths["cache_dir"] / "router"
    train_x, train_gain, train_helpful = _load(router_root / "router_train.parquet")
    validation_x, validation_gain, validation_helpful = _load(router_root / "router_validation.parquet")
    expected_train_rows = int(config["splits"]["router_train_pairs"]) * 2 * 3
    expected_validation_rows = int(config["splits"]["router_validation_pairs"]) * 2 * 3
    if train_x.shape != (expected_train_rows, 53) or validation_x.shape != (expected_validation_rows, 53):
        raise ArtifactError(
            "Router matrices must have exactly "
            f"{expected_train_rows}/{expected_validation_rows} rows and 53 columns"
        )
    ridge = RidgeRouter.fit_select(
        train_x,
        train_gain,
        validation_x,
        validation_gain,
        alphas=router_config["ridge_alphas"],
    )
    ridge.save(router_root / "ridge_router.joblib", overwrite=args.force)
    outcomes = []
    results = []
    for seed in args.seeds:
        result = train_mlp_router(
            train_x,
            train_gain,
            train_helpful,
            validation_x,
            validation_gain,
            validation_helpful,
            seed=int(seed),
            input_dim=int(router_config["input_dim"]),
            hidden_dims=tuple(router_config["hidden_dims"]),
            dropout=float(router_config["dropout"]),
            batch_size=int(router_config["batch_size"]),
            learning_rate=float(router_config["learning_rate"]),
            weight_decay=float(router_config["weight_decay"]),
            max_epochs=int(router_config["max_epochs"]),
            patience=int(router_config["patience"]),
            gradient_clip_norm=float(router_config["gradient_clip_norm"]),
            huber_beta=float(router_config["huber_beta"]),
            auxiliary_weight=float(router_config["auxiliary_weight"]),
        )
        checkpoint = router_root / f"mlp_seed_{seed}.safetensors"
        result.router.save(
            checkpoint,
            overwrite=args.force,
            metadata={
                "seed": int(seed),
                "best_epoch": result.best_epoch,
                "best_validation_regression_loss": result.best_validation_regression_loss,
            },
        )
        atomic_write_json(router_root / f"mlp_seed_{seed}.history.json", {"history": result.history}, overwrite=args.force)
        validation_mae = float(np.mean(np.abs(result.router.predict_gain(validation_x) - validation_gain)))
        outcomes.append(
            {
                "seed": int(seed),
                "checkpoint": checkpoint.name,
                "best_epoch": result.best_epoch,
                "validation_regression_loss": result.best_validation_regression_loss,
                "validation_mae": validation_mae,
                "result": result,
            }
        )
        results.append({key: value for key, value in outcomes[-1].items() if key != "result"})
    selected = min(outcomes, key=lambda item: (item["validation_regression_loss"], item["seed"]))
    constant_mae = float(np.mean(np.abs(validation_gain - train_gain.mean())))
    if not selected["validation_mae"] < constant_mae:
        raise ArtifactError(
            f"Selected MLP validation MAE {selected['validation_mae']:.6f} does not beat constant predictor {constant_mae:.6f}"
        )
    selected_result = selected["result"]
    selected_result.router.save(
        router_root / "best_validation.safetensors",
        overwrite=args.force,
        metadata={
            "selected_seed": selected["seed"],
            "best_epoch": selected["best_epoch"],
            "best_validation_regression_loss": selected["validation_regression_loss"],
        },
    )
    selected_result.last_router.save(
        router_root / "last.safetensors",
        overwrite=args.force,
        metadata={"selected_seed": selected["seed"], "last_epoch": selected_result.history[-1]["epoch"]},
    )
    _torch_save_atomic(
        {"selected_seed": selected["seed"], "best_epoch": selected["best_epoch"], "history": selected_result.history},
        router_root / "training_state.pt",
        overwrite=args.force,
    )
    atomic_write_json(
        router_root / "router_selection.json",
        {
            "ridge_alpha": ridge.alpha,
            "ridge_validation_mae": ridge.validation_mae,
            "constant_validation_mae": constant_mae,
            "selected_seed": selected["seed"],
            "selected_checkpoint": "best_validation.safetensors",
            "all_mlp_seeds": results,
        },
        overwrite=args.force,
    )
    print(f"Selected MLP seed {selected['seed']} with validation regression loss {selected['validation_regression_loss']:.6f}")


if __name__ == "__main__":
    main()
