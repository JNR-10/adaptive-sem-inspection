"""Deterministic CPU multi-task MLP Value-of-Resolution router."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from seminspect.utils.device import require_torch
from seminspect.utils.io import ArtifactError, atomic_write_json
from seminspect.utils.reproducibility import seed_everything


def _module(input_dim: int = 53, hidden_dims: tuple[int, int] = (64, 32), dropout: float = 0.2) -> Any:
    torch = require_torch()
    from torch import nn

    class ValueMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.GELU(),
            )
            self.gain_head = nn.Linear(hidden_dims[1], 1)
            self.helpful_head = nn.Linear(hidden_dims[1], 1)

        def forward(self, values: Any) -> tuple[Any, Any]:
            representation = self.backbone(values)
            return self.gain_head(representation).squeeze(-1), self.helpful_head(representation).squeeze(-1)

    return ValueMLP()


@dataclass
class MLPTrainingResult:
    router: "MLPRouter"
    last_router: "MLPRouter"
    seed: int
    best_epoch: int
    best_validation_regression_loss: float
    history: list[dict[str, float]]


class MLPRouter:
    """Inference wrapper around a CPU MLP checkpoint (dropout disabled)."""

    def __init__(
        self,
        *,
        input_dim: int = 53,
        hidden_dims: tuple[int, int] = (64, 32),
        dropout: float = 0.2,
        model: Any | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(item) for item in hidden_dims)
        self.dropout = float(dropout)
        self.model = model if model is not None else _module(self.input_dim, self.hidden_dims, self.dropout)
        self.model.cpu().eval()

    def predict_gain(self, features: np.ndarray) -> np.ndarray:
        torch = require_torch()
        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[1] != self.input_dim:
            raise ValueError("MLP input does not match the sealed router feature width")
        self.model.eval()
        with torch.inference_mode():
            gain, _ = self.model(torch.from_numpy(values))
        return gain.detach().cpu().numpy().astype(np.float64)

    def predict_helpful_logit(self, features: np.ndarray) -> np.ndarray:
        torch = require_torch()
        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        self.model.eval()
        with torch.inference_mode():
            _, helpful = self.model(torch.from_numpy(values))
        return helpful.detach().cpu().numpy().astype(np.float64)

    def save(self, path: str | Path, *, overwrite: bool = False, metadata: dict[str, Any] | None = None) -> Path:
        """Save only tensors in Safetensors and metadata in a neighboring JSON file."""

        destination = Path(path)
        if destination.exists() and not overwrite:
            raise ArtifactError(f"Refusing to overwrite MLP router checkpoint: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ArtifactError("safetensors is required to write MLP router checkpoints") from exc
        state = {name: value.detach().cpu().contiguous() for name, value in self.model.state_dict().items()}
        temporary = destination.with_name(f"{destination.name}.tmp")
        temporary.unlink(missing_ok=True)
        save_file(state, str(temporary))
        temporary.replace(destination)
        meta = {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            **(metadata or {}),
        }
        atomic_write_json(destination.with_suffix(destination.suffix + ".json"), meta, overwrite=overwrite)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "MLPRouter":
        checkpoint = Path(path)
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ArtifactError("safetensors is required to read MLP router checkpoints") from exc
        import json

        metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        router = cls(
            input_dim=int(metadata["input_dim"]),
            hidden_dims=tuple(int(item) for item in metadata["hidden_dims"]),
            dropout=float(metadata["dropout"]),
        )
        router.model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
        router.model.eval()
        return router


def train_mlp_router(
    train_features: np.ndarray,
    train_gain: np.ndarray,
    train_helpful: np.ndarray,
    validation_features: np.ndarray,
    validation_gain: np.ndarray,
    validation_helpful: np.ndarray,
    *,
    seed: int,
    input_dim: int = 53,
    hidden_dims: tuple[int, int] = (64, 32),
    dropout: float = 0.2,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 20,
    gradient_clip_norm: float = 1.0,
    huber_beta: float = 0.1,
    auxiliary_weight: float = 0.25,
) -> MLPTrainingResult:
    """Train one fully deterministic CPU router seed and retain its best state."""

    torch = require_torch()
    from torch import nn

    seed_everything(seed, deterministic=True)
    arrays = [
        np.asarray(train_features, dtype=np.float32),
        np.asarray(train_gain, dtype=np.float32).reshape(-1),
        np.asarray(train_helpful, dtype=np.float32).reshape(-1),
        np.asarray(validation_features, dtype=np.float32),
        np.asarray(validation_gain, dtype=np.float32).reshape(-1),
        np.asarray(validation_helpful, dtype=np.float32).reshape(-1),
    ]
    train_x, train_y, train_h, validation_x, validation_y, validation_h = arrays
    if train_x.shape[1] != input_dim or validation_x.shape[1] != input_dim:
        raise ValueError("MLP train/validation feature width must equal input_dim")
    if not (len(train_x) == len(train_y) == len(train_h)) or not (len(validation_x) == len(validation_y) == len(validation_h)):
        raise ValueError("MLP target lengths do not match feature rows")
    if not np.isfinite(np.concatenate([train_x.ravel(), train_y, train_h, validation_x.ravel(), validation_y, validation_h])).all():
        raise ValueError("MLP inputs and targets must be finite")

    model = _module(input_dim, hidden_dims, dropout).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    regression_loss = nn.SmoothL1Loss(beta=huber_beta, reduction="mean")
    classification_loss = nn.BCEWithLogitsLoss(reduction="mean")
    tensors = [torch.from_numpy(item) for item in arrays]
    train_x_t, train_y_t, train_h_t, validation_x_t, validation_y_t, validation_h_t = tensors
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    waiting = 0
    history: list[dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(train_x_t), generator=generator)
        running = 0.0
        examples = 0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            gain_hat, helpful_logit = model(train_x_t[index])
            loss_reg = regression_loss(gain_hat, train_y_t[index])
            loss_cls = classification_loss(helpful_logit, train_h_t[index])
            loss = loss_reg + auxiliary_weight * loss_cls
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            batch_count = len(index)
            running += float(loss.detach()) * batch_count
            examples += batch_count
        model.eval()
        with torch.inference_mode():
            validation_gain_hat, validation_helpful_logit = model(validation_x_t)
            val_reg = float(regression_loss(validation_gain_hat, validation_y_t))
            val_cls = float(classification_loss(validation_helpful_logit, validation_h_t))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running / examples,
                "validation_regression_loss": val_reg,
                "validation_classification_loss": val_cls,
            }
        )
        if best_loss - val_reg > 1e-8:
            best_loss = val_reg
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            waiting = 0
        else:
            waiting += 1
            if waiting >= patience:
                break
    if best_state is None:
        raise RuntimeError("MLP did not produce a checkpoint")
    last_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    last_model = _module(input_dim, hidden_dims, dropout).cpu()
    last_model.load_state_dict(last_state, strict=True)
    last_model.eval()
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return MLPTrainingResult(
        router=MLPRouter(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, model=model),
        last_router=MLPRouter(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, model=last_model),
        seed=seed,
        best_epoch=best_epoch,
        best_validation_regression_loss=best_loss,
        history=history,
    )
