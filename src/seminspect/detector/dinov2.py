"""Pinned DINOv2 and ResNet-18 feature extraction interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from seminspect.utils.device import require_torch
from seminspect.utils.io import file_sha256


@dataclass
class EncoderOutput:
    patch_descriptors: Any
    cls_embeddings: Any
    patch_grid: tuple[int, int]


class DINOv2Encoder:
    """Frozen DINOv2 ViT-S/14-with-registers at a specific source commit."""

    def __init__(
        self,
        *,
        source_repository: str = "facebookresearch/dinov2",
        source_commit: str = "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        model_name: str = "dinov2_vits14_reg",
        device: str = "cuda",
        weight_path: str | Path | None = None,
        patch_pool: int = 2,
    ) -> None:
        self.source_repository = source_repository
        self.source_commit = source_commit
        self.model_name = model_name
        self.device = device
        self.weight_path = Path(weight_path) if weight_path else None
        self.patch_pool = int(patch_pool)
        if self.patch_pool <= 0 or 32 % self.patch_pool:
            raise ValueError("DINOv2 patch_pool must be a positive divisor of 32")
        self.model: Any | None = None

    @property
    def weight_sha256(self) -> str | None:
        return file_sha256(self.weight_path) if self.weight_path and self.weight_path.is_file() else None

    def load(self) -> "DINOv2Encoder":
        torch = require_torch()
        if self.weight_path is not None and not self.weight_path.is_file():
            raise FileNotFoundError(f"Configured DINO weight file does not exist: {self.weight_path}")
        repository = f"{self.source_repository}:{self.source_commit}"
        self.model = torch.hub.load(repository, self.model_name, source="github")
        if self.weight_path is None:
            checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
            candidates = sorted(
                path
                for path in checkpoints.glob("*.pth")
                if "dinov2" in path.name.lower() and "vits14" in path.name.lower() and "reg" in path.name.lower()
            )
            # The released hub model has one corresponding file. If a user has
            # several ambiguous cached variants, force an explicit provenance path.
            if len(candidates) == 1:
                self.weight_path = candidates[0]
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        return self

    def extract(self, batch: Any) -> EncoderOutput:
        """Return 16×16 pooled L2-normalized patch descriptors and CLS vectors."""

        if self.model is None:
            self.load()
        torch = require_torch()
        import torch.nn.functional as functional

        if tuple(batch.shape[1:]) != (3, 448, 448):
            raise ValueError(f"DINOv2 expects [B,3,448,448], received {tuple(batch.shape)}")
        batch = batch.to(self.device, dtype=torch.float32, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            features = self.model.forward_features(batch)
        patch = features["x_norm_patchtokens"].float()
        cls = features["x_norm_clstoken"].float()
        expected_patch = (batch.shape[0], 1024, 384)
        expected_cls = (batch.shape[0], 384)
        if tuple(patch.shape) != expected_patch or tuple(cls.shape) != expected_cls:
            raise RuntimeError(
                f"Unexpected DINOv2 feature shapes: patch={tuple(patch.shape)}, cls={tuple(cls.shape)}"
            )
        grid = patch.reshape(batch.shape[0], 32, 32, 384).permute(0, 3, 1, 2)
        pooled = functional.avg_pool2d(grid, kernel_size=self.patch_pool, stride=self.patch_pool)
        side = 32 // self.patch_pool
        descriptors = pooled.permute(0, 2, 3, 1).reshape(batch.shape[0], side * side, 384)
        descriptors = functional.normalize(descriptors, dim=-1)
        cls = functional.normalize(cls, dim=-1)
        return EncoderOutput(descriptors, cls, (16, 16))


class ResNet18Encoder:
    """Predeclared frozen fallback used only when the DINO feasibility gate fails."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.model: Any | None = None
        self.weight_path: Path | None = None

    @property
    def weight_sha256(self) -> str | None:
        return file_sha256(self.weight_path) if self.weight_path and self.weight_path.is_file() else None

    def load(self) -> "ResNet18Encoder":
        torch = require_torch()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = resnet18(weights=weights).to(self.device).eval()
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
        if checkpoint.is_file():
            self.weight_path = checkpoint
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        return self

    def extract(self, batch: Any) -> EncoderOutput:
        if self.model is None:
            self.load()
        torch = require_torch()
        import torch.nn.functional as functional

        if tuple(batch.shape[1:]) != (3, 448, 448):
            raise ValueError("ResNet-18 fallback expects [B,3,448,448]")
        x = batch.to(self.device, dtype=torch.float32, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            layer2 = self.model.layer2(x)
            layer3 = self.model.layer3(layer2)
        layer2 = functional.interpolate(layer2.float(), size=(28, 28), mode="bilinear", align_corners=False)
        fused = torch.cat([layer2, layer3.float()], dim=1)  # 128 + 256 = 384
        cls = functional.normalize(fused.mean(dim=(2, 3)), dim=-1)
        patches = functional.avg_pool2d(fused, kernel_size=2, stride=2)
        descriptors = functional.normalize(patches.permute(0, 2, 3, 1).reshape(x.shape[0], 196, 384), dim=-1)
        return EncoderOutput(descriptors, cls, (14, 14))


def features_to_numpy(output: EncoderOutput) -> tuple[np.ndarray, np.ndarray]:
    """Move a no-grad encoder result to contiguous float32 CPU arrays."""

    return (
        output.patch_descriptors.detach().cpu().contiguous().numpy().astype(np.float32, copy=False),
        output.cls_embeddings.detach().cpu().contiguous().numpy().astype(np.float32, copy=False),
    )


def build_encoder(detector: Mapping[str, Any], *, device: str, weight_path: str | Path | None = None) -> DINOv2Encoder | ResNet18Encoder:
    """Construct the explicitly configured frozen encoder.

    `core_v1` uses DINOv2.  The only alternate value is the prespecified
    `resnet18` feasibility fallback, which must live under a separately named
    protocol configuration and therefore never silently changes a sealed run.
    """

    encoder_type = str(detector.get("encoder_type", "dinov2"))
    if encoder_type == "dinov2":
        return DINOv2Encoder(
            source_repository=str(detector["source_repository"]),
            source_commit=str(detector["source_commit"]),
            model_name=str(detector["backbone"]),
            device=device,
            weight_path=weight_path,
            patch_pool=int(detector.get("patch_pool", 2)),
        )
    if encoder_type == "resnet18":
        if weight_path is not None:
            raise ValueError("DINO_WEIGHT_PATH is not applicable to the ResNet-18 fallback")
        if int(detector.get("patch_pool", 2)) != 2:
            raise ValueError("The ResNet-18 fallback has a fixed 2×2 descriptor pooling contract")
        return ResNet18Encoder(device=device)
    raise ValueError(f"Unsupported detector.encoder_type: {encoder_type!r}")
