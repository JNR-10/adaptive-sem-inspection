"""Optional-PyTorch device boundary used by GPU-only components."""

from __future__ import annotations

from typing import Any


class DependencyError(RuntimeError):
    """Raised when a runtime-only dependency has not been installed."""


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise DependencyError(
            "PyTorch is required for this stage. Install requirements/gpu-cu126.lock in a CUDA-capable environment."
        ) from exc
    return torch


def resolve_device(requested: str = "cuda") -> str:
    torch = require_torch()
    if requested == "cuda" and not torch.cuda.is_available():
        raise DependencyError("core_v1 reference detector requires a CUDA-capable runtime")
    if requested not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported device {requested!r}; use 'cuda' or 'cpu'")
    return requested
