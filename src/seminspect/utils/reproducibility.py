"""Determinism controls and run-environment capture."""

from __future__ import annotations

import os
import platform
import random
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from seminspect.utils.io import atomic_write_json, utc_now


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Set every available random source and return the applied settings.

    `PYTHONHASHSEED` only controls a newly started interpreter; the bootstrap
    notebook exports it before launching scripts. We still record and set it
    here to surface a misconfigured interactive invocation.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    settings: dict[str, Any] = {
        "seed": seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "pythonhashseed_matches": os.environ.get("PYTHONHASHSEED") == str(seed),
        "torch_available": False,
        "deterministic": deterministic,
    }
    try:
        import torch

        settings["torch_available"] = True
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        settings.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            }
        )
    except ImportError:
        pass
    return settings


def environment_info() -> dict[str, Any]:
    """Collect immutable-ish package/runtime facts before a computation starts."""

    package_names = [
        "numpy",
        "pandas",
        "Pillow",
        "scipy",
        "scikit-learn",
        "scikit-image",
        "torch",
        "torchvision",
        "timm",
        "opencv-python-headless",
        "pyarrow",
    ]
    packages: dict[str, str | None] = {}
    for name in package_names:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    result: dict[str, Any] = {
        "captured_utc": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }
    try:
        import torch

        result["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except ImportError:
        result["torch"] = None
    return result


def write_environment(path: str | Path, seed: int, deterministic: bool = True) -> dict[str, Any]:
    result = environment_info()
    result["determinism"] = seed_everything(seed, deterministic=deterministic)
    atomic_write_json(path, result)
    return result
