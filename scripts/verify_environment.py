#!/usr/bin/env python3
"""Print a concise native-GPU environment check before a long experiment."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys


def _command_output(argv: list[str]) -> str | None:
    """Return a short command result without turning a diagnostic into a crash."""

    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true", help="Exit non-zero when CUDA is unavailable")
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        print("WARNING: core_v1 is locked for Python 3.11 or 3.12.")

    nvidia_smi = _command_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    print(f"nvidia-smi: {nvidia_smi or 'not available'}")

    seven_zip = shutil.which("7z")
    print(f"7z: {seven_zip or 'not found (install p7zip-full before MIIC extraction)'}")

    try:
        import torch
    except ImportError:
        print("PyTorch: not installed")
        if args.require_gpu:
            raise SystemExit(2)
        return

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda or 'CPU-only'}")
    available = bool(torch.cuda.is_available())
    print(f"CUDA available: {available}")
    if available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory: {total_gib:.1f} GiB")
    if args.require_gpu and (not available or seven_zip is None):
        missing = []
        if not available:
            missing.append("CUDA-enabled PyTorch/GPU")
        if seven_zip is None:
            missing.append("7z")
        raise SystemExit("Missing required native dependencies: " + ", ".join(missing))


if __name__ == "__main__":
    main()
