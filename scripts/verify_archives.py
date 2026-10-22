#!/usr/bin/env python3
"""Verify declared archive bytes/checksums and inspect extractable members safely."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import add_config_argument, runtime_root
from seminspect.data.miic import inspect_archive
from seminspect.utils.config import load_config
from seminspect.utils.io import ArtifactError, file_md5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--raw-dir", type=Path, default=runtime_root() / "data/raw")
    args = parser.parse_args()
    config = load_config(args.config)
    datasets = config["datasets"]
    if len(datasets) != 1:
        raise ArtifactError("Archive config must name exactly one dataset")
    name, dataset = next(iter(datasets.items()))
    for archive in dataset["archives"]:
        path = args.raw_dir / archive["filename"]
        if not path.is_file():
            raise ArtifactError(f"Missing declared archive: {path}")
        if path.stat().st_size != int(archive["expected_bytes"]):
            raise ArtifactError(f"Archive size mismatch: {path}")
        if file_md5(path).lower() != str(archive["md5"]).lower():
            raise ArtifactError(f"Archive MD5 mismatch: {path}")
        if path.suffix.lower() in {".rar", ".zip"}:
            members = inspect_archive(path)
            print(f"{name}/{path.name}: verified; {len(members)} safe members")
        else:
            print(f"{name}/{path.name}: verified")


if __name__ == "__main__":
    main()
