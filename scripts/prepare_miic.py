#!/usr/bin/env python3
"""Transactionally extract MIIC and emit its unpartitioned canonical manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import add_config_argument, load_core_config, write_failure
from seminspect.data.manifests import read_manifest, write_manifest
from seminspect.data.miic import build_miic_manifest, extract_archives_transactionally, validate_miic_extraction
from seminspect.utils.io import ArtifactError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--force", action="store_true", help="Replace development manifest only; never use for sealed output")
    args = parser.parse_args()
    config, paths = load_core_config(args.config)
    try:
        dataset = config["datasets"]["miic"]
        archives = [paths["raw_dir"] / item["filename"] for item in dataset["archives"] if item["filename"].endswith(".rar")]
        for path in archives:
            if not path.is_file():
                raise ArtifactError(f"MIIC archive has not been downloaded: {path}")
        extracted = paths["extracted_dir"] / "miic"
        extract_archives_transactionally(
            archives,
            extracted,
            validator=lambda root: validate_miic_extraction(root, dataset["expected"]),
        )
        manifest_path = paths["manifest_dir"] / "miic_raw.parquet"
        if manifest_path.exists() and not args.force:
            frame = read_manifest(manifest_path)
            print(f"Reusing complete MIIC raw manifest with {len(frame)} rows: {manifest_path}")
            return
        frame = build_miic_manifest(extracted, dataset["expected"])
        write_manifest(frame, manifest_path, overwrite=args.force)
        print(f"Wrote {len(frame)} MIIC manifest rows to {manifest_path}")
    except Exception as exc:
        write_failure(paths["artifacts_dir"], "prepare_miic", exc)
        raise


if __name__ == "__main__":
    main()
