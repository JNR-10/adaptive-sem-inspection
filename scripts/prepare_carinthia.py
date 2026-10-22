#!/usr/bin/env python3
"""Transactionally extract Carinthia-S and audit its masks before modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import add_config_argument, load_core_config, write_failure
from seminspect.data.carinthia import build_carinthia_manifest, validate_carinthia_extraction
from seminspect.data.manifests import read_manifest, write_manifest
from seminspect.data.miic import extract_archives_transactionally
from seminspect.utils.io import ArtifactError
from seminspect.utils.tabular import write_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--force", action="store_true", help="Replace development outputs only")
    args = parser.parse_args()
    config, paths = load_core_config(args.config)
    try:
        dataset = config["datasets"]["carinthia_s"]
        archive = paths["raw_dir"] / dataset["archives"][0]["filename"]
        if not archive.is_file():
            raise ArtifactError(f"Carinthia-S archive has not been downloaded: {archive}")
        extracted = paths["extracted_dir"] / "carinthia_s"
        extract_archives_transactionally(
            [archive], extracted, validator=lambda root: validate_carinthia_extraction(root, dataset["expected"])
        )
        manifest_path = paths["manifest_dir"] / "carinthia_s.parquet"
        audit_path = paths["artifacts_dir"] / "audits" / "carinthia_mask_audit.parquet"
        if manifest_path.exists() and audit_path.exists() and not args.force:
            frame = read_manifest(manifest_path)
            print(f"Reusing complete Carinthia-S manifest with {len(frame)} rows: {manifest_path}")
            return
        manifest, audit = build_carinthia_manifest(extracted, dataset["expected"])
        write_manifest(manifest, manifest_path, overwrite=args.force)
        write_parquet(audit, audit_path, overwrite=args.force)
        print(f"Wrote {len(manifest)} Carinthia-S manifest rows and mask audit")
    except Exception as exc:
        write_failure(paths["artifacts_dir"], "prepare_carinthia", exc)
        raise


if __name__ == "__main__":
    main()
