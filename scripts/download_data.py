#!/usr/bin/env python3
"""Stream declared archives transactionally and write download provenance."""

from __future__ import annotations

import argparse
from pathlib import Path

from seminspect.utils.config import load_config
from seminspect.utils.io import ArtifactError, atomic_write_json, read_json, transactional_download, utc_now


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["miic", "carinthia_s"])
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    config_path = ROOT / "configs" / "data" / ("miic.yaml" if args.dataset == "miic" else "carinthia_s.yaml")
    dataset = load_config(config_path)["datasets"][args.dataset]
    destination = args.destination
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "download_manifest.json"
    existing = read_json(manifest_path) if manifest_path.exists() else {"downloads": []}
    records = {item["filename"]: item for item in existing.get("downloads", [])}
    for archive in dataset["archives"]:
        url = archive.get("url") or dataset["url_template"].format(file_id=archive["file_id"])
        try:
            result = transactional_download(
                url=url,
                destination=destination / archive["filename"],
                expected_bytes=int(archive["expected_bytes"]),
                expected_md5=str(archive["md5"]),
            )
        except ArtifactError as exc:
            if args.dataset == "miic":
                raise ArtifactError(
                    f"{exc}\nThe NTU Dataverse may have returned an invalid response or dropped the transfer. "
                    "Do not bypass any verification: download the five official files in a browser or use "
                    "a verified folder, then rerun the pipeline with --miic-manual-bundle <folder>."
                ) from exc
            raise
        records[archive["filename"]] = {"filename": archive["filename"], **result}
        print(f"{archive['filename']}: {result['status']} ({result['bytes']} bytes)")
    atomic_write_json(
        manifest_path,
        {
            "dataset": args.dataset,
            "license_notice": dataset["license"],
            "generated_utc": utc_now(),
            "downloads": [records[name] for name in sorted(records)],
        },
        overwrite=True,
    )


if __name__ == "__main__":
    main()
