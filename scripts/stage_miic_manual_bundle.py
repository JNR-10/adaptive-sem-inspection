#!/usr/bin/env python3
"""Stage an officially downloaded MIIC bundle with full verification.

Some networks are challenged by the NTU Dataverse anti-bot page.  This script
does not attempt to bypass that protection: it accepts an archive obtained by a
person in a browser or a folder of declared files, copies only those files, and
verifies the published byte counts and MD5 values before making them available
to the normal pipeline. A folder may contain lossless numeric shards named
``<archive>.part-000``, ``<archive>.part-001``, and so on; they are streamed
together and verified before promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator
from zipfile import BadZipFile, ZipFile

from seminspect.utils.config import load_config
from seminspect.utils.io import ArtifactError, file_md5


ROOT = Path(__file__).resolve().parents[1]


def _verified_existing(path: Path, *, expected_bytes: int, expected_md5: str) -> bool:
    if not path.exists():
        return False
    if not path.is_file():
        raise ArtifactError(f"Expected a file at the MIIC destination: {path}")
    actual_bytes = path.stat().st_size
    actual_md5 = file_md5(path)
    if actual_bytes == expected_bytes and actual_md5.lower() == expected_md5.lower():
        print(f"{path.name}: reused verified file")
        return True
    raise ArtifactError(
        "Existing MIIC archive does not match the published metadata and will not be overwritten: "
        f"{path}"
    )


def _chunks_from_stream(source: BinaryIO) -> Iterator[bytes]:
    while chunk := source.read(1024 * 1024):
        yield chunk


def _chunks_from_paths(paths: Iterable[Path]) -> Iterator[bytes]:
    for path in paths:
        with path.open("rb") as source:
            yield from _chunks_from_stream(source)


def _copy_verified(chunks: Iterable[bytes], target: Path, *, expected_bytes: int, expected_md5: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.manual-staging")
    temporary.unlink(missing_ok=True)
    digest = hashlib.md5()
    count = 0
    try:
        with temporary.open("xb") as destination:
            for chunk in chunks:
                destination.write(chunk)
                digest.update(chunk)
                count += len(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        actual_md5 = digest.hexdigest()
        if count != expected_bytes or actual_md5.lower() != expected_md5.lower():
            raise ArtifactError(
                f"Manual MIIC source failed verification for {target.name}: expected "
                f"{expected_bytes} bytes/{expected_md5}, got {count} bytes/{actual_md5}"
            )
        os.replace(temporary, target)
        print(f"{target.name}: staged and verified ({count} bytes)")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _ordered_shards(source: Path, filename: str) -> list[Path]:
    """Return an unambiguous, gap-free sequence of lossless archive shards."""

    prefix = f"{filename}.part-"
    found = [
        path
        for path in source.rglob(f"{filename}.part-*")
        if path.is_file() and path.name.removeprefix(prefix).isdigit()
    ]
    if not found:
        raise ArtifactError(f"Expected exactly one {filename} or numeric shards under {source}")
    indexed: dict[int, Path] = {}
    for path in found:
        index = int(path.name.removeprefix(prefix))
        if index in indexed:
            raise ArtifactError(f"Found duplicate shard index {index} for {filename} under {source}")
        indexed[index] = path
    expected_indexes = list(range(len(indexed)))
    if sorted(indexed) != expected_indexes:
        raise ArtifactError(f"Shards for {filename} must be numbered consecutively from 000 under {source}")
    return [indexed[index] for index in expected_indexes]


def _stage_from_directory(source: Path, archive: dict, destination: Path) -> None:
    filename = str(archive["filename"])
    target = destination / filename
    if _verified_existing(target, expected_bytes=int(archive["expected_bytes"]), expected_md5=str(archive["md5"])):
        return
    matches = [path for path in source.rglob(filename) if path.is_file() and path.name == filename]
    if len(matches) == 1:
        with matches[0].open("rb") as input_stream:
            _copy_verified(
                _chunks_from_stream(input_stream),
                target,
                expected_bytes=int(archive["expected_bytes"]),
                expected_md5=str(archive["md5"]),
            )
        return
    if len(matches) > 1:
        raise ArtifactError(f"Expected exactly one {filename} under {source}, found {len(matches)}")
    _copy_verified(
        _chunks_from_paths(_ordered_shards(source, filename)),
        target,
        expected_bytes=int(archive["expected_bytes"]),
        expected_md5=str(archive["md5"]),
    )


def _stage_from_zip(source: Path, archive: dict, destination: Path, bundle: ZipFile) -> None:
    filename = str(archive["filename"])
    target = destination / filename
    if _verified_existing(target, expected_bytes=int(archive["expected_bytes"]), expected_md5=str(archive["md5"])):
        return
    matches = [item for item in bundle.infolist() if not item.is_dir() and Path(item.filename).name == filename]
    if len(matches) != 1:
        raise ArtifactError(f"Expected exactly one {filename} in {source}, found {len(matches)}")
    with bundle.open(matches[0], "r") as input_stream:
        _copy_verified(
            _chunks_from_stream(input_stream),
            target,
            expected_bytes=int(archive["expected_bytes"]),
            expected_md5=str(archive["md5"]),
        )


def stage_bundle(source: Path, destination: Path, archives: list[dict]) -> None:
    """Stage declared files from an official ZIP or a folder of files/shards."""

    if source.is_dir():
        for archive in archives:
            _stage_from_directory(source, archive, destination)
        return
    if not source.is_file():
        raise ArtifactError(f"Manual MIIC source does not exist: {source}")
    if source.suffix.lower() != ".zip":
        raise ArtifactError("Manual MIIC source must be the official ZIP file or a folder of verified files/shards")
    try:
        with ZipFile(source) as bundle:
            for archive in archives:
                _stage_from_zip(source, archive, destination, bundle)
    except (BadZipFile, OSError) as exc:
        raise ArtifactError(f"Could not read manual MIIC ZIP: {source}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Official MIIC ZIP or a folder of files (including .part-000 shards)",
    )
    parser.add_argument("--destination", required=True, type=Path, help="Pipeline data/raw directory")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "data" / "miic.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    archives = config["datasets"]["miic"]["archives"]
    stage_bundle(args.source.resolve(), args.destination.resolve(), archives)


if __name__ == "__main__":
    main()
