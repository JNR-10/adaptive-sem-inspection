"""MIIC archive safety, transactional extraction, and manifest construction."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

import pandas as pd

from seminspect.data.hashing import image_hash_metadata
from seminspect.data.manifests import normalize_manifest
from seminspect.utils.io import ArtifactError, temporary_directory


class ArchiveSecurityError(ArtifactError):
    """Raised when an archive contains an unsafe member or cannot be inspected."""


def _validate_member_name(name: str) -> None:
    normalized = name.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if not normalized or normalized.startswith("/") or candidate.is_absolute():
        raise ArchiveSecurityError(f"Archive contains an absolute path: {name!r}")
    if ".." in candidate.parts or any(":" in part for part in candidate.parts[:1]):
        raise ArchiveSecurityError(f"Archive contains a traversal or drive path: {name!r}")


def _validate_zip(path: Path) -> list[str]:
    members: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            _validate_member_name(entry.filename)
            mode = (entry.external_attr >> 16) & 0o177777
            kind = stat.S_IFMT(mode)
            if kind in {stat.S_IFLNK, stat.S_IFCHR, stat.S_IFBLK, stat.S_IFIFO, stat.S_IFSOCK}:
                raise ArchiveSecurityError(f"Archive contains a link/device entry: {entry.filename!r}")
            members.append(entry.filename)
    return members


def _seven_zip() -> str:
    binary = shutil.which("7z")
    if binary is None:
        raise ArchiveSecurityError("7z is required for RAR archive inspection/extraction; install p7zip-full")
    return binary


def _validate_rar(path: Path) -> list[str]:
    """List RAR entries using 7z before extraction and reject unsafe metadata."""

    result = subprocess.run(
        [_seven_zip(), "l", "-slt", str(path)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ArchiveSecurityError(f"7z could not inspect {path}: {result.stderr.strip()}")
    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}
    in_entries = False
    for raw_line in result.stdout.splitlines():
        if raw_line.strip() == "----------":
            in_entries = True
            continue
        if not in_entries:
            continue
        if not raw_line.strip():
            if current:
                entries.append(current)
                current = {}
            continue
        if " = " in raw_line:
            key, value = raw_line.split(" = ", 1)
            current[key.strip()] = value.strip()
    if current:
        entries.append(current)
    names: list[str] = []
    for entry in entries:
        name = entry.get("Path")
        if not name:
            continue
        _validate_member_name(name)
        attributes = entry.get("Attributes", "").lower()
        link = entry.get("Symbolic Link", "") or entry.get("Hard Link", "")
        if link or "l" in attributes.split():
            raise ArchiveSecurityError(f"Archive contains a link entry: {name!r}")
        names.append(name)
    if not names:
        raise ArchiveSecurityError(f"Archive has no extractable members: {path}")
    return names


def inspect_archive(path: str | Path) -> list[str]:
    """Return vetted member paths without extracting the archive."""

    archive = Path(path)
    if not archive.is_file():
        raise ArchiveSecurityError(f"Archive does not exist: {archive}")
    suffix = archive.suffix.lower()
    if suffix == ".zip":
        return _validate_zip(archive)
    if suffix == ".rar":
        return _validate_rar(archive)
    raise ArchiveSecurityError(f"Unsupported archive format: {archive}")


def extract_archives_transactionally(
    archives: Iterable[str | Path],
    destination: str | Path,
    *,
    validator: Callable[[Path], None] | None = None,
) -> Path:
    """Extract a declared archive set into a new directory then atomically publish it.

    Existing destination trees are never merged with or overwritten. A complete
    existing tree may be reused only if the caller's validator accepts it.
    """

    archive_paths = [Path(item) for item in archives]
    if not archive_paths:
        raise ArchiveSecurityError("At least one archive is required for transactional extraction")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if validator is None:
            raise ArtifactError(f"Extraction target already exists: {target}")
        validator(target)
        return target

    for archive in archive_paths:
        inspect_archive(archive)
    temporary = temporary_directory(target.parent, f".{target.name}.extract-")
    try:
        for archive in archive_paths:
            if archive.suffix.lower() == ".zip":
                with zipfile.ZipFile(archive) as zip_archive:
                    zip_archive.extractall(temporary)
            else:
                result = subprocess.run(
                    [_seven_zip(), "x", "-y", f"-o{temporary}", str(archive)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    raise ArchiveSecurityError(
                        f"7z extraction failed for {archive}: {(result.stderr or result.stdout).strip()}"
                    )
        if validator is not None:
            validator(temporary)
        os.replace(temporary, target)
        return target
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace(os.sep, "/")


def _record(
    *,
    root: Path,
    path: Path,
    image_id: str,
    source_archive: str,
    label: int | None,
    pair_id: str | None = None,
    mask_path: str | None = None,
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "dataset": "miic",
        "source_archive": source_archive,
        "relative_path": _relative(path, root),
        "label": label,
        "pair_id": pair_id,
        "mask_path": mask_path,
        "class_id": None,
        **image_hash_metadata(path),
        "split": "unassigned",
        "duplicate_group": None,
        "quarantined": False,
        "quarantine_reason": None,
    }


def _expected_test_mask(image_path: Path, masks_root: Path) -> Path:
    candidate = masks_root / f"{image_path.stem}_mask.jpg"
    if candidate.is_file():
        return candidate
    alternatives = sorted(masks_root.glob(f"{image_path.stem}_mask.*"))
    if len(alternatives) == 1:
        return alternatives[0]
    raise ArtifactError(f"Expected exactly one mask for {image_path.name}; found {len(alternatives)}")


def _pair_rows(root: Path, folder_name: str, expected_pairs: int) -> list[dict[str, Any]]:
    folder = root / folder_name
    files = sorted(path for path in folder.glob("*.jpg") if path.is_file())
    by_pair: dict[str, dict[str, Path]] = {}
    for path in files:
        stem = path.stem
        if stem.endswith("_anomaly"):
            pair, role = stem[: -len("_anomaly")], "anomaly"
        elif stem.endswith("_original"):
            pair, role = stem[: -len("_original")], "original"
        else:
            raise ArtifactError(f"Unrecognized MIIC inpainting filename: {path.name}")
        by_pair.setdefault(pair, {})[role] = path
    if len(by_pair) != expected_pairs:
        raise ArtifactError(f"{folder_name} has {len(by_pair)} pairs; expected {expected_pairs}")
    rows: list[dict[str, Any]] = []
    for pair, members in sorted(by_pair.items()):
        if set(members) != {"original", "anomaly"}:
            raise ArtifactError(f"Incomplete or duplicated pair {folder_name}/{pair}")
        pair_id = f"miic:{folder_name}:{pair}"
        for role, label in (("original", 0), ("anomaly", 1)):
            path = members[role]
            rows.append(
                _record(
                    root=root,
                    path=path,
                    image_id=f"{pair_id}:{role}",
                    source_archive=f"{folder_name}.rar",
                    label=label,
                    pair_id=pair_id,
                )
            )
    return rows


def validate_miic_extraction(root: str | Path, expected: dict[str, Any]) -> None:
    """Validate structural counts before a temporary extraction becomes visible."""

    source = Path(root)
    test = source / "Anomaly_test"
    checks = {
        "test_normal": len(list((test / "normal_img").glob("*.jpg"))),
        "test_anomaly": len(list((test / "abnormal_img").glob("*.jpg"))),
        "test_masks": len(list((test / "abnormal_mask").glob("*.jpg"))),
        "inpainting_train_pairs": len(list((source / "Inpainting_train").glob("*_original.jpg"))),
        "inpainting_test_pairs": len(list((source / "Inpainting_test").glob("*_original.jpg"))),
    }
    for key, observed in checks.items():
        if observed != int(expected[key]):
            raise ArtifactError(f"MIIC extraction count mismatch for {key}: {observed} != {expected[key]}")


def build_miic_manifest(root: str | Path, expected: dict[str, Any]) -> pd.DataFrame:
    """Build the unpartitioned MIIC manifest from a validated extraction tree."""

    source = Path(root)
    validate_miic_extraction(source, expected)
    records: list[dict[str, Any]] = []
    test_root = source / "Anomaly_test"
    for path in sorted((test_root / "normal_img").glob("*.jpg")):
        records.append(
            _record(
                root=source,
                path=path,
                image_id=f"miic:anomaly_test:normal:{path.stem}",
                source_archive="Anomaly_test.rar",
                label=0,
            )
        )
    masks_root = test_root / "abnormal_mask"
    for path in sorted((test_root / "abnormal_img").glob("*.jpg")):
        mask = _expected_test_mask(path, masks_root)
        records.append(
            _record(
                root=source,
                path=path,
                image_id=f"miic:anomaly_test:anomaly:{path.stem}",
                source_archive="Anomaly_test.rar",
                label=1,
                mask_path=_relative(mask, source),
            )
        )
    train_root = source / "Anomaly_train"
    train_images = sorted(path for path in train_root.glob("*.jpg") if path.is_file())
    if len(train_images) != int(expected["train_total"]):
        raise ArtifactError(
            f"Anomaly_train contains {len(train_images)} images; expected {expected['train_total']}"
        )
    for path in train_images:
        records.append(
            _record(
                root=source,
                path=path,
                image_id=f"miic:anomaly_train:{path.stem}",
                source_archive="Anomaly_train.rar",
                label=None,
            )
        )
    records.extend(_pair_rows(source, "Inpainting_train", int(expected["inpainting_train_pairs"])))
    records.extend(_pair_rows(source, "Inpainting_test", int(expected["inpainting_test_pairs"])))
    frame = normalize_manifest(pd.DataFrame.from_records(records))
    widths = frame["width"].dropna().astype(int)
    heights = frame["height"].dropna().astype(int)
    if not ((widths == int(expected["native_side"])).all() and (heights == int(expected["native_side"])).all()):
        raise ArtifactError("MIIC decoded image dimensions do not match the declared 512×512 contract")
    return frame
