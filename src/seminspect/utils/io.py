"""Durable file, hashing, and download primitives.

No caller is allowed to overwrite a completed incompatible artifact.  The
helpers below make the intended state transition explicit: write a sibling
temporary file, fsync it, validate it, then atomically rename it into place.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Iterable


class ArtifactError(RuntimeError):
    """Raised when an artifact cannot be safely created or reused."""


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    """Return sealed-protocol JSON bytes: UTF-8, sorted keys, compact syntax."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_md5(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # nosec B324: dataset publisher specifies MD5 integrity hashes.
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_tree_sha256(paths: Iterable[str | Path], root: str | Path | None = None) -> str:
    """Hash paths and bytes in deterministic relative-path order.

    This is intentionally not a directory timestamp hash. It is the artifact
    primitive used by `test_lock.json`.
    """

    root_path = Path(root).resolve() if root is not None else None
    items = sorted(Path(path).resolve() for path in paths)
    digest = hashlib.sha256()
    for path in items:
        if not path.is_file():
            raise ArtifactError(f"Cannot hash missing/non-file artifact: {path}")
        relative = str(path.relative_to(root_path) if root_path else path.name).replace(os.sep, "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory fsync; unsupported filesystems remain usable."""

    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: str | Path, data: bytes, *, overwrite: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise ArtifactError(f"Refusing to overwrite completed artifact: {target}")
    temporary = target.with_name(f"{target.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(path: str | Path, text: str, *, overwrite: bool = False) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"), overwrite=overwrite)


def atomic_write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> Path:
    return atomic_write_bytes(path, canonical_json_bytes(value) + b"\n", overwrite=overwrite)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _stream_copy_and_hash(source: BinaryIO, destination: BinaryIO, chunk_size: int) -> tuple[int, str]:
    digest = hashlib.md5()  # nosec B324: required only to compare publisher metadata.
    count = 0
    while True:
        chunk = source.read(chunk_size)
        if not chunk:
            break
        destination.write(chunk)
        digest.update(chunk)
        count += len(chunk)
    return count, digest.hexdigest()


def transactional_download(
    *,
    url: str,
    destination: str | Path,
    expected_bytes: int,
    expected_md5: str,
    timeout_seconds: int = 60,
    chunk_size: int = 1024 * 1024,
) -> dict[str, Any]:
    """Download one declared archive without replacing a mismatch.

    A matching final file is reused. A mismatching final file is a hard failure;
    callers must inspect it rather than receiving an implicit overwrite.
    """

    import requests

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        current_bytes = target.stat().st_size
        current_md5 = file_md5(target)
        if current_bytes == expected_bytes and current_md5.lower() == expected_md5.lower():
            return {
                "url": url,
                "retrieved_utc": None,
                "status": "reused_verified",
                "bytes": current_bytes,
                "md5": current_md5,
                "etag": None,
                "last_modified": None,
            }
        raise ArtifactError(
            f"Existing archive does not match declared metadata and will not be overwritten: {target}"
        )

    partial = target.with_name(f"{target.name}.partial")
    partial.unlink(missing_ok=True)  # Explicitly created transfer residue only.
    response = requests.get(url, stream=True, timeout=timeout_seconds)
    try:
        response.raise_for_status()
        if not response.raw.readable():
            raise ArtifactError(f"HTTP response body is not readable for {url}")
        with partial.open("xb") as handle:
            count, md5 = _stream_copy_and_hash(response.raw, handle, chunk_size)
            handle.flush()
            os.fsync(handle.fileno())
        if count != expected_bytes or md5.lower() != expected_md5.lower():
            raise ArtifactError(
                f"Downloaded archive failed verification: expected {expected_bytes} bytes/{expected_md5}, "
                f"got {count} bytes/{md5}"
            )
        os.replace(partial, target)
        _fsync_directory(target.parent)
        return {
            "url": url,
            "retrieved_utc": utc_now(),
            "status": "downloaded_verified",
            "bytes": count,
            "md5": md5,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        response.close()


def atomic_copy(source: str | Path, destination: str | Path, *, overwrite: bool = False) -> Path:
    """Copy a completed file through a fsynced sibling temp path."""

    source_path = Path(source)
    target = Path(destination)
    if not source_path.is_file():
        raise ArtifactError(f"Copy source is not a file: {source_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise ArtifactError(f"Refusing to overwrite completed artifact: {target}")
    temporary = target.with_name(f"{target.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with source_path.open("rb") as src, temporary.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def temporary_directory(parent: str | Path, prefix: str) -> Path:
    """Create an explicitly owned temporary extraction or staging directory."""

    parent_path = Path(parent)
    parent_path.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(parent_path)))
