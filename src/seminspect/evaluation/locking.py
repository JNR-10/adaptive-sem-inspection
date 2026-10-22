"""Test-lock hashing and validation for the one controlled final benchmark unlock."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from seminspect.utils.io import ArtifactError, canonical_json_sha256, file_sha256, read_json


class TestLockError(ArtifactError):
    """Raised when frozen inputs differ from the current execution state."""


def verify_test_lock(lock_path: str | Path, current: dict[str, str]) -> dict[str, Any]:
    lock = read_json(lock_path)
    for key, value in current.items():
        if lock.get(key) != value:
            raise TestLockError(f"Test lock mismatch for {key}: locked={lock.get(key)!r}, current={value!r}")
    return lock


def ids_hash(ids: list[str]) -> str:
    return canonical_json_sha256(sorted(str(item) for item in ids))


def hash_file_or_error(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        raise TestLockError(f"Required locked artifact does not exist: {candidate}")
    return file_sha256(candidate)
