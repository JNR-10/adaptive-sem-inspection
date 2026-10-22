"""Canonical image decoding and content hashes used for leakage prevention."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile, ImageOps

from seminspect.utils.io import file_sha256


def decode_grayscale(path: str | Path, *, allow_truncated: bool = False) -> tuple[np.ndarray, str]:
    """Decode one image with the protocol's exact EXIF and grayscale rules."""

    original_policy = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = allow_truncated
    try:
        with Image.open(path) as source:
            mode = source.mode
            image = ImageOps.exif_transpose(source).convert("L")
            array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = original_policy
    if array.ndim != 2:
        raise ValueError(f"Canonical grayscale image must be two-dimensional: {path}")
    return array, mode


def decoded_pixel_sha256(array: np.ndarray) -> str:
    """Hash shape plus contiguous pixels, avoiding ambiguous byte-only hashes."""

    if array.ndim != 2 or array.dtype != np.uint8:
        raise ValueError("decoded_pixel_sha256 requires a two-dimensional uint8 image")
    contiguous = np.ascontiguousarray(array)
    height, width = contiguous.shape
    payload = struct.pack("<II", height, width) + contiguous.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def phash64(array: np.ndarray) -> int:
    """Return ImageHash-compatible 64-bit pHash (`hash_size=8`, `highfreq_factor=4`).

    ImageHash is a declared runtime dependency. The local implementation is kept
    as an equivalent fallback for lightweight static tests, but a sealed run
    records the installed package version in `environment.json`.
    """

    if array.ndim != 2:
        raise ValueError("phash64 requires a grayscale two-dimensional array")
    try:
        import imagehash

        image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L")
        digest = imagehash.phash(image, hash_size=8, highfreq_factor=4)
        return int(str(digest), 16)
    except ImportError:
        from scipy.fftpack import dct

        image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").resize(
            (32, 32), Image.Resampling.LANCZOS
        )
        pixels = np.asarray(image, dtype=np.float64)
        transformed = dct(dct(pixels, axis=0), axis=1)
        low = transformed[:8, :8]
        median = np.median(low)
        bits = (low > median).reshape(-1)
        output = 0
        for bit in bits:
            output = (output << 1) | int(bit)
        return output


def image_hash_metadata(path: str | Path, *, allow_truncated: bool = False) -> dict[str, Any]:
    """Produce all raw and decoded identity fields for an image-manifest row."""

    pixels, mode = decode_grayscale(path, allow_truncated=allow_truncated)
    return {
        "width": int(pixels.shape[1]),
        "height": int(pixels.shape[0]),
        "mode": mode,
        "file_sha256": file_sha256(path),
        "pixel_sha256": decoded_pixel_sha256(pixels),
        "phash64": phash64(pixels),
    }


def canonical_float448(array: np.ndarray, side: int = 448) -> np.ndarray:
    """Build the source-independent Q3 grayscale field for duplicate confirmation."""

    if array.dtype != np.uint8 or array.ndim != 2:
        raise ValueError("canonical_float448 requires a two-dimensional uint8 image")
    image = Image.fromarray(array, mode="L").resize((side, side), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / np.float32(255.0)
