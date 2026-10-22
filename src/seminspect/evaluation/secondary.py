"""Prespecified secondary mask-size and prevalence utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.ndimage import label as connected_components

from seminspect.data.access import resolve_manifest_path
from seminspect.data.carinthia import decode_carinthia_mask


def miic_binary_mask(path: str | Path) -> np.ndarray:
    """Decode an annotated MIIC JPEG mask into an explicit native binary array.

    MIIC calls the masks binary while distributing JPEG files. A threshold of 128
    reconstructs their black/white foreground and avoids treating JPEG ringing as
    tiny defects; any non-binary source level is thus handled deterministically.
    """

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        values = np.asarray(image, dtype=np.uint8)
    if values.shape != (512, 512):
        raise ValueError(f"MIIC native masks must be 512×512, got {values.shape} for {path}")
    return values >= 128


def mask_statistics(mask: np.ndarray) -> dict[str, int | float]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("Mask statistics require a two-dimensional mask")
    foreground = int(values.sum())
    if foreground == 0:
        raise ValueError("A real anomaly mask must have at least one foreground pixel")
    components, count = connected_components(values, structure=np.ones((3, 3), dtype=np.int8))
    areas = np.bincount(components.ravel())[1:]
    rows, columns = np.where(values)
    return {
        "mask_pixels": foreground,
        "mask_fraction": foreground / float(values.size),
        "connected_component_count": int(count),
        "largest_component_area": int(areas.max()),
        "bbox_width": int(columns.max() + 1 - columns.min()),
        "bbox_height": int(rows.max() + 1 - rows.min()),
    }


def miic_mask_statistics(manifest: pd.DataFrame, extracted_dir: str | Path) -> pd.DataFrame:
    anomalies = manifest.loc[(manifest["dataset"] == "miic") & (manifest["label"].astype("Int64") == 1)].copy()
    if len(anomalies) != 116:
        raise ValueError(f"MIIC final anomaly-size analysis requires 116 masks, got {len(anomalies)}")
    rows: list[dict] = []
    for row in anomalies.sort_values("image_id", kind="mergesort").itertuples(index=False):
        mask = miic_binary_mask(resolve_manifest_path(row, extracted_dir, column="mask_path"))
        rows.append({"image_id": str(row.image_id), **mask_statistics(mask)})
    stats = pd.DataFrame(rows).sort_values(["mask_pixels", "image_id"], kind="mergesort").reset_index(drop=True)
    if len(stats) != 116:
        raise RuntimeError("Unexpected MIIC mask-statistic cardinality")
    labels = np.repeat(["Tiny", "Small", "Medium", "Large"], 29)
    stats["size_quartile"] = labels
    return stats


def carinthia_mask_statistics(manifest: pd.DataFrame, extracted_dir: str | Path) -> pd.DataFrame:
    rows: list[dict] = []
    for row in manifest.sort_values("image_id", kind="mergesort").itertuples(index=False):
        foreground, mask_mode = decode_carinthia_mask(resolve_manifest_path(row, extracted_dir, column="mask_path"))
        entry = {
            "image_id": str(row.image_id),
            "class_id": str(row.class_id),
            "mask_mode": mask_mode,
            "image_label": int(row.label),
            "decoded_foreground_pixels": int(foreground.sum()),
        }
        if str(row.class_id) == "6":
            entry.update({"mask_pixels": np.nan, "mask_fraction": np.nan})
        else:
            if not foreground.any():
                raise ValueError(f"Carinthia defect-class mask is empty: {row.image_id}")
            entry.update({"mask_pixels": int(foreground.sum()), "mask_fraction": float(foreground.mean())})
        rows.append(entry)
    return pd.DataFrame(rows).sort_values("image_id", kind="mergesort").reset_index(drop=True)


def prevalence_lots(
    labels: Mapping[str, int],
    *,
    prevalence_counts: Iterable[int] = (10, 20, 50),
    lot_size: int = 1000,
    seed: int = 20260917,
    repeats: int = 20,
) -> list[tuple[int, int, list[str]]]:
    """Return exact-composition virtual lots sampled without replacement per lot."""

    normal = np.asarray(sorted(identifier for identifier, value in labels.items() if value == 0), dtype=object)
    anomaly = np.asarray(sorted(identifier for identifier, value in labels.items() if value == 1), dtype=object)
    result: list[tuple[int, int, list[str]]] = []
    for anomaly_count in prevalence_counts:
        if anomaly_count > len(anomaly) or lot_size - anomaly_count > len(normal):
            raise ValueError("Requested prevalence lot exceeds available final-test class counts")
        for repeat in range(repeats):
            rng = np.random.Generator(np.random.PCG64(seed + repeat))
            selected_anomaly = rng.choice(anomaly, size=anomaly_count, replace=False)
            selected_normal = rng.choice(normal, size=lot_size - anomaly_count, replace=False)
            selected = np.concatenate([selected_normal, selected_anomaly]).tolist()
            result.append((anomaly_count, repeat, [str(item) for item in selected]))
    return result
