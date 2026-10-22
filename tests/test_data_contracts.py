from __future__ import annotations

import os
import unittest
from pathlib import Path

import pandas as pd

from seminspect.data.carinthia import decode_carinthia_mask
from seminspect.data.access import resolve_manifest_path
from seminspect.data.manifests import read_manifest
from seminspect.data.manifests import MANIFEST_COLUMNS, ManifestError, normalize_manifest
from seminspect.utils.io import file_md5


def _row(image_id: str, *, split: str = "unassigned", duplicate_group: str | None = "a" * 64) -> dict:
    return {
        "image_id": image_id,
        "dataset": "miic",
        "source_archive": "Anomaly_train.rar",
        "relative_path": f"Anomaly_train/{image_id}.jpg",
        "label": 0,
        "pair_id": None,
        "mask_path": None,
        "class_id": None,
        "width": 512,
        "height": 512,
        "mode": "L",
        "file_sha256": "b" * 64,
        "pixel_sha256": "a" * 64,
        "phash64": 0,
        "split": split,
        "duplicate_group": duplicate_group,
        "quarantined": False,
        "quarantine_reason": None,
    }


class DataContractsTest(unittest.TestCase):
    def test_manifest_has_canonical_columns(self) -> None:
        frame = normalize_manifest(pd.DataFrame([_row("one")]))
        self.assertEqual(frame.columns.tolist(), MANIFEST_COLUMNS)
        self.assertEqual(frame.iloc[0].image_id, "one")

    def test_manifest_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ManifestError):
            normalize_manifest(pd.DataFrame([_row("one"), _row("one")]))



@unittest.skipUnless(
    os.environ.get("SEMSPECT_INTEGRATION_DATA") == "1",
    "Set SEMSPECT_INTEGRATION_DATA=1 in Colab to run licensed downloaded-data integration tests.",
)
class DownloadedDataContractsTest(unittest.TestCase):
    @property
    def root(self):
        value = os.environ.get("SEMSPECT_RUNTIME_ROOT")
        if not value:
            self.fail("SEMSPECT_RUNTIME_ROOT must point at the prepared Colab runtime")
        return Path(value)

    def test_miic_archive_checksums(self) -> None:
        expected = {
            "Anomaly_train.rar": (666594169, "b7c7201863c4fe92e494caec3c35429f"),
            "Anomaly_test.rar": (57307813, "8564777749e0de87348c785221d06d69"),
            "Inpainting_train.rar": (76171335, "05692f52bf2f011fea72548cc8420785"),
            "Inpainting_test.rar": (29813335, "671576e5302bc1b64bc2f369b2223cf1"),
        }
        for name, (size, checksum) in expected.items():
            path = self.root / "data" / "raw" / name
            self.assertEqual(path.stat().st_size, size, name)
            self.assertEqual(file_md5(path), checksum, name)

    def test_miic_final_counts_1272_116(self) -> None:
        manifest = read_manifest(self.root / "data" / "manifests" / "miic_raw.parquet")
        final = manifest.loc[manifest["source_archive"].eq("Anomaly_test.rar")]
        self.assertEqual(int((final["label"].astype("Int64") == 0).sum()), 1272)
        self.assertEqual(int((final["label"].astype("Int64") == 1).sum()), 116)

    def test_miic_masks_116_and_512_square(self) -> None:
        manifest = read_manifest(self.root / "data" / "manifests" / "miic_raw.parquet")
        anomalies = manifest.loc[(manifest["source_archive"] == "Anomaly_test.rar") & (manifest["label"].astype("Int64") == 1)]
        self.assertEqual(len(anomalies), 116)
        self.assertTrue(anomalies["mask_path"].notna().all())
        self.assertTrue((anomalies["width"].astype(int) == 512).all())
        self.assertTrue((anomalies["height"].astype(int) == 512).all())
        for row in anomalies.itertuples(index=False):
            self.assertTrue(resolve_manifest_path(row, self.root / "data" / "extracted", column="mask_path").is_file())

    def test_miic_train_minus_test_equals_23888_before_near_duplicate_filter(self) -> None:
        manifest = read_manifest(self.root / "data" / "manifests" / "miic_raw.parquet")
        train = manifest.loc[manifest["source_archive"].eq("Anomaly_train.rar")]
        final = manifest.loc[manifest["source_archive"].eq("Anomaly_test.rar")]
        remainder = train.loc[~train["pixel_sha256"].isin(set(final["pixel_sha256"]))]
        self.assertEqual(len(remainder), 23888)

    def test_carinthia_count_dimensions_and_class6_override(self) -> None:
        manifest = read_manifest(self.root / "data" / "manifests" / "carinthia_s.parquet")
        self.assertEqual(len(manifest), 4591)
        self.assertTrue((manifest["width"].astype(int) == 480).all())
        self.assertTrue((manifest["height"].astype(int) == 480).all())
        class6 = manifest.loc[manifest["class_id"].astype(str).eq("6")]
        self.assertEqual(len(class6), 227)
        self.assertTrue(class6["label"].astype("Int64").eq(0).all())
        # The released class-6 mask mark is retained for audit but cannot
        # change the image-level no-visible-defect definition.
        foregrounds = [
            decode_carinthia_mask(resolve_manifest_path(row, self.root / "data" / "extracted", column="mask_path"))[0].sum()
            for row in class6.itertuples(index=False)
        ]
        self.assertGreaterEqual(max(foregrounds), 50)
