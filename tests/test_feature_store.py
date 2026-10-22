from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from seminspect.detector.feature_store import validate_feature_shard, write_feature_shard


class FeatureStoreTest(unittest.TestCase):
    def test_resnet_sized_square_maps_are_valid_without_full_patch_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, map_reference = write_feature_shard(
                root,
                split="fallback",
                quality_id="Q0",
                part=0,
                image_ids=["a", "b"],
                cls_embeddings=np.ones((2, 384), dtype=np.float32),
                anomaly_maps=np.ones((2, 14, 14), dtype=np.float32),
                metadata={
                    "config_hash": "a",
                    "source_manifest_hash": "b",
                    "model_commit": "c",
                    "dtype": "float32",
                    "shape": "[2,384]",
                },
            )
            self.assertEqual(reference.row, 0)
            self.assertIsNotNone(map_reference)
            validate_feature_shard(root / reference.path)
