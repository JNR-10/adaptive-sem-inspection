from __future__ import annotations

import unittest

import numpy as np

from seminspect.detector.scoring import summarize_patch_distances


class DetectorScoringTest(unittest.TestCase):
    def test_top_one_aggregation_is_the_patch_maximum(self) -> None:
        distances = np.arange(16, dtype=np.float32)
        raw, maximum, top_std, _, anomaly_map = summarize_patch_distances(distances, (4, 4), top_k=1)
        self.assertEqual(raw, 15.0)
        self.assertEqual(maximum, 15.0)
        self.assertEqual(top_std, 0.0)
        self.assertEqual(anomaly_map.shape, (4, 4))

    def test_top_five_aggregation_is_distinct_from_top_one(self) -> None:
        distances = np.arange(16, dtype=np.float32)
        top_one, *_ = summarize_patch_distances(distances, (4, 4), top_k=1)
        top_five, *_ = summarize_patch_distances(distances, (4, 4), top_k=5)
        self.assertLess(top_five, top_one)
