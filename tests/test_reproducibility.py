from __future__ import annotations

import unittest

from seminspect.evaluation.thresholds import select_normal_fpr_threshold
from seminspect.detector.calibration import reliability_bins
from seminspect.utils.io import canonical_json_sha256


class ReproducibilityTest(unittest.TestCase):
    def test_canonical_json_hash_is_key_order_invariant(self) -> None:
        self.assertEqual(canonical_json_sha256({"a": 1, "b": 2}), canonical_json_sha256({"b": 2, "a": 1}))

    def test_threshold_order_statistic_is_700_of_735_and_ties_are_normal(self) -> None:
        threshold = select_normal_fpr_threshold([(f"id-{index:04d}", float(index)) for index in range(735)], alpha=0.05)
        self.assertEqual(threshold.rank_one_based, 700)
        self.assertEqual(threshold.value, 699.0)
        self.assertEqual(sum(float(index) > threshold.value for index in range(735)), 35)

    def test_reliability_bins_are_fixed_width_and_keep_empty_bins(self) -> None:
        rows = reliability_bins([0.01, 0.99], [0, 1], bins=4)
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(int(row["count"]) for row in rows), 2)
        self.assertEqual(rows[1]["count"], 0)
