from __future__ import annotations

import unittest

import pandas as pd

from seminspect.data.duplicates import (
    DuplicateAuditError,
    assert_no_cross_partition_components,
    assign_duplicate_groups,
    exact_duplicate_pairs,
    hamming64,
    phash_candidate_pairs,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"image_id": "a", "phash64": 0, "pixel_sha256": "a" * 64, "split": "router_train", "pair_id": "p1", "quarantined": False},
            {"image_id": "b", "phash64": 1, "pixel_sha256": "b" * 64, "split": "final_test", "pair_id": None, "quarantined": False},
        ]
    )


class DuplicateTest(unittest.TestCase):
    def test_hamming_distance(self) -> None:
        self.assertEqual(hamming64(0b1010, 0b0011), 2)

    def test_phash_candidate_pairs(self) -> None:
        pairs = phash_candidate_pairs(_frame(), radius=1)
        self.assertEqual([(pair.first_id, pair.second_id) for pair in pairs], [("a", "b")])

    def test_exact_duplicate_pairs_uses_decoded_pixel_identity(self) -> None:
        frame = _frame()
        frame.loc[1, "pixel_sha256"] = frame.loc[0, "pixel_sha256"]
        pairs = exact_duplicate_pairs(frame)
        self.assertEqual([(pair.first_id, pair.second_id, pair.source) for pair in pairs], [("a", "b", "pixel_sha256")])

    def test_cross_partition_duplicate_component_hard_fails(self) -> None:
        frame = _frame()
        grouped, _ = assign_duplicate_groups(frame, [])
        grouped.loc[:, "duplicate_group"] = "component"
        with self.assertRaises(DuplicateAuditError):
            assert_no_cross_partition_components(grouped)
