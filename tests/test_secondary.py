from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from seminspect.evaluation.failure_cases import select_failure_cases
from seminspect.evaluation.secondary import mask_statistics, prevalence_lots


class SecondaryAnalysisTest(unittest.TestCase):
    def test_mask_statistics_use_eight_connected_components_and_half_open_bbox(self) -> None:
        mask = np.zeros((5, 6), dtype=bool)
        mask[1, 2] = True
        mask[2, 3] = True  # diagonal connection is one 8-connected component
        values = mask_statistics(mask)
        self.assertEqual(values["mask_pixels"], 2)
        self.assertEqual(values["connected_component_count"], 1)
        self.assertEqual(values["bbox_width"], 2)
        self.assertEqual(values["bbox_height"], 2)

    def test_prevalence_lots_have_exact_composition_and_are_deterministic(self) -> None:
        labels = {f"n{index:04d}": 0 for index in range(1272)}
        labels.update({f"a{index:04d}": 1 for index in range(116)})
        first = prevalence_lots(labels)
        second = prevalence_lots(labels)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 60)
        for anomaly_count, _, identifiers in first:
            self.assertEqual(len(identifiers), 1000)
            self.assertEqual(len(set(identifiers)), 1000)
            self.assertEqual(sum(labels[item] for item in identifiers), anomaly_count)

    def test_failure_selection_is_bounded_and_tie_stable(self) -> None:
        rows = []
        for index in range(25):
            rows.append(
                {
                    "image_id": f"id-{index:02d}",
                    "label": int(index % 2 == 0),
                    "prediction": int(index % 3 == 0),
                    "score_margin": float(index - 10),
                    "q0_prediction": 0,
                    "q3_minus_q0_probability": float(index),
                    "q0_entropy": float(index),
                    "realized_gain": -float(index),
                    "stopped_with_positive_unacquired_gain": int(index % 4 == 0),
                    "unacquired_true_gain": float(index),
                    "final_quality": 3,
                    "absolute_q3_q0_probability_change": float(index),
                    "vor_uncertainty_quality_difference": float(index % 4),
                }
            )
        selected = select_failure_cases(pd.DataFrame(rows), cap=20)
        self.assertEqual(set(selected), {
            "false_negatives",
            "false_positives",
            "corrected_low_resolution_misses",
            "high_uncertainty_no_help",
            "harmful_upgrades",
            "stopped_but_correctable",
            "q3_no_decision_change",
            "policy_disagreements",
        })
        self.assertTrue(all(len(frame) <= 20 for frame in selected.values()))
