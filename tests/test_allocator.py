from __future__ import annotations

import unittest

from seminspect.policies.allocator import CostSchedule, Observation, ObservationTable, allocate_deterministic, full_q3, survey_only
from seminspect.policies.vor import predicted_gain_policy


def _table() -> ObservationTable:
    rows = []
    for image_id in ("a", "b"):
        for quality in range(4):
            rows.append(
                Observation(
                    image_id=image_id,
                    quality=quality,
                    calibrated_probability=0.1 + quality * 0.1,
                    detector_raw_score=0.1,
                    predicted_gain=1.0 if quality < 3 else None,
                )
            )
    return ObservationTable(rows)


class AllocatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = CostSchedule.from_values("pixel", [0.015625, 0.0625, 0.25, 1.0])

    def test_budget_never_exceeded(self) -> None:
        cap = (2 * self.schedule.q0_cost + self.schedule.incremental(0)) / 2
        result = allocate_deterministic(_table(), predicted_gain_policy(), self.schedule, cap)
        self.assertLessEqual(result.actual_cost, cap + 1e-12)

    def test_heap_tie_break_is_deterministic(self) -> None:
        cap = (2 * self.schedule.q0_cost + self.schedule.incremental(0)) / 2
        result = allocate_deterministic(_table(), predicted_gain_policy(), self.schedule, cap)
        self.assertEqual(result.final_qualities["a"], 1)
        self.assertEqual(result.final_qualities["b"], 0)

    def test_survey_and_full_reference_costs(self) -> None:
        self.assertAlmostEqual(survey_only(_table(), self.schedule, self.schedule.q0_cost).actual_cost, 0.015625)
        self.assertAlmostEqual(full_q3(_table(), self.schedule).actual_cost, 1.0)

    def test_no_quality_downgrade_and_at_most_three_upgrades(self) -> None:
        result = allocate_deterministic(_table(), predicted_gain_policy(), self.schedule, 1.0)
        self.assertTrue(all(quality == 3 for quality in result.final_qualities.values()))
        for path in result.trajectories.values():
            qualities = [int(item["quality"]) for item in path]
            self.assertEqual(qualities, sorted(qualities))
            self.assertLessEqual(len(qualities) - 1, 3)

    def test_infeasible_survey_cap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            allocate_deterministic(_table(), predicted_gain_policy(), self.schedule, 0.01)
