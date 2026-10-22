from __future__ import annotations

import unittest

from seminspect.policies.allocator import CostSchedule


class CostTest(unittest.TestCase):
    def test_costs_strictly_increase(self) -> None:
        schedule = CostSchedule.from_values("pixel", [0.015625, 0.0625, 0.25, 1.0])
        self.assertEqual(schedule.q3_cost, 1.0)

    def test_incremental_costs_sum_to_cumulative_difference(self) -> None:
        schedule = CostSchedule.from_values("pixel", [0.015625, 0.0625, 0.25, 1.0])
        self.assertAlmostEqual(sum(schedule.incremental(index) for index in range(3)), 1.0 - 0.015625)

    def test_infeasible_cap_rejected_for_each_cost_schedule(self) -> None:
        schedule = CostSchedule.from_values("front", [0.1, 0.25, 0.5, 1.0])
        with self.assertRaises(ValueError):
            schedule.validate_budget(0.099)
