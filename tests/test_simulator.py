from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from seminspect.simulator.noise import StressScenario, apply_stress, stress_seed
from seminspect.simulator.resolution import Quality, QualityLadder, canonical_q3_from_array, simulate_resolution


TORCH = importlib.util.find_spec("torch") is not None


class SimulatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ladder = QualityLadder(
            [
                Quality(0, "Q0", 56, 0.015625),
                Quality(1, "Q1", 112, 0.0625),
                Quality(2, "Q2", 224, 0.25),
                Quality(3, "Q3", 448, 1.0),
            ]
        )
        self.q3 = canonical_q3_from_array(np.arange(512 * 512, dtype=np.uint8).reshape(512, 512))

    def test_q3_is_canonical_448(self) -> None:
        q3 = simulate_resolution(self.q3, self.ladder["Q3"])
        self.assertEqual(q3.shape, (448, 448))
        self.assertEqual(q3.dtype, np.float32)
        np.testing.assert_array_equal(q3, self.q3)

    @unittest.skipUnless(TORCH, "Torch-specific primary simulator test runs in Colab reference environment")
    def test_quality_effective_sides_exact(self) -> None:
        levels = [simulate_resolution(self.q3, quality) for quality in self.ladder]
        self.assertTrue(all(level.shape == (448, 448) for level in levels))
        self.assertFalse(np.array_equal(levels[0], levels[-1]))

    def test_stress_seed_deterministic(self) -> None:
        scenario = StressScenario("test", 32, 0.8, 0.5, 0.01)
        first = apply_stress(self.q3, image_id="id", quality_id="Q1", scenario=scenario, global_seed=20260917)
        second = apply_stress(self.q3, image_id="id", quality_id="Q1", scenario=scenario, global_seed=20260917)
        self.assertEqual(stress_seed("id", "Q1", "test", 20260917), stress_seed("id", "Q1", "test", 20260917))
        np.testing.assert_array_equal(first, second)
