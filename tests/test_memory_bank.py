from __future__ import annotations

import unittest

import numpy as np

from seminspect.detector.memory_bank import ReservoirSampler, descriptor_coordinates


class MemoryBankTest(unittest.TestCase):
    def test_reservoir_is_deterministic(self) -> None:
        coordinates = descriptor_coordinates((2, 2))
        first = ReservoirSampler(4, 3, 17)
        second = ReservoirSampler(4, 3, 17)
        values = np.arange(24, dtype=np.float32).reshape(8, 3) + 1
        for sampler in (first, second):
            sampler.add(values[:4], "a", coordinates)
            sampler.add(values[4:], "b", coordinates)
        one, two = first.finish(), second.finish()
        np.testing.assert_array_equal(one.descriptors, two.descriptors)
        self.assertEqual(one.size, 4)
        self.assertEqual(one.dimension, 3)
