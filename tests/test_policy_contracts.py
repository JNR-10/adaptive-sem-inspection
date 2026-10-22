from __future__ import annotations

from dataclasses import fields
import unittest

from seminspect.evaluation.locking import TestLockError, verify_test_lock
from seminspect.policies.allocator import Observation
from seminspect.policies.uncertainty import entropy_uncertainty_policy


class PolicyContractTest(unittest.TestCase):
    def test_observation_contract_excludes_labels_masks_cost_and_budget(self) -> None:
        names = {field.name for field in fields(Observation)}
        self.assertFalse({"label", "mask", "cost", "budget", "future_score"} & names)

    def test_uncertainty_policy_is_a_nonnegative_local_signal(self) -> None:
        policy = entropy_uncertainty_policy()
        observation = Observation("id", 0, 0.5, 0.1)
        self.assertGreater(policy.signal(observation, None), 0.0)

    def test_test_lock_rejects_changed_config_value(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            path.write_text(json.dumps({"config": "original"}), encoding="utf-8")
            with self.assertRaises(TestLockError):
                verify_test_lock(path, {"config": "changed"})
