from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("colab_pipeline", ROOT / "scripts" / "run_colab_pipeline.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ColabPipelineTest(unittest.TestCase):
    def test_supported_python_versions_match_the_locked_environment(self) -> None:
        self.assertTrue(MODULE._is_supported_python((3, 11)))
        self.assertTrue(MODULE._is_supported_python((3, 12)))
        self.assertFalse(MODULE._is_supported_python((3, 10)))
        self.assertFalse(MODULE._is_supported_python((3, 13)))
        with self.assertRaisesRegex(RuntimeError, "Python 3.13"):
            MODULE._require_supported_python((3, 13))

    def test_development_plan_has_the_required_gate_order(self) -> None:
        keys = [item.key for item in MODULE._commands_for_development(ROOT / "configs/experiment/core_v1.yaml", Path("/tmp/runtime"))]
        self.assertLess(keys.index("audit_leakage"), keys.index("memory_features"))
        self.assertLess(keys.index("memory_banks"), keys.index("score_calibration"))
        self.assertLess(keys.index("detector_gate"), keys.index("router_dataset"))
        self.assertEqual(keys[-1], "policy_calibration")

    def test_development_plan_can_stage_a_verified_manual_bundle_first(self) -> None:
        commands = MODULE._commands_for_development(
            ROOT / "configs/experiment/core_v1.yaml",
            Path("/tmp/runtime"),
            miic_manual_bundle=Path("/tmp/miic-download.zip"),
        )
        self.assertEqual(commands[0].key, "stage_miic_manual_bundle")
        self.assertEqual(commands[1].key, "download_miic")

    def test_final_plan_keeps_final_features_after_the_lock(self) -> None:
        keys = [item.key for item in MODULE._commands_for_final(ROOT / "configs/experiment/core_v1.yaml", Path("/tmp/runtime"))]
        self.assertEqual(keys, ["freeze_protocol", "final_features", "final_benchmark", "primary_report"])

    def test_postlock_plan_builds_every_stressed_pair_before_evaluation(self) -> None:
        commands = MODULE._commands_for_postlock(
            ROOT / "configs/experiment/core_v1.yaml", Path("/tmp/runtime"), ("poisson_mild", "blur_severe")
        )
        keys = [item.key for item in commands]
        self.assertLess(keys.index("stress_calibration_poisson_mild"), keys.index("stress_robustness"))
        self.assertLess(keys.index("stress_final_blur_severe"), keys.index("stress_robustness"))

