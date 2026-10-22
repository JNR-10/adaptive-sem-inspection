from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_pipeline", ROOT / "scripts" / "run_pipeline.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NativePipelineTest(unittest.TestCase):
    def test_native_environment_is_headless_and_runtime_scoped(self) -> None:
        environment = MODULE._environment(Path("/tmp/adaptive-sem-runtime"))
        self.assertEqual(environment["SEMSPECT_RUNTIME_ROOT"], "/tmp/adaptive-sem-runtime")
        self.assertEqual(environment["MPLBACKEND"], "Agg")
        self.assertTrue(environment["PYTHONPATH"].split(":")[0].endswith("src"))

    def test_native_pipeline_keeps_final_lock_before_final_features(self) -> None:
        commands = MODULE._commands_for_final(ROOT / "configs/experiment/core_v1.yaml", Path("/tmp/runtime"))
        self.assertEqual([command.key for command in commands], ["freeze_protocol", "final_features", "final_benchmark", "primary_report"])

    def test_supported_python_range_matches_native_documentation(self) -> None:
        self.assertTrue(MODULE._is_supported_python((3, 11)))
        self.assertTrue(MODULE._is_supported_python((3, 12)))
        self.assertFalse(MODULE._is_supported_python((3, 13)))
