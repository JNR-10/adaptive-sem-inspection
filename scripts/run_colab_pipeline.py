#!/usr/bin/env python3
"""Run the complete protocol as resumable, Drive-persistent Colab stages.

This is the supported entry point for Google Colab.  It deliberately keeps the
development, sealed-final, and exploratory post-lock phases separate so a final
test cannot be inspected while development choices are still being made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "core_v1.yaml"
DEFAULT_STRESS_SCENARIOS = (
    "poisson_mild",
    "poisson_moderate",
    "poisson_severe",
    "contrast_moderate",
    "contrast_severe",
    "blur_moderate",
    "blur_severe",
    "compound_moderate",
    "compound_severe",
)
SUPPORTED_PYTHON_MIN = (3, 11)
SUPPORTED_PYTHON_MAX_EXCLUSIVE = (3, 13)


@dataclass(frozen=True)
class Command:
    """One restart-safe command boundary in the Colab workflow."""

    key: str
    label: str
    argv: tuple[str, ...]


def _is_supported_python(version: tuple[int, int]) -> bool:
    """Return whether a Python major/minor version has locked wheels."""

    return SUPPORTED_PYTHON_MIN <= version < SUPPORTED_PYTHON_MAX_EXCLUSIVE


def _require_supported_python(version: tuple[int, int] | None = None) -> None:
    """Fail early instead of letting a locked GPU dependency fail obscurely."""

    current = sys.version_info[:2] if version is None else version
    if _is_supported_python(current):
        return
    raise RuntimeError(
        "This project is locked for Python 3.11 or 3.12, but this launcher is running "
        f"on Python {current[0]}.{current[1]}. Current Colab runtimes can use Python 3.13. "
        "Run the Python 3.12 bootstrap in RUN_ADAPTIVE_SEM_IN_COLAB.ipynb or "
        "COLAB_QUICKSTART.md; do not call this launcher with Colab's /usr/bin/python3."
    )


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_state(path: Path, *, runtime_root: Path, config: Path) -> dict:
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("schema_version") != 1:
            raise RuntimeError(f"Unsupported Colab state file schema: {path}")
        return state
    return {
        "schema_version": 1,
        "project_root": str(ROOT),
        "runtime_root": str(runtime_root),
        "config": str(config),
        "created_utc": _utc_now(),
        "stages": {},
        "commands": {},
    }


def _save_state(path: Path, state: dict) -> None:
    state["updated_utc"] = _utc_now()
    _atomic_json(path, state)


def _stage_key(command: Command) -> str:
    digest = hashlib.sha256("\0".join(command.argv).encode("utf-8")).hexdigest()[:12]
    return f"{command.key}-{digest}"


def _environment(runtime_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    source = str(ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not existing else f"{source}{os.pathsep}{existing}"
    environment["SEMSPECT_RUNTIME_ROOT"] = str(runtime_root)
    environment["SEMSPECT_ENVIRONMENT_LOCK"] = str(ROOT / "requirements-colab.lock")
    environment["PYTHONHASHSEED"] = "20260917"
    environment["MPLCONFIGDIR"] = str(runtime_root / "cache" / "matplotlib")
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return environment


def _stream_process(argv: Sequence[str], *, environment: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] $ {' '.join(argv)}\n")
        process = subprocess.Popen(
            list(argv),
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        result = process.wait()
        if result != 0:
            raise RuntimeError(
                f"Command failed with exit code {result}: {' '.join(argv)}\n"
                f"Full command log: {log_path}"
            )


def _run_command(
    command: Command,
    *,
    state: dict,
    state_path: Path,
    environment: dict[str, str],
    logs_root: Path,
) -> None:
    key = _stage_key(command)
    prior = state["commands"].get(key)
    if prior and prior.get("status") == "completed":
        print(f"[resume] {command.label}")
        return
    print(f"\n=== {command.label} ===")
    state["commands"][key] = {
        "key": command.key,
        "label": command.label,
        "argv": list(command.argv),
        "status": "running",
        "started_utc": _utc_now(),
    }
    _save_state(state_path, state)
    try:
        _stream_process((sys.executable, *command.argv), environment=environment, log_path=logs_root / f"{key}.log")
    except BaseException as exc:
        state["commands"][key].update(
            {"status": "failed", "finished_utc": _utc_now(), "error": f"{type(exc).__name__}: {exc}"}
        )
        _save_state(state_path, state)
        raise
    state["commands"][key].update({"status": "completed", "finished_utc": _utc_now()})
    _save_state(state_path, state)


def _commands_for_development(
    config: Path,
    runtime_root: Path,
    *,
    miic_manual_bundle: Path | None = None,
) -> tuple[Command, ...]:
    raw = runtime_root / "data" / "raw"
    commands: list[Command] = []
    if miic_manual_bundle is not None:
        commands.append(
            Command(
                "stage_miic_manual_bundle",
                "Stage and verify browser-downloaded MIIC files",
                (
                    "scripts/stage_miic_manual_bundle.py",
                    "--source",
                    str(miic_manual_bundle),
                    "--destination",
                    str(raw),
                ),
            )
        )
    commands.extend(
        (
            Command("download_miic", "Download and checksum MIIC", ("scripts/download_data.py", "--dataset", "miic", "--destination", str(raw))),
            Command("download_carinthia", "Download and checksum Carinthia-S", ("scripts/download_data.py", "--dataset", "carinthia_s", "--destination", str(raw))),
            Command("verify_miic", "Verify MIIC archives", ("scripts/verify_archives.py", "--config", str(ROOT / "configs/data/miic.yaml"))),
            Command("verify_carinthia", "Verify Carinthia-S archive", ("scripts/verify_archives.py", "--config", str(ROOT / "configs/data/carinthia_s.yaml"))),
            Command("prepare_miic", "Extract and manifest MIIC", ("scripts/prepare_miic.py", "--config", str(config))),
            Command("prepare_carinthia", "Extract and audit Carinthia-S", ("scripts/prepare_carinthia.py", "--config", str(config))),
            Command("audit_leakage", "Build leakage-safe MIIC splits", ("scripts/audit_leakage.py", "--config", str(config))),
            Command("unit_tests", "Run protocol tests", ("-m", "pytest", "-q")),
            Command("memory_features", "Build Q0–Q3 normal memory features", ("scripts/build_quality_features.py", "--config", str(config), "--split", "normal_memory_train")),
            Command("memory_banks", "Build quality-specific memory banks", ("scripts/build_memory_banks.py", "--config", str(config))),
            Command("development_features", "Build development detector features", ("scripts/build_quality_features.py", "--config", str(config), "--split", "score_calibration,router_train,router_validation,policy_calibration")),
            Command("score_calibration", "Fit detector score calibrators", ("scripts/fit_calibrators.py", "--config", str(config))),
            Command("detector_gate", "Run detector feasibility gate", ("scripts/run_detector_feasibility.py", "--config", str(config))),
            Command("router_dataset", "Build Value-of-Resolution dataset", ("scripts/build_router_dataset.py", "--config", str(config))),
            Command("router_training", "Train Ridge and multi-seed MLP routers", ("scripts/train_router.py", "--config", str(config), "--seeds", "0", "1", "2", "3", "4")),
            Command("policy_calibration", "Calibrate policy thresholds", ("scripts/calibrate_policies.py", "--config", str(config))),
        )
    )
    return tuple(commands)


def _commands_for_final(config: Path, runtime_root: Path) -> tuple[Command, ...]:
    lock = runtime_root / "artifacts" / "test_lock.json"
    run_dir = runtime_root / "runs" / "core_v1_final"
    return (
        Command("freeze_protocol", "Seal the committed core_v1 protocol", ("scripts/freeze_protocol.py", "--config", str(config))),
        Command("final_features", "Extract sealed final-test features", ("scripts/build_quality_features.py", "--config", str(config), "--split", "final_test", "--lock", str(lock))),
        Command("final_benchmark", "Run the sealed final benchmark", ("scripts/run_final_benchmark.py", "--lock", str(lock), "--run-dir", str(run_dir))),
        Command("primary_report", "Generate the primary benchmark report", ("scripts/generate_report.py", "--run-dir", str(run_dir))),
    )


def _commands_for_postlock(config: Path, runtime_root: Path, scenarios: Sequence[str]) -> tuple[Command, ...]:
    lock = runtime_root / "artifacts" / "test_lock.json"
    run_dir = runtime_root / "runs" / "core_v1_final"
    commands: list[Command] = [
        Command("carinthia_features", "Build Carinthia-S secondary features", ("scripts/build_quality_features.py", "--config", str(config), "--split", "carinthia_secondary", "--lock", str(lock))),
        Command("secondary_analyses", "Run defect-size, prevalence, and Carinthia analyses", ("scripts/run_secondary_analyses.py", "--lock", str(lock), "--run-dir", str(run_dir), "--analysis", "all")),
    ]
    for scenario in scenarios:
        commands.extend(
            (
                Command(
                    f"stress_calibration_{scenario}",
                    f"Build stressed policy-calibration features ({scenario})",
                    ("scripts/build_quality_features.py", "--config", str(config), "--split", "policy_calibration", "--stress-scenario", scenario, "--lock", str(lock)),
                ),
                Command(
                    f"stress_final_{scenario}",
                    f"Build stressed final-test features ({scenario})",
                    ("scripts/build_quality_features.py", "--config", str(config), "--split", "final_test", "--stress-scenario", scenario, "--lock", str(lock)),
                ),
            )
        )
    scenario_key = "-".join(scenarios)
    commands.extend(
        (
            Command("stress_robustness", "Run post-lock stress robustness analysis", ("scripts/run_stress_robustness.py", "--lock", str(lock), "--run-dir", str(run_dir), "--scenarios", *scenarios)),
            Command("router_ablations", "Run post-lock router ablations", ("scripts/run_ablations.py", "--lock", str(lock), "--run-dir", str(run_dir))),
            Command("failure_galleries", "Render deterministic failure galleries", ("scripts/render_failure_galleries.py", "--lock", str(lock), "--run-dir", str(run_dir))),
            Command(f"postlock_report_{scenario_key}", "Refresh report with exploratory analyses", ("scripts/generate_report.py", "--run-dir", str(run_dir), "--force")),
        )
    )
    return tuple(commands)


def _install_dependencies(environment: dict[str, str], logs_root: Path) -> None:
    print("\n=== Install the pinned Colab environment ===")
    _stream_process((sys.executable, "-m", "pip", "install", "--upgrade", "pip"), environment=environment, log_path=logs_root / "install_pip.log")
    _stream_process((sys.executable, "-m", "pip", "install", "-r", "requirements-colab.lock"), environment=environment, log_path=logs_root / "install_requirements.log")
    _stream_process((sys.executable, "-m", "pip", "install", "-e", "."), environment=environment, log_path=logs_root / "install_project.log")


def _require_gpu(environment: dict[str, str], logs_root: Path) -> None:
    check = "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.cuda.get_device_name(0))"
    try:
        _stream_process((sys.executable, "-c", check), environment=environment, log_path=logs_root / "gpu_check.log")
    except RuntimeError as exc:
        raise RuntimeError("No Colab GPU is available. Choose Runtime > Change runtime type > T4 GPU, then rerun.") from exc


def _git_output(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(ROOT), *argv], capture_output=True, text=True, check=False)


def _ensure_git_commit(args: argparse.Namespace) -> None:
    head = _git_output(("rev-parse", "HEAD"))
    if head.returncode == 0:
        return
    if not args.initialize_git:
        raise RuntimeError(
            "The final test is intentionally blocked until source is committed. "
            "Rerun the final stage with --initialize-git, or initialize and commit this Drive-backed project yourself."
        )
    if not shutil.which("git"):
        raise RuntimeError("Git is unavailable in this runtime; use a standard Colab runtime and retry.")
    init = _git_output(("init",))
    if init.returncode != 0:
        raise RuntimeError(init.stderr.strip() or "git init failed")
    for setting, value in (("user.name", args.git_name), ("user.email", args.git_email)):
        configured = _git_output(("config", "--get", setting))
        if configured.returncode != 0:
            outcome = _git_output(("config", setting, value))
            if outcome.returncode != 0:
                raise RuntimeError(outcome.stderr.strip() or f"Could not configure git {setting}")
    add = _git_output(("add", "--all"))
    if add.returncode != 0:
        raise RuntimeError(add.stderr.strip() or "git add failed")
    commit = _git_output(("commit", "-m", "Lock source for adaptive SEM Colab run"))
    if commit.returncode != 0:
        raise RuntimeError(commit.stderr.strip() or "git commit failed")
    print(f"Created source commit { _git_output(('rev-parse', 'HEAD')).stdout.strip() }")


def _run_stage(
    name: str,
    commands: Sequence[Command],
    *,
    state: dict,
    state_path: Path,
    environment: dict[str, str],
    logs_root: Path,
) -> None:
    state["stages"][name] = {"status": "running", "started_utc": _utc_now()}
    _save_state(state_path, state)
    try:
        for command in commands:
            _run_command(command, state=state, state_path=state_path, environment=environment, logs_root=logs_root)
    except BaseException as exc:
        state["stages"][name] = {"status": "failed", "finished_utc": _utc_now(), "error": f"{type(exc).__name__}: {exc}"}
        _save_state(state_path, state)
        raise
    state["stages"][name] = {"status": "completed", "finished_utc": _utc_now()}
    _save_state(state_path, state)


def _print_status(state: dict, state_path: Path) -> None:
    print(f"Colab state: {state_path}")
    for name, details in state.get("stages", {}).items():
        print(f"  {name}: {details.get('status', 'unknown')}")
    completed = sum(item.get("status") == "completed" for item in state.get("commands", {}).values())
    failed = sum(item.get("status") == "failed" for item in state.get("commands", {}).values())
    print(f"  command checkpoints: {completed} complete, {failed} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "final", "postlock", "all", "status"), default="development")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runtime-root", type=Path, help="Persistent Drive directory for data, caches, logs, and results")
    parser.add_argument("--install", action="store_true", help="Install the pinned Colab packages before running the requested stage")
    parser.add_argument("--accept-miic-license", action="store_true", help="Confirm CC BY-NC 4.0 non-commercial research use before MIIC download")
    parser.add_argument(
        "--miic-manual-bundle",
        type=Path,
        help="Official MIIC ZIP or verified folder of archives (including .part-000 shards) when automatic download is unreliable",
    )
    parser.add_argument("--initialize-git", action="store_true", help="Create the one required local source commit before the sealed final stage")
    parser.add_argument("--git-name", default="Adaptive SEM Colab")
    parser.add_argument("--git-email", default="adaptive-sem@local")
    parser.add_argument("--stress-scenarios", nargs="+", default=list(DEFAULT_STRESS_SCENARIOS))
    args = parser.parse_args()

    config = args.config.resolve()
    if not config.is_file():
        raise SystemExit(f"Configuration does not exist: {config}")
    runtime_root = (args.runtime_root or Path(os.environ.get("SEMSPECT_RUNTIME_ROOT", "/content/sem-inspection"))).resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    if "/content/drive/" not in f"{runtime_root}/":
        print("WARNING: runtime root is not under mounted Drive; a Colab reset will remove experiment state.")
    invalid = sorted(set(args.stress_scenarios) - set(DEFAULT_STRESS_SCENARIOS))
    if invalid:
        raise SystemExit(f"Unknown core_v1 stress scenarios: {', '.join(invalid)}")
    miic_manual_bundle = args.miic_manual_bundle.resolve() if args.miic_manual_bundle else None
    if miic_manual_bundle is not None and not miic_manual_bundle.exists():
        raise SystemExit(f"Manual MIIC bundle does not exist: {miic_manual_bundle}")

    state_path = runtime_root / "artifacts" / "colab_pipeline_state.json"
    logs_root = runtime_root / "logs" / "colab_pipeline"
    environment = _environment(runtime_root)
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    state = _load_state(state_path, runtime_root=runtime_root, config=config)
    if args.stage == "status":
        _print_status(state, state_path)
        return
    _require_supported_python()
    if args.install:
        _install_dependencies(environment, logs_root)
    _require_gpu(environment, logs_root)

    wants_development = args.stage in {"development", "all"}
    wants_final = args.stage in {"final", "all"}
    wants_postlock = args.stage in {"postlock", "all"}
    if wants_development:
        if not args.accept_miic_license:
            raise SystemExit(
                "MIIC is CC BY-NC 4.0 and is not bundled. Read its license, then rerun with --accept-miic-license to download it."
            )
        _run_stage(
            "development",
            _commands_for_development(config, runtime_root, miic_manual_bundle=miic_manual_bundle),
            state=state,
            state_path=state_path,
            environment=environment,
            logs_root=logs_root,
        )
        print("\nDevelopment complete. The final test remains sealed. Run the final stage only when you are ready to lock the protocol.")
    if wants_final:
        _ensure_git_commit(args)
        _run_stage("final", _commands_for_final(config, runtime_root), state=state, state_path=state_path, environment=environment, logs_root=logs_root)
        print(f"\nFinal benchmark complete: {runtime_root / 'runs' / 'core_v1_final'}")
    if wants_postlock:
        _run_stage("postlock", _commands_for_postlock(config, runtime_root, args.stress_scenarios), state=state, state_path=state_path, environment=environment, logs_root=logs_root)
        print(f"\nPost-lock analyses complete: {runtime_root / 'runs' / 'core_v1_final'}")


if __name__ == "__main__":
    main()
