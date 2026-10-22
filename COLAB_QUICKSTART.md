# Run the complete project in one Colab workflow

You do **not** need to open the eight internal notebooks. They remain as
reference launchers, but the supported Colab path is one blank notebook that
runs the persistent sequential launcher below.

Use a GPU runtime (`Runtime` → `Change runtime type` → `T4 GPU`). Upload the
project ZIP to `MyDrive` first, then create a blank Colab notebook and run the
following cells in order. Current Colab notebooks can run Python 3.13, while
this reproducible GPU environment has locked Python 3.12 wheels. The second
cell creates the compatible Python 3.12 environment automatically; it is
temporary, while the project and every run artifact stay in Drive.

## 1. Mount Drive and stage the source once

```python
from google.colab import drive
from pathlib import Path
import zipfile

drive.mount("/content/drive")

# Change only this filename/path if you stored the ZIP elsewhere in Drive.
# Use new paths once so an older Python-3.13 launcher cannot be reused.
ZIP_ON_DRIVE = Path("/content/drive/MyDrive/adaptive-sem-inspection-v0.1.6-colab-python312.zip")
PROJECT = Path("/content/drive/MyDrive/adaptive-sem-inspection-v0.1.6")
RUNTIME_ROOT = Path("/content/drive/MyDrive/adaptive-sem-inspection-runtime-v0.1.6")

# Leave as None for the normal automatic download. If NTU's connection is
# unreliable, put the five verified MIIC files in one Drive folder and set this
# to that folder. The launcher also accepts Anomaly_train.rar.part-000 and
# Anomaly_train.rar.part-001 in place of the single training archive.
MIIC_MANUAL_BUNDLE = None
# MIIC_MANUAL_BUNDLE = Path("/content/drive/MyDrive/MIIC_verified_files")

if not ZIP_ON_DRIVE.is_file():
    raise FileNotFoundError(f"Upload the project ZIP first: {ZIP_ON_DRIVE}")

# The source and its Git history live in Drive; extracted data/results live in
# the separate runtime directory and therefore survive a Colab reset.
if not (PROJECT / "scripts" / "run_colab_pipeline.py").is_file():
    PROJECT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_ON_DRIVE) as archive:
        archive.extractall(PROJECT)

RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
print("Project:", PROJECT)
print("Persistent runtime:", RUNTIME_ROOT)
```

## 2. Create the compatible Python 3.12 runner

```python
import json
import os
import shutil
import subprocess
import sys

# This environment intentionally lives outside Drive: it is recreated after a
# Colab reset, while source/data/results remain in the two Drive folders above.
COLAB_VENV = Path("/content/adaptive-sem-python312")
COLAB_PYTHON = COLAB_VENV / "bin" / "python"

bootstrap_env = os.environ.copy()
bootstrap_env.setdefault("UV_CACHE_DIR", "/content/uv-cache")
bootstrap_env.setdefault("UV_PYTHON_INSTALL_DIR", "/content/uv-python")
UV = shutil.which("uv")
if UV is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "uv"], check=True)
    UV = shutil.which("uv")
if UV is None:
    raise RuntimeError("uv did not install correctly")

if not COLAB_PYTHON.is_file():
    subprocess.run([UV, "python", "install", "3.12"], env=bootstrap_env, check=True)
    subprocess.run(
        [UV, "venv", "--python", "3.12", "--managed-python", "--seed", "--allow-existing", str(COLAB_VENV)],
        env=bootstrap_env,
        check=True,
    )

# `--seed` gives the runner its own pip. Do not call ensurepip: newer uv
# environments can correctly reject it after pip has already been seeded.
if not (COLAB_VENV / "bin" / "pip").is_file():
    subprocess.run([UV, "pip", "install", "--python", str(COLAB_PYTHON), "pip"], env=bootstrap_env, check=True)

python_version = subprocess.check_output([str(COLAB_PYTHON), "--version"], text=True).strip()
if not python_version.startswith("Python 3.12."):
    raise RuntimeError(f"Expected the local Python 3.12 runner, found: {python_version}")
print("Stage runner:", COLAB_PYTHON)
print(python_version)
```

## 3. Install a current RAR extractor once per Colab runtime

MIIC's RAR5 archives require the current official 7-Zip console binary. This
cell is fast and needs to be rerun only after Colab resets the runtime.

```python
SEVEN_ZIP_URL = "https://github.com/ip7z/7zip/releases/download/26.04/7z2604-linux-x64.tar.xz"
SEVEN_ZIP_ROOT = Path("/content/7zip-26.04")
SEVEN_ZIP_ARCHIVE = Path("/content/7z2604-linux-x64.tar.xz")
SEVEN_ZIP_BIN = Path("/content/modern-7zip-bin")

if not (SEVEN_ZIP_ROOT / "7zz").is_file():
    subprocess.run(["wget", "--quiet", "--show-progress", "-O", str(SEVEN_ZIP_ARCHIVE), SEVEN_ZIP_URL], check=True)
    SEVEN_ZIP_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xJf", str(SEVEN_ZIP_ARCHIVE), "-C", str(SEVEN_ZIP_ROOT)], check=True)
    (SEVEN_ZIP_ROOT / "7zz").chmod(0o755)

SEVEN_ZIP_BIN.mkdir(parents=True, exist_ok=True)
shim = SEVEN_ZIP_BIN / "7z"
if shim.exists() or shim.is_symlink():
    shim.unlink()
shim.symlink_to(SEVEN_ZIP_ROOT / "7zz")
os.environ["PATH"] = f"{SEVEN_ZIP_BIN}:{os.environ['PATH']}"
subprocess.run(["7z", "i"], stdout=subprocess.DEVNULL, check=True)
print("Modern 7-Zip is ready:", shim)
```

## 4. Define the single-stage runner

```python
def _show_last_failure():
    state_path = RUNTIME_ROOT / "artifacts" / "colab_pipeline_state.json"
    if not state_path.is_file():
        return
    state = json.loads(state_path.read_text())
    failed = [(key, value) for key, value in state.get("commands", {}).items() if value.get("status") == "failed"]
    if not failed:
        return
    key, record = failed[-1]
    print("Failed command:", record["label"])
    print("Error:", record["error"])
    log_path = RUNTIME_ROOT / "logs" / "colab_pipeline" / f"{key}.log"
    if log_path.is_file():
        print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:]))


def run_stage(stage, *extra):
    environment = os.environ.copy()
    environment["SEMSPECT_RUNTIME_ROOT"] = str(RUNTIME_ROOT)
    environment["MPLCONFIGDIR"] = str(RUNTIME_ROOT / "cache" / "matplotlib")
    command = [
        str(COLAB_PYTHON),
        "scripts/run_colab_pipeline.py",
        "--stage", stage,
        "--runtime-root", str(RUNTIME_ROOT),
        *extra,
    ]
    try:
        subprocess.run(command, cwd=PROJECT, env=environment, check=True)
    except subprocess.CalledProcessError as error:
        print(f"\nStage {stage!r} stopped with exit code {error.returncode}.")
        _show_last_failure()
        print("Full logs:", RUNTIME_ROOT / "logs" / "colab_pipeline")
        print("After resolving the printed error, rerun this same stage; completed commands are retained.")
        raise
```

## 5. Run development once

Read the MIIC CC BY-NC 4.0 license first. This command records that you accept
it, downloads/verifies both datasets, builds all development artifacts, and
stops before the held-out final test.

```python
development_args = ["--install", "--accept-miic-license"]
if MIIC_MANUAL_BUNDLE is not None:
    development_args.extend(["--miic-manual-bundle", str(MIIC_MANUAL_BUNDLE)])
run_stage("development", *development_args)
```

### If the MIIC host drops or blocks the automatic download

The official NTU Dataverse may return an invalid page to automated Colab
requests or drop a long transfer. Do not bypass any verification. In a normal
browser, open the [official MIIC dataset page](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi%3A10.21979%2FN9%2FWBLTFI),
accept its CC BY-NC 4.0 terms, and obtain the five official files. Put them in
one Drive folder and set `MIIC_MANUAL_BUNDLE` in cell 1 to that folder. If the
large `Anomaly_train.rar` was delivered as the two lossless files
`Anomaly_train.rar.part-000` and `Anomaly_train.rar.part-001`, leave both in
the same folder; the launcher streams them together. It verifies every
published size and MD5 before proceeding, so no manual extraction or
reassembly step is required.

## 6. Seal and run the final benchmark

Run this only after the development stage passes. `--initialize-git` creates
the one local source commit required by the sealed protocol; because the source
is in Drive, that commit persists for later post-lock analyses.

```python
run_stage("final", "--install", "--initialize-git")
```

## 7. Run post-lock analyses and refresh the report

```python
run_stage("postlock", "--install")
```

The final output lives here:

```python
print(RUNTIME_ROOT / "runs" / "core_v1_final" / "report.md")
```

If Colab disconnects or a stage stops, remount Drive, rerun cells 1–3, and
rerun the same stage. Completed command boundaries are recorded in
`artifacts/colab_pipeline_state.json` and are skipped safely. The command that
was interrupted is retried; no hidden notebook state is required. Passing
`--install` on every stage makes a fresh Colab runtime recreate its temporary
Python environment without losing persistent work.

## Persistent files

| Location | Contents |
| --- | --- |
| `adaptive-sem-inspection-v0.1.6/` | Source, configuration, launcher, and committed Git metadata |
| `adaptive-sem-inspection-runtime-v0.1.6/data/` | Downloaded archives and extracted datasets |
| `adaptive-sem-inspection-runtime-v0.1.6/cache/` | Features, banks, calibrators, and router checkpoints |
| `adaptive-sem-inspection-runtime-v0.1.6/artifacts/` | Audits, policy thresholds, lock, and pipeline checkpoint state |
| `adaptive-sem-inspection-runtime-v0.1.6/runs/core_v1_final/` | Metrics, plots, report, trajectories, and secondary analyses |

`RUN_ADAPTIVE_SEM_IN_COLAB.ipynb` contains the same cells if you prefer to
upload one notebook rather than create a blank one.
