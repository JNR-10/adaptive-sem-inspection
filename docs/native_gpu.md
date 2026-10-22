# Native GPU runbook

This is the supported path for running the project on your own NVIDIA GPU. It
uses the same staged protocol as the completed Colab run, but stores data,
caches, logs, checkpoints, and reports in a local runtime directory instead
of Google Drive.

## What you need

- Linux x86_64 with an NVIDIA GPU and a driver compatible with CUDA 12.6;
- Python **3.11 or 3.12**;
- enough local SSD space for MIIC, Carinthia-S, extracted images, feature
  caches, and output artifacts; use a path outside the Git checkout;
- `7z` / `p7zip-full`, because the official MIIC files are RAR archives;
- Git, because the final stage deliberately seals a committed source revision.

For Ubuntu/Debian, install the archive tool with:

```bash
sudo apt-get update
sudo apt-get install -y p7zip-full git
```

Windows users should run the project in WSL2 with NVIDIA CUDA support. macOS
is suitable for CPU tests but not for the locked CUDA detector path.

## 1. Create the environment

```bash
git clone <your-fork-url> adaptive-sem-inspection
cd adaptive-sem-inspection

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/gpu-cu126.lock
python -m pip install -e .

python scripts/verify_environment.py --require-gpu
```

`requirements/gpu-cu126.lock` selects the pinned CUDA 12.6 PyTorch wheels used
by the reference run. If your driver cannot support them, do not silently swap
the environment and call it a reproduction. Record a new environment and treat
the result as a separate run.

## 2. Choose a runtime location

Keep runtime outputs out of Git. A fast local SSD is much better than a network
home directory for feature extraction and parquet reads.

```bash
export SEMSPECT_RUNTIME_ROOT=/fast-ssd/adaptive-sem-runtime
mkdir -p "$SEMSPECT_RUNTIME_ROOT"
```

The launcher writes these directories below that root:

```text
data/raw/          Original, checksum-verified archives
data/extracted/    Decoded images and masks
data/manifests/    Audit and split manifests
cache/             DINOv2 features, memory banks, router state
artifacts/         Calibrators, thresholds, locks, resumable state
runs/              Final reports, figures, diagnostics, secondary results
logs/pipeline/     One log per restart-safe command boundary
```

## 3. Acquire the data legally

MIIC is **CC BY-NC 4.0**. Read its terms and use it only for permitted
non-commercial research. The project never bundles the raw data.

The normal path is:

```bash
python scripts/run_native_pipeline.py \
  --stage development \
  --runtime-root "$SEMSPECT_RUNTIME_ROOT" \
  --accept-miic-license
```

The official NTU host can be slow or disconnect on long downloads. In that
case, download the five files from the official MIIC page with a browser, keep
their official names, and pass the directory to the launcher:

```bash
python scripts/run_native_pipeline.py \
  --stage development \
  --runtime-root "$SEMSPECT_RUNTIME_ROOT" \
  --accept-miic-license \
  --miic-manual-bundle /path/to/MIIC_verified_files
```

The manual bundle may include the lossless browser shards
`Anomaly_train.rar.part-000` and `Anomaly_train.rar.part-001`. Leave both in
the same folder. The staging step streams them together and verifies the
declared bytes and MD5 before extraction; do not rename, concatenate, or
extract them by hand.

## 4. Run the staged protocol

### Development

Development builds data manifests, leakage-safe splits, Q0–Q3 features, memory
banks, calibrators, router training data, routers, and policy thresholds. It
stops before the final test.

```bash
python scripts/run_native_pipeline.py \
  --stage development \
  --runtime-root "$SEMSPECT_RUNTIME_ROOT" \
  --accept-miic-license \
  --miic-manual-bundle /path/to/MIIC_verified_files
```

### Final

Review development output first. Then commit the source/configuration you are
about to lock. The final stage refuses to run against an uncommitted revision
unless you explicitly ask it to initialize a local commit.

```bash
git add -A
git commit -m "Lock core_v1 native GPU protocol"

python scripts/run_native_pipeline.py \
  --stage final \
  --runtime-root "$SEMSPECT_RUNTIME_ROOT"
```

### Post-lock analysis

These are exploratory and must not replace the locked primary result.

```bash
python scripts/run_native_pipeline.py \
  --stage postlock \
  --runtime-root "$SEMSPECT_RUNTIME_ROOT"
```

## 5. Resume and inspect progress

Every command boundary has a durable state record. If the machine restarts or
a process is interrupted, rerun the same stage. Completed commands are reused.

```bash
python scripts/run_native_pipeline.py \
  --stage status \
  --runtime-root "$SEMSPECT_RUNTIME_ROOT"

tail -f "$SEMSPECT_RUNTIME_ROOT/logs/pipeline"/*.log
```

Use `--force` only on exploratory scripts where it is documented. The locked
final artifacts are intentionally immutable; change the protocol version and
run a new experiment instead of overwriting them.

## Expected workload

The reference T4 run took roughly 40 minutes for development and 20 minutes
for the final benchmark, excluding data acquisition and later post-lock
analyses. Native performance depends on GPU, storage, CPU, and DINOv2 download
cache. Treat those times as orientation, not a service-level estimate.

## Colab remains available

The earlier Colab launcher is retained for provenance in
`scripts/run_colab_pipeline.py` and `COLAB_QUICKSTART.md`. For a local GPU,
use `scripts/run_native_pipeline.py`; it sets a headless plotting backend,
uses a local runtime path, and writes native pipeline state/log locations.
