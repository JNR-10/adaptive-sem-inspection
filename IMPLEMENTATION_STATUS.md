# Implementation status

This repository is a reproducible **offline simulated-acquisition benchmark**
for multi-resolution SEM image inspection. It does not claim physical SEM
control, scan time, electron dose, wafer throughput, yield, or fab deployment
performance.

## Included

- checksum-verified MIIC and Carinthia-S ingestion, canonical manifests,
  overlap removal, duplicate audit, and strict Carinthia mask handling;
- deterministic Q0–Q3 resolution simulation and predeclared stress scenarios;
- frozen DINOv2 patch-memory detector, quality-specific normal banks,
  calibration, and an isolated ResNet-18 feasibility fallback;
- 53-feature Ridge/MLP Value-of-Resolution routers and a deterministic
  allocator with survey, full-Q3, random, score, uncertainty, instability, and
  oracle baselines;
- policy-specific threshold calibration, final-test locking, paired bootstrap,
  secondary analyses, stress robustness, ablations, and failure galleries;
- a resumable native-GPU launcher, unit/contract tests, and GitHub Actions CPU
  test workflow;
- a small, auditable snapshot of the completed sealed final run in `results/`.

## Completed reference run

The included locked result shows that VoR MLP did not outperform the
cost-normalized entropy-uncertainty baseline at the predeclared primary cap.
That result is preserved rather than edited away. See `README.md`,
`docs/results.md`, and `results/SEALED_RUN_REPORT.md` for the exact numbers and
interpretation limits.

## What remains runtime-dependent

The licensed data, DINOv2 weights, feature extraction, training, and new final
locks must be generated locally by the user. Follow `docs/native_gpu.md` and
commit the source/configuration before sealing a final run. A changed data
manifest, model, configuration, source revision, or dependency lock is a new
protocol version—not an overwrite of the included result.

The earlier Drive-persistent Colab path is retained in the repository for
provenance, but it is no longer the recommended entry point for native GPU
users.
