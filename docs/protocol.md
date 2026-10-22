# core_v1 protocol contract

The source of truth is `configs/experiment/core_v1.yaml` plus the immutable
test lock created by `scripts/freeze_protocol.py`. The primary endpoint is the
paired difference in real-anomaly recall between MLP Value-of-Resolution and
cost-normalized entropy uncertainty at a pixel-information mean cost cap of
`0.20`, with thresholds selected only on policy-calibration normals.

The implementation intentionally separates:

1. **observation**: deterministic Q0–Q3 simulated SEM observations;
2. **detector**: frozen normal-memory anomaly evidence and calibrated score;
3. **router**: a current-state prediction of next-step risk reduction;
4. **allocator**: a label-blind, deterministic cost-constrained sequence; and
5. **evaluation**: labels and masks are used only after a policy has produced
   its final state.

Any changed model, data manifest, score calibrator, configuration, source
revision, or test ID set invalidates the lock and requires a new named protocol
version.

## Gates and post-lock discipline

`scripts/run_detector_feasibility.py` is a hard development gate: the Q3/Q0
resolution opportunity must pass its prespecified checks before
`build_router_dataset.py` will construct supervision. If DINOv2 fails, use the
separately named `core_v1_resnet` configuration or report a negative
feasibility result; never select the fallback using final-test labels.

`run_stress_robustness.py`, `run_secondary_analyses.py`,
`run_ablations.py`, and `render_failure_galleries.py` all require an existing
primary final run and write only exploratory outputs. Stress scenarios use
frozen detector/calibrator/router artifacts; only policy thresholds are
reselected from stressed calibration normals. Detector-side ablations change
the cached observations and therefore run as separately named full protocol
variants, not as a post-hoc overwrite of `core_v1`.
