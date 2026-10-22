# Reproducibility notes

## What is versioned

The repository versions source code, configuration, unit tests, a CUDA 12.6
dependency list, selected non-sensitive result artifacts, and the exact final
test-lock snapshot from the completed run. The lock captures hashes for the
resolved configuration, data manifests, duplicate decisions, memory banks,
encoder weights, calibrators, router checkpoint, threshold set, final-test IDs,
and source revision.

## What is intentionally excluded

Raw MIIC archives/images, Carinthia-S archives/images, DINOv2 weights, cached
feature tables, memory banks, trained checkpoints, logs, and future runtime
outputs are not source-controlled. They are large, often license-controlled,
and must be recreated under the user's own permitted environment.

## Re-running the protocol

1. Use the native setup in `docs/native_gpu.md`.
2. Run `development`; inspect the leakage audit and detector feasibility gate.
3. Commit your code/configuration, then run `final` exactly once for that
   protocol version.
4. Treat a changed source/config/data manifest/model as a new protocol version
   rather than an overwrite of the previous final test.
5. Run `postlock` only after final completion, and label those analyses
   exploratory.

Runtime state is checkpointed in `artifacts/pipeline_state.json`; logs live in
`logs/pipeline/`. Re-running an interrupted stage resumes completed command
boundaries. The post-lock router-ablation script is additionally safe to resume
after a checkpoint-only interruption.

## Determinism boundary

The code fixes seeds and configures deterministic GPU behavior where supported,
but exact bit-level equality across NVIDIA drivers, CUDA builds, PyTorch builds,
and CPU libraries is not guaranteed. Record `scripts/verify_environment.py`
output with every new run. If the environment differs from the supplied lock,
report the difference rather than calling it the original sealed result.
