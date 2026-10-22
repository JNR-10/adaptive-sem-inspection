# Colab notebooks

These notebooks are deliberately thin, documented launchers. All logic lives in
`src/seminspect` and every command may be rerun from a shell, so no final
experiment depends on hidden notebook state.

For Google Colab, use the root-level `RUN_ADAPTIVE_SEM_IN_COLAB.ipynb` or the
blank-notebook instructions in `../COLAB_QUICKSTART.md`. The persistent
`scripts/run_colab_pipeline.py` launcher runs the stages sequentially and
records resumable checkpoints in Drive. You do **not** need to open this folder
one notebook at a time.

| Notebook | Purpose |
| --- | --- |
| `00_colab_bootstrap.ipynb` | Mount Drive, stage the repository on local disk, install pins, record runtime. |
| `01_data_audit.ipynb` | Download, verify, extract, manifest, and audit fixed splits. |
| `02_quality_ladder.ipynb` | Build Q0–Q3 normal embeddings and inspect the deterministic ladder. |
| `03_detector.ipynb` | Build memory banks, detector features, and score calibrators. |
| `04_router.ipynb` | Build target/features and train Ridge/MLP routers. |
| `05_policy_validation.ipynb` | Calibrate policy thresholds and freeze a committed protocol. |
| `06_final_benchmark.ipynb` | Unlock final features once, run the sealed benchmark, and report it. |
| `07_failure_analysis.ipynb` | Run secondary analyses and generate the final report. |

Open each notebook only after its stated gate passes. Do not inspect final-test
metrics while selecting a detector, simulator, or router.
