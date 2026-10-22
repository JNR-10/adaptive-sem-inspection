# Results: how to read the completed run

The versioned files in `results/` are a snapshot of the completed sealed
`core_v1` benchmark. They are included so a reviewer can inspect the evidence
without downloading the licensed data. They are not a substitute for re-running
the protocol on a compatible native GPU environment.

## Primary endpoint

The predeclared endpoint is the paired difference in **real-anomaly recall**
between the MLP Value-of-Resolution router and the cost-normalized entropy
uncertainty policy at the pixel-information cap of 0.20.

| Metric | Locked outcome |
| --- | --- |
| VoR MLP recall | 83.62% |
| Uncertainty-entropy recall | 85.34% |
| Difference (VoR − uncertainty) | −1.72 percentage points |
| Paired bootstrap 95% CI | [−5.17, +0.88] percentage points |
| VoR MLP FPR | 0.86% |
| Uncertainty-entropy FPR | 0.71% |
| Bootstrap replicates | 10,000 |

The interval includes zero. The appropriate conclusion is therefore that the
experiment **did not establish a learned-routing improvement** over the
uncertainty baseline at the locked endpoint. It should not be reframed as a
throughput or yield result.

![Recall curve](../results/figures/recall_by_cost.png)

## What still worked

At the same cost schedule, survey-only recall was 64.66%. Both uncertainty and
VoR policies improved substantially as they allocated selected regions to
higher-quality observations. The non-deployable oracle diagnostic reached
87.07% recall, suggesting that the observation ladder contains some useful
remaining information. The gap is in estimating that value from the router's
currently available state—not in claiming that every region needs maximum
quality.

## Interpretation boundaries

- The router was trained on synthetic paired supervision but assessed on real
  MIIC anomalies. That transfer is the central limitation.
- A result with higher recall but materially higher FPR is a calibration
  trade-off, not an automatic win.
- Carinthia-S, stress, defect-size, prevalence, and ablation outputs are
  exploratory post-lock analyses. They cannot replace the primary endpoint.
- `oracle_gain` uses privileged future/label-derived information and is only a
  diagnostic upper bound.

See `results/SEALED_RUN_REPORT.md` for the exact reported table and
`results/primary_endpoint.json` for the machine-readable primary endpoint.
