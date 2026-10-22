# core_v1 final benchmark report

## Scope

This is an offline simulated-acquisition result on public SEM image regions. Normalized acquisition cost does not represent SEM scan time, electron dose, wafer throughput, yield, or commercial-tool savings.

## Primary endpoint

At a 0.20 normalized cost cap, VoR changed recall relative to cost-normalized uncertainty by -1.72 percentage points (paired 95% CI [-5.17, 0.88]); observed FPRs were 0.86% and 0.71%.

## Pixel-information schedule at cap 0.20

| policy | actual_cost | recall | fpr | precision | q0_rate | q1_rate | q2_rate | q3_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| survey_only | 0.015625 | 64.66% | 0.79% | 88.24% | 100.00% | 0.00% | 0.00% | 0.00% |
| anomaly_score | 0.199883 | 82.76% | 0.71% | 91.43% | 0.00% | 51.22% | 42.65% | 6.12% |
| uncertainty_entropy | 0.199984 | 85.34% | 0.71% | 91.67% | 1.80% | 24.42% | 73.78% | 0.00% |
| vor_ridge | 0.199984 | 83.62% | 1.42% | 84.35% | 13.04% | 37.18% | 43.08% | 6.70% |
| vor_mlp | 0.199984 | 83.62% | 0.86% | 89.81% | 11.60% | 25.14% | 60.01% | 3.24% |
| oracle_gain | 0.199984 | 87.07% | 1.18% | 87.07% | 16.21% | 13.33% | 68.73% | 1.73% |

`oracle_gain` is a privileged, non-deployable diagnostic and is not a deployment baseline.

## Generated figures

- `figures/recall_by_cost.png`
- `figures/fpr_by_cost.png`
- `figures/precision_by_cost.png`
- `figures/auroc_by_cost.png`
- `figures/auprc_by_cost.png`

## Interpretation discipline

The synthetic-pair router target may not transfer to real MIIC anomalies. Treat the primary interval and final-test FPR together; a recall increase with materially worse FPR is a calibration trade-off, not an unqualified win. Secondary cost schedules, stress scenarios, ablations, and Carinthia-S results must remain labeled exploratory.