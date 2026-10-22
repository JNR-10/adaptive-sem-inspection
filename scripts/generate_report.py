#!/usr/bin/env python3
"""Generate a data-derived Markdown handoff report and core cost curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from seminspect.evaluation.plots import (
    plot_metric_by_category,
    plot_metric_by_cost,
    plot_quality_distribution,
    plot_reliability,
    plot_scatter_relationship,
)
from seminspect.utils.io import ArtifactError, atomic_write_text, read_json
from seminspect.utils.tabular import read_parquet


def _metric_row(metrics: pd.DataFrame, policy: str, cost_model: str, budget: float, seed=None) -> pd.Series:
    selected = metrics.loc[
        (metrics["policy"] == policy)
        & (metrics["cost_model"] == cost_model)
        & (metrics["budget"] == budget)
        & ((metrics["seed"].isna()) if seed is None else (metrics["seed"] == seed))
    ]
    if len(selected) != 1:
        raise ArtifactError(f"Could not locate unique report row for {policy}/{cost_model}/{budget}")
    return selected.iloc[0]


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = frame.columns.tolist()
    rows = frame.astype(str).values.tolist()
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="Replace a generated report/figures without changing raw results")
    args = parser.parse_args()
    run_dir = args.run_dir
    metrics_path = run_dir / "metrics.parquet"
    primary_path = run_dir / "primary_endpoint.json"
    if not metrics_path.is_file() or not primary_path.is_file():
        raise ArtifactError("Run directory is missing locked metrics or primary endpoint")
    metrics = read_parquet(metrics_path)
    primary = read_json(primary_path)
    figures = run_dir / "figures"
    stable_seed = metrics["seed"].isna() | metrics["seed"].eq("mean_30_seeds")
    primary_metrics = metrics.loc[(metrics["cost_model"] == "pixel_information") & stable_seed].copy()
    for metric in ("recall", "fpr", "precision", "auroc", "auprc"):
        plot_metric_by_cost(
            primary_metrics,
            metric,
            figures / f"{metric}_by_cost.png",
        )
    quality_input = primary_metrics.loc[
        (primary_metrics["budget"] == 0.20)
        & primary_metrics["policy"].isin(
            ["survey_only", "anomaly_score", "uncertainty_entropy", "vor_ridge", "vor_mlp", "oracle_gain", "full_q3", "random"]
        )
    ]
    if not quality_input.empty:
        plot_quality_distribution(quality_input, figures / "quality_distribution_at_0_20.png")
    sensitivity = metrics.loc[
        stable_seed & metrics["policy"].isin(["uncertainty_entropy", "vor_mlp"])
    ].copy()
    if not sensitivity.empty:
        sensitivity["policy"] = sensitivity["policy"] + " / " + sensitivity["cost_model"]
        plot_metric_by_cost(sensitivity, "recall", figures / "cost_model_sensitivity.png", "Recall by cost-model sensitivity")
    reliability_path = run_dir / "calibration_reliability.parquet"
    if reliability_path.is_file():
        reliability = read_parquet(reliability_path)
        validation = reliability.loc[reliability["split"].eq("router_validation")]
        if not validation.empty:
            plot_reliability(validation, figures / "calibration_reliability.png")
    secondary = run_dir / "secondary"
    defect_size_path = secondary / "defect_size_summary.parquet"
    if defect_size_path.is_file():
        defect_size = read_parquet(defect_size_path)
        plot_metric_by_category(
            defect_size,
            category="size_quartile",
            metric="recall",
            series="policy",
            destination=figures / "recall_by_defect_size.png",
            title="Exploratory recall by real-anomaly size quartile",
        )
    stress_path = secondary / "stress_results.parquet"
    if stress_path.is_file():
        stress = read_parquet(stress_path)
        stress = stress.loc[
            stress["threshold_mode"].eq("scenario_controlled_fpr")
            & stress["cost_model"].eq("pixel_information")
            & (stress["budget"] == 0.20)
            & stress["seed"].isna()
            & stress["policy"].isin(["uncertainty_entropy", "vor_mlp"])
        ]
        if not stress.empty:
            plot_metric_by_category(
                stress,
                category="scenario",
                metric="recall",
                series="policy",
                destination=figures / "stress_robustness_recall.png",
                title="Exploratory stress robustness at controlled FPR",
            )
    diagnostic_path = run_dir / "failure_cases" / "vor_mlp" / "diagnostic_frame.parquet"
    if diagnostic_path.is_file():
        diagnostic = read_parquet(diagnostic_path)
        plot_scatter_relationship(
            diagnostic,
            x="predicted_q0_gain",
            y="realized_gain",
            destination=figures / "predicted_vs_realized_gain.png",
            color="label",
        )
        plot_scatter_relationship(
            diagnostic,
            x="q0_entropy",
            y="realized_gain",
            destination=figures / "uncertainty_vs_realized_gain.png",
            color="label",
        )
    mlp = _metric_row(metrics, "vor_mlp", "pixel_information", 0.20)
    uncertainty = _metric_row(metrics, "uncertainty_entropy", "pixel_information", 0.20)
    primary_text = (
        f"At a 0.20 normalized cost cap, VoR changed recall relative to cost-normalized uncertainty by "
        f"{100 * primary['estimate']:.2f} percentage points (paired 95% CI "
        f"[{100 * primary['ci95'][0]:.2f}, {100 * primary['ci95'][1]:.2f}]); observed FPRs were "
        f"{100 * mlp['fpr']:.2f}% and {100 * uncertainty['fpr']:.2f}%."
    )
    table = metrics.loc[
        (metrics["cost_model"] == "pixel_information")
        & (metrics["budget"] == 0.20)
        & (~metrics["seed"].notna())
        & (metrics["policy"].isin(["survey_only", "anomaly_score", "uncertainty_entropy", "vor_ridge", "vor_mlp", "oracle_gain"]))
    ][["policy", "actual_cost", "recall", "fpr", "precision", "q0_rate", "q1_rate", "q2_rate", "q3_rate"]].copy()
    table["actual_cost"] = table["actual_cost"].map(lambda value: f"{value:.6f}")
    for column in ["recall", "fpr", "precision", "q0_rate", "q1_rate", "q2_rate", "q3_rate"]:
        table[column] = table[column].map(lambda value: f"{100 * value:.2f}%")
    markdown = "\n".join(
        [
            "# core_v1 final benchmark report",
            "",
            "## Scope",
            "",
            "This is an offline simulated-acquisition result on public SEM image regions. Normalized acquisition cost does not represent SEM scan time, electron dose, wafer throughput, yield, or commercial-tool savings.",
            "",
            "## Primary endpoint",
            "",
            primary_text,
            "",
            "## Pixel-information schedule at cap 0.20",
            "",
            _markdown_table(table),
            "",
            "`oracle_gain` is a privileged, non-deployable diagnostic and is not a deployment baseline.",
            "",
            "## Generated figures",
            "",
            "- `figures/recall_by_cost.png`",
            "- `figures/fpr_by_cost.png`",
            "- `figures/precision_by_cost.png`",
            "- `figures/auroc_by_cost.png`",
            "- `figures/auprc_by_cost.png`",
            "",
            "## Interpretation discipline",
            "",
            "The synthetic-pair router target may not transfer to real MIIC anomalies. Treat the primary interval and final-test FPR together; a recall increase with materially worse FPR is a calibration trade-off, not an unqualified win. Secondary cost schedules, stress scenarios, ablations, and Carinthia-S results must remain labeled exploratory.",
        ]
    )
    atomic_write_text(run_dir / "report.md", markdown, overwrite=args.force)
    print(f"Wrote report and figures to {run_dir}")


if __name__ == "__main__":
    main()
