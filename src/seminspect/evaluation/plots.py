"""Small reproducible plotting helpers for required cost curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Reports are generated from CLI subprocesses as well as notebooks.  A
# headless backend keeps that path stable when an interactive backend happens
# to be inherited from a notebook session.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metric_by_cost(summary: pd.DataFrame, metric: str, destination: str | Path, title: str | None = None) -> Path:
    required = {"policy", "actual_cost", metric}
    if not required.issubset(summary.columns):
        raise ValueError(f"Missing plot columns: {required - set(summary.columns)}")
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for policy, group in summary.groupby("policy", sort=True):
        ordered = group.sort_values("actual_cost", kind="mergesort")
        axis.plot(ordered["actual_cost"], ordered[metric], marker="o", label=policy)
    axis.set_xlabel("Actual normalized acquisition cost")
    axis.set_ylabel(metric.replace("_", " ").upper())
    axis.set_title(title or f"{metric.replace('_', ' ').title()} by acquisition cost")
    axis.set_xlim(left=0)
    axis.legend(fontsize=8)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_quality_distribution(summary: pd.DataFrame, destination: str | Path, title: str | None = None) -> Path:
    """Render Q0–Q3 final-state shares for an already-filtered result table."""

    quality_columns = ["q0_rate", "q1_rate", "q2_rate", "q3_rate"]
    required = {"policy", "budget", *quality_columns}
    if not required.issubset(summary.columns):
        raise ValueError(f"Missing quality-distribution columns: {required - set(summary.columns)}")
    ordered = summary.sort_values(["budget", "policy"], kind="mergesort").reset_index(drop=True)
    labels = [f"{row.policy}\nB={row.budget:g}" for row in ordered.itertuples(index=False)]
    colors = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072"]
    figure, axis = plt.subplots(figsize=(max(8, len(ordered) * 0.85), 5), constrained_layout=True)
    bottom = np.zeros(len(ordered), dtype=float)
    for column, color, quality in zip(quality_columns, colors, ["Q0", "Q1", "Q2", "Q3"], strict=True):
        values = ordered[column].to_numpy(dtype=float)
        axis.bar(range(len(ordered)), values, bottom=bottom, color=color, label=quality, width=0.8)
        bottom += values
    axis.set_xticks(range(len(ordered)), labels, rotation=0, fontsize=8)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Fraction of regions")
    axis.set_title(title or "Final-quality distribution")
    axis.legend(title="Final quality", ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_metric_by_category(
    frame: pd.DataFrame,
    *,
    category: str,
    metric: str,
    destination: str | Path,
    series: str | None = None,
    title: str | None = None,
) -> Path:
    """Plot a small categorical comparison, optionally with one line per series."""

    required = {category, metric}
    if series is not None:
        required.add(series)
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing categorical-plot columns: {required - set(frame.columns)}")
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if series is None:
        values = frame.sort_values(category, kind="mergesort")
        axis.bar(values[category].astype(str), values[metric].to_numpy(dtype=float))
    else:
        for name, group in frame.groupby(series, sort=True):
            ordered = group.sort_values(category, kind="mergesort")
            axis.plot(ordered[category].astype(str), ordered[metric], marker="o", label=str(name))
        axis.legend(fontsize=8)
    axis.set_xlabel(category.replace("_", " ").title())
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_title(title or f"{metric.replace('_', ' ').title()} by {category.replace('_', ' ')}")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_scatter_relationship(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    destination: str | Path,
    color: str | None = None,
    title: str | None = None,
) -> Path:
    """Draw a finite-value scatter relationship for diagnostic quantities."""

    required = {x, y}
    if color is not None:
        required.add(color)
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing scatter-plot columns: {required - set(frame.columns)}")
    values = frame.loc[np.isfinite(frame[x].to_numpy(dtype=float)) & np.isfinite(frame[y].to_numpy(dtype=float))].copy()
    figure, axis = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    if color is None:
        axis.scatter(values[x], values[y], s=12, alpha=0.65)
    else:
        for name, group in values.groupby(color, sort=True):
            axis.scatter(group[x], group[y], s=12, alpha=0.65, label=str(name))
        axis.legend(fontsize=8)
    axis.axhline(0, color="0.6", linewidth=0.8)
    axis.set_xlabel(x.replace("_", " ").title())
    axis.set_ylabel(y.replace("_", " ").title())
    axis.set_title(title or f"{y.replace('_', ' ').title()} versus {x.replace('_', ' ').title()}")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_reliability(reliability: pd.DataFrame, destination: str | Path, title: str | None = None) -> Path:
    """Plot fixed-bin calibration reliability curves by quality level."""

    required = {"quality_id", "mean_probability", "empirical_rate", "count"}
    if not required.issubset(reliability.columns):
        raise ValueError(f"Missing reliability columns: {required - set(reliability.columns)}")
    figure, axis = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    axis.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1, label="perfect")
    for quality, group in reliability.groupby("quality_id", sort=True):
        nonempty = group.loc[group["count"].astype(int) > 0]
        axis.plot(nonempty["mean_probability"], nonempty["empirical_rate"], marker="o", label=str(quality))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean calibrated probability")
    axis.set_ylabel("Empirical anomaly rate")
    axis.set_title(title or "Calibration reliability by quality")
    axis.legend(fontsize=8)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output
