"""Per-policy endpoint metrics with explicit strict-threshold predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from seminspect.evaluation.thresholds import Threshold, predict_with_threshold


@dataclass(frozen=True)
class BinaryMetrics:
    recall: float
    false_negative_rate: float
    fpr: float
    precision: float
    specificity: float
    auroc: float
    auprc: float
    tp: int
    fp: int
    tn: int
    fn: int


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: Threshold | float) -> BinaryMetrics:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores) or not set(labels).issubset({0, 1}):
        raise ValueError("Metrics need equal-length binary labels and scores")
    prediction = predict_with_threshold(scores, threshold)
    true_positive = int(np.sum((labels == 1) & prediction))
    false_positive = int(np.sum((labels == 0) & prediction))
    true_negative = int(np.sum((labels == 0) & ~prediction))
    false_negative = int(np.sum((labels == 1) & ~prediction))
    positives, negatives = true_positive + false_negative, true_negative + false_positive
    recall = true_positive / positives if positives else float("nan")
    fpr = false_positive / negatives if negatives else float("nan")
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    auroc = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan")
    auprc = float(average_precision_score(labels, scores)) if len(set(labels)) == 2 else float("nan")
    return BinaryMetrics(
        recall=recall,
        false_negative_rate=1.0 - recall if np.isfinite(recall) else float("nan"),
        fpr=fpr,
        precision=precision,
        specificity=1.0 - fpr if np.isfinite(fpr) else float("nan"),
        auroc=auroc,
        auprc=auprc,
        tp=true_positive,
        fp=false_positive,
        tn=true_negative,
        fn=false_negative,
    )


def policy_result_frame(
    labels: Mapping[str, int],
    final_scores: Mapping[str, float],
    final_qualities: Mapping[str, int],
    threshold: Threshold,
    *,
    policy: str,
    budget: float,
    actual_cost: float,
) -> pd.DataFrame:
    """Produce a stable per-image endpoint table for metrics and bootstrap reuse."""

    ids = sorted(labels)
    if set(ids) != set(final_scores) or set(ids) != set(final_qualities):
        raise ValueError("Policy results must have one score and final quality per label ID")
    scores = np.asarray([final_scores[item] for item in ids], dtype=np.float64)
    result = pd.DataFrame(
        {
            "image_id": ids,
            "label": [int(labels[item]) for item in ids],
            "final_score": scores,
            "final_quality": [int(final_qualities[item]) for item in ids],
            "prediction": predict_with_threshold(scores, threshold).astype(np.int8),
            "threshold": threshold.value,
            "policy": policy,
            "budget": float(budget),
            "actual_cost": float(actual_cost),
        }
    )
    return result


def summarize_policy_results(frame: pd.DataFrame) -> dict[str, float | int]:
    required = {"label", "final_score", "threshold", "final_quality", "actual_cost"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing policy result fields: {required - set(frame.columns)}")
    threshold = float(frame["threshold"].iloc[0])
    metrics = binary_metrics(frame["label"].to_numpy(), frame["final_score"].to_numpy(), threshold)
    output: dict[str, float | int] = asdict(metrics)
    output.update(
        {
            "actual_cost": float(frame["actual_cost"].iloc[0]),
            "q0_rate": float((frame["final_quality"] == 0).mean()),
            "q1_rate": float((frame["final_quality"] == 1).mean()),
            "q2_rate": float((frame["final_quality"] == 2).mean()),
            "q3_rate": float((frame["final_quality"] == 3).mean()),
            "mean_upgrades": float(frame["final_quality"].mean()),
        }
    )
    return output


def recall_cost_auc(summary: pd.DataFrame) -> float:
    """Trapezoidal recall-cost area on a shared actual-cost support table."""

    if not {"actual_cost", "recall"}.issubset(summary.columns):
        raise ValueError("Recall-cost AUC needs actual_cost and recall columns")
    ordered = summary.sort_values("actual_cost", kind="mergesort")
    if ordered["actual_cost"].duplicated().any():
        ordered = ordered.groupby("actual_cost", as_index=False, sort=True)["recall"].mean()
    return float(np.trapezoid(ordered["recall"].to_numpy(), ordered["actual_cost"].to_numpy()))
