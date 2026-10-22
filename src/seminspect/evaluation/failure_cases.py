"""Deterministic post-hoc diagnostic selections; none are extra endpoints."""

from __future__ import annotations

from typing import Callable

import pandas as pd


def _select(frame: pd.DataFrame, query: str, sort_columns: list[str], ascending: list[bool], cap: int = 20) -> pd.DataFrame:
    selected = frame.query(query, engine="python").copy()
    for column in sort_columns:
        if column not in selected.columns:
            raise ValueError(f"Failure gallery cannot rank absent column {column!r}")
    return selected.sort_values(sort_columns + ["image_id"], ascending=ascending + [True], kind="mergesort").head(cap)


def select_failure_cases(frame: pd.DataFrame, cap: int = 20) -> dict[str, pd.DataFrame]:
    """Return protocol-defined galleries from a sufficiently rich result table."""

    output = {
        "false_negatives": _select(frame, "label == 1 and prediction == 0", ["score_margin"], [True], cap),
        "false_positives": _select(frame, "label == 0 and prediction == 1", ["score_margin"], [False], cap),
        "corrected_low_resolution_misses": _select(frame, "label == 1 and q0_prediction == 0 and prediction == 1", ["q3_minus_q0_probability"], [False], cap),
        "high_uncertainty_no_help": _select(frame, "realized_gain <= 0", ["q0_entropy", "realized_gain"], [False, True], cap),
        "harmful_upgrades": _select(frame, "realized_gain < 0", ["realized_gain"], [True], cap),
        "stopped_but_correctable": _select(frame, "stopped_with_positive_unacquired_gain == 1", ["unacquired_true_gain"], [False], cap),
        "q3_no_decision_change": _select(frame, "final_quality == 3", ["absolute_q3_q0_probability_change"], [True], cap),
        "policy_disagreements": _select(frame, "vor_uncertainty_quality_difference == vor_uncertainty_quality_difference", ["vor_uncertainty_quality_difference"], [False], cap),
    }
    return output
