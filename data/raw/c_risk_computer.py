"""Risk computation module for supplier risk prediction system.

This module computes risk scores and categorical labels based on supplier
characteristics and transaction history.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Derive heuristic risk scores and categorical labels."""

    def score_row(row: pd.Series) -> Tuple[float, str]:
        score = 0.0
        if row["avg_payment_delay_days"] > 45:
            score += 2
        if row["past_disputes"] > 5:
            score += 2
        if row["financial_stability_index"] < 0.5:
            score += 2
        if row["late_ratio"] > 0.3:
            score += 1
        if row["dispute_rate"] > 0.2:
            score += 1
        if row["clause_risk_score"] > 60:
            score += 1
        if row["avg_delivery_quality"] < 70:
            score += 1
        risk_score = round(min(1.0, score / 7), 3)
        if score >= 5:
            label = "high"
        elif score >= 3:
            label = "medium"
        else:
            label = "low"
        return risk_score, label

    risks = df.apply(score_row, axis=1, result_type="expand")
    df["risk_score"] = risks[0]
    df["risk_label"] = risks[1].astype(str)
    return df
