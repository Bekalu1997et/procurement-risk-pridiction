"""Visualization helpers for notebooks, dashboards, and reports."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"


def create_pairplot(df: pd.DataFrame, output_name: str) -> Path:
    """Generate a seaborn pairplot for key numeric features."""

    sns.set_theme(style="whitegrid")
    numeric_cols = [
        "annual_spend",
        "credit_score",
        "late_ratio",
        "dispute_rate",
        "avg_delay",
    ]

    pairplot = sns.pairplot(df[numeric_cols + ["risk_label"]], hue="risk_label")
    output_path = REPORTS_DIR / f"{output_name}_pairplot.png"
    pairplot.savefig(output_path)
    plt.close(pairplot.fig)
    return output_path


def plot_shap_summary(shap_values: Iterable[float], feature_names: Iterable[str], output_name: str) -> Path:
    """Create a horizontal bar chart summarising SHAP contributions."""

    shap_series = pd.Series(shap_values, index=feature_names).sort_values()
    plt.figure(figsize=(8, 5))
    shap_series.plot(kind="barh", color=["#f0ad4e" if val > 0 else "#5bc0de" for val in shap_series])
    plt.title("Top Feature Contributions")
    plt.xlabel("SHAP Value")
    output_path = REPORTS_DIR / "shap_summary_plots" / f"{output_name}_shap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def create_weekly_plots(predictions: pd.DataFrame, run_id: str) -> None:
    """Produce charts for the weekly scoring report."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    label_counts = predictions["prediction"].value_counts()
    plt.figure(figsize=(6, 4))
    label_counts.plot(kind="bar", color="#007bff")
    plt.title("Prediction Distribution")
    plt.ylabel("Count")
    plt.xlabel("Risk Label")
    plt.tight_layout()
    output = REPORTS_DIR / "weekly_reports" / f"{run_id}_distribution.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close()

    plt.figure(figsize=(6, 4))
    sorted_predictions = predictions.sort_values("prob_high").reset_index(drop=True)
    sns.lineplot(data=sorted_predictions, x=sorted_predictions.index, y="prob_high")
    plt.title("High Risk Probability Trend")
    plt.ylabel("Probability")
    plt.xlabel("Observation")
    plt.tight_layout()
    trend_output = REPORTS_DIR / "weekly_reports" / f"{run_id}_trend.png"
    plt.savefig(trend_output)
    plt.close()

