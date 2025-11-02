"""Visualization helpers for notebooks, dashboards, and reports."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import base64
import io
import uuid

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


def plot_top_feature_trends_from_history(
    payload: Dict[str, object],
    top_feature_names: List[str],
    supplier_id: Optional[int] = None,
    history_path: Optional[Path] = None,
) -> List[str]:
    """Return base64-encoded PNGs showing historical trends for each top feature.

    The function will try a few sensible historical datasets in order:
    - data/processed/new_data_weekly.csv
    - data/processed/merged_training.csv

    If a `supplier_id` column exists, it will filter rows for that supplier; if not,
    it will attempt to match on `region` and `industry` from the payload. If no
    historical data can be found, an empty list is returned.
    """

    # locate history file
    base = Path(__file__).resolve().parents[1]
    candidates = []
    if history_path:
        candidates.append(Path(history_path))
    candidates.extend(
        [
            base / "data" / "processed" / "new_data_weekly.csv",
            base / "data" / "processed" / "merged_training.csv",
        ]
    )

    hist_df = None
    for p in candidates:
        try:
            if p.exists():
                hist_df = pd.read_csv(p)
                break
        except Exception:
            continue

    if hist_df is None or hist_df.empty:
        return []

    # Filter to supplier-level series if possible
    selector = None
    if supplier_id is not None and "supplier_id" in hist_df.columns:
        selector = hist_df[hist_df["supplier_id"] == supplier_id]
    elif "region" in payload and "industry" in payload and {
        "region",
        "industry",
    }.issubset(hist_df.columns):
        selector = hist_df[
            (hist_df["region"] == payload.get("region"))
            & (hist_df["industry"] == payload.get("industry"))
        ]
    else:
        # fallback to whole dataset
        selector = hist_df

    if selector.empty:
        return []

    # try to find a time column for plotting order
    time_col = None
    for candidate in ("date", "week", "timestamp", "created_at"):
        if candidate in selector.columns:
            time_col = candidate
            break

    if time_col is not None:
        try:
            selector[time_col] = pd.to_datetime(selector[time_col], errors="coerce")
            selector = selector.sort_values(time_col)
            x_values = selector[time_col]
        except Exception:
            x_values = selector.index
    else:
        x_values = selector.index

    images_base64: List[str] = []
    for feat in top_feature_names:
        if feat not in selector.columns:
            continue

        plt.figure(figsize=(6, 3))
        sns.lineplot(x=x_values, y=selector[feat])
        plt.title(f"Historical trend — {feat}")
        plt.xlabel("Time")
        plt.ylabel(feat)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        images_base64.append(f"data:image/png;base64,{img_b64}")

    return images_base64

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

