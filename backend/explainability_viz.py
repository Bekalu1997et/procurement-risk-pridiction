"""Explainability visualizations for SHAP and risk scores."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports" / "explainability"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_feature_importance(
    feature_names: list[str],
    shap_values: list[float],
    output_name: str = "feature_importance"
) -> Path:
    """Plot horizontal bar chart of SHAP feature importance."""
    df = pd.DataFrame({"feature": feature_names, "importance": shap_values})
    df["abs_importance"] = df["importance"].abs()
    df = df.sort_values("abs_importance", ascending=True).tail(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["red" if x < 0 else "green" for x in df["importance"]]
    ax.barh(df["feature"], df["importance"], color=colors, alpha=0.7)
    ax.set_xlabel("SHAP Value (Impact on Risk)", fontsize=12)
    ax.set_title("Top 10 Feature Importance", fontsize=14)
    ax.axvline(0, color="black", linewidth=0.8)
    
    path = REPORTS_DIR / f"{output_name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def plot_risk_score_timeline(
    df: pd.DataFrame,
    date_col: str = "prediction_date",
    risk_col: str = "risk_score",
    supplier_col: str = "supplier_id",
    output_name: str = "risk_timeline"
) -> Path:
    """Plot line graph of risk scores over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if supplier_col in df.columns:
        for supplier in df[supplier_col].unique()[:5]:
            supplier_df = df[df[supplier_col] == supplier]
            ax.plot(supplier_df[date_col], supplier_df[risk_col], marker="o", label=f"Supplier {supplier}")
    else:
        ax.plot(df[date_col], df[risk_col], marker="o", color="blue")
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Risk Score", fontsize=12)
    ax.set_title("Risk Score Timeline", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    
    path = REPORTS_DIR / f"{output_name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def plot_waterfall(
    feature_names: list[str],
    shap_values: list[float],
    base_value: float = 0.5,
    output_name: str = "waterfall"
) -> Path:
    """Plot waterfall chart showing cumulative SHAP contributions."""
    df = pd.DataFrame({"feature": feature_names, "shap": shap_values})
    df = df.sort_values("shap", ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    cumulative = base_value
    positions = []
    
    for i, row in df.iterrows():
        color = "green" if row["shap"] > 0 else "red"
        ax.barh(row["feature"], row["shap"], left=cumulative, color=color, alpha=0.7)
        positions.append(cumulative + row["shap"] / 2)
        cumulative += row["shap"]
    
    ax.set_xlabel("Risk Score Contribution", fontsize=12)
    ax.set_title("Waterfall: Feature Contributions to Risk", fontsize=14)
    ax.axvline(base_value, color="black", linestyle="--", label="Base Value")
    
    path = REPORTS_DIR / f"{output_name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path
