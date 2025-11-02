"""Visualization engine for matplotlib and seaborn charts."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports" / "visualizations"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def pairplot(df: pd.DataFrame, hue: str | None = None, output_name: str = "pairplot") -> Path:
    """Generate seaborn pairplot."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    plot_df = df[numeric_cols[:5]].copy()  # Limit to 5 cols for performance
    if hue and hue in df.columns:
        plot_df[hue] = df[hue]
    
    g = sns.pairplot(plot_df, hue=hue, diag_kind="kde", corner=True)
    path = REPORTS_DIR / f"{output_name}.png"
    g.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def scatter(df: pd.DataFrame, x: str, y: str, hue: str | None = None, output_name: str = "scatter") -> Path:
    """Generate scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, s=100, alpha=0.7)
    ax.set_title(f"{y} vs {x}", fontsize=14)
    path = REPORTS_DIR / f"{output_name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def heatmap(df: pd.DataFrame, output_name: str = "heatmap") -> Path:
    """Generate correlation heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Correlation Heatmap", fontsize=14)
    path = REPORTS_DIR / f"{output_name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def histogram(df: pd.DataFrame, column: str, bins: int = 30, output_name: str = "histogram") -> Path:
    """Generate histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df[column].dropna(), bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Distribution of {column}", fontsize=14)
    path = REPORTS_DIR / f"{output_name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def bar_chart(df: pd.DataFrame, x: str, y: str, output_name: str = "bar_chart") -> Path:
    """Generate bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x=x, y=y, ax=ax, palette="viridis")
    ax.set_title(f"{y} by {x}", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    path = REPORTS_DIR / f"{output_name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def line_plot(df: pd.DataFrame, x: str, y: str, hue: str | None = None, output_name: str = "line_plot") -> Path:
    """Generate line plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax, marker="o")
    ax.set_title(f"{y} over {x}", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    path = REPORTS_DIR / f"{output_name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def box_plot(df: pd.DataFrame, x: str, y: str, output_name: str = "box_plot") -> Path:
    """Generate box plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x=x, y=y, ax=ax, palette="Set2")
    ax.set_title(f"{y} distribution by {x}", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    path = REPORTS_DIR / f"{output_name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def violin_plot(df: pd.DataFrame, x: str, y: str, output_name: str = "violin_plot") -> Path:
    """Generate violin plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x=x, y=y, ax=ax, palette="muted")
    ax.set_title(f"{y} distribution by {x}", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    path = REPORTS_DIR / f"{output_name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def count_plot(df: pd.DataFrame, column: str, output_name: str = "count_plot") -> Path:
    """Generate count plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x=column, ax=ax, palette="pastel")
    ax.set_title(f"Count of {column}", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    path = REPORTS_DIR / f"{output_name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path
