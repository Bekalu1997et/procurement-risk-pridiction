"""Data ingestion and preprocessing routines for the supplier risk system.

This module demonstrates a production-inspired pandas pipeline that loads raw
and processed artifacts, performs consolidated cleaning, and records auditing
metadata.  The functions here are intentionally granular so that the Jupyter
notebooks, API layer, and MLOps simulation can reuse them without duplication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

import auditing


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_processed_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the canonical processed datasets produced by ``demo_pipeline``.
    The function loads the training and weekly scoring datasets from the processed directory.
    It also records the lineage of the datasets using the auditing module.
    """

    training_path = DATA_PROCESSED_DIR / "merged_training.csv"
    weekly_path = DATA_PROCESSED_DIR / "new_data_weekly.csv"
    if not training_path.exists() or not weekly_path.exists():
        raise FileNotFoundError(
            "Processed datasets missing. Run `python data/raw/demo_pipeline.py` first."
        )

    training_df = pd.read_csv(training_path)
    weekly_df = pd.read_csv(weekly_path)

    auditing.record_lineage(
        table_name="processed_training",
        source_files=[str(training_path)]
    )
    auditing.record_lineage(
        table_name="weekly_scoring",
        source_files=[str(weekly_path)]
    )

    return training_df, weekly_df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply canonical cleaning steps: strip whitespace and harmonise names.
    The function cleans the columns of the dataframe by stripping whitespace and harmonising the names.
    It also logs the data quality of the dataframe using the auditing module.
    """

    cleaned = df.copy()
    cleaned.columns = [column.strip().lower() for column in cleaned.columns]

    string_cols = cleaned.select_dtypes(include="object").columns
    for column in string_cols:
        cleaned[column] = cleaned[column].astype(str).str.strip()

    auditing.log_data_quality(cleaned)
    return cleaned


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal helper columns used by dashboards and the MLOps loop.
    The function adds temporal helper columns to the dataframe used by dashboards and the MLOps loop.
    """

    enriched = df.copy()
    if "created_ts" in enriched.columns:
        enriched["created_ts"] = pd.to_datetime(enriched["created_ts"], errors="coerce")
        enriched["created_date"] = enriched["created_ts"].dt.date
    if "snapshot_week" in enriched.columns:
        enriched["snapshot_week"] = enriched["snapshot_week"].fillna(0).astype(int)
    return enriched


def prepare_training_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Return model-ready feature matrix ``X`` and label vector ``y``.
    The function loads the training and weekly scoring datasets, cleans the columns,
    engineers the temporal features, and prepares the training data for the model.
    """

    training_df, _ = load_processed_datasets()
    cleaned = engineer_temporal_features(clean_columns(training_df))

    feature_columns = [
        "region",
        "industry",
        "contract_criticality",
        "annual_spend",
        "credit_score",
        "late_ratio",
        "dispute_rate",
        "avg_delay",
        "clause_risk_score",
    ]
    auditing.persist_audit_log(
        event_type="training_dataset_prepared",
        payload={
            "rows": int(cleaned.shape[0]),
            "feature_columns": feature_columns,
        },
    )

    X = cleaned[feature_columns]
    y = cleaned["risk_label"]
    return X, y


def load_weekly_scoring_data() -> pd.DataFrame:
    """Fetch and clean the simulated weekly scoring dataset.
    The function loads the weekly scoring dataset, cleans the columns,
    engineers the temporal features, and prepares the weekly scoring data for the model.
    """

    _, weekly_df = load_processed_datasets()
    cleaned = engineer_temporal_features(clean_columns(weekly_df))
    auditing.persist_audit_log(
        event_type="weekly_dataset_loaded",
        payload={
            "rows": int(cleaned.shape[0]),
        },
    )
    return cleaned

