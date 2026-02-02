"""Pandas-first auditing utilities for the supplier risk demo platform.

The auditing module provides a thin abstraction around data-quality checks,
lineage capture, and event persistence.  Each function is intentionally
lightweight and observation-friendly, making it easy to showcase how modern
MLOps platforms weave auditing into day-to-day data science workflows.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd
import yaml

from . import db_connector


LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports" / "audit_logs"
CONFIG_PATH = BASE_DIR / "config" / "auditing_config.yaml"


@dataclass
class AuditEvent:
    """Representation of a single audit event destined for storage."""

    event_type: str
    payload: Dict[str, Any]


def load_auditing_rules() -> Dict[str, Any]:
    """Load auditing configuration from YAML, returning sensible defaults."""

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as stream:
            return yaml.safe_load(stream) or {}
    return {
        "data_quality_checks": {
            "max_null_ratio": 0.1,
            "acceptable_regions": ["North America", "Europe", "Asia-Pacific", "LATAM"],
        }
    }


def persist_audit_log(event_type: str, payload: Mapping[str, Any]) -> None:
    """Write an audit event both to CSV (for transparency) and SQLite."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    audit_frame = pd.DataFrame(
        [
            {
                "event_type": event_type,
                "payload": json.dumps(payload, default=str),
                "created_at": pd.Timestamp.utcnow(),
            }
        ]
    )

    csv_path = REPORTS_DIR / "audit_events_log.csv"
    if csv_path.exists():
        audit_frame.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        audit_frame.to_csv(csv_path, index=False)

    db_connector.write_audit_event(event_type=event_type, payload=dict(payload))


def record_lineage(table_name: str, source_files: Sequence[str]) -> None:
    """Capture data lineage relationships for downstream traceability."""

    persist_audit_log(
        event_type="data_lineage",
        payload={
            "table": table_name,
            "source_files": list(source_files),
        },
    )


def _compute_null_ratios(df: pd.DataFrame) -> Dict[str, float]:
    """Utility to compute column-wise null ratios."""

    return {column: float(df[column].isna().mean()) for column in df.columns}


def log_data_quality(df: pd.DataFrame, checks: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Execute simple pandas-based data quality checks and log the results.

    Parameters
    ----------
    df:
        Input dataframe to analyse.
    checks:
        Optional overrides for auditing rules. If not provided, configuration
        is loaded from `auditing_config.yaml`.

    Returns
    -------
    pd.DataFrame
        Tabular summary of data quality metrics and pass/fail flags.
    """

    rules = checks or load_auditing_rules().get("data_quality_checks", {})
    max_null_ratio = float(rules.get("max_null_ratio", 0.1))
    acceptable_regions = set(rules.get("acceptable_regions", []))

    null_ratios = _compute_null_ratios(df)
    has_region_issue = (
        "region" in df.columns
        and acceptable_regions
        and not set(df["region"].dropna().unique()).issubset(acceptable_regions)
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "max_null_ratio",
                "value": max(null_ratios.values()) if null_ratios else 0.0,
                "threshold": max_null_ratio,
                "passed": max(null_ratios.values()) <= max_null_ratio if null_ratios else True,
            },
            {
                "metric": "region_domain_check",
                "value": int(has_region_issue),
                "threshold": 0,
                "passed": not has_region_issue,
            },
        ]
    )

    persist_audit_log(
        event_type="data_quality",
        payload={
            "max_null_ratio": null_ratios,
            "rules": rules,
            "passed": bool(summary["passed"].all()),
        },
    )
    return summary


def log_batch_predictions(
    predictions: pd.DataFrame,
    context: Mapping[str, Any],
) -> None:
    """Persist summary statistics for batch scoring outputs."""

    summary = predictions.groupby("risk_label").size().to_dict()
    persist_audit_log(
        event_type="batch_scoring",
        payload={
            "label_distribution": summary,
            **context,
        },
    )


def export_audit_dataframe(events: Iterable[AuditEvent]) -> pd.DataFrame:
    """Helper used in notebooks to materialise audit events as a dataframe."""

    records = [
        {
            "event_type": event.event_type,
            **event.payload,
        }
        for event in events
    ]
    return pd.DataFrame(records)

