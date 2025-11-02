"""Cleaned auditing utilities used by tests and notebooks.

This module provides small, dependency-resilient helpers for recording
audit events and simple data-quality checks. It prefers importing the
project-local `db_connector` package but will tolerate its absence in
lightweight test runs (writing only CSV logs).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports" / "audit_logs"
CONFIG_PATH = BASE_DIR / "config" / "auditing_config.yaml"

# Try to import db_connector from package; fall back gracefully if not available.
try:
    # Preferred: absolute package import when running from tests or app
    from src import db_connector  # type: ignore
except Exception:
    try:
        # Relative import when module executed as package sibling
        from . import db_connector  # type: ignore
    except Exception:
        db_connector = None  # type: ignore


@dataclass
class AuditEvent:
    event_type: str
    payload: Dict[str, Any]


def load_auditing_rules() -> Dict[str, Any]:
    """Load auditing configuration from YAML, returning sensible defaults."""

    default = {
        "data_quality_checks": {
            "max_null_ratio": 0.1,
            "acceptable_regions": ["North America", "Europe", "Asia-Pacific", "LATAM"],
        }
    }
    if CONFIG_PATH.exists():
        try:
            import yaml

            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                return {**default, **(yaml.safe_load(fh) or {})}
        except Exception:
            LOGGER.debug("Failed to load auditing config; using defaults", exc_info=True)
            return default
    return default


def persist_audit_log(event_type: str, payload: Mapping[str, Any]) -> None:
    """Persist an audit event to a CSV and, if available, the DB connector.

    This function never raises on persistence errors; it logs them and continues.
    """

    try:
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

        if db_connector is not None:
            try:
                db_connector.write_audit_event(event_type=event_type, payload=dict(payload))
            except Exception:
                LOGGER.debug("db_connector failed to write audit event", exc_info=True)
    except Exception:
        LOGGER.exception("Failed to persist audit log")


def record_lineage(table_name: str, source_files: Sequence[str]) -> None:
    persist_audit_log(event_type="data_lineage", payload={"table": table_name, "source_files": list(source_files)})


def _compute_null_ratios(df: pd.DataFrame) -> Dict[str, float]:
    return {column: float(df[column].isna().mean()) for column in df.columns}


def log_data_quality(df: pd.DataFrame, checks: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Run simple data-quality checks and log results. Returns a summary DataFrame.

    The function records results to the audit log and returns a small
    DataFrame summarising whether checks passed.
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

    persist_audit_log(event_type="data_quality", payload={"max_null_ratio": null_ratios, "rules": rules, "passed": bool(summary["passed"].all())})
    return summary


def log_batch_predictions(predictions: pd.DataFrame, context: Mapping[str, Any]) -> None:
    summary = predictions.groupby("risk_label").size().to_dict()
    persist_audit_log(event_type="batch_scoring", payload={"label_distribution": summary, **context})


def export_audit_dataframe(events: Iterable[AuditEvent]) -> pd.DataFrame:
    records = [{"event_type": event.event_type, **event.payload} for event in events]
    return pd.DataFrame(records)

