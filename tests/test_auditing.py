"""Test auditing functions."""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src import auditing


def test_log_data_quality():
    """Test data quality logging."""
    df = pd.DataFrame({"a": [1, 2, None], "b": [4, 5, 6]})
    report = auditing.log_data_quality(df)
    assert isinstance(report, pd.DataFrame)
    assert not report.empty


def test_persist_audit_log():
    """Test audit log persistence."""
    auditing.persist_audit_log(
        event_type="test_event",
        payload={"test_key": "test_value"}
    )
    csv_path = BASE_DIR / "reports" / "audit_logs" / "audit_events_log.csv"
    assert csv_path.exists()
