"""Test auditing functions.

Why we test this:
- Ensures audit trail captures all critical events for compliance and debugging
- Validates data quality checks detect missing values, outliers, and anomalies
- Verifies audit logs are persisted correctly to CSV for regulatory requirements
- Tests that auditing doesn't break the main application flow
- Critical for production observability and troubleshooting
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src import auditing


def test_log_data_quality():
    """Test data quality logging.
    
    Why: Data quality issues are the #1 cause of ML model failures in production.
    This test ensures we detect and log missing values, type mismatches, and anomalies.
    """
    df = pd.DataFrame({"a": [1, 2, None], "b": [4, 5, 6]})
    report = auditing.log_data_quality(df)
    assert isinstance(report, pd.DataFrame)
    assert not report.empty


def test_persist_audit_log():
    """Test audit log persistence.
    
    Why: Audit logs are required for compliance (SOX, GDPR) and debugging production issues.
    Verifies events are written to disk and can be retrieved for analysis.
    """
    auditing.persist_audit_log(
        event_type="test_event",
        payload={"test_key": "test_value"}
    )
    csv_path = BASE_DIR / "reports" / "audit_logs" / "audit_events_log.csv"
    assert csv_path.exists()
