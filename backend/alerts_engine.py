"""Mock alerts engine that surfaces high-risk suppliers for review."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src import auditing, db_connector


def generate_alerts(predictions: pd.DataFrame, threshold: float = 0.6) -> List[Dict[str, object]]:
    """Return alerts for suppliers with high predicted risk probability."""

    alerts_df = predictions[predictions["prob_high"] >= threshold]
    alerts = alerts_df[
        ["supplier_id", "prob_high", "prediction"]
    ].to_dict(orient="records")

    auditing.persist_audit_log(
        event_type="alerts_generated",
        payload={
            "count": len(alerts),
            "threshold": threshold,
        },
    )

    if alerts:
        db_connector.save_predictions(
            {
                "supplier_id": alert["supplier_id"],
                "model_name": "weekly_scoring",
                "risk_label": alert["prediction"],
                "risk_score": alert["prob_high"],
                "explanation": "Flagged by weekly scoring pipeline",
                "created_at": pd.Timestamp.utcnow(),
            }
            for alert in alerts
        )

    return alerts

