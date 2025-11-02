"""Simulated MLOps workflow orchestrating weekly scoring and alerting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
from apscheduler.schedulers.background import BackgroundScheduler

import auditing, data_pipeline, model_pipeline, visualization
from backend import alerts_engine


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports" / "weekly_reports"
CONFIG_PATH = BASE_DIR / "config" / "data_refresh.yaml"


def _load_feature_columns() -> Dict[str, list[str]]:
    """Load feature definitions shared across training and inference."""

    model_config_path = BASE_DIR / "config" / "model_config.yaml"
    with open(model_config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    return {
        "categorical": config["categorical_features"],
        "numeric": config["numeric_features"],
    }


def _load_schedule_configuration() -> Dict[str, int]:
    """Fetch scheduling parameters for the APScheduler."""

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
            return config.get("schedule", {"day_of_week": "sun", "hour": 2})
    return {"day_of_week": "sun", "hour": 2}


def run_weekly_scoring(model_name: str = "random_forest") -> pd.DataFrame:
    """Execute a single pass of the weekly scoring workflow."""

    weekly_df = data_pipeline.load_weekly_scoring_data()
    feature_columns = _load_feature_columns()
    feature_df = weekly_df[
        feature_columns["categorical"] + feature_columns["numeric"]
    ]

    predictions = model_pipeline.predict_batch(model_name=model_name, data=feature_df)
    predictions.insert(0, "supplier_id", weekly_df["supplier_id"].values)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / f"weekly_predictions_{pd.Timestamp.utcnow().date()}.csv"
    predictions.to_csv(output_path, index=False)

    visualization.create_weekly_plots(predictions, output_path.stem)
    alerts = alerts_engine.generate_alerts(predictions)
    auditing.persist_audit_log(
        event_type="weekly_scoring_complete",
        payload={
            "model": model_name,
            "predictions_path": str(output_path),
            "alerts": alerts,
        },
    )
    return predictions


def schedule_weekly_job() -> BackgroundScheduler:
    """Configure and start the APScheduler job for weekly scoring."""

    scheduler = BackgroundScheduler()
    schedule_cfg = _load_schedule_configuration()

    scheduler.add_job(
        run_weekly_scoring,
        "cron",
        id="weekly_scoring",
        replace_existing=True,
        **schedule_cfg,
    )
    scheduler.start()
    return scheduler


if __name__ == "__main__":  # pragma: no cover - manual execution path.
    run_weekly_scoring()

