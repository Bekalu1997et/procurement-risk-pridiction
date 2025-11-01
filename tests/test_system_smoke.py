"""Lightweight smoke tests verifying the end-to-end demo pipeline."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src import auditing, data_pipeline, model_pipeline  # noqa: E402

DEMO_PIPELINE_PATH = BASE_DIR / "data" / "raw" / "demo_pipeline.py"

spec = importlib.util.spec_from_file_location("demo_pipeline", DEMO_PIPELINE_PATH)
assert spec is not None and spec.loader is not None  # pragma: no cover - sanity check
_demo_pipeline_module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(_demo_pipeline_module)
except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
    pytest.skip(
        f"Skipping demo pipeline tests: missing optional dependency ({exc}).",
        allow_module_level=True,
    )
demo_pipeline = _demo_pipeline_module


def test_demo_pipeline_generates_datasets():
    """Ensure the synthetic data pipeline runs end-to-end and writes artefacts."""

    artifacts = demo_pipeline.run_pipeline(num_suppliers=250, num_transactions=600, random_state=123)
    assert not artifacts.merged.empty
    assert (BASE_DIR / "data" / "processed" / "merged_training.csv").exists()
    assert (BASE_DIR / "models" / "rf_model.pkl").exists()
    assert (BASE_DIR / "models" / "feature_bundle.json").exists()


def test_model_training_and_prediction():
    """Train models on the generated data and ensure predictions work."""

    X, y = data_pipeline.prepare_training_data()
    artifacts = model_pipeline.train_models(X, y)
    assert artifacts

    sample = X.iloc[0].to_dict()
    result = model_pipeline.predict_single("random_forest", sample)
    assert "prediction" in result and "probabilities" in result


def test_demo_pipeline_predictor():
    """Validate the helper prediction wrapper uses persisted artefacts."""

    merged = pd.read_csv(BASE_DIR / "data" / "processed" / "merged_training.csv")
    payload = merged.iloc[0].to_dict()
    result = demo_pipeline.predict_supplier_risk(payload)
    assert {"risk_score", "risk_label", "probabilities", "recommendation"}.issubset(result.keys())


def test_auditing_log_quality():
    """Check that auditing logs are produced in CSV format."""

    training_df, _ = data_pipeline.load_processed_datasets()
    report = auditing.log_data_quality(training_df.head())
    assert isinstance(report, pd.DataFrame)
    csv_path = BASE_DIR / "reports" / "audit_logs" / "audit_events_log.csv"
    assert csv_path.exists()

