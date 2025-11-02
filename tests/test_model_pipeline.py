"""Test model pipeline functions."""

import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src import data_pipeline, model_pipeline


def test_train_models():
    """Test model training."""
    X, y = data_pipeline.prepare_training_data()
    artifacts = model_pipeline.train_models(X, y)
    assert artifacts is not None
    assert (BASE_DIR / "models" / "random_forest.joblib").exists()


def test_predict_single():
    """Test single prediction."""
    X, _ = data_pipeline.prepare_training_data()
    sample = X.iloc[0].to_dict()
    result = model_pipeline.predict_single("random_forest", sample)
    assert "prediction" in result
    assert "probabilities" in result
    assert "shap_values" in result
