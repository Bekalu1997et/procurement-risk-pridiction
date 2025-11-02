"""Test explainability functions."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src import explainability


def test_summarize_shap_values():
    """Test SHAP value summarization."""
    shap_vals = [0.5, -0.3, 0.8, -0.1]
    features = ["f1", "f2", "f3", "f4"]
    top = explainability.summarize_shap_values(shap_vals, features, top_k=2)
    assert len(top) == 2
    assert top[0][0] == "f3"


def test_build_explanation():
    """Test explanation building."""
    exp = explainability.build_explanation(
        risk_level="High",
        probabilities={"High": 0.85, "Medium": 0.10, "Low": 0.05},
        shap_values=[0.5, -0.3, 0.8],
        feature_names=["f1", "f2", "f3"],
    )
    assert exp.risk_level == "High"
    assert exp.confidence == 85.0
    assert len(exp.top_features) > 0
    assert exp.narrative is not None
