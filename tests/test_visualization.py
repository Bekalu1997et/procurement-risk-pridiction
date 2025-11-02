"""Test visualization functions.

Why we test this:
- Visualizations are critical for model interpretability and stakeholder communication
- Validates SHAP plots are generated without errors
- Ensures pairplots help identify feature correlations
- Tests that plots are saved to disk for reports and dashboards
- Catches matplotlib/seaborn compatibility issues
- Verifies visualizations work with different data shapes and types
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src import visualization


def test_plot_shap_summary():
    """Test SHAP summary plot generation.
    
    Why: SHAP plots visually explain feature importance.
    Ensures plots are created and saved for explainability reports.
    """
    shap_vals = [0.5, -0.3, 0.8]
    features = ["f1", "f2", "f3"]
    path = visualization.plot_shap_summary(shap_vals, features, output_name="test")
    assert path.exists()


def test_create_pairplot():
    """Test pairplot creation.
    
    Why: Pairplots help identify feature correlations and data distributions.
    Validates seaborn integration works with our data schema.
    """
    df = pd.DataFrame({
        "credit_score": [700, 650, 800],
        "late_ratio": [0.1, 0.2, 0.05],
        "risk_label": ["Low", "Medium", "High"]
    })
    path = visualization.create_pairplot(df, output_name="test_pair")
    assert path.exists()
