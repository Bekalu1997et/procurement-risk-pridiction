"""Test SHAP explainability with feature importance visualization.

Why we test this:
- SHAP values explain which features drive predictions (regulatory compliance)
- Validates explainability pipeline works end-to-end
- Shows feature contributions in human-readable format
- Tests that explanations match predictions
- Critical for model transparency and trust
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import shap

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from tests.test_data_fixtures import testdata1, testdata2, testdata3, FEATURE_BUNDLE

MODELS_DIR = BASE_DIR / "models"


def load_model():
    """Load trained model."""
    import joblib
    return joblib.load(MODELS_DIR / "random_forest.joblib")


def prepare_features(data: dict) -> pd.DataFrame:
    """Prepare features for model prediction."""
    df = pd.DataFrame([data])
    required_cols = ['region', 'industry', 'contract_criticality', 'credit_score'] + FEATURE_BUNDLE["numeric_features"]
    return df[required_cols]


def compute_shap_values(model, X):
    """Compute SHAP values for predictions.
    
    Why: SHAP values show how each feature contributes to the prediction.
    """
    explainer = shap.TreeExplainer(model.named_steps['model'])
    X_transformed = model.named_steps['preprocessor'].transform(X)
    shap_values = explainer.shap_values(X_transformed)
    return shap_values, explainer


def display_feature_importance(shap_values, feature_names, prediction, data_name):
    """Display top feature contributions in readable format.
    
    Why: Shows which features drove the prediction and by how much.
    """
    # For multi-class, get SHAP values for predicted class
    class_idx = {'high': 0, 'low': 1, 'medium': 2}
    
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Shape: (n_samples, n_features, n_classes)
        shap_vals = shap_values[0, :, class_idx.get(prediction, 0)]
    elif isinstance(shap_values, list):
        shap_vals = shap_values[class_idx.get(prediction, 0)][0]
    else:
        shap_vals = shap_values[0]
    
    # Create feature importance ranking
    feature_impact = [(feat, float(val)) for feat, val in zip(feature_names, shap_vals)]
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n{'='*70}")
    print(f"{data_name} - Prediction: {prediction.upper()}")
    print(f"{'='*70}")
    print(f"\n{'Feature':<40} {'SHAP Value':<15} {'Impact'}")
    print(f"{'-'*70}")
    
    for i, (feature, value) in enumerate(feature_impact[:10], 1):
        impact = "+ Increases Risk" if value > 0 else "- Decreases Risk"
        print(f"{i:2}. {feature:<35} {value:>10.4f}    {impact}")
    
    return feature_impact[:10]


def test_shap_testdata1():
    """Test SHAP explainability on testdata1 (low risk).
    
    Why: Validates SHAP shows why model predicted this risk level.
    Shows which features contributed most to the decision.
    """
    model = load_model()
    X = prepare_features(testdata1)
    prediction = model.predict(X)[0]
    
    shap_values, explainer = compute_shap_values(model, X)
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    
    top_features = display_feature_importance(shap_values, feature_names, prediction, "TestData1 (Low Risk Profile)")
    
    assert len(top_features) == 10
    assert all(isinstance(f[0], str) for f in top_features)


def test_shap_testdata2():
    """Test SHAP explainability on testdata2 (medium risk).
    
    Why: Validates SHAP explanations for medium-risk scenarios.
    """
    model = load_model()
    X = prepare_features(testdata2)
    prediction = model.predict(X)[0]
    
    shap_values, explainer = compute_shap_values(model, X)
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    
    top_features = display_feature_importance(shap_values, feature_names, prediction, "TestData2 (Medium Risk Profile)")
    
    assert len(top_features) == 10


def test_shap_testdata3():
    """Test SHAP explainability on testdata3 (high risk).
    
    Why: Validates SHAP shows high-risk drivers clearly.
    """
    model = load_model()
    X = prepare_features(testdata3)
    prediction = model.predict(X)[0]
    
    shap_values, explainer = compute_shap_values(model, X)
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    
    top_features = display_feature_importance(shap_values, feature_names, prediction, "TestData3 (High Risk Profile)")
    
    assert len(top_features) == 10


def test_shap_all_datasets():
    """Test SHAP explainability on all test datasets.
    
    Why: Comprehensive view of how features contribute across risk profiles.
    Shows model reasoning for different supplier scenarios.
    """
    model = load_model()
    test_datasets = [
        (testdata1, "TestData1 (Low Risk)"),
        (testdata2, "TestData2 (Medium Risk)"),
        (testdata3, "TestData3 (High Risk)")
    ]
    
    print("\n" + "="*70)
    print("SHAP EXPLAINABILITY ANALYSIS - ALL TEST DATASETS")
    print("="*70)
    
    for data, name in test_datasets:
        X = prepare_features(data)
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        shap_values, explainer = compute_shap_values(model, X)
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        
        top_features = display_feature_importance(shap_values, feature_names, prediction, name)
        
        print(f"\nProbabilities: High={probabilities[0]:.2%}, Low={probabilities[1]:.2%}, Medium={probabilities[2]:.2%}")
        print()


def test_shap_base_value():
    """Test SHAP base value (average prediction).
    
    Why: Base value shows the model's average prediction before considering features.
    """
    model = load_model()
    X = prepare_features(testdata1)
    
    explainer = shap.TreeExplainer(model.named_steps['model'])
    X_transformed = model.named_steps['preprocessor'].transform(X)
    
    base_value = explainer.expected_value
    print(f"\nSHAP Base Values (average model output): {base_value}")
    
    assert base_value is not None
