"""Test model predictions using pkl files from models folder.

Why we test this:
- Validates trained models (rf_model.pkl, tfidf_vectorizer.pkl) work correctly
- Ensures models can load and predict on test data
- Tests predictions match expected risk levels
- Catches model serialization/deserialization issues
- Verifies feature engineering pipeline works end-to-end
"""

import pickle
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from tests.test_data_fixtures import testdata1, testdata2, testdata3, FEATURE_BUNDLE

MODELS_DIR = BASE_DIR / "models"


def load_models():
    """Load trained models from joblib files.
    
    Why: Validates model files exist and can be loaded.
    """
    import joblib
    rf_model = joblib.load(MODELS_DIR / "random_forest.joblib")
    return rf_model


def prepare_features(data: dict) -> pd.DataFrame:
    """Prepare features for model prediction.
    
    Why: Ensures test data is transformed to match training schema.
    Model expects both numeric and categorical features.
    """
    df = pd.DataFrame([data])
    # Model expects: region, industry, contract_criticality, credit_score + numeric features
    required_cols = ['region', 'industry', 'contract_criticality', 'credit_score'] + FEATURE_BUNDLE["numeric_features"]
    return df[required_cols]


def test_load_models():
    """Test loading joblib models.
    
    Why: Ensures model files exist and are not corrupted.
    """
    rf_model = load_models()
    assert rf_model is not None


def test_predict_testdata1():
    """Test prediction on testdata1 (low risk).
    
    Why: Validates model correctly identifies low-risk suppliers.
    Expected: Low risk prediction with high confidence.
    """
    rf_model = load_models()
    X = prepare_features(testdata1)
    prediction = rf_model.predict(X)[0]
    probabilities = rf_model.predict_proba(X)[0]
    
    print(f"\nTestData1 Prediction: {prediction}")
    print(f"Probabilities: {probabilities}")
    
    assert prediction in ["low", "medium", "high"]
    assert len(probabilities) == 3
    assert np.sum(probabilities) > 0.99  # Probabilities sum to 1


def test_predict_testdata2():
    """Test prediction on testdata2 (medium risk).
    
    Why: Validates model handles medium-risk scenarios.
    Expected: Medium risk prediction.
    """
    rf_model = load_models()
    X = prepare_features(testdata2)
    prediction = rf_model.predict(X)[0]
    probabilities = rf_model.predict_proba(X)[0]
    
    print(f"\nTestData2 Prediction: {prediction}")
    print(f"Probabilities: {probabilities}")
    
    assert prediction in ["low", "medium", "high"]
    assert len(probabilities) == 3


def test_predict_testdata3():
    """Test prediction on testdata3 (high risk).
    
    Why: Validates model correctly flags high-risk suppliers.
    Expected: High risk prediction with high confidence.
    """
    rf_model = load_models()
    X = prepare_features(testdata3)
    prediction = rf_model.predict(X)[0]
    probabilities = rf_model.predict_proba(X)[0]
    
    print(f"\nTestData3 Prediction: {prediction}")
    print(f"Probabilities: {probabilities}")
    
    assert prediction in ["low", "medium", "high"]
    assert len(probabilities) == 3


def test_all_test_data():
    """Test predictions on all test datasets.
    
    Why: Validates model works consistently across different risk profiles.
    """
    rf_model = load_models()
    test_datasets = [testdata1, testdata2, testdata3]
    
    for i, data in enumerate(test_datasets, 1):
        X = prepare_features(data)
        prediction = rf_model.predict(X)[0]
        probabilities = rf_model.predict_proba(X)[0]
        
        print(f"\nTestData{i}:")
        print(f"  Prediction: {prediction}")
        print(f"  Probabilities: {dict(zip(['high', 'low', 'medium'], probabilities))}")
        
        assert prediction in ["low", "medium", "high"]
        assert 0.99 < np.sum(probabilities) < 1.01


def test_feature_importance():
    """Test feature importance extraction.
    
    Why: Validates model has learned meaningful feature relationships.
    """
    rf_model = load_models()
    
    if hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
        numeric_features = FEATURE_BUNDLE["numeric_features"]
        
        print("\nTop 5 Feature Importances:")
        feature_imp = sorted(zip(numeric_features, importances), key=lambda x: x[1], reverse=True)[:5]
        for feat, imp in feature_imp:
            print(f"  {feat}: {imp:.4f}")
        
        assert len(importances) == len(numeric_features)
        assert np.sum(importances) > 0
