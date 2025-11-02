"""Test data fixtures based on feature bundle.

Why we create test fixtures:
- Provides consistent, reusable test data across all test files
- Ensures test data matches production feature schema
- Makes tests more maintainable - change once, update everywhere
- Covers different risk scenarios (low, medium, high)
- Validates model works with edge cases and typical inputs
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURE_BUNDLE_PATH = BASE_DIR / "models" / "feature_bundle.json"

# Load feature bundle to ensure test data matches production schema
with open(FEATURE_BUNDLE_PATH, "r") as f:
    FEATURE_BUNDLE = json.load(f)


# Test Data 1: Low Risk Supplier
testdata1 = {
    "annual_revenue": 5000000.0,
    "annual_spend": 100000.0,
    "avg_payment_delay_days": 2.0,
    "contract_value": 250000.0,
    "contract_duration_months": 24,
    "past_disputes": 0,
    "delivery_score": 95.0,
    "financial_stability_index": 85.0,
    "relationship_years": 10,
    "txn_count": 150,
    "avg_txn_amount": 5000.0,
    "avg_delay": 1.5,
    "late_ratio": 0.02,
    "dispute_rate": 0.0,
    "avg_delivery_quality": 92.0,
    "clause_risk_score": 15.0,
    "region": "North America",
    "industry": "Manufacturing",
    "contract_criticality": "Low",
    "credit_score": 800
}


# Test Data 2: Medium Risk Supplier
testdata2 = {
    "annual_revenue": 2000000.0,
    "annual_spend": 75000.0,
    "avg_payment_delay_days": 10.0,
    "contract_value": 150000.0,
    "contract_duration_months": 12,
    "past_disputes": 2,
    "delivery_score": 70.0,
    "financial_stability_index": 60.0,
    "relationship_years": 3,
    "txn_count": 80,
    "avg_txn_amount": 3000.0,
    "avg_delay": 8.0,
    "late_ratio": 0.15,
    "dispute_rate": 0.08,
    "avg_delivery_quality": 68.0,
    "clause_risk_score": 45.0,
    "region": "Europe",
    "industry": "Logistics",
    "contract_criticality": "Medium",
    "credit_score": 650
}


# Test Data 3: High Risk Supplier
testdata3 = {
    "annual_revenue": 500000.0,
    "annual_spend": 25000.0,
    "avg_payment_delay_days": 25.0,
    "contract_value": 50000.0,
    "contract_duration_months": 6,
    "past_disputes": 8,
    "delivery_score": 45.0,
    "financial_stability_index": 30.0,
    "relationship_years": 1,
    "txn_count": 20,
    "avg_txn_amount": 1500.0,
    "avg_delay": 20.0,
    "late_ratio": 0.45,
    "dispute_rate": 0.30,
    "avg_delivery_quality": 40.0,
    "clause_risk_score": 85.0,
    "region": "Asia-Pacific",
    "industry": "IT Services",
    "contract_criticality": "High",
    "credit_score": 450
}


def get_test_data(risk_level: str) -> dict:
    """Get test data by risk level.
    
    Args:
        risk_level: One of 'low', 'medium', 'high'
    
    Returns:
        Dictionary with supplier features
    """
    mapping = {
        "low": testdata1,
        "medium": testdata2,
        "high": testdata3
    }
    return mapping.get(risk_level.lower(), testdata2)


def validate_test_data(data: dict) -> bool:
    """Validate test data has all required numeric features.
    
    Why: Ensures test data matches production feature schema.
    Catches missing features before tests fail.
    """
    required_numeric = set(FEATURE_BUNDLE["numeric_features"])
    provided = set(k for k in data.keys() if k in required_numeric)
    return required_numeric.issubset(provided)


# Validate all test data on import
assert validate_test_data(testdata1), "testdata1 missing required features"
assert validate_test_data(testdata2), "testdata2 missing required features"
assert validate_test_data(testdata3), "testdata3 missing required features"
