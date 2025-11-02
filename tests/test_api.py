"""Test API endpoints."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from api.app import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint():
    """Test prediction endpoint."""
    payload = {
        "region": "North America",
        "industry": "Manufacturing",
        "contract_criticality": "High",
        "annual_spend": 50000.0,
        "credit_score": 700,
        "late_ratio": 0.2,
        "dispute_rate": 0.05,
        "avg_delay": 5.0,
        "clause_risk_score": 35.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probabilities" in data
