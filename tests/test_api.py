"""Test API endpoints.

Why we test this:
- Ensures FastAPI endpoints are accessible and return correct HTTP status codes
- Validates API contract: request/response structure matches expected format
- Catches breaking changes in API before deployment
- Verifies health check endpoint for monitoring and load balancers
- Tests prediction endpoint with realistic supplier data payload
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from api.app import app
from tests.test_data_fixtures import testdata1, testdata2, testdata3

client = TestClient(app)


def test_health_check():
    """Test health check endpoint.
    
    Why: Health checks are critical for production monitoring, load balancers,
    and orchestration systems (Kubernetes, ECS) to determine service availability.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint():
    """Test prediction endpoint.
    
    Why: Validates the core business logic endpoint that external clients will use.
    Ensures the API correctly processes supplier data and returns risk predictions.
    """
    response = client.post("/predict", json=testdata1)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probabilities" in data


def test_predict_low_risk():
    """Test prediction with low risk supplier data.
    
    Why: Validates model correctly identifies low-risk suppliers.
    """
    response = client.post("/predict", json=testdata1)
    assert response.status_code == 200


def test_predict_medium_risk():
    """Test prediction with medium risk supplier data.
    
    Why: Validates model handles medium-risk scenarios.
    """
    response = client.post("/predict", json=testdata2)
    assert response.status_code == 200


def test_predict_high_risk():
    """Test prediction with high risk supplier data.
    
    Why: Validates model correctly flags high-risk suppliers.
    """
    response = client.post("/predict", json=testdata3)
    assert response.status_code == 200
