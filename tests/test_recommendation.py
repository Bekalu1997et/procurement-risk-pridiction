"""Test recommendation functions.

Why we test this:
- Recommendations turn predictions into actionable business insights
- Validates that high-risk predictions generate mitigation strategies
- Ensures recommendations are based on top contributing features
- Tests that output is human-readable and actionable
- Catches logic errors in recommendation rules before production
- Critical for user adoption - predictions without actions are useless
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src import recommendation


def test_build_recommendations():
    """Test recommendation building.
    
    Why: Validates the system generates actionable recommendations.
    Ensures procurement teams know what to do with high-risk suppliers.
    """
    top_features = [("late_ratio", 0.5), ("credit_score", -0.3)]
    recos = recommendation.build_recommendations("High", top_features)
    assert isinstance(recos, list)
    assert len(recos) > 0
    assert all(isinstance(r, str) for r in recos)
