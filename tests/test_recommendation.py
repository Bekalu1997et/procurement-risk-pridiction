"""Test recommendation functions."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src import recommendation


def test_build_recommendations():
    """Test recommendation building."""
    top_features = [("late_ratio", 0.5), ("credit_score", -0.3)]
    recos = recommendation.build_recommendations("High", top_features)
    assert isinstance(recos, list)
    assert len(recos) > 0
    assert all(isinstance(r, str) for r in recos)
