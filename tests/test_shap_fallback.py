"""Test SHAP-based fallback explanations.

Why we test this:
- Validates system works even when Ollama is unavailable
- Ensures SHAP summaries are clear and actionable
- Tests different risk levels generate appropriate recommendations
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src import explainability


def test_shap_summary_high_risk():
    """Test SHAP summary for high risk scenario.
    
    Why: Ensures high-risk suppliers get appropriate warnings and recommendations.
    """
    top_features = [
        ("numeric__credit_score", 0.1121),
        ("numeric__dispute_rate", 0.0638),
        ("categorical__industry_IT Services", -0.0523),
        ("numeric__clause_risk_score", 0.0516),
        ("numeric__annual_spend", -0.0460)
    ]
    
    summary = explainability._generate_shap_summary("high", 57.0, top_features)
    
    print("\n" + "="*70)
    print("HIGH RISK SHAP SUMMARY:")
    print("="*70)
    print(summary)
    print("="*70)
    
    assert "elevated risk" in summary.lower()
    assert "57.0%" in summary
    assert "credit score" in summary.lower()
    assert "enhanced monitoring" in summary.lower()


def test_shap_summary_medium_risk():
    """Test SHAP summary for medium risk scenario.
    
    Why: Ensures medium-risk suppliers get balanced assessment.
    """
    top_features = [
        ("numeric__clause_risk_score", -0.1033),
        ("numeric__annual_spend", 0.0873),
        ("numeric__credit_score", 0.0548),
        ("numeric__late_ratio", -0.0416),
        ("categorical__region_North America", 0.0314)
    ]
    
    summary = explainability._generate_shap_summary("medium", 41.3, top_features)
    
    print("\n" + "="*70)
    print("MEDIUM RISK SHAP SUMMARY:")
    print("="*70)
    print(summary)
    print("="*70)
    
    assert "moderate risk" in summary.lower()
    assert "41.3%" in summary
    assert "monitor" in summary.lower()


def test_shap_summary_low_risk():
    """Test SHAP summary for low risk scenario.
    
    Why: Ensures low-risk suppliers get appropriate minimal monitoring recommendations.
    """
    top_features = [
        ("numeric__delivery_score", -0.0850),
        ("numeric__financial_stability_index", -0.0720),
        ("numeric__relationship_years", -0.0650),
        ("numeric__late_ratio", -0.0420),
        ("numeric__dispute_rate", -0.0310)
    ]
    
    summary = explainability._generate_shap_summary("low", 75.5, top_features)
    
    print("\n" + "="*70)
    print("LOW RISK SHAP SUMMARY:")
    print("="*70)
    print(summary)
    print("="*70)
    
    assert "minimal risk" in summary.lower()
    assert "75.5%" in summary
    assert "standard monitoring" in summary.lower()


def test_ollama_with_fallback():
    """Test Ollama call with SHAP fallback.
    
    Why: Validates system gracefully falls back to SHAP summary if Ollama fails.
    """
    top_features = [
        ("numeric__credit_score", 0.1028),
        ("numeric__annual_spend", -0.0925),
        ("numeric__late_ratio", 0.0529)
    ]
    
    prompt = explainability._build_business_prompt("high", 54.0, top_features)
    
    # This will try Ollama first, then fallback to SHAP if needed
    narrative = explainability._call_ollama(prompt, "high", 54.0, top_features)
    
    print("\n" + "="*70)
    print("NARRATIVE (Ollama or SHAP Fallback):")
    print("="*70)
    print(narrative)
    print("="*70)
    
    assert len(narrative) > 50
    assert "risk" in narrative.lower()


def test_all_risk_levels():
    """Test SHAP summaries for all risk levels.
    
    Why: Comprehensive validation that all risk scenarios are handled.
    """
    test_cases = [
        ("high", 85.0, [("numeric__dispute_rate", 0.15), ("numeric__credit_score", 0.12)]),
        ("medium", 50.0, [("numeric__late_ratio", 0.08), ("numeric__annual_spend", -0.06)]),
        ("low", 20.0, [("numeric__delivery_score", -0.10), ("numeric__relationship_years", -0.08)])
    ]
    
    print("\n" + "="*70)
    print("ALL RISK LEVELS SUMMARY:")
    print("="*70)
    
    for risk_level, confidence, features in test_cases:
        summary = explainability._generate_shap_summary(risk_level, confidence, features)
        print(f"\n{risk_level.upper()} Risk ({confidence}%):")
        print(f"  {summary}")
        
        # Check appropriate risk descriptor is present
        risk_descriptors = {"high": "elevated", "medium": "moderate", "low": "minimal"}
        assert risk_descriptors[risk_level] in summary.lower()
        assert f"{confidence:.1f}%" in summary
    
    print("\n" + "="*70)
