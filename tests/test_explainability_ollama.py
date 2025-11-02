"""Test explainability with Ollama tinyllama business narratives.

Why we test this:
- Validates Ollama integration works correctly
- Ensures business-friendly narratives are generated
- Tests SHAP values are translated to simple language
- Verifies explanations are conversational and actionable
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src import explainability


def test_build_business_prompt():
    """Test business prompt generation.
    
    Why: Ensures prompts translate technical SHAP values to business language.
    """
    top_features = [
        ("numeric__credit_score", 0.1028),
        ("numeric__late_ratio", -0.0529),
        ("categorical__industry_Manufacturing", 0.0180)
    ]
    
    prompt = explainability._build_business_prompt("high", 85.5, top_features)
    
    print("\n" + "="*70)
    print("GENERATED BUSINESS PROMPT:")
    print("="*70)
    print(prompt)
    print("="*70)
    
    assert "HIGH risk" in prompt
    assert "85.5%" in prompt
    assert "Credit Score" in prompt
    assert "increases risk" in prompt or "decreases risk" in prompt


def test_ollama_narrative_generation():
    """Test Ollama generates business narrative.
    
    Why: Validates end-to-end explanation generation with LLM.
    """
    top_features = [
        ("numeric__clause_risk_score", -0.1033),
        ("numeric__annual_spend", 0.0873),
        ("numeric__credit_score", 0.0548),
        ("numeric__late_ratio", -0.0416),
        ("categorical__region_North America", 0.0314)
    ]
    
    prompt = explainability._build_business_prompt("medium", 41.3, top_features)
    narrative = explainability._call_ollama(prompt)
    
    print("\n" + "="*70)
    print("OLLAMA GENERATED NARRATIVE:")
    print("="*70)
    print(narrative)
    print("="*70)
    
    assert isinstance(narrative, str)
    assert len(narrative) > 20


def test_full_explanation_pipeline():
    """Test complete explanation with SHAP and Ollama.
    
    Why: Validates the entire explainability pipeline works end-to-end.
    """
    # Mock data
    risk_level = "high"
    probabilities = {"high": 0.57, "medium": 0.38, "low": 0.05}
    shap_values = [0.1121, 0.0638, -0.0523, 0.0516, -0.0460]
    feature_names = [
        "numeric__credit_score",
        "numeric__dispute_rate", 
        "categorical__industry_IT Services",
        "numeric__clause_risk_score",
        "numeric__annual_spend"
    ]
    
    explanation = explainability.build_explanation(
        risk_level=risk_level,
        probabilities=probabilities,
        shap_values=shap_values,
        feature_names=feature_names
    )
    
    print("\n" + "="*70)
    print("COMPLETE EXPLANATION OBJECT:")
    print("="*70)
    print(f"Risk Level: {explanation.risk_level}")
    print(f"Confidence: {explanation.confidence}%")
    print(f"\nTop Features:")
    for feat, val in explanation.top_features[:5]:
        print(f"  - {feat}: {val:.4f}")
    print(f"\nBusiness Narrative:")
    print(f"  {explanation.narrative}")
    print("="*70)
    
    assert explanation.risk_level == "high"
    assert explanation.confidence == 57.0
    assert len(explanation.top_features) > 0
    assert len(explanation.narrative) > 0


def test_feature_name_cleaning():
    """Test feature names are cleaned for business users.
    
    Why: Technical names like 'numeric__credit_score' should become 'Credit Score'.
    """
    top_features = [
        ("numeric__avg_payment_delay_days", 0.05),
        ("categorical__contract_criticality_High", -0.03)
    ]
    
    prompt = explainability._build_business_prompt("medium", 50.0, top_features)
    
    # Check technical prefixes are removed
    assert "numeric__" not in prompt
    assert "categorical__" not in prompt
    
    # Check names are readable
    assert "Avg Payment Delay Days" in prompt or "payment" in prompt.lower()
    
    print("\n" + "="*70)
    print("CLEANED FEATURE NAMES IN PROMPT:")
    print("="*70)
    print(prompt)
    print("="*70)
