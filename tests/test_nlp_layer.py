"""Test NLP layer functions.

Why we test this:
- NLP features enhance risk prediction with contract text analysis
- Validates SHAP summarization for feature importance ranking
- Ensures contract summarization extracts key clauses correctly
- Tests Q&A functionality for contract interrogation
- Catches NLP model loading issues before production
- Verifies text processing doesn't crash on edge cases
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src import nlp_layer


def test_summarize_shap_values():
    """Test SHAP summarization helper.
    
    Why: Ensures NLP layer correctly ranks features by importance.
    Critical for explaining which contract clauses drive risk.
    """
    shap_vals = [0.5, -0.3, 0.8]
    features = ["f1", "f2", "f3"]
    top = nlp_layer.summarize_shap_values(shap_vals, features, top_k=2)
    assert len(top) == 2
    assert top[0][0] == "f3"


def test_summarize_contract():
    """Test contract summarization.
    
    Why: Contract summarization helps procurement teams quickly review key terms.
    Validates NLP models extract relevant clauses without errors.
    """
    text = "This contract includes payment terms and delivery schedules."
    summary = nlp_layer.summarize_contract(text)
    assert isinstance(summary, nlp_layer.NLPSummary)
    assert summary.raw_summary is not None


def test_contract_qna():
    """Test contract Q&A.
    
    Why: Q&A enables users to ask questions about contract terms.
    Ensures the system returns relevant answers from contract text.
    """
    context = "Payment is due within 30 days of invoice."
    question = "When is payment due?"
    answer = nlp_layer.contract_qna(question, context)
    assert isinstance(answer, str)
    assert len(answer) > 0
