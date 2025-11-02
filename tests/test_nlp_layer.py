"""Test NLP layer functions."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src import nlp_layer


def test_summarize_shap_values():
    """Test SHAP summarization helper."""
    shap_vals = [0.5, -0.3, 0.8]
    features = ["f1", "f2", "f3"]
    top = nlp_layer.summarize_shap_values(shap_vals, features, top_k=2)
    assert len(top) == 2
    assert top[0][0] == "f3"


def test_summarize_contract():
    """Test contract summarization."""
    text = "This contract includes payment terms and delivery schedules."
    summary = nlp_layer.summarize_contract(text)
    assert isinstance(summary, nlp_layer.NLPSummary)
    assert summary.raw_summary is not None


def test_contract_qna():
    """Test contract Q&A."""
    context = "Payment is due within 30 days of invoice."
    question = "When is payment due?"
    answer = nlp_layer.contract_qna(question, context)
    assert isinstance(answer, str)
    assert len(answer) > 0
