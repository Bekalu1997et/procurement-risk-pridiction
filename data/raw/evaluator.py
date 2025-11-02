"""Model evaluation and prediction module for supplier risk prediction system.

This module handles model evaluation, prediction, and risk scoring for individual
supplier records.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure the project root is available on the PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import auditing  # pylint: disable=wrong-import-position


def predict_supplier_risk(
    supplier_record: Dict[str, object],
    *,
    model: Optional[RandomForestClassifier] = None,
    vectorizer: Optional[TfidfVectorizer] = None,
    numeric_features: Optional[Sequence[str]] = None,
    models_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Score a supplier record using the saved model and heuristic modifiers."""

    models_dir = models_dir or PROJECT_ROOT / "models"
    if model is None:
        model = joblib.load(models_dir / "rf_model.pkl")  # type: ignore[assignment]
    if vectorizer is None:
        vectorizer = joblib.load(models_dir / "tfidf_vectorizer.pkl")
    feature_bundle_path = models_dir / "feature_bundle.json"
    if numeric_features is None:
        with open(feature_bundle_path, "r", encoding="utf-8") as handle:
            bundle = json.load(handle)
        numeric_features = bundle["numeric_features"]

    numeric_vector = np.array(
        [float(supplier_record.get(feature, 0)) for feature in numeric_features],
        dtype=float,
    ).reshape(1, -1)
    text_vector = vectorizer.transform([str(supplier_record.get("contract_text", ""))])
    combined = hstack([numeric_vector, text_vector])
    probabilities = model.predict_proba(combined)[0]
    probability_map = dict(zip(model.classes_, probabilities))

    modifier = 0.0
    if supplier_record.get("avg_payment_delay_days", 0) > 60:
        modifier += 0.12
    if supplier_record.get("past_disputes", 0) > 5:
        modifier += 0.08
    if supplier_record.get("financial_stability_index", 1.0) < 0.5:
        modifier += 0.12

    high_prob = probability_map.get("high", max(probabilities))
    final_score = min(1.0, float(high_prob) + modifier)
    if final_score >= 0.7:
        label = "high"
        recommendation = "Avoid supplier; identify alternate sourcing options."
    elif final_score >= 0.4:
        label = "medium"
        recommendation = "Renegotiate terms, request guarantees, monitor quarterly."
    else:
        label = "low"
        recommendation = "Continue partnership with routine monitoring cadence."

    auditing.persist_audit_log(
        event_type="predict_supplier_risk",
        payload={
            "supplier_id": supplier_record.get("supplier_id"),
            "score": final_score,
            "label": label,
            "modifier": modifier,
        },
    )

    return {
        "risk_score": round(final_score, 3),
        "risk_label": label,
        "probabilities": probability_map,
        "recommendation": recommendation,
    }
