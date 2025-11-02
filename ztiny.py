"""
Explainability Pipeline: SHAP + TinyLlama Narratives
----------------------------------------------------

This pipeline combines SHAP model explainability with TinyLlama-generated
business narratives. It runs predictions, evaluates performance,
explains results, and exports reports.
"""
# =============================================================================
# TROUBLESHOOTING: Pickle Load Error Handling Guidance
# =============================================================================

# If you encounter a RuntimeError when loading a pickled model or vectorizer,
# such as:
#   RuntimeError: Failed to load vectorizer from '...'. File may be corrupted.
#
# Common causes include:
#   - The file is not a valid pickle (e.g., it's empty or not a pickle file)
#   - The file was saved by a different Python version or environment
#   - The file is incomplete due to interrupted writing
#
# Recommended steps:
#   1. Check that the file exists and is not empty:
#        import os
#        print(os.path.getsize(vectorizer_path))
#   2. Recreate the model/vectorizer artifacts using the same Python version
#      and libraries as during training.
#   3. Confirm you are pointing to the correct file paths.
#   4. If possible, open the pickle file in a Python shell:
#        import pickle
#        with open(r"C:\Users\DABC\Desktop\Ninja\demo_pipeline_output\tfidf_vectorizer.pkl", "rb") as f:
#            obj = pickle.load(f)
#      and observe if the error reproduces.
#   5. If the problem persists, retrain the pipeline and export new .pkl files.

# The _load_pickle function already provides detailed error messages for these cases.


from __future__ import annotations
import os
import pickle
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import pandas as pd
import shap
import ollama
import yaml
from sklearn.metrics import classification_report


# =============================================================================
# Data Structure
# =============================================================================

@dataclass
class Explanation:
    """Holds explainability results for a single prediction."""
    text: str
    risk_level: str
    confidence: float
    top_features: List[Tuple[str, float]]
    narrative: str


# =============================================================================
# Main Explainability Routine
# =============================================================================

def explain_and_evaluate(
    model_path: str,
    vectorizer_path: str,
    test_path: str,
    text_column: str = "text",
    label_column: Optional[str] = "label",
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> List[Explanation]:
    """
    Run predictions, SHAP explainability, TinyLlama narratives,
    and model performance evaluation.
    """
    

    # --- Load vectorizer ----------------------------------------------------------
    vectorizer = _load_pickle(vectorizer_path, "vectorizer")

    # --- Load model ---------------------------------------------------------------
    model = _load_pickle(model_path, "model")

    # --- Load test dataset --------------------------------------------------------
    test_df = pd.read_csv(test_path)
    X_test = vectorizer.transform(test_df[text_column])

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    # Map numeric predictions to human-readable risk labels
    risk_labels = {0: "Low", 1: "Medium", 2: "High"}
    pred_labels = [risk_labels.get(p, "Unknown") for p in preds]

    # --- Evaluate model performance ----------------------------------------------
    if label_column and label_column in test_df.columns:
        y_true = [
            risk_labels.get(y, "Unknown") if isinstance(y, int) else y
            for y in test_df[label_column]
        ]
        print("\n=== MODEL PERFORMANCE REPORT ===")
        print(classification_report(y_true, pred_labels, digits=3))
        print("=================================\n")

    # --- Compute SHAP explanations -----------------------------------------------
    print("Computing SHAP explanations... (this may take a moment)")
    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(X_test)
    feature_names = vectorizer.get_feature_names_out()

    # --- Generate explanations ----------------------------------------------------
    explanations: List[Explanation] = []

    for idx, row in test_df.iterrows():
        text = row[text_column]
        pred_class = preds[idx]
        confidence = probs[idx][pred_class]
        shap_values = shap_values_all[pred_class][idx]

        top_features = summarize_shap_values(shap_values, feature_names)
        prompt_template = _load_prompt_template(config_path)

        prompt = prompt_template.format(
            risk_level=risk_labels[pred_class],
            confidence=round(confidence * 100, 2),
            top_features=", ".join(f"{f} ({round(v, 2)})" for f, v in top_features),
            context=text,
        )

        narrative = _generate_narrative_with_tinyllama(prompt)

        explanation = Explanation(
            text=text,
            risk_level=risk_labels[pred_class],
            confidence=confidence,
            top_features=top_features,
            narrative=narrative,
        )

        explanations.append(explanation)

        # Log progress
        print(f"\n[{idx}] Text: {text}")
        print(f"  → Predicted Risk: {risk_labels[pred_class]} ({confidence:.2%})")
        print(f"  → Top Features: {top_features}")
        print(f"  → Narrative: {narrative}\n")

    # --- Export results -----------------------------------------------------------
    if output_path:
        save_explanations_to_csv(explanations, output_path)
        print(f"\nResults exported to: {output_path}")

    return explanations


# =============================================================================
# Helper Utilities
# =============================================================================

def _load_pickle(path: str, name: str):
    """Load a pickled object safely with clear error messages."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except pickle.UnpicklingError as e:
        raise RuntimeError(f"Failed to load {name} from '{path}'. File may be corrupted.") from e
    except FileNotFoundError:
        raise RuntimeError(f"{name.capitalize()} file not found: {path}")


def summarize_shap_values(
    shap_values: Sequence[float],
    feature_names: Sequence[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Return the top-k SHAP features ranked by absolute contribution."""
    ranked = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)
    return ranked[:top_k]


def _load_prompt_template(config_path: Optional[str]) -> str:
    """Load or provide a default LLM prompt template."""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            if "mistral_prompt_template" in config:
                return config["mistral_prompt_template"]

    # Default template
    return (
        "You are an experienced business risk analyst.\n"
        "Prediction: {risk_level} risk ({confidence}%).\n"
        "Top contributing features: {top_features}.\n"
        "Context: {context}\n"
        "Explain in a concise, executive tone why this risk level was predicted."
    )


def _generate_narrative_with_tinyllama(prompt: str) -> str:
    """Generate a concise business narrative using TinyLlama."""
    try:
        response = ollama.chat(
            model="tinyllama",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior business analyst. "
                        "Explain model predictions clearly for executives and risk officers."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"(Fallback narrative: TinyLlama error - {e})"


def save_explanations_to_csv(explanations: List[Explanation], path: str):
    """Export all explanations (predictions + SHAP + narrative) to a CSV file."""
    rows = [
        {
            "text": exp.text,
            "risk_level": exp.risk_level,
            "confidence": round(exp.confidence * 100, 2),
            "top_features": ", ".join(f"{f} ({round(v, 2)})" for f, v in exp.top_features),
            "narrative": exp.narrative,
        }
        for exp in explanations
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

import pandas as pd
import numpy as np

NUMERIC_FEATURES = [
    "annual_revenue",
    "annual_spend",
    "avg_payment_delay_days",
    "contract_value",
    "contract_duration_months",
    "past_disputes",
    "delivery_score",
    "financial_stability_index",
    "relationship_years",
    "txn_count",
    "avg_txn_amount",
    "avg_delay",
    "late_ratio",
    "dispute_rate",
    "avg_delivery_quality",
    "clause_risk_score",
]

TEXT_FEATURES = [
    "30",
    "ambiguous",
    "annual",
    "causing",
    "clause",
    "clear",
    "contract",
    "days",
    "delayed",
    "delays",
    "delivery",
    "included",
    "long",
    "material",
    "net",
    "notified",
    "partnership",
    "payment",
    "penalty",
    "raw",
    "reviews",
    "shortages",
    "standard",
    "supplier",
    "term",
    "termination",
    "terms",
]

CLASS_LABELS = ["high", "medium", "low"]

np.random.seed(42)
single_data = {
    "annual_revenue": 2_345_678,
    "annual_spend": 1_500_000,
    "avg_payment_delay_days": 22,
    "contract_value": 350_000,
    "contract_duration_months": 36,
    "past_disputes": 3,
    "delivery_score": 85,
    "financial_stability_index": 0.72,
    "relationship_years": 5,
    "txn_count": 24,
    "avg_txn_amount": 12_500,
    "avg_delay": 4,
    "late_ratio": 0.19,
    "dispute_rate": 0.08,
    "avg_delivery_quality": 88,
    "clause_risk_score": 41,
    "contract_text": "Penalty clause included for delayed delivery. Payment terms: net 30 days. Supplier notified raw material shortages causing delays.",
    "risk_label": "medium",
}

df = pd.DataFrame([single_data])

assert all(f in df.columns for f in NUMERIC_FEATURES), "Missing numeric features"
assert "contract_text" in df.columns, "Missing contract_text feature"
assert "risk_label" in df.columns, "Missing risk_label"
print(df)

# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    explanations = explain_and_evaluate(
        model_path=r"C:\Users\DABC\Desktop\ai_procurment_manager\risk_prediction_system_production_demo\models\rf_model.pkl",
        vectorizer_path=r"C:\Users\DABC\Desktop\ai_procurment_manager\risk_prediction_system_production_demo\models\tfidf_vectorizer.pkl",
        test_path=r"C:\Users\DABC\Desktop\ai_procurment_manager\risk_prediction_system_production_demo\data\holdout_labels.csv",
        text_column="text",
        label_column="label",
        output_path=r"C:\Users\DABC\Desktop\ai_procurment_manager\risk_prediction_system_production_demo\reports\explanation_report.csv",
    )

