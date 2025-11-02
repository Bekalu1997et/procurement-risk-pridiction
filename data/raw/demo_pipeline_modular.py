"""Modular synthetic data generation and baseline modelling pipeline for the demo system.

This module uses the refactored modular components to generate synthetic supplier,
transaction, and contract data, engineers aggregate features, trains a RandomForest
model with TF-IDF text features, and seeds downstream artefacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import pandas as pd

# Ensure the project root is available on the PYTHONPATH when the module is
# executed as a script (e.g. `python data/raw/demo_pipeline_modular.py`).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import auditing  # pylint: disable=wrong-import-position

# Import the new modular components
from data.raw.synthetic_data_generator import (
    generate_supplier_profiles,
    generate_transactions,
    generate_contract_texts,
    simulate_weekly_drift,
)
from data.raw.data_aggregator import aggregate_supplier_features
from data.raw.risk_computer import compute_risk_scores
from data.raw.feature_processor import build_numeric_feature_list
from data.raw.trainer import train_random_forest_model
from data.raw.evaluator import predict_supplier_risk


DEFAULT_SUPPLIERS = 10500
DEFAULT_TRANSACTIONS = 5000
DEFAULT_RANDOM_STATE = 42
SAMPLE_PREDICTION_COUNT = 5


class PipelineArtifacts:
    """Simple container summarising artefacts produced by the pipeline."""

    def __init__(
        self,
        suppliers: pd.DataFrame,
        transactions: pd.DataFrame,
        contracts: pd.DataFrame,
        merged: pd.DataFrame,
        numeric_features: List[str],
        text_features: List[str],
        model_summary: Dict[str, object],
    ) -> None:
        self.suppliers = suppliers
        self.transactions = transactions
        self.contracts = contracts
        self.merged = merged
        self.numeric_features = numeric_features
        self.text_features = text_features
        self.model_summary = model_summary


def _directories(output_root: Optional[Path]) -> Dict[str, Path]:
    """Resolve key project directories, creating them if required."""

    root = output_root or PROJECT_ROOT
    data_processed = root / "data" / "processed"
    models_dir = root / "models"
    reports_dir = root / "reports"
    audit_dir = reports_dir / "audit_logs"
    weekly_dir = reports_dir / "weekly_reports"
    shap_dir = reports_dir / "shap_summary_plots"

    data_processed.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    weekly_dir.mkdir(parents=True, exist_ok=True)
    shap_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    return {
        "data_processed": data_processed,
        "models": models_dir,
        "weekly_reports": weekly_dir,
        "audit_logs": audit_dir,
        "shap_reports": shap_dir,
    }


def save_artifacts(
    artifacts: PipelineArtifacts,
    directories: Dict[str, Path],
    weekly_df: pd.DataFrame,
    random_state: int,
) -> None:
    """Persist datasets, models, and supporting metadata to disk."""

    data_dir = directories["data_processed"]
    models_dir = directories["models"]
    weekly_dir = directories["weekly_reports"]

    artifacts.suppliers.to_csv(data_dir / "suppliers.csv", index=False)
    artifacts.transactions.to_csv(data_dir / "transactions.csv", index=False)
    artifacts.contracts.to_csv(data_dir / "contracts.csv", index=False)
    artifacts.merged.to_csv(data_dir / "merged_training.csv", index=False)
    weekly_df.to_csv(data_dir / "new_data_weekly.csv", index=False)

    joblib.dump(artifacts.model_summary["model"], models_dir / "rf_model.pkl")
    joblib.dump(artifacts.model_summary["vectorizer"], models_dir / "tfidf_vectorizer.pkl")

    feature_bundle = {
        "numeric_features": artifacts.numeric_features,
        "text_features": artifacts.text_features,
        "class_labels": artifacts.model_summary["summary"]["class_labels"],
        "random_state": random_state,
    }
    with open(models_dir / "feature_bundle.json", "w", encoding="utf-8") as handle:
        json.dump(feature_bundle, handle, indent=2)

    training_summary = artifacts.model_summary["summary"].copy()
    training_summary.pop("classification_report", None)
    with open(models_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(training_summary, handle, indent=2)

    sample_rows = artifacts.merged.sample(
        min(SAMPLE_PREDICTION_COUNT, len(artifacts.merged)), random_state=random_state
    )
    predictions: List[Dict[str, object]] = []
    for row in sample_rows.to_dict(orient="records"):
        enriched = predict_supplier_risk(
            row,
            model=artifacts.model_summary["model"],
            vectorizer=artifacts.model_summary["vectorizer"],
            numeric_features=artifacts.numeric_features,
            models_dir=models_dir,
        )
        enriched["supplier_id"] = row["supplier_id"]
        enriched["company_name"] = row["company_name"]
        predictions.append(enriched)
    pd.DataFrame(predictions).to_csv(models_dir / "baseline_sample_predictions.csv", index=False)

    # Save a lightweight weekly report stub for dashboards.
    weekly_stub = weekly_df[["supplier_id", "risk_label", "risk_score", "late_ratio", "dispute_rate"]]
    weekly_stub.to_csv(weekly_dir / f"weekly_predictions_{pd.Timestamp.utcnow().date()}.csv", index=False)

    auditing.persist_audit_log(
        event_type="demo_pipeline_artifacts_saved",
        payload={
            "records": int(len(artifacts.merged)),
            "weekly_records": int(len(weekly_df)),
            "numeric_features": artifacts.numeric_features,
        },
    )


def run_pipeline(
    *,
    num_suppliers: int = DEFAULT_SUPPLIERS,
    num_transactions: int = DEFAULT_TRANSACTIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
    output_root: Optional[Path] = None,
) -> PipelineArtifacts:
    """Main orchestration function using modular components."""

    # Generate synthetic data using the data generator module
    suppliers = generate_supplier_profiles(num_suppliers, random_state)
    transactions = generate_transactions(suppliers, num_transactions, random_state)
    contracts = generate_contract_texts(suppliers, random_state)

    # Aggregate data using the data aggregator module
    merged = aggregate_supplier_features(suppliers, transactions, contracts)
    
    # Compute risk scores using the risk computer module
    merged = compute_risk_scores(merged)

    # Get feature lists using the feature processor module
    numeric_features = build_numeric_feature_list()
    
    # Train model using the trainer module
    model, vectorizer, summary = train_random_forest_model(merged, numeric_features, random_state)
    text_features = vectorizer.get_feature_names_out().tolist()

    directories = _directories(output_root)
    summary_payload = {"model": model, "vectorizer": vectorizer, "summary": summary}
    artifacts = PipelineArtifacts(
        suppliers=suppliers,
        transactions=transactions,
        contracts=contracts,
        merged=merged,
        numeric_features=list(numeric_features),
        text_features=text_features,
        model_summary=summary_payload,
    )

    weekly_df = simulate_weekly_drift(merged, random_state)
    save_artifacts(artifacts, directories, weekly_df, random_state)

    auditing.persist_audit_log(
        event_type="demo_pipeline_completed",
        payload={
            "suppliers": num_suppliers,
            "transactions": num_transactions,
            "class_labels": summary["class_labels"],
            "auc_macro": summary["auc_macro"],
        },
    )

    return artifacts


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the demo pipeline script."""

    parser = argparse.ArgumentParser(description="Generate supplier risk demo data (modular version)")
    parser.add_argument("--suppliers", type=int, default=DEFAULT_SUPPLIERS, help="Number of suppliers to generate")
    parser.add_argument(
        "--transactions", type=int, default=DEFAULT_TRANSACTIONS, help="Number of synthetic transactions"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed for reproducibility")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional override for the project root when generating artefacts",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entrypoint for CLI execution."""

    args = parse_args(argv)
    artifacts = run_pipeline(
        num_suppliers=args.suppliers,
        num_transactions=args.transactions,
        random_state=args.seed,
        output_root=args.output_root,
    )

    print(
        "Synthetic demo data generated successfully (modular version) — "
        f"{len(artifacts.suppliers)} suppliers, {len(artifacts.transactions)} transactions."
    )
    print(
        "RandomForest macro AUC:",
        round(artifacts.model_summary["summary"]["auc_macro"], 3),
    )


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
