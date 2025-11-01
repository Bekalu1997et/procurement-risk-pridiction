"""Synthetic data generation and baseline modelling pipeline for the demo system.

This module adapts the user-provided supplier risk script into a reusable, testable
component that integrates with the broader `risk_prediction_system_production_demo`
codebase.  The pipeline generates synthetic supplier, transaction, and contract
data, engineers aggregate features, trains a RandomForest model with TF-IDF text
features, and seeds downstream artefacts (CSV datasets, models, audit logs, and
sample predictions).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
try:  # pragma: no cover - optional dependency
    from faker import Faker
    HAS_FAKER = True
except ImportError:  # pragma: no cover - fallback path
    Faker = None  # type: ignore
    HAS_FAKER = False
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# Ensure the project root is available on the PYTHONPATH when the module is
# executed as a script (e.g. `python data/raw/demo_pipeline.py`).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import auditing  # pylint: disable=wrong-import-position


REGIONS = ["North America", "Europe", "Asia-Pacific", "LATAM"]
INDUSTRIES = ["Manufacturing", "Logistics", "IT Services", "Facilities", "Biotech"]
CONTRACT_CRITICALITY = ["High", "Medium", "Low"]
CONTRACT_PHRASES = [
    "Penalty clause included for delayed delivery.",
    "Payment terms: net 30 days.",
    "No clear penalty clause; termination ambiguous.",
    "Supplier notified raw material shortages causing delays.",
    "Long-term partnership clause with annual reviews.",
]

DEFAULT_SUPPLIERS = 10500
DEFAULT_TRANSACTIONS = 5000
DEFAULT_RANDOM_STATE = 42
SAMPLE_PREDICTION_COUNT = 5


class SimpleFaker:
    """Minimal fallback that mimics the Faker interface used in the pipeline."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def company(self) -> str:
        return f"Supplier {self._rng.randint(1000, 9999)}"

    def date_between(self, start_date: str, end_date: str) -> pd.Timestamp:
        days_back = self._rng.randint(0, 365)
        return (pd.Timestamp.utcnow() - pd.to_timedelta(days_back, unit="D")).normalize()


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


def _seed_generators(seed: int) -> object:
    """Seed Python, NumPy, and Faker PRNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    if HAS_FAKER and Faker is not None:
        faker = Faker()
        faker.seed_instance(seed)
        return faker
    return SimpleFaker(seed)


def generate_supplier_profiles(num_suppliers: int, seed: int) -> pd.DataFrame:
    """Create supplier master data with financial and categorical attributes."""

    faker = _seed_generators(seed)
    supplier_rows: List[Dict[str, object]] = []
    for idx in range(num_suppliers):
        annual_revenue = random.randint(50_000, 5_000_000)
        financial_stability = round(random.uniform(0.2, 1.0), 2)
        supplier_rows.append(
            {
                "supplier_id": f"S_{idx + 1:05}",
                "company_name": faker.company(),
                "region": random.choice(REGIONS),
                "industry": random.choice(INDUSTRIES),
                "annual_revenue": annual_revenue,
                "annual_spend": int(annual_revenue * random.uniform(0.4, 0.9)),
                "avg_payment_delay_days": random.randint(0, 90),
                "contract_value": random.randint(10_000, 2_000_000),
                "contract_duration_months": random.randint(6, 60),
                "past_disputes": random.randint(0, 12),
                "delivery_score": random.randint(50, 100),
                "financial_stability_index": financial_stability,
                "relationship_years": random.randint(0, 10),
                "contract_criticality": random.choices(
                    CONTRACT_CRITICALITY, weights=[0.3, 0.5, 0.2], k=1
                )[0],
                "credit_score": int(500 + financial_stability * 350),
                "created_ts": pd.Timestamp.utcnow(),
            }
        )
    suppliers = pd.DataFrame(supplier_rows)
    return suppliers


def generate_transactions(
    suppliers: pd.DataFrame,
    num_transactions: int,
    seed: int,
) -> pd.DataFrame:
    """Generate transactional history for suppliers to enable aggregation."""

    faker = _seed_generators(seed + 7)
    supplier_ids = suppliers["supplier_id"].tolist()
    rows: List[Dict[str, object]] = []
    for idx in range(num_transactions):
        payment_delay = random.randint(0, 90)
        on_time_delivery = int(payment_delay < 10)
        rows.append(
            {
                "transaction_id": f"T_{idx + 1:06}",
                "supplier_id": random.choice(supplier_ids),
                "buyer_company": faker.company(),
                "transaction_date": faker.date_between(start_date="-1y", end_date="today"),
                "payment_amount": random.randint(1_000, 50_000),
                "payment_delay_days": payment_delay,
                "delivery_quality_score": random.randint(50, 100),
                "contract_reference": f"C_{random.randint(100, 999)}",
                "product_category": random.choice(
                    ["IT Equipment", "Office Supplies", "Medical Devices", "Raw Materials", "Logistics"]
                ),
                "dispute_flag": int(random.random() < 0.1),
                "on_time_delivery": on_time_delivery,
            }
        )
    transactions = pd.DataFrame(rows)
    return transactions


def generate_contract_texts(suppliers: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign synthetic contract narratives and clause risk scores."""

    _seed_generators(seed + 13)
    contract_rows: List[Dict[str, object]] = []
    for supplier_id in suppliers["supplier_id"]:
        phrases = [phrase for phrase in CONTRACT_PHRASES if random.random() < 0.5]
        if not phrases:
            phrases = ["Standard contract with net 30 payment terms."]
        text = " ".join(phrases)
        clause_risk = 30.0
        if "ambiguous" in text.lower():
            clause_risk += 25.0
        if "Penalty clause" in text:
            clause_risk -= 10.0
        if "raw material shortages" in text.lower():
            clause_risk += 15.0
        clause_risk = max(5.0, min(95.0, clause_risk + random.uniform(-5, 5)))
        contract_rows.append(
            {
                "supplier_id": supplier_id,
                "contract_text": text,
                "clause_risk_score": round(clause_risk, 2),
            }
        )
    contracts = pd.DataFrame(contract_rows)
    return contracts


def aggregate_supplier_features(
    suppliers: pd.DataFrame,
    transactions: pd.DataFrame,
    contracts: pd.DataFrame,
) -> pd.DataFrame:
    """Combine disparate sources into a single modelling dataframe."""

    aggregated = transactions.groupby("supplier_id").agg(
        txn_count=("transaction_id", "count"),
        avg_txn_amount=("payment_amount", "mean"),
        avg_txn_delay=("payment_delay_days", "mean"),
        late_txn_ratio=("on_time_delivery", lambda series: 1 - float(series.mean())),
        dispute_ratio=("dispute_flag", "mean"),
        avg_delivery_quality=("delivery_quality_score", "mean"),
    )
    merged = (
        suppliers.merge(aggregated, on="supplier_id", how="left")
        .merge(contracts, on="supplier_id", how="left")
        .fillna(
            {
                "txn_count": 0,
                "avg_txn_amount": 0,
                "avg_txn_delay": suppliers["avg_payment_delay_days"],
                "late_txn_ratio": 0,
                "dispute_ratio": 0,
                "avg_delivery_quality": suppliers["delivery_score"],
                "contract_text": "Standard contract with net 30 payment terms.",
                "clause_risk_score": 40.0,
            }
        )
    )
    merged.rename(
        columns={
            "late_txn_ratio": "late_ratio",
            "dispute_ratio": "dispute_rate",
            "avg_txn_delay": "avg_delay",
        },
        inplace=True,
    )
    return merged


def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Derive heuristic risk scores and categorical labels."""

    def score_row(row: pd.Series) -> Tuple[float, str]:
        score = 0.0
        if row["avg_payment_delay_days"] > 45:
            score += 2
        if row["past_disputes"] > 5:
            score += 2
        if row["financial_stability_index"] < 0.5:
            score += 2
        if row["late_ratio"] > 0.3:
            score += 1
        if row["dispute_rate"] > 0.2:
            score += 1
        if row["clause_risk_score"] > 60:
            score += 1
        if row["avg_delivery_quality"] < 70:
            score += 1
        risk_score = round(min(1.0, score / 7), 3)
        if score >= 5:
            label = "high"
        elif score >= 3:
            label = "medium"
        else:
            label = "low"
        return risk_score, label

    risks = df.apply(score_row, axis=1, result_type="expand")
    df["risk_score"] = risks[0]
    df["risk_label"] = risks[1].astype(str)
    return df


def build_numeric_feature_list() -> List[str]:
    """Return the numeric feature list utilised for the demo model."""

    return [
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


def train_random_forest_model(
    data: pd.DataFrame,
    numeric_features: Sequence[str],
    random_state: int,
) -> Tuple[RandomForestClassifier, TfidfVectorizer, Dict[str, object]]:
    """Train a RandomForest classifier using numeric + TF-IDF text features."""

    vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
    text_matrix = vectorizer.fit_transform(data["contract_text"].astype(str))
    numeric_matrix = data.loc[:, numeric_features].fillna(0).to_numpy()
    feature_matrix = hstack([numeric_matrix, text_matrix])
    labels = data["risk_label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=random_state,
    )

    model = RandomForestClassifier(n_estimators=150, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    class_labels = model.classes_.tolist()
    y_test_bin = label_binarize(y_test, classes=class_labels)
    auc_macro = float(roc_auc_score(y_test_bin, y_proba, multi_class="ovr"))

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    feature_names = numeric_features + vectorizer.get_feature_names_out().tolist()
    importances = model.feature_importances_
    top_features = [
        {"feature": name, "importance": float(value)}
        for name, value in sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)[:15]
    ]

    summary: Dict[str, object] = {
        "auc_macro": auc_macro,
        "classification_report": report,
        "top_importances": top_features,
        "class_labels": class_labels,
    }
    return model, vectorizer, summary


def simulate_weekly_drift(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Create a weekly scoring dataset by adding light distributional drift."""

    _seed_generators(seed + 21)
    weekly = df.copy()
    weekly["late_ratio"] = np.clip(weekly["late_ratio"] + np.random.normal(0, 0.05, len(weekly)), 0, 1)
    weekly["dispute_rate"] = np.clip(weekly["dispute_rate"] + np.random.normal(0, 0.03, len(weekly)), 0, 1)
    weekly["avg_delay"] = np.maximum(0, weekly["avg_delay"] + np.random.normal(0, 2, len(weekly)))
    weekly["snapshot_week"] = pd.Timestamp.utcnow().isocalendar().week
    weekly["created_ts"] = pd.Timestamp.utcnow()
    return weekly


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
    """Main orchestration function used by scripts and tests."""

    suppliers = generate_supplier_profiles(num_suppliers, random_state)
    transactions = generate_transactions(suppliers, num_transactions, random_state)
    contracts = generate_contract_texts(suppliers, random_state)

    merged = aggregate_supplier_features(suppliers, transactions, contracts)
    merged = compute_risk_scores(merged)

    numeric_features = build_numeric_feature_list()
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

    parser = argparse.ArgumentParser(description="Generate supplier risk demo data")
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
        "Synthetic demo data generated successfully — "
        f"{len(artifacts.suppliers)} suppliers, {len(artifacts.transactions)} transactions."
    )
    print(
        "RandomForest macro AUC:",
        round(artifacts.model_summary["summary"]["auc_macro"], 3),
    )


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
