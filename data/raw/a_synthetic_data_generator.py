"""Synthetic data generation module for supplier risk prediction system.

This module generates synthetic supplier profiles, transactions, and contract texts
for training and testing the risk prediction model.
"""

from __future__ import annotations

import random
from typing import Dict, List

import pandas as pd
try:  # pragma: no cover - optional dependency
    from faker import Faker
    HAS_FAKER = True
except ImportError:  # pragma: no cover - fallback path
    Faker = None  # type: ignore
    HAS_FAKER = False


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


class SimpleFaker:
    """Minimal fallback that mimics the Faker interface used in the pipeline."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def company(self) -> str:
        return f"Supplier {self._rng.randint(1000, 9999)}"

    def date_between(self, start_date: str, end_date: str) -> pd.Timestamp:
        days_back = self._rng.randint(0, 365)
        return (pd.Timestamp.utcnow() - pd.to_timedelta(days_back, unit="D")).normalize()


def _seed_generators(seed: int) -> object:
    """Seed Python, NumPy, and Faker PRNGs for reproducibility."""
    import numpy as np
    
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


def simulate_weekly_drift(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Create a weekly scoring dataset by adding light distributional drift."""
    import numpy as np
    
    _seed_generators(seed + 21)
    weekly = df.copy()
    weekly["late_ratio"] = np.clip(weekly["late_ratio"] + np.random.normal(0, 0.05, len(weekly)), 0, 1)
    weekly["dispute_rate"] = np.clip(weekly["dispute_rate"] + np.random.normal(0, 0.03, len(weekly)), 0, 1)
    weekly["avg_delay"] = np.maximum(0, weekly["avg_delay"] + np.random.normal(0, 2, len(weekly)))
    weekly["snapshot_week"] = pd.Timestamp.utcnow().isocalendar().week
    weekly["created_ts"] = pd.Timestamp.utcnow()
    return weekly
