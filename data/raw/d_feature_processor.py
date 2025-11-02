"""Feature processing module for supplier risk prediction system.

This module handles the processing of numerical and categorical features
for model training and prediction.
"""

from __future__ import annotations

from typing import List


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


def build_text_feature_list() -> List[str]:
    """Return the text feature list for contract analysis."""
    
    return ["contract_text"]


def build_categorical_feature_list() -> List[str]:
    """Return the categorical feature list for encoding."""
    
    return [
        "region",
        "industry", 
        "contract_criticality",
    ]
