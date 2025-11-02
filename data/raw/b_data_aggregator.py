"""Data aggregation module for supplier risk prediction system.

This module combines disparate data sources (suppliers, transactions, contracts)
into a unified dataset for model training and prediction.
"""

from __future__ import annotations

import pandas as pd


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
