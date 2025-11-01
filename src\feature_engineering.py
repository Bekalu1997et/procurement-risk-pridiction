"""Feature engineering utilities for supplier risk analytics.

The functions below transform raw transaction and contract data into
model-ready features, including categorical buckets and NLP embeddings.  Each
step records metadata using the auditing pipeline so stakeholders can trace the
provenance of engineered features.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from . import auditing


LOGGER = logging.getLogger(__name__)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def calculate_behavioral_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction data into supplier-level behavioural metrics."""

    aggregated = transactions.groupby("supplier_id").agg(
        late_txn_ratio=("late_payment", "mean"),
        dispute_rate=("dispute_flag", "mean"),
        avg_delay_days=("delay_days", "mean"),
        invoice_volume=("invoice_amount", "sum"),
    )

    aggregated = aggregated.fillna(0)
    aggregated["payment_delay_category"] = pd.cut(
        aggregated["avg_delay_days"],
        bins=[-np.inf, 3, 10, np.inf],
        labels=["low", "medium", "high"],
    )

    auditing.persist_audit_log(
        event_type="feature_engineering_transactions",
        payload={
            "columns": aggregated.columns.tolist(),
            "rows": int(aggregated.shape[0]),
        },
    )
    return aggregated.reset_index()


@lru_cache(maxsize=1)
def _load_embedding_components() -> Optional[tuple[AutoTokenizer, AutoModel]]:
    """Lazy-load the transformer model and tokenizer for contract embeddings."""

    try:
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
        return tokenizer, model
    except OSError as exc:  # pragma: no cover - depends on external downloads.
        LOGGER.warning(
            "Falling back to hash-based contract features because model %s could not be loaded: %s",
            EMBEDDING_MODEL_NAME,
            exc,
        )
        return None


def _mean_pool(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Perform mean pooling on token embeddings respecting the attention mask."""

    mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    return torch.sum(model_output * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)


def generate_contract_embeddings(contracts: pd.DataFrame) -> pd.DataFrame:
    """Create dense embeddings for contract text using a miniature transformer.

    The embedding step falls back to hashed feature vectors when GPU / model
    weights are unavailable, ensuring the demo remains portable.
    """

    components = _load_embedding_components()
    if components is None:
        hashed = pd.util.hash_pandas_object(contracts["contract_terms"].astype(str)).astype(float)
        fallback = pd.DataFrame(
            {
                "contract_embedding_0": hashed,
                "supplier_id": contracts["supplier_id"].values,
            }
        )
        auditing.persist_audit_log(
            event_type="feature_engineering_contracts",
            payload={
                "embedding_dim": 1,
                "rows": int(fallback.shape[0]),
                "mode": "hash_fallback",
            },
        )
        return fallback

    tokenizer, model = components
    texts = contracts["contract_terms"].astype(str).tolist()
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        model_output = model(**encoded)
    sentence_embeddings = _mean_pool(model_output.last_hidden_state, encoded["attention_mask"]).numpy()

    embedding_df = pd.DataFrame(
        sentence_embeddings,
        columns=[f"contract_embedding_{i}" for i in range(sentence_embeddings.shape[1])],
    )
    embedding_df["supplier_id"] = contracts["supplier_id"].values

    auditing.persist_audit_log(
        event_type="feature_engineering_contracts",
        payload={
            "embedding_dim": int(sentence_embeddings.shape[1]),
            "rows": int(embedding_df.shape[0]),
        },
    )
    return embedding_df


def build_feature_dataframe(
    suppliers: pd.DataFrame,
    transactions: pd.DataFrame,
    contracts: pd.DataFrame,
) -> pd.DataFrame:
    """Combine raw tables into a model-ready feature dataframe."""

    behavioural = calculate_behavioral_features(transactions)
    embeddings = generate_contract_embeddings(contracts)

    feature_df = suppliers.merge(behavioural, on="supplier_id", how="left").merge(
        embeddings, on="supplier_id", how="left"
    )

    feature_df = feature_df.fillna({"late_txn_ratio": 0.0, "dispute_rate": 0.0, "avg_delay_days": 0.0})

    auditing.persist_audit_log(
        event_type="feature_dataframe_compiled",
        payload={
            "rows": int(feature_df.shape[0]),
            "columns": feature_df.columns.tolist(),
        },
    )
    return feature_df

