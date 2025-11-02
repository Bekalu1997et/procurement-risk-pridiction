"""
Feature engineering utilities for supplier risk analytics.

Feature engineering here is like creating new columns from raw transactional or contract data that capture risky supplier behavior. 
For example, "delayed ratio" means: for all transactions, how many are delayed divided by total transactions. 
Features like "mean delayed days" is the average time a supplier delays payment, and you can also compute how far actual transaction end date is from contract end date. 
These patterns are all useful features that tell our trained model about supplier risk in ways that are more informative than the raw data itself.

Even though we have an embedding model (pre-trained, used for extracting text features from contracts), we still do this numeric feature engineering, 
because models learn much better when they get clear, domain-informed behaviors—like ratios, averages, or category buckets—rather than just the raw data. 
So: embedding is for unstructured text (e.g., contract language), but engineered features like ratios and means (e.g., late transaction ratio, avg delay days) 
are crucial for quantifying behavioral risk. Both types go into our final model.

All steps record feature transformation metadata using the auditing pipeline, so stakeholders can always trace how engineered features were created.
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
    """Aggregate transaction data into supplier-level behavioural metrics.
    The function aggregates the transaction data into supplier-level behavioural metrics.
    It uses the pandas groupby function to aggregate the data.
    It also logs the feature engineering of the transactions using the auditing module.
    """

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
    """Lazy-load the transformer model and tokenizer for contract embeddings.
    The function lazy-loads the transformer model and tokenizer for contract embeddings.
    It uses the AutoTokenizer and AutoModel from the transformers library to load the model and tokenizer.
    It also logs the feature engineering of the contracts using the auditing module.
    """

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
    """Perform mean pooling on token embeddings respecting the attention mask.
    The function performs mean pooling on token embeddings respecting the attention mask.
    It uses the torch.sum and torch.clamp functions to perform the mean pooling.
    It also logs the feature engineering of the contracts using the auditing module.
    """

    mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    return torch.sum(model_output * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)


def generate_contract_embeddings(contracts: pd.DataFrame) -> pd.DataFrame:
    """Create dense embeddings for contract text using a miniature transformer.

    The embedding step falls back to hashed feature vectors when GPU / model
    weights are unavailable, ensuring the demo remains portable.
    The function creates dense embeddings for contract text using a miniature transformer.
    It uses the AutoTokenizer and AutoModel from the transformers library to load the model and tokenizer.
    It also logs the feature engineering of the contracts using the auditing module.
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
    """Combine raw tables into a model-ready feature dataframe.
    The function combines the raw tables into a model-ready feature dataframe.
    It uses the pandas merge function to combine the tables.
    It also fills the missing values with 0.0.
    It also logs the feature engineering of the dataframe using the auditing module.
    """

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

