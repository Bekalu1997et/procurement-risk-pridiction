"""Test data pipeline functions."""

import sys
from pathlib import Path

import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src import data_pipeline


def test_load_processed_datasets():
    """Test loading processed datasets."""
    training_df, weekly_df = data_pipeline.load_processed_datasets()
    assert isinstance(training_df, pd.DataFrame)
    assert isinstance(weekly_df, pd.DataFrame)
    assert not training_df.empty


def test_clean_columns():
    """Test column cleaning function."""
    df = pd.DataFrame({"col 1": [1, 2], "col-2": [3, 4]})
    cleaned = data_pipeline.clean_columns(df)
    assert "col_1" in cleaned.columns
    assert "col_2" in cleaned.columns


def test_prepare_training_data():
    """Test training data preparation."""
    X, y = data_pipeline.prepare_training_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
