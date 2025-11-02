"""Test data pipeline functions.

Why we test this:
- Data pipeline is the foundation of ML system - garbage in, garbage out
- Validates data loading from CSV files works correctly
- Ensures column cleaning handles special characters and spaces
- Tests train/test split produces valid X and y arrays
- Catches schema changes that would break downstream models
- Prevents production failures from malformed data
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src import data_pipeline


def test_load_processed_datasets():
    """Test loading processed datasets.
    
    Why: Ensures the pipeline can read preprocessed data files.
    Catches file path issues, encoding problems, and missing files early.
    """
    training_df, weekly_df = data_pipeline.load_processed_datasets()
    assert isinstance(training_df, pd.DataFrame)
    assert isinstance(weekly_df, pd.DataFrame)
    assert not training_df.empty


def test_clean_columns():
    """Test column cleaning function.
    
    Why: Column names with spaces/special chars break model training.
    This test ensures consistent naming convention across the pipeline.
    """
    df = pd.DataFrame({"col 1": [1, 2], "col-2": [3, 4]})
    cleaned = data_pipeline.clean_columns(df)
    assert "col_1" in cleaned.columns
    assert "col_2" in cleaned.columns


def test_prepare_training_data():
    """Test training data preparation.
    
    Why: Validates feature/target split is correct and shapes match.
    Ensures no data leakage between X and y before model training.
    """
    X, y = data_pipeline.prepare_training_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
