"""Tests for the ETL pipeline (data/process_data.py)."""

import os
import sys
import tempfile

import pandas as pd
import pytest

# Add project root and data directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

from data.process_data import load_data, clean_data, save_data


@pytest.fixture
def sample_csv_files(tmp_path):
    """Create small sample CSV files for testing."""
    messages_data = pd.DataFrame({
        'id': [1, 2, 3],
        'message': ['We need water', 'Help us please', 'Earthquake damage'],
        'original': ['We need water', 'Help us please', 'Earthquake damage'],
        'genre': ['direct', 'news', 'social']
    })
    categories_data = pd.DataFrame({
        'id': [1, 2, 3],
        'categories': [
            'related-1;request-1;offer-0',
            'related-1;request-0;offer-0',
            'related-2;request-1;offer-0'  # includes a 2 to test clipping
        ]
    })

    messages_path = str(tmp_path / 'messages.csv')
    categories_path = str(tmp_path / 'categories.csv')
    messages_data.to_csv(messages_path, index=False)
    categories_data.to_csv(categories_path, index=False)

    return messages_path, categories_path


def test_load_data(sample_csv_files):
    """Test that load_data merges datasets correctly."""
    messages_path, categories_path = sample_csv_files
    df = load_data(messages_path, categories_path)

    assert len(df) == 3
    assert 'message' in df.columns
    assert 'categories' in df.columns
    assert 'id' in df.columns


def test_clean_data(sample_csv_files):
    """Test that clean_data creates binary category columns."""
    messages_path, categories_path = sample_csv_files
    df = load_data(messages_path, categories_path)
    df_clean = clean_data(df)

    # Categories column should be replaced with individual columns
    assert 'categories' not in df_clean.columns
    assert 'related' in df_clean.columns
    assert 'request' in df_clean.columns
    assert 'offer' in df_clean.columns

    # All category values should be 0 or 1 (no 2s)
    category_cols = ['related', 'request', 'offer']
    for col in category_cols:
        assert df_clean[col].isin([0, 1]).all(), f"Column {col} has non-binary values"


def test_clean_data_removes_duplicates(sample_csv_files):
    """Test that clean_data removes duplicate rows."""
    messages_path, categories_path = sample_csv_files
    df = load_data(messages_path, categories_path)
    # Add a duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert len(df) == 4

    df_clean = clean_data(df)
    assert len(df_clean) == 3


def test_save_data(sample_csv_files, tmp_path):
    """Test that save_data creates a valid SQLite database."""
    messages_path, categories_path = sample_csv_files
    df = load_data(messages_path, categories_path)
    df_clean = clean_data(df)

    db_path = str(tmp_path / 'test.db')
    save_data(df_clean, db_path)

    # Verify the database was created and contains data
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///' + db_path)
    df_loaded = pd.read_sql_table('ETL', engine)
    assert len(df_loaded) == len(df_clean)


def test_save_data_replace(sample_csv_files, tmp_path):
    """Test that save_data can overwrite an existing table."""
    messages_path, categories_path = sample_csv_files
    df = load_data(messages_path, categories_path)
    df_clean = clean_data(df)

    db_path = str(tmp_path / 'test.db')
    save_data(df_clean, db_path)
    # Running again should not crash (if_exists='replace')
    save_data(df_clean, db_path)
