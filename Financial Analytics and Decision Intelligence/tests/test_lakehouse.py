"""
Unit tests for FLAME-AI Lakehouse Subsystem (Bronze, Silver, Gold & DuckDB).
"""

import os
import pytest
import pandas as pd
from lakehouse.query_engine import LakehouseQueryEngine


def test_lakehouse_files_exist():
    assert os.path.exists("data/silver/market_features.parquet"), "Silver market features file missing!"
    assert os.path.exists("data/silver/filings_chunks.parquet"), "Silver filing chunks file missing!"
    assert os.path.exists("data/gold/market_regimes_cube.parquet"), "Gold regime cube file missing!"
    assert os.path.exists("data/gold/financial_lakehouse.duckdb"), "DuckDB database file missing!"


def test_duckdb_views_and_query():
    engine = LakehouseQueryEngine()
    df = engine.execute_sql("SELECT COUNT(*) as cnt FROM silver_market_features")
    assert not df.empty, "Query returned empty DataFrame!"
    assert df.iloc[0]["cnt"] > 1000, f"Expected >1000 records, got {df.iloc[0]['cnt']}"


def test_point_in_time_enforcement():
    engine = LakehouseQueryEngine()
    # Test that no filings after 2024-01-01 are returned when as_of_date is 2024-01-01
    df = engine.get_filing_chunks_point_in_time(as_of_date="2024-01-01")
    for _, row in df.iterrows():
        assert row["filing_date"] <= "2024-01-01", f"Temporal violation: filing date {row['filing_date']} > 2024-01-01"


def test_active_regime_lookup():
    engine = LakehouseQueryEngine()
    regime = engine.get_active_regime("2024-06-01")
    assert "regime_name" in regime
    assert regime["regime_id"] in [0, 1, 2, 3]
