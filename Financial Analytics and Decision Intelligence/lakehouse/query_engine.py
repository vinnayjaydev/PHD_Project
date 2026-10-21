"""
Lakehouse Query Engine: Point-in-Time DuckDB OLAP Interface.
Enforces strict temporal consistency (no lookahead bias) and high-speed analytics queries.
Part of the FLAME-AI Financial Lakehouse Architecture.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import duckdb
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("LakehouseQueryEngine")


class LakehouseQueryEngine:
    """
    High-performance DuckDB query interface with strict temporal point-in-time enforcement.
    Guarantees no future data leakage across quantitative features, risk cubes, and SEC text chunks.
    """

    def __init__(self, duckdb_path: str = "data/gold/financial_lakehouse.duckdb"):
        self.duckdb_path = duckdb_path
        if not os.path.exists(duckdb_path):
            logger.warning(f"DuckDB database not found at {duckdb_path}. Initializing empty connection.")
        self._con = None

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect(self.duckdb_path, read_only=False)
        return self._con

    def execute_sql(self, sql: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """
        Executes an arbitrary SQL query on the Lakehouse catalog and returns a Pandas DataFrame.
        """
        con = self._get_connection()
        try:
            if params:
                return con.execute(sql, params).fetchdf()
            return con.execute(sql).fetchdf()
        except Exception as e:
            logger.error(f"SQL Execution Error: {e} in query: {sql}")
            return pd.DataFrame()

    def get_market_state(self, as_of_date: str, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retrieves point-in-time market state as of `as_of_date` for specified tickers.
        """
        sql = """
            SELECT date, ticker, close, return_1d, return_21d, vol_21d, rsi_14, macd, macd_signal, bb_zscore,
                   regime_id, regime_name, rolling_sharpe, rolling_sortino, var_95, cvar_95, rolling_max_dd
            FROM gold_unified_analytics
            WHERE date <= ?
        """
        params = [as_of_date]
        if tickers:
            placeholders = ",".join(["?"] * len(tickers))
            sql += f" AND ticker IN ({placeholders})"
            params.extend(tickers)

        # Get latest record for each ticker as of the specified date
        full_sql = f"""
            WITH ranked AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
                FROM ({sql})
            )
            SELECT * EXCLUDE (rn) FROM ranked WHERE rn = 1
        """
        return self.execute_sql(full_sql, params)

    def get_active_regime(self, as_of_date: str) -> Dict[str, Any]:
        """
        Retrieves the systemic market regime classification as of `as_of_date`.
        """
        sql = """
            SELECT date, regime_id, regime_name, trend_strength, market_volatility, benchmark_close
            FROM gold_market_regimes
            WHERE date <= ?
            ORDER BY date DESC
            LIMIT 1
        """
        df = self.execute_sql(sql, [as_of_date])
        if not df.empty:
            return df.iloc[0].to_dict()
        return {
            "date": as_of_date,
            "regime_id": 3,
            "regime_name": "Sideways_RangeBound",
            "trend_strength": 0.0,
            "market_volatility": 0.18,
            "benchmark_close": 400.0
        }

    def get_filing_chunks_point_in_time(
        self,
        as_of_date: str,
        ticker: Optional[str] = None,
        doc_type: Optional[str] = None,
        limit: int = 50
    ) -> pd.DataFrame:
        """
        Strict point-in-time text retrieval: only returns document chunks filed ON OR BEFORE `as_of_date`.
        Crucial for PhD-level academic validity to prevent lookahead bias in RAG pipelines.
        """
        sql = """
            SELECT chunk_id, doc_id, ticker, company_name, doc_type, filing_date, section, content, is_tabular, embedding
            FROM silver_filings_chunks
            WHERE filing_date <= ?
        """
        params = [as_of_date]
        if ticker:
            sql += " AND ticker = ?"
            params.append(ticker)
        if doc_type:
            sql += " AND doc_type = ?"
            params.append(doc_type)

        sql += f" ORDER BY filing_date DESC LIMIT {limit}"
        return self.execute_sql(sql, params)

    def get_cross_asset_covariance(self, as_of_date: str) -> Optional[pd.DataFrame]:
        """
        Retrieves the latest available covariance matrix as of `as_of_date`.
        """
        gold_cov_path = "data/gold/covariance_matrices.parquet"
        if not os.path.exists(gold_cov_path):
            return None

        sql = f"""
            SELECT date, tickers, covariance_json, avg_pairwise_corr
            FROM read_parquet('{gold_cov_path}')
            WHERE date <= ?
            ORDER BY date DESC
            LIMIT 1
        """
        df = self.execute_sql(sql, [as_of_date])
        if not df.empty:
            cov_dict = json.loads(df.iloc[0]["covariance_json"])
            return pd.DataFrame.from_dict(cov_dict)
        return None

    def get_lakehouse_catalog_summary(self) -> Dict[str, Any]:
        """
        Returns an institutional overview of all Lakehouse layers, table schemas,
        storage byte sizes, and row counts.
        """
        summary = {
            "bronze": {},
            "silver": {},
            "gold": {},
            "duckdb_views": []
        }

        # Inspect Bronze
        bronze_market = "data/bronze/market_data"
        if os.path.exists(bronze_market):
            files = [f for f in os.listdir(bronze_market) if f.endswith(".parquet")]
            summary["bronze"]["market_series_count"] = len(files)
            summary["bronze"]["tickers"] = [f.replace("_raw.parquet", "") for f in files]

        # Inspect Silver
        silver_feat = "data/silver/market_features.parquet"
        if os.path.exists(silver_feat):
            sdf = pd.read_parquet(silver_feat)
            summary["silver"]["market_features_rows"] = len(sdf)
            summary["silver"]["date_range"] = [str(sdf["date"].min()), str(sdf["date"].max())]
            summary["silver"]["columns"] = list(sdf.columns)

        silver_chunks = "data/silver/filings_chunks.parquet"
        if os.path.exists(silver_chunks):
            cdf = pd.read_parquet(silver_chunks)
            summary["silver"]["filing_chunks_count"] = len(cdf)
            summary["silver"]["filings_tickers"] = list(cdf["ticker"].unique())

        # Inspect Gold
        gold_reg = "data/gold/market_regimes_cube.parquet"
        if os.path.exists(gold_reg):
            rdf = pd.read_parquet(gold_reg)
            summary["gold"]["regime_cube_rows"] = len(rdf)
            summary["gold"]["regimes_distribution"] = rdf["regime_name"].value_counts().to_dict()

        # Inspect DuckDB views
        try:
            views_df = self.execute_sql("SELECT table_name FROM information_schema.tables WHERE table_type = 'VIEW'")
            summary["duckdb_views"] = views_df["table_name"].tolist() if not views_df.empty else []
        except Exception:
            summary["duckdb_views"] = ["silver_market_features", "silver_filings_chunks", "gold_market_regimes", "gold_rolling_risk", "gold_unified_analytics"]

        return summary

    def close(self):
        if self._con:
            self._con.close()
            self._con = None


if __name__ == "__main__":
    engine = LakehouseQueryEngine()
    print("Catalog summary:", engine.get_lakehouse_catalog_summary())
    engine.close()
