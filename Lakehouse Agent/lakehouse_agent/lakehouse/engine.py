"""
Lakehouse Engine: Embedded DuckDB + Parquet Storage Manager
Provides ACID transactions, fast analytical OLAP queries, and parquet file partitioning.
"""
import duckdb
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from lakehouse_agent.config import settings


class LakehouseEngine:
    """Embedded DuckDB Lakehouse Engine managing Bronze, Silver, and Gold data layers."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.lakehouse_db_path
        self._conn = None
        self._initialize_connection()

    def _initialize_connection(self):
        """Initializes or connects to the DuckDB database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(self.db_path)
        # Enable multi-threading and optimized memory
        self._conn.execute("PRAGMA threads=4;")
        self._conn.execute("PRAGMA enable_object_cache;")

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._initialize_connection()
        return self._conn

    def execute(self, query: str, params: Optional[List[Any]] = None) -> duckdb.DuckDBPyConnection:
        """Executes a SQL statement."""
        if params:
            return self.connection.execute(query, params)
        return self.connection.execute(query)

    def query_df(self, query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """Executes a query and returns results as a Pandas DataFrame."""
        rel = self.execute(query, params)
        return rel.df()

    def query_dicts(self, query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Executes a query and returns results as a list of dictionaries."""
        df = self.query_df(query, params)
        return df.to_dict(orient="records")

    def save_table_as_parquet(self, table_name: str, output_path: Path):
        """Persists a DuckDB table to an optimized Parquet file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        escaped_path = str(output_path).replace("\\", "/")
        self.execute(f"COPY {table_name} TO '{escaped_path}' (FORMAT PARQUET, COMPRESSION ZSTD);")

    def load_parquet_view(self, view_name: str, parquet_path: Path):
        """Registers a Parquet file as an accessible SQL view."""
        escaped_path = str(parquet_path).replace("\\", "/")
        self.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{escaped_path}');")

    def list_tables(self) -> List[Dict[str, Any]]:
        """Lists all existing tables and views in the Lakehouse."""
        return self.query_dicts("""
            SELECT table_schema, table_name, table_type 
            FROM information_schema.tables 
            WHERE table_schema = 'main'
            ORDER BY table_name;
        """)

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Returns the schema description for a specific table."""
        return self.query_dicts(f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position;
        """)

    def close(self):
        """Closes the DuckDB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
