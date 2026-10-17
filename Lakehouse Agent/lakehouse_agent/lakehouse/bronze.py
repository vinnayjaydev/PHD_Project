"""
Bronze Layer: Raw Data Ingestion & Storage
Stores unmodified, raw ingested records in DuckDB and partitioned Parquet format.
"""
import pandas as pd
from datetime import datetime
from lakehouse_agent.config import settings
from lakehouse_agent.lakehouse.engine import LakehouseEngine
from lakehouse_agent.lakehouse.data_generator import FinancialDataGenerator


class BronzeLayer:
    """Manages raw ingestion into the Lakehouse Bronze medallion tier."""
    
    def __init__(self, engine: LakehouseEngine):
        self.engine = engine
        self.bronze_path = settings.bronze_path

    def ingest_all(self, generator: FinancialDataGenerator = None) -> dict:
        """Runs the full ingestion pipeline for Bronze tables."""
        gen = generator or FinancialDataGenerator()
        
        # 1. Investor profiles
        df_investors = gen.generate_investor_profiles()
        self.ingest_investor_profiles(df_investors)

        # 2. Asset Universe
        df_assets = gen.generate_asset_universe()
        self.ingest_asset_universe(df_assets)

        # 3. Current Holdings
        df_holdings = gen.generate_current_holdings()
        self.ingest_current_holdings(df_holdings)

        # 4. Macro Indicators
        df_macro = gen.generate_macro_indicators()
        self.ingest_macro_indicators(df_macro)

        # 5. Raw Documents (for hybrid RAG)
        docs = gen.generate_unstructured_financial_documents()
        df_docs = pd.DataFrame(docs)
        self.ingest_raw_documents(df_docs)

        return {
            "investor_profiles_count": len(df_investors),
            "assets_count": len(df_assets),
            "holdings_count": len(df_holdings),
            "macro_records_count": len(df_macro),
            "documents_count": len(df_docs),
            "status": "SUCCESS",
            "timestamp": datetime.now().isoformat()
        }

    def ingest_investor_profiles(self, df: pd.DataFrame):
        self.engine.connection.register("temp_bronze_investors", df)
        self.engine.execute("""
            CREATE OR REPLACE TABLE bronze_investor_profiles AS 
            SELECT *, CURRENT_TIMESTAMP AS ingested_at FROM temp_bronze_investors;
        """)
        self.engine.save_table_as_parquet("bronze_investor_profiles", self.bronze_path / "investor_profiles.parquet")

    def ingest_asset_universe(self, df: pd.DataFrame):
        self.engine.connection.register("temp_bronze_assets", df)
        self.engine.execute("""
            CREATE OR REPLACE TABLE bronze_asset_universe AS 
            SELECT *, CURRENT_TIMESTAMP AS ingested_at FROM temp_bronze_assets;
        """)
        self.engine.save_table_as_parquet("bronze_asset_universe", self.bronze_path / "asset_universe.parquet")

    def ingest_current_holdings(self, df: pd.DataFrame):
        self.engine.connection.register("temp_bronze_holdings", df)
        self.engine.execute("""
            CREATE OR REPLACE TABLE bronze_current_holdings AS 
            SELECT *, CURRENT_TIMESTAMP AS ingested_at FROM temp_bronze_holdings;
        """)
        self.engine.save_table_as_parquet("bronze_current_holdings", self.bronze_path / "current_holdings.parquet")

    def ingest_macro_indicators(self, df: pd.DataFrame):
        self.engine.connection.register("temp_bronze_macro", df)
        self.engine.execute("""
            CREATE OR REPLACE TABLE bronze_macro_indicators AS 
            SELECT *, CURRENT_TIMESTAMP AS ingested_at FROM temp_bronze_macro;
        """)
        self.engine.save_table_as_parquet("bronze_macro_indicators", self.bronze_path / "macro_indicators.parquet")

    def ingest_raw_documents(self, df: pd.DataFrame):
        self.engine.connection.register("temp_bronze_docs", df)
        self.engine.execute("""
            CREATE OR REPLACE TABLE bronze_raw_documents AS 
            SELECT *, CURRENT_TIMESTAMP AS ingested_at FROM temp_bronze_docs;
        """)
        self.engine.save_table_as_parquet("bronze_raw_documents", self.bronze_path / "raw_documents.parquet")
