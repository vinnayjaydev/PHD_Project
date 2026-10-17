"""
Tests for Lakehouse Medallion ETL (Bronze -> Silver -> Gold) & DuckDB Engine
"""
import os
import unittest
from pathlib import Path
from lakehouse_agent.lakehouse.engine import LakehouseEngine
from lakehouse_agent.lakehouse.bronze import BronzeLayer
from lakehouse_agent.lakehouse.silver import SilverLayer
from lakehouse_agent.lakehouse.gold import GoldLayer
from lakehouse_agent.lakehouse.data_generator import FinancialDataGenerator


class TestLakehouseEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use an in-memory or dedicated test database
        cls.engine = LakehouseEngine(db_path=":memory:")
        cls.generator = FinancialDataGenerator(seed=42)
        cls.bronze = BronzeLayer(cls.engine)
        cls.silver = SilverLayer(cls.engine)
        cls.gold = GoldLayer(cls.engine)

    def test_01_bronze_ingestion(self):
        res = self.bronze.ingest_all(self.generator)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(res["investor_profiles_count"], 0)
        self.assertGreater(res["assets_count"], 0)
        self.assertGreater(res["holdings_count"], 0)

        # Verify tables exist in DuckDB
        tables = [t["table_name"] for t in self.engine.list_tables()]
        self.assertIn("bronze_investor_profiles", tables)
        self.assertIn("bronze_asset_universe", tables)
        self.assertIn("bronze_current_holdings", tables)
        self.assertIn("bronze_macro_indicators", tables)
        self.assertIn("bronze_raw_documents", tables)

    def test_02_silver_transformation(self):
        res = self.silver.process_all()
        self.assertEqual(res["status"], "SUCCESS")

        tables = [t["table_name"] for t in self.engine.list_tables()]
        self.assertIn("silver_investor_financial_health", tables)
        self.assertIn("silver_asset_metrics", tables)
        self.assertIn("silver_portfolio_holdings", tables)

        # Test Debt-to-Income and emergency runway calculations
        health = self.engine.query_dicts("SELECT * FROM silver_investor_financial_health WHERE investor_id = 'INV001';")
        self.assertEqual(len(health), 1)
        self.assertIn("debt_to_income_ratio", health[0])
        self.assertIn("emergency_fund_runway_months", health[0])
        self.assertGreater(health[0]["emergency_fund_runway_months"], 0)

    def test_03_gold_analytics_marts(self):
        res = self.gold.process_all()
        self.assertEqual(res["status"], "SUCCESS")

        tables = [t["table_name"] for t in self.engine.list_tables()]
        self.assertIn("gold_portfolio_analytics", tables)
        self.assertIn("gold_investment_universe", tables)
        self.assertIn("gold_recommendation_history", tables)

        # Verify portfolio beta calculation
        analytics = self.engine.query_dicts("SELECT * FROM gold_portfolio_analytics WHERE investor_id = 'INV001';")
        self.assertEqual(len(analytics), 1)
        self.assertGreater(analytics[0]["total_portfolio_value"], 0)
        self.assertIsNotNone(analytics[0]["portfolio_beta"])
        self.assertIsNotNone(analytics[0]["equity_pct"])

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()


if __name__ == "__main__":
    unittest.main()
