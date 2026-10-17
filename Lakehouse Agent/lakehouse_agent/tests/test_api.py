"""
Tests for FastAPI REST Endpoints & Web Service
"""
import unittest
from fastapi.testclient import TestClient
from lakehouse_agent.web.app import app


class TestWebAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_status_endpoint(self):
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertGreater(data["total_lakehouse_tables"], 0)

    def test_get_investors(self):
        response = self.client.get("/api/investors")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("investors", data)
        self.assertGreater(len(data["investors"]), 0)

    def test_investor_analytics(self):
        response = self.client.get("/api/investor/INV001/analytics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("analytics", data)
        self.assertIn("holdings", data)

    def test_recommendation_generate(self):
        response = self.client.post("/api/recommendation/generate", json={"investor_id": "INV001"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("recommendation_id", data)
        self.assertIn("target_allocation", data)
        self.assertIn("orders", data)
        self.assertIn("compliance", data)

    def test_lakehouse_sql_query(self):
        response = self.client.post("/api/lakehouse/query", json={"query": "SELECT COUNT(*) AS total_assets FROM bronze_asset_universe;"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["row_count"], 1)

    def test_scenario_simulate(self):
        response = self.client.post("/api/scenario/simulate", json={"investor_id": "INV001", "scenario": "tech_correction"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("scenario", data)
        self.assertIn("current_portfolio", data)
        self.assertIn("optimized_target_portfolio", data)
        self.assertIn("protection_alpha_pct", data)


if __name__ == "__main__":
    unittest.main()
