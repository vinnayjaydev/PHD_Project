"""
Tests for Multi-Agent Cognitive Pipeline & Compliance Guardrails
"""
import unittest
from lakehouse_agent.lakehouse.engine import LakehouseEngine
from lakehouse_agent.lakehouse.bronze import BronzeLayer
from lakehouse_agent.lakehouse.silver import SilverLayer
from lakehouse_agent.lakehouse.gold import GoldLayer
from lakehouse_agent.vector_store.store import FinancialVectorStore
from lakehouse_agent.agents.llm_provider import LLMClient
from lakehouse_agent.agents.orchestrator import FinancialAdvisorOrchestrator


class TestMultiAgentSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = LakehouseEngine(db_path=":memory:")
        cls.bronze = BronzeLayer(cls.engine)
        cls.silver = SilverLayer(cls.engine)
        cls.gold = GoldLayer(cls.engine)
        cls.vector_store = FinancialVectorStore()

        # Prime the test lakehouse
        cls.bronze.ingest_all()
        cls.silver.process_all()
        cls.gold.process_all()
        cls.vector_store.sync_from_lakehouse(cls.engine)

        cls.orchestrator = FinancialAdvisorOrchestrator(
            engine=cls.engine,
            vector_store=cls.vector_store
        )

    def test_full_agentic_pipeline_execution(self):
        report = self.orchestrator.run_advisory_pipeline("INV001")

        # 1. Check basic structure
        self.assertIn("recommendation_id", report)
        self.assertEqual(report["investor_id"], "INV001")
        self.assertEqual(report["risk_tier"], "Aggressive")

        # 2. Check Target Asset Allocation
        target = report["target_allocation"]
        self.assertIn("Equities", target)
        self.assertIn("Fixed Income", target)
        self.assertIn("Cash", target)
        total_weight = sum(target.values())
        self.assertAlmostEqual(total_weight, 100.0, delta=1.0)

        # 3. Check Rebalancing Orders
        orders = report["orders"]
        self.assertIsInstance(orders, list)
        self.assertGreater(len(orders), 0)
        first_order = orders[0]
        self.assertIn(first_order["action"], ["BUY", "SELL", "HOLD"])
        self.assertIn("ticker", first_order)
        self.assertGreater(first_order["shares"], 0)

        # 4. Check Compliance Certification
        compliance = report["compliance"]
        self.assertIn(compliance["status"], ["APPROVED", "CONDITIONALLY_APPROVED_WITH_CORRECTIONS"])
        self.assertIn("attestation", compliance)

        # 5. Check RAG Citations
        citations = report["rag_citations"]
        self.assertGreater(len(citations), 0)

        # 6. Check Audit Trail Persistence in Lakehouse
        audit_records = self.engine.query_dicts(f"""
            SELECT * FROM gold_recommendation_history 
            WHERE recommendation_id = '{report['recommendation_id']}';
        """)
        self.assertEqual(len(audit_records), 1)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()


if __name__ == "__main__":
    unittest.main()
