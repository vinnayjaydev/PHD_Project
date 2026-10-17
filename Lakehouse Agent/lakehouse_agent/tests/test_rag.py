"""
Tests for Vector Store, Embeddings, and Hybrid Semantic RAG Retrieval
"""
import unittest
from lakehouse_agent.vector_store.embeddings import UnifiedEmbeddings
from lakehouse_agent.vector_store.store import FinancialVectorStore


class TestVectorRAG(unittest.TestCase):
    def setUp(self):
        self.embeddings = UnifiedEmbeddings(provider="local")
        self.vector_store = FinancialVectorStore(embeddings=self.embeddings)
        self.sample_docs = [
            {
                "doc_id": "TEST_DOC_1",
                "ticker": "NVDA",
                "title": "NVIDIA Supply Chain Disclosures",
                "doc_type": "SEC_10K",
                "source": "SEC EDGAR",
                "published_at": "2024-03-01",
                "content": "NVIDIA faces semiconductor wafer packaging constraints at foundry partners. Hyperscale capex drives GPU demand."
            },
            {
                "doc_id": "TEST_DOC_2",
                "ticker": "MACRO",
                "title": "Federal Reserve Rate Outlook",
                "doc_type": "Fed_Statement",
                "source": "Federal Reserve",
                "published_at": "2024-09-18",
                "content": "The Federal Reserve policy committee anticipates benchmark rate cuts. Cash yields will face reinvestment drag."
            }
        ]

    def test_embeddings_generation(self):
        vec = self.embeddings.get_embedding("Risk and interest rate volatility in financial markets")
        self.assertIsInstance(vec, list)
        self.assertEqual(len(vec), 256)

    def test_vector_indexing_and_search(self):
        self.vector_store.add_documents(self.sample_docs)
        self.assertGreater(len(self.vector_store.chunks), 0)

        # Search for rate cuts
        results = self.vector_store.search("Federal Reserve interest rate cuts reinvestment drag", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertIn("citation", results[0])
        self.assertIn("similarity_score", results[0])

        # Verify filtering by ticker
        nvda_results = self.vector_store.search("GPU packaging", top_k=2, filter_ticker="NVDA")
        self.assertGreater(len(nvda_results), 0)
        self.assertEqual(nvda_results[0]["ticker"], "NVDA")


if __name__ == "__main__":
    unittest.main()
