"""
Unit tests for Financial RAG Subsystem (Hybrid Search, Guardrails & Evaluation).
"""

import pytest
from rag.hybrid_retriever import HybridFinancialRetriever
from rag.temporal_guardrail import TemporalGuardrail
from rag.evaluation_metrics import RAGEvaluationSuite


def test_hybrid_retriever_search():
    retriever = HybridFinancialRetriever()
    results = retriever.search("Data center AI GPU revenue and growth", ticker="NVDA", top_k=3)
    assert len(results) > 0, "Retriever returned 0 results!"
    top_hit = results[0]
    assert "content" in top_hit
    assert "hybrid_score" in top_hit
    assert "citation" in top_hit
    assert top_hit["ticker"] == "NVDA"


def test_temporal_guardrail():
    guard = TemporalGuardrail(strict_mode=True)
    sample_chunks = [
        {"chunk_id": "c1", "filing_date": "2023-11-01"},
        {"chunk_id": "c2", "filing_date": "2024-03-15"}
    ]
    # Decision date: 2023-12-31 (c2 is a future leak!)
    is_valid, score, clean_chunks = guard.audit_retrieved_context(sample_chunks, decision_date="2023-12-31")
    assert not is_valid, "Guardrail should flag future leakage!"
    assert score == 0.5, f"Expected 0.5 integrity score, got {score}"
    assert len(clean_chunks) == 1
    assert clean_chunks[0]["chunk_id"] == "c1"


def test_rag_evaluation_metrics():
    suite = RAGEvaluationSuite()
    q = "What was NVIDIA revenue in FY2024?"
    r = "NVIDIA's total revenue for FY2024 was $60.9 billion."
    ctx = ["Total revenue for Fiscal 2024 was $60.9 billion, up 126% from a year ago."]
    res = suite.evaluate(q, r, ctx)
    assert res["faithfulness"] >= 0.70, f"Faithfulness should be high, got {res['faithfulness']}"
    assert res["answer_relevance"] > 0.50
    assert res["hallucination_rate"] <= 0.30
