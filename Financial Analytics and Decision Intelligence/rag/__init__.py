"""
FLAME-AI: Retrieval-Augmented Generation (RAG) Subsystem for Financial Intelligence
Includes Dense-Sparse Hybrid Retriever, Point-in-Time Temporal Guardrails,
and Academic Ragas-Style Evaluation Metrics.
"""

from .hybrid_retriever import HybridFinancialRetriever
from .temporal_guardrail import TemporalGuardrail
from .evaluation_metrics import RAGEvaluationSuite

__all__ = [
    "HybridFinancialRetriever",
    "TemporalGuardrail",
    "RAGEvaluationSuite"
]
