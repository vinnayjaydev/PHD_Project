"""Retrieval-Augmented Generation (RAG) package for financial analysis."""

from metacog_finance.rag.chunker import FinancialChunker
from metacog_finance.rag.hybrid_retriever import HybridFinancialRetriever
from metacog_finance.rag.sec_edgar import SecEdgarClient
from metacog_finance.rag.tabular_parser import FinancialTableParser, parse_accounting_number

__all__ = [
    "FinancialChunker",
    "HybridFinancialRetriever",
    "SecEdgarClient",
    "FinancialTableParser",
    "parse_accounting_number",
]
