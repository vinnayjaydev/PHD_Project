"""Quantitative Data Ingestion and Verification Agent.

Retrieves point-in-time SEC statements, indexes granular document chunks into the hybrid retriever,
and executes deterministic ratio computation and accounting verification.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from metacog_finance.agents.base import BaseFinancialAgent
from metacog_finance.core.models import (
    FinancialRatios,
    FinancialStatement,
)
from metacog_finance.metacognition.financial_metrics import compute_financial_ratios
from metacog_finance.metacognition.verifier import (
    DeterministicFinancialVerifier,
    VerificationResult,
)
from metacog_finance.rag.hybrid_retriever import HybridFinancialRetriever
from metacog_finance.rag.sec_edgar import SecEdgarClient


class DataIngestionAgent(BaseFinancialAgent):
    """Agent responsible for raw data harvesting, parsing, and arithmetic sanity verification."""

    def __init__(self, sec_client: Optional[SecEdgarClient] = None):
        super().__init__(
            name="DataIngestionAgent",
            role_description="Harvests point-in-time SEC filings, parses financial statements, and performs deterministic ratio computation.",
        )
        self.sec_client = sec_client or SecEdgarClient()
        self.verifier = DeterministicFinancialVerifier()

    def execute(
        self, ticker: str, fiscal_year: int = 2023, market_cap: Optional[float] = None
    ) -> Dict[str, Any]:
        """Harvests filings, executes ratio calculations, verifies identities, and constructs retriever."""
        # 1. Fetch current and prior statements
        current_stmt, prior_stmt = self.sec_client.get_financial_statements(ticker, fiscal_year)

        # 2. Compute deterministic financial ratios & forensic scores
        ratios = compute_financial_ratios(current_stmt, prior_stmt, market_cap=market_cap)

        # 3. Verify core accounting identities
        verification = self.verifier.verify_statement_identities(current_stmt)

        # 4. Load chunks and index hybrid retriever
        chunks = self.sec_client.load_filing_chunks(ticker, fiscal_year)
        retriever = HybridFinancialRetriever(chunks)

        return {
            "current_stmt": current_stmt,
            "prior_stmt": prior_stmt,
            "ratios": ratios,
            "verification": verification,
            "retriever": retriever,
            "chunks_count": len(chunks),
        }
