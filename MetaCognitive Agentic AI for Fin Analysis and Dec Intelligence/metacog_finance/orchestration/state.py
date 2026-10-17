"""State management for the Meta-Cognitive Orchestration Pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from metacog_finance.core.models import (
    DecisionIntelligenceMemo,
    FinancialRatios,
    FinancialStatement,
    MetaCognitiveAudit,
)
from metacog_finance.metacognition.verifier import VerificationResult
from metacog_finance.rag.hybrid_retriever import HybridFinancialRetriever


class PipelineState(BaseModel):
    """Encapsulates mutable state throughout the multi-agent cognitive and meta-cognitive loops."""
    model_config = {"arbitrary_types_allowed": True}

    ticker: str
    fiscal_year: int = 2023
    market_cap: Optional[float] = None
    status: str = "INITIALIZED"
    iteration_count: int = 0

    # Data artifacts
    current_stmt: Optional[FinancialStatement] = None
    prior_stmt: Optional[FinancialStatement] = None
    ratios: Optional[FinancialRatios] = None
    verification: Optional[VerificationResult] = None
    retriever: Optional[HybridFinancialRetriever] = None

    # Cognitive outputs
    fundamental_narrative: str = ""
    verbalized_confidence: float = 0.85
    citations_used: List[str] = Field(default_factory=list)
    forensic_data: Dict[str, Any] = Field(default_factory=dict)

    # Meta-cognitive supervision
    audit_trace: List[MetaCognitiveAudit] = Field(default_factory=list)
    active_critique: str = ""

    # Synthesized intelligence
    final_memo: Optional[DecisionIntelligenceMemo] = None
