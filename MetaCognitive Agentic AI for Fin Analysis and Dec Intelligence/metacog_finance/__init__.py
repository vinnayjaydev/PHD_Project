"""Meta-Cognitive Agentic Framework for Adaptive Financial Analytics and Decision Intelligence."""

from metacog_finance.core.models import (
    DecisionIntelligenceMemo,
    DecisionRecommendation,
    DocumentChunk,
    DocumentSection,
    EpistemicDiscrepancy,
    FinancialRatios,
    FinancialStatement,
    MetaCognitiveAudit,
)
from metacog_finance.orchestration.engine import MetaCognitivePipeline
from metacog_finance.orchestration.state import PipelineState

__version__ = "1.0.0"
__all__ = [
    "MetaCognitivePipeline",
    "PipelineState",
    "FinancialStatement",
    "FinancialRatios",
    "DecisionIntelligenceMemo",
    "DecisionRecommendation",
    "EpistemicDiscrepancy",
    "MetaCognitiveAudit",
    "DocumentChunk",
    "DocumentSection",
]
