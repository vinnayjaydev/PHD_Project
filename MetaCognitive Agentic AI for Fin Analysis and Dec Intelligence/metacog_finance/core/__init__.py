"""Core package containing data models, configuration, and telemetry."""

from metacog_finance.core.config import settings
from metacog_finance.core.models import (
    DecisionIntelligenceMemo,
    DecisionRecommendation,
    DiscrepancySeverity,
    DocumentChunk,
    DocumentSection,
    EpistemicDiscrepancy,
    FinancialRatios,
    FinancialStatement,
    MetaCognitiveAudit,
)
from metacog_finance.core.telemetry import TelemetryTracer

__all__ = [
    "settings",
    "DocumentSection",
    "DocumentChunk",
    "FinancialStatement",
    "FinancialRatios",
    "DiscrepancySeverity",
    "EpistemicDiscrepancy",
    "MetaCognitiveAudit",
    "DecisionRecommendation",
    "DecisionIntelligenceMemo",
    "TelemetryTracer",
]
