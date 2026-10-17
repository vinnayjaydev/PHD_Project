"""Unit tests for core domain models."""

import pytest
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


def test_financial_statement_defaults():
    stmt = FinancialStatement(ticker="AAPL", fiscal_year=2023, revenue=383285.0)
    assert stmt.ticker == "AAPL"
    assert stmt.fiscal_year == 2023
    assert stmt.revenue == 383285.0
    assert stmt.net_income == 0.0


def test_document_chunk_model():
    chunk = DocumentChunk(
        chunk_id="chunk_1",
        ticker="AAPL",
        fiscal_year=2023,
        section=DocumentSection.ITEM_7_MDNA,
        content="Revenues increased in services.",
        is_tabular=False,
    )
    assert chunk.section == DocumentSection.ITEM_7_MDNA
    assert not chunk.is_tabular


def test_epistemic_discrepancy_model():
    disc = EpistemicDiscrepancy(
        discrepancy_id="DISC_1",
        category="Earnings_Quality_Divergence",
        claim_in_narrative="Strong operational cash flow",
        ground_truth_metric="Operating Cash Flow",
        ground_truth_value=-500.0,
        reported_period="FY2023",
        severity=DiscrepancySeverity.CRITICAL,
        explanation="Cash flow is negative.",
        source_citation="10-K Cash Flow Statement",
    )
    assert disc.severity == DiscrepancySeverity.CRITICAL
    assert disc.ground_truth_value == -500.0


def test_meta_cognitive_audit_model():
    audit = MetaCognitiveAudit(
        cycle_index=1,
        verbalized_confidence=0.90,
        calibrated_confidence=0.65,
        grounding_faithfulness_score=0.85,
        numerical_verification_pass=True,
    )
    assert audit.calibrated_confidence == 0.65
    assert not audit.backtracking_triggered
