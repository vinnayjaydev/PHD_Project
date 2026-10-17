"""Integration test for end-to-end multi-agent orchestration pipeline."""

from metacog_finance.core.models import DecisionRecommendation
from metacog_finance.orchestration.engine import MetaCognitivePipeline


def test_pipeline_aapl_execution():
    pipeline = MetaCognitivePipeline(max_iterations=2)
    memo, state = pipeline.run(ticker="AAPL", fiscal_year=2023)

    assert state.status == "COMPLETED"
    assert memo.ticker == "AAPL"
    assert memo.fiscal_year == 2023
    assert memo.calibrated_confidence > 0.70
    assert len(memo.prescriptive_allocations) > 0
    assert "Bull_Case" in memo.scenario_analysis
    assert len(state.audit_trace) >= 1


def test_pipeline_stress_case_execution():
    pipeline = MetaCognitivePipeline(max_iterations=2)
    memo, state = pipeline.run(ticker="STRESS", fiscal_year=2023)

    assert state.status == "COMPLETED"
    assert memo.ticker == "STRESS"
    # Should recommend avoid due to critical discrepancy and distress
    assert memo.recommendation == DecisionRecommendation.CRITICAL_AVOID
    assert memo.calibrated_confidence <= 0.50
    assert memo.confidence_tier.startswith("LOW")
    assert len(state.audit_trace) >= 1
    # Check that discrepancies were detected
    assert len(state.audit_trace[0].discrepancies) >= 1
