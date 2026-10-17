"""Unit tests for the Meta-Cognitive Supervisory Layer."""

from metacog_finance.core.models import (
    DiscrepancySeverity,
    EpistemicDiscrepancy,
    FinancialRatios,
    FinancialStatement,
)
from metacog_finance.metacognition.calibrator import ConfidenceCalibrator
from metacog_finance.metacognition.monitor import EpistemicDiscrepancyDetector
from metacog_finance.metacognition.reflection_loop import ReflectionLoopController


def test_discrepancy_detector_finds_cash_divergence():
    detector = EpistemicDiscrepancyDetector()
    stmt = FinancialStatement(
        ticker="DIVERGENT",
        fiscal_year=2023,
        net_income=1500.0,
        operating_cash_flow=-200.0,  # Negative operating cash flow
    )
    ratios = FinancialRatios(current_ratio=1.5, quick_ratio=1.2, accrual_ratio=0.15)
    narrative = "The company reported profitable operational performance and robust earnings growth."

    discrepancies = detector.detect_discrepancies(narrative, stmt, ratios)
    assert len(discrepancies) >= 1
    cash_disc = [d for d in discrepancies if d.category == "Earnings_Quality_Divergence"]
    assert len(cash_disc) == 1
    assert cash_disc[0].severity == DiscrepancySeverity.CRITICAL


def test_confidence_calibrator_penalizes_critical_discrepancies():
    calibrator = ConfidenceCalibrator()
    discrepancies = [
        EpistemicDiscrepancy(
            discrepancy_id="D1",
            category="Earnings_Quality_Divergence",
            claim_in_narrative="Profitable operations",
            ground_truth_metric="OCF",
            ground_truth_value=-100.0,
            reported_period="FY2023",
            severity=DiscrepancySeverity.CRITICAL,
            explanation="Critical divergence",
            source_citation="10-K",
        )
    ]

    audit = calibrator.calibrate(
        cycle_index=1,
        verbalized_confidence=0.95,
        grounding_faithfulness_score=0.90,
        verification_score=1.0,
        verification_passed=True,
        discrepancies=discrepancies,
    )

    # Should be heavily penalized due to critical discrepancy
    assert audit.calibrated_confidence <= 0.45
    assert audit.backtracking_triggered


def test_reflection_loop_formulates_plan():
    calibrator = ConfidenceCalibrator()
    controller = ReflectionLoopController(max_iterations=2)

    discrepancies = [
        EpistemicDiscrepancy(
            discrepancy_id="D1",
            category="Earnings_Quality_Divergence",
            claim_in_narrative="Strong growth",
            ground_truth_metric="OCF",
            ground_truth_value=-500.0,
            reported_period="FY2023",
            severity=DiscrepancySeverity.CRITICAL,
            explanation="Negative cash flow",
            source_citation="10-K",
        )
    ]
    audit = calibrator.calibrate(
        cycle_index=1,
        verbalized_confidence=0.90,
        grounding_faithfulness_score=0.80,
        verification_score=1.0,
        verification_passed=True,
        discrepancies=discrepancies,
    )

    plan = controller.formulate_backtracking_plan(audit, current_iteration=1)
    assert plan is not None
    assert plan["action"] == "ITERATIVE_REFINEMENT"
    assert "Cash Flows" in plan["secondary_query"]
    assert "CRITICAL WARNING" in plan["critique"]
