"""Formal Confidence Calibrator.

Calculates calibrated epistemic confidence by mathematically penalizing raw verbalized
confidence according to citation grounding, arithmetic verification, and detected
epistemic discrepancies.
"""

from __future__ import annotations

from typing import List, Tuple
from metacog_finance.core.config import settings
from metacog_finance.core.models import (
    DiscrepancySeverity,
    EpistemicDiscrepancy,
    MetaCognitiveAudit,
)


class ConfidenceCalibrator:
    """Calibrates verbalized confidence into true epistemic confidence."""

    def __init__(
        self,
        alpha: float = settings.calib_weight_faithfulness,
        beta: float = settings.calib_weight_consistency,
        gamma: float = settings.calib_weight_verification,
        backtrack_threshold: float = settings.confidence_backtrack_threshold,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.backtrack_threshold = backtrack_threshold

    def calculate_discrepancy_penalty(self, discrepancies: List[EpistemicDiscrepancy]) -> float:
        """Computes aggregate penalty based on severity of contradictions."""
        total_penalty = 0.0
        for d in discrepancies:
            if d.severity == DiscrepancySeverity.CRITICAL:
                total_penalty += settings.penalty_critical
            elif d.severity == DiscrepancySeverity.HIGH:
                total_penalty += settings.penalty_high
            elif d.severity == DiscrepancySeverity.MEDIUM:
                total_penalty += settings.penalty_medium
            elif d.severity == DiscrepancySeverity.LOW:
                total_penalty += settings.penalty_low

        return min(0.85, total_penalty)

    def calibrate(
        self,
        cycle_index: int,
        verbalized_confidence: float,
        grounding_faithfulness_score: float,
        verification_score: float,
        verification_passed: bool,
        discrepancies: List[EpistemicDiscrepancy],
    ) -> MetaCognitiveAudit:
        """Executes full calibration algorithm and produces a MetaCognitiveAudit record."""
        # 1. Compute discrepancy penalty
        penalty = self.calculate_discrepancy_penalty(discrepancies)
        consistency_score = max(0.0, 1.0 - penalty)

        # 2. Base weighted objective score
        base_score = (
            (self.alpha * grounding_faithfulness_score)
            + (self.beta * consistency_score)
            + (self.gamma * verification_score)
        )

        # 3. Penalize verbalized confidence (cannot exceed base evidence score)
        # If verbalized confidence is wildly high despite errors, force severe shrinkage
        calibrated = min(verbalized_confidence, base_score)

        # If any CRITICAL discrepancy exists, cap confidence at 0.50
        has_critical = any(d.severity == DiscrepancySeverity.CRITICAL for d in discrepancies)
        if has_critical:
            calibrated = min(calibrated, 0.45)

        # Bound strictly between 0.05 and 0.99
        calibrated_confidence = max(0.05, min(0.99, round(calibrated, 3)))

        # 4. Decide whether backtracking is triggered
        backtrack_triggered = False
        rationale = None

        if calibrated_confidence < self.backtrack_threshold:
            backtrack_triggered = True
            rationale = (
                f"Calibrated confidence {calibrated_confidence:.2f} fell below minimum required "
                f"threshold {self.backtrack_threshold:.2f}."
            )
        elif has_critical:
            backtrack_triggered = True
            rationale = "Critical accounting/cash flow discrepancy detected; requires targeted forensic re-examination."
        elif not verification_passed:
            backtrack_triggered = True
            rationale = "Accounting balance sheet or cash flow identity verification failed."

        audit_verdict = "PASSED" if not backtrack_triggered else "REVISED"

        return MetaCognitiveAudit(
            cycle_index=cycle_index,
            verbalized_confidence=round(verbalized_confidence, 3),
            calibrated_confidence=calibrated_confidence,
            grounding_faithfulness_score=round(grounding_faithfulness_score, 3),
            numerical_verification_pass=verification_passed,
            discrepancies=discrepancies,
            backtracking_triggered=backtrack_triggered,
            backtracking_rationale=rationale,
            audit_verdict=audit_verdict,
        )
