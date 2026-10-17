"""Meta-Cognitive supervisory package."""

from metacog_finance.metacognition.calibrator import ConfidenceCalibrator
from metacog_finance.metacognition.financial_metrics import (
    compute_altman_z_score,
    compute_beneish_m_score,
    compute_financial_ratios,
)
from metacog_finance.metacognition.monitor import EpistemicDiscrepancyDetector
from metacog_finance.metacognition.reflection_loop import ReflectionLoopController
from metacog_finance.metacognition.verifier import (
    DeterministicFinancialVerifier,
    VerificationResult,
)

__all__ = [
    "compute_financial_ratios",
    "compute_altman_z_score",
    "compute_beneish_m_score",
    "DeterministicFinancialVerifier",
    "VerificationResult",
    "EpistemicDiscrepancyDetector",
    "ConfidenceCalibrator",
    "ReflectionLoopController",
]
