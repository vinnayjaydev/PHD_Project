"""Decision Intelligence Synthesizer Agent.

Synthesizes quantitative metrics, qualitative fundamental thesis, forensic audit,
and meta-cognitive calibration into an actionable, stress-tested Decision Intelligence Memo.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List
from metacog_finance.agents.base import BaseFinancialAgent
from metacog_finance.core.models import (
    DecisionIntelligenceMemo,
    DecisionRecommendation,
    FinancialRatios,
    FinancialStatement,
    MetaCognitiveAudit,
)


class DecisionIntelligenceAgent(BaseFinancialAgent):
    """Synthesizes all cognitive and meta-cognitive streams into calibrated decision intelligence."""

    def __init__(self):
        super().__init__(
            name="DecisionIntelligenceAgent",
            role_description="Synthesizes investment recommendations, scenario stress-tests, and risk-calibrated allocation directives.",
        )

    def execute(
        self,
        ticker: str,
        stmt: FinancialStatement,
        ratios: FinancialRatios,
        fundamental_narrative: str,
        forensic_data: Dict[str, Any],
        audit_trace: List[MetaCognitiveAudit],
    ) -> DecisionIntelligenceMemo:
        """Constructs the final calibrated decision intelligence memo."""
        latest_audit = audit_trace[-1]
        calibrated_conf = latest_audit.calibrated_confidence

        # Determine confidence tier
        if calibrated_conf >= 0.85:
            conf_tier = "HIGH (>0.85)"
        elif calibrated_conf >= 0.65:
            conf_tier = "MODERATE (0.65-0.85)"
        else:
            conf_tier = "LOW (<0.65)"

        # Determine calibrated recommendation
        red_flags_count = len(forensic_data.get("red_flags", []))
        has_critical_discrepancy = any(
            d.severity.value == "CRITICAL" for audit in audit_trace for d in audit.discrepancies
        )

        if has_critical_discrepancy or ratios.altman_zone.startswith("Distress"):
            recommendation = DecisionRecommendation.CRITICAL_AVOID
        elif red_flags_count >= 2 or calibrated_conf < 0.60:
            recommendation = DecisionRecommendation.REDUCE_UNDERWEIGHT
        elif red_flags_count == 1:
            recommendation = DecisionRecommendation.HOLD_NEUTRAL
        elif ratios.return_on_invested_capital_roic > 15.0 and ratios.operating_margin > 15.0:
            recommendation = DecisionRecommendation.STRONG_BUY
        else:
            recommendation = DecisionRecommendation.ACCUMULATE

        # Formulate Scenario Analysis
        scenarios = {
            "Bull_Case": (
                f"Sustained pricing power drives ROIC toward {ratios.return_on_invested_capital_roic * 1.15:.1f}%. "
                f"Free cash flow conversion remains durable above {max(ratios.fcf_conversion_ratio, 0.8):.2f}x."
            ),
            "Base_Case": (
                f"Revenue stabilizes around ${stmt.revenue:,.1f}M with operating margin near {ratios.operating_margin:.1f}%. "
                f"Cash flow from operations covers capital expenditure requirements."
            ),
            "Bear_Case": (
                f"Macro deterioration compresses operating margins by 200 bps to {max(ratios.operating_margin - 2.0, 0.0):.1f}%. "
                f"Refinancing of ${stmt.short_term_debt:,.1f}M short-term debt incurs higher interest expense."
            ),
        }

        # Prescriptive Allocations
        if recommendation == DecisionRecommendation.STRONG_BUY:
            allocations = [
                "Overweight allocation recommended (target portfolio weight: 4.0% - 6.5%).",
                "Execute dollar-cost averaging tranches on technical pullbacks.",
                "Maintain trailing stop-loss at 12% below entry benchmark.",
            ]
        elif recommendation == DecisionRecommendation.ACCUMULATE:
            allocations = [
                "Benchmark neutral to moderate overweight allocation (target weight: 2.5% - 4.0%).",
                "Reinvest dividends; monitor quarterly working capital trends.",
            ]
        elif recommendation == DecisionRecommendation.HOLD_NEUTRAL:
            allocations = [
                "Hold existing position; avoid incremental capital deployment.",
                "Hedge downside tail risk with out-of-the-money protective put options.",
            ]
        elif recommendation == DecisionRecommendation.REDUCE_UNDERWEIGHT:
            allocations = [
                "Trim portfolio allocation by 30% - 50% to mitigate capital drawdown.",
                "Reallocate proceeds to high-liquidity short-duration sovereign instruments.",
            ]
        else:  # CRITICAL_AVOID
            allocations = [
                "Immediate liquidation or exclusion from approved institutional buy-list.",
                "Strict capital preservation mandate; avoid unsecured credit exposure.",
            ]

        # Executive summary
        exec_summary = (
            f"Decision Intelligence synthesis for {ticker} (FY{stmt.fiscal_year}) establishes a "
            f"**{recommendation.value}** recommendation backed by calibrated epistemic confidence of "
            f"{calibrated_conf*100:.1f}% ({conf_tier}). "
            f"Deterministic forensic audit produced a verdict of {forensic_data.get('verdict', 'REVIEW')}, "
            f"reflecting an Altman Z-Score of {ratios.altman_z_score:.2f} and Beneish M-Score of {ratios.beneish_m_score:.2f}."
        )

        return DecisionIntelligenceMemo(
            ticker=ticker,
            fiscal_year=stmt.fiscal_year,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            recommendation=recommendation,
            calibrated_confidence=calibrated_conf,
            confidence_tier=conf_tier,
            executive_thesis=exec_summary,
            forensic_health_assessment=forensic_data.get("forensic_report", ""),
            quantitative_summary={
                "Revenue": f"${stmt.revenue:,.1f}M",
                "Operating_Income": f"${stmt.operating_income:,.1f}M",
                "Net_Income": f"${stmt.net_income:,.1f}M",
                "Operating_Cash_Flow": f"${stmt.operating_cash_flow:,.1f}M",
                "Free_Cash_Flow": f"${stmt.free_cash_flow:,.1f}M",
                "Gross_Margin": f"{ratios.gross_margin:.1f}%",
                "Operating_Margin": f"{ratios.operating_margin:.1f}%",
                "ROIC": f"{ratios.return_on_invested_capital_roic:.1f}%",
                "Current_Ratio": f"{ratios.current_ratio:.2f}",
                "Altman_Z": f"{ratios.altman_z_score:.2f} ({ratios.altman_zone})",
                "Beneish_M": f"{ratios.beneish_m_score:.2f} ({ratios.manipulation_risk})",
            },
            epistemic_discrepancies_detected=latest_audit.discrepancies,
            scenario_analysis=scenarios,
            prescriptive_allocations=allocations,
            meta_cognitive_trace=audit_trace,
            citation_index={
                "Income_Statement": "SEC Form 10-K Item 8 (Operations)",
                "Balance_Sheet": "SEC Form 10-K Item 8 (Consolidated Balance Sheets)",
                "Cash_Flows": "SEC Form 10-K Item 8 (Consolidated Statements of Cash Flows)",
                "MD_and_A": "SEC Form 10-K Item 7 (MD&A)",
                "Footnotes": "SEC Form 10-K Notes to Financial Statements",
            },
        )
