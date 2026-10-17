"""Epistemic Discrepancy Detector.

Monitors dissonance between qualitative narrative claims (e.g. from MD&A or earnings calls)
and deterministic quantitative reality extracted from audited financial statements.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional
from metacog_finance.core.models import (
    DiscrepancySeverity,
    EpistemicDiscrepancy,
    FinancialRatios,
    FinancialStatement,
)


class EpistemicDiscrepancyDetector:
    """Supervisory monitor that identifies narrative-vs-quantitative discrepancies."""

    def __init__(self):
        # Patterns for positive/bullish rhetoric
        self.bullish_liquidity_patterns = [
            r"robust liquidity",
            r"strong liquidity position",
            r"ample cash reserves",
            r"well-positioned balance sheet",
            r"exceptional financial strength",
        ]
        self.bullish_growth_patterns = [
            r"strong top-line growth",
            r"accelerating revenue",
            r"record sales momentum",
            r"unprecedented demand",
        ]
        self.bullish_margin_patterns = [
            r"expanding margins",
            r"pricing power",
            r"margin resilience",
            r"operational cost discipline",
        ]

    def detect_discrepancies(
        self,
        narrative_text: str,
        stmt: FinancialStatement,
        ratios: FinancialRatios,
        prior_stmt: Optional[FinancialStatement] = None,
    ) -> List[EpistemicDiscrepancy]:
        """Detects discrepancies between claims in `narrative_text` and quantitative ground truth."""
        discrepancies: List[EpistemicDiscrepancy] = []
        lower_narrative = narrative_text.lower()

        # 1. Check Liquidity Claims vs Current/Quick Ratio
        claims_strong_liquidity = any(re.search(p, lower_narrative) for p in self.bullish_liquidity_patterns)
        if claims_strong_liquidity:
            if ratios.current_ratio < 1.0 or ratios.quick_ratio < 0.7:
                discrepancies.append(
                    EpistemicDiscrepancy(
                        discrepancy_id=f"DISC_LIQ_{len(discrepancies)+1}",
                        category="Narrative_vs_Liquidity",
                        claim_in_narrative="Management or analyst asserts strong/robust liquidity position",
                        ground_truth_metric="Current Ratio & Quick Ratio",
                        ground_truth_value=ratios.current_ratio,
                        reported_period=f"FY{stmt.fiscal_year}",
                        severity=DiscrepancySeverity.HIGH if ratios.current_ratio < 0.85 else DiscrepancySeverity.MEDIUM,
                        explanation=(
                            f"Narrative asserts robust liquidity, but Current Ratio is {ratios.current_ratio:.2f} "
                            f"(< 1.00) and Quick Ratio is {ratios.quick_ratio:.2f}. "
                            "Short-term obligations exceed immediately liquid resources."
                        ),
                        source_citation="SEC Form 10-K Balance Sheet",
                    )
                )

        # 2. Check Earnings Quality: Net Income vs Operating Cash Flow
        # If Net Income is positive, but Operating Cash Flow is negative or significantly lower (< 50% of NI)
        if stmt.net_income > 100.0:
            if stmt.operating_cash_flow < 0:
                discrepancies.append(
                    EpistemicDiscrepancy(
                        discrepancy_id=f"DISC_CASH_{len(discrepancies)+1}",
                        category="Earnings_Quality_Divergence",
                        claim_in_narrative="Profitable operational performance reported",
                        ground_truth_metric="Operating Cash Flow",
                        ground_truth_value=stmt.operating_cash_flow,
                        reported_period=f"FY{stmt.fiscal_year}",
                        severity=DiscrepancySeverity.CRITICAL,
                        explanation=(
                            f"Company reported positive Net Income of ${stmt.net_income:,.1f}M, "
                            f"yet Operating Cash Flow is NEGATIVE (${stmt.operating_cash_flow:,.1f}M). "
                            f"Accrual ratio is {ratios.accrual_ratio:.4f}, signaling severe divergence between "
                            "accounting profit and actual cash collection."
                        ),
                        source_citation="SEC Form 10-K Cash Flow Statement",
                    )
                )
            elif ratios.fcf_conversion_ratio < 0.40 and stmt.net_income > 500:
                discrepancies.append(
                    EpistemicDiscrepancy(
                        discrepancy_id=f"DISC_CASH_{len(discrepancies)+1}",
                        category="Low_FCF_Conversion",
                        claim_in_narrative="High bottom-line earnings quality",
                        ground_truth_metric="FCF Conversion Ratio",
                        ground_truth_value=ratios.fcf_conversion_ratio,
                        reported_period=f"FY{stmt.fiscal_year}",
                        severity=DiscrepancySeverity.HIGH,
                        explanation=(
                            f"Free Cash Flow conversion is only {ratios.fcf_conversion_ratio:.2f}x Net Income "
                            "(threshold >= 0.70x). Cash conversion is substantially impaired."
                        ),
                        source_citation="SEC Form 10-K Statement of Cash Flows",
                    )
                )

        # 3. Check Margin Rhetoric vs Actual Margin Trajectory
        claims_margin_strength = any(re.search(p, lower_narrative) for p in self.bullish_margin_patterns)
        if claims_margin_strength and prior_stmt and prior_stmt.revenue > 0:
            prior_op_margin = (prior_stmt.operating_income / prior_stmt.revenue) * 100
            curr_op_margin = ratios.operating_margin
            margin_change = curr_op_margin - prior_op_margin
            if margin_change < -1.5:  # Margin contracted by > 150 bps
                discrepancies.append(
                    EpistemicDiscrepancy(
                        discrepancy_id=f"DISC_MARG_{len(discrepancies)+1}",
                        category="Margin_Contradiction",
                        claim_in_narrative="Management narrative highlights margin resilience or pricing power",
                        ground_truth_metric="Operating Margin YoY Change",
                        ground_truth_value=round(margin_change, 2),
                        reported_period=f"FY{stmt.fiscal_year}",
                        severity=DiscrepancySeverity.MEDIUM,
                        explanation=(
                            f"Narrative emphasizes margin strength, but Operating Margin compressed by "
                            f"{abs(margin_change):.2f}% (from {prior_op_margin:.2f}% to {curr_op_margin:.2f}%)."
                        ),
                        source_citation="SEC Form 10-K Consolidated Statements of Operations",
                    )
                )

        # 4. Check Beneish Manipulation & Altman Distress Risk
        if ratios.beneish_m_score > -1.78:
            discrepancies.append(
                EpistemicDiscrepancy(
                    discrepancy_id=f"DISC_FORENSIC_{len(discrepancies)+1}",
                    category="Earnings_Manipulation_Probability",
                    claim_in_narrative="Standard financial reporting without qualification",
                    ground_truth_metric="Beneish M-Score",
                    ground_truth_value=ratios.beneish_m_score,
                    reported_period=f"FY{stmt.fiscal_year}",
                    severity=DiscrepancySeverity.CRITICAL,
                    explanation=(
                        f"Beneish M-Score is {ratios.beneish_m_score:.2f} (above critical threshold -1.78). "
                        "Statistical indicators flag high probability of earnings manipulation or aggressive accruals."
                    ),
                    source_citation="Forensic Econometric Model",
                )
            )

        if ratios.altman_z_score < 1.81:
            discrepancies.append(
                EpistemicDiscrepancy(
                    discrepancy_id=f"DISC_DISTRESS_{len(discrepancies)+1}",
                    category="Bankruptcy_Distress_Zone",
                    claim_in_narrative="Ongoing enterprise viability",
                    ground_truth_metric="Altman Z-Score",
                    ground_truth_value=ratios.altman_z_score,
                    reported_period=f"FY{stmt.fiscal_year}",
                    severity=DiscrepancySeverity.HIGH,
                    explanation=(
                        f"Altman Z-Score is {ratios.altman_z_score:.2f}, placing the firm firmly in the "
                        "Distress Zone (< 1.81). Substantial probability of insolvency within 24 months."
                    ),
                    source_citation="Forensic Econometric Model",
                )
            )

        return discrepancies
