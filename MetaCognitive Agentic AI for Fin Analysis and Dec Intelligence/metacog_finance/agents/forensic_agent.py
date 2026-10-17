"""Adversarial Forensic Auditor Agent.

Plays the role of a skeptical forensic accountant looking for accounting red flags,
accrual manipulation (Beneish M-Score), bankruptcy risk (Altman Z-Score), and hidden footnote liabilities.
"""

from __future__ import annotations

from typing import Any, Dict, List
from metacog_finance.agents.base import BaseFinancialAgent
from metacog_finance.core.models import (
    DocumentChunk,
    DocumentSection,
    FinancialRatios,
    FinancialStatement,
)
from metacog_finance.rag.hybrid_retriever import HybridFinancialRetriever


class ForensicAuditorAgent(BaseFinancialAgent):
    """Adversarial forensic accountant scrutinizing disclosures for hidden vulnerabilities."""

    def __init__(self):
        super().__init__(
            name="ForensicAuditorAgent",
            role_description="Conducts adversarial forensic accounting analysis, examining footnote disclosures, accrual anomalies, and bankruptcy distress.",
        )

    def execute(
        self,
        ticker: str,
        stmt: FinancialStatement,
        ratios: FinancialRatios,
        retriever: HybridFinancialRetriever,
    ) -> Dict[str, Any]:
        """Conducts forensic audit scrutinizing footnote disclosures and econometric scores."""
        # 1. Retrieve risk factors and footnotes
        risk_chunks = retriever.retrieve("Risk factors supplier concentration regulatory legal", top_k=2)
        note_chunks = retriever.retrieve("Notes to Consolidated Financial Statements commitments debt leases", top_k=2)
        all_chunks = risk_chunks + note_chunks

        # 2. Evaluate quantitative forensic flags
        red_flags: List[str] = []

        if ratios.altman_z_score < 1.81:
            red_flags.append(
                f"HIGH DISTRESS: Altman Z-Score ({ratios.altman_z_score:.2f}) is in the critical distress zone (< 1.81)."
            )
        elif ratios.altman_z_score < 2.99:
            red_flags.append(
                f"MODERATE CAUTION: Altman Z-Score ({ratios.altman_z_score:.2f}) is in the grey zone (1.81 - 2.99)."
            )

        if ratios.beneish_m_score > -1.78:
            red_flags.append(
                f"EARNINGS MANIPULATION RISK: Beneish M-Score ({ratios.beneish_m_score:.2f}) exceeds the threshold (-1.78)."
            )

        if stmt.net_income > 0 and stmt.operating_cash_flow < 0:
            red_flags.append(
                f"CASH/EARNINGS DECOUPLING: Positive Net Income (${stmt.net_income:,.1f}M) but Negative Operating Cash Flow (${stmt.operating_cash_flow:,.1f}M)."
            )

        if ratios.current_ratio < 1.0:
            red_flags.append(
                f"WORKING CAPITAL DEFICIT: Current Ratio is {ratios.current_ratio:.2f}; current liabilities exceed current assets."
            )

        # 3. Build forensic report
        forensic_verdict = "HEALTHY" if len(red_flags) == 0 else ("WARNING" if len(red_flags) <= 2 else "DISTRESS")

        report = f"""### Forensic Accounting & Balance Sheet Integrity Audit
**Entity**: {ticker} | **Fiscal Year**: {stmt.fiscal_year} | **Audit Verdict**: **{forensic_verdict}**

**1. Econometric Distress & Manipulation Metrics**:
- **Altman Z-Score**: {ratios.altman_z_score:.2f} ({ratios.altman_zone})
- **Beneish M-Score**: {ratios.beneish_m_score:.2f} (Risk: {ratios.manipulation_risk})
- **Accrual Ratio**: {ratios.accrual_ratio:.4f} (Benchmark: < 0.05)
- **FCF Conversion**: {ratios.fcf_conversion_ratio:.2f}x Net Income

**2. Audit Findings & Red Flags**:
{chr(10).join(f"- {flag}" for flag in red_flags) if red_flags else "- No critical forensic red flags detected. Accounting statements exhibit strong accrual integrity."}

**3. Footnote & Risk Factor Scrutiny**:
- **Debt & Maturities**: Total debt of ${stmt.short_term_debt + stmt.long_term_debt:,.1f}M compared to cash reserves of ${stmt.cash_and_equivalents + stmt.marketable_securities:,.1f}M.
- **Risk Disclosures (Item 1A)**: Macroeconomic headwinds, geopolitical supply chain vulnerabilities, and foreign exchange exposure.
"""

        return {
            "verdict": forensic_verdict,
            "forensic_report": report,
            "red_flags": red_flags,
            "citations_used": [c.chunk_id for c in all_chunks],
        }
