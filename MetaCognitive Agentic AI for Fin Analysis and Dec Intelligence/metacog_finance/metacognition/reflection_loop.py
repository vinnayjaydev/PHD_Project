"""Adaptive Reflection and Backtracking Controller.

Generates targeted secondary retrieval queries, critiques draft analyses,
and governs iterative self-correction loops when confidence or verification fails.
"""

from __future__ import annotations

from typing import List, Optional
from metacog_finance.core.models import (
    DiscrepancySeverity,
    EpistemicDiscrepancy,
    MetaCognitiveAudit,
)


class ReflectionLoopController:
    """Orchestrates adaptive backtracking and targeted secondary inquiries."""

    def __init__(self, max_iterations: int = 2):
        self.max_iterations = max_iterations

    def formulate_backtracking_plan(
        self,
        audit: MetaCognitiveAudit,
        current_iteration: int,
    ) -> Optional[dict]:
        """Examines the audit results and formulates a targeted corrective inquiry plan."""
        if not audit.backtracking_triggered:
            return None

        if current_iteration >= self.max_iterations:
            return {
                "action": "HALT_WITH_CAUTION",
                "reason": f"Maximum backtracking iterations ({self.max_iterations}) reached.",
                "critique": "Analysis remains subject to high epistemic uncertainty; downgrade final rating to CAUTION/HIGH_RISK.",
                "secondary_query": None,
            }

        targeted_queries: List[str] = []
        critique_points: List[str] = []

        for d in audit.discrepancies:
            if d.category == "Earnings_Quality_Divergence":
                targeted_queries.append("Statement of Cash Flows reconciliation and accounts receivable changes")
                critique_points.append(
                    f"CRITICAL WARNING: Cash flow divergence detected. Claim '{d.claim_in_narrative}' contradicts "
                    f"Operating Cash Flow (${d.ground_truth_value:,.1f}M). Force downward adjustment of earnings sustainability."
                )
            elif d.category == "Narrative_vs_Liquidity":
                targeted_queries.append("Item 8 Note: Short-term debt maturities, credit facilities, and working capital")
                critique_points.append(
                    f"LIQUIDITY WARNING: Current ratio is {d.ground_truth_value:.2f}. Management claims of robust liquidity "
                    "are unsubstantiated. Investigate covenant compliance and debt repayment schedule."
                )
            elif d.category == "Earnings_Manipulation_Probability":
                targeted_queries.append("Footnotes: Revenue recognition policy, unbilled receivables, and capitalized costs")
                critique_points.append(
                    f"FORENSIC ALERT: Beneish M-Score of {d.ground_truth_value:.2f} flags potential earnings management. "
                    "Scrutinize DSRI (receivables surge) and asset capitalization."
                )
            elif d.category == "Bankruptcy_Distress_Zone":
                targeted_queries.append("Liquidity and capital resources: Debt covenants, going concern notices")
                critique_points.append(
                    f"DISTRESS WARNING: Altman Z-score ({d.ground_truth_value:.2f}) indicates severe distress zone. "
                    "Assess debt restructuring risk."
                )

        if not audit.numerical_verification_pass:
            critique_points.append(
                "ARITHMETIC INTEGRITY: Numerical identities failed reconciliation. Ensure all quoted figures match audited tables exactly."
            )

        query = " AND ".join(targeted_queries) if targeted_queries else "Item 8 Footnotes and Significant Accounting Policies"
        composite_critique = "\n- ".join(critique_points)

        return {
            "action": "ITERATIVE_REFINEMENT",
            "iteration": current_iteration + 1,
            "secondary_query": query,
            "critique": f"Meta-Cognitive Auditor Feedback:\n- {composite_critique}",
        }
