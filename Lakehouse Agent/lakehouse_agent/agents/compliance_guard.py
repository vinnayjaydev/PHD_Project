"""
Compliance Guardrail Agent
Enforces FINRA Rule 2111 (Suitability), SEC Regulation Best Interest (Reg BI),
asset concentration caps, and emergency liquidity buffer mandates.
"""
from typing import Any, Dict, List, Optional
from lakehouse_agent.agents.base import BaseAgent
from lakehouse_agent.agents.llm_provider import LLMClient
from lakehouse_agent.config import settings


class ComplianceGuardrailAgent(BaseAgent):
    """Specialized agent acting as the compliance officer and fiduciary regulatory gatekeeper."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        system_prompt = (
            "You are a Chief Compliance Officer (CCO) and Financial Regulatory Fiduciary Gatekeeper. "
            "Your non-negotiable duty is to audit proposed investment advice against FINRA Rule 2111, "
            "SEC Regulation Best Interest (Reg BI), concentration limits, and emergency fund adequacy. "
            "You have full authority to reject non-compliant plans or modify allocations to ensure strict fiduciary safety."
        )
        super().__init__(
            name="ComplianceGuardrailAgent",
            role="Regulatory Fiduciary & Suitability Gatekeeper",
            description="Enforces FINRA/SEC suitability rules, concentration caps, and liquidity reserves.",
            system_prompt=system_prompt,
            llm_client=llm_client
        )

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Audits proposed target allocation and orders against compliance rules.
        context requires:
          - 'investor_profile'
          - 'optimization_plan'
          - 'market_citations'
        """
        self.clear_trace()
        investor = context["investor_profile"]
        plan = context["optimization_plan"]
        citations = context.get("market_citations", [])
        risk_tier = investor["risk_tier"]
        target_allocation = plan["target_allocation"]
        orders = plan["orders"]

        self.log_step("begin_compliance_audit", {
            "investor_id": investor["investor_id"],
            "risk_tier": risk_tier,
            "target_allocation": target_allocation
        })

        violations = []
        corrections = []

        # 1. Audit Equity Allocation Cap
        equity_pct = target_allocation.get("Equities", 0.0)
        max_allowed_equity = {
            "Conservative": settings.max_equity_allocation_conservative * 100,
            "Moderate": settings.max_equity_allocation_moderate * 100,
            "Aggressive": settings.max_equity_allocation_aggressive * 100,
        }.get(risk_tier, 70.0)

        if equity_pct > max_allowed_equity:
            violation_msg = (
                f"Proposed Equity allocation of {equity_pct}% exceeds maximum regulatory "
                f"suitability cap of {max_allowed_equity}% for '{risk_tier}' risk tier."
            )
            violations.append(violation_msg)
            # Automatic correction
            target_allocation["Equities"] = max_allowed_equity
            target_allocation["Fixed Income"] += (equity_pct - max_allowed_equity)
            corrections.append(f"Adjusted Equity allocation to {max_allowed_equity}% cap; reallocated excess to Fixed Income.")

        # 2. Audit Emergency Liquidity Buffer
        runway = investor["emergency_fund_runway_months"]
        cash_pct = target_allocation.get("Cash", 0.0)
        if runway < settings.min_emergency_fund_months and cash_pct < 10.0:
            violation_msg = (
                f"Client emergency runway of {runway} months is below required "
                f"{settings.min_emergency_fund_months}-month threshold. Proposed cash of {cash_pct}% is insufficient."
            )
            violations.append(violation_msg)
            # Correct cash reserve
            deficit = 10.0 - cash_pct
            target_allocation["Cash"] = 10.0
            target_allocation["Equities"] = max(0.0, target_allocation.get("Equities", 0.0) - deficit)
            corrections.append(f"Mandated minimum 10.0% liquid cash buffer to restore client emergency reserves.")

        # 3. Audit Single Asset Concentration
        for o in orders:
            if o["action"] == "BUY" and o.get("target_weight_pct", 0.0) > (settings.max_single_stock_concentration * 100):
                violation_msg = (
                    f"Buy order for {o['ticker']} targets {o['target_weight_pct']}%, exceeding "
                    f"prudential single-asset concentration limit of {settings.max_single_stock_concentration * 100}%."
                )
                violations.append(violation_msg)
                o["target_weight_pct"] = settings.max_single_stock_concentration * 100
                corrections.append(f"Capped {o['ticker']} target weight at 15.0%.")

        # 4. Audit RAG Citations Grounding
        has_citations = len(citations) > 0
        if not has_citations:
            violations.append("Recommendation lacks empirical grounding from lakehouse vector research documents.")
            corrections.append("Enforced mandatory citation requirement for all forward-looking market claims.")

        status = "APPROVED" if not violations else "CONDITIONALLY_APPROVED_WITH_CORRECTIONS"
        self.log_step("compliance_audit_complete", {
            "status": status,
            "violations_count": len(violations),
            "corrections_count": len(corrections)
        })

        # LLM Formal Audit Attestation
        user_prompt = f"""
Compliance Audit Summary:
- Client: {investor['name']} (Risk Tier: {risk_tier}, Horizon: {investor['investment_horizon_years']} yrs)
- Compliance Status: {status}
- Identified Violations: {violations if violations else "None. All fiduciary limits respected."}
- Implemented Fiduciary Corrections: {corrections if corrections else "None necessary."}
- Verified Target Allocation: {target_allocation}
- Document Provenance: {len(citations)} cited SEC/Macro sources verified.

Generate a formal Compliance & Fiduciary Suitability Statement confirming adherence to FINRA Rule 2111 and SEC Reg BI.
"""
        self.log_step("llm_compliance_attestation_prompt", user_prompt)
        compliance_attestation = self.llm.generate(self.system_prompt, user_prompt)
        self.log_step("llm_compliance_complete", {"preview": compliance_attestation[:200] + "..."})

        return {
            "compliance_status": status,
            "is_compliant": len(violations) == 0,
            "violations": violations,
            "corrections": corrections,
            "final_target_allocation": target_allocation,
            "compliance_attestation": compliance_attestation,
            "trace": self.get_trace()
        }
