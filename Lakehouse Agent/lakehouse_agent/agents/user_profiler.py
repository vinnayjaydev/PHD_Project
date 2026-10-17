"""
User Profiler Agent
Extracts investor financial health from the Lakehouse, evaluates risk capacity vs. tolerance,
and synthesizes liquidity requirements and life-stage investment constraints.
"""
from typing import Any, Dict, Optional
from lakehouse_agent.agents.base import BaseAgent
from lakehouse_agent.agents.llm_provider import LLMClient
from lakehouse_agent.lakehouse.engine import LakehouseEngine


class UserProfilerAgent(BaseAgent):
    """Specialized agent analyzing investor demographics, financial resilience, and behavioral risk."""

    def __init__(self, engine: LakehouseEngine, llm_client: Optional[LLMClient] = None):
        system_prompt = (
            "You are an Elite Certified Financial Planner (CFP) and Behavioral Finance Profiler. "
            "Your task is to objectively evaluate an investor's true risk capacity, liquidity constraints, "
            "debt burden, and life-stage objectives using verified data from the Lakehouse. "
            "Differentiate between stated risk tolerance (psychological appetite) and true risk capacity "
            "(financial ability to withstand drawdowns)."
        )
        super().__init__(
            name="UserProfileAgent",
            role="Investor Demographics & Behavioral Risk Profiler",
            description="Evaluates investor financial health, risk capacity, and liquidity runway from the Lakehouse.",
            system_prompt=system_prompt,
            llm_client=llm_client
        )
        self.engine = engine

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes profiling on a specific investor.
        context requires 'investor_id'.
        """
        self.clear_trace()
        investor_id = context.get("investor_id")
        self.log_step("fetch_lakehouse_profile", {"investor_id": investor_id})

        # Query Lakehouse Silver table
        records = self.engine.query_dicts(f"""
            SELECT * FROM silver_investor_financial_health 
            WHERE investor_id = '{investor_id}';
        """)

        if not records:
            raise ValueError(f"Investor ID '{investor_id}' not found in Lakehouse Silver layer.")

        profile_data = records[0]
        self.log_step("lakehouse_data_loaded", profile_data)

        # Risk capacity analysis
        age = profile_data["age"]
        dti = profile_data["debt_to_income_ratio"]
        runway = profile_data["emergency_fund_runway_months"]
        horizon = profile_data["investment_horizon_years"]
        stated_risk = profile_data["stated_risk_tolerance"]
        risk_score = profile_data["risk_score"]

        # Formulate prompt for LLM synthesis
        user_prompt = f"""
Analyze the following verified financial health data from our Lakehouse:
- Investor: {profile_data['name']} (Age: {age}, ID: {investor_id})
- Stated Risk Tolerance: {stated_risk} (Score: {risk_score}/100)
- Investment Horizon: {horizon} years
- Primary Goal: {profile_data['primary_goal']}
- Liquid Net Worth: ${profile_data['liquid_net_worth']:,.2f}
- Annual Income: ${profile_data['annual_income']:,.2f}
- Debt-to-Income (DTI): {dti}
- Emergency Fund Runway: {runway} months (Status: {profile_data['emergency_fund_status']})
- Monthly Savings Capacity: ${profile_data['monthly_savings_capacity']:,.2f}
- ESG Preference: {profile_data['esg_preference']}

Synthesize:
1. True Risk Capacity assessment vs Stated Tolerance.
2. Required Emergency Reserve adjustments.
3. Liquidity horizon constraints.
4. Strategic investment directives.
"""
        self.log_step("llm_reasoning_prompt", user_prompt)
        llm_analysis = self.llm.generate(self.system_prompt, user_prompt)
        self.log_step("llm_synthesis_complete", {"preview": llm_analysis[:200] + "..."})

        # Structured synthesis
        output = {
            "investor_id": investor_id,
            "name": profile_data["name"],
            "age": age,
            "risk_tier": profile_data["risk_tier"],
            "risk_score": risk_score,
            "stated_risk_tolerance": stated_risk,
            "investment_horizon_years": horizon,
            "primary_goal": profile_data["primary_goal"],
            "emergency_fund_runway_months": runway,
            "emergency_fund_status": profile_data["emergency_fund_status"],
            "debt_to_income_ratio": dti,
            "monthly_savings_capacity": profile_data["monthly_savings_capacity"],
            "esg_preference": profile_data["esg_preference"],
            "profiler_reasoning": llm_analysis,
            "trace": self.get_trace()
        }
        return output
