"""
Lakehouse Analytics Agent
Executes OLAP queries on DuckDB Gold & Silver marts, calculating asset exposures,
portfolio beta, diversification scores, and concentration violations.
"""
from typing import Any, Dict, List, Optional
from lakehouse_agent.agents.base import BaseAgent
from lakehouse_agent.agents.llm_provider import LLMClient
from lakehouse_agent.lakehouse.engine import LakehouseEngine


class LakehouseAnalyticsAgent(BaseAgent):
    """Specialized agent performing quantitative queries and portfolio risk diagnostics on the Lakehouse."""

    def __init__(self, engine: LakehouseEngine, llm_client: Optional[LLMClient] = None):
        system_prompt = (
            "You are a Quantitative Portfolio Analyst and Lakehouse Data Specialist. "
            "Your objective is to inspect portfolio analytics from DuckDB Gold and Silver marts, "
            "identify asset allocation skews, calculate weighted risk exposures (Beta, HHI Concentration), "
            "and diagnose discrepancies between the client's current portfolio and their target risk profile."
        )
        super().__init__(
            name="LakehouseAnalyticsAgent",
            role="Quantitative Portfolio Diagnostics & Lakehouse Query Specialist",
            description="Executes OLAP queries on DuckDB to inspect current holdings, beta, diversification, and risk drift.",
            system_prompt=system_prompt,
            llm_client=llm_client
        )
        self.engine = engine

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes quantitative portfolio diagnostics.
        context requires 'investor_id'.
        """
        self.clear_trace()
        investor_id = context.get("investor_id")
        self.log_step("query_gold_portfolio_analytics", {"investor_id": investor_id})

        # 1. Fetch Gold Portfolio Analytics
        analytics = self.engine.query_dicts(f"""
            SELECT * FROM gold_portfolio_analytics 
            WHERE investor_id = '{investor_id}';
        """)

        if not analytics:
            raise ValueError(f"No analytics found for investor '{investor_id}' in gold_portfolio_analytics.")

        portfolio_summary = analytics[0]
        self.log_step("gold_analytics_loaded", portfolio_summary)

        # 2. Fetch Detailed Silver Holdings
        holdings = self.engine.query_dicts(f"""
            SELECT ticker, asset_name, asset_class, sector, shares, 
                   current_price, current_market_value, weight_pct, 
                   beta, dividend_yield, unrealized_return_pct, concentration_status
            FROM silver_portfolio_holdings
            WHERE investor_id = '{investor_id}'
            ORDER BY weight_pct DESC;
        """)
        self.log_step("silver_holdings_loaded", {"total_positions": len(holdings)})

        # 3. Detect concentration flags
        flagged_positions = [h for h in holdings if h["concentration_status"] != "Compliant"]
        self.log_step("concentration_check", {"flagged_count": len(flagged_positions)})

        # LLM Diagnostic prompt
        user_prompt = f"""
Inspect the following quantitative Lakehouse analytics for {portfolio_summary['name']} (ID: {investor_id}):
- Total Portfolio Value: ${portfolio_summary['total_portfolio_value']:,.2f}
- Portfolio Weighted Beta: {portfolio_summary['portfolio_beta']} (Market Benchmark: 1.00)
- Weighted Dividend Yield: {portfolio_summary['portfolio_dividend_yield_pct']}%
- Diversification Score: {portfolio_summary['diversification_score']}/100
- Risk Drift Status: {portfolio_summary['risk_drift_status']}
- Asset Allocation:
    * Equities: {portfolio_summary['equity_pct']}%
    * Fixed Income: {portfolio_summary['bond_pct']}%
    * Real Estate: {portfolio_summary['real_estate_pct']}%
    * Commodities: {portfolio_summary['commodities_pct']}%
    * Cash Equivalents: {portfolio_summary['cash_pct']}%

Individual Holdings Breakdown:
{self._format_holdings_for_prompt(holdings)}

Identified Flagged Positions:
{[f"{p['ticker']} ({p['weight_pct']}% of portfolio - {p['concentration_status']})" for p in flagged_positions]}

Provide an analytical diagnostic of:
1. Asset allocation imbalances and beta deviation.
2. Single-stock / sector concentration risks.
3. Quantified rebalancing necessity score (1-10) and primary portfolio vulnerabilities.
"""
        self.log_step("llm_diagnostic_prompt", user_prompt)
        llm_diagnostic = self.llm.generate(self.system_prompt, user_prompt)
        self.log_step("llm_diagnostic_complete", {"preview": llm_diagnostic[:200] + "..."})

        return {
            "investor_id": investor_id,
            "portfolio_summary": portfolio_summary,
            "holdings": holdings,
            "flagged_positions": flagged_positions,
            "analytics_diagnostic": llm_diagnostic,
            "trace": self.get_trace()
        }

    def _format_holdings_for_prompt(self, holdings: List[Dict[str, Any]]) -> str:
        lines = []
        for h in holdings:
            lines.append(
                f"- {h['ticker']} ({h['asset_name']}): ${h['current_market_value']:,.2f} "
                f"({h['weight_pct']}%), Beta: {h['beta']}, PnL: {h['unrealized_return_pct']}%"
            )
        return "\n".join(lines)
