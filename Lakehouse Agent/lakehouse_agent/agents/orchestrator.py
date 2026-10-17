"""
Financial Advisor Orchestrator
Master coordinator governing the multi-agent cognitive pipeline:
Profiler -> Lakehouse Analytics -> Market RAG Intelligence -> Optimizer -> Compliance -> Final Advisory.
Persists execution logs and audit trails into the Lakehouse Gold tier.
"""
import uuid
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from lakehouse_agent.lakehouse.engine import LakehouseEngine
from lakehouse_agent.vector_store.store import FinancialVectorStore
from lakehouse_agent.agents.llm_provider import LLMClient
from lakehouse_agent.agents.user_profiler import UserProfilerAgent
from lakehouse_agent.agents.lakehouse_analyst import LakehouseAnalyticsAgent
from lakehouse_agent.agents.market_intelligence import MarketIntelligenceAgent
from lakehouse_agent.agents.portfolio_optimizer import PortfolioOptimizerAgent
from lakehouse_agent.agents.compliance_guard import ComplianceGuardrailAgent


class FinancialAdvisorOrchestrator:
    """Master Multi-Agent Orchestrator for Adaptive Personalized Financial Recommendations."""

    def __init__(
        self,
        engine: LakehouseEngine,
        vector_store: FinancialVectorStore,
        llm_client: Optional[LLMClient] = None
    ):
        self.engine = engine
        self.vector_store = vector_store
        self.llm = llm_client or LLMClient()

        # Initialize the 5 specialized agents
        self.profiler_agent = UserProfilerAgent(engine=engine, llm_client=self.llm)
        self.analyst_agent = LakehouseAnalyticsAgent(engine=engine, llm_client=self.llm)
        self.market_agent = MarketIntelligenceAgent(engine=engine, vector_store=vector_store, llm_client=self.llm)
        self.optimizer_agent = PortfolioOptimizerAgent(engine=engine, llm_client=self.llm)
        self.compliance_agent = ComplianceGuardrailAgent(llm_client=self.llm)

    def run_advisory_pipeline(self, investor_id: str) -> Dict[str, Any]:
        """
        Executes the complete end-to-end multi-agent recommendation lifecycle for an investor.
        """
        rec_id = f"REC_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
        pipeline_start = datetime.now()

        # Step 1: User Profiler Agent
        profile_result = self.profiler_agent.execute({"investor_id": investor_id})

        # Step 2: Lakehouse Analytics Agent
        analytics_result = self.analyst_agent.execute({"investor_id": investor_id})

        # Step 3: Market Intelligence & RAG Agent
        holdings_tickers = [h["ticker"] for h in analytics_result["holdings"]]
        market_result = self.market_agent.execute({
            "tickers": holdings_tickers,
            "risk_tier": profile_result["risk_tier"],
            "primary_goal": profile_result["primary_goal"]
        })

        # Step 4: Quantitative Portfolio Optimizer Agent
        optimizer_result = self.optimizer_agent.execute({
            "investor_profile": profile_result,
            "portfolio_analytics": analytics_result,
            "market_intelligence": market_result
        })

        # Step 5: Compliance Guardrail Agent
        compliance_result = self.compliance_agent.execute({
            "investor_profile": profile_result,
            "optimization_plan": optimizer_result,
            "market_citations": market_result["citations"]
        })

        # Step 6: Master Executive Advisory Synthesis
        executive_summary = self._synthesize_master_advisory(
            profile=profile_result,
            analytics=analytics_result,
            market=market_result,
            optimizer=optimizer_result,
            compliance=compliance_result
        )

        pipeline_duration = (datetime.now() - pipeline_start).total_seconds()

        # Compile final consolidated report
        final_report = {
            "recommendation_id": rec_id,
            "investor_id": investor_id,
            "investor_name": profile_result["name"],
            "timestamp": pipeline_start.isoformat(),
            "duration_seconds": round(pipeline_duration, 2),
            "risk_tier": profile_result["risk_tier"],
            "primary_goal": profile_result["primary_goal"],
            "current_portfolio_value": analytics_result["portfolio_summary"]["total_portfolio_value"],
            "current_allocation": {
                "Equities": analytics_result["portfolio_summary"]["equity_pct"],
                "Fixed Income": analytics_result["portfolio_summary"]["bond_pct"],
                "Real Estate": analytics_result["portfolio_summary"]["real_estate_pct"],
                "Commodities": analytics_result["portfolio_summary"]["commodities_pct"],
                "Cash": analytics_result["portfolio_summary"]["cash_pct"],
            },
            "target_allocation": compliance_result["final_target_allocation"],
            "orders": optimizer_result["orders"],
            "projected_metrics": optimizer_result["projected_metrics"],
            "compliance": {
                "status": compliance_result["compliance_status"],
                "is_compliant": compliance_result["is_compliant"],
                "violations": compliance_result["violations"],
                "corrections": compliance_result["corrections"],
                "attestation": compliance_result["compliance_attestation"]
            },
            "rag_citations": market_result["citations"],
            "retrieved_excerpts": market_result["retrieved_chunks"],
            "executive_summary": executive_summary,
            "agent_traces": {
                "user_profiler": profile_result["trace"],
                "lakehouse_analytics": analytics_result["trace"],
                "market_intelligence": market_result["trace"],
                "portfolio_optimizer": optimizer_result["trace"],
                "compliance_guard": compliance_result["trace"]
            }
        }

        # Persist audit record to Lakehouse Gold table
        self._save_recommendation_to_lakehouse(final_report)

        return final_report

    def _synthesize_master_advisory(
        self,
        profile: Dict[str, Any],
        analytics: Dict[str, Any],
        market: Dict[str, Any],
        optimizer: Dict[str, Any],
        compliance: Dict[str, Any]
    ) -> str:
        """Synthesizes an executive, client-ready advisory report with citations."""
        system_prompt = (
            "You are an Elite Private Wealth Management Advisor delivering a personalized, "
            "empirically grounded financial advisory strategy to a high-net-worth client. "
            "Synthesize the multi-agent findings into a polished, authoritative, actionable briefing. "
            "Integrate the retrieved citations smoothly, outline the exact rebalancing actions, "
            "and emphasize the fiduciary compliance certification."
        )

        user_prompt = f"""
Synthesize the master advisory plan for:
Client: {profile['name']} (Age: {profile['age']}, Risk Tier: {profile['risk_tier']})
Primary Goal: {profile['primary_goal']}
Portfolio Size: ${analytics['portfolio_summary']['total_portfolio_value']:,.2f}

Agent Findings:
1. Profiler: Runway is {profile['emergency_fund_runway_months']} mos ({profile['emergency_fund_status']}). Savings capacity ${profile['monthly_savings_capacity']:,.2f}/mo.
2. Analytics: Current Beta is {analytics['portfolio_summary']['portfolio_beta']}, Risk drift: {analytics['portfolio_summary']['risk_drift_status']}. Flagged positions: {[f"{p['ticker']} ({p['weight_pct']}%)" for p in analytics['flagged_positions']]}.
3. Market & RAG Intelligence: {market['market_intelligence_brief'][:350]}...
   Citations: {market['citations']}
4. Optimizer: Rebalancing target: {compliance['final_target_allocation']}. Projected Beta: {optimizer['projected_metrics']['target_beta']}.
   Trades: {[f"{o['action']} {o['shares']}x {o['ticker']} (${o['estimated_amount']:,.2f})" for o in optimizer['orders']]}
5. Compliance: Status {compliance['compliance_status']}. Violations resolved: {compliance['corrections']}

Deliver an Executive Financial Advisory Briefing containing:
1. Strategic Executive Overview
2. Asset Allocation & Rebalancing Roadmap
3. Grounded Market Context & RAG Evidence (citing sources)
4. Fiduciary Compliance & Downside Risk Protection
"""
        return self.llm.generate(system_prompt, user_prompt)

    def _save_recommendation_to_lakehouse(self, report: Dict[str, Any]):
        """Inserts the recommendation report into gold_recommendation_history."""
        try:
            self.engine.execute("""
                INSERT INTO gold_recommendation_history VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                );
            """, [
                report["recommendation_id"],
                report["investor_id"],
                datetime.fromisoformat(report["timestamp"]),
                "SUCCESS",
                report["risk_tier"],
                json.dumps(report["current_allocation"]),
                json.dumps(report["target_allocation"]),
                json.dumps(report["orders"]),
                report["compliance"]["status"],
                json.dumps(report["rag_citations"]),
                report["executive_summary"]
            ])
        except Exception as e:
            print(f"[Orchestrator] Warning: could not persist to gold_recommendation_history: {e}")
