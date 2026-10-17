"""
Market Intelligence & RAG Agent
Executes hybrid semantic search over SEC filings, macroeconomic prints, and equity research.
Grounds recommendations with cited evidence and provides market regime context.
"""
from typing import Any, Dict, List, Optional
from lakehouse_agent.agents.base import BaseAgent
from lakehouse_agent.agents.llm_provider import LLMClient
from lakehouse_agent.vector_store.store import FinancialVectorStore
from lakehouse_agent.lakehouse.engine import LakehouseEngine


class MarketIntelligenceAgent(BaseAgent):
    """Specialized agent performing semantic RAG retrieval and macro regime synthesis."""

    def __init__(
        self,
        engine: LakehouseEngine,
        vector_store: FinancialVectorStore,
        llm_client: Optional[LLMClient] = None
    ):
        system_prompt = (
            "You are a Senior Macro Strategist and Financial Research Specialist. "
            "Your role is to analyze retrieved unstructured SEC filings, Federal Reserve statements, "
            "and analyst notes to provide grounded market regime intelligence. "
            "You MUST explicitly cite retrieved sources [Source | Title (Ref: ChunkID)] for any claims, "
            "ensuring strict grounding and preventing hallucinations."
        )
        super().__init__(
            name="MarketIntelligenceAgent",
            role="Macro Strategist & RAG Knowledge Retrieval Specialist",
            description="Performs semantic vector search across SEC filings and macro reports to ground investment decisions.",
            system_prompt=system_prompt,
            llm_client=llm_client
        )
        self.engine = engine
        self.vector_store = vector_store

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes hybrid semantic search for the portfolio context.
        context can provide: 'tickers', 'risk_tier', 'primary_goal'.
        """
        self.clear_trace()
        tickers = context.get("tickers", [])
        risk_tier = context.get("risk_tier", "Moderate")

        # 1. Fetch latest macro snapshot from Lakehouse
        self.log_step("query_macro_indicators", {})
        macro_df = self.engine.query_df("""
            SELECT * FROM bronze_macro_indicators 
            ORDER BY record_date DESC LIMIT 1;
        """)
        macro_snapshot = macro_df.to_dict(orient="records")[0] if not macro_df.empty else {}
        self.log_step("macro_snapshot_loaded", macro_snapshot)

        # 2. Perform semantic searches across relevant queries
        search_queries = [
            f"Macroeconomic interest rate outlook and bond duration risks {macro_snapshot.get('fed_funds_rate', '')}%",
            "Asset allocation diversification international equities gold REITs",
        ]
        # Add ticker-specific queries for top positions
        for t in tickers[:3]:
            search_queries.append(f"{t} stock risk factors valuation supply constraints earnings")

        retrieved_chunks: List[Dict[str, Any]] = []
        citations: List[str] = []
        seen_chunks = set()

        for q in search_queries:
            self.log_step("semantic_vector_search", {"query": q})
            results = self.vector_store.search(q, top_k=2)
            for r in results:
                if r["chunk_id"] not in seen_chunks:
                    seen_chunks.add(r["chunk_id"])
                    retrieved_chunks.append(r)
                    citations.append(r["citation"])

        self.log_step("retrieval_complete", {"retrieved_count": len(retrieved_chunks)})

        # 3. LLM synthesis with citations
        context_text = "\n\n".join([
            f"--- Citation: {c['citation']} ---\n{c['text']}" for c in retrieved_chunks
        ])

        user_prompt = f"""
Analyze the following retrieved market research and macroeconomic data:

Macro Environment:
- Fed Funds Rate: {macro_snapshot.get('fed_funds_rate', 'N/A')}%
- US 10-Year Treasury Yield: {macro_snapshot.get('us_10y_treasury', 'N/A')}%
- CPI Inflation (YoY): {macro_snapshot.get('cpi_inflation_yoy', 'N/A')}%
- VIX Volatility Index: {macro_snapshot.get('vix_volatility_index', 'N/A')}
- Client Risk Tier: {risk_tier}

Retrieved SEC Filings & Research Documents:
{context_text}

Provide a grounded Market Intelligence Briefing addressing:
1. Macro regime assessment (rates, inflation, duration risks).
2. Valuation and concentration risks in major holdings with explicit citations.
3. Tactical asset class perspectives (Equities vs. Intermediate Bonds vs. Real Assets).
Every key argument MUST include its supporting citation in square brackets.
"""
        self.log_step("llm_market_synthesis_prompt", user_prompt)
        llm_synthesis = self.llm.generate(self.system_prompt, user_prompt)
        self.log_step("llm_synthesis_complete", {"preview": llm_synthesis[:200] + "..."})

        return {
            "macro_snapshot": macro_snapshot,
            "retrieved_chunks": retrieved_chunks,
            "citations": list(set(citations)),
            "market_intelligence_brief": llm_synthesis,
            "trace": self.get_trace()
        }
