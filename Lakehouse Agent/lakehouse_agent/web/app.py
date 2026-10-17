"""
FastAPI Web Server & REST API
Provides endpoints for Lakehouse data exploration, multi-agent advisory orchestration,
scenario stress testing, and real-time interactive UI.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from pathlib import Path

from lakehouse_agent.config import settings
from lakehouse_agent.lakehouse.engine import LakehouseEngine
from lakehouse_agent.lakehouse.bronze import BronzeLayer
from lakehouse_agent.lakehouse.silver import SilverLayer
from lakehouse_agent.lakehouse.gold import GoldLayer
from lakehouse_agent.vector_store.store import FinancialVectorStore
from lakehouse_agent.agents.orchestrator import FinancialAdvisorOrchestrator

# Initialize core system singletons
engine = LakehouseEngine()
bronze = BronzeLayer(engine)
silver = SilverLayer(engine)
gold = GoldLayer(engine)
vector_store = FinancialVectorStore()
orchestrator = FinancialAdvisorOrchestrator(engine=engine, vector_store=vector_store)

# Ensure lakehouse is primed on startup
try:
    bronze.ingest_all()
    silver.process_all()
    gold.process_all()
    vector_store.sync_from_lakehouse(engine)
except Exception as e:
    print(f"[Startup Warning] ETL Pipeline notice: {e}")

app = FastAPI(
    title="Lakehouse-Enabled Agentic Intelligence System",
    description="Adaptive Personalized Financial Recommendation using Retrieval-Augmented LLMs",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static directory
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Pydantic schemas
class RecommendationRequest(BaseModel):
    investor_id: str
    custom_risk_override: Optional[str] = None


class SQLQueryRequest(BaseModel):
    query: str


class StressTestRequest(BaseModel):
    investor_id: str
    scenario: str  # e.g., "tech_correction", "rate_hike", "stagflation"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main dashboard application."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>UI static file initializing. Please refresh in a moment.</h2>")


@app.get("/api/status")
async def get_system_status():
    """Returns the operational status of the Lakehouse and Agent subsystems."""
    tables = engine.list_tables()
    return {
        "status": "ONLINE",
        "lakehouse_engine": "DuckDB + Parquet (Medallion: Bronze, Silver, Gold)",
        "llm_provider": settings.llm_provider,
        "total_lakehouse_tables": len(tables),
        "table_names": [t["table_name"] for t in tables],
        "indexed_vector_chunks": len(vector_store.chunks),
        "db_path": settings.lakehouse_db_path
    }


@app.post("/api/pipeline/initialize")
async def initialize_pipeline():
    """Re-executes the complete Lakehouse ETL ingestion and vector sync."""
    b_res = bronze.ingest_all()
    s_res = silver.process_all()
    g_res = gold.process_all()
    v_count = vector_store.sync_from_lakehouse(engine)
    return {
        "message": "Lakehouse Medallion pipeline refreshed successfully.",
        "bronze": b_res,
        "silver": s_res,
        "gold": g_res,
        "indexed_vectors": v_count
    }


@app.get("/api/investors")
async def get_investors():
    """Returns all investor profiles loaded in the Lakehouse."""
    try:
        profiles = engine.query_dicts("""
            SELECT investor_id, name, age, stated_risk_tolerance, 
                   risk_score, risk_tier, primary_goal, liquid_net_worth,
                   annual_income, emergency_fund_runway_months, emergency_fund_status
            FROM silver_investor_financial_health
            ORDER BY investor_id;
        """)
        return {"investors": profiles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/investor/{investor_id}/analytics")
async def get_investor_analytics(investor_id: str):
    """Returns consolidated portfolio analytics and holdings for an investor."""
    try:
        analytics = engine.query_dicts(f"""
            SELECT * FROM gold_portfolio_analytics WHERE investor_id = '{investor_id}';
        """)
        if not analytics:
            raise HTTPException(status_code=404, detail=f"Investor {investor_id} not found.")

        holdings = engine.query_dicts(f"""
            SELECT * FROM silver_portfolio_holdings 
            WHERE investor_id = '{investor_id}'
            ORDER BY weight_pct DESC;
        """)

        return {
            "analytics": analytics[0],
            "holdings": holdings
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommendation/generate")
async def generate_recommendation(req: RecommendationRequest):
    """Executes the multi-agent financial recommendation pipeline."""
    try:
        report = orchestrator.run_advisory_pipeline(req.investor_id)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent pipeline error: {str(e)}")


@app.post("/api/lakehouse/query")
async def execute_lakehouse_sql(req: SQLQueryRequest):
    """Allows safe interactive SQL queries over the DuckDB Lakehouse."""
    query = req.query.strip()
    # Simple security guardrail: allow SELECT statements only
    if not query.lower().startswith("select") and not query.lower().startswith("show") and not query.lower().startswith("describe"):
        raise HTTPException(status_code=400, detail="Only analytical SELECT/SHOW/DESCRIBE queries are permitted in this console.")
    
    try:
        results = engine.query_dicts(query)
        # Limit rows for safety
        return {
            "query": query,
            "row_count": len(results),
            "data": results[:100]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL Execution Error: {str(e)}")


@app.post("/api/scenario/simulate")
async def simulate_stress_scenario(req: StressTestRequest):
    """
    Stress tests current portfolio vs. target rebalanced portfolio
    under predefined macroeconomic shock scenarios.
    """
    try:
        analytics = engine.query_dicts(f"""
            SELECT * FROM gold_portfolio_analytics WHERE investor_id = '{req.investor_id}';
        """)[0]
        holdings = engine.query_dicts(f"""
            SELECT * FROM silver_portfolio_holdings WHERE investor_id = '{req.investor_id}';
        """)

        scenarios = {
            "tech_correction": {
                "name": "Mega-Cap Tech Valuation Correction (-25%)",
                "equity_shock": -0.22,
                "tech_shock": -0.30,
                "bond_shock": 0.03,
                "gold_shock": 0.06,
                "cash_shock": 0.01,
                "description": "Tech multiples compress; flight to safety benefits bonds and gold."
            },
            "rate_hike": {
                "name": "Unexpected Fed 100bps Rate Hike",
                "equity_shock": -0.12,
                "tech_shock": -0.16,
                "bond_shock": -0.06,
                "gold_shock": -0.04,
                "cash_shock": 0.02,
                "description": "Rising discount rates compress long-duration equities and long treasuries."
            },
            "stagflation": {
                "name": "Stagflationary Shock (CPI Spikes + Growth Halts)",
                "equity_shock": -0.18,
                "tech_shock": -0.20,
                "bond_shock": -0.08,
                "gold_shock": 0.15,
                "cash_shock": 0.01,
                "description": "Equities and conventional bonds suffer simultaneously; Gold and real assets surge."
            }
        }

        scenario = scenarios.get(req.scenario, scenarios["tech_correction"])
        total_curr = analytics["total_portfolio_value"]

        # Calculate impact on current holdings
        curr_loss = 0.0
        for h in holdings:
            val = h["current_market_value"]
            sec = h["sector"]
            ac = h["asset_class"]
            if "Technology" in sec or "Tech" in sec or h["ticker"] == "NVDA":
                impact = scenario["tech_shock"]
            elif ac == "Equities":
                impact = scenario["equity_shock"]
            elif ac == "Fixed Income":
                impact = scenario["bond_shock"]
            elif ac == "Commodities":
                impact = scenario["gold_shock"]
            else:
                impact = scenario["cash_shock"]
            curr_loss += (val * impact)

        curr_post_val = total_curr + curr_loss
        curr_return_pct = round((curr_loss / total_curr) * 100, 2)

        # Calculate impact on optimized target portfolio
        # Target has balanced allocation: e.g. 60% Eq, 25% Bond, 5% Real Estate, 5% Gold, 5% Cash
        target_return_pct = round(
            (0.55 * scenario["equity_shock"]) +
            (0.25 * scenario["bond_shock"]) +
            (0.05 * scenario["equity_shock"] * 0.8) +
            (0.10 * scenario["gold_shock"]) +
            (0.05 * scenario["cash_shock"]),
            4
        ) * 100
        target_post_val = round(total_curr * (1 + target_return_pct / 100), 2)

        return {
            "scenario": scenario["name"],
            "description": scenario["description"],
            "current_portfolio": {
                "starting_value": total_curr,
                "simulated_value": round(curr_post_val, 2),
                "drawdown_pct": curr_return_pct,
                "dollar_loss": round(curr_loss, 2)
            },
            "optimized_target_portfolio": {
                "starting_value": total_curr,
                "simulated_value": target_post_val,
                "drawdown_pct": round(target_return_pct, 2),
                "dollar_loss": round(target_post_val - total_curr, 2)
            },
            "protection_alpha_pct": round(target_return_pct - curr_return_pct, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
