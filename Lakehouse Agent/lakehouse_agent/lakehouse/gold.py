"""
Gold Layer: Analytics Feature Store & Agent Decision Marts
Powers the multi-agent reasoning framework with aggregated portfolio analytics,
risk drift detection, asset class exposures, and recommendation audit logs.
"""
from lakehouse_agent.config import settings
from lakehouse_agent.lakehouse.engine import LakehouseEngine


class GoldLayer:
    """Manages business-ready marts and feature stores in the Gold medallion tier."""
    
    def __init__(self, engine: LakehouseEngine):
        self.engine = engine
        self.gold_path = settings.gold_path

    def process_all(self) -> dict:
        """Executes all Gold layer analytical aggregations."""
        # 1. Portfolio Analytics Mart
        self.build_portfolio_analytics()

        # 2. Investment Universe Mart
        self.build_investment_universe_mart()

        # 3. Create Recommendation Audit Log Table
        self.init_recommendation_history()

        return {
            "status": "SUCCESS",
            "tables": [
                "gold_portfolio_analytics",
                "gold_investment_universe",
                "gold_recommendation_history"
            ]
        }

    def build_portfolio_analytics(self):
        """Aggregates portfolio-level beta, dividend yield, asset class weights, and risk drift."""
        self.engine.execute("""
            CREATE OR REPLACE TABLE gold_portfolio_analytics AS
            WITH asset_class_breakdown AS (
                SELECT
                    investor_id,
                    SUM(CASE WHEN asset_class = 'Equities' THEN current_market_value ELSE 0 END) AS equity_value,
                    SUM(CASE WHEN asset_class = 'Fixed Income' THEN current_market_value ELSE 0 END) AS bond_value,
                    SUM(CASE WHEN asset_class = 'Real Estate' THEN current_market_value ELSE 0 END) AS real_estate_value,
                    SUM(CASE WHEN asset_class = 'Commodities' THEN current_market_value ELSE 0 END) AS commodities_value,
                    SUM(CASE WHEN asset_class = 'Cash' THEN current_market_value ELSE 0 END) AS cash_value,
                    -- Weighted portfolio beta & yield
                    SUM(current_market_value * beta) / NULLIF(SUM(current_market_value), 0) AS weighted_beta,
                    SUM(current_market_value * dividend_yield) / NULLIF(SUM(current_market_value), 0) AS weighted_dividend_yield,
                    -- Concentration / Diversification metric: Normalized inverse HHI
                    SUM(POW(weight_pct / 100.0, 2)) AS hhi_concentration
                FROM silver_portfolio_holdings
                GROUP BY investor_id
            )
            SELECT
                h.investor_id,
                h.name,
                h.age,
                h.stated_risk_tolerance,
                h.risk_score,
                h.risk_tier,
                h.primary_goal,
                h.debt_to_income_ratio,
                h.emergency_fund_runway_months,
                h.emergency_fund_status,
                t.total_portfolio_value,
                ROUND(a.weighted_beta, 3) AS portfolio_beta,
                ROUND(a.weighted_dividend_yield * 100, 2) AS portfolio_dividend_yield_pct,
                ROUND((a.equity_value / t.total_portfolio_value) * 100, 1) AS equity_pct,
                ROUND((a.bond_value / t.total_portfolio_value) * 100, 1) AS bond_pct,
                ROUND((a.real_estate_value / t.total_portfolio_value) * 100, 1) AS real_estate_pct,
                ROUND((a.commodities_value / t.total_portfolio_value) * 100, 1) AS commodities_pct,
                ROUND((a.cash_value / t.total_portfolio_value) * 100, 1) AS cash_pct,
                ROUND((1.0 - a.hhi_concentration) * 100, 1) AS diversification_score,
                -- Risk Drift Detection Logic
                CASE
                    WHEN h.risk_tier = 'Conservative' AND a.weighted_beta > 0.85 THEN 'High Risk Drift (Aggressive Mismatch)'
                    WHEN h.risk_tier = 'Conservative' AND (a.equity_value / t.total_portfolio_value) > 0.50 THEN 'High Equity Exposure for Conservative Investor'
                    WHEN h.risk_tier = 'Aggressive' AND a.weighted_beta < 0.70 THEN 'Low Risk Drift (Cash Drag / Under-invested)'
                    ELSE 'Aligned with Risk Profile'
                END AS risk_drift_status,
                CURRENT_TIMESTAMP AS calculated_at
            FROM silver_investor_financial_health h
            JOIN (
                SELECT investor_id, SUM(current_market_value) AS total_portfolio_value
                FROM silver_portfolio_holdings
                GROUP BY investor_id
            ) t ON h.investor_id = t.investor_id
            JOIN asset_class_breakdown a ON h.investor_id = a.investor_id;
        """)
        self.engine.save_table_as_parquet(
            "gold_portfolio_analytics", 
            self.gold_path / "portfolio_analytics.parquet"
        )

    def build_investment_universe_mart(self):
        """Constructs curated candidate asset universe with risk-adjusted ranking metrics."""
        self.engine.execute("""
            CREATE OR REPLACE TABLE gold_investment_universe AS
            SELECT
                ticker,
                name,
                asset_class,
                sector,
                current_price,
                pe_ratio,
                dividend_yield_pct,
                beta,
                volatility_profile,
                valuation_tier,
                esg_score,
                -- Opportunity Score heuristic (combining value, dividend, and reasonable beta)
                ROUND(
                    (CASE WHEN pe_ratio > 0 AND pe_ratio < 25 THEN 30 WHEN pe_ratio >= 25 AND pe_ratio <= 40 THEN 20 ELSE 10 END) +
                    (dividend_yield_pct * 10) +
                    (CASE WHEN beta BETWEEN 0.8 AND 1.2 THEN 25 WHEN beta < 0.8 THEN 20 ELSE 15 END) +
                    (esg_score * 0.2), 
                    1
                ) AS composite_opportunity_score,
                CURRENT_TIMESTAMP AS calculated_at
            FROM silver_asset_metrics
            ORDER BY composite_opportunity_score DESC;
        """)
        self.engine.save_table_as_parquet(
            "gold_investment_universe", 
            self.gold_path / "investment_universe.parquet"
        )

    def init_recommendation_history(self):
        """Initializes the recommendation audit log table."""
        self.engine.execute("""
            CREATE TABLE IF NOT EXISTS gold_recommendation_history (
                recommendation_id VARCHAR PRIMARY KEY,
                investor_id VARCHAR,
                timestamp TIMESTAMP,
                status VARCHAR,
                risk_tier VARCHAR,
                current_allocation VARCHAR,
                target_allocation VARCHAR,
                rebalancing_orders VARCHAR,
                compliance_status VARCHAR,
                rag_citations VARCHAR,
                executive_summary VARCHAR
            );
        """)
