"""
Silver Layer: Cleaned, Conformed, and Enriched Financial Data
Performs validations, joins, risk metric computations, and portfolio health indicators.
"""
from lakehouse_agent.config import settings
from lakehouse_agent.lakehouse.engine import LakehouseEngine


class SilverLayer:
    """Manages data transformation and feature enrichment into the Silver medallion tier."""
    
    def __init__(self, engine: LakehouseEngine):
        self.engine = engine
        self.silver_path = settings.silver_path

    def process_all(self) -> dict:
        """Executes all Silver layer transformation pipelines."""
        # 1. Enriched Investor Financial Health
        self.build_investor_financial_health()

        # 2. Enriched Asset Metrics
        self.build_asset_metrics()

        # 3. Enriched Current Portfolio Holdings & Exposures
        self.build_portfolio_holdings()

        return {
            "status": "SUCCESS",
            "tables": [
                "silver_investor_financial_health",
                "silver_asset_metrics",
                "silver_portfolio_holdings"
            ]
        }

    def build_investor_financial_health(self):
        """Computes debt-to-income, emergency runway, and financial resilience metrics."""
        self.engine.execute("""
            CREATE OR REPLACE TABLE silver_investor_financial_health AS
            SELECT 
                investor_id,
                name,
                age,
                annual_income,
                liquid_net_worth,
                total_debt,
                monthly_expenses,
                stated_risk_tolerance,
                risk_score,
                investment_horizon_years,
                primary_goal,
                esg_preference,
                -- Key Financial Ratios
                ROUND(total_debt / NULLIF(annual_income, 0), 3) AS debt_to_income_ratio,
                ROUND(liquid_net_worth / NULLIF(monthly_expenses, 0), 1) AS emergency_fund_runway_months,
                ROUND((annual_income / 12.0) - monthly_expenses, 2) AS monthly_savings_capacity,
                CASE 
                    WHEN (liquid_net_worth / NULLIF(monthly_expenses, 0)) >= 6.0 THEN 'Adequate'
                    WHEN (liquid_net_worth / NULLIF(monthly_expenses, 0)) >= 3.0 THEN 'Marginal'
                    ELSE 'Vulnerable'
                END AS emergency_fund_status,
                CASE
                    WHEN risk_score < 40 THEN 'Conservative'
                    WHEN risk_score <= 70 THEN 'Moderate'
                    ELSE 'Aggressive'
                END AS risk_tier,
                CURRENT_TIMESTAMP AS processed_at
            FROM bronze_investor_profiles;
        """)
        self.engine.save_table_as_parquet(
            "silver_investor_financial_health", 
            self.silver_path / "investor_financial_health.parquet"
        )

    def build_asset_metrics(self):
        """Enriches asset universe with risk factors, valuation tiers, and yield classifications."""
        self.engine.execute("""
            CREATE OR REPLACE TABLE silver_asset_metrics AS
            SELECT
                ticker,
                name,
                asset_class,
                sector,
                current_price,
                pe_ratio,
                dividend_yield,
                ROUND(dividend_yield * 100, 2) AS dividend_yield_pct,
                beta,
                market_cap_b,
                esg_score,
                CASE 
                    WHEN beta >= 1.3 THEN 'High Volatility'
                    WHEN beta >= 0.8 THEN 'Moderate Volatility'
                    ELSE 'Low Volatility'
                END AS volatility_profile,
                CASE
                    WHEN pe_ratio = 0 THEN 'Not Applicable'
                    WHEN pe_ratio < 18 THEN 'Value'
                    WHEN pe_ratio <= 30 THEN 'Fair'
                    ELSE 'Premium Growth'
                END AS valuation_tier,
                CURRENT_TIMESTAMP AS processed_at
            FROM bronze_asset_universe;
        """)
        self.engine.save_table_as_parquet(
            "silver_asset_metrics", 
            self.silver_path / "asset_metrics.parquet"
        )

    def build_portfolio_holdings(self):
        """Computes position values, unrealized gains, portfolio weights, and concentration flags."""
        self.engine.execute("""
            CREATE OR REPLACE TABLE silver_portfolio_holdings AS
            WITH base_holdings AS (
                SELECT
                    h.investor_id,
                    h.ticker,
                    h.shares,
                    h.cost_basis,
                    a.name AS asset_name,
                    a.asset_class,
                    a.sector,
                    a.current_price,
                    a.beta,
                    a.dividend_yield,
                    ROUND(h.shares * h.cost_basis, 2) AS total_cost_basis,
                    ROUND(h.shares * a.current_price, 2) AS current_market_value,
                    ROUND((h.shares * a.current_price) - (h.shares * h.cost_basis), 2) AS unrealized_pnl,
                    ROUND((((a.current_price - h.cost_basis) / NULLIF(h.cost_basis, 0)) * 100), 2) AS unrealized_return_pct
                FROM bronze_current_holdings h
                JOIN bronze_asset_universe a ON h.ticker = a.ticker
            ),
            portfolio_totals AS (
                SELECT 
                    investor_id, 
                    SUM(current_market_value) AS total_portfolio_value
                FROM base_holdings
                GROUP BY investor_id
            )
            SELECT
                b.*,
                t.total_portfolio_value,
                ROUND((b.current_market_value / t.total_portfolio_value) * 100, 2) AS weight_pct,
                CASE 
                    WHEN (b.current_market_value / t.total_portfolio_value) > 0.20 THEN 'Severe Concentration Risk'
                    WHEN (b.current_market_value / t.total_portfolio_value) > 0.15 THEN 'Moderate Concentration Risk'
                    ELSE 'Compliant'
                END AS concentration_status,
                CURRENT_TIMESTAMP AS processed_at
            FROM base_holdings b
            JOIN portfolio_totals t ON b.investor_id = t.investor_id;
        """)
        self.engine.save_table_as_parquet(
            "silver_portfolio_holdings", 
            self.silver_path / "portfolio_holdings.parquet"
        )
