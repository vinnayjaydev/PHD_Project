"""
Synthetic Financial Data Generator
Produces realistic market feeds, investor profiles, historical trades, macro indicators,
and unstructured research documents for the Lakehouse.
"""
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


class FinancialDataGenerator:
    """Generates comprehensive financial data for the Lakehouse Medallion layers."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)

    def generate_investor_profiles(self) -> pd.DataFrame:
        """Generates representative investor personas across life stages and risk levels."""
        investors = [
            {
                "investor_id": "INV001",
                "name": "Sarah Jenkins",
                "age": 29,
                "annual_income": 195000.0,
                "liquid_net_worth": 145000.0,
                "total_debt": 28000.0,
                "monthly_expenses": 5800.0,
                "stated_risk_tolerance": "Aggressive",
                "risk_score": 85,  # 1-100
                "investment_horizon_years": 30,
                "primary_goal": "Long-Term Wealth Accumulation & Tech Growth",
                "esg_preference": True,
                "created_at": datetime.now() - timedelta(days=365),
            },
            {
                "investor_id": "INV002",
                "name": "Robert Chang",
                "age": 62,
                "annual_income": 130000.0,
                "liquid_net_worth": 920000.0,
                "total_debt": 0.0,
                "monthly_expenses": 6500.0,
                "stated_risk_tolerance": "Conservative",
                "risk_score": 30,
                "investment_horizon_years": 5,
                "primary_goal": "Capital Preservation & Steady Dividend Income",
                "esg_preference": False,
                "created_at": datetime.now() - timedelta(days=730),
            },
            {
                "investor_id": "INV003",
                "name": "Elena Rostova",
                "age": 38,
                "annual_income": 275000.0,
                "liquid_net_worth": 480000.0,
                "total_debt": 310000.0,  # Mortgage
                "monthly_expenses": 9200.0,
                "stated_risk_tolerance": "Moderate",
                "risk_score": 58,
                "investment_horizon_years": 12,
                "primary_goal": "Balanced Growth with Down Payment for Vacation Home in 3 Years",
                "esg_preference": False,
                "created_at": datetime.now() - timedelta(days=500),
            },
            {
                "investor_id": "INV004",
                "name": "Marcus Vance",
                "age": 47,
                "annual_income": 220000.0,
                "liquid_net_worth": 650000.0,
                "total_debt": 85000.0,
                "monthly_expenses": 8000.0,
                "stated_risk_tolerance": "Moderate-Aggressive",
                "risk_score": 72,
                "investment_horizon_years": 18,
                "primary_goal": "College Tuition Funding & Retirement Alpha",
                "esg_preference": True,
                "created_at": datetime.now() - timedelta(days=400),
            }
        ]
        return pd.DataFrame(investors)

    def generate_asset_universe(self) -> pd.DataFrame:
        """Defines the investment asset universe with fundamental metrics."""
        assets = [
            # US Large Cap Equities
            {"ticker": "AAPL", "name": "Apple Inc.", "asset_class": "Equities", "sector": "Information Technology", "current_price": 234.50, "pe_ratio": 33.2, "dividend_yield": 0.005, "beta": 1.12, "market_cap_b": 3580.0, "esg_score": 78},
            {"ticker": "MSFT", "name": "Microsoft Corp.", "asset_class": "Equities", "sector": "Information Technology", "current_price": 448.20, "pe_ratio": 35.8, "dividend_yield": 0.007, "beta": 1.18, "market_cap_b": 3320.0, "esg_score": 85},
            {"ticker": "NVDA", "name": "NVIDIA Corp.", "asset_class": "Equities", "sector": "Information Technology", "current_price": 138.60, "pe_ratio": 48.5, "dividend_yield": 0.001, "beta": 1.68, "market_cap_b": 3400.0, "esg_score": 72},
            {"ticker": "GOOGL", "name": "Alphabet Inc.", "asset_class": "Equities", "sector": "Communication Services", "current_price": 182.10, "pe_ratio": 24.1, "dividend_yield": 0.004, "beta": 1.05, "market_cap_b": 2240.0, "esg_score": 74},
            {"ticker": "AMZN", "name": "Amazon.com Inc.", "asset_class": "Equities", "sector": "Consumer Discretionary", "current_price": 194.80, "pe_ratio": 41.2, "dividend_yield": 0.000, "beta": 1.25, "market_cap_b": 2020.0, "esg_score": 68},
            {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "asset_class": "Equities", "sector": "Financials", "current_price": 222.40, "pe_ratio": 12.4, "dividend_yield": 0.023, "beta": 0.95, "market_cap_b": 630.0, "esg_score": 71},
            {"ticker": "JNJ", "name": "Johnson & Johnson", "asset_class": "Equities", "sector": "Health Care", "current_price": 162.30, "pe_ratio": 16.8, "dividend_yield": 0.031, "beta": 0.54, "market_cap_b": 390.0, "esg_score": 82},
            {"ticker": "XOM", "name": "Exxon Mobil Corp.", "asset_class": "Equities", "sector": "Energy", "current_price": 118.90, "pe_ratio": 14.2, "dividend_yield": 0.034, "beta": 0.78, "market_cap_b": 510.0, "esg_score": 52},
            
            # Broad Market & International ETFs
            {"ticker": "SPY", "name": "SPDR S&P 500 ETF Trust", "asset_class": "Equities", "sector": "Broad Market", "current_price": 585.10, "pe_ratio": 26.5, "dividend_yield": 0.013, "beta": 1.00, "market_cap_b": 590.0, "esg_score": 75},
            {"ticker": "QQQ", "name": "Invesco QQQ Trust (Nasdaq 100)", "asset_class": "Equities", "sector": "Tech / Growth", "current_price": 492.30, "pe_ratio": 31.8, "dividend_yield": 0.006, "beta": 1.22, "market_cap_b": 280.0, "esg_score": 76},
            {"ticker": "VXUS", "name": "Vanguard Total International Stock ETF", "asset_class": "Equities", "sector": "International Equities", "current_price": 64.20, "pe_ratio": 15.1, "dividend_yield": 0.029, "beta": 0.82, "market_cap_b": 85.0, "esg_score": 77},
            
            # Fixed Income & Bonds
            {"ticker": "BND", "name": "Vanguard Total Bond Market ETF", "asset_class": "Fixed Income", "sector": "Aggregate Bonds", "current_price": 73.80, "pe_ratio": 0.0, "dividend_yield": 0.043, "beta": 0.12, "market_cap_b": 110.0, "esg_score": 80},
            {"ticker": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "asset_class": "Fixed Income", "sector": "Long-Term US Treasuries", "current_price": 92.50, "pe_ratio": 0.0, "dividend_yield": 0.046, "beta": 0.28, "market_cap_b": 58.0, "esg_score": 90},
            {"ticker": "SHY", "name": "iShares 1-3 Year Treasury Bond ETF", "asset_class": "Fixed Income", "sector": "Short-Term US Treasuries", "current_price": 82.40, "pe_ratio": 0.0, "dividend_yield": 0.048, "beta": 0.02, "market_cap_b": 26.0, "esg_score": 90},
            {"ticker": "TIP", "name": "iShares TIPS Bond ETF", "asset_class": "Fixed Income", "sector": "Inflation-Protected Treasuries", "current_price": 107.60, "pe_ratio": 0.0, "dividend_yield": 0.038, "beta": 0.10, "market_cap_b": 19.0, "esg_score": 90},
            
            # Real Estate & Commodities
            {"ticker": "VNQ", "name": "Vanguard Real Estate ETF", "asset_class": "Real Estate", "sector": "Real Estate Investment Trusts", "current_price": 88.30, "pe_ratio": 28.4, "dividend_yield": 0.041, "beta": 0.88, "market_cap_b": 35.0, "esg_score": 73},
            {"ticker": "GLD", "name": "SPDR Gold Shares", "asset_class": "Commodities", "sector": "Precious Metals", "current_price": 248.50, "pe_ratio": 0.0, "dividend_yield": 0.000, "beta": 0.08, "market_cap_b": 72.0, "esg_score": 60},
            
            # Cash Equivalent
            {"ticker": "BIL", "name": "SPDR 1-3 Month T-Bill ETF", "asset_class": "Cash", "sector": "Cash Equivalent", "current_price": 91.70, "pe_ratio": 0.0, "dividend_yield": 0.051, "beta": 0.00, "market_cap_b": 38.0, "esg_score": 95},
        ]
        return pd.DataFrame(assets)

    def generate_current_holdings(self) -> pd.DataFrame:
        """Generates realistic existing portfolios for each investor archetype."""
        holdings = [
            # Sarah Jenkins (INV001): Overconcentrated in Mega-Cap Tech, Low Cash, 0% Bonds
            {"investor_id": "INV001", "ticker": "NVDA", "shares": 320.0, "cost_basis": 78.50},
            {"investor_id": "INV001", "ticker": "AAPL", "shares": 180.0, "cost_basis": 172.00},
            {"investor_id": "INV001", "ticker": "MSFT", "shares": 95.0, "cost_basis": 340.00},
            {"investor_id": "INV001", "ticker": "QQQ", "shares": 80.0, "cost_basis": 410.00},
            {"investor_id": "INV001", "ticker": "BIL", "shares": 45.0, "cost_basis": 91.50},  # Small cash

            # Robert Chang (INV002): Conservative, but suffering from cash drag and unhedged bond duration
            {"investor_id": "INV002", "ticker": "BND", "shares": 3800.0, "cost_basis": 81.20},
            {"investor_id": "INV002", "ticker": "TLT", "shares": 2200.0, "cost_basis": 105.00},
            {"investor_id": "INV002", "ticker": "JNJ", "shares": 1100.0, "cost_basis": 155.00},
            {"investor_id": "INV002", "ticker": "JPM", "shares": 650.0, "cost_basis": 180.00},
            {"investor_id": "INV002", "ticker": "SPY", "shares": 150.0, "cost_basis": 460.00},
            {"investor_id": "INV002", "ticker": "BIL", "shares": 1800.0, "cost_basis": 91.60},

            # Elena Rostova (INV003): Moderate, balanced but lacking international and inflation hedging
            {"investor_id": "INV003", "ticker": "SPY", "shares": 350.0, "cost_basis": 490.00},
            {"investor_id": "INV003", "ticker": "MSFT", "shares": 160.0, "cost_basis": 380.00},
            {"investor_id": "INV003", "ticker": "BND", "shares": 1400.0, "cost_basis": 74.20},
            {"investor_id": "INV003", "ticker": "VNQ", "shares": 500.0, "cost_basis": 82.50},
            {"investor_id": "INV003", "ticker": "BIL", "shares": 600.0, "cost_basis": 91.70},

            # Marcus Vance (INV004): Moderate-Aggressive
            {"investor_id": "INV004", "ticker": "SPY", "shares": 420.0, "cost_basis": 510.00},
            {"investor_id": "INV004", "ticker": "QQQ", "shares": 280.0, "cost_basis": 430.00},
            {"investor_id": "INV004", "ticker": "GOOGL", "shares": 450.0, "cost_basis": 152.00},
            {"investor_id": "INV004", "ticker": "BND", "shares": 850.0, "cost_basis": 73.50},
            {"investor_id": "INV004", "ticker": "GLD", "shares": 220.0, "cost_basis": 215.00},
            {"investor_id": "INV004", "ticker": "BIL", "shares": 400.0, "cost_basis": 91.70},
        ]
        return pd.DataFrame(holdings)

    def generate_macro_indicators(self) -> pd.DataFrame:
        """Generates historical macroeconomic indicators for regime detection."""
        records = []
        base_date = datetime.now()
        for i in range(12):  # Last 12 months
            d = base_date - timedelta(days=30 * i)
            records.append({
                "record_date": d.strftime("%Y-%m-%d"),
                "fed_funds_rate": 5.25 - (i * 0.05),
                "us_10y_treasury": 4.25 + (np.sin(i) * 0.2),
                "cpi_inflation_yoy": 2.8 + (i * 0.08),
                "gdp_growth_annualized": 2.3 + (np.cos(i) * 0.3),
                "vix_volatility_index": 16.5 + (np.random.normal(0, 1.8)),
                "unemployment_rate": 4.1 + (i * 0.02),
            })
        return pd.DataFrame(records)

    def generate_unstructured_financial_documents(self) -> List[Dict[str, str]]:
        """Generates rich text corpus for RAG semantic retrieval: SEC filings, macro outlooks, analyst ratings."""
        docs = [
            {
                "doc_id": "SEC_10K_NVDA_2024",
                "ticker": "NVDA",
                "title": "NVIDIA Form 10-K Annual Report: AI Data Center Demand & Supply Constraints",
                "doc_type": "SEC_10K",
                "source": "SEC EDGAR",
                "published_at": "2024-03-01",
                "content": (
                    "Item 1A. Risk Factors: Our Compute & Networking segment experienced exponential growth driven by "
                    "accelerated computing architectures and generative AI models. However, our customer concentration has increased, "
                    "with top hyperscale cloud providers representing over 35% of total revenues. We face semiconductor supply "
                    "chain bottlenecks at foundry partners (TSMC CoWoS packaging) and geopolitical export restrictions to China. "
                    "Valuation multiples are elevated relative to historical semiconductor cycles, exposing equity holders to "
                    "volatility in case of capex deceleration by major cloud customers."
                )
            },
            {
                "doc_id": "SEC_10K_AAPL_2024",
                "ticker": "AAPL",
                "title": "Apple Inc. Form 10-K: Services Margin Expansion and Capital Return Program",
                "doc_type": "SEC_10K",
                "source": "SEC EDGAR",
                "published_at": "2024-02-15",
                "content": (
                    "Management Discussion and Analysis: Services revenue reached an all-time high of $85.2B with gross margins "
                    "exceeding 72%. Active installed base surpasses 2.2 billion active devices, creating sticky recurring cash flows. "
                    "We returned over $110 billion to shareholders via share repurchases and regular quarterly dividends. "
                    "Risks include regulatory antitrust scrutiny in the European Union and US DOJ pertaining to the App Store, "
                    "alongside lengthening smartphone replacement cycles in mature markets."
                )
            },
            {
                "doc_id": "SEC_10K_JPM_2024",
                "ticker": "JPM",
                "title": "JPMorgan Chase Annual Report: Net Interest Margin & Credit Reserve Resilience",
                "doc_type": "SEC_10K",
                "source": "SEC EDGAR",
                "published_at": "2024-02-28",
                "content": (
                    "Financial Summary: Net Interest Income (NII) benefited substantially from higher policy benchmark rates. "
                    "Common Equity Tier 1 (CET1) ratio stands robust at 15.0%, significantly exceeding Basel III requirements. "
                    "Credit loss allowances for commercial real estate (CRE) office exposure have been conservatively bolstered. "
                    "The firm is uniquely positioned to capitalize on corporate advisory and wealth management inflows while "
                    "maintaining an attractive dividend payout ratio of under 30%."
                )
            },
            {
                "doc_id": "MACRO_FOMC_OUTLOOK_2024",
                "ticker": "MACRO",
                "title": "Federal Open Market Committee: Monetary Policy Trajectory & Neutral Rate Assessment",
                "doc_type": "Fed_Statement",
                "source": "Federal Reserve Board",
                "published_at": "2024-09-18",
                "content": (
                    "The Committee observes that inflation has eased notably toward the 2% target, while labor market conditions "
                    "have normalized. In considering adjustments to the target range for the federal funds rate, the Committee "
                    "will carefully assess incoming data and the balance of risks. Fixed income investors face asymmetric reinvestment "
                    "risk on short-term cash instruments (money market funds yielding ~5%) as rate easing cycles commence. "
                    "Extending duration into intermediate Treasuries and investment-grade corporate bonds provides compelling total return protection."
                )
            },
            {
                "doc_id": "ANALYST_ASSET_ALLOC_2024",
                "ticker": "MULTI",
                "title": "Global Wealth Management Outlook: Dynamic Diversification and Adaptive Rebalancing",
                "doc_type": "Research_Report",
                "source": "Global Institutional Research",
                "published_at": "2024-08-10",
                "content": (
                    "Overconcentration in US Mega-Cap Technology has pushed portfolio correlation to 15-year highs. Investors with "
                    "equity exposure exceeding 85% carry acute drawdown risk during valuation resets. Recommended strategic adjustments: "
                    "(1) Reallocate tactical gains into international equities (VXUS) trading at 35% valuation discounts to the S&P 500; "
                    "(2) Allocate 5-10% to real assets (Gold GLD and REITs VNQ) for stagflation hedging; "
                    "(3) Lock in yields in core fixed income (BND) before money market yields decay."
                )
            },
            {
                "doc_id": "SEC_10K_JNJ_2024",
                "ticker": "JNJ",
                "title": "Johnson & Johnson Form 10-K: Innovative Medicine Pipeline and Dividend Aristocrat Status",
                "doc_type": "SEC_10K",
                "source": "SEC EDGAR",
                "published_at": "2024-02-18",
                "content": (
                    "Johnson & Johnson has increased dividends for 62 consecutive years, maintaining AAA-rated balance sheet health. "
                    "Following the successful separation of Kenvue consumer health, the company focuses exclusively on high-margin "
                    "Pharmaceuticals and MedTech devices. Low market beta (0.54) and steady 3.1% dividend yield offer vital defensive "
                    "cushioning for conservative investors and retirees seeking capital preservation during market downturns."
                )
            }
        ]
        return docs
