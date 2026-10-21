"""
Macro Regime Agent: Systemic Macroeconomic & Market Regime Intelligence.
Part of the FLAME-AI Financial Agentic Framework.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
from .base_agent import BaseFinancialAgent
from lakehouse.query_engine import LakehouseQueryEngine

logger = logging.getLogger("MacroRegimeAgent")


class MacroRegimeAgent(BaseFinancialAgent):
    """
    Evaluates global macroeconomic conditions, Federal Reserve policy stance,
    yield curve inversion/steepening, and market volatility regimes.
    """

    def __init__(self, weight: float = 0.25, llm_provider: str = "deterministic", api_key: Optional[str] = None):
        super().__init__(
            name="MacroRegimeAgent",
            role="Systemic Macroeconomic & Regime Analysis",
            weight=weight,
            llm_provider=llm_provider,
            api_key=api_key
        )
        self.query_engine = LakehouseQueryEngine()

    def analyze(self, ticker: str, as_of_date: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.trajectory_log.clear()
        self.log_trajectory_step("THOUGHT", {
            "thought": f"Querying Lakehouse Gold Layer for systemic macro indicators and regime classification as of {as_of_date}."
        })

        # 1. Fetch active regime from Gold Layer
        regime_info = self.query_engine.get_active_regime(as_of_date)
        regime_id = regime_info.get("regime_id", 3)
        regime_name = regime_info.get("regime_name", "Sideways_RangeBound")
        trend_strength = regime_info.get("trend_strength", 0.0)
        market_vol = regime_info.get("market_volatility", 0.18)

        # 2. Fetch macro indicators: 10Y Yield, VIX, Gold
        macro_df = self.query_engine.get_market_state(as_of_date, tickers=["INDEX_TNX", "INDEX_VIX", "GLD", "TLT"])
        vix_val = 20.0
        tnx_val = 4.2
        if not macro_df.empty:
            vix_row = macro_df[macro_df["ticker"] == "INDEX_VIX"]
            if not vix_row.empty:
                vix_val = float(vix_row.iloc[0]["close"])
            tnx_row = macro_df[macro_df["ticker"] == "INDEX_TNX"]
            if not tnx_row.empty:
                tnx_val = float(tnx_row.iloc[0]["close"])

        self.log_trajectory_step("OBSERVATION", {
            "regime_name": regime_name,
            "trend_strength": trend_strength,
            "market_volatility": market_vol,
            "vix": vix_val,
            "treasury_10y": tnx_val
        })

        # 3. Macro synthesis logic
        citations = [f"[Lakehouse Gold Cube § MarketRegimes as_of {as_of_date}]"]
        if regime_id == 0:  # Bull
            stance = "BULLISH"
            confidence = min(0.70 + abs(trend_strength) * 2.0, 0.92)
            rationale = (
                f"Macro environment is expansionary ({regime_name}). 10-Yr yield is stabilizing at {tnx_val:.2f}% "
                f"with moderate market volatility (VIX: {vix_val:.1f}). Favorable liquidity and equity risk premium support long exposures."
            )
        elif regime_id == 1:  # Bear
            stance = "BEARISH"
            confidence = min(0.68 + abs(trend_strength) * 2.5, 0.90)
            rationale = (
                f"Macro conditions exhibit contractionary pressures ({regime_name}). Benchmark trend is negative ({trend_strength*100:.1f}%) "
                f"with rising volatility ({market_vol*100:.1f}%). Macro liquidity tightening warrants capital preservation."
            )
        elif regime_id == 2:  # High Volatility Crisis
            stance = "DEFENSIVE"
            confidence = 0.94
            rationale = (
                f"SYSTEMIC VOLATILITY ALERT: Extreme market turbulence detected ({regime_name}, VIX: {vix_val:.1f}). "
                f"Realized volatility at {market_vol*100:.1f}% triggers mandatory capital preservation buffers and defensive allocation."
            )
        else:  # RangeBound
            stance = "NEUTRAL"
            confidence = 0.65
            rationale = (
                f"Market is consolidating in a sideways rangebound regime. Trend strength ({trend_strength*100:.2f}%) "
                f"and VIX ({vix_val:.1f}) indicate balanced risk-reward without strong directional momentum."
            )

        self.log_trajectory_step("DECISION", {
            "stance": stance,
            "confidence": round(confidence, 3),
            "rationale": rationale
        })

        return {
            "agent": self.name,
            "ticker": ticker,
            "as_of_date": as_of_date,
            "stance": stance,
            "confidence": round(confidence, 3),
            "rationale": rationale,
            "key_metrics": {
                "regime_id": regime_id,
                "regime_name": regime_name,
                "trend_strength_pct": round(trend_strength * 100, 2),
                "market_volatility_pct": round(market_vol * 100, 2),
                "vix_level": round(vix_val, 2),
                "treasury_10y_yield": round(tnx_val, 2)
            },
            "citations": citations,
            "trajectory": list(self.trajectory_log)
        }
