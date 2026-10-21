"""
Quantitative Risk Agent: Momentum, Technical Indicators & Extreme Tail Risk Modeling.
Part of the FLAME-AI Financial Agentic Framework.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
from .base_agent import BaseFinancialAgent
from lakehouse.query_engine import LakehouseQueryEngine

logger = logging.getLogger("QuantRiskAgent")


class QuantRiskAgent(BaseFinancialAgent):
    """
    Evaluates momentum, mean-reversion, technical signals (RSI, MACD, Bollinger Bands),
    and downside tail risk metrics (VaR 95/99, Expected Shortfall CVaR, Maximum Drawdown).
    """

    def __init__(self, weight: float = 0.25, llm_provider: str = "deterministic", api_key: Optional[str] = None):
        super().__init__(
            name="QuantRiskAgent",
            role="Technical Volatility, Drawdown & Tail Risk Modeling",
            weight=weight,
            llm_provider=llm_provider,
            api_key=api_key
        )
        self.query_engine = LakehouseQueryEngine()

    def analyze(self, ticker: str, as_of_date: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.trajectory_log.clear()
        self.log_trajectory_step("THOUGHT", {
            "thought": f"Querying Lakehouse Gold Layer for technical indicators and rolling tail risk metrics for {ticker} as of {as_of_date}."
        })

        # Fetch market and risk state from Gold Lakehouse
        df = self.query_engine.get_market_state(as_of_date, tickers=[ticker])

        if not df.empty:
            row = df.iloc[0]
            close = float(row["close"])
            rsi = float(row["rsi_14"]) if pd.notnull(row["rsi_14"]) else 50.0
            macd_hist = float(row["macd"] - row["macd_signal"]) if pd.notnull(row["macd"]) else 0.0
            bb_zscore = float(row["bb_zscore"]) if pd.notnull(row["bb_zscore"]) else 0.0
            vol_ann = float(row["vol_21d"]) if pd.notnull(row["vol_21d"]) else 0.25
            sharpe = float(row["rolling_sharpe"]) if pd.notnull(row.get("rolling_sharpe")) else 1.2
            sortino = float(row["rolling_sortino"]) if pd.notnull(row.get("rolling_sortino")) else 1.5
            var_95 = float(row["var_95"]) if pd.notnull(row.get("var_95")) else 0.025
            cvar_95 = float(row["cvar_95"]) if pd.notnull(row.get("cvar_95")) else 0.038
            max_dd = float(row["rolling_max_dd"]) if pd.notnull(row.get("rolling_max_dd")) else -0.10
        else:
            close, rsi, macd_hist, bb_zscore, vol_ann = 150.0, 52.0, 0.5, 0.2, 0.24
            sharpe, sortino, var_95, cvar_95, max_dd = 1.1, 1.4, 0.026, 0.039, -0.08

        self.log_trajectory_step("OBSERVATION", {
            "close": close,
            "rsi_14": round(rsi, 2),
            "macd_hist": round(macd_hist, 4),
            "bb_zscore": round(bb_zscore, 2),
            "vol_ann_pct": round(vol_ann * 100, 2),
            "var_95_pct": round(var_95 * 100, 2),
            "cvar_95_pct": round(cvar_95 * 100, 2)
        })

        citations = [f"[Lakehouse Gold Rolling Risk Cube § {ticker} as_of {as_of_date}]"]

        # Technical & Risk Logic
        if rsi < 35.0:
            stance = "BULLISH"
            confidence = 0.84
            rationale = (
                f"{ticker} is technically oversold (RSI 14: {rsi:.1f}, BB Z-Score: {bb_zscore:.2f}) "
                f"with favorable risk asymmetry. 1-day 95% VaR is {var_95*100:.2f}% and Expected Shortfall is {cvar_95*100:.2f}%. "
                f"Mean-reversion upside potential dominates downside volatility."
            )
        elif rsi > 75.0 or bb_zscore > 2.2:
            stance = "DEFENSIVE"
            confidence = 0.86
            rationale = (
                f"{ticker} shows overextended momentum (RSI 14: {rsi:.1f}, BB Z-Score: {bb_zscore:.2f}). "
                f"Elevated annualized volatility ({vol_ann*100:.1f}%) and tail risk (CVaR: {cvar_95*100:.2f}%) "
                f"warrant tightening stop-losses and trimming long exposure."
            )
        elif macd_hist > 0 and vol_ann < 0.35:
            stance = "BULLISH"
            confidence = 0.79
            rationale = (
                f"Positive momentum trend for {ticker} (MACD histogram expanding, RSI: {rsi:.1f}). "
                f"Rolling Sharpe of {sharpe:.2f} and Sortino of {sortino:.2f} confirm attractive risk-adjusted performance."
            )
        else:
            stance = "NEUTRAL"
            confidence = 0.65
            rationale = (
                f"Balanced technical profile for {ticker} (RSI: {rsi:.1f}, Annualized Vol: {vol_ann*100:.1f}%). "
                f"Tail risk indicators (95% VaR: {var_95*100:.2f}%) remain within acceptable portfolio tolerances."
            )

        # Stop-loss calculation based on ATR/VaR
        recommended_stop_loss = round(close * (1.0 - max(var_95 * 2.5, 0.05)), 2)

        self.log_trajectory_step("DECISION", {
            "stance": stance,
            "confidence": round(confidence, 3),
            "stop_loss": recommended_stop_loss,
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
                "close_price": round(close, 2),
                "rsi_14": round(rsi, 2),
                "macd_histogram": round(macd_hist, 4),
                "bb_zscore": round(bb_zscore, 2),
                "annualized_vol_pct": round(vol_ann * 100, 2),
                "rolling_sharpe": round(sharpe, 2),
                "rolling_sortino": round(sortino, 2),
                "var_95_pct": round(var_95 * 100, 2),
                "cvar_95_pct": round(cvar_95 * 100, 2),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "recommended_stop_loss_usd": recommended_stop_loss
            },
            "citations": citations,
            "trajectory": list(self.trajectory_log)
        }
