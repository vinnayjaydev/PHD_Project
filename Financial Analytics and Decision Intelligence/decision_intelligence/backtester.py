"""
Institutional Event-Driven Backtester: Realistic Transaction Costs, Turnover, Slippage & Benchmarks.
Part of the FLAME-AI Decision Intelligence Subsystem.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from .portfolio_optimizer import AdaptivePortfolioOptimizer
from .regime_detector import RegimeTransitionEngine
from lakehouse.query_engine import LakehouseQueryEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InstitutionalBacktester")


class InstitutionalBacktester:
    """
    Event-driven walk-forward backtesting engine with realistic institutional frictions:
    - Transaction fees (bps)
    - Execution slippage (bps)
    - Turnover constraints
    - Multi-benchmark tracking (SPY Buy & Hold, 60/40 Equity/Bond)
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
        rebalance_days: int = 21  # Approx monthly
    ):
        self.initial_capital = initial_capital
        self.cost_rate = (transaction_cost_bps + slippage_bps) / 10000.0
        self.rebalance_days = rebalance_days
        self.optimizer = AdaptivePortfolioOptimizer()
        self.regime_engine = RegimeTransitionEngine()
        self.query_engine = LakehouseQueryEngine()

    def run_backtest(
        self,
        strategy_name: str = "Adaptive_Agentic",  # "Adaptive_Agentic", "HRP", "Risk_Parity", "Equal_Weight"
        start_date: str = "2021-06-01",
        end_date: str = "2024-12-31"
    ) -> Dict[str, Any]:
        """
        Runs walk-forward backtest over Lakehouse market features and gold cubes.
        """
        logger.info(f"Running backtest [{strategy_name}] from {start_date} to {end_date}...")

        # Load clean market features
        full_market = pd.read_parquet("data/silver/market_features.parquet")
        full_market["date"] = pd.to_datetime(full_market["date"])

        # Full historical return pivot for seamless rolling lookback
        equity_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "JNJ"]
        sub_full = full_market[full_market["ticker"].isin(equity_tickers)]
        full_returns = sub_full.pivot(index="date", columns="ticker", values="return_1d").fillna(0.0)

        # Slice backtest evaluation window
        market_df = full_market[(full_market["date"] >= start_date) & (full_market["date"] <= end_date)]
        sub_market = market_df[market_df["ticker"].isin(equity_tickers)]
        price_pivot = sub_market.pivot(index="date", columns="ticker", values="close").dropna()
        returns_pivot = sub_market.pivot(index="date", columns="ticker", values="return_1d").fillna(0.0)

        # Benchmark SPY and TLT
        spy_series = full_market[full_market["ticker"] == "SPY"].set_index("date")["close"]
        tlt_series = full_market[full_market["ticker"] == "TLT"].set_index("date")["close"]

        dates = price_pivot.index
        n_dates = len(dates)
        if n_dates < 50:
            raise ValueError("Insufficient historical data for backtesting.")

        # Agent views baseline estimates (e.g. higher expected returns for strong momentum/growth)
        agent_views_map = {
            "NVDA": 0.35, "MSFT": 0.22, "AAPL": 0.18, "GOOGL": 0.19,
            "AMZN": 0.20, "JPM": 0.14, "JNJ": 0.08
        }

        # Initialize tracking arrays
        equity_curve = np.zeros(n_dates)
        equity_curve[0] = self.initial_capital
        current_weights = {t: 1.0 / len(equity_tickers) for t in equity_tickers}
        weights_history = []
        turnover_history = []
        regime_history = []
        costs_total = 0.0

        days_since_rebalance = self.rebalance_days  # Force rebalance on day 1

        for t in range(n_dates):
            current_date_str = str(dates[t])[:10]

            # 1. Check if Rebalance is due
            if days_since_rebalance >= self.rebalance_days:
                days_since_rebalance = 0

                # Determine active regime
                regime_info = self.query_engine.get_active_regime(current_date_str)
                reg_id = regime_info.get("regime_id", 3)
                regime_history.append({"date": dates[t], "regime_id": reg_id, "regime_name": regime_info.get("regime_name")})

                # Rolling 63-day lookback covariance slice from full historical returns
                prior_returns = full_returns[full_returns.index < dates[t]]
                if len(prior_returns) >= 21:
                    cov_slice = prior_returns.iloc[-63:].cov() * 252.0
                else:
                    cov_slice = full_returns.iloc[:63].cov() * 252.0

                # Solve target weights
                if strategy_name == "Adaptive_Agentic":
                    target_weights = self.optimizer.optimize(
                        cov_slice, regime_id=reg_id, agent_views=agent_views_map, method="regime_adaptive"
                    )
                elif strategy_name == "HRP":
                    target_weights = self.optimizer.optimize(cov_slice, method="hrp")
                elif strategy_name == "Risk_Parity":
                    target_weights = self.optimizer.optimize(cov_slice, method="risk_parity")
                else:  # Equal Weight
                    target_weights = {ticker: 1.0 / len(equity_tickers) for ticker in equity_tickers}

                # Compute turnover and frictions
                turnover = float(sum(abs(target_weights.get(k, 0.0) - current_weights.get(k, 0.0)) for k in equity_tickers))
                if np.isnan(turnover):
                    turnover = 0.0
                turnover_history.append({"date": dates[t], "turnover": turnover})
                rebalance_cost = turnover * equity_curve[t] * self.cost_rate
                if not np.isnan(rebalance_cost):
                    equity_curve[t] = max(equity_curve[t] - rebalance_cost, 0.0)
                    costs_total += rebalance_cost

                current_weights = target_weights

            days_since_rebalance += 1
            weights_history.append({"date": dates[t], **current_weights})

            # 2. Portfolio return update for next day
            if t < n_dates - 1:
                daily_asset_returns = returns_pivot.iloc[t + 1]
                port_daily_ret = sum(current_weights[ticker] * daily_asset_returns[ticker] for ticker in equity_tickers)
                equity_curve[t + 1] = equity_curve[t] * (1.0 + port_daily_ret)

        # 3. Construct Backtest DataFrame
        backtest_df = pd.DataFrame({
            "date": dates,
            "portfolio_equity": equity_curve,
            "portfolio_return": pd.Series(equity_curve).pct_change().fillna(0.0)
        })

        # 4. Benchmarks alignment
        spy_aligned = spy_series.reindex(dates).ffill()
        spy_equity = self.initial_capital * (spy_aligned / spy_aligned.iloc[0])
        backtest_df["spy_benchmark_equity"] = spy_equity.values

        # 60/40 Equity/Bond
        tlt_aligned = tlt_series.reindex(dates).ffill()
        balanced_equity = self.initial_capital * (
            0.60 * (spy_aligned / spy_aligned.iloc[0]) + 0.40 * (tlt_aligned / tlt_aligned.iloc[0])
        )
        backtest_df["balanced_60_40_equity"] = balanced_equity.values

        # Drawdown curves
        backtest_df["portfolio_drawdown"] = (backtest_df["portfolio_equity"] - backtest_df["portfolio_equity"].cummax()) / backtest_df["portfolio_equity"].cummax()
        backtest_df["spy_drawdown"] = (backtest_df["spy_benchmark_equity"] - backtest_df["spy_benchmark_equity"].cummax()) / backtest_df["spy_benchmark_equity"].cummax()

        weights_df = pd.DataFrame(weights_history)
        turnover_df = pd.DataFrame(turnover_history)

        return {
            "strategy_name": strategy_name,
            "date_range": [str(dates[0])[:10], str(dates[-1])[:10]],
            "initial_capital": self.initial_capital,
            "final_equity": float(equity_curve[-1]),
            "total_friction_costs_usd": float(np.round(costs_total, 2)),
            "backtest_df": backtest_df,
            "weights_df": weights_df,
            "turnover_df": turnover_df,
            "regimes_history": regime_history
        }


if __name__ == "__main__":
    bt = InstitutionalBacktester()
    res = bt.run_backtest("Adaptive_Agentic")
    print(f"Strategy: {res['strategy_name']} | Final Equity: ${res['final_equity']:,.2f}")
