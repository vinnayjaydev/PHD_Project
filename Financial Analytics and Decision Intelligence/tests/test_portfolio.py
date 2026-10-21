"""
Unit tests for Portfolio Optimization, Backtesting & Quantitative Metrics.
"""

import pytest
import numpy as np
import pandas as pd
from decision_intelligence.portfolio_optimizer import AdaptivePortfolioOptimizer
from decision_intelligence.backtester import InstitutionalBacktester
from evaluation.financial_metrics import FinancialPerformanceEvaluator


def test_portfolio_optimizer():
    opt = AdaptivePortfolioOptimizer()
    cov = pd.DataFrame(
        np.array([[0.04, 0.01], [0.01, 0.05]]),
        index=["AAPL", "NVDA"],
        columns=["AAPL", "NVDA"]
    )
    # Test HRP
    w_hrp = opt.optimize(cov, method="hrp")
    assert abs(sum(w_hrp.values()) - 1.0) < 1e-3
    assert w_hrp["AAPL"] > 0
    assert w_hrp["NVDA"] > 0

    # Test Risk Parity
    w_rp = opt.optimize(cov, method="risk_parity")
    assert abs(sum(w_rp.values()) - 1.0) < 1e-3

    # Test Regime Adaptive
    w_ada = opt.optimize(cov, regime_id=2, method="regime_adaptive")
    assert abs(sum(w_ada.values()) - 1.0) < 1e-3


def test_financial_metrics_evaluator():
    evaluator = FinancialPerformanceEvaluator()
    equity_series = pd.Series([1000.0 * (1.0 + 0.0006)**t for t in range(252)])
    metrics = evaluator.evaluate_strategy(equity_series)
    assert metrics["cagr_pct"] > 0.0
    assert metrics["annualized_volatility_pct"] >= 0.0
    assert "sharpe_ratio" in metrics
    assert "max_drawdown_pct" in metrics
    assert "var_95_pct" in metrics
    assert "cvar_95_pct" in metrics


def test_backtester_execution():
    bt = InstitutionalBacktester()
    res = bt.run_backtest("Adaptive_Agentic", start_date="2022-01-01", end_date="2023-01-01")
    assert "final_equity" in res
    assert res["final_equity"] > 0
    assert len(res["backtest_df"]) > 100
    assert "portfolio_drawdown" in res["backtest_df"].columns
