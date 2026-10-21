"""
Financial Performance Evaluator: PhD-Grade Quantitative Metric Suite.
Computes CAGR, Volatility, Sharpe, Sortino, Calmar, MaxDD, VaR, CVaR, Alpha, Beta, IR.
Part of the FLAME-AI Evaluation Framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class FinancialPerformanceEvaluator:
    """
    Computes rigorous institutional risk and return metrics from equity curves and return series.
    """

    def __init__(self, risk_free_rate_annual: float = 0.045):
        self.rf_annual = risk_free_rate_annual
        self.rf_daily = risk_free_rate_annual / 252.0

    def evaluate_strategy(
        self,
        portfolio_equity: pd.Series,
        benchmark_equity: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Computes full statistical risk-return profile.
        """
        n_days = len(portfolio_equity)
        if n_days < 2:
            return {}

        returns = portfolio_equity.pct_change().dropna().values
        cum_ret = (portfolio_equity.iloc[-1] / portfolio_equity.iloc[0]) - 1.0
        years = n_days / 252.0

        # 1. CAGR (Annualized Return)
        cagr = (portfolio_equity.iloc[-1] / portfolio_equity.iloc[0]) ** (1.0 / max(years, 0.1)) - 1.0

        # 2. Annualized Volatility
        vol_ann = np.std(returns) * np.sqrt(252)

        # 3. Sharpe Ratio
        excess_mean = np.mean(returns) - self.rf_daily
        sharpe = (excess_mean / (np.std(returns) + 1e-9)) * np.sqrt(252)

        # 4. Sortino Ratio (Downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else vol_ann
        sortino = (excess_mean * 252.0) / (downside_std + 1e-9)

        # 5. Maximum Drawdown & Duration
        peak = portfolio_equity.cummax()
        drawdowns = (portfolio_equity - peak) / peak
        max_dd = float(drawdowns.min())

        # Max drawdown duration (in business days)
        is_zero = (drawdowns == 0)
        dd_durations = (~is_zero).astype(int).groupby(is_zero.cumsum()).cumsum()
        max_dd_duration_days = int(dd_durations.max()) if len(dd_durations) > 0 else 0

        # 6. Calmar Ratio
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0

        # 7. Value at Risk (VaR 95% and 99%)
        var_95 = float(-np.percentile(returns, 5))
        var_99 = float(-np.percentile(returns, 1))

        # 8. Expected Shortfall / CVaR 95%
        tail_95 = returns[returns <= -var_95]
        cvar_95 = float(-np.mean(tail_95)) if len(tail_95) > 0 else var_95

        # 9. Benchmark-relative metrics (Alpha, Beta, Treynor, Information Ratio)
        alpha, beta, info_ratio, treynor = 0.0, 1.0, 0.0, 0.0
        if benchmark_equity is not None:
            combined_df = pd.DataFrame({"port": portfolio_equity.values, "bench": benchmark_equity.values}).dropna()
            port_ret = combined_df["port"].pct_change().dropna()
            bench_ret = combined_df["bench"].pct_change().dropna()
            common_idx = port_ret.index.intersection(bench_ret.index)

            if len(common_idx) > 10:
                r_p = port_ret.loc[common_idx].values
                r_b = bench_ret.loc[common_idx].values
                cov_mat = np.cov(r_p, r_b)
                cov = cov_mat[0, 1]
                var_bench = np.var(r_b) + 1e-9
                beta = float(cov / var_bench)

                bench_cagr = (combined_df["bench"].iloc[-1] / combined_df["bench"].iloc[0]) ** (1.0 / max(years, 0.1)) - 1.0
                alpha = float(cagr - (self.rf_annual + beta * (bench_cagr - self.rf_annual)))

                tracking_diff = r_p - r_b
                tracking_error = np.std(tracking_diff) * np.sqrt(252) + 1e-9
                info_ratio = float((cagr - bench_cagr) / tracking_error)
                treynor = float((cagr - self.rf_annual) / (beta + 1e-6))

        return {
            "cumulative_return_pct": float(np.round(cum_ret * 100, 2)),
            "cagr_pct": float(np.round(cagr * 100, 2)),
            "annualized_volatility_pct": float(np.round(vol_ann * 100, 2)),
            "sharpe_ratio": float(np.round(sharpe, 2)),
            "sortino_ratio": float(np.round(sortino, 2)),
            "calmar_ratio": float(np.round(calmar, 2)),
            "max_drawdown_pct": float(np.round(max_dd * 100, 2)),
            "max_drawdown_duration_days": max_dd_duration_days,
            "var_95_pct": float(np.round(var_95 * 100, 2)),
            "var_99_pct": float(np.round(var_99 * 100, 2)),
            "cvar_95_pct": float(np.round(cvar_95 * 100, 2)),
            "capm_beta": float(np.round(beta, 2)),
            "jensen_alpha_pct": float(np.round(alpha * 100, 2)),
            "information_ratio": float(np.round(info_ratio, 2)),
            "treynor_ratio": float(np.round(treynor, 2))
        }


if __name__ == "__main__":
    evaluator = FinancialPerformanceEvaluator()
    dummy_equity = pd.Series([100 * (1 + 0.0008)**i for i in range(500)])
    m = evaluator.evaluate_strategy(dummy_equity)
    print("Evaluator Test Output:", m)
