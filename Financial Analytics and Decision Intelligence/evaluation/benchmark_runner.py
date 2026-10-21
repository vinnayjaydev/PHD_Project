"""
Benchmark Runner: End-to-End Comparative Ablation Studies.
Executes multi-strategy walk-forward simulations and outputs PhD-grade comparison tables & LaTeX code.
Part of the FLAME-AI Evaluation Framework.
"""

import os
import json
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from decision_intelligence.backtester import InstitutionalBacktester
from evaluation.financial_metrics import FinancialPerformanceEvaluator
from evaluation.agent_metrics import AgenticSystemEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BenchmarkRunner")


class BenchmarkRunner:
    """
    Orchestrates comparative benchmarking across:
    1. FLAME-AI Adaptive Agentic Lakehouse Strategy
    2. Hierarchical Risk Parity (HRP)
    3. Equal Risk Contribution (Risk Parity)
    4. Equal Weight (1/N)
    5. SPY S&P 500 Buy & Hold
    6. 60/40 Equity/Bond Balanced Benchmark
    """

    def __init__(self, output_dir: str = "data/gold"):
        self.output_dir = output_dir
        self.backtester = InstitutionalBacktester()
        self.fin_eval = FinancialPerformanceEvaluator(risk_free_rate_annual=0.045)
        self.agent_eval = AgenticSystemEvaluator()

    def run_full_benchmark(
        self,
        start_date: str = "2021-06-01",
        end_date: str = "2024-12-31"
    ) -> Dict[str, Any]:
        """
        Executes all benchmark strategies and compiles comparative metric scorecard.
        """
        strategies = ["Adaptive_Agentic", "HRP", "Risk_Parity", "Equal_Weight"]
        results = {}
        backtest_results = {}

        logger.info("Starting PhD Comparative Benchmark Suite...")

        for strat in strategies:
            bt_res = self.backtester.run_backtest(
                strategy_name=strat,
                start_date=start_date,
                end_date=end_date
            )
            backtest_results[strat] = bt_res

            # Evaluate metrics against SPY benchmark
            equity = bt_res["backtest_df"]["portfolio_equity"]
            spy_equity = bt_res["backtest_df"]["spy_benchmark_equity"]

            metrics = self.fin_eval.evaluate_strategy(equity, spy_equity)
            results[strat] = metrics

        # Add passive benchmarks
        sample_df = backtest_results["Adaptive_Agentic"]["backtest_df"]
        spy_eq = sample_df["spy_benchmark_equity"]
        bal_eq = sample_df["balanced_60_40_equity"]

        results["SPY_Buy_and_Hold"] = self.fin_eval.evaluate_strategy(spy_eq, spy_eq)
        results["Balanced_60_40"] = self.fin_eval.evaluate_strategy(bal_eq, spy_eq)

        # Build Comparative DataFrame
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df[[
            "cagr_pct", "annualized_volatility_pct", "sharpe_ratio", "sortino_ratio",
            "calmar_ratio", "max_drawdown_pct", "var_95_pct", "cvar_95_pct",
            "capm_beta", "jensen_alpha_pct", "information_ratio"
        ]]

        # Format column names for academic display
        comparison_df.columns = [
            "CAGR (%)", "Ann. Vol (%)", "Sharpe", "Sortino",
            "Calmar", "Max DD (%)", "VaR 95% (%)", "CVaR 95% (%)",
            "Beta", "Alpha (%)", "Info Ratio"
        ]

        # Generate LaTeX table code
        latex_table = comparison_df.to_latex(
            float_format="%.2f",
            caption="Comparative Performance Across Strategies and Benchmarks (2021-2024)",
            label="tab:phd_performance_comparison"
        )

        output_payload = {
            "comparison_table": comparison_df.to_dict(orient="index"),
            "latex_table": latex_table,
            "backtest_results": backtest_results,
            "agentic_metrics": self.agent_eval.evaluate_multi_agent_session([])
        }

        # Save to disk
        out_path = os.path.join(self.output_dir, "benchmark_comparison_results.json")
        with open(out_path, "w") as f:
            json.dump({
                "comparison_table": comparison_df.to_dict(orient="index"),
                "latex_table": latex_table,
                "agentic_metrics": output_payload["agentic_metrics"]
            }, f, indent=2)

        logger.info(f"Benchmark results successfully saved to {out_path}")
        return output_payload


if __name__ == "__main__":
    runner = BenchmarkRunner()
    res = runner.run_full_benchmark()
    df = pd.DataFrame(res["comparison_table"]).T
    print("Comparative Performance Table:\n", df)
    print("\nLaTeX Code:\n", res["latex_table"])
