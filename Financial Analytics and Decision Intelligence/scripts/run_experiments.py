"""
Experiment Runner: Executes PhD Benchmark Comparisons, Ablation Studies,
and Saves Experimental Artifacts.
"""

import sys
import os
import json
import pandas as pd

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmark_runner import BenchmarkRunner
from agents.executive_synthesizer import ExecutiveSynthesizer


def run_phd_experiments():
    print("=" * 80)
    print("FLAME-AI: EXECUTING PHD BENCHMARK EXPERIMENTS & ABLATION STUDIES")
    print("=" * 80)

    runner = BenchmarkRunner()
    results = runner.run_full_benchmark(start_date="2021-06-01", end_date="2024-12-31")

    comp_dict = results["comparison_table"]
    df = pd.DataFrame(comp_dict).T

    print("\n" + "=" * 80)
    print("PHD COMPARATIVE FINANCIAL PERFORMANCE MATRIX (2021 - 2024)")
    print("=" * 80)
    print(df.to_string())

    print("\n" + "=" * 80)
    print("AGENTIC SYSTEM EVALUATION SCORECARD")
    print("=" * 80)
    for k, v in results["agentic_metrics"].items():
        print(f"  * {k.replace('_', ' ').title()}: {v}")

    # Test single-asset agentic deliberation memo
    print("\n" + "=" * 80)
    print("SAMPLE INSTITUTIONAL DECISION MEMO SYNTHESIS (NVDA)")
    print("=" * 80)
    cio = ExecutiveSynthesizer()
    memo = cio.analyze("NVDA", as_of_date="2024-05-15")
    print(memo["memo_text"])

    # Write Markdown summary report to docs/BENCHMARK_RESULTS.md
    report_md = f"""# PhD Experimental Results: FLAME-AI Comparative Evaluation

## Quantitative Performance Matrix (2021-2024)

| Strategy | CAGR (%) | Ann. Vol (%) | Sharpe | Sortino | Calmar | Max DD (%) | VaR 95% (%) | CVaR 95% (%) | Beta | Alpha (%) | Info Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for strat, row in comp_dict.items():
        report_md += f"| **{strat}** | {row['CAGR (%)']:.2f}% | {row['Ann. Vol (%)']:.2f}% | {row['Sharpe']:.2f} | {row['Sortino']:.2f} | {row['Calmar']:.2f} | {row['Max DD (%)']:.2f}% | {row['VaR 95% (%)']:.2f}% | {row['CVaR 95% (%)']:.2f}% | {row['Beta']:.2f} | {row['Alpha (%)']:.2f}% | {row['Info Ratio']:.2f} |\n"

    report_md += f"""
## Multi-Agent Decision Intelligence Metrics
- **Average Trajectory Steps**: {results['agentic_metrics']['avg_trajectory_steps']}
- **Average Consensus Entropy**: {results['agentic_metrics']['avg_consensus_entropy']} nats
- **Tool-Calling Accuracy**: {results['agentic_metrics']['tool_calling_accuracy_pct']}%
- **Regime Adaptation Latency**: {results['agentic_metrics']['regime_adaptation_latency_days']} trading day
- **Hallucination Suppression**: {results['agentic_metrics']['hallucination_suppression_pct']}%

## LaTeX Table Source
```latex
{results['latex_table']}
```
"""
    os.makedirs("docs", exist_ok=True)
    with open("docs/BENCHMARK_RESULTS.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n[SUCCESS] Experimental artifacts written to docs/BENCHMARK_RESULTS.md")


if __name__ == "__main__":
    run_phd_experiments()
