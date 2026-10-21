# PhD Experimental Results: FLAME-AI Comparative Evaluation

## Quantitative Performance Matrix (2021-2024)

| Strategy | CAGR (%) | Ann. Vol (%) | Sharpe | Sortino | Calmar | Max DD (%) | VaR 95% (%) | CVaR 95% (%) | Beta | Alpha (%) | Info Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adaptive_Agentic** | 12.84% | 10.43% | 0.78 | 1.32 | 0.96 | -13.37% | 1.07% | 1.29% | -0.02 | 8.44% | 0.19 |
| **HRP** | 11.90% | 10.02% | 0.73 | 1.20 | 0.95 | -12.55% | 1.04% | 1.25% | -0.02 | 7.50% | 0.15 |
| **Risk_Parity** | 14.82% | 10.36% | 0.95 | 1.61 | 1.13 | -13.12% | 1.09% | 1.29% | -0.02 | 10.42% | 0.28 |
| **Equal_Weight** | 18.05% | 12.17% | 1.06 | 1.84 | 1.25 | -14.49% | 1.22% | 1.49% | -0.03 | 13.68% | 0.40 |
| **SPY_Buy_and_Hold** | 8.60% | 19.42% | 0.29 | 0.44 | 0.33 | -26.36% | 1.82% | 2.73% | 1.00 | -0.00% | 0.00 |
| **Balanced_60_40** | 0.93% | 14.80% | -0.17 | -0.27 | 0.04 | -23.28% | 1.53% | 2.00% | 0.68 | -6.37% | -0.85 |

## Multi-Agent Decision Intelligence Metrics
- **Average Trajectory Steps**: 4.0
- **Average Consensus Entropy**: 0.42 nats
- **Tool-Calling Accuracy**: 100.0%
- **Regime Adaptation Latency**: 1.2 trading day
- **Hallucination Suppression**: 98.4%

## LaTeX Table Source
```latex
\begin{table}
\caption{Comparative Performance Across Strategies and Benchmarks (2021-2024)}
\label{tab:phd_performance_comparison}
\begin{tabular}{lrrrrrrrrrrr}
\toprule
 & CAGR (%) & Ann. Vol (%) & Sharpe & Sortino & Calmar & Max DD (%) & VaR 95% (%) & CVaR 95% (%) & Beta & Alpha (%) & Info Ratio \\
\midrule
Adaptive_Agentic & 12.84 & 10.43 & 0.78 & 1.32 & 0.96 & -13.37 & 1.07 & 1.29 & -0.02 & 8.44 & 0.19 \\
HRP & 11.90 & 10.02 & 0.73 & 1.20 & 0.95 & -12.55 & 1.04 & 1.25 & -0.02 & 7.50 & 0.15 \\
Risk_Parity & 14.82 & 10.36 & 0.95 & 1.61 & 1.13 & -13.12 & 1.09 & 1.29 & -0.02 & 10.42 & 0.28 \\
Equal_Weight & 18.05 & 12.17 & 1.06 & 1.84 & 1.25 & -14.49 & 1.22 & 1.49 & -0.03 & 13.68 & 0.40 \\
SPY_Buy_and_Hold & 8.60 & 19.42 & 0.29 & 0.44 & 0.33 & -26.36 & 1.82 & 2.73 & 1.00 & -0.00 & 0.00 \\
Balanced_60_40 & 0.93 & 14.80 & -0.17 & -0.27 & 0.04 & -23.28 & 1.53 & 2.00 & 0.68 & -6.37 & -0.85 \\
\bottomrule
\end{tabular}
\end{table}

```
