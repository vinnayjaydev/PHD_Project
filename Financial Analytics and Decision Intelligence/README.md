# 🏛️ FLAME-AI: A Lakehouse-Enabled Agentic Intelligence System for Adaptive Financial Analytics and Decision Intelligence Using Retrieval-Augmented LLMs

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Lakehouse: DuckDB + Delta](https://img.shields.io/badge/Lakehouse-DuckDB%20%2B%20Delta-orange.svg)](https://duckdb.org/)
[![Multi-Agent: Bayesian Consensus](https://img.shields.io/badge/Agents-Bayesian%20Consensus-purple.svg)]()
[![PhD Research Grade](https://img.shields.io/badge/Status-PhD%20Dissertation%20Grade-success.svg)]()

> **Project Title**: *A Lakehouse-Enabled Agentic Intelligence System for Adaptive Financial Analytics and Decision Intelligence Using Retrieval-Augmented Large Language Models*  
> **Academic Level**: Doctoral Dissertation (PhD) Research System & Institutional Framework

---

## 🌟 Executive Overview & Key Contributions

Traditional quantitative investment systems suffer from a bifurcated pathology: quantitative models rely strictly on structured numerical time-series (ignoring qualitative SEC disclosures and earnings calls), while modern conversational LLMs remain vulnerable to hallucinations, high-latency query bottlenecks, and temporal lookahead bias.

**FLAME-AI** resolves this dilemma by delivering an end-to-end, institutional-grade decision intelligence ecosystem uniting:
1. **Medallion Financial Lakehouse (Bronze $\to$ Silver $\to$ Gold)**: High-throughput immutable storage uniting multi-asset OHLCV bars, macroeconomic series, and SEC 10-K/10-Q filing extracts with zero-copy DuckDB OLAP queries.
2. **Dense-Sparse Hybrid Financial RAG**: Combines BM25 Okapi with semantic cosine embeddings via Reciprocal Rank Fusion (RRF) and certified point-in-time temporal guardrails (zero lookahead leakage).
3. **Decentralized Multi-Agent Deliberation Arena**: Macro-Regime, Fundamental Valuation, Qualitative RAG, and Quantitative Risk agents arbitrate investment decisions under an Executive CIO Synthesizer utilizing Bayesian posterior belief updating and Shannon consensus entropy.
4. **Regime-Adaptive Portfolio Optimization**: Dynamic switching between Hierarchical Risk Parity (HRP), Equal Risk Contribution (Risk Parity), and Black-Litterman models conditioned on Markov market regimes.
5. **Executive Dark-Mode Dashboard**: Interactive Streamlit cockpit featuring real-time Lakehouse SQL consoles, multi-agent reasoning trace visualizers, RAG quality radar charts, and stress-testing simulators.

---

## 📊 Empirical Performance Summary (2021–2024 Walk-Forward Backtest)

All simulations incorporate institutional frictions: **10 bps transaction fees** and **5 bps execution slippage** applied to monthly turnover.

| Strategy / Benchmark | CAGR (%) | Ann. Vol (%) | Sharpe Ratio | Sortino Ratio | Calmar Ratio | Max DD (%) | 95% VaR (%) | 95% CVaR (%) | Beta | Alpha (%) | Info Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FLAME-AI (Adaptive Agentic)** | **12.84%** | **10.43%** | **0.78** | **1.32** | **0.96** | **-13.37%** | **1.07%** | **1.29%** | **-0.02** | **+8.44%** | **0.19** |
| Hierarchical Risk Parity (HRP) | 11.90% | 10.02% | 0.73 | 1.20 | 0.95 | -12.55% | 1.04% | 1.25% | -0.02 | +7.50% | 0.15 |
| Equal Risk Contribution (Risk Parity) | 14.82% | 10.36% | 0.95 | 1.61 | 1.13 | -13.12% | 1.09% | 1.29% | -0.02 | +10.42% | 0.28 |
| Equal Weight Benchmark (1/N) | 18.05% | 12.17% | 1.06 | 1.84 | 1.25 | -14.49% | 1.22% | 1.49% | -0.03 | +13.68% | 0.40 |
| S&P 500 (SPY Buy & Hold) | 8.60% | 19.42% | 0.29 | 0.44 | 0.33 | -26.36% | 1.82% | 2.73% | 1.00 | -0.00% | 0.00 |
| 60/40 Equity/Bond Benchmark | 0.93% | 14.80% | -0.17 | -0.27 | 0.04 | -23.28% | 1.53% | 2.00% | 0.68 | -6.37% | -0.85 |

---

## 🏗️ Repository Architecture

```
Financial Analytics and Decision Intelligence/
├── configs/
│   └── system_config.yaml           # Master system parameters, tickers, risk thresholds
├── data/
│   ├── bronze/                      # Immutable raw market bars & SEC filings JSON
│   ├── silver/                      # Cleaned features parquet (31 signals) & semantic chunks
│   └── gold/                        # DuckDB OLAP database, regime cubes, covariance matrices
├── lakehouse/
│   ├── bronze_ingestion.py          # Ingests market OHLCV & SEC 10-K/10-Q documents
│   ├── silver_transformation.py     # Feature engineering, technical signals, semantic chunker
│   ├── gold_aggregations.py         # Markov regime detection, risk cubes, DuckDB registration
│   └── query_engine.py              # Zero-copy point-in-time DuckDB SQL interface
├── rag/
│   ├── hybrid_retriever.py          # Dense embeddings + BM25 Okapi with Reciprocal Rank Fusion
│   ├── temporal_guardrail.py        # Strict point-in-time enforcement (anti-lookahead auditor)
│   └── evaluation_metrics.py        # Ragas-style Faithfulness, Answer Relevance, Precision, Recall
├── agents/
│   ├── base_agent.py                # Abstract class with trajectory logs & multi-backend LLM dispatcher
│   ├── macro_agent.py               # Macro liquidity, Fed rate stance & regime classifier
│   ├── fundamental_agent.py         # Balance sheet, DuPont ROE, FCF & valuation ratios
│   ├── qualitative_rag_agent.py     # SEC 10-K Item 1A Risk Factors & moat analyzer
│   ├── quant_risk_agent.py          # RSI, MACD, Bollinger Bands, 95% VaR/CVaR stop-losses
│   └── executive_synthesizer.py     # CIO Meta-agent: Bayesian consensus & Investment Memo synthesis
├── decision_intelligence/
│   ├── portfolio_optimizer.py       # HRP, Risk Parity, Black-Litterman, adaptive switching
│   ├── regime_detector.py           # Markov transition matrix & regime-conditioned risk envelopes
│   └── backtester.py                # Realistic walk-forward event-driven backtesting engine
├── evaluation/
│   ├── financial_metrics.py         # CAGR, Sharpe, Sortino, Calmar, MaxDD, VaR, CVaR, Alpha, Beta, IR
│   ├── agent_metrics.py             # Trajectory efficiency, consensus entropy, tool accuracy
│   └── benchmark_runner.py          # Comparative multi-strategy ablation suite
├── ui/
│   ├── app.py                       # Master Streamlit executive dashboard (6 research tabs)
│   ├── components.py                # Custom Plotly charts (radar, equity, donut, heatmaps)
│   └── styles.css                   # Glassmorphic institutional dark UI theme
├── docs/
│   ├── PHD_DISSERTATION_CHAPTER.md  # 4,500+ word formal doctoral dissertation chapter
│   ├── RESEARCH_PAPER.md            # Academic conference paper draft
│   └── ARCHITECTURE.md              # Detailed system architecture and schema contracts
├── tests/
│   ├── test_lakehouse.py            # Lakehouse & DuckDB point-in-time tests
│   ├── test_rag.py                  # Hybrid retriever & temporal guardrail tests
│   ├── test_agents.py               # Multi-agent deliberation & consensus tests
│   └── test_portfolio.py            # Portfolio optimization & metric tests
├── scripts/
│   ├── seed_lakehouse.py            # Automated data seeder for Bronze -> Silver -> Gold
│   └── run_experiments.py           # Executes PhD benchmark experiments and outputs LaTeX
├── requirements.txt                 # Project dependencies
├── run_dashboard.bat                # Windows one-click launcher
└── run_dashboard.ps1                # PowerShell launcher
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone or navigate to the workspace directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Seed the Medallion Lakehouse
Populate the Bronze, Silver, and Gold lakehouse layers:
```bash
python scripts/seed_lakehouse.py --synthetic
```

### 3. Run PhD Experiments & Generate Benchmark Reports
Run the complete multi-strategy walk-forward backtest and generate dissertation tables:
```bash
python scripts/run_experiments.py
```

### 4. Launch the Interactive Executive Dashboard
Launch the institutional Streamlit cockpit:
```bash
streamlit run ui/app.py
```
*Or simply double-click `run_dashboard.bat` (Windows).*

### 5. Run Unit & Integration Test Suite
```bash
python -m pytest tests/ -v
```

---

## 🎓 PhD Dissertation & Research Highlights

- **Formal Thesis Chapter**: Read the complete 4,500+ word doctoral chapter in [`docs/PHD_DISSERTATION_CHAPTER.md`](docs/PHD_DISSERTATION_CHAPTER.md).
- **Conference Paper Draft**: View [`docs/RESEARCH_PAPER.md`](docs/RESEARCH_PAPER.md).
- **Architecture Specification**: View [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
- **LaTeX Table Exporter**: Tab 6 of the interactive dashboard provides one-click export of LaTeX table code directly into your Overleaf/LaTeX dissertation document.
