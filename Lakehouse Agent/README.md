# Lakehouse-Enabled Agentic Intelligence System for Adaptive Personalized Financial Recommendation using Retrieval-Augmented LLMs

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DuckDB](https://img.shields.io/badge/Lakehouse-DuckDB%20%2B%20Parquet-orange.svg)](https://duckdb.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI%20%26%20REST-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-12%2F12%20passing-brightgreen.svg)]()

An end-to-end autonomous financial intelligence system integrating a **Medallion Lakehouse architecture (DuckDB + Parquet)** with a **Multi-Agent Cognitive Framework** and **Hybrid Retrieval-Augmented Generation (RAG)** to deliver real-time, adaptive, personalized investment recommendations backed by fiduciary guardrails (FINRA Rule 2111 & SEC Reg BI).

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       Client / Interactive Web UI     |
                      |   (Chart.js, Tailwind, REST / SSE)    |
                      +-------------------+-------------------+
                                          |
                                          v
+---------------------------------------------------------------------------------+
|                         FastAPI Application Gateway                             |
+---------------------------------------------------------------------------------+
                                          |
       +----------------------------------+----------------------------------+
       |                                                                     |
       v                                                                     v
+-------------------------------+                     +-----------------------------------+
| Multi-Agent Reasoning Engine  |                     | Embedded DuckDB Lakehouse Engine  |
|                               |                     |                                   |
| 1. User Profiler Agent        |<------------------->| Bronze Tier (Raw Ingestion)       |
| 2. Lakehouse Analyst Agent    |    (Direct OLAP     |   - Ticker Feeds, SEC 10-K Docs   |
| 3. Market Intelligence Agent  |     SQL Queries)    |   - Investor Profiles & Trades    |
| 4. Portfolio Optimizer Agent  |                     |                                   |
| 5. Compliance Guardrail Agent |                     | Silver Tier (Cleaned & Enriched)  |
| 6. Master Advisory Synthesizer|                     |   - Financial Health Ratios, DTI  |
+---------------+---------------+                     |   - Portfolio Beta & Exposures    |
                |                                     |                                   |
                v                                     | Gold Tier (Analytical Marts)      |
+-------------------------------+                     |   - Portfolio Risk Feature Store  |
|     Hybrid RAG Vector Store   |<------------------->|   - Curated Asset Universe Mart   |
|   (Dense Embeddings, Cosine   |   (Syncs Unstruc-   |   - Recommendation Audit History  |
|    Similarity, Provenance)    |    tured Filings)   +-----------------------------------+
+-------------------------------+
```

---

## Key Theoretical & Engineering Pillars

### 1. Lakehouse Medallion Architecture (DuckDB + Parquet)
- **Bronze Tier**: Ingests raw structured ticker streams, historical transactions, macro prints (CPI, Fed funds rate, 10Y Treasury), and raw SEC 10-K disclosures.
- **Silver Tier**: Executes conformed joins and feature enrichment:
  - Debt-to-Income (DTI) ratio: $\text{DTI} = \frac{\text{Total Debt}}{\text{Annual Income}}$
  - Emergency Fund Runway: $\text{Runway} = \frac{\text{Liquid Net Worth}}{\text{Monthly Expenses}}$
  - Holding weights and concentration anomaly detection ($w_i > 15\%$).
- **Gold Tier**: High-performance analytical marts:
  - Portfolio-weighted Beta: $\beta_p = \sum_{i=1}^N w_i \cdot \beta_i$
  - Herfindahl-Hirschman Diversification Score: $\text{Score} = \left(1 - \sum w_i^2\right) \times 100$
  - Risk Drift Detection: Automated alerts when actual portfolio beta deviates from declared risk capacity.

### 2. Multi-Agent Collaborative Reasoning Framework
- **UserProfileAgent**: Diagnoses behavioral risk appetite vs. true financial capacity and emergency reserves.
- **LakehouseAnalyticsAgent**: Runs direct OLAP SQL aggregations over Gold/Silver tables to diagnose asset class skews.
- **MarketIntelligenceAgent**: Queries unstructured financial disclosures (SEC Form 10-K, FOMC statements, analyst outlooks) using hybrid RAG to ground claims with verifiable citations (`[Source | Title (Ref: ChunkID)]`).
- **PortfolioOptimizerAgent**: Solves strategic asset allocation weights across 5 asset classes (Equities, Fixed Income, Real Estate, Commodities, Cash) and generates exact share rebalancing orders.
- **ComplianceGuardrailAgent**: Enforces regulatory fiduciary standards (FINRA Rule 2111 / SEC Regulation Best Interest). Caps equity exposure by risk tier, protects liquidity buffers, and corrects allocation violations.
- **FinancialAdvisorOrchestrator**: Master coordinator assembling the executive briefing and persisting recommendations to `gold_recommendation_history`.

### 3. Adaptive Personalization & Scenario Stress Testing
- Dynamic trigger adaptation when client goals or macroeconomic regimes shift.
- Real-time stress simulator testing portfolio resilience against:
  1. *Tech Multiple Contraction (-25%)*
  2. *Unexpected Fed 100bps Rate Hike*
  3. *Stagflationary Shock (CPI spike + GDP contraction)*

---

## Quickstart Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
cd "d:\Antigravity\Lakehouse Agent"
pip install -r requirements.txt
```

### 2. Configuration (`.env`)
Copy `.env.example` to `.env` and optionally configure your API keys:
```bash
cp .env.example .env
```
Supported LLM Providers:
- `gemini`: Uses Google Gemini API (Gemini 2.5 Flash / Pro).
- `openai`: Uses OpenAI API (GPT-4o / GPT-4o-mini).
- `mock`: Built-in domain heuristic engine (100% offline, zero API keys required).

### 3. Launch the Interactive Web Dashboard
Run the server:
```bash
python run_server.py
```
Open your browser to: **`http://127.0.0.1:8000`**

### 4. Command-Line Interface (CLI)
You can also run batch tasks directly from the terminal:
```bash
# Initialize Lakehouse Medallion ETL tables
python -m lakehouse_agent.cli --init

# Run multi-agent recommendation for investor INV001
python -m lakehouse_agent.cli --investor INV001

# Run custom SQL query on the Lakehouse
python -m lakehouse_agent.cli --query "SELECT * FROM gold_portfolio_analytics;"

# List all Lakehouse tables
python -m lakehouse_agent.cli --list-tables
```

---

## Running Automated Tests

Run the complete test suite:
```bash
python -m unittest discover -s lakehouse_agent/tests -v
```

All 12 unit and integration tests validate:
- Bronze ingestion and schema integrity
- Silver derived metrics and risk profiling
- Gold feature marts and DuckDB Parquet persistence
- Vector embedding generation and cosine similarity search
- Multi-agent cognitive pipeline execution
- Fiduciary suitability compliance corrections
- FastAPI REST endpoints and scenario simulations

---

## Directory Structure

```
d:\Antigravity\Lakehouse Agent\
├── .env.example
├── requirements.txt
├── README.md
├── run_server.py
├── lakehouse_agent/
│   ├── config.py
│   ├── cli.py
│   ├── lakehouse/
│   │   ├── engine.py          # DuckDB Lakehouse connection & ACID queries
│   │   ├── bronze.py          # Raw feeds, transactions, profiles, filings
│   │   ├── silver.py          # Cleaned, enriched metrics, DTI, portfolio weights
│   │   ├── gold.py            # Feature marts: portfolio analytics, investment universe
│   │   └── data_generator.py  # Realistic financial market & investor data generator
│   ├── vector_store/
│   │   ├── embeddings.py      # Unified Embeddings (Gemini, OpenAI, Local Dense)
│   │   └── store.py           # Hybrid RAG Semantic Vector Store
│   ├── agents/
│   │   ├── base.py            # Base agent with chain-of-thought trace logger
│   │   ├── llm_provider.py    # Multi-backend LLM client with local fallback
│   │   ├── user_profiler.py   # Investor profiling & risk capacity agent
│   │   ├── lakehouse_analyst.py # DuckDB OLAP quantitative analytics agent
│   │   ├── market_intelligence.py # SEC/Macro RAG knowledge agent
│   │   ├── portfolio_optimizer.py # MPT allocation & rebalancing trade generator
│   │   ├── compliance_guard.py# FINRA 2111 / SEC Reg BI suitability gatekeeper
│   │   └── orchestrator.py    # Master multi-agent coordinator
│   ├── web/
│   │   ├── app.py             # FastAPI REST server & scenario endpoints
│   │   └── static/            # Responsive web UI (Tailwind CSS, Chart.js)
│   └── tests/
│       ├── test_lakehouse.py
│       ├── test_rag.py
│       ├── test_agents.py
│       └── test_api.py
```

---

## License & Attribution
Developed for Advanced Financial AI Research. Released under the MIT License.
