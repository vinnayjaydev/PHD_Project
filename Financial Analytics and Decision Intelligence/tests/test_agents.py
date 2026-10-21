"""
Unit tests for Multi-Agent Deliberation Subsystem.
"""

import pytest
from agents.executive_synthesizer import ExecutiveSynthesizer
from agents.macro_agent import MacroRegimeAgent
from agents.fundamental_agent import FundamentalValuationAgent
from agents.quant_risk_agent import QuantRiskAgent


def test_individual_agents():
    macro = MacroRegimeAgent()
    m_res = macro.analyze("AAPL", as_of_date="2024-05-01")
    assert m_res["stance"] in ["BULLISH", "BEARISH", "DEFENSIVE", "NEUTRAL"]
    assert 0.0 <= m_res["confidence"] <= 1.0
    assert len(m_res["trajectory"]) > 0

    fund = FundamentalValuationAgent()
    f_res = fund.analyze("MSFT", as_of_date="2024-05-01")
    assert f_res["stance"] in ["BULLISH", "BEARISH", "DEFENSIVE", "NEUTRAL"]
    assert "revenue_growth_yoy_pct" in f_res["key_metrics"]

    quant = QuantRiskAgent()
    q_res = quant.analyze("NVDA", as_of_date="2024-05-01")
    assert "rsi_14" in q_res["key_metrics"]
    assert "var_95_pct" in q_res["key_metrics"]


def test_executive_synthesizer_consensus():
    cio = ExecutiveSynthesizer()
    memo = cio.analyze("NVDA", as_of_date="2024-05-01")
    assert "recommendation" in memo
    assert memo["recommendation"] in ["STRONG_BUY", "MODERATE_BUY", "HOLD_NEUTRAL", "TRIM_DEFENSIVE", "SELL"]
    assert 0.0 <= memo["conviction_pct"] <= 100.0
    assert memo["consensus_entropy"] >= 0.0
    assert "memo_text" in memo
    assert len(memo["citations_ledger"]) > 0
