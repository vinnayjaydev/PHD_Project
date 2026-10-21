"""
FLAME-AI: Multi-Agent Adaptive Intelligence Framework
Contains Base Agent, Macro Regime Agent, Fundamental Valuation Agent,
Qualitative RAG Agent, Quant Risk Agent, and Executive CIO Synthesizer.
"""

from .base_agent import BaseFinancialAgent
from .macro_agent import MacroRegimeAgent
from .fundamental_agent import FundamentalValuationAgent
from .qualitative_rag_agent import QualitativeRAGAgent
from .quant_risk_agent import QuantRiskAgent
from .executive_synthesizer import ExecutiveSynthesizer

__all__ = [
    "BaseFinancialAgent",
    "MacroRegimeAgent",
    "FundamentalValuationAgent",
    "QualitativeRAGAgent",
    "QuantRiskAgent",
    "ExecutiveSynthesizer"
]
