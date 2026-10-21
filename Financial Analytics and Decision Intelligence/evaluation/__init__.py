"""
FLAME-AI: Quantitative Evaluation Suite for PhD Financial Analytics & Agentic Systems
Includes Formal Financial Metrics, Multi-Agent Trajectory Metrics, and Comparative Benchmarking.
"""

from .financial_metrics import FinancialPerformanceEvaluator
from .agent_metrics import AgenticSystemEvaluator
from .benchmark_runner import BenchmarkRunner

__all__ = [
    "FinancialPerformanceEvaluator",
    "AgenticSystemEvaluator",
    "BenchmarkRunner"
]
