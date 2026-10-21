"""
FLAME-AI: Decision Intelligence & Regime-Adaptive Portfolio Optimization Engine
Includes Hierarchical Risk Parity (HRP), Black-Litterman Agent Views,
Regime-Adaptive Allocator, and Institutional Event-Driven Backtester.
"""

from .portfolio_optimizer import AdaptivePortfolioOptimizer
from .regime_detector import RegimeTransitionEngine
from .backtester import InstitutionalBacktester

__all__ = [
    "AdaptivePortfolioOptimizer",
    "RegimeTransitionEngine",
    "InstitutionalBacktester"
]
