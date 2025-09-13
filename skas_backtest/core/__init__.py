"""
Core backtesting framework components.

This module contains the fundamental classes for running backtests:
- Backtest: Main backtesting engine
- Strategy: Base strategy interface
- Portfolio: Position and portfolio management
- DataManager: Data fetching and management
- RiskManager: Risk controls and validation
- MetricsCalculator: Performance metrics calculation
"""

from .backtest import Backtest
from .strategy import Strategy, StrategyResult
from .portfolio import Portfolio, Position
from ..data import DataManager
from .risk_manager import RiskManager
from .metrics import MetricsCalculator, BacktestResults

__all__ = [
    "Backtest",
    "Strategy", 
    "StrategyResult",
    "Portfolio",
    "Position",
    "DataManager",
    "RiskManager",
    "MetricsCalculator",
    "BacktestResults",
]