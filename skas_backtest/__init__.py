"""
SKAS Backtesting Framework

A comprehensive backtesting framework for NSE options trading strategies.
Integrates with the NSE Options Pricing library for real-time and historical data.

Main Components:
    Backtest: Core backtesting engine
    Strategy: Base strategy class and implementations
    Portfolio: Portfolio management and position tracking
    DataManager: Historical and live data management
    RiskManager: Risk management and validation
    MetricsCalculator: Performance analysis and reporting

Quick Start:
    >>> from skas_backtest import Backtest, Strategy
    >>> from skas_backtest.strategies import SimpleCallStrategy
    >>> 
    >>> strategy = SimpleCallStrategy()
    >>> backtest = Backtest(strategy=strategy, initial_capital=100000)
    >>> results = backtest.run(start_date="2024-01-01", end_date="2024-06-30")
    >>> print(results.summary())

For more examples and documentation, see the examples/ directory.
"""

__version__ = "0.1.0"
__author__ = "SKAS Trading"
__email__ = "noreply@example.com"
__license__ = "MIT"

# Core exports - importing individually to avoid circular imports
try:
    from .data import DataManager
    from .core.strategy import Strategy, StrategyResult
    from .core.portfolio import Portfolio, Position
    from .core.risk_manager import RiskManager
    from .core.metrics import MetricsCalculator, BacktestResults
    from .core.backtest import Backtest
except ImportError as e:
    # Handle import errors gracefully during development
    print(f"Warning: Some modules not available: {e}")

# Utility functions
try:
    from .utils.helpers import (
        calculate_profit_loss,
        get_option_greeks,
        format_results,
    )
except ImportError:
    pass

__all__ = [
    # Version info
    "__version__",
    
    # Core classes
    "DataManager",
    "Strategy",
    "StrategyResult", 
    "Portfolio",
    "Position",
    "RiskManager",
    "MetricsCalculator",
    "BacktestResults",
    "Backtest",
    
    # Utilities
    "calculate_profit_loss",
    "get_option_greeks",
    "format_results",
]