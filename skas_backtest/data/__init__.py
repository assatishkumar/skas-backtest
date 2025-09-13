"""
Data management package for NSE backtesting.

This package handles all data fetching, caching, and processing for the backtesting framework.
It integrates with the NSE options pricing library to provide historical and real-time data.

Main Components:
    DataManager: Primary interface for all data operations
    MarketData: Data structures for market information
    OptionQuote: Data structures for option pricing

Example:
    >>> from skas_backtest.data import DataManager
    >>> dm = DataManager()
    >>> price = dm.get_stock_price("RELIANCE", date(2024, 1, 15))
"""

from .data_manager import DataManager, MarketData, OptionQuote

__all__ = [
    "DataManager",
    "MarketData", 
    "OptionQuote",
]