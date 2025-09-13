"""
Core backtesting engine for NSE options strategies.

This module provides the main Backtest class that orchestrates the backtesting process,
integrating with the NSE options pricing library for real-time and historical data.
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .strategy import Strategy, StrategyResult
from .portfolio import Portfolio
from ..data import DataManager
from .risk_manager import RiskManager
from .metrics import MetricsCalculator, BacktestResults


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    initial_capital: float = 100000.0
    max_positions: int = 10
    commission_per_trade: float = 20.0
    commission_rate: float = 0.0005  # 0.05%
    slippage_bps: int = 2  # 2 basis points
    margin_multiplier: float = 1.0
    allow_fractional_lots: bool = False
    risk_free_rate: float = 0.06  # 6% annual
    enable_risk_checks: bool = True


class Backtest:
    """
    Main backtesting engine for NSE options strategies.
    
    This class orchestrates the entire backtesting process, from data fetching
    to strategy execution to performance analysis.
    
    Example:
        >>> from skas_backtest import Backtest
        >>> from skas_backtest.strategies import SimpleCallStrategy
        >>> 
        >>> strategy = SimpleCallStrategy()
        >>> backtest = Backtest(strategy=strategy, initial_capital=100000)
        >>> results = backtest.run(start_date="2024-01-01", end_date="2024-06-30")
        >>> print(results.summary())
    """
    
    def __init__(
        self,
        strategy: Strategy,
        config: Optional[BacktestConfig] = None,
        data_manager: Optional[DataManager] = None,
        risk_manager: Optional[RiskManager] = None,
        **kwargs
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            strategy: Trading strategy to backtest
            config: Backtest configuration
            data_manager: Custom data manager (optional)
            risk_manager: Custom risk manager (optional)
            **kwargs: Additional configuration parameters
        """
        self.strategy = strategy
        self.config = config or BacktestConfig(**kwargs)
        
        # Initialize core components
        self.portfolio = Portfolio(
            initial_capital=self.config.initial_capital,
            commission_per_trade=self.config.commission_per_trade,
            commission_rate=self.config.commission_rate,
        )
        
        self.data_manager = data_manager or DataManager()
        self.risk_manager = risk_manager or RiskManager(self.config)
        self.metrics_calculator = MetricsCalculator()
        
        # Tracking variables
        self.current_date: Optional[date] = None
        self.trade_log: List[Dict] = []
        self.daily_portfolio_values: List[Dict] = []
        self.is_running = False
        
    def run(
        self,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        symbols: Optional[List[str]] = None,
        rebalance_frequency: str = "daily",
        progress_callback: Optional[callable] = None,
    ) -> BacktestResults:
        """
        Run the backtest over the specified date range.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest  
            symbols: List of symbols to trade (if None, strategy decides)
            rebalance_frequency: How often to rebalance ("daily", "weekly", "monthly")
            progress_callback: Optional callback for progress updates
            
        Returns:
            BacktestResults object with performance metrics and data
        """
        # Convert dates to datetime.date objects
        start_date = self._parse_date(start_date)
        end_date = self._parse_date(end_date)
        
        self.is_running = True
        
        try:
            # Get trading dates
            trading_dates = self.data_manager.get_trading_dates(start_date, end_date)
            
            # Initialize strategy
            self.strategy.initialize(self.data_manager, self.portfolio, self.config)
            
            # Main backtesting loop
            for i, current_date in enumerate(trading_dates):
                self.current_date = current_date
                
                # Progress callback
                if progress_callback:
                    progress = (i + 1) / len(trading_dates)
                    progress_callback(progress, current_date)
                
                # Update portfolio with current market data
                self._update_portfolio_values(current_date)
                
                # Generate strategy signals
                strategy_result = self.strategy.generate_signals(
                    current_date, symbols or self.strategy.get_symbols()
                )
                
                # Process signals through risk management
                if self.config.enable_risk_checks:
                    strategy_result = self.risk_manager.validate_signals(
                        strategy_result, self.portfolio, current_date
                    )
                
                # Execute trades
                self._execute_trades(strategy_result, current_date)
                
                # Record daily portfolio state
                self._record_daily_state(current_date)
                
                # Strategy end-of-day processing
                self.strategy.end_of_day(current_date, self.portfolio)
            
            # Calculate final results
            results = self._calculate_results(start_date, end_date)
            
        finally:
            self.is_running = False
            
        return results
    
    def _update_portfolio_values(self, current_date: date) -> None:
        """Update portfolio positions with current market values."""
        option_prices = self.data_manager.get_portfolio_option_prices(
            self.portfolio.positions, current_date
        )
        self.portfolio.update_values(option_prices, current_date)
    
    def _execute_trades(self, strategy_result: StrategyResult, current_date: date) -> None:
        """Execute trades from strategy signals."""
        for signal in strategy_result.signals:
            try:
                # Get current market price
                market_price = self.data_manager.get_option_price(
                    symbol=signal.symbol,
                    strike=signal.strike,
                    expiry=signal.expiry,
                    option_type=signal.option_type,
                    as_of_date=current_date,
                )
                
                # Apply slippage
                execution_price = self._apply_slippage(market_price, signal.action)
                
                # Get lot size for accurate quantity calculation
                lot_size = self.data_manager.get_lot_size(signal.symbol, current_date)
                actual_quantity = signal.quantity * lot_size  # Convert contracts to shares
                
                # Execute the trade
                trade_result = self.portfolio.execute_trade(
                    symbol=signal.symbol,
                    strike=signal.strike,
                    expiry=signal.expiry,
                    option_type=signal.option_type,
                    action=signal.action,
                    quantity=actual_quantity,
                    price=execution_price,
                    date=current_date,
                )
                
                # Enhanced trade execution logging
                if trade_result and trade_result.success:
                    # Get action and option type as strings
                    action_str = signal.action.value if hasattr(signal.action, 'value') else str(signal.action)
                    option_str = signal.option_type.value if hasattr(signal.option_type, 'value') else str(signal.option_type)
                    
                    # Calculate trade value and commission using actual quantity
                    trade_value = execution_price * actual_quantity
                    slippage = execution_price - market_price
                    slippage_pct = (slippage / market_price * 100) if market_price > 0 else 0
                    
                    # Check if this is a BUY order closing a position and show P&L
                    leg_pnl_info = ""
                    if action_str.upper() == "BUY" and hasattr(signal, 'metadata') and signal.metadata:
                        if signal.metadata.get('position_type') == 'exit':
                            # This is closing a position, get entry price from signal reason or metadata
                            # We can extract entry price from the portfolio or calculate from current position
                            existing_pos = None
                            # Try to find existing position to get entry price
                            for pos in self.portfolio.positions:
                                if (pos.symbol == signal.symbol and pos.strike == signal.strike and 
                                    pos.expiry == signal.expiry and pos.option_type == signal.option_type.value):
                                    existing_pos = pos
                                    break
                            
                            if existing_pos and existing_pos.entry_price:
                                # Calculate P&L for this leg
                                entry_price = existing_pos.entry_price
                                leg_pnl = (entry_price - execution_price) * actual_quantity  # For short positions
                                leg_pnl_info = f" | Leg P&L: ₹{leg_pnl:,.0f} (Entry: ₹{entry_price:.2f} → Exit: ₹{execution_price:.2f})"
                    
                    # Log detailed trade execution
                    print(f"✅ {action_str.upper()} {signal.symbol} {option_str.upper()}@{signal.strike} x{signal.quantity} @ ₹{execution_price:.2f}{leg_pnl_info}")
                    if abs(slippage) > 0.01:  # Only show slippage if significant
                        print(f"   Slippage: ₹{slippage:+.2f} ({slippage_pct:+.2f}%) | Value: ₹{trade_value:,.0f} | Commission: ₹{trade_result.commission:.0f}")
                    
                    # Log portfolio impact
                    portfolio_value_after = self.portfolio.get_total_value()
                    print(f"   Portfolio Value: ₹{portfolio_value_after:,.0f} | Cash: ₹{self.portfolio.cash:,.0f}")
                    
                    self.trade_log.append({
                        "date": current_date,
                        "symbol": signal.symbol,
                        "strike": signal.strike,
                        "expiry": signal.expiry,
                        "option_type": signal.option_type,
                        "action": signal.action,
                        "quantity": signal.quantity,  # Contract count for strategy compatibility
                        "actual_quantity": actual_quantity,  # Actual shares traded
                        "lot_size": lot_size,  # For reference
                        "price": execution_price,
                        "market_price": market_price,
                        "slippage": slippage,
                        "strategy_reason": signal.reason,
                    })
                elif trade_result and not trade_result.success:
                    # Log failed trades
                    action_str = signal.action.value if hasattr(signal.action, 'value') else str(signal.action)
                    option_str = signal.option_type.value if hasattr(signal.option_type, 'value') else str(signal.option_type)
                    print(f"❌ FAILED: {action_str.upper()} {signal.symbol} {option_str.upper()}@{signal.strike} - {trade_result.error_message}")
                    
            except Exception as e:
                print(f"Error executing trade for {signal.symbol}: {e}")
                continue
    
    def _apply_slippage(self, market_price: float, action: str) -> float:
        """Apply slippage to market price based on trade direction."""
        slippage_factor = self.config.slippage_bps / 10000.0
        
        action_str = action.value if hasattr(action, 'value') else str(action).lower()
        
        if action_str.lower() == "buy":
            return market_price * (1 + slippage_factor)
        else:  # sell
            return market_price * (1 - slippage_factor)
    
    def _record_daily_state(self, current_date: date) -> None:
        """Record daily portfolio state for performance tracking."""
        portfolio_value = self.portfolio.get_total_value()
        cash = self.portfolio.cash
        positions_value = portfolio_value - cash
        
        self.daily_portfolio_values.append({
            "date": current_date,
            "total_value": portfolio_value,
            "cash": cash,
            "positions_value": positions_value,
            "num_positions": len(self.portfolio.positions),
            "unrealized_pnl": self.portfolio.get_unrealized_pnl(),
            "realized_pnl": self.portfolio.realized_pnl,
        })
    
    def _calculate_results(self, start_date: date, end_date: date) -> BacktestResults:
        """Calculate final backtest results and performance metrics."""
        # Convert daily values to DataFrame
        daily_values_df = pd.DataFrame(self.daily_portfolio_values)
        daily_values_df.set_index("date", inplace=True)
        
        # Convert trade log to DataFrame
        trades_df = pd.DataFrame(self.trade_log)
        
        # Generate API performance summary if performance monitoring is enabled
        performance_summary = None
        if hasattr(self.data_manager, 'perf_monitor') and self.data_manager.perf_monitor:
            performance_summary = self.data_manager.perf_monitor.get_summary()
            print("\n" + "="*80)
            print("API PERFORMANCE ANALYSIS")
            print("="*80)
            print(performance_summary)
            print("="*80)
        
        # Calculate performance metrics
        metrics = self.metrics_calculator.calculate_metrics(
            daily_values_df,
            trades_df,
            self.config.initial_capital,
            self.config.risk_free_rate,
        )
        
        # Add performance summary to metrics if available
        if performance_summary:
            metrics['api_performance'] = performance_summary
        
        return BacktestResults(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.config.initial_capital,
            final_value=self.portfolio.get_total_value(),
            total_return=metrics["total_return"],
            annualized_return=metrics["annualized_return"],
            volatility=metrics["volatility"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
            profit_factor=metrics["profit_factor"],
            daily_values=daily_values_df,
            trades=trades_df,
            portfolio=self.portfolio,
            metrics=metrics,
        )
    
    def _parse_date(self, date_input: Union[str, date, datetime]) -> date:
        """Parse various date input formats to datetime.date."""
        if isinstance(date_input, str):
            return datetime.strptime(date_input, "%Y-%m-%d").date()
        elif isinstance(date_input, datetime):
            return date_input.date()
        elif isinstance(date_input, date):
            return date_input
        else:
            raise ValueError(f"Invalid date format: {date_input}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current backtest state (useful for real-time monitoring)."""
        if not self.is_running:
            return {"status": "not_running"}
            
        return {
            "status": "running",
            "current_date": self.current_date,
            "portfolio_value": self.portfolio.get_total_value(),
            "cash": self.portfolio.cash,
            "num_positions": len(self.portfolio.positions),
            "unrealized_pnl": self.portfolio.get_unrealized_pnl(),
            "realized_pnl": self.portfolio.realized_pnl,
            "num_trades": len(self.trade_log),
        }