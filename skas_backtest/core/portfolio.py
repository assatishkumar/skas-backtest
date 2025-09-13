"""
Portfolio and position management for the backtesting framework.

This module handles position tracking, portfolio valuation, profit/loss calculation,
and trade execution within the backtesting environment.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from enum import Enum


class PositionType(Enum):
    """Types of positions that can be held."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """
    Represents a single option position in the portfolio.
    
    Tracks all relevant information about an options position including
    entry details, current market value, and profit/loss calculations.
    """
    symbol: str
    strike: float
    expiry: date
    option_type: str  # "call" or "put"
    position_type: PositionType
    quantity: int  # Number of contracts (positive for long, negative for short)
    entry_price: float
    entry_date: date
    current_price: float = 0.0
    last_updated: Optional[date] = None
    
    # P&L tracking
    unrealized_pnl: float = field(init=False, default=0.0)
    realized_pnl: float = field(init=False, default=0.0)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize calculated fields after creation."""
        if isinstance(self.position_type, str):
            self.position_type = PositionType(self.position_type.lower())
        self.update_pnl()
    
    def update_price(self, new_price: float, as_of_date: date) -> None:
        """
        Update the current market price of the position.
        
        Args:
            new_price: New market price
            as_of_date: Date of the price update
        """
        self.current_price = new_price
        self.last_updated = as_of_date
        self.update_pnl()
    
    def update_pnl(self) -> None:
        """Calculate and update unrealized P&L."""
        price_diff = self.current_price - self.entry_price
        
        if self.position_type == PositionType.LONG:
            self.unrealized_pnl = price_diff * self.quantity
        else:  # SHORT
            self.unrealized_pnl = -price_diff * self.quantity
    
    def get_market_value(self) -> float:
        """Get current market value of the position."""
        return self.current_price * abs(self.quantity)
    
    def get_notional_value(self) -> float:
        """Get notional value of the position at entry."""
        return self.entry_price * abs(self.quantity)
    
    def is_expired(self, current_date: date) -> bool:
        """Check if the position has expired."""
        return current_date >= self.expiry
    
    def get_days_to_expiry(self, current_date: date) -> int:
        """Get number of days until expiry."""
        return (self.expiry - current_date).days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary representation."""
        return {
            "symbol": self.symbol,
            "strike": self.strike,
            "expiry": self.expiry,
            "option_type": self.option_type,
            "position_type": self.position_type.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_date": self.entry_date,
            "current_price": self.current_price,
            "last_updated": self.last_updated,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "market_value": self.get_market_value(),
            "notional_value": self.get_notional_value(),
        }


@dataclass
class TradeResult:
    """Result of a trade execution."""
    success: bool
    position: Optional[Position] = None
    error_message: str = ""
    trade_value: float = 0.0
    commission: float = 0.0
    net_cash_flow: float = 0.0


class Portfolio:
    """
    Portfolio management class for tracking positions and calculating performance.
    
    Handles position tracking, trade execution, profit/loss calculation,
    and portfolio valuation for the backtesting framework.
    
    Example:
        >>> portfolio = Portfolio(initial_capital=100000)
        >>> trade_result = portfolio.execute_trade(
        ...     symbol="RELIANCE", strike=2500, expiry=date(2024, 1, 25),
        ...     option_type="call", action="buy", quantity=1, price=50.0
        ... )
        >>> print(f"Portfolio value: {portfolio.get_total_value()}")
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_trade: float = 20.0,
        commission_rate: float = 0.0005,  # 0.05%
        max_positions: int = 50,
    ):
        """
        Initialize the portfolio.
        
        Args:
            initial_capital: Starting cash amount
            commission_per_trade: Fixed commission per trade
            commission_rate: Commission as percentage of trade value
            max_positions: Maximum number of concurrent positions
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_per_trade = commission_per_trade
        self.commission_rate = commission_rate
        self.max_positions = max_positions
        
        # Position tracking
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        
        # P&L tracking
        self.realized_pnl = 0.0
        self.total_commissions = 0.0
        
        # Trade history
        self.trade_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.daily_values: List[Dict[str, Any]] = []
        self.high_water_mark = initial_capital
        self.max_drawdown = 0.0
    
    def execute_trade(
        self,
        symbol: str,
        strike: float,
        expiry: date,
        option_type: str,
        action: str,
        quantity: int,
        price: float,
        date: date,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TradeResult:
        """
        Execute a trade (buy/sell options).
        
        Args:
            symbol: Stock symbol
            strike: Strike price
            expiry: Expiry date
            option_type: "call" or "put"
            action: "buy" or "sell"
            quantity: Number of contracts
            price: Execution price per contract
            date: Trade date
            metadata: Additional trade metadata
            
        Returns:
            TradeResult indicating success/failure and details
        """
        try:
            # Validate inputs
            if quantity <= 0:
                return TradeResult(False, error_message="Quantity must be positive")
            
            if price <= 0:
                return TradeResult(False, error_message="Price must be positive")
            
            # Check position limits
            action_str = action.value if hasattr(action, 'value') else str(action).lower()
            if action_str == "buy" and len(self.positions) >= self.max_positions:
                return TradeResult(False, error_message="Maximum positions reached")
            
            # Calculate trade value and commission
            trade_value = price * quantity
            commission = self._calculate_commission(trade_value)
            
            # Check if this is opening or closing a position
            existing_position = self._find_matching_position(
                symbol, strike, expiry, option_type
            )
            
            if action_str == "buy":
                return self._execute_buy(
                    symbol, strike, expiry, option_type, quantity,
                    price, date, trade_value, commission, existing_position, metadata
                )
            elif action_str == "sell":
                return self._execute_sell(
                    symbol, strike, expiry, option_type, quantity,
                    price, date, trade_value, commission, existing_position, metadata
                )
            else:
                return TradeResult(False, error_message=f"Invalid action: {action}")
                
        except Exception as e:
            return TradeResult(False, error_message=f"Trade execution error: {str(e)}")
    
    def _execute_buy(
        self,
        symbol: str, strike: float, expiry: date, option_type: str,
        quantity: int, price: float, date: date, trade_value: float,
        commission: float, existing_position: Optional[Position],
        metadata: Optional[Dict[str, Any]],
    ) -> TradeResult:
        """Execute a buy order."""
        total_cost = trade_value + commission
        
        # Check if we have enough cash
        if self.cash < total_cost:
            return TradeResult(False, error_message="Insufficient cash")
        
        if existing_position and existing_position.position_type == PositionType.SHORT:
            # Closing short position
            if quantity >= abs(existing_position.quantity):
                # Fully closing the position
                self._close_position(existing_position, price, date, quantity)
                net_cash_flow = -total_cost
            else:
                # Partially closing the position
                existing_position.quantity += quantity
                existing_position.update_pnl()
                net_cash_flow = -total_cost
        else:
            # Opening new long position or adding to existing long
            if existing_position and existing_position.position_type == PositionType.LONG:
                # Adding to existing long position - average the price
                total_quantity = existing_position.quantity + quantity
                total_cost_basis = (existing_position.entry_price * existing_position.quantity + 
                                  price * quantity)
                existing_position.entry_price = total_cost_basis / total_quantity
                existing_position.quantity = total_quantity
                position = existing_position
            else:
                # New long position
                position = Position(
                    symbol=symbol,
                    strike=strike,
                    expiry=expiry,
                    option_type=option_type,
                    position_type=PositionType.LONG,
                    quantity=quantity,
                    entry_price=price,
                    entry_date=date,
                    current_price=price,
                    last_updated=date,
                    metadata=metadata or {},
                )
                self.positions.append(position)
            
            net_cash_flow = -total_cost
        
        # Update cash and record trade
        self.cash += net_cash_flow
        self.total_commissions += commission
        self._record_trade(symbol, strike, expiry, option_type, "buy", 
                          quantity, price, date, commission, net_cash_flow)
        
        return TradeResult(
            success=True,
            position=existing_position or position,
            trade_value=trade_value,
            commission=commission,
            net_cash_flow=net_cash_flow,
        )
    
    def _execute_sell(
        self,
        symbol: str, strike: float, expiry: date, option_type: str,
        quantity: int, price: float, date: date, trade_value: float,
        commission: float, existing_position: Optional[Position],
        metadata: Optional[Dict[str, Any]],
    ) -> TradeResult:
        """Execute a sell order."""
        net_proceeds = trade_value - commission
        
        if existing_position and existing_position.position_type == PositionType.LONG:
            # Closing long position
            if quantity >= existing_position.quantity:
                # Fully closing the position
                self._close_position(existing_position, price, date, quantity)
            else:
                # Partially closing the position
                existing_position.quantity -= quantity
                existing_position.update_pnl()
            
            net_cash_flow = net_proceeds
        else:
            # Opening new short position or adding to existing short
            if existing_position and existing_position.position_type == PositionType.SHORT:
                # Adding to existing short position
                total_quantity = abs(existing_position.quantity) + quantity
                total_credit = (existing_position.entry_price * abs(existing_position.quantity) + 
                               price * quantity)
                existing_position.entry_price = total_credit / total_quantity
                existing_position.quantity = -total_quantity
                position = existing_position
            else:
                # New short position
                position = Position(
                    symbol=symbol,
                    strike=strike,
                    expiry=expiry,
                    option_type=option_type,
                    position_type=PositionType.SHORT,
                    quantity=-quantity,  # Negative for short
                    entry_price=price,
                    entry_date=date,
                    current_price=price,
                    last_updated=date,
                    metadata=metadata or {},
                )
                self.positions.append(position)
            
            net_cash_flow = net_proceeds
        
        # Update cash and record trade
        self.cash += net_cash_flow
        self.total_commissions += commission
        self._record_trade(symbol, strike, expiry, option_type, "sell",
                          quantity, price, date, commission, net_cash_flow)
        
        return TradeResult(
            success=True,
            position=existing_position or position,
            trade_value=trade_value,
            commission=commission,
            net_cash_flow=net_cash_flow,
        )
    
    def _close_position(self, position: Position, close_price: float, 
                       close_date: date, quantity: int) -> None:
        """Close a position and calculate realized P&L."""
        # Calculate realized P&L
        if position.position_type == PositionType.LONG:
            realized_pnl = (close_price - position.entry_price) * min(quantity, position.quantity)
        else:  # SHORT
            realized_pnl = (position.entry_price - close_price) * min(quantity, abs(position.quantity))
        
        # Update realized P&L
        self.realized_pnl += realized_pnl
        position.realized_pnl += realized_pnl
        
        # Remove position from active positions if fully closed
        if quantity >= abs(position.quantity):
            self.positions.remove(position)
            self.closed_positions.append(position)
        else:
            # Partially closed - reduce quantity
            if position.position_type == PositionType.LONG:
                position.quantity -= quantity
            else:
                position.quantity += quantity  # Adding to negative quantity
    
    def _find_matching_position(
        self, symbol: str, strike: float, expiry: date, option_type: str
    ) -> Optional[Position]:
        """Find existing position that matches the trade parameters."""
        for position in self.positions:
            if (position.symbol == symbol and 
                position.strike == strike and
                position.expiry == expiry and
                position.option_type == option_type):
                return position
        return None
    
    def _calculate_commission(self, trade_value: float) -> float:
        """Calculate commission for a trade."""
        return self.commission_per_trade + (trade_value * self.commission_rate)
    
    def _record_trade(
        self, symbol: str, strike: float, expiry: date, option_type: str,
        action: str, quantity: int, price: float, date: date,
        commission: float, net_cash_flow: float
    ) -> None:
        """Record a trade in the trade history."""
        trade_record = {
            "date": date,
            "symbol": symbol,
            "strike": strike,
            "expiry": expiry,
            "option_type": option_type,
            "action": action,
            "quantity": quantity,
            "price": price,
            "trade_value": price * quantity,
            "commission": commission,
            "net_cash_flow": net_cash_flow,
            "portfolio_value_after": self.get_total_value(),
        }
        self.trade_history.append(trade_record)
    
    def update_values(self, market_data: Dict[str, Any], current_date: date) -> None:
        """
        Update all position values with current market data.
        
        Args:
            market_data: Dictionary containing current market prices
            current_date: Current date for valuation
        """
        for position in self.positions:
            # Get current market price for this position
            option_key = f"{position.symbol}_{position.strike}_{position.expiry}_{position.option_type}"
            
            if option_key in market_data:
                current_price = market_data[option_key]
                position.update_price(current_price, current_date)
        
        # Check for expired positions
        self._handle_expired_positions(current_date)
        
        # Update performance metrics
        self._update_performance_metrics(current_date)
    
    def _handle_expired_positions(self, current_date: date) -> None:
        """Handle positions that have expired."""
        expired_positions = [p for p in self.positions if p.is_expired(current_date)]
        
        for position in expired_positions:
            # Options expire worthless
            if position.position_type == PositionType.LONG:
                # Long positions lose their value
                realized_pnl = -position.entry_price * position.quantity
            else:
                # Short positions keep the premium
                realized_pnl = position.entry_price * abs(position.quantity)
            
            self.realized_pnl += realized_pnl
            position.realized_pnl += realized_pnl
            
            # Move to closed positions
            self.positions.remove(position)
            self.closed_positions.append(position)
    
    def _update_performance_metrics(self, current_date: date) -> None:
        """Update performance metrics like drawdown."""
        current_value = self.get_total_value()
        
        # Update high water mark
        if current_value > self.high_water_mark:
            self.high_water_mark = current_value
        
        # Calculate current drawdown
        current_drawdown = (self.high_water_mark - current_value) / self.high_water_mark
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def get_total_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        positions_value = sum(pos.get_market_value() 
                             if pos.position_type == PositionType.LONG 
                             else -pos.get_market_value() 
                             for pos in self.positions)
        return self.cash + positions_value
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions."""
        return sum(pos.unrealized_pnl for pos in self.positions)
    
    def get_positions_summary(self) -> pd.DataFrame:
        """Get summary of all current positions."""
        if not self.positions:
            return pd.DataFrame()
        
        position_data = [pos.to_dict() for pos in self.positions]
        return pd.DataFrame(position_data)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get portfolio performance summary."""
        total_value = self.get_total_value()
        total_return = (total_value - self.initial_capital) / self.initial_capital
        
        return {
            "initial_capital": self.initial_capital,
            "current_value": total_value,
            "cash": self.cash,
            "positions_value": total_value - self.cash,
            "total_return": total_return,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.get_unrealized_pnl(),
            "total_commissions": self.total_commissions,
            "num_positions": len(self.positions),
            "num_trades": len(self.trade_history),
            "max_drawdown": self.max_drawdown,
            "high_water_mark": self.high_water_mark,
        }