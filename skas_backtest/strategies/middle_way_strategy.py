"""
Middle Way Strategy Implementation

A short strangle strategy with trend-following adjustments. This strategy:
1. Enters short strangles based on previous month's range at the start of each month
2. Monitors for breakouts and adjusts positions accordingly
3. Includes NIFTY hedge to manage portfolio risk
4. Uses strict risk management with stop-loss and profit targets

Starting with single stock implementation for validation before scaling to full portfolio.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import logging

from ..core.strategy import Strategy, TradingSignal, StrategyResult, SignalAction, OptionType


class PositionType(Enum):
    """Types of positions in Middle Way strategy."""
    STRANGLE = "strangle"        # Initial short strangle
    BULLISH = "bullish"         # Short put only (after upside breakout)
    BEARISH = "bearish"         # Short call only (after downside breakout)
    HEDGE = "hedge"             # Long NIFTY strangle


class BreakoutDirection(Enum):
    """Direction of breakout from initial range."""
    UPSIDE = "upside"
    DOWNSIDE = "downside"
    NONE = "none"


@dataclass
class MiddleWayConfig:
    """Configuration parameters for Middle Way strategy."""
    
    # Single stock for testing (will scale to full list later)
    test_stock: str = "RELIANCE"
    
    # Full stock list for future scaling
    stock_list: List[str] = field(default_factory=lambda: [
        "HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "BHARTIARTL",
        "LT", "ITC", "TCS", "SBIN", "AXISBANK", "KOTAKBANK", "M&M",
        "HINDUNILVR", "BAJFINANCE", "ZOMATO", "MARUTI", "SUNPHARMA",
        "NTPC", "HCLTECH", "ULTRACEMCO", "TITAN", "TATAMOTORS",
        "BEL", "TATASTEEL", "POWERGRID"
    ])
    
    # Capital allocation
    total_capital: float = 5000000  # 50 lakhs
    capital_per_stock: float = 200000  # 2 lakhs per stock
    
    # Risk parameters
    portfolio_stop_loss_pct: float = 0.02  # 2% of total capital
    position_profit_target_pct: float = 0.80  # 80% of premium collected
    
    # Hedge parameters
    hedge_symbol: str = "NIFTY"
    hedge_otm_pct: float = 0.04  # 4% OTM for NIFTY hedge
    
    # Adjustment constraints
    max_adjustments_per_stock: int = 2
    min_days_between_adjustments: int = 1
    
    # Exit timing
    exit_days_before_expiry: int = 1  # Exit T-1 day before expiry
    
    # Leg-level profit booking
    enable_leg_profit_booking: bool = True
    leg_profit_target_pct: float = 0.80  # Book profit when 80% of max profit captured
    
    # Adjustment trigger buffer
    adjustment_trigger_buffer_pct: float = 0.00  # 0% = at strike, 0.01 = 1% buffer
    
    # Premium threshold configuration
    min_leg_premium_pct: float = 0.05  # Skip leg if premium < 5% of total
    skip_low_premium_legs: bool = True  # Enable/disable skipping low premium legs
    
    # Position sizing configuration
    lots_per_stock_position: int = 1  # Number of lots per stock position
    lots_per_hedge_position: int = 1  # Number of lots for NIFTY hedge
    
    # Margin configuration
    margin_percentage: float = 0.25  # 25% of exposure (notional value)


@dataclass
class PositionState:
    """Tracks the state of a position in the Middle Way strategy."""
    
    symbol: str
    entry_date: date
    expiry: date
    position_type: PositionType
    
    # Strike and premium information
    call_strike: Optional[float] = None
    put_strike: Optional[float] = None
    call_premium: Optional[float] = None
    put_premium: Optional[float] = None
    
    # Position tracking
    call_quantity: int = 0  # Negative for short positions
    put_quantity: int = 0   # Negative for short positions
    lot_size: int = 1  # Cached lot size for the symbol (fetched once at entry)
    
    # Strategy state
    adjustment_count: int = 0
    last_adjustment_date: Optional[date] = None
    entry_spot_price: float = 0.0
    
    # Current market prices (updated daily)
    current_spot_price: float = 0.0
    current_call_premium: Optional[float] = None
    current_put_premium: Optional[float] = None
    
    # Financial tracking
    margin_deployed: float = 0.0
    exposure: float = 0.0  # Total notional value (lot_size × quantity × spot_price)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_commissions: float = 0.0  # Track commission costs for this position
    
    # P&L tracking by leg
    call_pnl: float = 0.0  # Final P&L for call leg
    put_pnl: float = 0.0   # Final P&L for put leg
    final_pnl: float = 0.0 # Total final P&L including commissions
    
    # Daily P&L tracking
    daily_pnl_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Original range for breakout detection
    prev_month_high: Optional[float] = None
    prev_month_low: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class MiddleWayStrategy(Strategy):
    """
    Middle Way Strategy Implementation.
    
    This strategy implements a systematic approach to options trading:
    1. Monthly entry based on previous month's high/low
    2. Short strangle at range extremes
    3. Breakout-based adjustments with limits
    4. Strict risk management and profit targets
    """
    
    def __init__(self, config: MiddleWayConfig = None, **kwargs):
        """
        Initialize Middle Way Strategy.
        
        Args:
            config: Strategy configuration object
            **kwargs: Additional parameters
        """
        super().__init__(name="MiddleWayStrategy", **kwargs)
        
        self.strategy_config = config or MiddleWayConfig()
        self.logger = logging.getLogger(__name__)
        
        # Position tracking
        self.positions: Dict[str, PositionState] = {}
        self.hedge_position: Optional[PositionState] = None
        
        # Strategy state
        self.current_month = None
        self.entry_this_month = False
        
        # Performance tracking
        self.monthly_pnl = {}
        self.total_trades = 0
        self.winning_trades = 0
        
        # Enhanced monthly tracking
        self.monthly_stats = {}  # Detailed monthly statistics
        self.completed_positions = []  # Store completed positions for analysis
    
    def get_symbols(self) -> List[str]:
        """Return symbols this strategy trades."""
        # For now, return single test stock
        # Later will return self.strategy_config.stock_list
        return [self.strategy_config.test_stock]
    
    def on_initialize(self) -> None:
        """Strategy-specific initialization."""
        self.logger.info(f"Initializing Middle Way Strategy")
        self.logger.info(f"Trading symbol: {self.strategy_config.test_stock}")
        self.logger.info(f"Capital per stock: ₹{self.strategy_config.capital_per_stock}")
    
    def generate_signals(self, current_date: date, symbols: List[str]) -> StrategyResult:
        """
        Main signal generation logic for Middle Way strategy.
        
        Args:
            current_date: Current trading date
            symbols: List of symbols to analyze
            
        Returns:
            StrategyResult with trading signals
        """
        signals = []
        analytics = {}
        
        try:
            # Check if we're in a new month
            current_month_key = (current_date.year, current_date.month)
            if self.current_month != current_month_key:
                self._on_new_month(current_date)
                self.current_month = current_month_key
                self.entry_this_month = False
            
            # Generate signals for each symbol
            for symbol in symbols:
                symbol_signals = self._generate_symbol_signals(symbol, current_date)
                signals.extend(symbol_signals)
                
                # Add analytics for this symbol
                analytics[symbol] = self._get_symbol_analytics(symbol, current_date)
            
            # Check hedge signals
            hedge_signals = self._generate_hedge_signals(current_date)
            signals.extend(hedge_signals)
            
            # Update position analytics
            analytics['portfolio'] = self._get_portfolio_analytics(current_date)
            
            return StrategyResult(
                signals=signals,
                analytics=analytics,
                metadata={'strategy': 'MiddleWay', 'date': current_date}
            )
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {current_date}: {e}")
            return StrategyResult(signals=[], analytics={'error': str(e)})
    
    def _on_new_month(self, current_date: date) -> None:
        """Handle new month initialization."""
        self.logger.info(f"New month detected: {current_date.strftime('%B %Y')}")
        
        # Log monthly summary for previous month
        if self.current_month:
            self._log_monthly_summary(current_date)
        
        # Reset monthly tracking
        self.entry_this_month = False
    
    def _generate_symbol_signals(self, symbol: str, current_date: date) -> List[TradingSignal]:
        """Generate signals for a specific symbol."""
        signals = []
        
        try:
            # Check if we should enter new positions
            if not self.entry_this_month and self._should_enter_new_position(symbol, current_date):
                entry_signals = self._generate_entry_signals(symbol, current_date)
                signals.extend(entry_signals)
                self.entry_this_month = True
            
            # Update existing positions with current market prices
            if symbol in self.positions:
                position = self.positions[symbol]
                self._update_position_mtm(position, current_date)
                
                # Check for adjustments or exits
                position_signals = self._check_existing_position(symbol, current_date)
                signals.extend(position_signals)
        
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
        
        return signals
    
    def _should_enter_new_position(self, symbol: str, current_date: date) -> bool:
        """Check if we should enter a new position."""
        
        # Don't enter if we already have a position
        if symbol in self.positions:
            return False
        
        # Only enter on first few trading days of the month
        month_start = date(current_date.year, current_date.month, 1)
        trading_dates = self.data_manager.get_trading_dates(month_start, current_date)
        
        if len(trading_dates) > 3:  # Only enter in first 3 trading days
            return False
        
        # Check if we have sufficient data for previous month
        prev_high, prev_low = self.data_manager.get_previous_month_range(symbol, current_date)
        if not prev_high or not prev_low:
            self.logger.warning(f"No previous month data for {symbol}")
            return False
        
        return True
    
    def _generate_entry_signals(self, symbol: str, current_date: date) -> List[TradingSignal]:
        """Generate entry signals for new strangle position."""
        signals = []
        
        try:
            # Get previous month's high and low
            prev_high, prev_low = self.data_manager.get_previous_month_range(symbol, current_date)
            
            # Get current month's expiry
            expiry = self.data_manager.get_monthly_expiry(current_date.year, current_date.month)
            if not expiry:
                self.logger.error(f"Cannot get expiry for {current_date}")
                return signals
            
            # Get current spot price
            spot_price = self.data_manager.get_stock_price(symbol, current_date)
            if not spot_price:
                self.logger.error(f"Cannot get spot price for {symbol}")
                return signals
            
            # Find closest strikes to previous month's high and low
            call_strike = self.data_manager.find_closest_strike(symbol, prev_high, expiry, current_date)
            put_strike = self.data_manager.find_closest_strike(symbol, prev_low, expiry, current_date)
            
            if not call_strike or not put_strike:
                self.logger.error(f"Cannot find strikes for {symbol}")
                return signals
            
            # Get option prices
            call_price = self.data_manager.get_option_price(symbol, call_strike, expiry, "call", current_date)
            put_price = self.data_manager.get_option_price(symbol, put_strike, expiry, "put", current_date)
            
            if not call_price or not put_price:
                self.logger.warning(f"Cannot get option prices for {symbol}, using estimated prices")
                # Use rough estimation if option prices not available
                call_price = call_price or max(10, (spot_price - call_strike) * 0.1)
                put_price = put_price or max(10, (put_strike - spot_price) * 0.1)
            
            # Calculate position size
            lot_size = self.data_manager.get_lot_size(symbol, current_date)
            total_premium = call_price + put_price
            # Use configured lot size instead of capital-based calculation
            quantity = self.strategy_config.lots_per_stock_position
            
            # Check premium thresholds if enabled
            final_call_price = call_price
            final_put_price = put_price
            skip_call = False
            skip_put = False
            
            if self.strategy_config.skip_low_premium_legs:
                min_premium_threshold = total_premium * self.strategy_config.min_leg_premium_pct
                
                if call_price < min_premium_threshold:
                    self.logger.info(f"Skipping call leg: premium ₹{call_price:.2f} < threshold ₹{min_premium_threshold:.2f}")
                    skip_call = True
                    final_call_price = None
                    call_strike = None
                
                if put_price < min_premium_threshold:
                    self.logger.info(f"Skipping put leg: premium ₹{put_price:.2f} < threshold ₹{min_premium_threshold:.2f}")
                    skip_put = True
                    final_put_price = None
                    put_strike = None
                
                # If both legs are skipped, don't enter
                if skip_call and skip_put:
                    self.logger.warning(f"Both legs below premium threshold, skipping entry for {symbol}")
                    return signals
                
                # Recalculate total premium for actual legs
                total_premium = (final_call_price or 0) + (final_put_price or 0)
            
            # Calculate proper margin as 25% of exposure and premium collected
            exposure = lot_size * quantity * spot_price
            margin_deployed = exposure * self.strategy_config.margin_percentage
            premium_collected = total_premium * lot_size * quantity
            
            # Create position state
            position = PositionState(
                symbol=symbol,
                entry_date=current_date,
                expiry=expiry,
                position_type=PositionType.STRANGLE,
                call_strike=call_strike,
                put_strike=put_strike,
                call_premium=final_call_price,
                put_premium=final_put_price,
                call_quantity=-quantity if not skip_call else 0,  # Short position or 0 if skipped
                put_quantity=-quantity if not skip_put else 0,   # Short position or 0 if skipped
                lot_size=lot_size,  # Cache lot size for reuse throughout position lifetime
                entry_spot_price=spot_price,
                current_spot_price=spot_price,  # Initialize current prices
                current_call_premium=final_call_price,
                current_put_premium=final_put_price,
                margin_deployed=margin_deployed,
                exposure=exposure,  # Store exposure for ROI calculations
                prev_month_high=prev_high,
                prev_month_low=prev_low
            )
            
            # Store premium collected for profit target calculation
            position.metadata['premium_collected'] = premium_collected
            
            self.positions[symbol] = position
            
            # Generate sell signals for non-skipped legs only
            if not skip_call and call_strike:
                call_signal = TradingSignal(
                    symbol=symbol,
                    strike=call_strike,
                    expiry=expiry,
                    option_type=OptionType.CALL,
                    action=SignalAction.SELL,
                    quantity=quantity,
                    reason=f"Middle Way entry: Sell call at prev month high ₹{prev_high}",
                    metadata={'position_type': 'strangle_entry', 'leg': 'call'}
                )
                signals.append(call_signal)
            
            if not skip_put and put_strike:
                put_signal = TradingSignal(
                    symbol=symbol,
                    strike=put_strike,
                    expiry=expiry,
                    option_type=OptionType.PUT,
                    action=SignalAction.SELL,
                    quantity=quantity,
                    reason=f"Middle Way entry: Sell put at prev month low ₹{prev_low}",
                    metadata={'position_type': 'strangle_entry', 'leg': 'put'}
                )
                signals.append(put_signal)
            
            # Enhanced entry logging
            total_call_premium = (final_call_price * quantity * lot_size) if not skip_call else 0
            total_put_premium = (final_put_price * quantity * lot_size) if not skip_put else 0
            total_premium_collected = total_call_premium + total_put_premium
            
            self.logger.info(f"📈 ENTRY: {symbol} Strangle | Spot: ₹{spot_price:.2f} | Prev Range: ₹{prev_low:.0f}-₹{prev_high:.0f} | Lot Size: {lot_size}")
            
            if not skip_call and call_strike:
                self.logger.info(f"  ├─ SELL Call@{call_strike} x{quantity} @ ₹{final_call_price:.2f} (Premium: ₹{total_call_premium:,.0f})")
            else:
                self.logger.info(f"  ├─ SKIPPED Call@{call_strike or 'N/A'} - Premium too low (₹{call_price:.2f})")
            
            if not skip_put and put_strike:
                self.logger.info(f"  ├─ SELL Put@{put_strike} x{quantity} @ ₹{final_put_price:.2f} (Premium: ₹{total_put_premium:,.0f})")
            else:
                self.logger.info(f"  ├─ SKIPPED Put@{put_strike or 'N/A'} - Premium too low (₹{put_price:.2f})")
            
            self.logger.info(f"  └─ Premium: ₹{total_premium_collected:,.0f} | Margin: ₹{position.margin_deployed:,.0f} | Exposure: ₹{exposure:,.0f} | Expiry: {expiry}")
            
        except Exception as e:
            self.logger.error(f"Error generating entry signals for {symbol}: {e}")
        
        return signals
    
    def _check_existing_position(self, symbol: str, current_date: date) -> List[TradingSignal]:
        """Check existing position for adjustments or exits."""
        signals = []
        position = self.positions[symbol]
        
        try:
            # Check exit conditions first
            exit_signals = self._check_exit_conditions(position, current_date)
            if exit_signals:
                signals.extend(exit_signals)
                return signals  # Exit takes priority
            
            # Check for breakouts and adjustments
            breakout_signals = self._check_breakout_adjustments(position, current_date)
            signals.extend(breakout_signals)
            
        except Exception as e:
            self.logger.error(f"Error checking position for {symbol}: {e}")
        
        return signals
    
    def _check_exit_conditions(self, position: PositionState, current_date: date) -> List[TradingSignal]:
        """Check if position should be exited."""
        signals = []
        
        try:
            # Check expiry exit (T-1) - highest priority
            days_to_expiry = (position.expiry - current_date).days
            if days_to_expiry <= self.strategy_config.exit_days_before_expiry:
                exit_signals = self._generate_exit_signals(position, current_date, "Expiry exit T-1")
                signals.extend(exit_signals)
                return signals
            
            # Check leg-level profit targets before position-level checks
            leg_exit_signals = self._check_leg_profit_targets(position, current_date)
            if leg_exit_signals:
                signals.extend(leg_exit_signals)
                # Don't return immediately - position may still exist with remaining legs
            
            # Check position-level profit target only if we still have at least one leg
            if (position.call_quantity != 0 or position.put_quantity != 0):
                current_pnl = self._calculate_position_pnl(position, current_date)
                # Calculate profit target as percentage of premium collected
                premium_collected = position.metadata.get('premium_collected', position.margin_deployed)
                profit_target = premium_collected * self.strategy_config.position_profit_target_pct
                
                if current_pnl >= profit_target:
                    exit_signals = self._generate_exit_signals(position, current_date, 
                                                             f"Position profit target hit: ₹{current_pnl:,.0f} (Target: ₹{profit_target:,.0f})")
                    signals.extend(exit_signals)
                    return signals
            
            # Check portfolio stop loss
            total_portfolio_pnl = self._calculate_total_portfolio_pnl(current_date)
            stop_loss_limit = self.strategy_config.total_capital * self.strategy_config.portfolio_stop_loss_pct
            
            if total_portfolio_pnl <= -stop_loss_limit:
                exit_signals = self._generate_exit_signals(position, current_date, 
                                                         f"Portfolio stop loss: ₹{total_portfolio_pnl:,.0f}")
                signals.extend(exit_signals)
                return signals
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
        
        return signals
    
    def _check_leg_profit_targets(self, position: PositionState, current_date: date) -> List[TradingSignal]:
        """Check if individual legs have hit profit targets."""
        signals = []
        
        if not self.strategy_config.enable_leg_profit_booking:
            return signals
        
        try:
            # Check CALL leg profit
            if position.call_quantity != 0 and position.call_strike and position.call_premium:
                call_pnl, call_profit_pct = self._calculate_leg_pnl(position, "call", current_date)
                
                if call_profit_pct >= self.strategy_config.leg_profit_target_pct:
                    # Generate buy signal to close only call leg
                    call_exit = TradingSignal(
                        symbol=position.symbol,
                        strike=position.call_strike,
                        expiry=position.expiry,
                        option_type=OptionType.CALL,
                        action=SignalAction.BUY,
                        quantity=abs(position.call_quantity),
                        reason=f"Call leg profit {call_profit_pct:.0%} hit target {self.strategy_config.leg_profit_target_pct:.0%}",
                        metadata={'exit_type': 'leg_profit_booking', 'leg': 'call', 'profit_pct': call_profit_pct}
                    )
                    signals.append(call_exit)
                    
                    # Update position to remove call leg and store P&L
                    position.call_pnl = call_pnl  # Store the P&L for this leg
                    position.call_quantity = 0
                    position.call_premium = None
                    position.call_strike = None
                    position.current_call_premium = None
                    
                    self.logger.info(f"Call leg profit booking: {position.symbol} Call@{call_exit.strike}, "
                                   f"Profit: ₹{call_pnl:,.0f} ({call_profit_pct:.0%})")
            
            # Check PUT leg profit
            if position.put_quantity != 0 and position.put_strike and position.put_premium:
                put_pnl, put_profit_pct = self._calculate_leg_pnl(position, "put", current_date)
                
                if put_profit_pct >= self.strategy_config.leg_profit_target_pct:
                    # Generate buy signal to close only put leg
                    put_exit = TradingSignal(
                        symbol=position.symbol,
                        strike=position.put_strike,
                        expiry=position.expiry,
                        option_type=OptionType.PUT,
                        action=SignalAction.BUY,
                        quantity=abs(position.put_quantity),
                        reason=f"Put leg profit {put_profit_pct:.0%} hit target {self.strategy_config.leg_profit_target_pct:.0%}",
                        metadata={'exit_type': 'leg_profit_booking', 'leg': 'put', 'profit_pct': put_profit_pct}
                    )
                    signals.append(put_exit)
                    
                    # Update position to remove put leg and store P&L
                    position.put_pnl = put_pnl  # Store the P&L for this leg
                    position.put_quantity = 0
                    position.put_premium = None
                    position.put_strike = None
                    position.current_put_premium = None
                    
                    self.logger.info(f"Put leg profit booking: {position.symbol} Put@{put_exit.strike}, "
                                   f"Profit: ₹{put_pnl:,.0f} ({put_profit_pct:.0%})")
            
            # If both legs are closed, remove position from tracking
            if position.call_quantity == 0 and position.put_quantity == 0:
                if position.symbol in self.positions:
                    # Calculate final P&L by summing stored leg P&L
                    final_pnl = position.call_pnl + position.put_pnl
                    position.final_pnl = final_pnl
                    position.realized_pnl = final_pnl
                    
                    # Add to completed positions and log summary
                    self.completed_positions.append(position)
                    self._log_position_summary(position, current_date)
                    
                    del self.positions[position.symbol]
        
        except Exception as e:
            self.logger.error(f"Error checking leg profit targets for {position.symbol}: {e}")
        
        return signals
    
    def _check_breakout_adjustments(self, position: PositionState, current_date: date) -> List[TradingSignal]:
        """Check for breakouts and generate adjustment signals."""
        signals = []
        
        try:
            # Only adjust if we haven't hit max adjustments
            if position.adjustment_count >= self.strategy_config.max_adjustments_per_stock:
                return signals
            
            # Check minimum days between adjustments
            if (position.last_adjustment_date and 
                (current_date - position.last_adjustment_date).days < self.strategy_config.min_days_between_adjustments):
                return signals
            
            # Get current spot price
            spot_price = self.data_manager.get_stock_price(position.symbol, current_date)
            if not spot_price:
                return signals
            
            # Detect breakout
            breakout_direction = self._detect_breakout(position, spot_price)
            
            if breakout_direction != BreakoutDirection.NONE:
                adjustment_signals = self._generate_adjustment_signals(position, breakout_direction, current_date)
                signals.extend(adjustment_signals)
            
        except Exception as e:
            self.logger.error(f"Error checking breakout adjustments: {e}")
        
        return signals
    
    def _detect_breakout(self, position: PositionState, spot_price: float) -> BreakoutDirection:
        """Detect if spot price has broken out based on current position type with configurable buffer."""
        
        buffer = self.strategy_config.adjustment_trigger_buffer_pct
        
        # For strangle positions, check against actual option strikes with buffer
        if position.position_type == PositionType.STRANGLE:
            # Check if we have active call position
            if position.call_strike and position.call_quantity != 0:
                # Upside breakout: spot price crosses call strike (with buffer for cushion)
                # If buffer=0.01 (1%), trigger when spot > call_strike * (1 + 0.01) = call_strike * 1.01
                upside_trigger = position.call_strike * (1 + buffer)
                if spot_price > upside_trigger:
                    return BreakoutDirection.UPSIDE
            
            # Check if we have active put position
            if position.put_strike and position.put_quantity != 0:
                # Downside breakout: spot price crosses put strike (with buffer for cushion)  
                # If buffer=0.01 (1%), trigger when spot < put_strike * (1 - 0.01) = put_strike * 0.99
                downside_trigger = position.put_strike * (1 - buffer)
                if spot_price < downside_trigger:
                    return BreakoutDirection.DOWNSIDE
                
            return BreakoutDirection.NONE
        
        # For bullish positions (short put only), check downside breakout against put strike
        elif position.position_type == PositionType.BULLISH:
            if position.put_strike:
                # Buffer provides cushion before triggering
                # If buffer=0.01, trigger when spot < put_strike * 0.99
                trigger_level = position.put_strike * (1 - buffer)
                if spot_price < trigger_level:
                    return BreakoutDirection.DOWNSIDE
            return BreakoutDirection.NONE
        
        # For bearish positions (short call only), check upside breakout against call strike
        elif position.position_type == PositionType.BEARISH:
            if position.call_strike:
                # Buffer provides cushion before triggering
                # If buffer=0.01, trigger when spot > call_strike * 1.01
                trigger_level = position.call_strike * (1 + buffer)
                if spot_price > trigger_level:
                    return BreakoutDirection.UPSIDE
            return BreakoutDirection.NONE
        
        return BreakoutDirection.NONE
    
    def _generate_adjustment_signals(self, position: PositionState, breakout_direction: BreakoutDirection, 
                                   current_date: date) -> List[TradingSignal]:
        """Generate adjustment signals based on breakout direction."""
        signals = []
        
        try:
            # Get current market context
            current_spot = self.data_manager.get_stock_price(position.symbol, current_date)
            buffer = self.strategy_config.adjustment_trigger_buffer_pct
            
            # Calculate trigger levels based on actual position strikes (corrected buffer direction)
            if position.position_type == PositionType.STRANGLE:
                if breakout_direction == BreakoutDirection.UPSIDE and position.call_strike:
                    trigger_level = position.call_strike * (1 + buffer)
                elif breakout_direction == BreakoutDirection.DOWNSIDE and position.put_strike:
                    trigger_level = position.put_strike * (1 - buffer)
                else:
                    trigger_level = current_spot  # Fallback
            elif position.position_type == PositionType.BULLISH:
                trigger_level = position.put_strike * (1 - buffer)
            else:  # BEARISH
                trigger_level = position.call_strike * (1 + buffer)
            
            # Log pre-adjustment state
            current_pnl = self._calculate_position_pnl(position, current_date)
            self.logger.info(f"🔄 ADJUSTMENT #{position.adjustment_count + 1}: {position.symbol} | Date: {current_date} | Breakout: {breakout_direction.value}")
            self.logger.info(f"  ├─ Spot: ₹{current_spot:.2f} | Trigger: ₹{trigger_level:.2f} | Buffer: {buffer:.1%}")
            self.logger.info(f"  ├─ Current Position: {position.position_type.value} | P&L: ₹{current_pnl:,.0f}")
            
            # Log what's being closed
            if position.call_quantity != 0 and position.put_quantity != 0:
                self.logger.info(f"  ├─ Closing: Call@{position.call_strike} & Put@{position.put_strike}")
            elif position.call_quantity != 0:
                self.logger.info(f"  ├─ Closing: Call@{position.call_strike}")
            elif position.put_quantity != 0:
                self.logger.info(f"  ├─ Closing: Put@{position.put_strike}")
            
            # First close existing strangle
            close_signals = self._generate_exit_signals(position, current_date, 
                                                      f"Breakout adjustment: {breakout_direction.value}",
                                                      is_adjustment=True)
            signals.extend(close_signals)
            
            # Then enter directional position
            if breakout_direction == BreakoutDirection.UPSIDE:
                # Enter bullish position (short put only)
                directional_signals = self._generate_bullish_adjustment(position, current_date)
                new_position_type = "Bullish (Put only)"
            else:  # DOWNSIDE
                # Enter bearish position (short call only)
                directional_signals = self._generate_bearish_adjustment(position, current_date)
                new_position_type = "Bearish (Call only)"
            
            signals.extend(directional_signals)
            
            # Update position state
            position.adjustment_count += 1
            position.last_adjustment_date = current_date
            
            # Log post-adjustment state
            self.logger.info(f"  ├─ Opening: {new_position_type}")
            if directional_signals:
                signal = directional_signals[0]
                atm_strike = signal.strike
                self.logger.info(f"  ├─ New Position: {signal.option_type.value}@{atm_strike}")
            
            adjustment_pnl = self._calculate_position_pnl(position, current_date)
            self.logger.info(f"  └─ Adjustment Complete | New P&L: ₹{adjustment_pnl:,.0f}")
            
        except Exception as e:
            self.logger.error(f"Error generating adjustment signals: {e}")
        
        return signals
    
    def _generate_bullish_adjustment(self, position: PositionState, current_date: date) -> List[TradingSignal]:
        """Generate signals for bullish adjustment (short put only)."""
        signals = []
        
        try:
            # Get current spot and ATM strike
            spot_price = self.data_manager.get_stock_price(position.symbol, current_date)
            atm_strike = self.data_manager.get_atm_strike(position.symbol, position.expiry, current_date)
            
            if not atm_strike or not spot_price:
                return signals
            
            # Get put price
            put_price = self.data_manager.get_option_price(position.symbol, atm_strike, position.expiry, "put", current_date)
            if not put_price:
                put_price = max(10, (atm_strike - spot_price) * 0.1)  # Estimate
            
            # Calculate quantity - use whichever leg was active
            if position.position_type == PositionType.BEARISH:
                quantity = abs(position.call_quantity)
            elif position.position_type == PositionType.STRANGLE:
                # For strangle, use the non-zero quantity (either call or put)
                if position.call_quantity != 0:
                    quantity = abs(position.call_quantity)
                elif position.put_quantity != 0:
                    quantity = abs(position.put_quantity)
                else:
                    # Both legs closed/empty - shouldn't happen
                    self.logger.warning(f"Bullish adjustment skipped: no active legs for {position.symbol}")
                    return signals
            else:
                # For existing BULLISH position
                quantity = abs(position.put_quantity)
            
            # Update position state for bullish adjustment
            position.position_type = PositionType.BULLISH
            position.put_strike = atm_strike
            position.put_premium = put_price
            position.call_strike = None
            position.call_premium = None
            position.call_quantity = 0
            position.put_quantity = -quantity
            
            # Calculate proper margin and store premium collected  
            exposure = position.lot_size * quantity * spot_price
            position.margin_deployed = exposure * self.strategy_config.margin_percentage
            position.exposure = exposure  # Update exposure for new position
            position.metadata['premium_collected'] = put_price * position.lot_size * quantity
            
            # Update current market prices
            position.current_spot_price = spot_price
            position.current_put_premium = put_price
            position.current_call_premium = None
            
            # Generate sell signal for ATM put
            put_signal = TradingSignal(
                symbol=position.symbol,
                strike=atm_strike,
                expiry=position.expiry,
                option_type=OptionType.PUT,
                action=SignalAction.SELL,
                quantity=quantity,
                reason=f"Bullish adjustment: Sell ATM put at ₹{spot_price}",
                metadata={'position_type': 'bullish_adjustment', 'adjustment_count': position.adjustment_count}
            )
            
            signals.append(put_signal)
            
        except Exception as e:
            self.logger.error(f"Error generating bullish adjustment: {e}")
        
        return signals
    
    def _generate_bearish_adjustment(self, position: PositionState, current_date: date) -> List[TradingSignal]:
        """Generate signals for bearish adjustment (short call only)."""
        signals = []
        
        try:
            # Get current spot and ATM strike
            spot_price = self.data_manager.get_stock_price(position.symbol, current_date)
            atm_strike = self.data_manager.get_atm_strike(position.symbol, position.expiry, current_date)
            
            if not atm_strike or not spot_price:
                return signals
            
            # Get call price
            call_price = self.data_manager.get_option_price(position.symbol, atm_strike, position.expiry, "call", current_date)
            if not call_price:
                call_price = max(10, (spot_price - atm_strike) * 0.1)  # Estimate
            
            # Calculate quantity - use whichever leg was active
            if position.position_type == PositionType.BULLISH:
                quantity = abs(position.put_quantity)
            elif position.position_type == PositionType.STRANGLE:
                # For strangle, use the non-zero quantity (either call or put)
                if position.put_quantity != 0:
                    quantity = abs(position.put_quantity)
                elif position.call_quantity != 0:
                    quantity = abs(position.call_quantity)
                else:
                    # Both legs closed/empty - shouldn't happen
                    self.logger.warning(f"Bearish adjustment skipped: no active legs for {position.symbol}")
                    return signals
            else:
                # For existing BEARISH position
                quantity = abs(position.call_quantity)
            
            # Update position state for bearish adjustment
            position.position_type = PositionType.BEARISH
            position.call_strike = atm_strike
            position.call_premium = call_price
            position.put_strike = None
            position.put_premium = None
            position.put_quantity = 0
            position.call_quantity = -quantity
            
            # Calculate proper margin and store premium collected
            exposure = position.lot_size * quantity * spot_price
            position.margin_deployed = exposure * self.strategy_config.margin_percentage
            position.exposure = exposure  # Update exposure for new position
            position.metadata['premium_collected'] = call_price * position.lot_size * quantity
            
            # Update current market prices
            position.current_spot_price = spot_price
            position.current_call_premium = call_price
            position.current_put_premium = None
            
            # Generate sell signal for ATM call
            call_signal = TradingSignal(
                symbol=position.symbol,
                strike=atm_strike,
                expiry=position.expiry,
                option_type=OptionType.CALL,
                action=SignalAction.SELL,
                quantity=quantity,
                reason=f"Bearish adjustment: Sell ATM call at ₹{spot_price}",
                metadata={'position_type': 'bearish_adjustment', 'adjustment_count': position.adjustment_count}
            )
            
            signals.append(call_signal)
            
        except Exception as e:
            self.logger.error(f"Error generating bearish adjustment: {e}")
        
        return signals
    
    def _generate_exit_signals(self, position: PositionState, current_date: date, reason: str, is_adjustment: bool = False) -> List[TradingSignal]:
        """Generate exit signals for a position."""
        signals = []
        
        try:
            # Get current option prices for P&L calculation
            call_pnl = 0.0
            put_pnl = 0.0
            current_call_price = None
            current_put_price = None
            
            # Close call leg if it exists
            if position.call_quantity != 0:
                current_call_price = self.data_manager.get_option_price(
                    position.symbol, position.call_strike, position.expiry, "call", current_date
                )
                if current_call_price and position.call_premium:
                    call_pnl_per_unit = position.call_premium - current_call_price
                    call_pnl = call_pnl_per_unit * abs(position.call_quantity) * position.lot_size
                
                call_signal = TradingSignal(
                    symbol=position.symbol,
                    strike=position.call_strike,
                    expiry=position.expiry,
                    option_type=OptionType.CALL,
                    action=SignalAction.BUY,  # Buy to close short position
                    quantity=abs(position.call_quantity),
                    reason=reason,
                    metadata={'position_type': 'exit', 'leg': 'call'}
                )
                signals.append(call_signal)
            
            # Close put leg if it exists
            if position.put_quantity != 0:
                current_put_price = self.data_manager.get_option_price(
                    position.symbol, position.put_strike, position.expiry, "put", current_date
                )
                if current_put_price and position.put_premium:
                    put_pnl_per_unit = position.put_premium - current_put_price
                    put_pnl = put_pnl_per_unit * abs(position.put_quantity) * position.lot_size
                
                put_signal = TradingSignal(
                    symbol=position.symbol,
                    strike=position.put_strike,
                    expiry=position.expiry,
                    option_type=OptionType.PUT,
                    action=SignalAction.BUY,  # Buy to close short position
                    quantity=abs(position.put_quantity),
                    reason=reason,
                    metadata={'position_type': 'exit', 'leg': 'put'}
                )
                signals.append(put_signal)
            
            # Calculate final P&L and update position
            final_pnl = self._calculate_position_pnl(position, current_date)
            position.realized_pnl = final_pnl
            position.unrealized_pnl = 0.0
            
            # Store leg P&L for detailed tracking
            if position.call_quantity != 0:
                position.call_pnl = call_pnl
            if position.put_quantity != 0:
                position.put_pnl = put_pnl
            position.final_pnl = final_pnl
            
            # Remove position from tracking only if it's not an adjustment
            if not is_adjustment and position.symbol in self.positions:
                # Add to completed positions for analysis
                self.completed_positions.append(position)
                del self.positions[position.symbol]
                
                # Log comprehensive position summary
                self._log_position_summary(position, current_date)
            
            # Update trade statistics
            self.total_trades += 1
            if final_pnl > 0:
                self.winning_trades += 1
            
            # Enhanced exit logging with P&L breakdown
            current_spot = self.data_manager.get_stock_price(position.symbol, current_date)
            self.logger.info(f"📉 EXIT: {position.symbol} {position.position_type.value} | Reason: {reason} | Spot: ₹{current_spot:.2f}")
            
            if position.call_quantity != 0 and current_call_price:
                self.logger.info(f"  ├─ BUY Call@{position.call_strike} x{abs(position.call_quantity)} | "
                               f"Entry: ₹{position.call_premium:.2f} → Exit: ₹{current_call_price:.2f} | Leg P&L: ₹{call_pnl:,.0f}")
            
            if position.put_quantity != 0 and current_put_price:
                self.logger.info(f"  ├─ BUY Put@{position.put_strike} x{abs(position.put_quantity)} | "
                               f"Entry: ₹{position.put_premium:.2f} → Exit: ₹{current_put_price:.2f} | Leg P&L: ₹{put_pnl:,.0f}")
            
            days_held = (current_date - position.entry_date).days
            self.logger.info(f"  └─ Position P&L: ₹{final_pnl:,.0f} | Days Held: {days_held} | Win: {'✅' if final_pnl > 0 else '❌'}")
            
        except Exception as e:
            self.logger.error(f"Error generating exit signals: {e}")
        
        return signals
    
    def _generate_hedge_signals(self, current_date: date) -> List[TradingSignal]:
        """Generate hedge signals for NIFTY positions."""
        signals = []
        
        # TODO: Implement NIFTY hedge logic
        # For now, skip hedge implementation
        
        return signals
    
    def _calculate_position_pnl(self, position: PositionState, current_date: date) -> float:
        """Calculate current P&L for a position."""
        total_pnl = 0.0
        
        try:
            # Calculate call leg P&L
            if position.call_quantity != 0 and position.call_strike:
                current_call_price = self.data_manager.get_option_price(
                    position.symbol, position.call_strike, position.expiry, "call", current_date
                )
                if current_call_price:
                    # P&L = (Entry Premium - Current Premium) * Quantity * Lot Size
                    # For short positions, we profit when option price decreases
                    call_pnl = (position.call_premium - current_call_price) * abs(position.call_quantity)
                    call_pnl *= position.lot_size
                    total_pnl += call_pnl
            
            # Calculate put leg P&L
            if position.put_quantity != 0 and position.put_strike:
                current_put_price = self.data_manager.get_option_price(
                    position.symbol, position.put_strike, position.expiry, "put", current_date
                )
                if current_put_price:
                    put_pnl = (position.put_premium - current_put_price) * abs(position.put_quantity)
                    put_pnl *= position.lot_size
                    total_pnl += put_pnl
            
        except Exception as e:
            self.logger.error(f"Error calculating P&L for {position.symbol}: {e}")
        
        return total_pnl
    
    def _calculate_leg_pnl(self, position: PositionState, leg: str, current_date: date) -> Tuple[float, float]:
        """Calculate P&L and profit percentage for a specific leg.
        
        Returns:
            Tuple[float, float]: (pnl_amount, profit_percentage)
        """
        
        if leg == "call" and position.call_quantity != 0 and position.call_strike:
            current_price = self.data_manager.get_option_price(
                position.symbol, position.call_strike, position.expiry, "call", current_date
            )
            if current_price and position.call_premium:
                # For short positions, profit when current price < entry price
                pnl_per_unit = position.call_premium - current_price
                pnl = pnl_per_unit * abs(position.call_quantity)
                pnl *= position.lot_size
                profit_pct = pnl_per_unit / position.call_premium
                return pnl, profit_pct
        
        elif leg == "put" and position.put_quantity != 0 and position.put_strike:
            current_price = self.data_manager.get_option_price(
                position.symbol, position.put_strike, position.expiry, "put", current_date
            )
            if current_price and position.put_premium:
                # For short positions, profit when current price < entry price
                pnl_per_unit = position.put_premium - current_price
                pnl = pnl_per_unit * abs(position.put_quantity)
                pnl *= position.lot_size
                profit_pct = pnl_per_unit / position.put_premium
                return pnl, profit_pct
        
        return 0.0, 0.0
    
    def _calculate_total_portfolio_pnl(self, current_date: date) -> float:
        """Calculate total portfolio P&L."""
        total_pnl = 0.0
        
        for position in self.positions.values():
            position_pnl = self._calculate_position_pnl(position, current_date)
            total_pnl += position_pnl
        
        # Add hedge position P&L if exists
        if self.hedge_position:
            hedge_pnl = self._calculate_position_pnl(self.hedge_position, current_date)
            total_pnl += hedge_pnl
        
        return total_pnl
    
    def log_position_status(self, position: PositionState, current_date: date) -> None:
        """Log current position status with MTM details."""
        try:
            current_spot = self.data_manager.get_stock_price(position.symbol, current_date)
            self.logger.debug(f"📊 POSITION STATUS: {position.symbol} {position.position_type.value}")
            self.logger.debug(f"  ├─ Spot: ₹{current_spot:.2f} | Entry: ₹{position.entry_spot_price:.2f} | Move: ₹{current_spot - position.entry_spot_price:+.2f}")
            
            if position.call_quantity != 0 and position.call_strike:
                current_call_price = self.data_manager.get_option_price(
                    position.symbol, position.call_strike, position.expiry, "call", current_date
                )
                if current_call_price:
                    call_move = current_call_price - position.call_premium
                    self.logger.debug(f"  ├─ Call@{position.call_strike}: ₹{current_call_price:.2f} (Entry: ₹{position.call_premium:.2f}, Move: ₹{call_move:+.2f})")
            
            if position.put_quantity != 0 and position.put_strike:
                current_put_price = self.data_manager.get_option_price(
                    position.symbol, position.put_strike, position.expiry, "put", current_date
                )
                if current_put_price:
                    put_move = current_put_price - position.put_premium
                    self.logger.debug(f"  ├─ Put@{position.put_strike}: ₹{current_put_price:.2f} (Entry: ₹{position.put_premium:.2f}, Move: ₹{put_move:+.2f})")
            
            unrealized_pnl = self._calculate_position_pnl(position, current_date)
            days_held = (current_date - position.entry_date).days
            self.logger.debug(f"  └─ Unrealized P&L: ₹{unrealized_pnl:,.0f} | Days: {days_held} | Adjustments: {position.adjustment_count}")
            
        except Exception as e:
            self.logger.error(f"Error logging position status: {e}")
    
    def _update_position_mtm(self, position: PositionState, current_date: date) -> None:
        """Update position with current market prices and P&L."""
        try:
            # Get current spot price
            current_spot = self.data_manager.get_stock_price(position.symbol, current_date)
            if current_spot:
                position.current_spot_price = current_spot
            else:
                position.current_spot_price = position.entry_spot_price  # Fallback
            
            # Update call premium if holding call position
            if position.call_quantity != 0 and position.call_strike:
                call_price = self.data_manager.get_option_price(
                    position.symbol, position.call_strike, position.expiry, "call", current_date
                )
                position.current_call_premium = call_price
            
            # Update put premium if holding put position
            if position.put_quantity != 0 and position.put_strike:
                put_price = self.data_manager.get_option_price(
                    position.symbol, position.put_strike, position.expiry, "put", current_date
                )
                position.current_put_premium = put_price
            
            # Calculate and store current unrealized P&L
            position.unrealized_pnl = self._calculate_position_pnl(position, current_date)
            
            # Add to daily P&L history
            daily_entry = {
                'date': current_date,
                'spot_price': position.current_spot_price,
                'call_premium': position.current_call_premium,
                'put_premium': position.current_put_premium,
                'unrealized_pnl': position.unrealized_pnl,
                'position_type': position.position_type.value,
                'adjustment_count': position.adjustment_count
            }
            
            # Avoid duplicate entries for the same date
            if not position.daily_pnl_history or position.daily_pnl_history[-1]['date'] != current_date:
                position.daily_pnl_history.append(daily_entry)
                
        except Exception as e:
            self.logger.error(f"Error updating position MTM for {position.symbol}: {e}")
    
    def _calculate_monthly_pnl(self) -> float:
        """Calculate P&L for the previous month."""
        # This would typically look at realized P&L from closed positions
        # For now, return 0 as placeholder
        return 0.0
    
    def _calculate_trigger_distance(self, position: PositionState, direction: str, current_date: date) -> float:
        """Calculate distance to adjustment trigger levels.
        
        Returns:
            float: Percentage distance to trigger (positive = above trigger, negative = below trigger)
        """
        try:
            current_spot = self.data_manager.get_stock_price(position.symbol, current_date)
            if not current_spot:
                current_spot = position.current_spot_price
            
            buffer = self.strategy_config.adjustment_trigger_buffer_pct
            
            if direction == 'upside':
                if position.position_type == PositionType.STRANGLE and position.call_strike:
                    trigger_level = position.call_strike * (1 + buffer)
                elif position.position_type == PositionType.BEARISH and position.call_strike:
                    trigger_level = position.call_strike * (1 + buffer)
                else:
                    return 0.0
                    
                # Return percentage distance (negative means still below trigger)
                return (current_spot - trigger_level) / trigger_level
                
            elif direction == 'downside':
                if position.position_type == PositionType.STRANGLE and position.put_strike:
                    trigger_level = position.put_strike * (1 - buffer)
                elif position.position_type == PositionType.BULLISH and position.put_strike:
                    trigger_level = position.put_strike * (1 - buffer)
                else:
                    return 0.0
                    
                # Return percentage distance (negative means still above trigger)
                return (trigger_level - current_spot) / trigger_level
                
        except Exception as e:
            self.logger.error(f"Error calculating trigger distance: {e}")
            
        return 0.0
    
    def _get_symbol_analytics(self, symbol: str, current_date: date) -> Dict[str, Any]:
        """Get analytics for a specific symbol with individual leg performance."""
        analytics = {}
        
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Calculate individual leg P&L and profit percentages
            call_pnl, call_profit_pct = self._calculate_leg_pnl(position, "call", current_date)
            put_pnl, put_profit_pct = self._calculate_leg_pnl(position, "put", current_date)
            
            analytics.update({
                'position_type': position.position_type.value,
                'entry_date': position.entry_date,
                'expiry': position.expiry,
                'days_to_expiry': (position.expiry - current_date).days,
                'adjustment_count': position.adjustment_count,
                
                # Overall position
                'total_unrealized_pnl': call_pnl + put_pnl,
                'margin_deployed': position.margin_deployed,
                
                # Call leg analytics
                'call_strike': position.call_strike,
                'call_entry_premium': position.call_premium,
                'call_current_premium': position.current_call_premium,
                'call_quantity': position.call_quantity,
                'call_pnl': call_pnl,
                'call_profit_pct': call_profit_pct,
                'call_profit_target_hit': call_profit_pct >= self.strategy_config.leg_profit_target_pct if call_profit_pct > 0 else False,
                
                # Put leg analytics
                'put_strike': position.put_strike,
                'put_entry_premium': position.put_premium,
                'put_current_premium': position.current_put_premium,
                'put_quantity': position.put_quantity,
                'put_pnl': put_pnl,
                'put_profit_pct': put_profit_pct,
                'put_profit_target_hit': put_profit_pct >= self.strategy_config.leg_profit_target_pct if put_profit_pct > 0 else False,
                
                # Current market
                'current_spot': position.current_spot_price,
                'entry_spot': position.entry_spot_price,
                'spot_move_pct': (position.current_spot_price - position.entry_spot_price) / position.entry_spot_price if position.entry_spot_price > 0 else 0,
                
                # Risk metrics
                'prev_month_high': position.prev_month_high,
                'prev_month_low': position.prev_month_low,
                'distance_to_upside_trigger': self._calculate_trigger_distance(position, 'upside', current_date),
                'distance_to_downside_trigger': self._calculate_trigger_distance(position, 'downside', current_date)
            })
        else:
            analytics['position_type'] = 'none'
        
        return analytics
    
    def _get_portfolio_analytics(self, current_date: date) -> Dict[str, Any]:
        """Get portfolio-level analytics."""
        total_pnl = self._calculate_total_portfolio_pnl(current_date)
        active_positions = len(self.positions)
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'total_pnl': total_pnl,
            'active_positions': active_positions,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate_pct': win_rate,
            'capital_deployed': sum(p.margin_deployed for p in self.positions.values())
        }
    
    def on_end_of_day(self, current_date: date, portfolio) -> None:
        """End-of-day processing and logging."""
        try:
            # Log daily position status for each active position
            for symbol, position in self.positions.items():
                self._update_position_mtm(position, current_date)
                self.log_position_status(position, current_date)
            
            # Log portfolio summary if there are active positions
            if self.positions:
                total_pnl = self._calculate_total_portfolio_pnl(current_date)
                total_margin = sum(p.margin_deployed for p in self.positions.values())
                self.logger.debug(f"📋 PORTFOLIO SUMMARY: Total P&L: ₹{total_pnl:,.0f} | Margin: ₹{total_margin:,.0f} | Positions: {len(self.positions)}")
                
        except Exception as e:
            self.logger.error(f"Error in end_of_day processing: {e}")
    
    def _log_position_summary(self, position: PositionState, close_date: date) -> None:
        """Log comprehensive position summary when fully closed."""
        try:
            days_held = (close_date - position.entry_date).days
            premium_collected = position.metadata.get('premium_collected', 0)
            
            # Calculate different return metrics
            premium_retention = (position.final_pnl / premium_collected * 100) if premium_collected > 0 else 0
            return_on_margin = (position.final_pnl / position.margin_deployed * 100) if position.margin_deployed > 0 else 0
            return_on_exposure = (position.final_pnl / position.exposure * 100) if position.exposure > 0 else 0
            
            self.logger.info("=" * 80)
            self.logger.info(f"📊 POSITION SUMMARY: {position.symbol} ({position.entry_date} → {close_date})")
            self.logger.info("=" * 80)
            self.logger.info(f"Position Type: {position.position_type.value.upper()} | Days Held: {days_held}")
            self.logger.info(f"Entry Spot: ₹{position.entry_spot_price:.2f} | Exit Spot: ₹{position.current_spot_price:.2f}")
            
            # Leg breakdown
            if position.call_quantity != 0:
                call_entry_value = position.call_premium * position.lot_size
                call_return_pct = (position.call_pnl / call_entry_value * 100) if call_entry_value > 0 else 0
                self.logger.info(f"├─ CALL@{position.call_strike}: P&L ₹{position.call_pnl:,.0f} ({call_return_pct:+.1f}%)")
            
            if position.put_quantity != 0:
                put_entry_value = position.put_premium * position.lot_size  
                put_return_pct = (position.put_pnl / put_entry_value * 100) if put_entry_value > 0 else 0
                self.logger.info(f"├─ PUT@{position.put_strike}: P&L ₹{position.put_pnl:,.0f} ({put_return_pct:+.1f}%)")
            
            # Financial Summary with clear metrics
            win_status = "🎉 WIN" if position.final_pnl > 0 else "💸 LOSS"
            self.logger.info(f"├─ Premium Collected: ₹{premium_collected:,.0f}")
            self.logger.info(f"├─ Margin Deployed: ₹{position.margin_deployed:,.0f}")
            self.logger.info(f"├─ Exposure: ₹{position.exposure:,.0f}")
            self.logger.info(f"├─ Total P&L: ₹{position.final_pnl:,.0f}")
            self.logger.info(f"├─ Premium Retention: {premium_retention:+.1f}% (kept ₹{position.final_pnl:,.0f} of ₹{premium_collected:,.0f})")
            self.logger.info(f"├─ Return on Margin: {return_on_margin:+.1f}%")
            self.logger.info(f"└─ Return on Exposure: {return_on_exposure:+.1f}% | Status: {win_status}")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Error logging position summary: {e}")
    
    def _log_monthly_summary(self, current_date: date) -> None:
        """Log monthly performance summary."""
        try:
            if not self.current_month:
                return
                
            year, month = self.current_month
            month_name = current_date.replace(day=1).strftime('%B %Y')
            
            # Calculate monthly stats
            monthly_positions = [p for p in self.completed_positions 
                               if p.entry_date.year == year and p.entry_date.month == month]
            
            if not monthly_positions:
                self.logger.info(f"📅 {month_name}: No positions completed")
                return
            
            total_pnl = sum(p.final_pnl for p in monthly_positions)
            wins = len([p for p in monthly_positions if p.final_pnl > 0])
            losses = len(monthly_positions) - wins
            win_rate = (wins / len(monthly_positions) * 100) if monthly_positions else 0
            
            best_trade = max(monthly_positions, key=lambda p: p.final_pnl) if monthly_positions else None
            worst_trade = min(monthly_positions, key=lambda p: p.final_pnl) if monthly_positions else None
            
            self.logger.info("\n" + "🗓️ " * 20 + " MONTHLY SUMMARY " + "🗓️ " * 20)
            self.logger.info(f"Month: {month_name}")
            self.logger.info(f"Positions: {len(monthly_positions)} | Wins: {wins} | Losses: {losses}")
            self.logger.info(f"Win Rate: {win_rate:.1f}% | Total P&L: ₹{total_pnl:,.0f}")
            
            if best_trade:
                self.logger.info(f"Best Trade: {best_trade.symbol} +₹{best_trade.final_pnl:,.0f}")
            if worst_trade:
                self.logger.info(f"Worst Trade: {worst_trade.symbol} ₹{worst_trade.final_pnl:,.0f}")
            
            self.logger.info("🗓️ " * 65 + "\n")
            
            # Store monthly stats
            self.monthly_stats[self.current_month] = {
                'positions': len(monthly_positions),
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'best_trade': best_trade.final_pnl if best_trade else 0,
                'worst_trade': worst_trade.final_pnl if worst_trade else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error logging monthly summary: {e}")