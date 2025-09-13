# Middle Way Strategy Implementation Plan

## Overview
Implementation of the Middle Way trading strategy - a short strangle strategy with trend-following adjustments. Starting with a simplified version (1 stock + NIFTY hedge) before scaling to full 25-stock portfolio.

## Phase 1: Simplified Implementation (1 Stock + Hedge)

### Step 1: Data Infrastructure
**File**: `skas_backtest/core/data_manager.py`

#### 1.1 Create NSE Data Manager
- Extend base DataManager class
- Integrate with NSE options pricing library
- Implement caching for performance

#### 1.2 Core Data Methods
```python
- get_stock_price(symbol, date)
- get_option_chain(symbol, expiry, date)
- get_trading_dates(start_date, end_date)
- get_monthly_expiry(year, month)
- get_lot_size(symbol, date)
```

#### 1.3 Strike Selection Utilities
```python
- find_closest_strike(symbol, target_price, expiry, date)
- get_atm_strike(symbol, date, expiry)
- get_otm_strikes(symbol, spot_price, otm_percentage, expiry, date)
```

#### 1.4 Historical Data Methods
```python
- get_month_high_low(symbol, year, month)
- get_previous_month_range(symbol, current_date)
- get_historical_prices(symbol, start_date, end_date)
```

### Step 2: Middle Way Strategy Class
**File**: `skas_backtest/strategies/middle_way_strategy.py`

#### 2.1 Configuration
```python
class MiddleWayConfig:
    # Single stock for testing
    test_stock = "RELIANCE"
    
    # Full stock list (for future scaling)
    stock_list = [
        "HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "BHARTIARTL",
        "LT", "ITC", "TCS", "SBIN", "AXISBANK", "KOTAKBANK", "M&M",
        "HINDUNILVR", "BAJFINANCE", "ZOMATO", "MARUTI", "SUNPHARMA",
        "NTPC", "HCLTECH", "ULTRACEMCO", "TITAN", "TATAMOTORS",
        "BEL", "TATASTEEL", "POWERGRID"
    ]
    
    # Capital allocation
    total_capital = 5000000  # 50 lakhs
    capital_per_stock = 200000  # 2 lakhs
    
    # Risk parameters
    portfolio_stop_loss_pct = 0.02  # 2% of capital
    position_profit_target_pct = 0.02  # 2% of margin
    
    # Hedge parameters
    hedge_otm_pct = 0.04  # 4% OTM for NIFTY hedge
    
    # Adjustment constraints
    max_adjustments_per_stock = 2
    min_days_between_adjustments = 1
```

#### 2.2 Strategy State Tracking
```python
class PositionState:
    symbol: str
    entry_date: date
    expiry: date
    call_strike: float
    put_strike: float
    call_premium: float
    put_premium: float
    position_type: str  # "strangle", "bullish", "bearish"
    adjustment_count: int
    last_adjustment_date: date
    entry_spot_price: float
    margin_deployed: float
    realized_pnl: float
    unrealized_pnl: float
```

#### 2.3 Core Strategy Methods
```python
class MiddleWayStrategy(Strategy):
    def initialize(self):
        # Initialize position tracking
        # Set up configuration
        
    def generate_signals(self, current_date, symbols):
        # Check for new month entry
        # Monitor existing positions
        # Detect breakouts
        # Generate adjustment signals
        # Check risk limits
        
    def enter_monthly_positions(self, current_date):
        # Calculate previous month high/low
        # Select strikes
        # Enter short strangle
        # Enter NIFTY hedge
        
    def check_breakouts(self, current_date):
        # Compare spot vs strikes
        # Return breakout signals
        
    def adjust_position(self, symbol, breakout_type):
        # Exit existing strangle
        # Enter directional position
        # Update adjustment count
        
    def check_exit_conditions(self, current_date):
        # Check portfolio stop loss
        # Check individual profit targets
        # Check expiry (T-1 day)
```

### Step 3: Position Management
**File**: `skas_backtest/core/position_manager.py`

#### 3.1 Position Types
- Short Strangle (initial position)
- Bullish (short put only after upside breakout)
- Bearish (short call only after downside breakout)
- Hedge (long NIFTY strangle)

#### 3.2 Position Tracking
```python
class PositionManager:
    def open_strangle(self, symbol, call_strike, put_strike, quantities, date):
        # Open short strangle position
        
    def close_strangle(self, symbol, date):
        # Close both legs
        
    def convert_to_bullish(self, symbol, date):
        # Close strangle, sell ATM put
        
    def convert_to_bearish(self, symbol, date):
        # Close strangle, sell ATM call
        
    def calculate_position_pnl(self, position, current_prices):
        # Calculate MTM for position
```

### Step 4: Risk Manager
**File**: `skas_backtest/core/risk_manager.py`

#### 4.1 Risk Checks
```python
class MiddleWayRiskManager(RiskManager):
    def check_portfolio_stop_loss(self, portfolio, stop_loss_amount):
        # Check if total MTM loss exceeds limit
        
    def check_position_profit_target(self, position, target_amount):
        # Check if position profit exceeds target
        
    def check_expiry_exit(self, current_date, expiry_date):
        # Check if T-1 day before expiry
        
    def validate_adjustment(self, position, current_date):
        # Check adjustment count and timing constraints
```

### Step 5: Testing Framework
**File**: `tests/test_middle_way_strategy.py`

#### 5.1 Unit Tests
- Test strike selection logic
- Test breakout detection
- Test adjustment constraints
- Test risk management triggers

#### 5.2 Integration Tests
- Test 2-month backtest with RELIANCE
- Verify entry/exit mechanics
- Validate P&L calculations
- Check hedge positions

#### 5.3 Test Scenarios
1. **Normal Month**: No breakouts, exit at expiry
2. **Single Breakout**: One adjustment triggered
3. **Multiple Breakouts**: Hit max adjustment limit
4. **Stop Loss**: Portfolio loss triggers exit
5. **Profit Target**: Individual position target hit

## Phase 2: Validation & Debugging

### Backtest Configuration
```python
test_config = {
    "start_date": "2024-01-01",
    "end_date": "2024-03-31",  # 3 months
    "symbol": "RELIANCE",
    "initial_capital": 200000,
    "enable_logging": True,
    "log_level": "DEBUG"
}
```

### Validation Checklist
- [ ] Entry on first trading day of month
- [ ] Correct strike selection (previous month high/low)
- [ ] Proper margin calculation
- [ ] Breakout detection accuracy
- [ ] Adjustment execution (max 2)
- [ ] NIFTY hedge entry/exit
- [ ] Stop loss trigger at 2%
- [ ] Profit target trigger at 2%
- [ ] Exit 1 day before expiry
- [ ] P&L calculation accuracy

### Debug Output
Generate detailed logs including:
- Daily position status
- Breakout events
- Adjustment triggers
- P&L breakdown
- Risk metric violations
- Trade execution details

## Phase 3: Scaling to Full Portfolio

### 3.1 Multi-Stock Extension
Once single stock is validated:
1. Loop through all 25 stocks
2. Proportional capital allocation
3. Aggregate portfolio metrics
4. Parallel position management

### 3.2 Performance Optimization
- Batch data fetching
- Parallel processing for signals
- Caching frequently accessed data
- Optimize strike selection algorithms

### 3.3 Additional Features
- Position sizing based on volatility
- Dynamic hedge adjustment
- Correlation-based risk management
- Advanced analytics and reporting

## Implementation Timeline

### Week 1: Data Infrastructure
- Day 1-2: Data manager implementation
- Day 3-4: Strike selection utilities
- Day 5: Testing data fetching

### Week 2: Core Strategy
- Day 1-2: Strategy class structure
- Day 3-4: Entry/exit logic
- Day 5: Breakout detection

### Week 3: Risk & Adjustments
- Day 1-2: Adjustment mechanics
- Day 3-4: Risk management
- Day 5: Integration testing

### Week 4: Validation
- Day 1-3: Backtest runs
- Day 4-5: Debugging and fixes

## Success Metrics

### Functional Metrics
- All test scenarios pass
- Accurate P&L calculation
- Correct position management
- Proper risk control execution

### Performance Metrics (Target)
- Sharpe Ratio > 1.0
- Max Drawdown < 10%
- Win Rate > 60%
- Average Monthly Return > 2%

## Notes

### Key Considerations
1. **Slippage**: Account for bid-ask spreads in backtesting
2. **Margin Requirements**: Use realistic margin calculations
3. **Transaction Costs**: Include brokerage and taxes
4. **Market Hours**: Use EOD prices only as specified
5. **Holidays**: Handle market holidays properly

### Data Requirements
- Historical stock prices (daily OHLC)
- Option chain data with strikes and premiums
- Expiry calendar
- Lot sizes for each stock
- NIFTY index values for hedge

### Risk Warnings
- Backtesting results may not reflect actual trading
- Consider impact costs and liquidity
- Account for assignment risk on short options
- Monitor margin requirements dynamically

---

*This plan will be updated as implementation progresses and new requirements emerge.*