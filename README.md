# SKAS Backtest Framework

A professional backtesting framework for NSE options strategies, integrated with the NSE options pricing library.

## Features

- **Real-time NSE options data integration** with Black-Scholes pricing
- **Multiple strategy implementations** with configurable parameters
- **Performance monitoring and optimization** with API call tracking
- **Comprehensive P&L tracking and metrics** with leg-level detail
- **Support for single and multi-stock backtesting** across 25 NSE stocks
- **Advanced risk management** with position sizing and stop-loss controls
- **Professional logging and reporting** with detailed position summaries

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd skas-backtest

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage
```bash
# Default: RELIANCE for 1 month
python run_middle_way_backtest.py

# Custom stock and date range
python run_middle_way_backtest.py --stocks RELIANCE --start-date 2024-01-01 --months 3
```

## CLI Options

### Stock Selection
- `--stocks SYMBOL` : Single stock (e.g., `RELIANCE`)
- `--stocks SYMBOL1,SYMBOL2` : Multiple stocks (comma-separated)
- `--stocks ALL` : All 25 available stocks

### Date Range Options
- `--start-date YYYY-MM-DD` : Start date (default: `2024-01-01`)
- `--end-date YYYY-MM-DD` : Specific end date
- `--months N` : Number of months from start date (default: `1`)

### Predefined Scenarios
- `--scenario quick` : 1 month test with single stock
- `--scenario monthly` : Current month backtest
- `--scenario quarterly` : 3 months backtest
- `--scenario yearly` : 12 months backtest
- `--scenario multi-stock` : Test with top 5 stocks

### Other Options
- `--log-level {DEBUG,INFO,WARNING,ERROR}` : Logging level (default: `INFO`)
- `--output-dir PATH` : Results directory (default: `results`)

## Examples

### Single Stock Backtesting
```bash
# RELIANCE for Q1 2024
python run_middle_way_backtest.py --stocks RELIANCE --start-date 2024-01-01 --end-date 2024-03-31

# TCS for 6 months with debug logging
python run_middle_way_backtest.py --stocks TCS --start-date 2024-01-01 --months 6 --log-level DEBUG

# HDFCBANK using predefined quarterly scenario
python run_middle_way_backtest.py --scenario quarterly --stocks HDFCBANK
```

### Multiple Stocks
```bash
# Top IT stocks for Q2 2024
python run_middle_way_backtest.py --stocks INFY,TCS,HCLTECH --start-date 2024-04-01 --months 3

# Banking stocks for 1 year
python run_middle_way_backtest.py --stocks HDFCBANK,ICICIBANK,SBIN,AXISBANK --start-date 2023-01-01 --months 12

# All available stocks for 6 months
python run_middle_way_backtest.py --stocks ALL --start-date 2024-01-01 --months 6
```

### Using Scenarios
```bash
# Quick test with default stock
python run_middle_way_backtest.py --scenario quick

# Multi-stock scenario with custom output directory
python run_middle_way_backtest.py --scenario multi-stock --output-dir custom_results

# Yearly backtest with warning-level logging
python run_middle_way_backtest.py --scenario yearly --stocks RELIANCE --log-level WARNING
```

## Available Stocks

The framework supports 25 liquid NSE stocks across major sectors:

**Banking & Finance:**
- HDFCBANK, ICICIBANK, SBIN, AXISBANK, KOTAKBANK, BAJFINANCE

**Information Technology:**
- INFY, TCS, HCLTECH

**Energy & Power:**
- RELIANCE, NTPC, POWERGRID

**Automotive:**
- MARUTI, TATAMOTORS, M&M

**FMCG & Consumer:**
- HINDUNILVR, ITC

**Diversified & Others:**
- LT, BHARTIARTL, ZOMATO, MARUTI, SUNPHARMA, ULTRACEMCO, TITAN, BEL, TATASTEEL

## Strategy: Middle Way

The framework implements the **Middle Way Strategy** - a systematic options selling approach:

### Strategy Logic
1. **Monthly Entry**: Enter short strangles at previous month's high/low levels
2. **Breakout Adjustments**: Convert to directional positions on price breakouts
3. **Risk Management**: Strict stop-loss and profit targets
4. **Leg-level Profit Booking**: Close profitable legs independently

### Key Parameters
- **Capital Allocation**: ₹2L per stock (configurable)
- **Position Sizing**: 1 lot per position (250-300 shares depending on stock)
- **Margin Requirement**: 25% of exposure (notional value)
- **Stop Loss**: 2% of total portfolio capital
- **Profit Target**: 80% of premium collected
- **Max Adjustments**: 2 per stock per month

## Output & Results

### File Format
Results are saved as: `middle_way_{STOCKS}_{TIMESTAMP}.json`

### Performance Metrics
The framework provides comprehensive performance analysis:

#### Return Metrics
- **Premium Retention**: % of collected premium retained
- **Return on Margin**: Return on capital deployed
- **Return on Exposure**: Return on total notional value

#### Monthly Performance (NEW FORMAT)
```
📅 MONTHLY PERFORMANCE
----------------------------------------
  2024-01: Monthly: -1.1% (₹-2,183) | Cumulative: -1.1% (₹-2,183)
  2024-02: Monthly: +6.4% (₹12,827) | Cumulative: +5.3% (₹10,644)
  2024-03: Monthly: +6.0% (₹12,609) | Cumulative: +11.6% (₹23,253)
```

#### Position Summary Example
```
📊 POSITION SUMMARY: RELIANCE (2024-02-01 → 2024-02-26)
================================================================================
Position Type: BULLISH | Days Held: 25
Entry Spot: ₹2853.30 | Exit Spot: ₹2974.65
├─ Premium Collected: ₹18,596
├─ Margin Deployed: ₹183,138
├─ Exposure: ₹713,325
├─ Total P&L: ₹11,816
├─ Premium Retention: +63.5% (kept ₹11,816 of ₹18,596)
├─ Return on Margin: +6.5%
└─ Return on Exposure: +1.7% | Status: 🎉 WIN
```

## Advanced Features

### API Performance Monitoring
The framework tracks NSE API calls for optimization:
```
API PERFORMANCE SUMMARY
────────────────────────────────────────
API Call                       Count    Total(s)  Cache Hit%
get_lot_size                   5        1.25      0.0
option_calculator.calculate_premium 23  0.42      0.0
Total API Time: 2.87s | Overall Cache Hit Rate: 74.0%
```

### Risk Management Features
- **Portfolio-level stop loss** (2% of total capital)
- **Position-level profit targets** (80% of premium collected)
- **Leg-level profit booking** (80% of individual leg premium)
- **Maximum position limits** (configurable per strategy)
- **Breakout-based adjustments** with 1% buffer

### Logging Levels
- **DEBUG**: Detailed trade execution and position tracking
- **INFO**: Standard operation with key trade and P&L information
- **WARNING**: Only warnings and errors
- **ERROR**: Critical errors only

## Configuration

### Strategy Parameters
Modify `run_middle_way_backtest.py` to customize:

```python
strategy_config = MiddleWayConfig(
    # Risk parameters
    portfolio_stop_loss_pct=0.02,        # 2% of total capital
    position_profit_target_pct=0.80,     # 80% of premium collected
    leg_profit_target_pct=0.80,          # 80% profit target for individual legs
    
    # Adjustment controls
    adjustment_trigger_buffer_pct=0.01,  # 1% buffer for adjustments
    max_adjustments_per_stock=2,         # Maximum 2 adjustments per position
    
    # Position sizing
    lots_per_stock_position=1,           # 1 lot per position
    margin_percentage=0.25,              # 25% of exposure as margin
    
    # Exit timing
    exit_days_before_expiry=1,           # Exit T-1 day before expiry
)
```

### Capital Allocation
- **Single Stock**: Uses full `capital_per_stock` (₹2L default)
- **Multiple Stocks**: Divides `total_capital` among stocks
- **Maximum**: ₹2L per stock regardless of total capital

## Development

### Project Structure
```
skas-backtest/
├── skas_backtest/
│   ├── core/              # Core backtesting engine
│   │   ├── backtest.py    # Main backtest orchestration
│   │   ├── portfolio.py   # Position and portfolio management
│   │   └── metrics.py     # Performance calculation and reporting
│   ├── strategies/        # Strategy implementations
│   │   └── middle_way_strategy.py  # Middle Way strategy logic
│   ├── data/             # Data management and NSE integration
│   │   └── data_manager.py        # NSE options data interface
│   └── utils/            # Utility functions
│       └── performance_monitor.py  # API performance tracking
├── run_middle_way_backtest.py      # Main CLI entry point
├── results/              # Backtest output files
└── logs/                # Detailed execution logs
```

### Testing
```bash
# Quick validation test
python run_middle_way_backtest.py --scenario quick --log-level DEBUG

# Multi-stock performance test
python run_middle_way_backtest.py --scenario multi-stock --log-level INFO

# Custom date range test
python run_middle_way_backtest.py --stocks RELIANCE,INFY --start-date 2024-01-01 --months 6
```

## Requirements

- Python 3.8+
- NSE Options Pricing Library
- pandas, numpy, logging
- Access to NSE historical and options data

## License

Proprietary - All rights reserved

---

**Note**: This framework requires access to the NSE options pricing library and historical data. Ensure proper data connectivity before running backtests.