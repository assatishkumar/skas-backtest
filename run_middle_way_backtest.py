#!/usr/bin/env python3
"""
Middle Way Strategy Backtest Runner

This script runs a comprehensive backtest of the Middle Way strategy with configurable
parameters for stocks, date ranges, and scenarios.

Usage:
    python run_middle_way_backtest.py --stocks RELIANCE --start-date 2024-01-01 --months 3
    python run_middle_way_backtest.py --stocks RELIANCE,INFY,TCS --start-date 2024-01-01 --end-date 2024-06-30
    python run_middle_way_backtest.py --scenario quarterly --stocks HDFCBANK
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from skas_backtest.core.backtest import Backtest, BacktestConfig
from skas_backtest.strategies.middle_way_strategy import MiddleWayStrategy, MiddleWayConfig
from skas_backtest.data.data_manager import DataManager

# Available stocks for the Middle Way strategy
AVAILABLE_STOCKS = [
    "HDFCBANK", "ICICIBANK", "RELIANCE", "INFY", "BHARTIARTL",
    "LT", "ITC", "TCS", "SBIN", "AXISBANK", "KOTAKBANK", "M&M",
    "HINDUNILVR", "BAJFINANCE", "ZOMATO", "MARUTI", "SUNPHARMA",
    "NTPC", "HCLTECH", "ULTRACEMCO", "TITAN", "TATAMOTORS",
    "BEL", "TATASTEEL", "POWERGRID"
]

def calculate_end_date(start_date: str, months: int) -> str:
    """Calculate end date from start date plus number of months."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    # Add months (roughly 30 days per month for trading purposes)
    end_dt = start_dt + timedelta(days=months * 30)
    return end_dt.strftime("%Y-%m-%d")

def parse_stock_list(stocks_str: str) -> List[str]:
    """Parse comma-separated stock string and validate against available stocks."""
    if stocks_str.upper() == "ALL":
        return AVAILABLE_STOCKS.copy()
    
    stocks = [stock.strip().upper() for stock in stocks_str.split(",")]
    
    # Validate stocks
    invalid_stocks = [stock for stock in stocks if stock not in AVAILABLE_STOCKS]
    if invalid_stocks:
        raise ValueError(f"Invalid stocks: {invalid_stocks}. Available stocks: {AVAILABLE_STOCKS}")
    
    return stocks

def get_predefined_scenario(scenario_name: str, stocks: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return predefined test scenarios."""
    base_date = datetime.now().strftime("%Y-%m-%d")
    default_stocks = stocks or ["RELIANCE"]
    
    scenarios = {
        "quick": {
            "name": "Quick Test (1 Month)",
            "start_date": "2024-01-01",
            "months": 1,
            "stocks": default_stocks[:1]  # Single stock
        },
        "monthly": {
            "name": "Monthly Test",
            "start_date": "2024-01-01", 
            "months": 1,
            "stocks": default_stocks
        },
        "quarterly": {
            "name": "Quarterly Test (3 Months)",
            "start_date": "2024-01-01",
            "months": 3,
            "stocks": default_stocks
        },
        "yearly": {
            "name": "Yearly Test (12 Months)",
            "start_date": "2023-01-01",
            "months": 12,
            "stocks": default_stocks
        },
        "multi-stock": {
            "name": "Multi-Stock Test (Top 5)",
            "start_date": "2024-01-01",
            "months": 3,
            "stocks": ["RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK"]
        }
    }
    
    if scenario_name not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(scenarios.keys())}")
    
    scenario = scenarios[scenario_name].copy()
    scenario["end_date"] = calculate_end_date(scenario["start_date"], scenario["months"])
    
    return scenario

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up comprehensive logging for the backtest."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Setup logging configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"middle_way_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("MiddleWayBacktest")
    return logger

def create_strategy_config(stocks: List[str]) -> MiddleWayConfig:
    """Create strategy configuration for given stocks."""
    
    # Calculate capital allocation based on number of stocks
    total_capital = 5000000  # 50 lakhs base capital
    num_stocks = len(stocks)
    capital_per_stock = min(total_capital // num_stocks, 200000)  # Max 2 lakhs per stock
    
    config = MiddleWayConfig(
        # Primary stock (first in list for single-stock mode)
        test_stock=stocks[0] if stocks else "RELIANCE",
        
        # Full stock list for future multi-stock implementation
        stock_list=stocks,
        
        # Capital allocation based on number of stocks
        total_capital=total_capital,
        capital_per_stock=capital_per_stock,
        
        # Risk parameters
        portfolio_stop_loss_pct=0.02,  # 2% of total capital
        position_profit_target_pct=0.80,  # 80% of premium collected
        
        # Adjustment constraints
        max_adjustments_per_stock=2,
        min_days_between_adjustments=1,
        
        # Exit timing
        exit_days_before_expiry=1,
        
        # Configurable features
        enable_leg_profit_booking=True,
        leg_profit_target_pct=0.80,  # 80% profit target for individual legs
        adjustment_trigger_buffer_pct=0.01,  # 1% buffer for early adjustment triggers
        min_leg_premium_pct=0.05,  # Skip leg if premium < 5% of total
        skip_low_premium_legs=True,  # Enable skipping low premium legs
        
        # Position sizing
        lots_per_stock_position=1,  # Trade 1 lot per stock position
        lots_per_hedge_position=1,   # Trade 1 lot for hedge positions
        
        # Margin configuration
        margin_percentage=0.25  # 25% of exposure (notional value)
    )
    
    return config

def create_backtest_config(stocks: List[str]) -> BacktestConfig:
    """Create backtest configuration based on number of stocks."""
    
    # Calculate initial capital based on number of stocks
    base_capital_per_stock = 200000  # 2 lakhs per stock
    total_capital = len(stocks) * base_capital_per_stock
    
    config = BacktestConfig(
        initial_capital=total_capital,
        commission_per_trade=20.0,  # ₹20 per trade
        commission_rate=0.0005,  # 0.05% of trade value
        enable_risk_checks=True,
        slippage_bps=2,  # 2 basis points
        max_positions=len(stocks) * 5  # 5 positions per stock max
    )
    
    return config

def run_validation_backtest(
    start_date: str = "2024-01-01",
    end_date: str = "2024-03-31",
    stocks: List[str] = None,
    log_level: str = "INFO",
    output_dir: str = "results"
) -> None:
    """
    Run validation backtest for Middle Way strategy.
    
    Args:
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        stocks: List of stock symbols to test
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        output_dir: Directory for results output
    """
    
    if stocks is None:
        stocks = ["RELIANCE"]
    
    logger = setup_logging(log_level)
    logger.info("="*80)
    logger.info("MIDDLE WAY STRATEGY PARAMETRIZED BACKTEST")
    logger.info("="*80)
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Stocks: {', '.join(stocks)} ({len(stocks)} stock{'s' if len(stocks) > 1 else ''})")
    logger.info("="*80)
    
    try:
        # Create strategy and backtest configurations
        strategy_config = create_strategy_config(stocks)
        backtest_config = create_backtest_config(stocks)
        
        logger.info("Configuration:")
        logger.info(f"  Initial Capital: ₹{backtest_config.initial_capital:,}")
        logger.info(f"  Capital per Stock: ₹{strategy_config.capital_per_stock:,}")
        logger.info(f"  Portfolio Stop Loss: {strategy_config.portfolio_stop_loss_pct:.1%}")
        logger.info(f"  Position Profit Target: {strategy_config.position_profit_target_pct:.1%}")
        logger.info(f"  Leg Profit Target: {strategy_config.leg_profit_target_pct:.1%}")
        logger.info(f"  Adjustment Buffer: {strategy_config.adjustment_trigger_buffer_pct:.1%}")
        logger.info(f"  Max Adjustments: {strategy_config.max_adjustments_per_stock}")
        logger.info("")
        
        # Initialize components
        logger.info("Initializing backtest components...")
        
        # Create strategy
        strategy = MiddleWayStrategy(strategy_config)
        logger.info(f"✓ Strategy initialized: {strategy.name}")
        
        # Create data manager
        data_manager = DataManager()
        logger.info("✓ Data manager initialized")
        
        # Create backtest engine
        backtest = Backtest(
            strategy=strategy,
            config=backtest_config,
            data_manager=data_manager
        )
        logger.info("✓ Backtest engine initialized")
        logger.info("")
        
        # Progress callback
        def progress_callback(progress: float, current_date: date):
            if progress * 100 % 10 == 0:  # Log every 10%
                logger.info(f"Progress: {progress:.1%} - {current_date}")
        
        # Run backtest
        logger.info("Starting backtest execution...")
        start_time = datetime.now()
        
        results = backtest.run(
            start_date=start_date,
            end_date=end_date,
            symbols=stocks,
            progress_callback=progress_callback
        )
        
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        logger.info(f"✓ Backtest completed in {execution_time}")
        logger.info("")
        
        # Display results using new formatted summary
        logger.info("\n" + results.summary())
        
        # Save detailed results
        results_dir = Path(output_dir)
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stocks_suffix = "_".join(stocks[:3])  # Use first 3 stocks in filename
        if len(stocks) > 3:
            stocks_suffix += f"_plus{len(stocks)-3}"
        
        results_file = results_dir / f"middle_way_{stocks_suffix}_{timestamp}.json"
        
        logger.info(f"Results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Backtest failed with error: {e}")
        logger.exception("Full traceback:")
        raise

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Middle Way Strategy Backtest with configurable parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single stock for 3 months
  python run_middle_way_backtest.py --stocks RELIANCE --start-date 2024-01-01 --months 3

  # Multiple stocks for specific period  
  python run_middle_way_backtest.py --stocks RELIANCE,INFY,TCS --start-date 2024-01-01 --end-date 2024-06-30

  # Use predefined scenario
  python run_middle_way_backtest.py --scenario quarterly --stocks HDFCBANK

  # All available stocks
  python run_middle_way_backtest.py --stocks ALL --start-date 2024-01-01 --months 6
        """
    )
    
    # Stock selection
    parser.add_argument(
        "--stocks", 
        type=str, 
        default="RELIANCE",
        help="Comma-separated stock symbols or 'ALL' for all available stocks (default: RELIANCE)"
    )
    
    # Date range options (mutually exclusive groups)
    date_group = parser.add_mutually_exclusive_group(required=False)
    date_group.add_argument(
        "--scenario",
        choices=["quick", "monthly", "quarterly", "yearly", "multi-stock"],
        help="Use predefined scenario (overrides other date options)"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date in YYYY-MM-DD format (default: 2024-01-01)"
    )
    
    end_group = parser.add_mutually_exclusive_group(required=False)
    end_group.add_argument(
        "--end-date",
        type=str,
        help="End date in YYYY-MM-DD format"
    )
    end_group.add_argument(
        "--months",
        type=int,
        default=1,
        help="Number of months to backtest from start date (default: 1)"
    )
    
    # Other options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for results output (default: results)"
    )
    
    return parser.parse_args()

def main():
    """Main function to run the parametrized backtest."""
    
    try:
        args = parse_arguments()
        
        # Handle scenario-based execution
        if args.scenario:
            stocks_list = None
            if args.stocks and args.stocks.upper() != "RELIANCE":
                stocks_list = parse_stock_list(args.stocks)
            
            scenario = get_predefined_scenario(args.scenario, stocks_list)
            
            print(f"\n🚀 Running predefined scenario: {scenario['name']}")
            print(f"   Period: {scenario['start_date']} to {scenario['end_date']}")
            print(f"   Stocks: {', '.join(scenario['stocks'])}")
            
            run_validation_backtest(
                start_date=scenario['start_date'],
                end_date=scenario['end_date'],
                stocks=scenario['stocks'],
                log_level=args.log_level,
                output_dir=args.output_dir
            )
            
        else:
            # Handle custom parameters
            stocks = parse_stock_list(args.stocks)
            
            # Calculate end date
            if args.end_date:
                end_date = args.end_date
            else:
                end_date = calculate_end_date(args.start_date, args.months)
            
            print(f"\n🚀 Running custom backtest configuration:")
            print(f"   Period: {args.start_date} to {end_date}")
            print(f"   Stocks: {', '.join(stocks)} ({len(stocks)} stock{'s' if len(stocks) > 1 else ''})")
            print(f"   Log Level: {args.log_level}")
            
            run_validation_backtest(
                start_date=args.start_date,
                end_date=end_date,
                stocks=stocks,
                log_level=args.log_level,
                output_dir=args.output_dir
            )
        
        print(f"✅ Backtest completed successfully")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()