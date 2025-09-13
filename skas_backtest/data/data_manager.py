"""
Enhanced Data Manager for NSE Options Backtesting.

This module extends the base DataManager to integrate with the NSE options pricing library,
providing historical stock prices, option chains, and utilities for the Middle Way strategy.
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from abc import abstractmethod

# Import NSE options library
try:
    from nse_options import OptionCalculator, get_lot_size, get_monthly_expiry, get_next_monthly_expiry
    from nse_options.data import NSEDataFetcher
    from nse_options.utils import get_atm_strike, get_typical_strikes
except ImportError as e:
    raise ImportError(f"NSE options library not available: {e}")

# Import performance monitoring
from ..utils.performance_monitor import PerformanceMonitor


@dataclass
class OptionQuote:
    """Structure for option price data."""
    symbol: str
    strike: float
    expiry: date
    option_type: str  # "call" or "put"
    ltp: float
    bid: float
    ask: float
    volume: int
    oi: int
    iv: Optional[float] = None


@dataclass
class MarketData:
    """Structure for market data."""
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class DataManager:
    """
    Enhanced Data Manager integrating with NSE options pricing library.
    
    Provides historical stock prices, option chains, and utilities needed
    for backtesting the Middle Way strategy.
    """
    
    def __init__(
        self,
        cache_enabled: bool = True,
        cache_size: int = 1000,
        data_source: str = "nse_options",
        enable_performance_monitoring: bool = True
    ):
        """
        Initialize the Data Manager.
        
        Args:
            cache_enabled: Enable data caching for performance
            cache_size: Maximum number of cached items
            data_source: Data source to use ("nse_options")
            enable_performance_monitoring: Enable API performance monitoring
        """
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size
        self.data_source = data_source
        
        # Initialize NSE components
        self.option_calculator = OptionCalculator()
        self.nse_fetcher = NSEDataFetcher()
        
        # Cache dictionaries
        self._price_cache: Dict[str, Any] = {}
        self._option_cache: Dict[str, Any] = {}
        self._calendar_cache: Dict[str, List[date]] = {}
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor() if enable_performance_monitoring else None
        
        # Logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def get_stock_price(self, symbol: str, target_date: date) -> Optional[float]:
        """
        Get stock closing price for a specific date.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            target_date: Date for which to get price
            
        Returns:
            Closing price if available, None otherwise
        """
        cache_key = f"{symbol}_{target_date}"
        
        if self.cache_enabled and cache_key in self._price_cache:
            # Record cache hit
            if self.perf_monitor:
                self.perf_monitor.record_cache_hit('get_stock_price')
            return self._price_cache[cache_key]
        
        try:
            # Use NSE data fetcher to get EOD price with performance monitoring
            context = f"{symbol} {target_date}"
            if self.perf_monitor:
                with self.perf_monitor.track_api_call('nse_fetcher.get_eod_price', context=context):
                    close_price = self.nse_fetcher.get_eod_price(symbol, target_date)
            else:
                close_price = self.nse_fetcher.get_eod_price(symbol, target_date)
            
            if close_price is not None:
                if self.cache_enabled:
                    self._price_cache[cache_key] = close_price
                    self._trim_cache(self._price_cache)
                
                return close_price
            
            self.logger.warning(f"No price data found for {symbol} on {target_date}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching price for {symbol} on {target_date}: {e}")
            return None
    
    def get_market_data(self, symbol: str, target_date: date) -> Optional[MarketData]:
        """
        Get complete market data (OHLC + Volume) for a specific date.
        
        Args:
            symbol: Stock symbol
            target_date: Date for market data
            
        Returns:
            MarketData object if available, None otherwise
        """
        cache_key = f"md_{symbol}_{target_date}"
        
        if self.cache_enabled and cache_key in self._price_cache:
            # Record cache hit
            if self.perf_monitor:
                self.perf_monitor.record_cache_hit('get_market_data')
            return self._price_cache[cache_key]
        
        try:
            # Fetch OHLC data using historical data method with performance monitoring
            context = f"{symbol} {target_date}"
            if self.perf_monitor:
                with self.perf_monitor.track_api_call('nse_fetcher.get_historical_data', context=context):
                    price_data = self.nse_fetcher.get_historical_data(symbol, target_date, target_date, "EQ")
            else:
                price_data = self.nse_fetcher.get_historical_data(symbol, target_date, target_date, "EQ")
            
            if price_data is not None and not price_data.empty:
                row = price_data.iloc[-1]
                
                # Handle column names that may vary
                close_price = row.get('CLOSE', row.get('close', None))
                open_price = row.get('OPEN', row.get('open', close_price))
                high_price = row.get('HIGH', row.get('high', close_price))
                low_price = row.get('LOW', row.get('low', close_price))
                volume = row.get('VOLUME', row.get('volume', 0))
                
                if close_price is not None:
                    market_data = MarketData(
                        symbol=symbol,
                        date=target_date,
                        open=float(open_price),
                        high=float(high_price),
                        low=float(low_price),
                        close=float(close_price),
                        volume=int(volume) if volume else 0
                    )
                    
                    if self.cache_enabled:
                        self._price_cache[cache_key] = market_data
                        self._trim_cache(self._price_cache)
                    
                    return market_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol} on {target_date}: {e}")
            return None
    
    def get_option_price(
        self,
        symbol: str,
        strike: float,
        expiry: date,
        option_type: str,
        as_of_date: date
    ) -> Optional[float]:
        """
        Get option price using NSE options pricing library.
        
        Args:
            symbol: Underlying stock symbol
            strike: Strike price
            expiry: Expiry date
            option_type: "call" or "put"
            as_of_date: Date for valuation
            
        Returns:
            Option price if available, None otherwise
        """
        cache_key = f"{symbol}_{strike}_{expiry}_{option_type}_{as_of_date}"
        
        if self.cache_enabled and cache_key in self._option_cache:
            # Record cache hit
            if self.perf_monitor:
                self.perf_monitor.record_cache_hit('get_option_price')
            return self._option_cache[cache_key]
        
        try:
            # Convert option_type to string if it's an enum
            option_type_str = option_type.value if hasattr(option_type, 'value') else str(option_type).lower()
            
            # Use NSE options calculator with performance monitoring
            context = f"{symbol} {strike}{option_type_str[0].upper()}E {as_of_date}"
            if self.perf_monitor:
                with self.perf_monitor.track_api_call('option_calculator.calculate_premium', context=context):
                    result = self.option_calculator.calculate_premium(
                        symbol=symbol,
                        strike=strike,
                        as_of_date=as_of_date.strftime('%Y-%m-%d'),
                        option_type=option_type_str,
                        expiry=expiry.strftime('%Y-%m-%d')
                    )
            else:
                result = self.option_calculator.calculate_premium(
                    symbol=symbol,
                    strike=strike,
                    as_of_date=as_of_date.strftime('%Y-%m-%d'),
                    option_type=option_type_str,
                    expiry=expiry.strftime('%Y-%m-%d')
                )
            
            if result and 'theoretical_price' in result:
                price = result['theoretical_price']
                
                if self.cache_enabled:
                    self._option_cache[cache_key] = price
                    self._trim_cache(self._option_cache)
                
                return price
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating option price: {e}")
            return None
    
    def get_option_chain(
        self,
        symbol: str,
        expiry: date,
        as_of_date: date,
        strike_range: Optional[Tuple[float, float]] = None
    ) -> List[OptionQuote]:
        """
        Get option chain for a specific expiry.
        
        Args:
            symbol: Underlying stock symbol
            expiry: Expiry date
            as_of_date: Date for option chain
            strike_range: Optional (min_strike, max_strike) filter
            
        Returns:
            List of OptionQuote objects
        """
        cache_key = f"chain_{symbol}_{expiry}_{as_of_date}_{strike_range}"
        
        if self.cache_enabled and cache_key in self._option_cache:
            return self._option_cache[cache_key]
        
        try:
            # Get spot price
            spot_price = self.get_stock_price(symbol, as_of_date)
            if not spot_price:
                return []
            
            # Generate typical strikes around spot with performance monitoring
            context = f"{symbol} {spot_price}"
            if self.perf_monitor:
                with self.perf_monitor.track_api_call('get_typical_strikes', context=context):
                    strikes = get_typical_strikes(symbol, spot_price)
            else:
                strikes = get_typical_strikes(symbol, spot_price)
            
            # Filter by range if provided
            if strike_range:
                min_strike, max_strike = strike_range
                strikes = [s for s in strikes if min_strike <= s <= max_strike]
            
            option_quotes = []
            
            for strike in strikes:
                for option_type in ["call", "put"]:
                    price = self.get_option_price(symbol, strike, expiry, option_type, as_of_date)
                    
                    if price is not None:
                        quote = OptionQuote(
                            symbol=symbol,
                            strike=strike,
                            expiry=expiry,
                            option_type=option_type,
                            ltp=price,
                            bid=price * 0.99,  # Approximate bid-ask spread
                            ask=price * 1.01,
                            volume=0,  # Not available in historical data
                            oi=0,      # Not available in historical data
                        )
                        option_quotes.append(quote)
            
            if self.cache_enabled:
                self._option_cache[cache_key] = option_quotes
                self._trim_cache(self._option_cache)
            
            return option_quotes
            
        except Exception as e:
            self.logger.error(f"Error fetching option chain: {e}")
            return []
    
    def get_trading_dates(self, start_date: date, end_date: date) -> List[date]:
        """
        Get list of trading dates between start and end dates.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of trading dates (excluding weekends and holidays)
        """
        cache_key = f"trading_dates_{start_date}_{end_date}"
        
        if self.cache_enabled and cache_key in self._calendar_cache:
            return self._calendar_cache[cache_key]
        
        try:
            # Generate date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Filter out weekends (Saturday=5, Sunday=6)
            trading_dates = [
                d.date() for d in date_range 
                if d.weekday() < 5  # Monday=0, Friday=4
            ]
            
            # TODO: Add NSE holiday calendar filtering
            # For now, we'll use weekdays only
            
            if self.cache_enabled:
                self._calendar_cache[cache_key] = trading_dates
                self._trim_cache(self._calendar_cache)
            
            return trading_dates
            
        except Exception as e:
            self.logger.error(f"Error getting trading dates: {e}")
            return []
    
    def get_monthly_expiry(self, year: int, month: int) -> Optional[date]:
        """
        Get monthly expiry date for given year and month.
        
        Args:
            year: Year
            month: Month (1-12)
            
        Returns:
            Expiry date (last Thursday of month)
        """
        try:
            # Add performance monitoring for get_monthly_expiry
            context = f"{year}-{month:02d}"
            if self.perf_monitor:
                with self.perf_monitor.track_api_call('get_monthly_expiry', context=context):
                    return get_monthly_expiry(year, month)
            else:
                return get_monthly_expiry(year, month)
        except Exception as e:
            self.logger.error(f"Error getting monthly expiry for {year}-{month}: {e}")
            return None
    
    def get_next_monthly_expiry(self, reference_date: date) -> Optional[date]:
        """
        Get next monthly expiry after reference date.
        
        Args:
            reference_date: Reference date
            
        Returns:
            Next monthly expiry date
        """
        try:
            # Add performance monitoring for get_next_monthly_expiry
            context = f"{reference_date}"
            if self.perf_monitor:
                with self.perf_monitor.track_api_call('get_next_monthly_expiry', context=context):
                    return get_next_monthly_expiry(reference_date)
            else:
                return get_next_monthly_expiry(reference_date)
        except Exception as e:
            self.logger.error(f"Error getting next monthly expiry after {reference_date}: {e}")
            return None
    
    def get_lot_size(self, symbol: str, as_of_date: date) -> int:
        """
        Get lot size for a symbol on a specific date.
        
        Args:
            symbol: Stock symbol
            as_of_date: Date for lot size lookup
            
        Returns:
            Lot size (returns 1 as default for invalid symbols)
        """
        try:
            # Get lot size with performance monitoring
            context = f"{symbol} {as_of_date}"
            if self.perf_monitor:
                with self.perf_monitor.track_api_call('get_lot_size', context=context):
                    lot_size = get_lot_size(symbol, as_of_date)
            else:
                lot_size = get_lot_size(symbol, as_of_date)
                
            if lot_size is None:
                self.logger.warning(f"No lot size found for {symbol}, using default of 1")
                return 1  # Default fallback for invalid symbols
            return lot_size
        except Exception as e:
            self.logger.error(f"Error getting lot size for {symbol}: {e}")
            return 1  # Default fallback
    
    def find_closest_strike(
        self,
        symbol: str,
        target_price: float,
        expiry: date,
        as_of_date: date
    ) -> Optional[float]:
        """
        Find the closest available strike to target price.
        
        Args:
            symbol: Underlying stock symbol
            target_price: Target price to match
            expiry: Option expiry date
            as_of_date: Date for strike lookup
            
        Returns:
            Closest available strike price
        """
        try:
            # Add performance monitoring for get_typical_strikes
            context = f"{symbol} {target_price}"
            if self.perf_monitor:
                with self.perf_monitor.track_api_call('get_typical_strikes', context=context):
                    strikes = get_typical_strikes(symbol, target_price)
            else:
                strikes = get_typical_strikes(symbol, target_price)
            
            if not strikes:
                return None
            
            # Find closest strike
            closest_strike = min(strikes, key=lambda x: abs(x - target_price))
            return closest_strike
            
        except Exception as e:
            self.logger.error(f"Error finding closest strike: {e}")
            return None
    
    def get_atm_strike(self, symbol: str, expiry: date, as_of_date: date) -> Optional[float]:
        """
        Get At-The-Money strike for given symbol and date.
        
        Args:
            symbol: Stock symbol
            expiry: Option expiry date
            as_of_date: Date for ATM calculation
            
        Returns:
            ATM strike price
        """
        try:
            spot_price = self.get_stock_price(symbol, as_of_date)
            if not spot_price:
                return None
            
            # Add performance monitoring for get_atm_strike
            context = f"{symbol} {spot_price}"
            if self.perf_monitor:
                with self.perf_monitor.track_api_call('get_atm_strike', context=context):
                    return get_atm_strike(symbol, spot_price)
            else:
                return get_atm_strike(symbol, spot_price)
            
        except Exception as e:
            self.logger.error(f"Error getting ATM strike: {e}")
            return None
    
    def get_otm_strikes(
        self,
        symbol: str,
        spot_price: float,
        otm_percentage: float,
        expiry: date,
        as_of_date: date
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Get Out-of-The-Money strikes for calls and puts.
        
        Args:
            symbol: Stock symbol
            spot_price: Current spot price
            otm_percentage: Percentage OTM (e.g., 0.04 for 4%)
            expiry: Option expiry date
            as_of_date: Date for calculation
            
        Returns:
            Tuple of (call_strike, put_strike) - both OTM
        """
        try:
            # Calculate target prices
            call_target = spot_price * (1 + otm_percentage)
            put_target = spot_price * (1 - otm_percentage)
            
            # Find closest strikes
            call_strike = self.find_closest_strike(symbol, call_target, expiry, as_of_date)
            put_strike = self.find_closest_strike(symbol, put_target, expiry, as_of_date)
            
            return call_strike, put_strike
            
        except Exception as e:
            self.logger.error(f"Error getting OTM strikes: {e}")
            return None, None
    
    def get_month_high_low(self, symbol: str, year: int, month: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Get highest high and lowest low for a given month.
        
        Args:
            symbol: Stock symbol
            year: Year
            month: Month (1-12)
            
        Returns:
            Tuple of (highest_high, lowest_low)
        """
        try:
            # Get first and last day of month
            first_day = date(year, month, 1)
            
            if month == 12:
                last_day = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                last_day = date(year, month + 1, 1) - timedelta(days=1)
            
            # Get trading dates for the month
            trading_dates = self.get_trading_dates(first_day, last_day)
            
            if not trading_dates:
                return None, None
            
            highs = []
            lows = []
            
            for trading_date in trading_dates:
                market_data = self.get_market_data(symbol, trading_date)
                if market_data:
                    highs.append(market_data.high)
                    lows.append(market_data.low)
            
            if not highs or not lows:
                return None, None
            
            return max(highs), min(lows)
            
        except Exception as e:
            self.logger.error(f"Error getting month high/low: {e}")
            return None, None
    
    def get_previous_month_range(self, symbol: str, current_date: date) -> Tuple[Optional[float], Optional[float]]:
        """
        Get previous month's high and low relative to current date.
        
        Args:
            symbol: Stock symbol
            current_date: Current date
            
        Returns:
            Tuple of (previous_month_high, previous_month_low)
        """
        try:
            # Calculate previous month
            if current_date.month == 1:
                prev_year = current_date.year - 1
                prev_month = 12
            else:
                prev_year = current_date.year
                prev_month = current_date.month - 1
            
            return self.get_month_high_low(symbol, prev_year, prev_month)
            
        except Exception as e:
            self.logger.error(f"Error getting previous month range: {e}")
            return None, None
    
    def get_historical_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Get historical price data for a date range.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Use NSE fetcher to get historical data directly with performance monitoring
            context = f"{symbol} {start_date} to {end_date}"
            if self.perf_monitor:
                with self.perf_monitor.track_api_call('nse_fetcher.get_historical_data', context=context):
                    price_data = self.nse_fetcher.get_historical_data(symbol, start_date, end_date, "EQ")
            else:
                price_data = self.nse_fetcher.get_historical_data(symbol, start_date, end_date, "EQ")
            
            if price_data is not None and not price_data.empty:
                # Standardize column names
                columns_map = {
                    'OPEN': 'open',
                    'HIGH': 'high', 
                    'LOW': 'low',
                    'CLOSE': 'close',
                    'VOLUME': 'volume'
                }
                
                # Rename columns if they exist
                for old_col, new_col in columns_map.items():
                    if old_col in price_data.columns:
                        price_data.rename(columns={old_col: new_col}, inplace=True)
                
                # Ensure we have required columns
                required_cols = ['open', 'high', 'low', 'close']
                available_cols = [col for col in required_cols if col in price_data.columns]
                
                if available_cols:
                    # Select and return available columns
                    result_cols = available_cols + (['volume'] if 'volume' in price_data.columns else [])
                    return price_data[result_cols].copy()
                else:
                    self.logger.warning(f"No OHLC data found for {symbol}")
                    return pd.DataFrame()
            
            return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error getting historical prices: {e}")
            return pd.DataFrame()
    
    def _trim_cache(self, cache_dict: Dict[str, Any]) -> None:
        """Trim cache to maximum size."""
        if len(cache_dict) > self.cache_size:
            # Remove oldest entries (simple FIFO)
            items_to_remove = len(cache_dict) - self.cache_size
            keys_to_remove = list(cache_dict.keys())[:items_to_remove]
            for key in keys_to_remove:
                del cache_dict[key]
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._price_cache.clear()
        self._option_cache.clear()
        self._calendar_cache.clear()
        self.logger.info("All caches cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "price_cache_size": len(self._price_cache),
            "option_cache_size": len(self._option_cache),
            "calendar_cache_size": len(self._calendar_cache),
            "total_cache_items": len(self._price_cache) + len(self._option_cache) + len(self._calendar_cache)
        }
    
    def get_portfolio_option_prices(self, positions: List, target_date: date) -> Dict[str, float]:
        """
        Get option prices for all positions in a portfolio.
        
        Args:
            positions: List of Position objects
            target_date: Date for pricing
            
        Returns:
            Dictionary with option keys and their current prices
        """
        option_prices = {}
        
        for position in positions:
            option_key = f"{position.symbol}_{position.strike}_{position.expiry}_{position.option_type}"
            
            try:
                price = self.get_option_price(
                    symbol=position.symbol,
                    strike=position.strike,
                    expiry=position.expiry,
                    option_type=position.option_type,
                    as_of_date=target_date
                )
                
                if price is not None:
                    option_prices[option_key] = price
                    
            except Exception as e:
                self.logger.warning(f"Could not get price for {option_key} on {target_date}: {e}")
                continue
        
        return option_prices