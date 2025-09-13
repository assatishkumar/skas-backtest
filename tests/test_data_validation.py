"""
Data validation tests for NSE Data Manager.

These tests verify that the DataManager fetches accurate, realistic data
and maintains data quality standards. They use known historical values
for verification.
"""

import pytest
from datetime import date, timedelta
from skas_backtest.data import DataManager, MarketData


class TestKnownHistoricalData:
    """Test against known historical data points."""
    
    @pytest.fixture
    def data_manager(self):
        """DataManager for real data testing."""
        return DataManager(cache_enabled=False)  # Disable cache for fresh data
    
    def test_reliance_jan_15_2024(self, data_manager):
        """Test RELIANCE data for a specific known date."""
        test_date = date(2024, 1, 15)
        symbol = "RELIANCE"
        
        # Test stock price
        price = data_manager.get_stock_price(symbol, test_date)
        assert price is not None, f"No price data for {symbol} on {test_date}"
        assert isinstance(price, (int, float)), "Price should be numeric"
        
        # Known value verification
        expected_close = 2788.25
        assert price == expected_close, f"Expected {expected_close}, got {price}"
        
        # Test market data
        market_data = data_manager.get_market_data(symbol, test_date)
        assert market_data is not None, f"No market data for {symbol} on {test_date}"
        assert market_data.close == expected_close
        assert market_data.open == 2750.0
        assert market_data.high == 2792.9
        assert market_data.low == 2732.0
    
    def test_monthly_expiry_dates(self, data_manager):
        """Test known monthly expiry dates."""
        test_cases = [
            (2024, 1, date(2024, 1, 25)),  # January 2024 expiry
            (2024, 2, date(2024, 2, 29)),  # February 2024 expiry (leap year)
            (2024, 3, date(2024, 3, 28)),  # March 2024 expiry
            (2024, 12, date(2024, 12, 26)), # December 2024 expiry
        ]
        
        for year, month, expected_expiry in test_cases:
            expiry = data_manager.get_monthly_expiry(year, month)
            assert expiry == expected_expiry, f"Expected {expected_expiry} for {year}-{month}, got {expiry}"
            assert expiry.weekday() == 3, f"Expiry {expiry} should be Thursday"
    
    def test_known_lot_sizes(self, data_manager):
        """Test known lot sizes as of specific dates."""
        test_date = date(2024, 1, 15)
        
        expected_lot_sizes = {
            "RELIANCE": 250,
            "TCS": 300,
            "INFY": 300,
            "HDFCBANK": 550,    # Correct as of Jan 2024
            "ICICIBANK": 1375,  # Correct as of Jan 2024
        }
        
        for symbol, expected_lot_size in expected_lot_sizes.items():
            lot_size = data_manager.get_lot_size(symbol, test_date)
            assert lot_size == expected_lot_size, f"{symbol} lot size: expected {expected_lot_size}, got {lot_size}"


class TestDataQuality:
    """Test data quality and consistency."""
    
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    def test_ohlc_consistency(self, data_manager):
        """Test OHLC data consistency rules."""
        test_symbols = ["RELIANCE", "TCS", "INFY"]
        test_date = date(2024, 1, 15)
        
        for symbol in test_symbols:
            market_data = data_manager.get_market_data(symbol, test_date)
            if market_data:
                # OHLC consistency rules
                assert market_data.low <= market_data.open, f"{symbol}: Low > Open"
                assert market_data.low <= market_data.close, f"{symbol}: Low > Close"
                assert market_data.low <= market_data.high, f"{symbol}: Low > High"
                assert market_data.high >= market_data.open, f"{symbol}: High < Open"
                assert market_data.high >= market_data.close, f"{symbol}: High < Close"
                
                # Reasonable price ranges (not zero or negative)
                assert market_data.open > 0, f"{symbol}: Open price <= 0"
                assert market_data.high > 0, f"{symbol}: High price <= 0"
                assert market_data.low > 0, f"{symbol}: Low price <= 0"
                assert market_data.close > 0, f"{symbol}: Close price <= 0"
                
                # Volume should be non-negative
                assert market_data.volume >= 0, f"{symbol}: Negative volume"
    
    def test_price_reasonableness(self, data_manager):
        """Test that prices are within reasonable ranges."""
        test_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK"]
        test_date = date(2024, 1, 15)
        
        # Define reasonable price ranges for major stocks (as of 2024)
        price_ranges = {
            "RELIANCE": (2000, 3500),
            "TCS": (3000, 5000),
            "INFY": (1200, 2000),
            "HDFCBANK": (1400, 1800),
        }
        
        for symbol in test_symbols:
            price = data_manager.get_stock_price(symbol, test_date)
            if price and symbol in price_ranges:
                min_price, max_price = price_ranges[symbol]
                assert min_price <= price <= max_price, f"{symbol} price {price} outside range [{min_price}, {max_price}]"
    
    def test_trading_dates_quality(self, data_manager):
        """Test trading dates are business days and in correct order."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        trading_dates = data_manager.get_trading_dates(start_date, end_date)
        
        assert len(trading_dates) > 0, "No trading dates found"
        assert len(trading_dates) >= 20, "Too few trading dates in January 2024"  # At least 20 trading days
        
        # All should be weekdays
        for trading_date in trading_dates:
            assert trading_date.weekday() < 5, f"{trading_date} is not a weekday"
        
        # Should be in chronological order
        assert trading_dates == sorted(trading_dates), "Trading dates not in order"
        
        # Should be within requested range
        assert all(start_date <= td <= end_date for td in trading_dates), "Dates outside requested range"


class TestDataConsistency:
    """Test consistency across different data methods."""
    
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    def test_price_consistency(self, data_manager):
        """Test that get_stock_price and get_market_data return consistent prices."""
        test_symbols = ["RELIANCE", "TCS"]
        test_date = date(2024, 1, 15)
        
        for symbol in test_symbols:
            stock_price = data_manager.get_stock_price(symbol, test_date)
            market_data = data_manager.get_market_data(symbol, test_date)
            
            if stock_price and market_data:
                assert stock_price == market_data.close, f"{symbol}: Inconsistent close prices"
    
    def test_strike_generation_consistency(self, data_manager):
        """Test that strike generation methods return reasonable values."""
        symbol = "RELIANCE"
        test_date = date(2024, 1, 15)
        expiry = data_manager.get_monthly_expiry(2024, 1)
        
        if not expiry:
            pytest.skip("Cannot get expiry date")
        
        spot_price = data_manager.get_stock_price(symbol, test_date)
        if not spot_price:
            pytest.skip(f"Cannot get spot price for {symbol}")
        
        # Test ATM strike
        atm_strike = data_manager.get_atm_strike(symbol, expiry, test_date)
        if atm_strike:
            # ATM should be close to spot price (within 10%)
            assert abs(atm_strike - spot_price) / spot_price <= 0.1, f"ATM strike {atm_strike} too far from spot {spot_price}"
        
        # Test closest strike finding
        target_price = spot_price
        closest_strike = data_manager.find_closest_strike(symbol, target_price, expiry, test_date)
        if closest_strike:
            # Should be reasonably close to target
            assert abs(closest_strike - target_price) / target_price <= 0.05, f"Closest strike {closest_strike} too far from target {target_price}"
        
        # Test OTM strikes
        call_strike, put_strike = data_manager.get_otm_strikes(symbol, spot_price, 0.04, expiry, test_date)
        if call_strike and put_strike:
            # Call strike should be above spot
            assert call_strike > spot_price, f"OTM call strike {call_strike} not above spot {spot_price}"
            # Put strike should be below spot
            assert put_strike < spot_price, f"OTM put strike {put_strike} not below spot {spot_price}"
            
            # Should be approximately 4% away
            call_diff = (call_strike - spot_price) / spot_price
            put_diff = (spot_price - put_strike) / spot_price
            assert 0.03 <= call_diff <= 0.05, f"Call OTM percentage {call_diff:.3f} not around 4%"
            assert 0.03 <= put_diff <= 0.05, f"Put OTM percentage {put_diff:.3f} not around 4%"


class TestBoundaryConditions:
    """Test edge cases and boundary conditions."""
    
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    def test_weekend_dates(self, data_manager):
        """Test behavior with weekend dates."""
        # Saturday and Sunday in January 2024
        weekend_dates = [
            date(2024, 1, 6),   # Saturday
            date(2024, 1, 7),   # Sunday
            date(2024, 1, 13),  # Saturday
            date(2024, 1, 14),  # Sunday
        ]
        
        for weekend_date in weekend_dates:
            price = data_manager.get_stock_price("RELIANCE", weekend_date)
            market_data = data_manager.get_market_data("RELIANCE", weekend_date)
            
            # Should either return None or handle gracefully
            # (Current implementation may return None for weekends)
            if price is not None:
                assert isinstance(price, (int, float))
            if market_data is not None:
                assert isinstance(market_data, MarketData)
    
    def test_future_dates(self, data_manager):
        """Test behavior with future dates."""
        future_date = date(2025, 12, 31)
        
        price = data_manager.get_stock_price("RELIANCE", future_date)
        market_data = data_manager.get_market_data("RELIANCE", future_date)
        
        # Should return None for future dates
        assert price is None, "Should not return data for future dates"
        assert market_data is None, "Should not return data for future dates"
    
    def test_invalid_symbols(self, data_manager):
        """Test behavior with invalid symbols."""
        invalid_symbols = ["INVALID", "NOTREAL", "TEST123"]
        test_date = date(2024, 1, 15)
        
        for symbol in invalid_symbols:
            price = data_manager.get_stock_price(symbol, test_date)
            market_data = data_manager.get_market_data(symbol, test_date)
            lot_size = data_manager.get_lot_size(symbol, test_date)
            
            # Should handle gracefully (return None or default values)
            if price is not None:
                assert isinstance(price, (int, float))
            if market_data is not None:
                assert isinstance(market_data, MarketData)
            # Lot size should return a positive integer (fallback to 1)
            assert isinstance(lot_size, int) and lot_size > 0


# Performance and reliability markers
@pytest.mark.slow
class TestDataReliability:
    """Test data reliability and performance."""
    
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    def test_multiple_dates_consistency(self, data_manager):
        """Test data consistency across multiple dates."""
        symbol = "RELIANCE"
        test_dates = [
            date(2024, 1, 8),   # Monday
            date(2024, 1, 9),   # Tuesday
            date(2024, 1, 10),  # Wednesday
            date(2024, 1, 11),  # Thursday
            date(2024, 1, 12),  # Friday
        ]
        
        prices = []
        for test_date in test_dates:
            price = data_manager.get_stock_price(symbol, test_date)
            if price:
                prices.append(price)
        
        if len(prices) >= 2:
            # Prices should be within reasonable daily movement (max 20% change)
            for i in range(1, len(prices)):
                prev_price = prices[i-1]
                curr_price = prices[i]
                daily_change = abs(curr_price - prev_price) / prev_price
                assert daily_change <= 0.20, f"Unrealistic daily change: {daily_change:.2%}"