"""
Unit tests for the NSE Data Manager.

Tests the data manager functionality using proper pytest conventions.
"""

import pytest
from datetime import date, timedelta
from unittest.mock import Mock, patch

from skas_backtest.data import DataManager, MarketData, OptionQuote


class TestDataManagerBasics:
    """Test basic DataManager functionality."""
    
    @pytest.fixture
    def data_manager(self):
        """Create DataManager instance for testing."""
        return DataManager(cache_enabled=True, cache_size=10)
    
    def test_initialization(self, data_manager):
        """Test DataManager initialization."""
        assert data_manager.cache_enabled is True
        assert data_manager.cache_size == 10
        assert data_manager.data_source == "nse_options"
        assert hasattr(data_manager, 'option_calculator')
        assert hasattr(data_manager, 'nse_fetcher')
    
    def test_cache_functionality(self, data_manager):
        """Test cache operations."""
        # Test cache stats
        stats = data_manager.get_cache_stats()
        assert isinstance(stats, dict)
        assert 'price_cache_size' in stats
        assert 'option_cache_size' in stats
        assert 'calendar_cache_size' in stats
        
        # Test cache clearing
        data_manager._price_cache['test'] = 100.0
        data_manager.clear_cache()
        assert len(data_manager._price_cache) == 0
    
    def test_cache_trimming(self, data_manager):
        """Test cache size limiting."""
        # Fill cache beyond limit
        for i in range(15):
            data_manager._price_cache[f'key_{i}'] = float(i)
        
        # Trigger trim
        data_manager._trim_cache(data_manager._price_cache)
        
        # Should be at or below cache size limit
        assert len(data_manager._price_cache) <= data_manager.cache_size


class TestTradingDates:
    """Test trading date functionality."""
    
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    def test_get_trading_dates(self, data_manager):
        """Test trading dates calculation."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        trading_dates = data_manager.get_trading_dates(start_date, end_date)
        
        assert isinstance(trading_dates, list)
        assert len(trading_dates) > 0
        assert all(isinstance(d, date) for d in trading_dates)
        
        # Should only contain weekdays
        assert all(d.weekday() < 5 for d in trading_dates)
        
        # Should be in order
        assert trading_dates == sorted(trading_dates)
    
    @pytest.mark.parametrize("year,month,expected_weekday", [
        (2024, 1, 3),  # January 2024 expiry should be Thursday
        (2024, 2, 3),  # February 2024 expiry should be Thursday
        (2024, 12, 3), # December 2024 expiry should be Thursday
    ])
    def test_monthly_expiry_dates(self, data_manager, year, month, expected_weekday):
        """Test monthly expiry calculation."""
        expiry = data_manager.get_monthly_expiry(year, month)
        
        assert expiry is not None
        assert isinstance(expiry, date)
        assert expiry.year == year
        assert expiry.month == month
        assert expiry.weekday() == expected_weekday  # Thursday
    
    def test_next_monthly_expiry(self, data_manager):
        """Test next monthly expiry calculation."""
        reference_date = date(2024, 1, 1)
        next_expiry = data_manager.get_next_monthly_expiry(reference_date)
        
        assert next_expiry is not None
        assert isinstance(next_expiry, date)
        assert next_expiry > reference_date
        assert next_expiry.weekday() == 3  # Thursday


class TestStockData:
    """Test stock data fetching functionality."""
    
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    def test_get_stock_price_with_mock(self, data_manager):
        """Test stock price fetching with mocked data."""
        test_symbol = "RELIANCE"
        test_date = date(2024, 1, 15)
        expected_price = 2788.25
        
        # Mock the NSE fetcher
        with patch.object(data_manager.nse_fetcher, 'get_eod_price', return_value=expected_price):
            price = data_manager.get_stock_price(test_symbol, test_date)
            assert price == expected_price
    
    def test_get_stock_price_cache(self, data_manager):
        """Test stock price caching."""
        test_symbol = "RELIANCE"
        test_date = date(2024, 1, 15)
        expected_price = 2788.25
        
        with patch.object(data_manager.nse_fetcher, 'get_eod_price', return_value=expected_price) as mock_fetch:
            # First call
            price1 = data_manager.get_stock_price(test_symbol, test_date)
            # Second call (should use cache)
            price2 = data_manager.get_stock_price(test_symbol, test_date)
            
            assert price1 == price2 == expected_price
            # Should only call external API once due to caching
            assert mock_fetch.call_count == 1
    
    def test_get_stock_price_no_data(self, data_manager):
        """Test stock price when no data available."""
        test_symbol = "INVALID"
        test_date = date(2024, 1, 15)
        
        with patch.object(data_manager.nse_fetcher, 'get_eod_price', return_value=None):
            price = data_manager.get_stock_price(test_symbol, test_date)
            assert price is None
    
    def test_get_market_data_with_mock(self, data_manager):
        """Test market data fetching with mocked data."""
        import pandas as pd
        
        test_symbol = "RELIANCE"
        test_date = date(2024, 1, 15)
        
        # Mock historical data
        mock_data = pd.DataFrame({
            'OPEN': [2750.0],
            'HIGH': [2792.9],
            'LOW': [2732.0],
            'CLOSE': [2788.25],
            'VOLUME': [1000000]
        })
        
        with patch.object(data_manager.nse_fetcher, 'get_historical_data', return_value=mock_data):
            market_data = data_manager.get_market_data(test_symbol, test_date)
            
            assert market_data is not None
            assert isinstance(market_data, MarketData)
            assert market_data.symbol == test_symbol
            assert market_data.date == test_date
            assert market_data.open == 2750.0
            assert market_data.high == 2792.9
            assert market_data.low == 2732.0
            assert market_data.close == 2788.25
            assert market_data.volume == 1000000


class TestLotSizeAndStrikes:
    """Test lot size and strike calculation functionality."""
    
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    def test_get_lot_size(self, data_manager):
        """Test lot size retrieval."""
        test_symbol = "RELIANCE"
        test_date = date(2024, 1, 15)
        
        lot_size = data_manager.get_lot_size(test_symbol, test_date)
        
        assert isinstance(lot_size, int)
        assert lot_size > 0
        # RELIANCE lot size should be reasonable (typically 250)
        assert 100 <= lot_size <= 1000
    
    def test_find_closest_strike_with_mock(self, data_manager):
        """Test closest strike finding with mocked strikes."""
        test_symbol = "RELIANCE"
        target_price = 2788.25
        test_date = date(2024, 1, 15)
        test_expiry = date(2024, 1, 25)
        
        # Mock get_typical_strikes to return known strikes
        mock_strikes = [2700, 2750, 2800, 2850, 2900]
        
        with patch('skas_backtest.data.data_manager.get_typical_strikes', return_value=mock_strikes):
            closest_strike = data_manager.find_closest_strike(test_symbol, target_price, test_expiry, test_date)
            
            # Should find 2800 as closest to 2788.25
            assert closest_strike == 2800
    
    def test_get_otm_strikes_with_mock(self, data_manager):
        """Test OTM strike calculation."""
        test_symbol = "RELIANCE"
        spot_price = 2800.0
        otm_percentage = 0.04  # 4%
        test_date = date(2024, 1, 15)
        test_expiry = date(2024, 1, 25)
        
        # Mock strike finding
        with patch.object(data_manager, 'find_closest_strike', side_effect=[2920, 2688]):  # 4% OTM strikes
            call_strike, put_strike = data_manager.get_otm_strikes(
                test_symbol, spot_price, otm_percentage, test_expiry, test_date
            )
            
            assert call_strike == 2920  # 4% above spot
            assert put_strike == 2688   # 4% below spot


class TestOptionPricing:
    """Test option pricing functionality."""
    
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    def test_get_option_price_with_mock(self, data_manager):
        """Test option price calculation with mocked calculator."""
        test_symbol = "RELIANCE"
        strike = 2800
        expiry = date(2024, 1, 25)
        option_type = "call"
        as_of_date = date(2024, 1, 15)
        expected_price = 25.75
        
        # Mock the option calculator result
        mock_result = {'theoretical_price': expected_price}
        
        with patch.object(data_manager.option_calculator, 'calculate_premium', return_value=mock_result):
            price = data_manager.get_option_price(test_symbol, strike, expiry, option_type, as_of_date)
            
            assert price == expected_price
    
    def test_get_option_price_no_result(self, data_manager):
        """Test option price when calculation fails."""
        test_symbol = "RELIANCE"
        strike = 2800
        expiry = date(2024, 1, 25)
        option_type = "call"
        as_of_date = date(2024, 1, 15)
        
        with patch.object(data_manager.option_calculator, 'calculate_premium', return_value=None):
            price = data_manager.get_option_price(test_symbol, strike, expiry, option_type, as_of_date)
            
            assert price is None


class TestHistoricalData:
    """Test historical data functionality."""
    
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    def test_get_month_high_low_with_mock(self, data_manager):
        """Test monthly high/low calculation with mocked data."""
        test_symbol = "RELIANCE"
        year = 2024
        month = 1
        
        # Mock market data for different dates
        mock_data_1 = MarketData(test_symbol, date(2024, 1, 5), 2700, 2750, 2680, 2720, 1000)
        mock_data_2 = MarketData(test_symbol, date(2024, 1, 10), 2720, 2800, 2700, 2780, 1000)
        mock_data_3 = MarketData(test_symbol, date(2024, 1, 15), 2780, 2850, 2760, 2800, 1000)
        
        # Mock the trading dates and market data calls
        mock_dates = [date(2024, 1, 5), date(2024, 1, 10), date(2024, 1, 15)]
        
        with patch.object(data_manager, 'get_trading_dates', return_value=mock_dates), \
             patch.object(data_manager, 'get_market_data', side_effect=[mock_data_1, mock_data_2, mock_data_3]):
            
            high, low = data_manager.get_month_high_low(test_symbol, year, month)
            
            assert high == 2850  # Highest high from mock data
            assert low == 2680   # Lowest low from mock data
    
    def test_get_previous_month_range(self, data_manager):
        """Test previous month range calculation."""
        current_date = date(2024, 2, 1)  # February to get January range
        
        with patch.object(data_manager, 'get_month_high_low', return_value=(2850.0, 2680.0)):
            prev_high, prev_low = data_manager.get_previous_month_range("RELIANCE", current_date)
            
            assert prev_high == 2850.0
            assert prev_low == 2680.0


# Integration test markers
@pytest.mark.integration
class TestDataManagerIntegration:
    """Integration tests that may require actual data."""
    
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    @pytest.mark.skipif(True, reason="Requires live NSE data - run manually")
    def test_real_data_fetching(self, data_manager):
        """Test with real NSE data (skip in CI)."""
        # This test would run against real NSE data
        # Only enable when you want to test with actual data
        test_symbol = "RELIANCE"
        test_date = date(2024, 1, 15)
        
        price = data_manager.get_stock_price(test_symbol, test_date)
        # Would assert actual conditions here
        pass