"""
Real data integration tests (no mocks).

These tests run against actual NSE data to verify the complete data pipeline works correctly.
They may be slow and require network connectivity.
"""

import pytest
from datetime import date, timedelta
from skas_backtest.data import DataManager


@pytest.mark.integration
@pytest.mark.slow
class TestRealDataIntegration:
    """Integration tests using real NSE data."""
    
    @pytest.fixture(scope="class")
    def data_manager(self):
        """Shared DataManager for integration tests."""
        return DataManager(cache_enabled=True, cache_size=100)
    
    def test_end_to_end_data_workflow(self, data_manager):
        """Test complete data workflow for Middle Way strategy."""
        symbol = "RELIANCE"
        current_date = date(2024, 2, 1)  # February 1st to get January data
        
        # 1. Get previous month range (required for Middle Way strategy)
        prev_high, prev_low = data_manager.get_previous_month_range(symbol, current_date)
        assert prev_high is not None, "Could not get previous month high"
        assert prev_low is not None, "Could not get previous month low"
        assert prev_high >= prev_low, "High should be >= Low"
        
        print(f"\n{symbol} January 2024 range: High={prev_high}, Low={prev_low}")
        
        # 2. Get current month expiry
        expiry = data_manager.get_monthly_expiry(current_date.year, current_date.month)
        assert expiry is not None, "Could not get monthly expiry"
        assert expiry.weekday() == 3, "Expiry should be Thursday"
        
        print(f"February 2024 expiry: {expiry}")
        
        # 3. Find strikes based on previous month range
        call_strike = data_manager.find_closest_strike(symbol, prev_high, expiry, current_date)
        put_strike = data_manager.find_closest_strike(symbol, prev_low, expiry, current_date)
        
        assert call_strike is not None, "Could not find call strike"
        assert put_strike is not None, "Could not find put strike"
        assert call_strike > put_strike, "Call strike should be > Put strike"
        
        print(f"Strikes: Call={call_strike}, Put={put_strike}")
        
        # 4. Get option prices for strangle
        call_price = data_manager.get_option_price(symbol, call_strike, expiry, "call", current_date)
        put_price = data_manager.get_option_price(symbol, put_strike, expiry, "put", current_date)
        
        # These might be None if option pricing fails, but should be attempted
        print(f"Option prices: Call={call_price}, Put={put_price}")
        
        # 5. Get lot size for position sizing
        lot_size = data_manager.get_lot_size(symbol, current_date)
        assert lot_size > 0, "Lot size should be positive"
        
        print(f"Lot size: {lot_size}")
        
        # 6. Verify NIFTY hedge data
        nifty_expiry = data_manager.get_monthly_expiry(current_date.year, current_date.month)
        assert nifty_expiry == expiry, "NIFTY expiry should match stock expiry"
        
        print("✅ End-to-end workflow completed successfully")
    
    def test_historical_data_accuracy(self, data_manager):
        """Test historical data against known reference points."""
        # Test known historical data points
        test_cases = [
            {
                "symbol": "RELIANCE",
                "date": date(2024, 1, 15),
                "expected_close": 2788.25,
                "tolerance": 0.01  # Allow 1 paisa difference
            }
        ]
        
        for case in test_cases:
            symbol = case["symbol"]
            test_date = case["date"]
            expected = case["expected_close"]
            tolerance = case["tolerance"]
            
            # Test stock price
            actual_price = data_manager.get_stock_price(symbol, test_date)
            assert actual_price is not None, f"No price data for {symbol} on {test_date}"
            
            price_diff = abs(actual_price - expected)
            assert price_diff <= tolerance, f"{symbol} on {test_date}: expected {expected}, got {actual_price}, diff {price_diff}"
            
            # Test market data consistency
            market_data = data_manager.get_market_data(symbol, test_date)
            assert market_data is not None, f"No market data for {symbol} on {test_date}"
            assert market_data.close == actual_price, "Market data close should match stock price"
            
            print(f"✅ {symbol} on {test_date}: {actual_price} (expected {expected})")
    
    def test_multiple_symbols_consistency(self, data_manager):
        """Test data consistency across multiple symbols."""
        symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
        test_date = date(2024, 1, 15)
        
        results = {}
        for symbol in symbols:
            price = data_manager.get_stock_price(symbol, test_date)
            lot_size = data_manager.get_lot_size(symbol, test_date)
            market_data = data_manager.get_market_data(symbol, test_date)
            
            results[symbol] = {
                "price": price,
                "lot_size": lot_size,
                "market_data": market_data
            }
            
            # Basic validations
            if price:
                assert price > 0, f"{symbol}: Price should be positive"
            assert lot_size > 0, f"{symbol}: Lot size should be positive"
            
            if market_data:
                assert market_data.close == price, f"{symbol}: Inconsistent close price"
                assert market_data.high >= market_data.low, f"{symbol}: High < Low"
        
        # Print results for manual verification
        print(f"\nData for {test_date}:")
        for symbol, data in results.items():
            print(f"{symbol}: Price={data['price']}, Lot={data['lot_size']}")
    
    def test_trading_dates_accuracy(self, data_manager):
        """Test trading dates against known calendar."""
        # January 2024 known trading dates (no major holidays)
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        trading_dates = data_manager.get_trading_dates(start_date, end_date)
        
        # Known facts about January 2024
        assert len(trading_dates) == 23, f"Expected 23 trading dates in Jan 2024, got {len(trading_dates)}"
        
        # First trading day should be Jan 1 (Monday)
        assert trading_dates[0] == date(2024, 1, 1), "First trading day should be Jan 1"
        
        # Last trading day should be Jan 31 (Wednesday)
        assert trading_dates[-1] == date(2024, 1, 31), "Last trading day should be Jan 31"
        
        # Should not contain weekends
        for trading_date in trading_dates:
            assert trading_date.weekday() < 5, f"{trading_date} is a weekend"
        
        print(f"✅ January 2024: {len(trading_dates)} trading dates")
    
    def test_option_chain_generation(self, data_manager):
        """Test option chain generation with real data."""
        symbol = "RELIANCE"
        test_date = date(2024, 1, 15)
        expiry = data_manager.get_monthly_expiry(2024, 1)
        
        if not expiry:
            pytest.skip("Cannot get expiry date")
        
        spot_price = data_manager.get_stock_price(symbol, test_date)
        if not spot_price:
            pytest.skip(f"Cannot get spot price for {symbol}")
        
        # Define strike range around spot (±20%)
        strike_range = (spot_price * 0.8, spot_price * 1.2)
        
        option_chain = data_manager.get_option_chain(symbol, expiry, test_date, strike_range)
        
        assert len(option_chain) > 0, "Option chain should not be empty"
        
        # Verify structure
        for quote in option_chain[:5]:  # Check first 5
            assert quote.symbol == symbol
            assert quote.expiry == expiry
            assert quote.option_type in ["call", "put"]
            assert quote.ltp > 0, "Option price should be positive"
            assert quote.strike > 0, "Strike should be positive"
            assert strike_range[0] <= quote.strike <= strike_range[1], "Strike outside requested range"
        
        print(f"✅ Generated {len(option_chain)} option quotes for {symbol}")
    
    @pytest.mark.slow
    def test_performance_benchmarks(self, data_manager):
        """Test performance of data operations."""
        import time
        
        symbol = "RELIANCE"
        test_date = date(2024, 1, 15)
        
        # Benchmark stock price fetching
        start_time = time.time()
        price = data_manager.get_stock_price(symbol, test_date)
        price_time = time.time() - start_time
        
        assert price is not None, "Should get price data"
        assert price_time < 5.0, f"Stock price fetch too slow: {price_time:.2f}s"
        
        # Benchmark market data fetching
        start_time = time.time()
        market_data = data_manager.get_market_data(symbol, test_date)
        market_time = time.time() - start_time
        
        assert market_data is not None, "Should get market data"
        assert market_time < 5.0, f"Market data fetch too slow: {market_time:.2f}s"
        
        # Test caching effectiveness
        start_time = time.time()
        price2 = data_manager.get_stock_price(symbol, test_date)  # Should use cache
        cache_time = time.time() - start_time
        
        assert price2 == price, "Cached price should match original"
        assert cache_time < 0.1, f"Cached fetch too slow: {cache_time:.3f}s"
        
        print(f"✅ Performance: Price={price_time:.2f}s, Market={market_time:.2f}s, Cache={cache_time:.3f}s")


@pytest.mark.integration
class TestNSELibraryIntegration:
    """Test integration with NSE options library components."""
    
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    def test_nse_calculator_integration(self, data_manager):
        """Test NSE option calculator integration."""
        symbol = "RELIANCE"
        strike = 2800
        expiry = date(2024, 1, 25)
        test_date = date(2024, 1, 15)
        
        # Test option price calculation
        call_price = data_manager.get_option_price(symbol, strike, expiry, "call", test_date)
        put_price = data_manager.get_option_price(symbol, strike, expiry, "put", test_date)
        
        # May return None if calculation fails, but should handle gracefully
        if call_price:
            assert call_price > 0, "Call price should be positive"
            print(f"Call option price: ₹{call_price}")
        
        if put_price:
            assert put_price > 0, "Put price should be positive"
            print(f"Put option price: ₹{put_price}")
        
        print("✅ NSE calculator integration working")
    
    def test_nse_fetcher_integration(self, data_manager):
        """Test NSE data fetcher integration."""
        symbol = "RELIANCE"
        test_date = date(2024, 1, 15)
        
        # Test direct NSE fetcher methods
        fetcher = data_manager.nse_fetcher
        
        # Test EOD price
        eod_price = fetcher.get_eod_price(symbol, test_date)
        if eod_price:
            assert eod_price > 0, "EOD price should be positive"
            print(f"EOD price: ₹{eod_price}")
        
        # Test historical data
        historical_data = fetcher.get_historical_data(symbol, test_date, test_date, "EQ")
        if historical_data is not None and not historical_data.empty:
            assert len(historical_data) > 0, "Should have historical data"
            print(f"Historical data: {len(historical_data)} rows")
        
        print("✅ NSE fetcher integration working")
    
    def test_utilities_integration(self, data_manager):
        """Test NSE utilities integration."""
        symbol = "RELIANCE"
        test_date = date(2024, 1, 15)
        
        # Test lot size utility
        lot_size = data_manager.get_lot_size(symbol, test_date)
        assert lot_size == 250, f"RELIANCE lot size should be 250, got {lot_size}"
        
        # Test expiry utilities
        jan_expiry = data_manager.get_monthly_expiry(2024, 1)
        assert jan_expiry == date(2024, 1, 25), "January 2024 expiry should be 25th"
        
        next_expiry = data_manager.get_next_monthly_expiry(date(2024, 1, 1))
        assert next_expiry == jan_expiry, "Next expiry from Jan 1 should be Jan 25"
        
        print("✅ NSE utilities integration working")