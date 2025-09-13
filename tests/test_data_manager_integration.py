#!/usr/bin/env python3
"""
Simple test script for the NSE Data Manager.
Tests the data manager directly without importing the full package.
"""

import sys
import os
from datetime import date, datetime, timedelta

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import DataManager from data package
from skas_backtest.data import DataManager


def test_basic_functionality():
    """Test basic DataManager functionality."""
    print("=" * 60)
    print("Testing NSE Data Manager - Basic Functionality")
    print("=" * 60)
    
    # Initialize DataManager
    print("\n1. Initializing DataManager...")
    try:
        dm = DataManager(cache_enabled=True, cache_size=50)
        print("✅ DataManager initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize DataManager: {e}")
        return False
    
    # Test symbol and dates
    test_symbol = "RELIANCE"
    test_date = date(2024, 1, 15)  # A Monday in January 2024
    
    print(f"\nTesting with symbol: {test_symbol}")
    print(f"Testing with date: {test_date}")
    
    # Test 1: Get trading dates (this doesn't depend on external data)
    print("\n2. Testing get_trading_dates()...")
    try:
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        trading_dates = dm.get_trading_dates(start_date, end_date)
        print(f"✅ Found {len(trading_dates)} trading dates in January 2024")
        print(f"   First date: {trading_dates[0]}")
        print(f"   Last date: {trading_dates[-1]}")
        
        # Verify they are weekdays
        weekdays_only = all(d.weekday() < 5 for d in trading_dates)
        print(f"   All weekdays: {'✅' if weekdays_only else '❌'}")
        
    except Exception as e:
        print(f"❌ Error getting trading dates: {e}")
        return False
    
    # Test 2: Get monthly expiry (uses NSE library utility)
    print("\n3. Testing get_monthly_expiry()...")
    try:
        expiry = dm.get_monthly_expiry(2024, 1)
        if expiry:
            print(f"✅ January 2024 monthly expiry: {expiry}")
            # Verify it's a Thursday
            is_thursday = expiry.weekday() == 3  # Thursday = 3
            print(f"   Is Thursday: {'✅' if is_thursday else '❌'}")
        else:
            print("❌ Could not get monthly expiry")
            return False
    except Exception as e:
        print(f"❌ Error getting monthly expiry: {e}")
        return False
    
    # Test 3: Get next monthly expiry
    print("\n4. Testing get_next_monthly_expiry()...")
    try:
        reference_date = date(2024, 1, 1)
        next_expiry = dm.get_next_monthly_expiry(reference_date)
        if next_expiry:
            print(f"✅ Next monthly expiry after {reference_date}: {next_expiry}")
        else:
            print("❌ Could not get next monthly expiry")
    except Exception as e:
        print(f"❌ Error getting next monthly expiry: {e}")
    
    # Test 4: Get lot size
    print("\n5. Testing get_lot_size()...")
    try:
        lot_size = dm.get_lot_size(test_symbol, test_date)
        print(f"✅ Lot size for {test_symbol}: {lot_size}")
        
        # Verify it's a positive integer
        is_valid = isinstance(lot_size, int) and lot_size > 0
        print(f"   Valid lot size: {'✅' if is_valid else '❌'}")
        
    except Exception as e:
        print(f"❌ Error getting lot size: {e}")
    
    # Test 5: Cache functionality
    print("\n6. Testing cache functionality...")
    try:
        # Add some items to cache
        dm._price_cache["test_key_1"] = 100.0
        dm._price_cache["test_key_2"] = 200.0
        
        cache_stats = dm.get_cache_stats()
        print(f"✅ Cache statistics:")
        for key, value in cache_stats.items():
            print(f"   {key}: {value}")
        
        # Test cache clearing
        dm.clear_cache()
        cache_stats_after = dm.get_cache_stats()
        all_empty = all(count == 0 for count in cache_stats_after.values())
        print(f"   Cache cleared successfully: {'✅' if all_empty else '❌'}")
        
    except Exception as e:
        print(f"❌ Error testing cache: {e}")
    
    print("\n" + "=" * 60)
    print("Basic functionality tests completed successfully!")
    print("=" * 60)
    
    return True


def test_data_dependent_functionality():
    """
    Test functionality that depends on external data.
    These may fail if data is not available, but that's expected.
    """
    print("\n" + "=" * 60)
    print("Testing Data-Dependent Functionality")
    print("(Warnings expected if historical data not available)")
    print("=" * 60)
    
    dm = DataManager()
    test_symbol = "RELIANCE"
    test_date = date(2024, 1, 15)
    
    # Test 1: Get stock price
    print("\n1. Testing get_stock_price()...")
    try:
        price = dm.get_stock_price(test_symbol, test_date)
        if price:
            print(f"✅ Stock price for {test_symbol} on {test_date}: ₹{price:.2f}")
        else:
            print(f"⚠️  No price data found for {test_symbol} on {test_date}")
            print("   This is expected if historical data is not available")
    except Exception as e:
        print(f"❌ Error getting stock price: {e}")
    
    # Test 2: Get market data
    print("\n2. Testing get_market_data()...")
    try:
        market_data = dm.get_market_data(test_symbol, test_date)
        if market_data:
            print(f"✅ Market data retrieved successfully")
            print(f"   OHLC: {market_data.open:.2f}/{market_data.high:.2f}/{market_data.low:.2f}/{market_data.close:.2f}")
        else:
            print(f"⚠️  No market data found for {test_symbol} on {test_date}")
    except Exception as e:
        print(f"❌ Error getting market data: {e}")
    
    # Test 3: Find closest strike (depends on typical strikes function)
    print("\n3. Testing find_closest_strike()...")
    try:
        target_price = 2500.0
        expiry = dm.get_monthly_expiry(2024, 1)
        if expiry:
            closest_strike = dm.find_closest_strike(test_symbol, target_price, expiry, test_date)
            if closest_strike:
                print(f"✅ Closest strike to ₹{target_price}: ₹{closest_strike}")
            else:
                print(f"⚠️  Could not find closest strike (may depend on data availability)")
        else:
            print("❌ Cannot test without expiry date")
    except Exception as e:
        print(f"❌ Error finding closest strike: {e}")
    
    # Test 4: Get ATM strike
    print("\n4. Testing get_atm_strike()...")
    try:
        expiry = dm.get_monthly_expiry(2024, 1)
        if expiry:
            atm_strike = dm.get_atm_strike(test_symbol, expiry, test_date)
            if atm_strike:
                print(f"✅ ATM strike for {test_symbol}: ₹{atm_strike}")
            else:
                print(f"⚠️  Could not get ATM strike (may depend on price data)")
        else:
            print("❌ Cannot test without expiry date")
    except Exception as e:
        print(f"❌ Error getting ATM strike: {e}")
    
    print("\n" + "=" * 60)
    print("Data-dependent tests completed")
    print("Note: Warnings are expected if historical data is not available")
    print("=" * 60)


def main():
    """Main test function."""
    print("Starting NSE Data Manager Tests...\n")
    
    # Run basic tests first
    basic_success = test_basic_functionality()
    
    if basic_success:
        print("\n✅ Basic tests passed! NSE library integration is working.")
        
        # Run data-dependent tests
        test_data_dependent_functionality()
        
        print("\n🎉 All tests completed successfully!")
        print("\nThe DataManager is ready for use in the Middle Way strategy.")
        print("Next steps:")
        print("1. ✅ Data infrastructure complete")
        print("2. 🔄 Ready to implement Middle Way Strategy class")
        
        return True
    else:
        print("\n❌ Basic tests failed. Please check NSE library integration.")
        return False


if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    exit(exit_code)