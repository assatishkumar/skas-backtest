#!/usr/bin/env python3
"""
Data Verification Script for NSE Data Manager

This script fetches and verifies data against known reference values.
Use this to manually validate that the DataManager is working correctly
and returning accurate data.

Usage:
    python scripts/verify_data.py
    python scripts/verify_data.py --symbol RELIANCE --date 2024-01-15
    python scripts/verify_data.py --comprehensive
"""

import sys
import os
from datetime import date, datetime, timedelta
import argparse
from typing import Dict, List, Any
from tabulate import tabulate

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from skas_backtest.data import DataManager


class DataVerifier:
    """Verify NSE data accuracy and consistency."""
    
    def __init__(self):
        self.data_manager = DataManager(cache_enabled=False)  # Fresh data
        self.known_data = {
            # Known historical data points for verification
            "RELIANCE": {
                date(2024, 1, 15): {
                    "close": 2788.25,
                    "open": 2750.0,
                    "high": 2792.9,
                    "low": 2732.0,
                }
            }
        }
        self.known_lot_sizes = {
            "RELIANCE": 250,
            "TCS": 300,
            "INFY": 300,
            "HDFCBANK": 550,    # Correct as of Jan 2024
            "ICICIBANK": 1375,  # Correct as of Jan 2024
        }
        self.known_expiries = {
            (2024, 1): date(2024, 1, 25),
            (2024, 2): date(2024, 2, 29),
            (2024, 3): date(2024, 3, 28),
        }
    
    def verify_single_stock(self, symbol: str, target_date: date) -> Dict[str, Any]:
        """Verify data for a single stock on a specific date."""
        print(f"\n{'='*60}")
        print(f"Verifying {symbol} on {target_date}")
        print(f"{'='*60}")
        
        results = {
            "symbol": symbol,
            "date": target_date,
            "tests_passed": 0,
            "tests_failed": 0,
            "issues": []
        }
        
        # Test 1: Stock Price
        try:
            stock_price = self.data_manager.get_stock_price(symbol, target_date)
            if stock_price is not None:
                print(f"✅ Stock Price: ₹{stock_price}")
                
                # Check against known data
                if symbol in self.known_data and target_date in self.known_data[symbol]:
                    expected = self.known_data[symbol][target_date]["close"]
                    if abs(stock_price - expected) < 0.01:
                        print(f"   ✅ Matches expected: ₹{expected}")
                        results["tests_passed"] += 1
                    else:
                        print(f"   ❌ Expected ₹{expected}, got ₹{stock_price}")
                        results["tests_failed"] += 1
                        results["issues"].append(f"Price mismatch: expected {expected}, got {stock_price}")
                else:
                    results["tests_passed"] += 1
            else:
                print(f"❌ Stock Price: No data available")
                results["tests_failed"] += 1
                results["issues"].append("No stock price data")
        except Exception as e:
            print(f"❌ Stock Price: Error - {e}")
            results["tests_failed"] += 1
            results["issues"].append(f"Stock price error: {e}")
        
        # Test 2: Market Data
        try:
            market_data = self.data_manager.get_market_data(symbol, target_date)
            if market_data:
                print(f"✅ Market Data:")
                print(f"   Open: ₹{market_data.open}")
                print(f"   High: ₹{market_data.high}")
                print(f"   Low: ₹{market_data.low}")
                print(f"   Close: ₹{market_data.close}")
                print(f"   Volume: {market_data.volume:,}")
                
                # Verify OHLC consistency
                if (market_data.low <= market_data.open <= market_data.high and
                    market_data.low <= market_data.close <= market_data.high):
                    print("   ✅ OHLC consistency check passed")
                    results["tests_passed"] += 1
                else:
                    print("   ❌ OHLC consistency check failed")
                    results["tests_failed"] += 1
                    results["issues"].append("OHLC consistency violation")
                
                # Check against known data
                if symbol in self.known_data and target_date in self.known_data[symbol]:
                    known = self.known_data[symbol][target_date]
                    matches = 0
                    total = 0
                    
                    for field in ["open", "high", "low", "close"]:
                        if field in known:
                            total += 1
                            actual = getattr(market_data, field)
                            expected = known[field]
                            if abs(actual - expected) < 0.01:
                                matches += 1
                    
                    if matches == total:
                        print(f"   ✅ All {total} known values match")
                        results["tests_passed"] += 1
                    else:
                        print(f"   ❌ Only {matches}/{total} known values match")
                        results["tests_failed"] += 1
                        results["issues"].append(f"Market data mismatch: {matches}/{total} fields correct")
                else:
                    results["tests_passed"] += 1
            else:
                print(f"❌ Market Data: No data available")
                results["tests_failed"] += 1
                results["issues"].append("No market data")
        except Exception as e:
            print(f"❌ Market Data: Error - {e}")
            results["tests_failed"] += 1
            results["issues"].append(f"Market data error: {e}")
        
        # Test 3: Lot Size
        try:
            lot_size = self.data_manager.get_lot_size(symbol, target_date)
            print(f"✅ Lot Size: {lot_size}")
            
            if symbol in self.known_lot_sizes:
                expected_lot = self.known_lot_sizes[symbol]
                if lot_size == expected_lot:
                    print(f"   ✅ Matches expected: {expected_lot}")
                    results["tests_passed"] += 1
                else:
                    print(f"   ❌ Expected {expected_lot}, got {lot_size}")
                    results["tests_failed"] += 1
                    results["issues"].append(f"Lot size mismatch: expected {expected_lot}, got {lot_size}")
            else:
                results["tests_passed"] += 1
        except Exception as e:
            print(f"❌ Lot Size: Error - {e}")
            results["tests_failed"] += 1
            results["issues"].append(f"Lot size error: {e}")
        
        return results
    
    def verify_expiry_dates(self) -> Dict[str, Any]:
        """Verify monthly expiry date calculations."""
        print(f"\n{'='*60}")
        print("Verifying Monthly Expiry Dates")
        print(f"{'='*60}")
        
        results = {
            "tests_passed": 0,
            "tests_failed": 0,
            "issues": []
        }
        
        for (year, month), expected_expiry in self.known_expiries.items():
            try:
                calculated_expiry = self.data_manager.get_monthly_expiry(year, month)
                
                if calculated_expiry == expected_expiry:
                    print(f"✅ {year}-{month:02d}: {calculated_expiry} (Thursday)")
                    results["tests_passed"] += 1
                else:
                    print(f"❌ {year}-{month:02d}: Expected {expected_expiry}, got {calculated_expiry}")
                    results["tests_failed"] += 1
                    results["issues"].append(f"Expiry mismatch for {year}-{month}")
                
                # Verify it's a Thursday
                if calculated_expiry and calculated_expiry.weekday() != 3:
                    print(f"   ❌ Not a Thursday!")
                    results["tests_failed"] += 1
                    results["issues"].append(f"Expiry {calculated_expiry} not a Thursday")
                    
            except Exception as e:
                print(f"❌ {year}-{month:02d}: Error - {e}")
                results["tests_failed"] += 1
                results["issues"].append(f"Expiry calculation error for {year}-{month}: {e}")
        
        return results
    
    def verify_trading_dates(self, year: int = 2024, month: int = 1) -> Dict[str, Any]:
        """Verify trading dates calculation."""
        print(f"\n{'='*60}")
        print(f"Verifying Trading Dates for {year}-{month:02d}")
        print(f"{'='*60}")
        
        results = {
            "tests_passed": 0,
            "tests_failed": 0,
            "issues": []
        }
        
        try:
            start_date = date(year, month, 1)
            if month == 12:
                end_date = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = date(year, month + 1, 1) - timedelta(days=1)
            
            trading_dates = self.data_manager.get_trading_dates(start_date, end_date)
            
            print(f"✅ Found {len(trading_dates)} trading dates")
            print(f"   First: {trading_dates[0]} ({trading_dates[0].strftime('%A')})")
            print(f"   Last: {trading_dates[-1]} ({trading_dates[-1].strftime('%A')})")
            
            # Verify all are weekdays
            weekday_count = sum(1 for d in trading_dates if d.weekday() < 5)
            if weekday_count == len(trading_dates):
                print(f"   ✅ All {len(trading_dates)} dates are weekdays")
                results["tests_passed"] += 1
            else:
                print(f"   ❌ {len(trading_dates) - weekday_count} weekend dates found")
                results["tests_failed"] += 1
                results["issues"].append("Weekend dates in trading calendar")
            
            # Verify chronological order
            if trading_dates == sorted(trading_dates):
                print(f"   ✅ Dates in chronological order")
                results["tests_passed"] += 1
            else:
                print(f"   ❌ Dates not in chronological order")
                results["tests_failed"] += 1
                results["issues"].append("Trading dates not ordered")
            
            # Check specific counts for known months
            if year == 2024 and month == 1:
                expected_count = 23  # Known for January 2024
                if len(trading_dates) == expected_count:
                    print(f"   ✅ Correct count for January 2024: {expected_count}")
                    results["tests_passed"] += 1
                else:
                    print(f"   ❌ Expected {expected_count} days, got {len(trading_dates)}")
                    results["tests_failed"] += 1
                    results["issues"].append(f"Incorrect trading day count for Jan 2024")
            
        except Exception as e:
            print(f"❌ Trading Dates: Error - {e}")
            results["tests_failed"] += 1
            results["issues"].append(f"Trading dates error: {e}")
        
        return results
    
    def verify_option_calculations(self, symbol: str = "RELIANCE", target_date: date = date(2024, 1, 15)) -> Dict[str, Any]:
        """Verify option-related calculations."""
        print(f"\n{'='*60}")
        print(f"Verifying Option Calculations for {symbol}")
        print(f"{'='*60}")
        
        results = {
            "tests_passed": 0,
            "tests_failed": 0,
            "issues": []
        }
        
        try:
            # Get spot price and expiry
            spot_price = self.data_manager.get_stock_price(symbol, target_date)
            expiry = self.data_manager.get_monthly_expiry(target_date.year, target_date.month)
            
            if not spot_price or not expiry:
                print(f"❌ Cannot get spot price or expiry")
                results["tests_failed"] += 1
                results["issues"].append("Missing spot price or expiry")
                return results
            
            print(f"Spot Price: ₹{spot_price}")
            print(f"Expiry: {expiry}")
            
            # Test ATM strike
            atm_strike = self.data_manager.get_atm_strike(symbol, expiry, target_date)
            if atm_strike:
                atm_diff_pct = abs(atm_strike - spot_price) / spot_price * 100
                print(f"✅ ATM Strike: ₹{atm_strike} ({atm_diff_pct:.1f}% from spot)")
                
                if atm_diff_pct <= 5.0:  # ATM should be within 5% of spot
                    results["tests_passed"] += 1
                else:
                    print(f"   ❌ ATM strike too far from spot ({atm_diff_pct:.1f}%)")
                    results["tests_failed"] += 1
                    results["issues"].append(f"ATM strike {atm_diff_pct:.1f}% from spot")
            else:
                print(f"❌ ATM Strike: Could not calculate")
                results["tests_failed"] += 1
                results["issues"].append("ATM strike calculation failed")
            
            # Test closest strike finding
            target_strike = 2800
            closest_strike = self.data_manager.find_closest_strike(symbol, target_strike, expiry, target_date)
            if closest_strike:
                print(f"✅ Closest to ₹{target_strike}: ₹{closest_strike}")
                results["tests_passed"] += 1
            else:
                print(f"❌ Could not find closest strike to ₹{target_strike}")
                results["tests_failed"] += 1
                results["issues"].append("Closest strike finding failed")
            
            # Test OTM strikes
            call_strike, put_strike = self.data_manager.get_otm_strikes(symbol, spot_price, 0.04, expiry, target_date)
            if call_strike and put_strike:
                call_otm = (call_strike - spot_price) / spot_price * 100
                put_otm = (spot_price - put_strike) / spot_price * 100
                
                print(f"✅ 4% OTM Strikes:")
                print(f"   Call: ₹{call_strike} ({call_otm:.1f}% OTM)")
                print(f"   Put: ₹{put_strike} ({put_otm:.1f}% OTM)")
                
                if 3.0 <= call_otm <= 5.0 and 3.0 <= put_otm <= 5.0:
                    results["tests_passed"] += 1
                else:
                    print(f"   ❌ OTM percentages outside 3-5% range")
                    results["tests_failed"] += 1
                    results["issues"].append("OTM strikes not 4% from spot")
            else:
                print(f"❌ Could not calculate OTM strikes")
                results["tests_failed"] += 1
                results["issues"].append("OTM strike calculation failed")
            
        except Exception as e:
            print(f"❌ Option Calculations: Error - {e}")
            results["tests_failed"] += 1
            results["issues"].append(f"Option calculation error: {e}")
        
        return results
    
    def run_comprehensive_verification(self) -> Dict[str, Any]:
        """Run comprehensive verification across all components."""
        print(f"\n{'#'*80}")
        print("COMPREHENSIVE DATA VERIFICATION")
        print(f"{'#'*80}")
        
        all_results = {
            "total_tests_passed": 0,
            "total_tests_failed": 0,
            "all_issues": [],
            "component_results": {}
        }
        
        # Test 1: Expiry dates
        expiry_results = self.verify_expiry_dates()
        all_results["component_results"]["expiry_dates"] = expiry_results
        all_results["total_tests_passed"] += expiry_results["tests_passed"]
        all_results["total_tests_failed"] += expiry_results["tests_failed"]
        all_results["all_issues"].extend(expiry_results["issues"])
        
        # Test 2: Trading dates
        trading_results = self.verify_trading_dates()
        all_results["component_results"]["trading_dates"] = trading_results
        all_results["total_tests_passed"] += trading_results["tests_passed"]
        all_results["total_tests_failed"] += trading_results["tests_failed"]
        all_results["all_issues"].extend(trading_results["issues"])
        
        # Test 3: Stock data for major symbols
        major_symbols = ["RELIANCE", "TCS", "INFY"]
        test_date = date(2024, 1, 15)
        
        for symbol in major_symbols:
            stock_results = self.verify_single_stock(symbol, test_date)
            all_results["component_results"][f"{symbol}_data"] = stock_results
            all_results["total_tests_passed"] += stock_results["tests_passed"]
            all_results["total_tests_failed"] += stock_results["tests_failed"]
            all_results["all_issues"].extend(stock_results["issues"])
        
        # Test 4: Option calculations
        option_results = self.verify_option_calculations()
        all_results["component_results"]["option_calculations"] = option_results
        all_results["total_tests_passed"] += option_results["tests_passed"]
        all_results["total_tests_failed"] += option_results["tests_failed"]
        all_results["all_issues"].extend(option_results["issues"])
        
        return all_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print verification summary."""
        print(f"\n{'='*80}")
        print("VERIFICATION SUMMARY")
        print(f"{'='*80}")
        
        total_passed = results["total_tests_passed"]
        total_failed = results["total_tests_failed"]
        total_tests = total_passed + total_failed
        
        if total_tests > 0:
            success_rate = (total_passed / total_tests) * 100
            print(f"Tests Passed: {total_passed}/{total_tests} ({success_rate:.1f}%)")
        else:
            print("No tests were run")
        
        if total_failed > 0:
            print(f"\n❌ {total_failed} ISSUES FOUND:")
            for i, issue in enumerate(results["all_issues"], 1):
                print(f"  {i}. {issue}")
        else:
            print(f"\n✅ ALL TESTS PASSED! Data verification successful.")
        
        # Component breakdown
        print(f"\nComponent Results:")
        table_data = []
        for component, comp_results in results["component_results"].items():
            passed = comp_results["tests_passed"]
            failed = comp_results["tests_failed"]
            total_comp = passed + failed
            status = "✅ PASS" if failed == 0 else "❌ FAIL"
            table_data.append([component, f"{passed}/{total_comp}", status])
        
        print(tabulate(table_data, headers=["Component", "Passed", "Status"], tablefmt="grid"))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Verify NSE Data Manager accuracy")
    parser.add_argument("--symbol", default="RELIANCE", help="Stock symbol to verify")
    parser.add_argument("--date", default="2024-01-15", help="Date to verify (YYYY-MM-DD)")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive verification")
    
    args = parser.parse_args()
    
    verifier = DataVerifier()
    
    if args.comprehensive:
        results = verifier.run_comprehensive_verification()
        verifier.print_summary(results)
    else:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        results = verifier.verify_single_stock(args.symbol, target_date)
        
        print(f"\n{'='*60}")
        print("SINGLE STOCK VERIFICATION SUMMARY")
        print(f"{'='*60}")
        
        total = results["tests_passed"] + results["tests_failed"]
        if total > 0:
            success_rate = (results["tests_passed"] / total) * 100
            print(f"Tests Passed: {results['tests_passed']}/{total} ({success_rate:.1f}%)")
        
        if results["issues"]:
            print(f"\nIssues found:")
            for i, issue in enumerate(results["issues"], 1):
                print(f"  {i}. {issue}")
        else:
            print(f"\n✅ All tests passed for {args.symbol} on {args.date}")


if __name__ == "__main__":
    main()