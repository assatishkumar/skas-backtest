# Data Verification Guide

## How to Verify Your Tests Are Working Correctly

This guide explains how to ensure your tests are actually validating real data and working as expected.

## The Problem

When tests pass, it doesn't necessarily mean they're testing the right things:
- **Mocked tests** might use unrealistic values
- **Integration tests** might pass but fetch incorrect data
- **No baseline** to compare against known correct values

## Our Solution: Multi-Layer Verification

### 1. **Unit Tests with Realistic Mocks** (`tests/test_data_manager.py`)
- Uses mocks with **real values** (e.g., RELIANCE @ ₹2788.25 on 2024-01-15)
- Tests code logic without external dependencies
- Fast execution (~15 seconds)

```bash
pytest tests/test_data_manager.py -v
```

### 2. **Data Validation Tests** (`tests/test_data_validation.py`)
- Tests against **known historical data**
- Validates **data quality** (OHLC consistency, reasonable ranges)
- Checks **cross-method consistency**

```bash
pytest tests/test_data_validation.py -v
```

### 3. **Real Data Integration Tests** (`tests/test_real_data_integration.py`)
- No mocks - tests actual NSE data fetching
- Validates **end-to-end workflows**
- May be slower, requires network

```bash
pytest tests/test_real_data_integration.py -v
```

### 4. **Manual Verification Script** (`scripts/verify_data.py`)
- Interactive verification against known values
- Detailed reporting of discrepancies
- Manual cross-checking capability

```bash
python scripts/verify_data.py --symbol RELIANCE --date 2024-01-15
python scripts/verify_data.py --comprehensive
```

## Known Reference Values

### Stock Prices (Verified)
| Symbol   | Date       | Close    | Open     | High     | Low      |
|----------|------------|----------|----------|----------|----------|
| RELIANCE | 2024-01-15 | 2788.25  | 2750.0   | 2792.9   | 2732.0   |

### Lot Sizes (as of 2024-01-15)
| Symbol     | Lot Size | Last Changed |
|------------|----------|--------------|
| RELIANCE   | 250      | 2021-09-27   |
| TCS        | 300      | 2021-11-22   |
| INFY       | 300      | 2021-09-27   |
| HDFCBANK   | 550      | 2022-07-11   |
| ICICIBANK  | 1375     | 2022-08-15   |

### Monthly Expiries (Known)
| Month      | Expiry Date | Day       |
|------------|-------------|-----------|
| 2024-01    | 2024-01-25  | Thursday  |
| 2024-02    | 2024-02-29  | Thursday  |
| 2024-03    | 2024-03-28  | Thursday  |

## How to Verify Your Data

### Step 1: Quick Verification
```bash
# Test single stock with known data
python scripts/verify_data.py --symbol RELIANCE --date 2024-01-15
```

Expected output:
```
✅ Stock Price: ₹2788.25
   ✅ Matches expected: ₹2788.25
✅ Market Data:
   ✅ OHLC consistency check passed
   ✅ All 4 known values match
✅ Lot Size: 250
   ✅ Matches expected: 250
```

### Step 2: Comprehensive Verification
```bash
# Run full verification suite
python scripts/verify_data.py --comprehensive
```

### Step 3: Run All Test Suites
```bash
# Unit tests (fast)
pytest tests/test_data_manager.py -v

# Validation tests (medium)
pytest tests/test_data_validation.py -v

# Integration tests (slow, requires network)
pytest tests/test_real_data_integration.py -v
```

### Step 4: Check Coverage
```bash
pytest --cov=skas_backtest.data --cov-report=html
open htmlcov/index.html
```

## Interpreting Results

### ✅ Good Signs
- All known values match exactly
- OHLC consistency maintained
- Lot sizes match NSE specifications
- Expiry dates are Thursdays
- Price ranges are reasonable

### ❌ Warning Signs
- Price mismatches with known values
- OHLC violations (Low > High, etc.)
- Unrealistic price movements
- Wrong lot sizes
- Expiry dates not on Thursdays

## Adding New Reference Data

### 1. Add to Known Data
Edit `scripts/verify_data.py`:
```python
self.known_data = {
    "SYMBOL": {
        date(2024, 1, 15): {
            "close": 1234.56,
            "open": 1230.0,
            "high": 1240.0,
            "low": 1225.0,
        }
    }
}
```

### 2. Cross-Check with External Sources
- NSE website historical data
- Financial news reports
- Broker platforms
- Yahoo Finance / Google Finance

### 3. Update Test Mocks
Ensure test mocks in `test_data_manager.py` use realistic values:
```python
expected_price = 2788.25  # Use real known value
```

## Common Issues and Solutions

### Issue: Tests pass but data is wrong
**Solution**: Check against verification script
```bash
python scripts/verify_data.py --symbol YOUR_SYMBOL --date YOUR_DATE
```

### Issue: Integration tests fail
**Possible causes**:
- Network connectivity
- NSE data source unavailable
- Date is a holiday/weekend

**Solution**: Run data validation tests instead
```bash
pytest tests/test_data_validation.py -v
```

### Issue: Prices don't match known values
**Investigation steps**:
1. Check if date is correct
2. Verify symbol format
3. Check for stock splits/bonuses
4. Cross-reference with multiple sources

### Issue: Option calculations seem wrong
**Verification**:
```bash
# Test option calculations
python scripts/verify_data.py --comprehensive
# Look for "Option Calculations" section
```

## Best Practices

### 1. Regular Verification
- Run verification script weekly
- Update known reference values quarterly
- Cross-check with external sources

### 2. Add New Test Cases
When you find discrepancies:
1. Investigate the root cause
2. Add new test case if needed
3. Update reference values if confirmed

### 3. Document Changes
- Record any data source changes
- Note any formula updates
- Track new reference values

### 4. Continuous Monitoring
```bash
# Add to CI/CD pipeline
pytest tests/test_data_validation.py --tb=short
```

## Example Verification Session

```bash
$ python scripts/verify_data.py --comprehensive

################################################################################
COMPREHENSIVE DATA VERIFICATION
################################################################################

============================================================
Verifying Monthly Expiry Dates
============================================================
✅ 2024-01: 2024-01-25 (Thursday)
✅ 2024-02: 2024-02-29 (Thursday)
✅ 2024-03: 2024-03-28 (Thursday)

============================================================
Verifying Trading Dates for 2024-01
============================================================
✅ Found 23 trading dates
   First: 2024-01-01 (Monday)
   Last: 2024-01-31 (Wednesday)
   ✅ All 23 dates are weekdays
   ✅ Dates in chronological order
   ✅ Correct count for January 2024: 23

============================================================
Verifying RELIANCE on 2024-01-15
============================================================
✅ Stock Price: ₹2788.25
   ✅ Matches expected: ₹2788.25
✅ Market Data:
   ✅ OHLC consistency check passed
   ✅ All 4 known values match
✅ Lot Size: 250
   ✅ Matches expected: 250

================================================================================
VERIFICATION SUMMARY
================================================================================
Tests Passed: 15/15 (100.0%)

✅ ALL TESTS PASSED! Data verification successful.
```

This comprehensive verification approach ensures that your tests are not just passing, but actually validating correct, real-world data from NSE sources.