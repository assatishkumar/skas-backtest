# Trading Strategy Requirements

## 1. Strategy Overview
### 1.1 Core Strategy
- **Type:** Short Strangle with Trend-Following Adjustments
- **Market:** NSE (National Stock Exchange of India)
- **Instruments:** Single-stock options on NIFTY 50 constituents + NIFTY Index options for hedging
- **Capital:** ₹50,00,000 (50 lakhs)
- **Data Frequency:** Daily EOD (End of Day) data only
- **Option Pricing:** nse_options library

### 1.2 Strategy Logic
- Sell monthly strangles on 25 selected stocks
- Adjust to trend-following mode on breakouts
- Hedge portfolio with NIFTY OTM options
- Risk management via stop-loss and profit targets

## 2. Stock Selection
### 2.1 Universe
- **Source:** NIFTY 50 index constituents
- **Selected Stocks (25):** HDFCBANK, ICICIBANK, RELIANCE, INFY, BHARTIARTL, LT, ITC, TCS, SBIN, AXISBANK, KOTAKBANK, M&M, HINDUNILVR, BAJFINANCE, ZOMATO, MARUTI, SUNPHARMA, NTPC, HCLTECH, ULTRACEMCO, TITAN, TATAMOTORS, BEL, TATASTEEL, POWERGRID

## 3. Position Entry Rules
### 3.1 Entry Timing
- **Frequency:** Monthly
- **Entry Date:** First trading day of each calendar month
- **Execution Price:** Closing price on entry date
- **Expiry Selection:** Current month expiry (last Thursday of month)

### 3.2 Strike Selection Logic
- **Lookback Period:** Previous calendar month (all trading days)
- **Call Strike:** Closest available strike to previous month's highest high
- **Put Strike:** Closest available strike to previous month's lowest low

### 3.3 Initial Position Structure
Per selected stock:
- **Sell:** Call option at previous month's high strike
- **Sell:** Put option at previous month's low strike
- **Position Type:** Short Strangle
- **Margin Allocation:** ₹2,00,000 per stock

## 4. Daily Monitoring
- **Monitoring:** Daily at market close

## 5. Breakout Detection and Adjustments
### 5.1 Breakout Detection
- **Upside Breakout:** Closing price > Short Call Strike Price
- **Downside Breakout:** Closing price < Short Put Strike Price

### 5.2 Adjustment Actions
**On Upside Breakout:**
- Exit: Short call and Short put position
- Enter: Sell new ATM put option
- Result: Position becomes bullish

**On Downside Breakout:**
- Exit: Short call and Short put position
- Enter: Sell new ATM call option
- Result: Position becomes bearish

### 5.3 Adjustment Constraints
- **Maximum Adjustments:** 2 per stock per month
- **Minimum Gap:** 1 trading day between adjustments
- **Reset:** Adjustment counter resets on new month entry

## 6. Portfolio Hedging
### 6.1 Hedge Structure
- **Instrument:** NIFTY monthly options
- **Type:** Long Strangle
- **Call Strike:** 4% OTM (NIFTY spot × 1.04)
- **Put Strike:** 4% OTM (NIFTY spot × 0.96)

## 7. Exit Rules
### 7.1 Portfolio Stop Loss
- **Trigger Level:** 2% of initial capital (₹1,00,000 loss)
- **Calculation:** Sum of all positions' MTM including hedges
- **Action:** Close ALL positions immediately
- **Re-entry:** Not allowed in same month after stop loss

### 7.2 Individual Position Profit Target
- **Target:** 2% of margin deployed for that position (₹4,000 per stock)
- **Calculation:** Based on theoretical premium changes
- **Action:** Close both legs of the strangle for that stock
- **Re-entry:** No new position in that stock for remainder of the month

### 7.3 Expiry Management
- **Exit Timing:** 1 day before expiry
- **Action:** Close all positions and hedges
- **Next Step:** Enter new monthly positions on next trading day
