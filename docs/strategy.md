# Strategy Documentation

## Phase 1 Baseline Strategy

### Name
EMA Crossover with RSI + Volume Confirmation

### Timeframe
15-minute candles

### Assets
BTC/USD, ETH/USD on Kraken Pro

---

## Entry Conditions (ALL must be true)
1. **EMA Crossover:** EMA 9 crosses above EMA 21 (long) or below (short)
2. **RSI Confirmation:**
   - Long: RSI(14) is between 40-70 (not oversold bounce, not overbought entry)
   - Short: RSI(14) is between 30-60 (not overbought bounce, not oversold entry)
3. **Volume Confirmation:** Current candle volume > 20-period volume SMA

## Exit Conditions
- **Stop Loss:** Below the most recent swing low (long) / above swing high (short)
- **Take Profit:** 2x the risk distance from entry (1:2 R/R ratio)
- **Time stop:** Position closed at end of trading session if neither SL nor TP hit

## Risk Parameters
- Max risk per trade: 1% of total capital
- Position size calculated from: `(capital * 0.01) / (entry - stop_loss)`

---

## Backtest Targets (Phase 1 Pass Criteria)
Before moving to paper trading, backtesting must show:
- Win rate: > 40% (with 1:2 R/R, 40% win rate = profitable)
- Profit factor: > 1.3
- Maximum drawdown: < 15%
- Minimum sample: 100+ trades

---

## Strategy Evolution Notes
This file tracks strategy changes over time.
Never delete old entries — append new versions with dates.

### v1.0 — Initial baseline (Phase 1)
Simple EMA crossover with RSI and volume filter.
Purpose: Validate system architecture, not optimize for maximum returns.
