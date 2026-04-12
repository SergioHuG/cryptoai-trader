# Backtest Module — Claude Code Context

## Purpose
Validate strategy logic against historical data before risking real capital.
Phase 1 of the validation pipeline.

## Files
| File | Responsibility |
|---|---|
| `engine.py` | Core backtest runner — replays historical candles through signal agents |
| `metrics.py` | Performance metrics: Sharpe ratio, max drawdown, win rate, profit factor |
| `results/` | Output files (gitignored — can be large) |

## Backtest Parameters (Phase 1)
- **Data:** 6-12 months Kraken historical BTC/USD and ETH/USD
- **Timeframe:** 15-minute candles
- **Strategy:** EMA 9/21 crossover + RSI confirmation + volume filter
- **Risk model:** Same constants as live system (1% per trade, max 3 positions)

## Required Metrics Output
Every backtest run must report:
- Total return %
- Win rate %
- Profit factor (gross profit / gross loss)
- Maximum drawdown %
- Sharpe ratio
- Total trades
- Average trade duration
- Best/worst trade

## Critical Rules
- The backtest engine uses the SAME signal logic as `agents/technical.py` — no separate implementation
- Results are never committed to Git — only the engine and metrics code
- Backtests always run in isolation — no live data connections during backtest
- Look-ahead bias is a hard bug — never use future candle data to generate past signals

## Testing Requirements
- Engine must be tested with synthetic data of known outcomes
- Metrics calculations must have exact expected values in tests
- Look-ahead bias prevention must have explicit tests
