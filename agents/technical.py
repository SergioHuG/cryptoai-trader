"""
Technical Analysis Agent — agents/technical.py

Calculates EMA 9/21 crossover, RSI(14), and volume confirmation signals.
This is the primary signal generator for the Phase 1 baseline strategy.

Signal is generated when ALL THREE conditions are met:
  1. EMA 9 crosses EMA 21 (direction determines long/short)
  2. RSI(14) is in the confirmation zone (not overbought/oversold)
  3. Current volume > 20-period volume SMA

Built in Phase 1 — feature/technical-agent branch.
"""
