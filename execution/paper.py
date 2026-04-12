"""
Paper Trading Engine — execution/paper.py

Simulates trade execution using real live market data but without
placing real orders. Tracks simulated P&L, position state, and trade history.

This is the primary execution mode during Phase 2 validation.
All logic here mirrors what live.py will do in Phase 3.

Built in Phase 1 after risk gate tests pass — feature/paper-trading branch.
"""
