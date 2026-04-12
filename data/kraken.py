"""
Kraken Integration — data/kraken.py

Single point of contact with Kraken Pro via CCXT.
All exchange communication goes through this module — never use raw requests.

Provides:
  - OHLCV candle fetching (historical and live)
  - Account balance queries
  - Order placement and cancellation
  - WebSocket feed subscription

API keys come from os.getenv() — never hardcoded.

Built in Phase 1 — feature/data-pipeline branch.
"""
