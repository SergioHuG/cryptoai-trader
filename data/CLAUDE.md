# Data Module — Claude Code Context

## Purpose
All data ingestion, streaming, historical fetching, and caching lives here.
Nothing outside this module talks directly to Kraken or Redis.

## Files
| File | Responsibility |
|---|---|
| `kraken.py` | CCXT Kraken Pro integration — single source of truth for exchange communication |
| `feeds.py` | Real-time WebSocket data streaming, 15-minute candle assembly |
| `historical.py` | Historical OHLCV data fetching for backtesting |
| `cache.py` | Redis caching layer — signal state, recent candles, position state |

## Critical Rules
- CCXT is the only library that talks to Kraken — never use raw requests to Kraken API
- All API keys come from `os.getenv()` — never hardcoded
- Historical data fetching respects Kraken rate limits — always add delays between paginated requests
- Redis cache has explicit TTLs on every key — nothing persists indefinitely
- Data is always validated with Pydantic before leaving this module

## Assets (Phase 1)
- BTC/USD
- ETH/USD
- Timeframe: 15-minute candles

## Data Schema (OHLCV)
```python
class Candle(BaseModel):
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timeframe: str
```

## Testing Requirements
- Kraken integration tests use recorded responses (VCR cassettes or mocked CCXT)
- Never make real API calls in unit tests
- Historical fetcher must be tested with pagination edge cases
