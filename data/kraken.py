"""
Kraken Integration — data/kraken.py

Single point of contact with Kraken Pro via CCXT.
All exchange communication goes through this module — never use raw requests.

Public endpoints (OHLCV, ticker) work without API keys.
Private endpoints (orders, balances) require KRAKEN_API_KEY + KRAKEN_API_SECRET.
"""
import os
import time
import logging
from decimal import Decimal
from datetime import datetime, timezone
from dataclasses import dataclass

import ccxt

logger = logging.getLogger(__name__)

# ── Supported assets (Phase 1) ────────────────────────────────────────────────
SUPPORTED_SYMBOLS = ["BTC/USD", "ETH/USD"]
SUPPORTED_TIMEFRAMES = ["15m", "1h", "4h", "1d"]
DEFAULT_TIMEFRAME = "15m"

# ── Rate limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT_DELAY_MS = 1000  # 1 second between paginated requests


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class Candle:
    """A single OHLCV candle."""
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timeframe: str

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume),
            "timeframe": self.timeframe,
        }


@dataclass
class Ticker:
    """Current market price snapshot."""
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume_24h: Decimal
    timestamp: datetime


# ── Kraken Client ─────────────────────────────────────────────────────────────

class KrakenClient:
    """
    CCXT-based Kraken Pro client.

    Usage:
        client = KrakenClient()                    # Public endpoints only
        client = KrakenClient(authenticated=True)  # Requires env vars
    """

    def __init__(self, authenticated: bool = False):
        config = {
            "enableRateLimit": True,
            "rateLimit": RATE_LIMIT_DELAY_MS,
        }

        if authenticated:
            api_key = os.getenv("KRAKEN_API_KEY")
            api_secret = os.getenv("KRAKEN_API_SECRET")

            if not api_key or not api_secret:
                raise ValueError(
                    "KRAKEN_API_KEY and KRAKEN_API_SECRET must be set "
                    "in environment variables for authenticated requests."
                )

            config["apiKey"] = api_key
            config["secret"] = api_secret

        self._exchange = ccxt.kraken(config)
        self._authenticated = authenticated
        logger.info(
            "KrakenClient initialized (authenticated=%s)", authenticated
        )

    # ── Public endpoints ──────────────────────────────────────────────────────

    def fetch_candles(
        self,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[Candle]:
        """
        Fetch OHLCV candles from Kraken.

        Args:
            symbol:    Trading pair e.g. "BTC/USD"
            timeframe: Candle size e.g. "15m", "1h"
            since:     Fetch candles from this datetime onwards
            limit:     Max candles to return (Kraken max is 720)

        Returns:
            List of Candle objects ordered oldest to newest
        """
        self._validate_symbol(symbol)
        self._validate_timeframe(timeframe)

        since_ms = int(since.timestamp() * 1000) if since else None

        try:
            raw = self._exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=limit,
            )
        except ccxt.NetworkError as e:
            logger.error("Network error fetching candles for %s: %s", symbol, e)
            raise
        except ccxt.ExchangeError as e:
            logger.error("Exchange error fetching candles for %s: %s", symbol, e)
            raise

        candles = [self._parse_candle(row, symbol, timeframe) for row in raw]
        logger.info(
            "Fetched %d candles for %s (%s)", len(candles), symbol, timeframe
        )
        return candles

    def fetch_ticker(self, symbol: str) -> Ticker:
        """
        Fetch current bid/ask/last price for a symbol.

        Args:
            symbol: Trading pair e.g. "BTC/USD"

        Returns:
            Ticker with current market prices
        """
        self._validate_symbol(symbol)

        try:
            raw = self._exchange.fetch_ticker(symbol)
        except ccxt.NetworkError as e:
            logger.error("Network error fetching ticker for %s: %s", symbol, e)
            raise
        except ccxt.ExchangeError as e:
            logger.error("Exchange error fetching ticker for %s: %s", symbol, e)
            raise

        return Ticker(
            symbol=symbol,
            bid=Decimal(str(raw["bid"])),
            ask=Decimal(str(raw["ask"])),
            last=Decimal(str(raw["last"])),
            volume_24h=Decimal(str(raw["baseVolume"] or 0)),
            timestamp=datetime.fromtimestamp(
                raw["timestamp"] / 1000, tz=timezone.utc
            ),
        )

    def fetch_historical_candles(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime,
    ) -> list[Candle]:
        """
        Fetch a full historical range of candles with automatic pagination.
        Respects rate limits between paginated requests.

        Args:
            symbol:    Trading pair
            timeframe: Candle size
            since:     Start datetime (inclusive)
            until:     End datetime (inclusive)

        Returns:
            All candles in the range, ordered oldest to newest
        """
        self._validate_symbol(symbol)
        self._validate_timeframe(timeframe)

        all_candles: list[Candle] = []
        current_since = since

        logger.info(
            "Fetching historical candles for %s from %s to %s",
            symbol, since.date(), until.date()
        )

        while current_since < until:
            batch = self.fetch_candles(
                symbol=symbol,
                timeframe=timeframe,
                since=current_since,
                limit=500,
            )

            if not batch:
                break

            batch = [c for c in batch if c.timestamp <= until]
            all_candles.extend(batch)

            if len(batch) < 500:
                break

            current_since = batch[-1].timestamp
            logger.debug(
                "Paginating: fetched %d total candles, cursor at %s",
                len(all_candles), current_since
            )

            time.sleep(RATE_LIMIT_DELAY_MS / 1000)

        seen: set[datetime] = set()
        unique_candles = []
        for candle in all_candles:
            if candle.timestamp not in seen:
                seen.add(candle.timestamp)
                unique_candles.append(candle)

        logger.info(
            "Historical fetch complete: %d unique candles for %s",
            len(unique_candles), symbol
        )
        return unique_candles

    # ── Private endpoints (require authentication) ────────────────────────────

    def fetch_balance(self) -> dict[str, Decimal]:
        """
        Fetch account balances. Requires authenticated client.

        Returns:
            Dict of asset to available balance
        """
        self._require_auth()

        try:
            raw = self._exchange.fetch_balance()
        except ccxt.AuthenticationError as e:
            logger.error("Authentication failed fetching balance: %s", e)
            raise

        return {
            asset: Decimal(str(info["free"]))
            for asset, info in raw.items()
            if isinstance(info, dict) and info.get("free", 0) > 0
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_candle(
        self, raw: list, symbol: str, timeframe: str
    ) -> Candle:
        """Convert raw CCXT OHLCV list to a Candle dataclass."""
        timestamp_ms, open_, high, low, close, volume = raw
        return Candle(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(
                timestamp_ms / 1000, tz=timezone.utc
            ),
            open=Decimal(str(open_)),
            high=Decimal(str(high)),
            low=Decimal(str(low)),
            close=Decimal(str(close)),
            volume=Decimal(str(volume)),
            timeframe=timeframe,
        )

    def _validate_symbol(self, symbol: str) -> None:
        if symbol not in SUPPORTED_SYMBOLS:
            raise ValueError(
                f"Symbol '{symbol}' is not supported. "
                f"Supported symbols: {SUPPORTED_SYMBOLS}"
            )

    def _validate_timeframe(self, timeframe: str) -> None:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Timeframe '{timeframe}' is not supported. "
                f"Supported timeframes: {SUPPORTED_TIMEFRAMES}"
            )

    def _require_auth(self) -> None:
        if not self._authenticated:
            raise RuntimeError(
                "This operation requires an authenticated client. "
                "Initialize with KrakenClient(authenticated=True)."
            )
