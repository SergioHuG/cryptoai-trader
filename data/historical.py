"""
Historical Data Fetcher — data/historical.py

Fetches OHLCV candles from Kraken and stores them in PostgreSQL.
This feeds the backtest engine in Phase 1.

Two-phase seeding strategy:
  - 1h candles: 6-12 months deep history (Kraken allows this)
  - 15m candles: last 7 days only (Kraken hard limit ~720 candles)

Usage:
    python -m data.historical              # Seed 1h (6mo) + 15m (7 days)
    python -m data.historical --months 12  # Seed 1h (12mo) + 15m (7 days)
    python -m data.historical --timeframe 1h --months 6  # Single timeframe
"""
import logging
from datetime import datetime, timezone, timedelta

from sqlalchemy.dialects.postgresql import insert as pg_insert

from data.kraken import KrakenClient, Candle, SUPPORTED_SYMBOLS, DEFAULT_TIMEFRAME
from database.models import CandleModel
from database.session import get_session, create_tables

logger = logging.getLogger(__name__)

# Kraken's 15m endpoint returns ~720 candles max (~7.5 days)
KRAKEN_15M_MAX_DAYS = 7
DEEP_HISTORY_TIMEFRAME = "1h"


class HistoricalDataFetcher:
    """
    Fetches historical candles from Kraken and stores in PostgreSQL.
    Uses upsert to safely re-run without creating duplicates.
    """

    def __init__(self, client: KrakenClient | None = None):
        self._client = client or KrakenClient()

    def fetch_and_store(
        self,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
        months: int = 6,
    ) -> int:
        """
        Fetch historical candles for a symbol and store in the database.

        Args:
            symbol:    Trading pair e.g. "BTC/USD"
            timeframe: Candle size e.g. "15m" or "1h"
            months:    How many months of history to fetch

        Returns:
            Number of candles stored
        """
        until = datetime.now(timezone.utc)
        since = until - timedelta(days=months * 30)

        logger.info(
            "Fetching %d months of %s %s candles from Kraken...",
            months, symbol, timeframe
        )

        candles = self._client.fetch_historical_candles(
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            until=until,
        )

        if not candles:
            logger.warning("No candles returned for %s", symbol)
            return 0

        stored = self._store_candles(candles)
        logger.info(
            "Stored %d candles for %s (%s)", stored, symbol, timeframe
        )
        return stored

    def fetch_and_store_recent(
        self,
        symbol: str,
        days: int = KRAKEN_15M_MAX_DAYS,
    ) -> int:
        """
        Fetch recent 15m candles (within Kraken's ~720-candle limit).

        Args:
            symbol: Trading pair e.g. "BTC/USD"
            days:   Number of recent days to fetch (max ~7 for 15m)

        Returns:
            Number of candles stored
        """
        until = datetime.now(timezone.utc)
        since = until - timedelta(days=days)

        logger.info(
            "Fetching %d days of recent 15m candles for %s...", days, symbol
        )

        candles = self._client.fetch_historical_candles(
            symbol=symbol,
            timeframe=DEFAULT_TIMEFRAME,
            since=since,
            until=until,
        )

        if not candles:
            logger.warning("No recent 15m candles returned for %s", symbol)
            return 0

        stored = self._store_candles(candles)
        logger.info("Stored %d recent 15m candles for %s", stored, symbol)
        return stored

    def fetch_and_store_all(
        self,
        timeframe: str = DEFAULT_TIMEFRAME,
        months: int = 6,
    ) -> dict[str, int]:
        """
        Fetch and store historical candles for all supported symbols.

        Returns:
            Dict of symbol -> candles stored
        """
        results = {}
        for symbol in SUPPORTED_SYMBOLS:
            try:
                count = self.fetch_and_store(symbol, timeframe, months)
                results[symbol] = count
            except Exception as e:
                logger.error("Failed to fetch %s: %s", symbol, e)
                results[symbol] = 0
        return results

    def fetch_and_store_all_deep(self, months: int = 6) -> dict[str, dict[str, int]]:
        """
        Two-phase seed for all symbols:
          Phase A — 1h candles going back `months` months (deep backtest history)
          Phase B — 15m candles for the last 7 days (live trading resolution)

        Returns:
            Dict of symbol -> {"1h": count, "15m": count}
        """
        results = {}
        for symbol in SUPPORTED_SYMBOLS:
            results[symbol] = {"1h": 0, "15m": 0}
            try:
                count_1h = self.fetch_and_store(
                    symbol, timeframe=DEEP_HISTORY_TIMEFRAME, months=months
                )
                results[symbol]["1h"] = count_1h
            except Exception as e:
                logger.error("Failed 1h fetch for %s: %s", symbol, e)

            try:
                count_15m = self.fetch_and_store_recent(symbol)
                results[symbol]["15m"] = count_15m
            except Exception as e:
                logger.error("Failed 15m fetch for %s: %s", symbol, e)

        return results

    def get_candle_count(self, symbol: str, timeframe: str = DEFAULT_TIMEFRAME) -> int:
        """Return how many candles are stored for a symbol."""
        with get_session() as session:
            return (
                session.query(CandleModel)
                .filter(
                    CandleModel.symbol == symbol,
                    CandleModel.timeframe == timeframe,
                )
                .count()
            )

    def get_date_range(
        self, symbol: str, timeframe: str = DEFAULT_TIMEFRAME
    ) -> tuple[datetime | None, datetime | None]:
        """Return the earliest and latest candle timestamps for a symbol."""
        with get_session() as session:
            from sqlalchemy import func
            result = session.query(
                func.min(CandleModel.timestamp),
                func.max(CandleModel.timestamp),
            ).filter(
                CandleModel.symbol == symbol,
                CandleModel.timeframe == timeframe,
            ).one()
            return result[0], result[1]

    def _store_candles(self, candles: list[Candle]) -> int:
        """
        Upsert candles into the database.
        Uses PostgreSQL INSERT ... ON CONFLICT DO NOTHING to skip duplicates.
        Safe to run multiple times without creating duplicate rows.
        """
        rows = [
            {
                "symbol": c.symbol,
                "timeframe": c.timeframe,
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]

        with get_session() as session:
            stmt = pg_insert(CandleModel).values(rows)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["symbol", "timeframe", "timestamp"]
            )
            result = session.execute(stmt)
            return result.rowcount if result.rowcount >= 0 else len(rows)


def seed_database(months: int = 6, timeframe: str | None = None) -> None:
    """
    Main entry point for seeding the database with historical data.

    If timeframe is None (default): runs two-phase deep seed.
      - 1h candles for `months` months (backtest history)
      - 15m candles for last 7 days (recent live resolution)

    If timeframe is specified: single-timeframe fetch only.
    """
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    logger.info("Creating database tables if not exists...")
    create_tables()

    fetcher = HistoricalDataFetcher()

    if timeframe:
        # Single timeframe mode (explicit --timeframe flag)
        logger.info(
            "Starting single-timeframe seed (%d months, %s)...", months, timeframe
        )
        results_single = fetcher.fetch_and_store_all(timeframe=timeframe, months=months)
        logger.info("Seed complete:")
        for symbol, count in results_single.items():
            earliest, latest = fetcher.get_date_range(symbol, timeframe)
            logger.info(
                "  %s [%s]: %d candles (%s → %s)",
                symbol, timeframe, count,
                earliest.date() if earliest else "N/A",
                latest.date() if latest else "N/A",
            )
    else:
        # Two-phase deep seed (default)
        logger.info(
            "Starting two-phase deep seed: 1h (%d months) + 15m (7 days)...", months
        )
        results = fetcher.fetch_and_store_all_deep(months=months)
        logger.info("Seed complete:")
        for symbol, counts in results.items():
            for tf, count in counts.items():
                earliest, latest = fetcher.get_date_range(symbol, tf)
                logger.info(
                    "  %s [%s]: %d candles (%s → %s)",
                    symbol, tf, count,
                    earliest.date() if earliest else "N/A",
                    latest.date() if latest else "N/A",
                )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed historical candle data")
    parser.add_argument("--months", type=int, default=6, help="Months of history to fetch")
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Candle timeframe (default: two-phase 1h+15m seed)"
    )
    args = parser.parse_args()
    seed_database(months=args.months, timeframe=args.timeframe)