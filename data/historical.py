"""
Historical Data Fetcher — data/historical.py

Fetches OHLCV candles from Kraken and stores them in PostgreSQL.
This feeds the backtest engine in Phase 1.

Usage:
    python -m data.historical          # Fetch last 6 months for all assets
    python -m data.historical --months 12  # Fetch last 12 months
"""
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from sqlalchemy.dialects.postgresql import insert as pg_insert

from data.kraken import KrakenClient, Candle, SUPPORTED_SYMBOLS, DEFAULT_TIMEFRAME
from database.models import CandleModel
from database.session import get_session, create_tables

logger = logging.getLogger(__name__)


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
            timeframe: Candle size e.g. "15m"
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


def seed_database(months: int = 6, timeframe: str = DEFAULT_TIMEFRAME) -> None:
    # Load .env file if present
    from dotenv import load_dotenv
    load_dotenv()

    """
    Main entry point for seeding the database with historical data.
    Creates tables if they don't exist, then fetches all assets.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    logger.info("Creating database tables if not exists...")
    create_tables()

    fetcher = HistoricalDataFetcher()

    logger.info("Starting historical data seed (%d months, %s)...", months, timeframe)
    results = fetcher.fetch_and_store_all(timeframe=timeframe, months=months)

    logger.info("Seed complete:")
    for symbol, count in results.items():
        earliest, latest = fetcher.get_date_range(symbol, timeframe)
        logger.info(
            "  %s: %d candles (%s → %s)",
            symbol, count,
            earliest.date() if earliest else "N/A",
            latest.date() if latest else "N/A",
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed historical candle data")
    parser.add_argument("--months", type=int, default=6, help="Months of history to fetch")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe")
    args = parser.parse_args()
    seed_database(months=args.months, timeframe=args.timeframe)
