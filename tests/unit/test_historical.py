"""
Tests for data/historical.py

Uses mocked KrakenClient and an in-memory SQLite database.
Never makes real API calls or touches the development database.
"""
import pytest
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from data.kraken import Candle
from data.historical import HistoricalDataFetcher
from database.models import CandleModel
from database.session import create_tables, drop_tables, get_session


# ── Test database URL (SQLite in-memory for speed) ────────────────────────────
TEST_DB_URL = "sqlite:///:memory:"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def test_database():
    """
    Create a fresh in-memory SQLite database for each test.
    Automatically tears down after each test.
    """
    create_tables(TEST_DB_URL)
    yield
    drop_tables(TEST_DB_URL)


@pytest.fixture
def mock_client():
    """A mocked KrakenClient."""
    return MagicMock()


@pytest.fixture
def fetcher(mock_client):
    """HistoricalDataFetcher with mocked Kraken client."""
    return HistoricalDataFetcher(client=mock_client)


@pytest.fixture
def sample_candles():
    """A batch of BTC/USD candles for testing."""
    base_time = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
    return [
        Candle(
            symbol="BTC/USD",
            timestamp=base_time + timedelta(minutes=15 * i),
            open=Decimal("42000.00"),
            high=Decimal("42500.00"),
            low=Decimal("41800.00"),
            close=Decimal(f"{42000 + i * 100}.00"),
            volume=Decimal("125.50"),
            timeframe="15m",
        )
        for i in range(10)
    ]


# ── fetch_and_store tests ─────────────────────────────────────────────────────

class TestFetchAndStore:
    def test_stores_candles_from_kraken(self, fetcher, mock_client, sample_candles):
        mock_client.fetch_historical_candles.return_value = sample_candles

        with patch("data.historical.get_session") as mock_session_ctx:
            mock_session = MagicMock()
            mock_session_ctx.return_value.__enter__.return_value = mock_session
            mock_result = MagicMock()
            mock_result.rowcount = len(sample_candles)
            mock_session.execute.return_value = mock_result

            count = fetcher.fetch_and_store("BTC/USD", months=6)

        assert count == len(sample_candles)
        mock_client.fetch_historical_candles.assert_called_once()

    def test_returns_zero_when_no_candles(self, fetcher, mock_client):
        mock_client.fetch_historical_candles.return_value = []
        count = fetcher.fetch_and_store("BTC/USD", months=6)
        assert count == 0

    def test_passes_correct_date_range(self, fetcher, mock_client, sample_candles):
        mock_client.fetch_historical_candles.return_value = sample_candles

        with patch("data.historical.get_session") as mock_session_ctx:
            mock_session = MagicMock()
            mock_session_ctx.return_value.__enter__.return_value = mock_session
            mock_result = MagicMock()
            mock_result.rowcount = 1
            mock_session.execute.return_value = mock_result

            fetcher.fetch_and_store("BTC/USD", months=3)

        call_kwargs = mock_client.fetch_historical_candles.call_args
        since = call_kwargs.kwargs["since"]
        until = call_kwargs.kwargs["until"]
        diff_days = (until - since).days
        assert 85 <= diff_days <= 95  # ~3 months ± a few days


# ── fetch_and_store_all tests ─────────────────────────────────────────────────

class TestFetchAndStoreAll:
    def test_fetches_all_supported_symbols(self, fetcher, mock_client, sample_candles):
        mock_client.fetch_historical_candles.return_value = sample_candles

        with patch.object(fetcher, "fetch_and_store", return_value=10) as mock_fetch:
            results = fetcher.fetch_and_store_all()

        assert "BTC/USD" in results
        assert "ETH/USD" in results
        assert mock_fetch.call_count == 2

    def test_continues_on_individual_failure(self, fetcher, mock_client):
        """One symbol failing should not stop the others from fetching."""
        def side_effect(symbol, *args, **kwargs):
            if symbol == "BTC/USD":
                raise Exception("Network error")
            return 5

        with patch.object(fetcher, "fetch_and_store", side_effect=side_effect):
            results = fetcher.fetch_and_store_all()

        assert results["BTC/USD"] == 0  # Failed gracefully
        assert results["ETH/USD"] == 5  # Still succeeded


# ── Candle model tests ────────────────────────────────────────────────────────

class TestCandleModel:
    def test_repr(self):
        candle = CandleModel(
            symbol="BTC/USD",
            timeframe="15m",
            timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            open=Decimal("42000"),
            high=Decimal("42500"),
            low=Decimal("41800"),
            close=Decimal("42300"),
            volume=Decimal("125.5"),
        )
        assert "BTC/USD" in repr(candle)
        assert "15m" in repr(candle)
