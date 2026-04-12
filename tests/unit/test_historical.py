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
from data.historical import (
    HistoricalDataFetcher,
    KRAKEN_15M_MAX_DAYS,
    DEEP_HISTORY_TIMEFRAME,
)
from database.models import CandleModel
from database.session import create_tables, drop_tables

TEST_DB_URL = "sqlite:///:memory:"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def test_database():
    create_tables(TEST_DB_URL)
    yield
    drop_tables(TEST_DB_URL)


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def fetcher(mock_client):
    return HistoricalDataFetcher(client=mock_client)


@pytest.fixture
def sample_candles_15m():
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


@pytest.fixture
def sample_candles_1h():
    base_time = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    return [
        Candle(
            symbol="BTC/USD",
            timestamp=base_time + timedelta(hours=i),
            open=Decimal("42000.00"),
            high=Decimal("42500.00"),
            low=Decimal("41800.00"),
            close=Decimal(f"{42000 + i * 50}.00"),
            volume=Decimal("300.00"),
            timeframe="1h",
        )
        for i in range(20)
    ]


@pytest.fixture
def sample_candles(sample_candles_15m):
    return sample_candles_15m


def _mock_session(mock_session_ctx, rowcount=10):
    mock_session = MagicMock()
    mock_session_ctx.return_value.__enter__.return_value = mock_session
    mock_result = MagicMock()
    mock_result.rowcount = rowcount
    mock_session.execute.return_value = mock_result
    return mock_session


# ── fetch_and_store ───────────────────────────────────────────────────────────

class TestFetchAndStore:
    def test_stores_candles_from_kraken(self, fetcher, mock_client, sample_candles):
        mock_client.fetch_historical_candles.return_value = sample_candles
        with patch("data.historical.get_session") as mock_session_ctx:
            _mock_session(mock_session_ctx, rowcount=len(sample_candles))
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
            _mock_session(mock_session_ctx, rowcount=1)
            fetcher.fetch_and_store("BTC/USD", months=3)
        call_kwargs = mock_client.fetch_historical_candles.call_args
        since = call_kwargs.kwargs["since"]
        until = call_kwargs.kwargs["until"]
        assert 85 <= (until - since).days <= 95


# ── fetch_and_store_recent ────────────────────────────────────────────────────

class TestFetchAndStoreRecent:
    def test_fetches_and_returns_count(self, fetcher, mock_client, sample_candles_15m):
        mock_client.fetch_historical_candles.return_value = sample_candles_15m
        with patch("data.historical.get_session") as mock_session_ctx:
            _mock_session(mock_session_ctx, rowcount=len(sample_candles_15m))
            count = fetcher.fetch_and_store_recent("BTC/USD")
        assert count == len(sample_candles_15m)

    def test_always_uses_15m_timeframe(self, fetcher, mock_client, sample_candles_15m):
        mock_client.fetch_historical_candles.return_value = sample_candles_15m
        with patch("data.historical.get_session") as mock_session_ctx:
            _mock_session(mock_session_ctx, rowcount=1)
            fetcher.fetch_and_store_recent("BTC/USD")
        call_kwargs = mock_client.fetch_historical_candles.call_args
        assert call_kwargs.kwargs["timeframe"] == "15m"

    def test_date_range_within_kraken_limit(self, fetcher, mock_client, sample_candles_15m):
        mock_client.fetch_historical_candles.return_value = sample_candles_15m
        with patch("data.historical.get_session") as mock_session_ctx:
            _mock_session(mock_session_ctx, rowcount=1)
            fetcher.fetch_and_store_recent("BTC/USD")
        call_kwargs = mock_client.fetch_historical_candles.call_args
        since = call_kwargs.kwargs["since"]
        until = call_kwargs.kwargs["until"]
        assert (until - since).days <= KRAKEN_15M_MAX_DAYS

    def test_returns_zero_when_no_candles(self, fetcher, mock_client):
        mock_client.fetch_historical_candles.return_value = []
        count = fetcher.fetch_and_store_recent("BTC/USD")
        assert count == 0

    def test_custom_days_parameter(self, fetcher, mock_client, sample_candles_15m):
        mock_client.fetch_historical_candles.return_value = sample_candles_15m
        with patch("data.historical.get_session") as mock_session_ctx:
            _mock_session(mock_session_ctx, rowcount=1)
            fetcher.fetch_and_store_recent("ETH/USD", days=3)
        call_kwargs = mock_client.fetch_historical_candles.call_args
        since = call_kwargs.kwargs["since"]
        until = call_kwargs.kwargs["until"]
        assert 2 <= (until - since).days <= 4


# ── fetch_and_store_all ───────────────────────────────────────────────────────

class TestFetchAndStoreAll:
    def test_fetches_all_supported_symbols(self, fetcher):
        with patch.object(fetcher, "fetch_and_store", return_value=10) as mock_fetch:
            results = fetcher.fetch_and_store_all()
        assert "BTC/USD" in results
        assert "ETH/USD" in results
        assert mock_fetch.call_count == 2

    def test_continues_on_individual_failure(self, fetcher):
        def side_effect(symbol, *args, **kwargs):
            if symbol == "BTC/USD":
                raise Exception("Network error")
            return 5

        with patch.object(fetcher, "fetch_and_store", side_effect=side_effect):
            results = fetcher.fetch_and_store_all()

        assert results["BTC/USD"] == 0
        assert results["ETH/USD"] == 5


# ── fetch_and_store_all_deep ──────────────────────────────────────────────────

class TestFetchAndStoreAllDeep:
    def test_returns_both_timeframes_per_symbol(self, fetcher):
        with patch.object(fetcher, "fetch_and_store", return_value=100), \
             patch.object(fetcher, "fetch_and_store_recent", return_value=50):
            results = fetcher.fetch_and_store_all_deep(months=6)
        for symbol in ("BTC/USD", "ETH/USD"):
            assert results[symbol]["1h"] == 100
            assert results[symbol]["15m"] == 50

    def test_1h_phase_uses_deep_history_timeframe(self, fetcher):
        with patch.object(fetcher, "fetch_and_store", return_value=100) as mock_1h, \
             patch.object(fetcher, "fetch_and_store_recent", return_value=50):
            fetcher.fetch_and_store_all_deep(months=6)
        for call in mock_1h.call_args_list:
            assert call.kwargs.get("timeframe") == DEEP_HISTORY_TIMEFRAME

    def test_continues_if_1h_fails(self, fetcher):
        with patch.object(fetcher, "fetch_and_store", side_effect=Exception("API down")), \
             patch.object(fetcher, "fetch_and_store_recent", return_value=50):
            results = fetcher.fetch_and_store_all_deep()
        for symbol in results:
            assert results[symbol]["1h"] == 0
            assert results[symbol]["15m"] == 50

    def test_continues_if_15m_fails(self, fetcher):
        with patch.object(fetcher, "fetch_and_store", return_value=100), \
             patch.object(fetcher, "fetch_and_store_recent", side_effect=Exception("Timeout")):
            results = fetcher.fetch_and_store_all_deep()
        for symbol in results:
            assert results[symbol]["1h"] == 100
            assert results[symbol]["15m"] == 0

    def test_all_symbols_processed(self, fetcher):
        with patch.object(fetcher, "fetch_and_store", return_value=10), \
             patch.object(fetcher, "fetch_and_store_recent", return_value=5):
            results = fetcher.fetch_and_store_all_deep()
        assert set(results.keys()) == {"BTC/USD", "ETH/USD"}


# ── CandleModel ───────────────────────────────────────────────────────────────

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