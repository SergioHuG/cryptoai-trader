"""
Tests for data/kraken.py

Uses mocked CCXT responses — never makes real API calls in tests.
"""
import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from data.kraken import KrakenClient, Candle, Ticker, SUPPORTED_SYMBOLS, SUPPORTED_TIMEFRAMES


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_raw_candle(
    timestamp_ms: int = 1705312800000,
    open_: float = 42000.0,
    high: float = 42500.0,
    low: float = 41800.0,
    close: float = 42300.0,
    volume: float = 125.5,
) -> list:
    """Build a raw CCXT OHLCV list."""
    return [timestamp_ms, open_, high, low, close, volume]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_exchange():
    """A mocked CCXT exchange instance."""
    return MagicMock()


@pytest.fixture
def client(mock_exchange):
    """KrakenClient with mocked CCXT exchange."""
    with patch("data.kraken.ccxt.kraken", return_value=mock_exchange):
        return KrakenClient()


@pytest.fixture
def auth_client(mock_exchange):
    """Authenticated KrakenClient with mocked CCXT exchange."""
    with patch("data.kraken.ccxt.kraken", return_value=mock_exchange):
        with patch.dict("os.environ", {
            "KRAKEN_API_KEY": "test_key",
            "KRAKEN_API_SECRET": "test_secret",
        }):
            return KrakenClient(authenticated=True)


# ── Initialization tests ──────────────────────────────────────────────────────

class TestKrakenClientInit:
    def test_initializes_without_auth(self, mock_exchange):
        with patch("data.kraken.ccxt.kraken", return_value=mock_exchange):
            client = KrakenClient()
            assert client._authenticated is False

    def test_initializes_with_auth(self, mock_exchange):
        with patch("data.kraken.ccxt.kraken", return_value=mock_exchange):
            with patch.dict("os.environ", {
                "KRAKEN_API_KEY": "key",
                "KRAKEN_API_SECRET": "secret",
            }):
                client = KrakenClient(authenticated=True)
                assert client._authenticated is True

    def test_raises_without_api_key(self, mock_exchange):
        with patch("data.kraken.ccxt.kraken", return_value=mock_exchange):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="KRAKEN_API_KEY"):
                    KrakenClient(authenticated=True)


# ── Validation tests ──────────────────────────────────────────────────────────

class TestValidation:
    def test_rejects_unsupported_symbol(self, client):
        with pytest.raises(ValueError, match="not supported"):
            client.fetch_candles("DOGE/USD")

    def test_rejects_unsupported_timeframe(self, client):
        with pytest.raises(ValueError, match="not supported"):
            client.fetch_candles("BTC/USD", timeframe="1m")

    def test_accepts_all_supported_symbols(self, client, mock_exchange):
        mock_exchange.fetch_ohlcv.return_value = [make_raw_candle()]
        for symbol in SUPPORTED_SYMBOLS:
            result = client.fetch_candles(symbol)
            assert len(result) == 1

    def test_accepts_all_supported_timeframes(self, client, mock_exchange):
        mock_exchange.fetch_ohlcv.return_value = [make_raw_candle()]
        for timeframe in SUPPORTED_TIMEFRAMES:
            result = client.fetch_candles("BTC/USD", timeframe=timeframe)
            assert len(result) == 1


# ── fetch_candles tests ───────────────────────────────────────────────────────

class TestFetchCandles:
    def test_returns_list_of_candles(self, client, mock_exchange):
        mock_exchange.fetch_ohlcv.return_value = [
            make_raw_candle(),
            make_raw_candle(timestamp_ms=1705312800000 + 900000),
        ]
        result = client.fetch_candles("BTC/USD")
        assert len(result) == 2
        assert all(isinstance(c, Candle) for c in result)

    def test_candle_fields_are_correct(self, client, mock_exchange):
        mock_exchange.fetch_ohlcv.return_value = [
            make_raw_candle(
                timestamp_ms=1705312800000,
                open_=42000.0,
                high=42500.0,
                low=41800.0,
                close=42300.0,
                volume=125.5,
            )
        ]
        candles = client.fetch_candles("BTC/USD")
        c = candles[0]
        assert c.symbol == "BTC/USD"
        assert c.open == Decimal("42000.0")
        assert c.high == Decimal("42500.0")
        assert c.low == Decimal("41800.0")
        assert c.close == Decimal("42300.0")
        assert c.volume == Decimal("125.5")
        assert c.timeframe == "15m"

    def test_timestamp_is_utc_datetime(self, client, mock_exchange):
        mock_exchange.fetch_ohlcv.return_value = [make_raw_candle(timestamp_ms=1705312800000)]
        candles = client.fetch_candles("BTC/USD")
        assert candles[0].timestamp.tzinfo == timezone.utc

    def test_uses_decimal_not_float(self, client, mock_exchange):
        mock_exchange.fetch_ohlcv.return_value = [make_raw_candle(close=42300.12345678)]
        candles = client.fetch_candles("BTC/USD")
        assert isinstance(candles[0].close, Decimal)

    def test_passes_since_as_milliseconds(self, client, mock_exchange):
        mock_exchange.fetch_ohlcv.return_value = []
        since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        client.fetch_candles("BTC/USD", since=since)
        call_kwargs = mock_exchange.fetch_ohlcv.call_args
        assert call_kwargs.kwargs["since"] == int(since.timestamp() * 1000)

    def test_returns_empty_list_when_no_data(self, client, mock_exchange):
        mock_exchange.fetch_ohlcv.return_value = []
        result = client.fetch_candles("BTC/USD")
        assert result == []


# ── fetch_ticker tests ────────────────────────────────────────────────────────

class TestFetchTicker:
    def test_returns_ticker(self, client, mock_exchange):
        mock_exchange.fetch_ticker.return_value = {
            "bid": 42000.0,
            "ask": 42010.0,
            "last": 42005.0,
            "baseVolume": 1500.5,
            "timestamp": 1705312800000,
        }
        ticker = client.fetch_ticker("BTC/USD")
        assert isinstance(ticker, Ticker)
        assert ticker.symbol == "BTC/USD"
        assert ticker.bid == Decimal("42000.0")
        assert ticker.ask == Decimal("42010.0")
        assert ticker.last == Decimal("42005.0")

    def test_rejects_unsupported_symbol(self, client):
        with pytest.raises(ValueError):
            client.fetch_ticker("DOGE/USD")


# ── fetch_historical_candles tests ───────────────────────────────────────────

class TestFetchHistoricalCandles:
    def test_deduplicates_overlapping_candles(self, client, mock_exchange):
        """Pagination overlap should not produce duplicate candles."""
        raw = [make_raw_candle(timestamp_ms=1705312800000)]
        mock_exchange.fetch_ohlcv.return_value = raw

        since = datetime(2024, 1, 15, tzinfo=timezone.utc)
        until = datetime(2024, 1, 15, 1, 0, tzinfo=timezone.utc)

        result = client.fetch_historical_candles("BTC/USD", "15m", since, until)
        timestamps = [c.timestamp for c in result]
        assert len(timestamps) == len(set(timestamps))

    def test_filters_candles_beyond_until(self, client, mock_exchange):
        """Candles after the until boundary must be excluded."""
        until = datetime(2024, 1, 15, tzinfo=timezone.utc)
        until_ms = int(until.timestamp() * 1000)

        mock_exchange.fetch_ohlcv.return_value = [
            make_raw_candle(timestamp_ms=until_ms - 900000),   # before — include
            make_raw_candle(timestamp_ms=until_ms + 900000),   # after  — exclude
        ]

        since = datetime(2024, 1, 14, tzinfo=timezone.utc)
        result = client.fetch_historical_candles("BTC/USD", "15m", since, until)
        assert all(c.timestamp <= until for c in result)


# ── Auth protection tests ─────────────────────────────────────────────────────

class TestAuthProtection:
    def test_fetch_balance_requires_auth(self, client):
        with pytest.raises(RuntimeError, match="authenticated"):
            client.fetch_balance()

    def test_fetch_balance_works_with_auth(self, auth_client, mock_exchange):
        mock_exchange.fetch_balance.return_value = {
            "USD": {"free": 10000.0, "used": 0.0, "total": 10000.0},
            "BTC": {"free": 0.5, "used": 0.0, "total": 0.5},
        }
        balance = auth_client.fetch_balance()
        assert "USD" in balance
        assert isinstance(balance["USD"], Decimal)

    def test_candle_data_to_dict(self, client, mock_exchange):
        mock_exchange.fetch_ohlcv.return_value = [make_raw_candle()]
        candles = client.fetch_candles("BTC/USD")
        d = candles[0].to_dict()
        assert "symbol" in d
        assert "timestamp" in d
        assert "open" in d
        assert isinstance(d["open"], str)  # Decimals serialized as strings
