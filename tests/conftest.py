"""
Shared pytest fixtures for CryptoAI Trader test suite.
All tests import fixtures from here via pytest's automatic conftest discovery.
"""
import pytest
from decimal import Decimal
from datetime import datetime, timezone


# ── Candle fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_candle():
    """A single realistic BTC/USD 15min candle."""
    return {
        "symbol": "BTC/USD",
        "timestamp": datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
        "open": Decimal("42000.00"),
        "high": Decimal("42500.00"),
        "low": Decimal("41800.00"),
        "close": Decimal("42300.00"),
        "volume": Decimal("125.50"),
        "timeframe": "15m",
    }


@pytest.fixture
def bullish_candles():
    """Series of candles with a clear uptrend — EMA 9 should cross above EMA 21."""
    base_price = Decimal("40000.00")
    candles = []
    for i in range(30):
        price = base_price + Decimal(str(i * 100))
        candles.append({
            "symbol": "BTC/USD",
            "timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "open": price,
            "high": price + Decimal("200"),
            "low": price - Decimal("100"),
            "close": price + Decimal("150"),
            "volume": Decimal("100.00"),
            "timeframe": "15m",
        })
    return candles


@pytest.fixture
def bearish_candles():
    """Series of candles with a clear downtrend — EMA 9 should cross below EMA 21."""
    base_price = Decimal("45000.00")
    candles = []
    for i in range(30):
        price = base_price - Decimal(str(i * 100))
        candles.append({
            "symbol": "BTC/USD",
            "timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "open": price,
            "high": price + Decimal("100"),
            "low": price - Decimal("200"),
            "close": price - Decimal("150"),
            "volume": Decimal("100.00"),
            "timeframe": "15m",
        })
    return candles


# ── Risk parameter fixtures ───────────────────────────────────────────────────

@pytest.fixture
def risk_constants():
    """Hardcoded risk constants — tests use these, never hardcode values in tests."""
    return {
        "MAX_RISK_PER_TRADE_PCT": 0.01,
        "MAX_CONCURRENT_POSITIONS": 3,
        "DAILY_DRAWDOWN_KILL_SWITCH_PCT": 0.05,
        "SIGNAL_TIMEOUT_SECONDS": 90,
    }


@pytest.fixture
def sample_account():
    """Sample account state for position sizing and risk tests."""
    return {
        "total_capital": Decimal("10000.00"),
        "available_capital": Decimal("10000.00"),
        "open_positions": 0,
        "daily_pnl": Decimal("0.00"),
        "daily_pnl_pct": 0.0,
    }


# ── Signal fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_long_signal():
    """A valid long signal that should pass risk gate."""
    return {
        "symbol": "BTC/USD",
        "direction": "long",
        "entry_price": Decimal("42300.00"),
        "stop_loss": Decimal("41800.00"),
        "take_profit": Decimal("43300.00"),
        "risk_reward_ratio": Decimal("2.0"),
        "confidence": 0.75,
        "agent_reasoning": {
            "technical": "EMA 9 crossed above EMA 21, RSI at 55, volume 40% above average",
            "sentiment": "Neutral-positive, no major news events",
            "fundamental": "Network activity stable, no red flags",
        },
    }


@pytest.fixture
def sample_short_signal():
    """A valid short signal that should pass risk gate."""
    return {
        "symbol": "ETH/USD",
        "direction": "short",
        "entry_price": Decimal("2200.00"),
        "stop_loss": Decimal("2280.00"),
        "take_profit": Decimal("2040.00"),
        "risk_reward_ratio": Decimal("2.0"),
        "confidence": 0.68,
        "agent_reasoning": {
            "technical": "EMA 9 crossed below EMA 21, RSI at 42, volume above average",
            "sentiment": "Slightly negative, FUD on social media",
            "fundamental": "No significant fundamental changes",
        },
    }
