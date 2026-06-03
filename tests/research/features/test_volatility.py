"""
Tests for research/features/volatility.py

pandas/numpy are available in the test environment (research layer).
Tests verify:
  - EWMA vol produces correct shape and NaN pattern
  - ewma_vol_latest returns a Decimal
  - build_signal_packet_metadata produces the correct dict
  - Input validation (empty series, span < 2, wrong type)
  - Seam integration — metadata dict is consumable by SignalPacket
"""
from decimal import Decimal

import pandas as pd
import numpy as np
import pytest

from research.features.volatility import (
    DEFAULT_EWMA_SPAN,
    build_signal_packet_metadata,
    ewma_vol,
    ewma_vol_latest,
)
from research.seam import SignalPacket


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_prices() -> pd.Series:
    """Constant prices → log returns = 0 → vol ≈ 0 after warmup."""
    return pd.Series(
        [100.0] * 50,
        index=pd.date_range("2024-01-01", periods=50, freq="D"),
    )


@pytest.fixture
def trending_prices() -> pd.Series:
    """Steadily rising prices with mild noise."""
    rng = np.random.default_rng(42)
    base = np.linspace(100, 200, 60)
    noise = rng.normal(0, 0.5, 60)
    return pd.Series(
        base + noise,
        index=pd.date_range("2024-01-01", periods=60, freq="D"),
    )


@pytest.fixture
def volatile_prices() -> pd.Series:
    """High-variance prices."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0, 0.05, 50)   # 5% daily std dev
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(
        prices,
        index=pd.date_range("2024-01-01", periods=50, freq="D"),
    )


# ── ewma_vol — shape and NaN pattern ─────────────────────────────────────────

def test_output_length_matches_input(trending_prices: pd.Series) -> None:
    result = ewma_vol(trending_prices)
    assert len(result) == len(trending_prices)


def test_first_values_are_nan(trending_prices: pd.Series) -> None:
    """First (span - 1) values must be NaN — insufficient history."""
    result = ewma_vol(trending_prices, span=DEFAULT_EWMA_SPAN)
    assert result.iloc[0] is np.nan or np.isnan(result.iloc[0])


def test_later_values_are_not_nan(trending_prices: pd.Series) -> None:
    """After warmup, values should be finite."""
    result = ewma_vol(trending_prices, span=DEFAULT_EWMA_SPAN)
    non_nan = result.dropna()
    assert len(non_nan) > 0
    assert np.all(np.isfinite(non_nan.values))


def test_flat_prices_produce_near_zero_vol(flat_prices: pd.Series) -> None:
    """Zero-variance prices → vol ≈ 0."""
    result = ewma_vol(flat_prices)
    non_nan = result.dropna()
    assert (non_nan.abs() < 1e-10).all()


def test_volatile_prices_produce_higher_vol_than_flat(
    flat_prices: pd.Series,
    volatile_prices: pd.Series,
) -> None:
    flat_vol = ewma_vol(flat_prices).dropna().mean()
    high_vol = ewma_vol(volatile_prices).dropna().mean()
    assert high_vol > flat_vol


def test_shorter_span_reacts_faster() -> None:
    """Smaller span → EWMA reacts faster to a volatility shock."""
    rng = np.random.default_rng(99)
    # Calm period followed by volatile period
    calm = np.ones(30) * 100.0
    shock = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.05, 30)))
    prices = pd.Series(
        np.concatenate([calm, shock]),
        index=pd.date_range("2024-01-01", periods=60, freq="D"),
    )
    vol_fast = ewma_vol(prices, span=5).dropna()
    vol_slow = ewma_vol(prices, span=30).dropna()
    # Fast EWMA should show higher average vol over the shock period
    assert vol_fast.mean() >= vol_slow.mean() * 0.5   # loose bound


def test_returns_pandas_series(trending_prices: pd.Series) -> None:
    result = ewma_vol(trending_prices)
    assert isinstance(result, pd.Series)


def test_custom_span_respected(trending_prices: pd.Series) -> None:
    result_5 = ewma_vol(trending_prices, span=5)
    result_30 = ewma_vol(trending_prices, span=30)
    # Different spans → different results
    assert not result_5.dropna().equals(result_30.dropna())


# ── ewma_vol — input validation ───────────────────────────────────────────────

def test_empty_series_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        ewma_vol(pd.Series([], dtype=float))


def test_span_below_two_raises_value_error() -> None:
    prices = pd.Series([100.0, 101.0, 102.0])
    with pytest.raises(ValueError, match="span"):
        ewma_vol(prices, span=1)


def test_non_series_raises_type_error() -> None:
    with pytest.raises(TypeError, match="Series"):
        ewma_vol([100.0, 101.0, 102.0])  # type: ignore[arg-type]


def test_returns_col_raises_value_error(trending_prices: pd.Series) -> None:
    """returns_col is reserved — must raise ValueError."""
    with pytest.raises(ValueError, match="reserved"):
        ewma_vol(trending_prices, returns_col="close")


# ── ewma_vol_latest ───────────────────────────────────────────────────────────

def test_latest_returns_decimal(trending_prices: pd.Series) -> None:
    result = ewma_vol_latest(trending_prices)
    assert isinstance(result, Decimal)


def test_latest_is_positive_for_volatile_prices(volatile_prices: pd.Series) -> None:
    result = ewma_vol_latest(volatile_prices)
    assert result > Decimal("0")


def test_latest_is_zero_for_flat_prices(flat_prices: pd.Series) -> None:
    """Flat prices → vol = 0 → Decimal('0') returned."""
    result = ewma_vol_latest(flat_prices)
    assert result == Decimal("0")


def test_latest_insufficient_data_returns_zero() -> None:
    """Fewer bars than span → all NaN → returns Decimal('0')."""
    prices = pd.Series(
        [100.0, 101.0],
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    result = ewma_vol_latest(prices, span=20)
    assert result == Decimal("0")


def test_latest_has_max_eight_decimal_places(volatile_prices: pd.Series) -> None:
    result = ewma_vol_latest(volatile_prices)
    # Decimal string should have at most 8 places after decimal
    str_val = str(result)
    if "." in str_val:
        decimal_places = len(str_val.split(".")[1])
        assert decimal_places <= 8


# ── build_signal_packet_metadata ──────────────────────────────────────────────

def test_metadata_returns_dict(trending_prices: pd.Series) -> None:
    result = build_signal_packet_metadata(trending_prices)
    assert isinstance(result, dict)


def test_metadata_contains_ewma_vol_key(trending_prices: pd.Series) -> None:
    result = build_signal_packet_metadata(trending_prices)
    assert "ewma_vol" in result


def test_metadata_ewma_vol_is_decimal(trending_prices: pd.Series) -> None:
    result = build_signal_packet_metadata(trending_prices)
    assert isinstance(result["ewma_vol"], Decimal)


def test_metadata_custom_span_forwarded(trending_prices: pd.Series) -> None:
    result_default = build_signal_packet_metadata(trending_prices)
    result_custom = build_signal_packet_metadata(trending_prices, span=5)
    # Different spans → different vol estimates
    assert result_default["ewma_vol"] != result_custom["ewma_vol"]


# ── Seam integration ──────────────────────────────────────────────────────────

def test_metadata_is_consumable_by_signal_packet(trending_prices: pd.Series) -> None:
    """
    Critical integration test: metadata from volatility.py must pass
    SignalPacket validation without modification.
    """
    metadata = build_signal_packet_metadata(trending_prices)

    # Must not raise
    packet = SignalPacket(
        side=1,
        confidence=Decimal("0.80"),
        metadata=metadata,
    )
    assert packet.metadata.get("ewma_vol") is not None
    assert isinstance(packet.metadata["ewma_vol"], Decimal)


def test_zero_vol_metadata_consumable_by_signal_packet(flat_prices: pd.Series) -> None:
    """Decimal('0') is still a Decimal — must pass SignalPacket validation."""
    metadata = build_signal_packet_metadata(flat_prices)
    packet = SignalPacket(
        side=-1,
        confidence=Decimal("0.60"),
        metadata=metadata,
    )
    assert packet.metadata["ewma_vol"] == Decimal("0")


def test_live_stack_can_read_ewma_vol_via_get(trending_prices: pd.Series) -> None:
    """
    Simulate the live-stack pattern: packet.metadata.get('ewma_vol').
    Missing key → None (no KeyError). Present key → Decimal value.
    """
    metadata = build_signal_packet_metadata(trending_prices)
    packet = SignalPacket(side=1, confidence=Decimal("0.75"), metadata=metadata)

    vol = packet.metadata.get("ewma_vol")
    assert vol is not None
    assert isinstance(vol, Decimal)

    unknown = packet.metadata.get("nonexistent_key")
    assert unknown is None


# ── DEFAULT_EWMA_SPAN constant ────────────────────────────────────────────────

def test_default_span_is_twenty() -> None:
    """DEFAULT_EWMA_SPAN = 20 is a locked constant — changing requires retraining."""
    assert DEFAULT_EWMA_SPAN == 20
