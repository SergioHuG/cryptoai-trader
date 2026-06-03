"""
Tests for agents/technical.py

Pure unit tests — no I/O, no database, no API calls.
Synthetic candle factories give deterministic inputs for every assertion.
"""
import pytest
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from data.kraken import Candle
from agents.technical import (
    TechnicalAnalyst,
    TechnicalSignal,
    EMA_FAST_PERIOD,
    EMA_SLOW_PERIOD,
    RSI_PERIOD,
    VOLUME_SMA_PERIOD,
    SWING_LOOKBACK,
    RISK_REWARD_RATIO,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    MIN_CANDLES,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_candle(
    price: Decimal,
    volume: Decimal = Decimal("100"),
    symbol: str = "BTC/USD",
    timeframe: str = "1h",
    index: int = 0,
    high_offset: Decimal = Decimal("100"),
    low_offset: Decimal = Decimal("100"),
) -> Candle:
    return Candle(
        symbol=symbol,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=index),
        open=price,
        high=price + high_offset,
        low=price - low_offset,
        close=price,
        volume=volume,
        timeframe=timeframe,
    )


def flat_candles(
    count: int = 50,
    price: Decimal = Decimal("42000"),
    base_volume: Decimal = Decimal("100"),
    last_volume: Decimal = Decimal("150"),
) -> list[Candle]:
    """
    Flat-price candles. Last candle has elevated volume so ratio > 1.0.
    Used as the base for mocked analyze() tests.
    """
    candles = []
    for i in range(count):
        vol = last_volume if i == count - 1 else base_volume
        candles.append(make_candle(price=price, volume=vol, index=i))
    return candles


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def analyst():
    return TechnicalAnalyst()


@pytest.fixture
def candles():
    return flat_candles()


# ── EMA Calculation ───────────────────────────────────────────────────────────

class TestEMACalculation:
    def test_constant_prices_return_constant_ema(self, analyst):
        prices = [Decimal("100")] * 20
        result = analyst._calculate_ema(prices, 9)
        for val in result:
            assert val == Decimal("100")

    def test_output_length_is_correct(self, analyst):
        prices = [Decimal("100")] * 30
        result = analyst._calculate_ema(prices, EMA_FAST_PERIOD)
        # length = len(prices) - period + 1
        assert len(result) == 30 - EMA_FAST_PERIOD + 1

    def test_ema_reacts_faster_with_shorter_period(self, analyst):
        """EMA9 should move further toward a price spike than EMA21."""
        prices = [Decimal("100")] * 30 + [Decimal("200")]
        ema9 = analyst._calculate_ema(prices, 9)
        ema21 = analyst._calculate_ema(prices, 21)
        assert ema9[-1] > ema21[-1]

    def test_raises_on_insufficient_data(self, analyst):
        with pytest.raises(ValueError):
            analyst._calculate_ema([Decimal("100")] * 5, 9)

    def test_ema_seeds_with_sma(self, analyst):
        """First EMA value should equal SMA of first `period` prices."""
        prices = [Decimal("10"), Decimal("20"), Decimal("30")] + [Decimal("30")] * 5
        result = analyst._calculate_ema(prices, 3)
        expected_seed = (Decimal("10") + Decimal("20") + Decimal("30")) / Decimal("3")
        assert result[0] == expected_seed


# ── RSI Calculation ───────────────────────────────────────────────────────────

class TestRSICalculation:
    def test_all_gains_returns_100(self, analyst):
        prices = [Decimal(str(i * 10)) for i in range(1, 20)]
        rsi = analyst._calculate_rsi(prices, RSI_PERIOD)
        assert rsi == Decimal("100")

    def test_all_losses_returns_zero(self, analyst):
        prices = [Decimal(str(200 - i * 10)) for i in range(20)]
        rsi = analyst._calculate_rsi(prices, RSI_PERIOD)
        assert rsi == Decimal("0")

    def test_rsi_within_valid_range(self, analyst):
        prices = [Decimal("100"), Decimal("110"), Decimal("105"),
                  Decimal("115"), Decimal("108"), Decimal("112"),
                  Decimal("109"), Decimal("117"), Decimal("111"),
                  Decimal("119"), Decimal("113"), Decimal("120"),
                  Decimal("116"), Decimal("122"), Decimal("118"),
                  Decimal("125"), Decimal("120"), Decimal("128")]
        rsi = analyst._calculate_rsi(prices, RSI_PERIOD)
        assert Decimal("0") <= rsi <= Decimal("100")

    def test_raises_on_insufficient_data(self, analyst):
        with pytest.raises(ValueError):
            analyst._calculate_rsi([Decimal("100")] * 5, RSI_PERIOD)


# ── SMA Calculation ───────────────────────────────────────────────────────────

class TestSMACalculation:
    def test_known_values(self, analyst):
        values = [Decimal("10"), Decimal("20"), Decimal("30")]
        assert analyst._calculate_sma(values) == Decimal("20")

    def test_empty_list_returns_zero(self, analyst):
        assert analyst._calculate_sma([]) == Decimal("0")


# ── Crossover Detection ───────────────────────────────────────────────────────

class TestCrossoverDetection:
    def test_bullish_crossover(self, analyst):
        fast = [Decimal("99"), Decimal("101")]   # was below, now above
        slow = [Decimal("100"), Decimal("100")]
        assert analyst._detect_crossover(fast, slow) == "bullish"

    def test_bearish_crossover(self, analyst):
        fast = [Decimal("101"), Decimal("99")]   # was above, now below
        slow = [Decimal("100"), Decimal("100")]
        assert analyst._detect_crossover(fast, slow) == "bearish"

    def test_no_crossover_fast_stays_above(self, analyst):
        fast = [Decimal("102"), Decimal("101")]
        slow = [Decimal("100"), Decimal("100")]
        assert analyst._detect_crossover(fast, slow) is None

    def test_no_crossover_fast_stays_below(self, analyst):
        fast = [Decimal("98"), Decimal("99")]
        slow = [Decimal("100"), Decimal("100")]
        assert analyst._detect_crossover(fast, slow) is None

    def test_insufficient_series_returns_none(self, analyst):
        assert analyst._detect_crossover([Decimal("100")], [Decimal("100")]) is None


# ── Stop Loss ─────────────────────────────────────────────────────────────────

class TestStopLoss:
    def _make_candles_with_known_lows(self) -> list[Candle]:
        """Last 5 candles have lows: 39800, 39900, 39700, 39850, 39750."""
        lows = [Decimal("39800"), Decimal("39900"), Decimal("39700"),
                Decimal("39850"), Decimal("39750")]
        candles = [make_candle(Decimal("42000"), index=i) for i in range(45)]
        for i, low in enumerate(lows):
            candles.append(Candle(
                symbol="BTC/USD",
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=45 + i),
                open=Decimal("42000"),
                high=Decimal("42100"),
                low=low,
                close=Decimal("42000"),
                volume=Decimal("100"),
                timeframe="1h",
            ))
        return candles

    def test_long_stop_uses_swing_low(self, analyst):
        candles = self._make_candles_with_known_lows()
        sl = analyst._calculate_stop_loss(candles, "long")
        assert sl == Decimal("39700")  # min of last 5 lows

    def test_short_stop_uses_swing_high(self, analyst):
        candles = [make_candle(
            price=Decimal("42000"),
            high_offset=Decimal(str(100 + i * 10)),
            index=i,
        ) for i in range(50)]
        sl = analyst._calculate_stop_loss(candles, "short")
        expected = max(c.high for c in candles[-SWING_LOOKBACK:])
        assert sl == expected

    def test_uses_only_swing_lookback_candles(self, analyst):
        """A very low candle outside the lookback window must not affect the result."""
        candles = [make_candle(Decimal("42000"), low_offset=Decimal("5000"), index=i)
                   for i in range(44)]
        candles += [make_candle(Decimal("42000"), low_offset=Decimal("50"), index=i + 44)
                    for i in range(6)]
        sl = analyst._calculate_stop_loss(candles, "long")
        # Only last 5 candles counted — none has a huge low offset
        assert sl >= Decimal("41950")


# ── Take Profit and Risk/Reward ───────────────────────────────────────────────

class TestTakeProfitAndRR:
    def test_long_take_profit_is_2x_risk(self, analyst):
        entry = Decimal("42000")
        stop = Decimal("41000")   # risk = 1000
        tp = analyst._calculate_take_profit(entry, stop, "long")
        assert tp == Decimal("44000")  # entry + 2 * 1000

    def test_short_take_profit_is_2x_risk(self, analyst):
        entry = Decimal("42000")
        stop = Decimal("43000")   # risk = 1000
        tp = analyst._calculate_take_profit(entry, stop, "short")
        assert tp == Decimal("40000")  # entry - 2 * 1000

    def test_risk_reward_equals_constant(self, analyst):
        entry = Decimal("42000")
        stop = Decimal("41000")
        tp = analyst._calculate_take_profit(entry, stop, "long")
        rr = analyst._calculate_risk_reward(entry, stop, tp, "long")
        assert rr == RISK_REWARD_RATIO

    def test_zero_risk_returns_zero_rr(self, analyst):
        entry = Decimal("42000")
        rr = analyst._calculate_risk_reward(entry, entry, Decimal("44000"), "long")
        assert rr == Decimal("0")


# ── analyze() — gating logic ──────────────────────────────────────────────────

class TestAnalyzeGating:
    def test_returns_none_when_insufficient_candles(self, analyst):
        candles = flat_candles(count=MIN_CANDLES - 1)
        assert analyst.analyze(candles) is None

    def test_returns_none_when_no_crossover(self, analyst, candles):
        with patch.object(analyst, "_detect_crossover", return_value=None):
            assert analyst.analyze(candles) is None

    def test_returns_none_when_rsi_overbought_on_long(self, analyst, candles):
        with patch.object(analyst, "_detect_crossover", return_value="bullish"), \
             patch.object(analyst, "_calculate_rsi", return_value=RSI_OVERBOUGHT):
            assert analyst.analyze(candles) is None

    def test_returns_none_when_rsi_oversold_on_short(self, analyst, candles):
        with patch.object(analyst, "_detect_crossover", return_value="bearish"), \
             patch.object(analyst, "_calculate_rsi", return_value=RSI_OVERSOLD):
            assert analyst.analyze(candles) is None

    def test_returns_none_when_volume_below_sma(self, analyst):
        # All volumes identical → ratio = 1.0 → fails > 1.0 check
        equal_volume_candles = flat_candles(base_volume=Decimal("100"), last_volume=Decimal("100"))
        with patch.object(analyst, "_detect_crossover", return_value="bullish"), \
             patch.object(analyst, "_calculate_rsi", return_value=Decimal("50")):
            assert analyst.analyze(equal_volume_candles) is None

    def test_returns_signal_when_all_conditions_met(self, analyst, candles):
        with patch.object(analyst, "_detect_crossover", return_value="bullish"), \
             patch.object(analyst, "_calculate_rsi", return_value=Decimal("50")):
            signal = analyst.analyze(candles)
        assert signal is not None
        assert isinstance(signal, TechnicalSignal)


# ── analyze() — signal output fields ─────────────────────────────────────────

class TestAnalyzeSignalOutput:
    def _get_signal(self, analyst, candles, direction="bullish", rsi=Decimal("50")):
        with patch.object(analyst, "_detect_crossover", return_value=direction), \
             patch.object(analyst, "_calculate_rsi", return_value=rsi):
            return analyst.analyze(candles)

    def test_long_signal_on_bullish_crossover(self, analyst, candles):
        signal = self._get_signal(analyst, candles, direction="bullish")
        assert signal.direction == "long"

    def test_short_signal_on_bearish_crossover(self, analyst, candles):
        signal = self._get_signal(analyst, candles, direction="bearish")
        assert signal.direction == "short"

    def test_high_confidence_when_rsi_in_neutral_zone(self, analyst, candles):
        signal = self._get_signal(analyst, candles, rsi=Decimal("50"))
        assert signal.confidence == "high"

    def test_medium_confidence_when_rsi_near_limits(self, analyst, candles):
        signal = self._get_signal(analyst, candles, rsi=Decimal("65"))
        assert signal.confidence == "medium"

    def test_signal_risk_reward_equals_2(self, analyst, candles):
        signal = self._get_signal(analyst, candles)
        assert signal.risk_reward == RISK_REWARD_RATIO

    def test_signal_symbol_and_timeframe_match_candles(self, analyst, candles):
        signal = self._get_signal(analyst, candles)
        assert signal.symbol == candles[-1].symbol
        assert signal.timeframe == candles[-1].timeframe

    def test_signal_repr_contains_direction(self, analyst, candles):
        signal = self._get_signal(analyst, candles)
        assert "LONG" in repr(signal)
