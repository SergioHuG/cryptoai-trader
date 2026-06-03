"""
Technical Analysis Agent — agents/technical.py

Calculates EMA 9/21 crossover, RSI(14), and volume confirmation signals.
This is the primary signal generator for the Phase 1 baseline strategy.

Signal is generated when ALL THREE conditions are met:
  1. EMA 9 crosses EMA 21 (direction determines long/short)
  2. RSI(14) is in the confirmation zone (< 70 for long, > 30 for short)
  3. Current volume > 20-period volume SMA

Pure Python Decimal arithmetic — no pandas, no float precision issues.
Timeframe-agnostic: works on 1h (backtest) and 15m (live) candles.
"""
import logging
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Literal

from data.kraken import Candle

logger = logging.getLogger(__name__)

# ── Strategy constants ────────────────────────────────────────────────────────
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 21
RSI_PERIOD = 14
VOLUME_SMA_PERIOD = 20
RISK_REWARD_RATIO = Decimal("2.0")
SWING_LOOKBACK = 5          # candles used to find stop-loss swing high/low
RSI_OVERBOUGHT = Decimal("70")
RSI_OVERSOLD = Decimal("30")
RSI_HIGH_CONFIDENCE_LOW = Decimal("40")
RSI_HIGH_CONFIDENCE_HIGH = Decimal("60")

# Minimum candles needed for all indicators to be valid
# max(EMA_SLOW_PERIOD + 1 crossover check, RSI_PERIOD + 1, VOLUME_SMA_PERIOD) + buffer
MIN_CANDLES = 35


# ── Output model ──────────────────────────────────────────────────────────────

@dataclass
class TechnicalSignal:
    """
    A fully-formed trade signal from the technical analyst.
    All prices are Decimal for exact arithmetic downstream.
    """
    symbol: str
    timeframe: str
    direction: Literal["long", "short"]
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    risk_reward: Decimal
    ema_fast: Decimal          # EMA 9 at signal candle
    ema_slow: Decimal          # EMA 21 at signal candle
    rsi: Decimal               # RSI(14) at signal candle
    volume_ratio: Decimal      # current volume / volume SMA (> 1.0 means above average)
    confidence: Literal["high", "medium"]

    def __repr__(self) -> str:
        return (
            f"TechnicalSignal({self.symbol} {self.direction.upper()} | "
            f"entry={self.entry_price} sl={self.stop_loss} tp={self.take_profit} | "
            f"RSI={self.rsi:.1f} vol_ratio={self.volume_ratio:.2f} [{self.confidence}])"
        )


# ── Main class ────────────────────────────────────────────────────────────────

class TechnicalAnalyst:
    """
    Stateless technical analysis engine.
    Call analyze() with a list of Candle objects (oldest first).
    Returns a TechnicalSignal if all conditions are met, or None.
    """

    def analyze(self, candles: list[Candle]) -> TechnicalSignal | None:
        """
        Run full technical analysis on a candle series.

        Args:
            candles: Ordered list of Candle objects, oldest first.
                     Must all share the same symbol and timeframe.
                     Minimum MIN_CANDLES (35) required.

        Returns:
            TechnicalSignal if all three conditions are met, else None.
        """
        if len(candles) < MIN_CANDLES:
            logger.debug(
                "Insufficient candles: need %d, got %d", MIN_CANDLES, len(candles)
            )
            return None

        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]

        # ── Indicators ────────────────────────────────────────────────────────
        ema_fast_series = self._calculate_ema(closes, EMA_FAST_PERIOD)
        ema_slow_series = self._calculate_ema(closes, EMA_SLOW_PERIOD)
        rsi = self._calculate_rsi(closes, RSI_PERIOD)
        volume_sma = self._calculate_sma(volumes[-VOLUME_SMA_PERIOD:])

        current_ema_fast = ema_fast_series[-1]
        current_ema_slow = ema_slow_series[-1]
        current_volume = volumes[-1]
        volume_ratio = current_volume / volume_sma if volume_sma > 0 else Decimal("0")

        # ── Condition 1: EMA crossover ────────────────────────────────────────
        crossover = self._detect_crossover(ema_fast_series, ema_slow_series)
        if crossover is None:
            logger.debug("No EMA crossover detected")
            return None

        direction: Literal["long", "short"] = (
            "long" if crossover == "bullish" else "short"
        )

        # ── Condition 2: RSI confirmation ─────────────────────────────────────
        if direction == "long" and rsi >= RSI_OVERBOUGHT:
            logger.debug("Long signal rejected: RSI %.1f >= %s (overbought)", rsi, RSI_OVERBOUGHT)
            return None
        if direction == "short" and rsi <= RSI_OVERSOLD:
            logger.debug("Short signal rejected: RSI %.1f <= %s (oversold)", rsi, RSI_OVERSOLD)
            return None

        # ── Condition 3: Volume confirmation ──────────────────────────────────
        if volume_ratio <= Decimal("1"):
            logger.debug("Signal rejected: volume ratio %.2f <= 1.0", volume_ratio)
            return None

        # ── Price levels ──────────────────────────────────────────────────────
        entry = candles[-1].close
        stop_loss = self._calculate_stop_loss(candles, direction)
        take_profit = self._calculate_take_profit(entry, stop_loss, direction)
        risk_reward = self._calculate_risk_reward(entry, stop_loss, take_profit, direction)

        # ── Confidence ────────────────────────────────────────────────────────
        confidence: Literal["high", "medium"] = (
            "high"
            if RSI_HIGH_CONFIDENCE_LOW <= rsi <= RSI_HIGH_CONFIDENCE_HIGH
            else "medium"
        )

        signal = TechnicalSignal(
            symbol=candles[-1].symbol,
            timeframe=candles[-1].timeframe,
            direction=direction,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            ema_fast=current_ema_fast,
            ema_slow=current_ema_slow,
            rsi=rsi,
            volume_ratio=volume_ratio,
            confidence=confidence,
        )

        logger.info("Signal generated: %s", signal)
        return signal

    # ── Private helpers ───────────────────────────────────────────────────────

    def _calculate_ema(self, values: list[Decimal], period: int) -> list[Decimal]:
        """
        Calculate EMA for a price series using the standard multiplier.
        Seeds with SMA of the first `period` values.
        Returns a list of the same length as values (first period-1 entries are None-padded
        internally but the returned list always has len(values) entries from index period-1 onward).

        For simplicity, returns only the valid EMA values (len = len(values) - period + 1).
        """
        if len(values) < period:
            raise ValueError(f"Need at least {period} values for EMA({period})")

        multiplier = Decimal(2) / Decimal(period + 1)
        # Seed: SMA of first `period` candles
        seed_sma = sum(values[:period]) / Decimal(period)
        ema_values = [seed_sma]

        for price in values[period:]:
            ema = (price * multiplier) + (ema_values[-1] * (Decimal(1) - multiplier))
            ema_values.append(ema)

        return ema_values  # length = len(values) - period + 1

    def _calculate_rsi(self, closes: list[Decimal], period: int) -> Decimal:
        """
        Wilder's RSI using simple moving average for initial smoothing.
        Returns RSI as a Decimal 0-100.
        """
        if len(closes) < period + 1:
            raise ValueError(f"Need at least {period + 1} closes for RSI({period})")

        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [d if d > 0 else Decimal("0") for d in deltas]
        losses = [abs(d) if d < 0 else Decimal("0") for d in deltas]

        # Initial averages over first `period` changes
        avg_gain = sum(gains[:period]) / Decimal(period)
        avg_loss = sum(losses[:period]) / Decimal(period)

        # Wilder smoothing for remaining values
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * Decimal(period - 1) + gains[i]) / Decimal(period)
            avg_loss = (avg_loss * Decimal(period - 1) + losses[i]) / Decimal(period)

        if avg_loss == 0:
            return Decimal("100")

        rs = avg_gain / avg_loss
        rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))
        return rsi.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _calculate_sma(self, values: list[Decimal]) -> Decimal:
        """Simple arithmetic mean of a value list."""
        if not values:
            return Decimal("0")
        return sum(values) / Decimal(len(values))

    def _detect_crossover(
        self,
        ema_fast: list[Decimal],
        ema_slow: list[Decimal],
    ) -> Literal["bullish", "bearish"] | None:
        """
        Detect a crossover between the last two EMA values.
        Both series must have at least 2 elements.

        Bullish: fast was below slow, now above.
        Bearish: fast was above slow, now below.
        """
        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return None

        prev_fast, curr_fast = ema_fast[-2], ema_fast[-1]
        prev_slow, curr_slow = ema_slow[-2], ema_slow[-1]

        if prev_fast < prev_slow and curr_fast > curr_slow:
            return "bullish"
        if prev_fast > prev_slow and curr_fast < curr_slow:
            return "bearish"
        return None

    def _calculate_stop_loss(
        self,
        candles: list[Candle],
        direction: Literal["long", "short"],
    ) -> Decimal:
        """
        Stop loss based on recent swing high/low.
        Long  → swing low  of last SWING_LOOKBACK candles
        Short → swing high of last SWING_LOOKBACK candles
        """
        recent = candles[-SWING_LOOKBACK:]
        if direction == "long":
            return min(c.low for c in recent)
        return max(c.high for c in recent)

    def _calculate_take_profit(
        self,
        entry: Decimal,
        stop_loss: Decimal,
        direction: Literal["long", "short"],
    ) -> Decimal:
        """Take profit at exactly RISK_REWARD_RATIO × risk distance from entry."""
        risk = abs(entry - stop_loss)
        if direction == "long":
            return entry + (risk * RISK_REWARD_RATIO)
        return entry - (risk * RISK_REWARD_RATIO)

    def _calculate_risk_reward(
        self,
        entry: Decimal,
        stop_loss: Decimal,
        take_profit: Decimal,
        direction: Literal["long", "short"],
    ) -> Decimal:
        """Compute actual R/R ratio for audit — should always equal RISK_REWARD_RATIO."""
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        if risk == 0:
            return Decimal("0")
        return (reward / risk).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)