"""
EWMA Volatility — research/features/volatility.py

Implements exponentially-weighted moving standard deviation of returns
on dollar or volume bars. Used by:
  1. Labeling step  — sets stop/take-profit distances during training
  2. RiskGate (live) — reads metadata['ewma_vol'] from SignalPacket to
                       scale stop/PT widths consistently with training

The live value crosses the research → live boundary via:
    SignalPacket.metadata['ewma_vol'] = Decimal(str(round(ewma_vol, 8)))

All pandas/numpy computation stays inside this file.
research/seam.py is the ONLY import from research/ permitted in agents/.

References:
    AFML Ch.3 — Bars; Ch.17 — Structural Breaks / vol-scaling
    ADR-006 (amended) — RiskGate uses model-recommended sizes up to 1% ceiling
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Default EWMA span (days). Matches the 20-day window used in training.
# Changing this requires retraining — treat as a locked constant.
DEFAULT_EWMA_SPAN: int = 20


def ewma_vol(
    prices: "pd.Series",
    span: int = DEFAULT_EWMA_SPAN,
    returns_col: str | None = None,
) -> "pd.Series":
    """
    Compute the EWMA standard deviation of log returns.

    Works on dollar bars or volume bars — any bar type where price is
    a numeric Series. The caller is responsible for passing the correct
    price series (typically close prices of the bar type used in training).

    Args:
        prices:      A pandas Series of bar close prices, chronologically
                     ordered, with a DatetimeIndex.
        span:        EWMA span in bars (default 20). Controls how quickly
                     the vol estimate reacts to new data. Higher span =
                     slower, smoother estimate.
        returns_col: Unused; reserved for future multi-column DataFrame
                     support. Raises ValueError if provided.

    Returns:
        A pandas Series of the same length as prices, containing the EWMA
        standard deviation of log returns. The first value is NaN because
        log returns require at least two price observations.

    Raises:
        ImportError:  If pandas or numpy are not installed (research layer
                      only — never installed in the live container).
        ValueError:   If prices is empty, span < 2, or returns_col is passed.
        TypeError:    If prices is not a pandas Series.
    """
    try:
        import numpy as np
        import pandas as pd_mod
    except ImportError as exc:
        raise ImportError(
            "ewma_vol requires pandas and numpy. These are research-layer "
            "dependencies and must not be imported in the live stack."
        ) from exc

    if returns_col is not None:
        raise ValueError(
            "returns_col is reserved for future use and must not be passed."
        )

    if not isinstance(prices, pd_mod.Series):
        raise TypeError(
            f"prices must be a pandas Series, got {type(prices).__name__!r}."
        )

    if span < 2:
        raise ValueError(f"span must be >= 2, got {span}.")

    if prices.empty:
        raise ValueError("prices Series is empty.")

    log_returns = np.log(prices / prices.shift(1))
    vol = log_returns.ewm(span=span, min_periods=span).std()

    return vol


def ewma_vol_latest(
    prices: "pd.Series",
    span: int = DEFAULT_EWMA_SPAN,
) -> Decimal:
    """
    Return the most recent EWMA volatility estimate as a Decimal.

    This is the value that crosses the seam via SignalPacket.metadata['ewma_vol'].
    Rounds to 8 decimal places to keep Decimal arithmetic clean.

    Args:
        prices: Bar close prices, chronologically ordered.
        span:   EWMA span in bars (default 20).

    Returns:
        Most recent non-NaN EWMA vol as Decimal, or Decimal('0') if the
        series contains insufficient data to compute the estimate.

    Example (live-stack side, after seam crossing):
        packet = coordinator.build_signal_packet(signal, prices)
        vol = packet.metadata.get('ewma_vol', Decimal('0'))
        # vol is now the same unit used during training
    """
    series = ewma_vol(prices, span=span)
    latest = series.dropna().iloc[-1] if not series.dropna().empty else None

    if latest is None:
        logger.warning(
            "ewma_vol_latest: insufficient data for span=%d — returning 0.", span
        )
        return Decimal("0")

    return Decimal(str(round(float(latest), 8)))


def build_signal_packet_metadata(
    prices: "pd.Series",
    span: int = DEFAULT_EWMA_SPAN,
) -> dict[str, Decimal]:
    """
    Build the metadata dict for a SignalPacket from a price series.

    Centralises the research → seam handoff so callers don't construct
    the metadata dict manually. All keys added here must be documented
    in research/seam.py.

    Args:
        prices: Bar close prices used for vol estimation.
        span:   EWMA span in bars.

    Returns:
        {'ewma_vol': Decimal(...)}  — ready to pass to SignalPacket(metadata=...).
    """
    return {
        "ewma_vol": ewma_vol_latest(prices, span=span),
    }
