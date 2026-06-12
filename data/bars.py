"""Dollar bar constructor for the CryptoAI Trader research layer.

Streams Kraken trade history via an injected CCXT exchange, accumulates
trades until a dollar-volume threshold is crossed, and persists bars to
a resumable pd.HDFStore checkpoint file.

HDF5 group hierarchy
--------------------
    store_path.h5
    └── /{SYM_KEY}/          e.g. /BTC_USD/
        └── /thr_{N}/        e.g. /thr_1000000/
            └── bars         pd.HDFStore table format, appendable
                  attrs:
                    last_trade_ts: int  ← epoch ms, resume checkpoint

Public API
----------
Bar                  -- frozen dataclass; the bar boundary DTO
DollarBarAccumulator -- stateful accumulator (also exported for research)
build_dollar_bars()  -- bootstrap / resume historical dollar bars
load_dollar_bars()   -- read persisted bars from HDF5
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import pandas as pd

__all__ = [
    "Bar",
    "DollarBarAccumulator",
    "build_dollar_bars",
    "load_dollar_bars",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RATE_LIMIT_DELAY: float = 1.0   # seconds between API pages (Kraken limit)
DEFAULT_PAGE_SIZE: int = 1000           # trades per fetch_trades() call
DEFAULT_CHECKPOINT_INTERVAL: int = 50  # bars between HDF5 flush+checkpoint

_HDF5_COMPLEVEL: int = 5
_HDF5_COMPLIB: str = "blosc"

# Columns present in every bars DataFrame (excluding the bar_end_ts index).
BAR_COLUMNS: list[str] = [
    "bar_start_ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dollar_volume",
    "num_ticks",
]


# ---------------------------------------------------------------------------
# Bar dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Bar:
    """Immutable dollar bar boundary DTO.

    All price/volume fields are Decimal to preserve arithmetic precision
    during accumulation. Converted to float64 when written to HDF5 via
    :func:`_bars_to_df`.
    """

    bar_start_ts: int       # epoch ms — timestamp of first trade in bar
    bar_end_ts: int         # epoch ms — timestamp of last trade in bar
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal         # base currency (e.g. BTC)
    dollar_volume: Decimal  # quote currency (e.g. USD); the value that crosses threshold
    num_ticks: int          # number of raw trades sampled into this bar


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _symbol_to_key(symbol: str) -> str:
    """Normalise a CCXT symbol string to an HDF5-safe group name.

    Examples
    --------
    'BTC/USD' → 'BTC_USD'
    'btc/usd' → 'BTC_USD'
    """
    return symbol.replace("/", "_").upper()


def _threshold_to_key(threshold: Decimal) -> str:
    """Normalise a Decimal threshold to an HDF5-safe group name.

    HDF5 group names cannot start with a digit, so we prefix 'thr_'.

    Examples
    --------
    Decimal('1000000')    → 'thr_1000000'
    Decimal('500000.99')  → 'thr_500000'  (int-truncated)
    """
    return f"thr_{int(threshold)}"


def _hdf5_key(symbol: str, threshold: Decimal) -> str:
    """Return the full HDFStore key for (symbol, threshold).

    Example: '/BTC_USD/thr_1000000/bars'
    """
    return f"/{_symbol_to_key(symbol)}/{_threshold_to_key(threshold)}/bars"


# ---------------------------------------------------------------------------
# DollarBarAccumulator
# ---------------------------------------------------------------------------

class DollarBarAccumulator:
    """Stateful dollar bar accumulator.

    Feed raw trades one at a time via :meth:`ingest`. A :class:`Bar` is
    returned (and state reset) when accumulated dollar volume first reaches
    or exceeds ``threshold``.  Call :meth:`flush` at end-of-data to retrieve
    the trailing partial bar (if any).

    Not thread-safe — one instance per bootstrap run.
    """

    def __init__(self, threshold: Decimal) -> None:
        if threshold <= Decimal("0"):
            raise ValueError(f"threshold must be positive, got {threshold!r}")
        self._threshold = threshold
        self._reset()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        """True when no trades have been ingested since the last reset or flush."""
        return self._num_ticks == 0

    def ingest(self, ts_ms: int, price: Decimal, amount: Decimal) -> Bar | None:
        """Process one trade.

        Parameters
        ----------
        ts_ms:   Trade timestamp in epoch milliseconds.
        price:   Trade execution price in quote currency.
        amount:  Trade size in base currency.

        Returns
        -------
        Completed :class:`Bar` when dollar-volume threshold is crossed,
        ``None`` otherwise.
        """
        # Record open price on first tick of a new bar
        if self._start_ts is None:
            self._start_ts = ts_ms
            self._open = price

        self._last_ts = ts_ms
        self._high = max(self._high, price)
        self._low = min(self._low, price)
        self._close = price
        self._volume += amount
        self._dollar_volume += price * amount
        self._num_ticks += 1

        if self._dollar_volume >= self._threshold:
            return self._emit(ts_ms)

        return None

    def flush(self) -> Bar | None:
        """Force-complete the partial bar accumulated so far.

        Called at end of historical data to avoid discarding the trailing
        partial bar. Returns ``None`` if no trades have been ingested since
        the last emission or flush.
        """
        if self.is_empty:
            return None
        return self._emit(self._last_ts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emit(self, end_ts: int) -> Bar:
        """Package current state into a Bar, reset, and return."""
        bar = Bar(
            bar_start_ts=self._start_ts,  # type: ignore[arg-type]
            bar_end_ts=end_ts,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            dollar_volume=self._dollar_volume,
            num_ticks=self._num_ticks,
        )
        self._reset()
        return bar

    def _reset(self) -> None:
        self._start_ts: int | None = None
        self._last_ts: int = 0
        self._open: Decimal = Decimal("0")
        self._high: Decimal = Decimal("0")
        self._low: Decimal = Decimal("Inf")
        self._close: Decimal = Decimal("0")
        self._volume: Decimal = Decimal("0")
        self._dollar_volume: Decimal = Decimal("0")
        self._num_ticks: int = 0


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def _bars_to_df(bars: list[Bar]) -> pd.DataFrame:
    """Convert a list of :class:`Bar` to a DataFrame with DatetimeIndex.

    Index
    -----
    ``bar_end_ts`` — UTC DatetimeIndex built from each bar's ``bar_end_ts``
    (epoch milliseconds).

    Columns
    -------
    bar_start_ts : int64
    open, high, low, close, volume, dollar_volume : float64
    num_ticks : int64
    """
    if not bars:
        empty = pd.DataFrame(columns=BAR_COLUMNS)
        # Enforce dtypes on the empty frame so callers can rely on them
        for col in ["open", "high", "low", "close", "volume", "dollar_volume"]:
            empty[col] = empty[col].astype("float64")
        empty["num_ticks"] = empty["num_ticks"].astype("int64")
        empty["bar_start_ts"] = empty["bar_start_ts"].astype("int64")
        return empty

    records = [
        {
            "bar_start_ts": b.bar_start_ts,
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": float(b.volume),
            "dollar_volume": float(b.dollar_volume),
            "num_ticks": int(b.num_ticks),
        }
        for b in bars
    ]
    index = pd.to_datetime(
        [b.bar_end_ts for b in bars], unit="ms", utc=True
    )
    df = pd.DataFrame(records, index=index)
    df.index.name = "bar_end_ts"
    df["num_ticks"] = df["num_ticks"].astype("int64")
    return df


# ---------------------------------------------------------------------------
# HDF5 checkpoint helpers
# ---------------------------------------------------------------------------

def _read_checkpoint(store_path: Path, hdf_key: str) -> int | None:
    """Read the ``last_trade_ts`` checkpoint attribute from HDF5.

    Returns ``None`` if the file or key does not exist (fresh / first run).
    """
    if not store_path.exists():
        return None
    try:
        with pd.HDFStore(store_path, mode="r") as store:
            if hdf_key not in store:
                return None
            return store.get_storer(hdf_key).attrs["last_trade_ts"]
    except KeyError:
        return None


def _write_checkpoint(store_path: Path, hdf_key: str, last_trade_ts: int) -> None:
    """Persist the ``last_trade_ts`` checkpoint attribute to HDF5."""
    with pd.HDFStore(store_path, mode="a") as store:
        store.get_storer(hdf_key).attrs["last_trade_ts"] = last_trade_ts


def _flush_bars(
    store_path: Path, hdf_key: str, bars: list[Bar], last_trade_ts: int
) -> None:
    """Append a batch of :class:`Bar` objects to HDF5 and update checkpoint.

    Opens the store in append mode (creates the file if absent).  The
    checkpoint attribute is updated in the same open/close cycle to minimise
    data loss on crash.
    """
    df = _bars_to_df(bars)
    mode = "a" if store_path.exists() else "w"
    with pd.HDFStore(
        store_path, mode=mode, complevel=_HDF5_COMPLEVEL, complib=_HDF5_COMPLIB
    ) as store:
        store.append(hdf_key, df, format="table", data_columns=True)
        store.get_storer(hdf_key).attrs["last_trade_ts"] = last_trade_ts
    logger.debug(
        "Flushed %d bars → %s  (last_trade_ts=%d)",
        len(bars), hdf_key, last_trade_ts,
    )


# ---------------------------------------------------------------------------
# Public orchestration functions
# ---------------------------------------------------------------------------

def build_dollar_bars(
    symbol: str,
    threshold: Decimal,
    store_path: Path,
    *,
    exchange: Any,
    since_ms: int | None = None,
    rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
    page_size: int = DEFAULT_PAGE_SIZE,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    _sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Bootstrap (or resume) historical dollar bars from Kraken trade history.

    Fetches paginated trades from ``exchange.fetch_trades(symbol, since=...,
    limit=page_size)``, feeds them through :class:`DollarBarAccumulator`, and
    flushes completed bars + checkpoint to ``store_path`` (HDF5) at
    ``checkpoint_interval``-bar intervals.

    Termination conditions
    ----------------------
    - ``fetch_trades`` returns an empty list.
    - ``fetch_trades`` returns fewer trades than ``page_size`` (end of history).

    Resume behaviour
    ----------------
    On resumption the cursor starts at ``checkpoint + 1`` (ms).  Pass an
    explicit ``since_ms`` to override the checkpoint (e.g. for backfill).

    Parameters
    ----------
    symbol:               CCXT market symbol, e.g. ``'BTC/USD'``.
    threshold:            Dollar volume per bar (Decimal).
    store_path:           Path to the HDF5 output file.
    exchange:             Injected CCXT exchange instance (Kraken).
    since_ms:             Override start timestamp (epoch ms).
    rate_limit_delay:     Sleep between pages (default 1.0 s — Kraken limit).
    page_size:            Trades per API call (default 1000).
    checkpoint_interval:  Bars between flush+checkpoint cycles (default 50).
    _sleep:               Injectable sleep callable (substitute in tests).

    Returns
    -------
    Complete ``pd.DataFrame`` of all persisted bars for (symbol, threshold)
    after this run.
    """
    hdf_key = _hdf5_key(symbol, threshold)
    accumulator = DollarBarAccumulator(threshold)

    # Determine starting cursor
    if since_ms is not None:
        cursor: int | None = since_ms
    else:
        checkpoint = _read_checkpoint(store_path, hdf_key)
        cursor = (checkpoint + 1) if checkpoint is not None else None

    logger.info(
        "build_dollar_bars start: symbol=%s threshold=%s cursor=%s",
        symbol, threshold, cursor,
    )

    pending_bars: list[Bar] = []
    last_trade_ts: int = (cursor - 1) if cursor is not None else 0

    while True:
        trades: list[dict] = exchange.fetch_trades(
            symbol, since=cursor, limit=page_size
        )

        if not trades:
            break

        for trade in trades:
            ts_ms: int = trade["timestamp"]
            price = Decimal(str(trade["price"]))
            amount = Decimal(str(trade["amount"]))

            bar = accumulator.ingest(ts_ms, price, amount)
            if bar is not None:
                pending_bars.append(bar)

            last_trade_ts = ts_ms

        # Flush completed bars at checkpoint_interval
        if len(pending_bars) >= checkpoint_interval:
            _flush_bars(store_path, hdf_key, pending_bars, last_trade_ts)
            pending_bars = []

        # Partial page → end of available history; stop before sleeping
        if len(trades) < page_size:
            break

        cursor = last_trade_ts + 1
        _sleep(rate_limit_delay)

    # Final flush: remaining completed bars + trailing partial bar
    final_bar = accumulator.flush()
    if final_bar is not None:
        pending_bars.append(final_bar)

    if pending_bars:
        _flush_bars(store_path, hdf_key, pending_bars, last_trade_ts)

    return load_dollar_bars(symbol, threshold, store_path)


def load_dollar_bars(
    symbol: str,
    threshold: Decimal,
    store_path: Path,
) -> pd.DataFrame:
    """Load all persisted dollar bars for (symbol, threshold) from HDF5.

    Returns an empty :class:`pd.DataFrame` (with correct columns) when the
    file does not exist or the key has not been written yet.
    """
    hdf_key = _hdf5_key(symbol, threshold)
    if not store_path.exists():
        return _bars_to_df([])
    try:
        with pd.HDFStore(store_path, mode="r") as store:
            if hdf_key not in store:
                return _bars_to_df([])
            return store[hdf_key]
    except Exception as exc:  # noqa: BLE001
        logger.warning("load_dollar_bars failed for %s: %s", hdf_key, exc)
        return _bars_to_df([])
