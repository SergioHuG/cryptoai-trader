"""Acceptance tests for the symmetric CUSUM event filter (AFML Snippet 2.4)."""
import numpy as np
import pandas as pd
import pytest

from research.labels.filters import cusum_filter


def _close(values, start="2024-01-01", freq="15min"):
    """Build a UTC-indexed close Series from a list of prices."""
    idx = pd.date_range(start=start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series([float(v) for v in values], index=idx)


class TestCusumFilter:
    def test_constant_threshold_fires_on_cumulative_up_move(self):
        close = _close([100, 100, 103])  # diff2 = log(103/100) ~ 0.0296 > 0.02
        events = cusum_filter(close, threshold=0.02)
        assert len(events) == 1
        assert events[0] == close.index[2]

    def test_constant_threshold_fires_on_cumulative_down_move(self):
        close = _close([100, 100, 97])  # diff2 = log(97/100) ~ -0.0305 < -0.02
        events = cusum_filter(close, threshold=0.02)
        assert len(events) == 1
        assert events[0] == close.index[2]

    def test_arm_resets_after_fire(self):
        # alternating ~3% swings; each crossing fires AND resets that arm,
        # so we get one event per swing (3), not a single carried-over trigger.
        close = _close([100, 103, 100, 103])
        events = cusum_filter(close, threshold=0.02)
        assert len(events) == 3
        assert list(events) == [close.index[1], close.index[2], close.index[3]]

    def test_series_threshold_aligns_by_timestamp(self):
        # A ~3% up move at bar 2. With a high per-bar h there it is suppressed;
        # with a low h it fires. Proves the Series h is read per-timestamp.
        close = _close([100, 100, 103, 103])
        high = pd.Series(0.05, index=close.index)   # 0.0296 < 0.05 -> no fire
        low = pd.Series(0.01, index=close.index)     # 0.0296 > 0.01 -> fire
        assert len(cusum_filter(close, threshold=high)) == 0
        assert len(cusum_filter(close, threshold=low)) == 1

    def test_no_events_below_threshold_returns_empty_index(self):
        close = _close([100, 100.01, 100.02, 100.03])
        events = cusum_filter(close, threshold=0.02)
        assert isinstance(events, pd.DatetimeIndex)
        assert len(events) == 0

    def test_runs_on_log_returns_not_raw_diff(self):
        # Scale invariance: log returns are unchanged by a constant price factor,
        # so the event set must be identical for close and 1000*close.
        close = _close([100, 100, 103, 100, 103])
        scaled = close * 1000.0
        assert list(cusum_filter(close, 0.02)) == list(cusum_filter(scaled, 0.02))

    def test_returns_datetimeindex_preserving_utc(self):
        close = _close([100, 100, 103])
        events = cusum_filter(close, threshold=0.02)
        assert isinstance(events, pd.DatetimeIndex)
        assert str(events.tz) == "UTC"
