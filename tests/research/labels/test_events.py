"""Acceptance tests for get_events (AFML Snippet 3.3, bars-forward vertical)."""
import numpy as np
import pandas as pd
import pytest

from research.labels.barriers import Barrier, get_events


def _close(values, start="2024-01-01", freq="15min"):
    idx = pd.date_range(start=start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series([float(v) for v in values], index=idx)


def _trgt(close, overrides=None, default=0.02):
    """Constant per-bar target, with optional {position: value} overrides."""
    s = pd.Series(default, index=close.index, dtype="float64")
    for pos, val in (overrides or {}).items():
        s.iloc[pos] = val
    return s


class TestGetEvents:
    def test_symmetric_event_resolves_upper(self):
        close = _close([100, 101, 103, 103, 103])
        t_events = pd.DatetimeIndex([close.index[0]])
        out = get_events(close, t_events, (1.0, 1.0), _trgt(close), max_hp=4)
        assert "side" not in out.columns
        assert out.iloc[0]["t1"] == close.index[2]
        assert out.iloc[0]["barrier"] == Barrier.UPPER
        assert out.iloc[0]["trgt"] == pytest.approx(0.02)

    def test_min_ret_filters_low_target_events(self):
        close = _close([100, 101, 103, 103, 103])
        t_events = pd.DatetimeIndex([close.index[0], close.index[2]])
        trgt = _trgt(close, overrides={2: 0.005})  # below min_ret
        out = get_events(close, t_events, (1.0, 1.0), trgt, max_hp=2, min_ret=0.01)
        assert list(out.index) == [close.index[0]]

    def test_nan_target_events_dropped(self):
        close = _close([100, 101, 103, 103, 103])
        t_events = pd.DatetimeIndex([close.index[0], close.index[2]])
        trgt = _trgt(close, overrides={2: np.nan})
        out = get_events(close, t_events, (1.0, 1.0), trgt, max_hp=2)
        assert list(out.index) == [close.index[0]]

    def test_meta_mode_keeps_side_column(self):
        close = _close([100, 101, 103, 103, 103])
        t_events = pd.DatetimeIndex([close.index[0]])
        side = pd.Series(1, index=t_events, dtype="int8")
        out = get_events(close, t_events, (1.0, 1.0), _trgt(close), max_hp=4, side=side)
        assert "side" in out.columns
        assert out.iloc[0]["side"] == 1

    def test_meta_mode_orients_via_side(self):
        # short side: a price drop is a profit-take (UPPER) in the bet frame.
        close = _close([100, 99, 97, 97, 97])
        t_events = pd.DatetimeIndex([close.index[0]])
        side = pd.Series(-1, index=t_events, dtype="int8")
        out = get_events(close, t_events, (1.0, 1.0), _trgt(close), max_hp=4, side=side)
        assert out.iloc[0]["barrier"] == Barrier.UPPER
        assert out.iloc[0]["side"] == -1

    def test_vertical_timeout_sets_vertical_barrier(self):
        close = _close([100, 100.5, 101, 100.5, 100])
        t_events = pd.DatetimeIndex([close.index[0]])
        out = get_events(close, t_events, (1.0, 1.0), _trgt(close), max_hp=4)
        assert out.iloc[0]["t1"] == close.index[4]
        assert out.iloc[0]["barrier"] == Barrier.VERTICAL

    def test_unresolved_tail_event_kept_with_na_barrier(self):
        close = _close([100, 100, 100, 100, 100])
        t_events = pd.DatetimeIndex([close.index[2]])
        out = get_events(close, t_events, (1.0, 1.0), _trgt(close), max_hp=5)
        assert len(out) == 1
        assert pd.isna(out.iloc[0]["t1"])
        assert pd.isna(out.iloc[0]["barrier"])

    def test_barrier_is_nullable_int8(self):
        close = _close([100, 101, 103, 103, 103])
        t_events = pd.DatetimeIndex([close.index[0]])
        out = get_events(close, t_events, (1.0, 1.0), _trgt(close), max_hp=4)
        assert out["barrier"].dtype == pd.Int8Dtype()
        assert isinstance(out["t1"].dtype, pd.DatetimeTZDtype)
        assert out["trgt"].dtype == np.dtype("float64")

    def test_empty_after_filter_returns_typed_frame(self):
        close = _close([100, 101, 103])
        t_events = pd.DatetimeIndex([close.index[0]])
        trgt = _trgt(close, overrides={0: 0.005})
        out = get_events(close, t_events, (1.0, 1.0), trgt, max_hp=2, min_ret=0.01)
        assert len(out) == 0
        assert list(out.columns) == ["t1", "trgt", "barrier"]
        assert out["barrier"].dtype == pd.Int8Dtype()
        assert isinstance(out["t1"].dtype, pd.DatetimeTZDtype)

    def test_output_index_is_filtered_event_starts(self):
        close = _close([100, 101, 103, 97, 100, 100, 100])
        t_events = pd.DatetimeIndex([close.index[0], close.index[1]])
        out = get_events(close, t_events, (1.0, 1.0), _trgt(close), max_hp=3)
        assert list(out.index) == [close.index[0], close.index[1]]
