"""Acceptance tests for the Barrier enum and triple-barrier helpers."""
import numpy as np
import pandas as pd
import pytest

from research.labels.barriers import Barrier, _vertical_barrier


def _utc_index(n, start="2024-01-01", freq="15min"):
    """Build a UTC DatetimeIndex of length ``n``."""
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC")


class TestBarrier:
    def test_barrier_values(self):
        assert Barrier.LOWER == -1
        assert Barrier.VERTICAL == 0
        assert Barrier.UPPER == 1

    def test_barrier_is_int8_compatible(self):
        assert np.int8(Barrier.UPPER) == 1
        assert np.int8(Barrier.LOWER) == -1


class TestVerticalBarrier:
    def test_interior_event_is_max_hp_bars_forward(self):
        index = _utc_index(10)
        t_events = pd.DatetimeIndex([index[2]])
        t1 = _vertical_barrier(index, t_events, max_hp=3)
        assert t1.iloc[0] == index[5]

    def test_multiple_events_each_offset_independently(self):
        index = _utc_index(10)
        t_events = pd.DatetimeIndex([index[0], index[2], index[4]])
        t1 = _vertical_barrier(index, t_events, max_hp=2)
        assert list(t1) == [index[2], index[4], index[6]]

    def test_last_resolvable_event_maps_to_final_bar(self):
        # len=10 (positions 0..9), max_hp=3 -> last resolvable position is 6,
        # mapping to the final bar index[9].
        index = _utc_index(10)
        t_events = pd.DatetimeIndex([index[6]])
        t1 = _vertical_barrier(index, t_events, max_hp=3)
        assert t1.iloc[0] == index[9]

    def test_events_at_or_past_tail_get_nat(self):
        # positions 7, 9 with max_hp=3 -> 10, 12 -> out of range -> NaT.
        index = _utc_index(10)
        t_events = pd.DatetimeIndex([index[7], index[9]])
        t1 = _vertical_barrier(index, t_events, max_hp=3)
        assert t1.isna().all()

    def test_mixed_interior_and_tail(self):
        index = _utc_index(10)
        t_events = pd.DatetimeIndex([index[5], index[8]])
        t1 = _vertical_barrier(index, t_events, max_hp=3)
        assert t1.iloc[0] == index[8]
        assert pd.isna(t1.iloc[1])

    def test_empty_events_returns_empty_tz_aware_series(self):
        index = _utc_index(10)
        t_events = pd.DatetimeIndex([], tz="UTC")
        t1 = _vertical_barrier(index, t_events, max_hp=3)
        assert len(t1) == 0
        assert isinstance(t1.dtype, pd.DatetimeTZDtype)
        assert str(t1.dtype.tz) == "UTC"

    def test_result_index_name_and_tz_preserved(self):
        index = _utc_index(10)
        t_events = pd.DatetimeIndex([index[1], index[2]])
        t1 = _vertical_barrier(index, t_events, max_hp=2)
        assert list(t1.index) == list(t_events)
        assert t1.name == "t1"
        assert isinstance(t1.dtype, pd.DatetimeTZDtype)
        assert str(t1.dtype.tz) == "UTC"

    def test_non_member_event_raises(self):
        index = _utc_index(10)
        off_grid = index[0] + pd.Timedelta(seconds=7)  # not a bar boundary
        t_events = pd.DatetimeIndex([off_grid])
        with pytest.raises(ValueError):
            _vertical_barrier(index, t_events, max_hp=3)

    def test_max_hp_must_be_positive(self):
        index = _utc_index(10)
        t_events = pd.DatetimeIndex([index[0]])
        with pytest.raises(ValueError):
            _vertical_barrier(index, t_events, max_hp=0)