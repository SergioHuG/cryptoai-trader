"""Acceptance tests for the Barrier enum and triple-barrier helpers."""
import numpy as np
import pandas as pd
import pytest

from research.labels.barriers import (
    Barrier,
    _apply_pt_sl_on_t1,
    _vertical_barrier,
)


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


def _close(values, start="2024-01-01", freq="15min"):
    """Build a UTC-indexed float close Series from a list of prices."""
    idx = pd.date_range(start=start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series([float(v) for v in values], index=idx)


def _events(close, starts, max_hp, trgt, side=1):
    """Assemble an events frame (t1 vertical, trgt, side) for path-walk tests."""
    t_events = pd.DatetimeIndex([close.index[s] for s in starts])
    t1 = _vertical_barrier(close.index, t_events, max_hp)
    return pd.DataFrame(
        {"t1": t1, "trgt": float(trgt), "side": np.int8(side)},
        index=t_events,
    )


class TestApplyPtSlOnT1:
    def test_upper_barrier_touched_first(self):
        close = _close([100, 101, 103, 103, 103])  # bar2 ret +3% > 2%
        ev = _events(close, starts=[0], max_hp=4, trgt=0.02)
        out = _apply_pt_sl_on_t1(close, ev, pt_sl=(1.0, 1.0))
        assert out.iloc[0]["t1"] == close.index[2]
        assert out.iloc[0]["barrier"] == Barrier.UPPER

    def test_lower_barrier_touched_first(self):
        close = _close([100, 99, 97, 97, 97])  # bar2 ret -3% < -2%
        ev = _events(close, starts=[0], max_hp=4, trgt=0.02)
        out = _apply_pt_sl_on_t1(close, ev, pt_sl=(1.0, 1.0))
        assert out.iloc[0]["t1"] == close.index[2]
        assert out.iloc[0]["barrier"] == Barrier.LOWER

    def test_vertical_when_no_horizontal_touch(self):
        close = _close([100, 100.5, 101, 100.5, 100])  # never beyond +/-2%
        ev = _events(close, starts=[0], max_hp=4, trgt=0.02)
        out = _apply_pt_sl_on_t1(close, ev, pt_sl=(1.0, 1.0))
        assert out.iloc[0]["t1"] == close.index[4]
        assert out.iloc[0]["barrier"] == Barrier.VERTICAL

    def test_first_touch_is_earliest(self):
        # bar1 already +3% (PT) before bar2 would be -3% (SL): UPPER wins.
        close = _close([100, 103, 97, 97, 97])
        ev = _events(close, starts=[0], max_hp=4, trgt=0.02)
        out = _apply_pt_sl_on_t1(close, ev, pt_sl=(1.0, 1.0))
        assert out.iloc[0]["t1"] == close.index[1]
        assert out.iloc[0]["barrier"] == Barrier.UPPER

    def test_horizontal_wins_on_vertical_bar(self):
        # vertical falls on bar2; PT also first breaches exactly at bar2.
        close = _close([100, 100.5, 103])
        ev = _events(close, starts=[0], max_hp=2, trgt=0.02)
        assert pd.notna(ev.iloc[0]["t1"])  # vertical is index[2]
        out = _apply_pt_sl_on_t1(close, ev, pt_sl=(1.0, 1.0))
        assert out.iloc[0]["t1"] == close.index[2]
        assert out.iloc[0]["barrier"] == Barrier.UPPER  # not VERTICAL

    def test_side_short_orients_returns(self):
        # side=-1: a price drop is a *profit* (UPPER) on a short.
        close = _close([100, 99, 97, 97, 97])
        ev = _events(close, starts=[0], max_hp=4, trgt=0.02, side=-1)
        out = _apply_pt_sl_on_t1(close, ev, pt_sl=(1.0, 1.0))
        assert out.iloc[0]["t1"] == close.index[2]
        assert out.iloc[0]["barrier"] == Barrier.UPPER

    def test_nat_vertical_walks_to_end_and_resolves(self):
        # start=2, max_hp=5 on a len-5 series -> vertical NaT; PT touches at bar3.
        close = _close([100, 100, 100, 103, 103])
        ev = _events(close, starts=[2], max_hp=5, trgt=0.02)
        assert pd.isna(ev.iloc[0]["t1"])  # confirm NaT vertical
        out = _apply_pt_sl_on_t1(close, ev, pt_sl=(1.0, 1.0))
        assert out.iloc[0]["t1"] == close.index[3]
        assert out.iloc[0]["barrier"] == Barrier.UPPER

    def test_nat_vertical_no_touch_is_unresolved(self):
        close = _close([100, 100, 100, 100, 100])
        ev = _events(close, starts=[2], max_hp=5, trgt=0.02)
        assert pd.isna(ev.iloc[0]["t1"])
        out = _apply_pt_sl_on_t1(close, ev, pt_sl=(1.0, 1.0))
        assert pd.isna(out.iloc[0]["t1"])
        assert pd.isna(out.iloc[0]["barrier"])

    def test_strict_inequality_at_exact_threshold(self):
        # Make trgt bit-exactly equal to bar2's realized return, so r == pt.
        # AFML strict '>' means an exact landing is NOT a touch -> VERTICAL.
        close = _close([100, 100, 102, 100])
        exact = close.iloc[2] / close.iloc[0] - 1.0
        ev = _events(close, starts=[0], max_hp=3, trgt=exact)
        out = _apply_pt_sl_on_t1(close, ev, pt_sl=(1.0, 1.0))
        assert out.iloc[0]["t1"] == close.index[3]
        assert out.iloc[0]["barrier"] == Barrier.VERTICAL

    def test_zero_pt_mult_disables_profit_take(self):
        close = _close([100, 103, 103, 103])  # would be UPPER if PT enabled
        ev = _events(close, starts=[0], max_hp=3, trgt=0.02)
        out = _apply_pt_sl_on_t1(close, ev, pt_sl=(0.0, 1.0))
        assert out.iloc[0]["t1"] == close.index[3]
        assert out.iloc[0]["barrier"] == Barrier.VERTICAL

    def test_empty_events_returns_empty_typed_frame(self):
        close = _close([100, 101, 102])
        ev = _events(close, starts=[], max_hp=2, trgt=0.02)
        out = _apply_pt_sl_on_t1(close, ev, pt_sl=(1.0, 1.0))
        assert len(out) == 0
        assert list(out.columns) == ["t1", "barrier"]
        assert isinstance(out["t1"].dtype, pd.DatetimeTZDtype)
        assert out["barrier"].dtype == pd.Int8Dtype()

    def test_output_index_dtypes_and_alignment(self):
        close = _close([100, 101, 103, 97, 100, 100])
        ev = _events(close, starts=[0, 1], max_hp=3, trgt=0.02)
        out = _apply_pt_sl_on_t1(close, ev, pt_sl=(1.0, 1.0))
        assert list(out.index) == list(ev.index)
        assert isinstance(out["t1"].dtype, pd.DatetimeTZDtype)
        assert str(out["t1"].dtype.tz) == "UTC"
        assert out["barrier"].dtype == pd.Int8Dtype()