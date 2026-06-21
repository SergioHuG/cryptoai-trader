"""Acceptance tests for get_bins (AFML Snippet 3.5)."""
import numpy as np
import pandas as pd
import pytest

from research.labels.barriers import Barrier, get_bins


def _close(values, start="2024-01-01", freq="15min"):
    idx = pd.date_range(start=start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series([float(v) for v in values], index=idx)


def _events(close, rows, side=None):
    """rows: list of (start_pos, t1_pos_or_None, trgt, barrier_or_NA)."""
    starts = [close.index[r[0]] for r in rows]
    t1 = [close.index[r[1]] if r[1] is not None else pd.NaT for r in rows]
    trgt = [r[2] for r in rows]
    barrier = pd.array([r[3] for r in rows], dtype="Int8")
    df = pd.DataFrame(
        {
            "t1": pd.Series(pd.to_datetime(t1, utc=True), index=starts),
            "trgt": pd.Series(trgt, index=starts, dtype="float64"),
            "barrier": pd.Series(barrier, index=starts),
        },
        index=pd.DatetimeIndex(starts),
    )
    if side is not None:
        df["side"] = pd.Series(side, index=starts, dtype="int8")
    return df


class TestGetBins:
    def test_symmetric_positive_return(self):
        close = _close([100, 101, 103])
        ev = _events(close, [(0, 2, 0.02, Barrier.UPPER)])
        out = get_bins(ev, close)
        assert out.iloc[0]["ret"] == pytest.approx(0.03)
        assert out.iloc[0]["bin"] == 1

    def test_symmetric_negative_return(self):
        close = _close([100, 99, 97])
        ev = _events(close, [(0, 2, 0.02, Barrier.LOWER)])
        out = get_bins(ev, close)
        assert out.iloc[0]["ret"] == pytest.approx(-0.03)
        assert out.iloc[0]["bin"] == -1

    def test_symmetric_zero_return_is_zero_bin(self):
        close = _close([100, 100, 100])
        ev = _events(close, [(0, 2, 0.02, Barrier.VERTICAL)])
        out = get_bins(ev, close)
        assert out.iloc[0]["ret"] == 0.0
        assert out.iloc[0]["bin"] == 0

    def test_meta_profitable_bet_is_one(self):
        # short side; price drop -> ret*side > 0 -> profitable -> bin=1
        close = _close([100, 99, 97])
        ev = _events(close, [(0, 2, 0.02, Barrier.UPPER)], side=[-1])
        out = get_bins(ev, close)
        assert out.iloc[0]["ret"] == pytest.approx(0.03)  # oriented
        assert out.iloc[0]["bin"] == 1
        assert out.iloc[0]["side"] == -1

    def test_meta_unprofitable_bet_is_zero(self):
        # long side; price drop -> ret*side < 0 -> unprofitable -> bin=0
        close = _close([100, 99, 97])
        ev = _events(close, [(0, 2, 0.02, Barrier.LOWER)], side=[1])
        out = get_bins(ev, close)
        assert out.iloc[0]["ret"] == pytest.approx(-0.03)
        assert out.iloc[0]["bin"] == 0

    def test_meta_zero_return_is_zero_bin(self):
        close = _close([100, 100, 100])
        ev = _events(close, [(0, 2, 0.02, Barrier.VERTICAL)], side=[1])
        out = get_bins(ev, close)
        assert out.iloc[0]["bin"] == 0

    def test_unresolved_events_dropped(self):
        close = _close([100, 101, 103])
        ev = _events(
            close,
            [(0, 2, 0.02, Barrier.UPPER), (1, None, 0.02, pd.NA)],
        )
        out = get_bins(ev, close)
        assert len(out) == 1
        assert list(out.index) == [close.index[0]]

    def test_barrier_cast_to_plain_int8(self):
        close = _close([100, 101, 103])
        ev = _events(close, [(0, 2, 0.02, Barrier.UPPER)])
        out = get_bins(ev, close)
        assert out["barrier"].dtype == np.dtype("int8")
        assert out.iloc[0]["barrier"] == Barrier.UPPER

    def test_bin_dtype_is_int8(self):
        close = _close([100, 101, 103])
        ev = _events(close, [(0, 2, 0.02, Barrier.UPPER)])
        out = get_bins(ev, close)
        assert out["bin"].dtype == np.dtype("int8")

    def test_symmetric_output_columns_no_side(self):
        close = _close([100, 101, 103])
        ev = _events(close, [(0, 2, 0.02, Barrier.UPPER)])
        out = get_bins(ev, close)
        assert list(out.columns) == ["ret", "bin", "t1", "barrier"]

    def test_meta_output_columns_include_side(self):
        close = _close([100, 101, 103])
        ev = _events(close, [(0, 2, 0.02, Barrier.UPPER)], side=[1])
        out = get_bins(ev, close)
        assert list(out.columns) == ["ret", "bin", "t1", "barrier", "side"]

    def test_t1_dtype_preserved_tz_aware(self):
        close = _close([100, 101, 103])
        ev = _events(close, [(0, 2, 0.02, Barrier.UPPER)])
        out = get_bins(ev, close)
        assert isinstance(out["t1"].dtype, pd.DatetimeTZDtype)
        assert str(out["t1"].dtype.tz) == "UTC"
        assert out.iloc[0]["t1"] == close.index[2]

    def test_empty_input_returns_empty_typed_frame(self):
        close = _close([100, 101, 103])
        ev = _events(close, [(0, None, 0.02, pd.NA)])  # all unresolved
        out = get_bins(ev, close)
        assert len(out) == 0
        assert list(out.columns) == ["ret", "bin", "t1", "barrier"]
        assert out["bin"].dtype == np.dtype("int8")
        assert out["barrier"].dtype == np.dtype("int8")
