"""Acceptance tests for research.weights.concurrency (Step 2).

num_co_events (AFML 4.1) is implemented as a difference-array sweep --
O(events + bars), independent of span width -- rather than AFML's naive
per-bar accumulation loop. The naive loop is retained here as a *test
oracle* only (TestNumCoEventsSweepMatchesNaiveOracle), never as production
code (Q6).

avg_uniqueness (AFML 4.2) takes co_events as an explicit argument and never
recomputes it -- this is the seam that lets the orchestrator (Step 6)
guarantee a single shared num_co_events() call feeds every consumer (Q3).
"""
import numpy as np
import pandas as pd
import pytest

from research.weights.concurrency import avg_uniqueness, num_co_events


def _utc_index(n, start="2024-01-01", freq="15min"):
    """Build a UTC DatetimeIndex of length ``n``."""
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC")


def _t1(idx, pairs, name="t1"):
    """Build a t1 Series from [(start_pos, end_pos), ...] over ``idx``."""
    starts = [idx[s] for s, _ in pairs]
    ends = [idx[e] for _, e in pairs]
    return pd.Series(ends, index=pd.DatetimeIndex(starts, name=None), name=name)


def _naive_num_co_events(close_index, t1):
    """Independent oracle: AFML's per-event .loc-slice accumulation (4.1)."""
    counts = pd.Series(0, index=close_index, dtype="int64")
    for t0, t1v in t1.items():
        counts.loc[t0:t1v] += 1
    counts.name = "num_co_events"
    return counts


class TestNumCoEventsBasicCounting:
    def test_single_event_counts_one_across_span(self):
        idx = _utc_index(5)
        t1 = _t1(idx, [(1, 3)])

        result = num_co_events(idx, t1)

        assert list(result) == [0, 1, 1, 1, 0]

    def test_two_overlapping_events_sum_correctly(self):
        idx = _utc_index(5)
        t1 = _t1(idx, [(1, 3), (2, 4)])

        result = num_co_events(idx, t1)

        assert list(result) == [0, 1, 2, 2, 1]

    def test_two_disjoint_events_never_exceed_one(self):
        idx = _utc_index(5)
        t1 = _t1(idx, [(0, 1), (3, 4)])

        result = num_co_events(idx, t1)

        assert list(result) == [1, 1, 0, 1, 1]
        assert result.max() == 1

    def test_inclusive_of_both_endpoints(self):
        idx = _utc_index(5)
        t1 = _t1(idx, [(0, 2)])

        result = num_co_events(idx, t1)

        assert result.iloc[0] == 1  # t0 endpoint
        assert result.iloc[2] == 1  # t1 endpoint
        assert result.iloc[1] == 1

    def test_single_bar_event_counts_one_at_that_bar_only(self):
        idx = _utc_index(5)
        t1 = _t1(idx, [(2, 2)])

        result = num_co_events(idx, t1)

        assert list(result) == [0, 0, 1, 0, 0]


class TestNumCoEventsGridShape:
    def test_result_index_equals_full_close_index(self):
        idx = _utc_index(6)
        t1 = _t1(idx, [(1, 3)])

        result = num_co_events(idx, t1)

        pd.testing.assert_index_equal(result.index, idx)

    def test_result_is_zero_filled_outside_event_range(self):
        idx = _utc_index(6)
        t1 = _t1(idx, [(2, 3)])

        result = num_co_events(idx, t1)

        assert result.iloc[0] == 0
        assert result.iloc[1] == 0
        assert result.iloc[4] == 0
        assert result.iloc[5] == 0

    def test_empty_events_returns_all_zero_series(self):
        idx = _utc_index(5)
        t1 = pd.Series([], dtype="datetime64[ns, UTC]", name="t1")

        result = num_co_events(idx, t1)

        pd.testing.assert_index_equal(result.index, idx)
        assert (result == 0).all()


class TestNumCoEventsValidation:
    def test_null_t1_raises(self):
        idx = _utc_index(5)
        t1 = pd.Series([idx[2], pd.NaT], index=[idx[0], idx[1]], name="t1")

        with pytest.raises(ValueError):
            num_co_events(idx, t1)

    def test_t0_not_member_of_close_index_raises(self):
        idx = _utc_index(5)
        foreign_t0 = pd.Timestamp("1999-01-01", tz="UTC")
        t1 = pd.Series([idx[2]], index=[foreign_t0], name="t1")

        with pytest.raises(ValueError):
            num_co_events(idx, t1)

    def test_t1_value_not_member_of_close_index_raises(self):
        idx = _utc_index(5)
        foreign_t1 = pd.Timestamp("1999-01-01", tz="UTC")
        t1 = pd.Series([foreign_t1], index=[idx[0]], name="t1")

        with pytest.raises(ValueError):
            num_co_events(idx, t1)


class TestNumCoEventsSweepMatchesNaiveOracle:
    def test_sweep_equals_naive_oracle_on_randomized_overlaps(self):
        """The difference-array sweep must be exactly equal -- not an
        approximation -- to AFML's naive per-event .loc-slice accumulation,
        across many randomized overlapping spans (Q6 equivalence pin)."""
        rng = np.random.default_rng(42)
        idx = _utc_index(60)
        n_events = 25
        starts = rng.integers(0, 55, size=n_events)
        spans = rng.integers(1, 8, size=n_events)
        ends = np.minimum(starts + spans, 59)

        t1 = pd.Series(
            [idx[e] for e in ends],
            index=pd.DatetimeIndex([idx[s] for s in starts]),
            name="t1",
        )

        actual = num_co_events(idx, t1)
        expected = _naive_num_co_events(idx, t1)

        pd.testing.assert_series_equal(actual, expected)

    def test_sweep_equals_naive_oracle_with_duplicate_start_bars(self):
        """Multiple events sharing the same t0 bar (a realistic CUSUM-filter
        edge case at high sampling density) must still sum correctly."""
        idx = _utc_index(10)
        t1 = _t1(idx, [(1, 4), (1, 6), (3, 8)])

        actual = num_co_events(idx, t1)
        expected = _naive_num_co_events(idx, t1)

        pd.testing.assert_series_equal(actual, expected)


class TestAvgUniqueness:
    def test_single_event_no_overlap_uniqueness_one(self):
        idx = _utc_index(5)
        t1 = _t1(idx, [(1, 3)])
        co_events = pd.Series(1, index=idx, dtype="int64")

        result = avg_uniqueness(t1, co_events)

        assert result.iloc[0] == pytest.approx(1.0)

    def test_partial_overlap_averages_reciprocals(self):
        """Two events overlapping over 2 of their 3 bars each -- hand
        computed against AFML's mean(1/c_t) over each event's own span."""
        idx = _utc_index(5)
        t1 = _t1(idx, [(0, 2), (1, 3)])
        co_events = num_co_events(idx, t1)  # [1, 2, 2, 1, 0]

        result = avg_uniqueness(t1, co_events)

        # event A spans bars 0,1,2 -> c_t = [1,2,2] -> mean(1,0.5,0.5) = 2/3
        assert result.iloc[0] == pytest.approx(2 / 3)
        # event B spans bars 1,2,3 -> c_t = [2,2,1] -> mean(0.5,0.5,1) = 2/3
        assert result.iloc[1] == pytest.approx(2 / 3)

    def test_takes_co_events_as_explicit_arg_no_internal_recompute(self):
        """Passing a co_events series that disagrees with the 'true' one
        must change the output -- proves there is no internal recompute."""
        idx = _utc_index(5)
        t1 = _t1(idx, [(0, 2)])  # true co_events here would be all 1s
        fake_co_events = pd.Series(5, index=idx, dtype="int64")

        result = avg_uniqueness(t1, fake_co_events)

        assert result.iloc[0] == pytest.approx(1.0 / 5.0)

    def test_indexed_by_t0(self):
        idx = _utc_index(5)
        t1 = _t1(idx, [(0, 2), (1, 3)])
        co_events = pd.Series(1, index=idx, dtype="int64")

        result = avg_uniqueness(t1, co_events)

        pd.testing.assert_index_equal(result.index, t1.index)

    def test_empty_input_returns_empty(self):
        t1 = pd.Series([], dtype="datetime64[ns, UTC]", name="t1")
        co_events = pd.Series([], dtype="int64")

        result = avg_uniqueness(t1, co_events)

        assert len(result) == 0
