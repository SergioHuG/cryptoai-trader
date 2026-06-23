"""Acceptance tests for research.weights.attribution (Step 3, AFML 4.10).

return_attribution uses *log* returns -- a deliberate, documented split
from research.labels.barriers, which uses simple returns (AFML 3.2/3.5).
Both are AFML-faithful; this is a different snippet (4.10) with different
intent, not an inconsistency to "fix" (Q4b).

Indexing follows AFML's mpSampleW exactly: ret.loc[t0:t1] is INCLUSIVE of
t0 (the same convention research.weights.concurrency.avg_uniqueness uses
for co_events.loc[t0:t1]), summed (not averaged) and divided by
co_events, then absolute-valued (Q4a, Q4c).

return_attribution returns the raw absolute attribution -- normalize_weights
is a separate, explicitly-called step (Q4c), so the decay-before-renormalize
seam (Q5) stays visible at the orchestrator level.
"""
import numpy as np
import pandas as pd
import pytest

from research.weights.attribution import normalize_weights, return_attribution


def _close(values, start="2024-01-01", freq="15min"):
    idx = pd.date_range(start=start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series([float(v) for v in values], index=idx)


def _t1(idx, pairs, name="t1"):
    """Build a t1 Series from [(start_pos, end_pos), ...] over ``idx``."""
    starts = [idx[s] for s, _ in pairs]
    ends = [idx[e] for _, e in pairs]
    return pd.Series(ends, index=pd.DatetimeIndex(starts), name=name)


_LN_1_1 = float(np.log(1.1))  # 0.0953101798...


class TestReturnAttributionUsesLogReturns:
    def test_uses_log_returns_not_simple_returns(self):
        """Constant 10% per-bar growth: simple returns sum to exactly 0.30
        over 3 steps, but log returns sum to 3*ln(1.1) ~= 0.28593 -- a
        detectable, hand-computable difference that pins log-return usage."""
        close = _close([100.0, 110.0, 121.0, 133.1])
        t1 = _t1(close.index, [(1, 3)])
        co_events = pd.Series(1, index=close.index, dtype="int64")

        result = return_attribution(close, t1, co_events)

        assert result.iloc[0] == pytest.approx(3 * _LN_1_1, rel=1e-9)
        assert result.iloc[0] != pytest.approx(0.30, rel=1e-3)


class TestReturnAttributionDividesByConcurrency:
    def test_doubling_co_events_halves_attribution(self):
        close = _close([100.0, 200.0, 400.0])
        t1 = _t1(close.index, [(1, 2)])

        co_events_one = pd.Series(1, index=close.index, dtype="int64")
        co_events_two = pd.Series(2, index=close.index, dtype="int64")

        result_one = return_attribution(close, t1, co_events_one)
        result_two = return_attribution(close, t1, co_events_two)

        assert result_two.iloc[0] == pytest.approx(result_one.iloc[0] / 2.0)


class TestReturnAttributionAbsoluteValue:
    def test_negative_log_return_yields_positive_weight(self):
        close = _close([100.0, 50.0])
        t1 = _t1(close.index, [(1, 1)])
        co_events = pd.Series(1, index=close.index, dtype="int64")

        result = return_attribution(close, t1, co_events)

        assert result.iloc[0] > 0
        assert result.iloc[0] == pytest.approx(float(np.log(2.0)))


class TestReturnAttributionExplicitCoEventsArg:
    def test_takes_co_events_as_explicit_arg_no_internal_recompute(self):
        close = _close([100.0, 110.0])
        t1 = _t1(close.index, [(1, 1)])
        fake_co_events = pd.Series(5, index=close.index, dtype="int64")

        result = return_attribution(close, t1, fake_co_events)

        assert result.iloc[0] == pytest.approx(_LN_1_1 / 5.0)


class TestReturnAttributionShape:
    def test_indexed_by_t0(self):
        close = _close([100.0, 110.0, 121.0, 100.0])
        t1 = _t1(close.index, [(0, 1), (2, 3)])
        co_events = pd.Series(1, index=close.index, dtype="int64")

        result = return_attribution(close, t1, co_events)

        pd.testing.assert_index_equal(result.index, t1.index)

    def test_single_bar_event(self):
        close = _close([100.0, 110.0, 121.0])
        t1 = _t1(close.index, [(1, 1)])
        co_events = pd.Series(1, index=close.index, dtype="int64")

        result = return_attribution(close, t1, co_events)

        assert result.iloc[0] == pytest.approx(_LN_1_1)


class TestNormalizeWeights:
    def test_normalizes_to_sum_n(self):
        w = pd.Series([1.0, 2.0, 3.0])

        result = normalize_weights(w)

        assert result.sum() == pytest.approx(3.0)

    def test_preserves_relative_proportions(self):
        w = pd.Series([1.0, 2.0, 3.0])

        result = normalize_weights(w)

        assert result.iloc[1] / result.iloc[0] == pytest.approx(2.0)
        assert result.iloc[2] / result.iloc[0] == pytest.approx(3.0)

    def test_zero_sum_guarded_no_divide_by_zero(self):
        w = pd.Series([0.0, 0.0, 0.0])

        result = normalize_weights(w)

        assert (result == 0).all()
        assert not result.isna().any()

    def test_empty_input_returns_empty(self):
        w = pd.Series([], dtype="float64")

        result = normalize_weights(w)

        assert len(result) == 0

    def test_single_weight_normalizes_to_one(self):
        w = pd.Series([5.0])

        result = normalize_weights(w)

        assert result.iloc[0] == pytest.approx(1.0)
