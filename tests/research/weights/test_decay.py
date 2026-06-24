"""Acceptance tests for research.weights.decay (Step 4, AFML 4.11).

time_decay (AFML's getTimeDecay) returns DECAY FACTORS, not pre-multiplied
weights (Q5a) -- a pure function of (avg_uniqueness, decay_c) with no
attribution input at all. The decay axis is CUMULATIVE UNIQUENESS
(tW.sort_index().cumsum()), not calendar time (Q5c) -- the most easily
mis-ported detail in AFML's snippet, and the one
TestDecayAxisIsCumulativeUniquenessNotCalendarTime exists to pin.

decay_c <= 1 is enforced in WeightConfig (Q7), not here; this module
accepts any float and applies AFML's formula verbatim, including the
c < 0 truncation regime (Q5b) where clfW[clfW < 0] = 0 zeroes out the
oldest (1 + decay_c) fraction of cumulative uniqueness entirely.
"""
import inspect

import pandas as pd
import pytest

from research.weights.decay import time_decay


def _utc_index(n, start="2024-01-01", freq="15min"):
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC")


def _tw():
    """Three chronologically-ordered events; cumulative uniqueness sums to
    exactly 1.0 ([0.2, 0.5, 1.0]) so every formula branch hand-computes
    cleanly."""
    return pd.Series([0.2, 0.3, 0.5], index=_utc_index(3), name="avg_uniqueness")


class TestTimeDecayNoDecay:
    def test_c_one_is_no_decay(self):
        result = time_decay(_tw(), decay_c=1.0)

        assert list(result) == pytest.approx([1.0, 1.0, 1.0])


class TestTimeDecayLinearRegime:
    def test_c_half_linear_combination(self):
        """slope=(1-0.5)/1.0=0.5, const=1-0.5=0.5 -> 0.5 + 0.5*cumsum."""
        result = time_decay(_tw(), decay_c=0.5)

        assert list(result) == pytest.approx([0.6, 0.75, 1.0])

    def test_c_zero_factors_equal_cumulative_uniqueness(self):
        """slope=1/1.0=1, const=1-1=0 -> factors equal cumsum exactly."""
        result = time_decay(_tw(), decay_c=0.0)

        assert list(result) == pytest.approx([0.2, 0.5, 1.0])


class TestTimeDecayTruncationRegime:
    def test_negative_c_truncates_oldest_cumulative_mass_to_zero(self):
        """c=-0.5: slope=1/((c+1)*1.0)=2.0, const=1-2.0=-1.0 ->
        raw = -1 + 2*cumsum = [-0.6, 0.0, 1.0] -> clipped to [0.0, 0.0, 1.0].
        The oldest (1 + c) = 50% of cumulative uniqueness mass is zeroed
        outright -- exactly the truncation regime locked in Q5b."""
        result = time_decay(_tw(), decay_c=-0.5)

        assert list(result) == pytest.approx([0.0, 0.0, 1.0])

    def test_negative_c_never_produces_negative_factors(self):
        result = time_decay(_tw(), decay_c=-0.9)

        assert (result >= 0).all()


class TestDecayAxisIsCumulativeUniquenessNotCalendarTime:
    def test_identical_tw_values_different_calendar_spacing_give_identical_factors(
        self,
    ):
        """Two fixtures with the SAME avg_uniqueness values but wildly
        different calendar spacing between events must produce identical
        decay factors -- the axis is cumulative uniqueness, never wall-clock
        time (Q5c, the most easily mis-ported AFML detail)."""
        close_together = pd.Series([0.2, 0.3, 0.5], index=_utc_index(3))
        far_apart = pd.Series(
            [0.2, 0.3, 0.5],
            index=pd.DatetimeIndex(
                ["2020-01-01", "2021-06-15", "2024-11-30"], tz="UTC"
            ),
        )

        result_close = time_decay(close_together, decay_c=0.5)
        result_far = time_decay(far_apart, decay_c=0.5)

        assert list(result_close) == pytest.approx(list(result_far))


class TestTimeDecayReturnsFactorsNotWeights:
    def test_signature_takes_no_attribution_input(self):
        """time_decay is a pure factor producer -- it must not accept any
        attribution/weight argument at all (Q5a)."""
        params = list(inspect.signature(time_decay).parameters)

        assert params == ["avg_uniqueness", "decay_c"]


class TestTimeDecayShape:
    def test_indexed_by_t0(self):
        tw = _tw()

        result = time_decay(tw, decay_c=0.5)

        pd.testing.assert_index_equal(result.index, tw.index)

    def test_factors_monotonic_nondecreasing_in_time(self):
        tw = pd.Series(
            [0.1, 0.4, 0.2, 0.3], index=_utc_index(4), name="avg_uniqueness"
        )

        result = time_decay(tw, decay_c=0.3)

        assert all(
            result.iloc[i] <= result.iloc[i + 1] + 1e-12
            for i in range(len(result) - 1)
        )

    def test_newest_event_factor_is_always_one(self):
        tw = _tw()

        for c in (1.0, 0.5, 0.0, -0.5):
            result = time_decay(tw, decay_c=c)
            assert result.iloc[-1] == pytest.approx(1.0)

    def test_empty_input_returns_empty(self):
        tw = pd.Series([], dtype="float64")

        result = time_decay(tw, decay_c=0.5)

        assert len(result) == 0

    def test_single_event_factor_is_always_one_regardless_of_c(self):
        tw = pd.Series([0.7], index=[_utc_index(1)[0]])

        for c in (1.0, 0.3, 0.0, -0.4):
            result = time_decay(tw, decay_c=c)
            assert result.iloc[0] == pytest.approx(1.0)
