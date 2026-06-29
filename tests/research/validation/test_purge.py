"""Acceptance tests for research/validation/purge.py (Step 2).

Step 2a: get_embargo_times (AFML Snippet 7.2) -- maps every bar in the full
sample grid to the timestamp `step = int(n * embargo_pct)` bars ahead,
clamping the trailing `step` bars to the last timestamp (AFML's own tail
behavior, not a guard we added).

Two defensive checks are added at the primitive level, beyond AFML's bare
snippet, consistent with the house fail-loud posture (num_co_events
validates its own preconditions rather than trusting the caller already
checked):
  * `times` must be strictly monotonic increasing -- "step bars ahead" is
    only meaningful chronologically if the grid is ascending.
  * `embargo_pct` must satisfy 0 <= embargo_pct < 1 -- mirrors
    ValidationConfig's bound, enforced again here so the primitive is
    trustworthy when called directly, not only via the config layer.

Step 2b: get_train_times (AFML Snippet 7.1) -- the NAIVE loop, kept in
production deliberately (Q5 asymmetry from num_co_events: the outer loop
is over a handful of test spans, not every bar, so there is no hot path
to vectorize away). Drops every train observation whose [t0, t1] span
overlaps ANY test span under all three AFML conditions: starts inside,
ends inside, or envelops. `test_times` may carry more than one row (a
union of disjoint blocks, the CPCV shape, Q6) -- each row is purged
against independently and the drops are unioned.

Step 2c: _purge_embargo -- the shared kernel composing 2a + 2b (Q4/Q6).
One row per CONTIGUOUS TEST BLOCK (not per individual test event): each
block's right edge (the max t1 among its events) is extended forward by
the embargo via get_embargo_times looked up against the FULL bar grid,
then get_train_times purges train around the (now-embargoed) block
spans. Deliberately tested DIRECTLY despite the leading underscore --
unlike storage.py's _weights_key (trivial, only exercised indirectly via
its public callers), this kernel is the single load-bearing leak-proofing
unit both PurgedKFold and CombinatorialPurgedKFold are thin adapters
over (Q6); their own tests check positional mapping and combination
enumeration only, not re-derive purge+embargo correctness.
"""
import pandas as pd
import pytest

from research.validation.purge import _purge_embargo, get_embargo_times, get_train_times


def _bars(n: int, freq: str = "1h"):
    return pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")


class TestGetEmbargoTimesStepZero:
    def test_zero_pct_is_identity(self):
        times = _bars(10)
        result = get_embargo_times(times, embargo_pct=0.0)
        for t in times:
            assert result.loc[t] == t

    def test_pct_too_small_to_produce_a_step_is_identity(self):
        """int(10 * 0.05) == 0 -- same identity branch as pct == 0."""
        times = _bars(10)
        result = get_embargo_times(times, embargo_pct=0.05)
        for t in times:
            assert result.loc[t] == t


class TestGetEmbargoTimesGeneralCase:
    def test_matches_afml_formula_step_two(self):
        """n=10, embargo_pct=0.2 -> step=2.

        AFML 7.2: index[:-step] -> values[step:] (shift-ahead-by-step),
        then index[-step:] -> last timestamp (tail clamp).
        """
        times = _bars(10)
        result = get_embargo_times(times, embargo_pct=0.2)

        assert len(result) == 10
        for i in range(8):  # 0..7 -> shifted ahead by step=2
            assert result.loc[times[i]] == times[i + 2]
        for i in range(8, 10):  # 8, 9 -> clamped to the last bar
            assert result.loc[times[i]] == times[-1]

    def test_matches_afml_formula_step_three_on_larger_grid(self):
        """n=20, embargo_pct=0.15 -> step=3 -- a second, independent
        parametrization of the same formula on a different grid size."""
        times = _bars(20)
        result = get_embargo_times(times, embargo_pct=0.15)

        for i in range(17):
            assert result.loc[times[i]] == times[i + 3]
        for i in range(17, 20):
            assert result.loc[times[i]] == times[-1]

    def test_near_full_embargo_collapses_every_bar_to_last(self):
        """n=10, embargo_pct=0.99 -> step=9. Every single bar -- including
        the first -- maps to the last bar (step spans the whole grid)."""
        times = _bars(10)
        result = get_embargo_times(times, embargo_pct=0.99)
        for t in times:
            assert result.loc[t] == times[-1]


class TestGetEmbargoTimesShape:
    def test_indexed_by_the_full_input_grid_in_order(self):
        times = _bars(15)
        result = get_embargo_times(times, embargo_pct=0.2)
        assert list(result.index) == list(times)

    def test_values_are_datetime_like_and_tz_aware(self):
        times = _bars(10)
        result = get_embargo_times(times, embargo_pct=0.2)
        assert result.dtype == times.dtype  # tz-aware datetime64[ns, UTC]


class TestGetEmbargoTimesValidation:
    def test_non_monotonic_times_raises(self):
        times = _bars(10)
        shuffled = times[[0, 2, 1, 3, 4, 5, 6, 7, 8, 9]]
        with pytest.raises(ValueError):
            get_embargo_times(shuffled, embargo_pct=0.2)

    def test_negative_embargo_pct_raises(self):
        times = _bars(10)
        with pytest.raises(ValueError):
            get_embargo_times(times, embargo_pct=-0.01)

    def test_embargo_pct_one_raises(self):
        times = _bars(10)
        with pytest.raises(ValueError):
            get_embargo_times(times, embargo_pct=1.0)

    def test_embargo_pct_above_one_raises(self):
        times = _bars(10)
        with pytest.raises(ValueError):
            get_embargo_times(times, embargo_pct=1.5)


# ── get_train_times (Step 2b) ────────────────────────────────────────────────
#
# Shared 20-bar grid for every get_train_times test below. A single test
# span is [bars[5], bars[8]] unless a test builds its own.

_BARS20 = _bars(20)


def _train(t0_idx: int, t1_idx: int) -> pd.Series:
    """A one-row train t1 Series: {bars[t0_idx]: bars[t1_idx]}."""
    return pd.Series({_BARS20[t0_idx]: _BARS20[t1_idx]})


def _test_span(*pairs: tuple) -> pd.Series:
    """A test_times Series from (t0_idx, t1_idx) pairs."""
    return pd.Series(
        {_BARS20[i]: _BARS20[j] for i, j in pairs}
    )


class TestGetTrainTimesOverlapConditionsInIsolation:
    def test_train_starts_inside_test_span_is_dropped(self):
        """df0: i <= train_t0 <= j. Train [6,10] vs test [5,8]: starts at
        6 (inside [5,8]), ends at 10 (outside) -- df0 only."""
        train = _train(6, 10)
        test_times = _test_span((5, 8))
        result = get_train_times(train, test_times)
        assert len(result) == 0

    def test_train_ends_inside_test_span_is_dropped(self):
        """df1: i <= train_t1 <= j. Train [2,6] vs test [5,8]: starts at 2
        (before), ends at 6 (inside [5,8]) -- df1 only."""
        train = _train(2, 6)
        test_times = _test_span((5, 8))
        result = get_train_times(train, test_times)
        assert len(result) == 0

    def test_train_envelops_test_span_is_dropped(self):
        """df2: train_t0 <= i and j <= train_t1. Train [3,12] vs test
        [5,8]: starts before (3<=5), ends after (8<=12) -- df2 only."""
        train = _train(3, 12)
        test_times = _test_span((5, 8))
        result = get_train_times(train, test_times)
        assert len(result) == 0

    def test_boundary_train_end_equals_test_start_is_dropped(self):
        """Inclusive bounds: train_t1 exactly equals test i (5) counts as
        'ends inside' -- df1 with i==train_t1==5."""
        train = pd.Series({_BARS20[3]: _BARS20[5]})  # ends exactly at i=5
        test_times = _test_span((5, 8))
        result = get_train_times(train, test_times)
        assert len(result) == 0

    def test_boundary_touching_at_test_end_is_dropped(self):
        """train_t0 exactly equals test j (8) counts as 'starts inside' --
        df0 with train_t0==j==8."""
        train = _train(8, 10)
        test_times = _test_span((5, 8))
        result = get_train_times(train, test_times)
        assert len(result) == 0


class TestGetTrainTimesNoOverlapSurvives:
    def test_train_entirely_before_test_span_survives(self):
        train = _train(0, 3)
        test_times = _test_span((5, 8))
        result = get_train_times(train, test_times)
        assert len(result) == 1
        assert result.loc[_BARS20[0]] == _BARS20[3]

    def test_train_entirely_after_test_span_survives(self):
        train = _train(10, 12)
        test_times = _test_span((5, 8))
        result = get_train_times(train, test_times)
        assert len(result) == 1
        assert result.loc[_BARS20[10]] == _BARS20[12]

    def test_surviving_values_are_unchanged_from_input(self):
        """get_train_times filters -- it never mutates surviving t1
        values."""
        train = pd.concat([_train(0, 3), _train(10, 12)])
        test_times = _test_span((5, 8))
        result = get_train_times(train, test_times)
        assert result.loc[_BARS20[0]] == _BARS20[3]
        assert result.loc[_BARS20[10]] == _BARS20[12]


class TestGetTrainTimesMultiBlockUnion:
    """test_times carries two disjoint blocks -- the CPCV shape (Q6): each
    train observation is purged against EVERY block independently, and the
    drops are unioned."""

    def test_drops_overlap_with_either_block(self):
        test_times = _test_span((5, 8), (14, 16))
        train = pd.concat(
            [
                _train(6, 7),  # inside block 1 [5,8] -> dropped
                _train(9, 12),  # between blocks, no overlap -> survives
                pd.Series({_BARS20[15]: _BARS20[15]}),  # zero-width, inside block 2 -> dropped
                _train(17, 18),  # entirely after block 2 -> survives
            ]
        )
        result = get_train_times(train, test_times)

        assert len(result) == 2
        assert _BARS20[9] in result.index
        assert _BARS20[17] in result.index
        assert _BARS20[6] not in result.index
        assert _BARS20[15] not in result.index

    def test_survivor_values_unchanged(self):
        test_times = _test_span((5, 8), (14, 16))
        train = pd.concat([_train(9, 12), _train(17, 18)])
        result = get_train_times(train, test_times)
        assert result.loc[_BARS20[9]] == _BARS20[12]
        assert result.loc[_BARS20[17]] == _BARS20[18]


class TestGetTrainTimesShape:
    def test_empty_train_returns_empty(self):
        empty_idx = pd.DatetimeIndex([], tz="UTC")
        train = pd.Series(empty_idx, index=empty_idx, dtype="datetime64[ns, UTC]")
        test_times = _test_span((5, 8))
        result = get_train_times(train, test_times)
        assert len(result) == 0

    def test_index_dtype_matches_input(self):
        train = pd.concat([_train(0, 3), _train(10, 12)])
        test_times = _test_span((5, 8))
        result = get_train_times(train, test_times)
        assert result.dtype == train.dtype


# ── _purge_embargo kernel (Step 2c) ──────────────────────────────────────────
#
# test_times for the kernel is ONE ROW PER CONTIGUOUS BLOCK: index = block
# start, value = block's max t1 (the latest resolution time among that
# block's events) -- already block-aggregated by the caller (the splitter).
# A fresh 20-bar grid, embargo_pct=0.1 -> step=2, used throughout.

_KBARS = _bars(20)


def _kernel_train(t0_idx: int, t1_idx: int) -> pd.Series:
    return pd.Series({_KBARS[t0_idx]: _KBARS[t1_idx]})


def _kernel_block(start_idx: int, max_t1_idx: int) -> pd.Series:
    return pd.Series({_KBARS[start_idx]: _KBARS[max_t1_idx]})


class TestPurgeEmbargoKernelSingleBlock:
    """Block: start=bars[4], max_t1=bars[7]. embargo_pct=0.1 (step=2) ->
    embargoed end = bars[9]."""

    def test_train_before_block_survives_unaffected_by_embargo(self):
        """Forward-only embargo: train entirely before the block's start
        is governed by neither purge nor embargo."""
        train = _kernel_train(0, 2)
        test_times = _kernel_block(4, 7)
        result = _purge_embargo(train, test_times, _KBARS, embargo_pct=0.1)
        assert _KBARS[0] in result

    def test_train_overlapping_original_span_is_purge_governed(self):
        """Train inside [4,7] is dropped by ordinary purge -- true with or
        without embargo, confirms the kernel still purges correctly."""
        train = _kernel_train(5, 6)
        test_times = _kernel_block(4, 7)
        result = _purge_embargo(train, test_times, _KBARS, embargo_pct=0.1)
        assert _KBARS[5] not in result

    def test_train_in_embargo_extension_zone_is_dropped(self):
        """Train at bar 8 -- strictly AFTER the original span [4,7], so
        plain purge alone would NOT catch it -- falls inside the embargoed
        zone (7, 9] and IS dropped. This is the embargo-governed case."""
        train = _kernel_train(8, 8)
        test_times = _kernel_block(4, 7)
        result = _purge_embargo(train, test_times, _KBARS, embargo_pct=0.1)
        assert _KBARS[8] not in result

    def test_train_beyond_embargo_zone_survives(self):
        """Train at bar 10 -- beyond the embargoed boundary (9) -- survives.
        Confirms the embargo's extent is finite, not open-ended."""
        train = _kernel_train(10, 11)
        test_times = _kernel_block(4, 7)
        result = _purge_embargo(train, test_times, _KBARS, embargo_pct=0.1)
        assert _KBARS[10] in result


class TestPurgeEmbargoKernelZeroEmbargoReducesToPurge:
    def test_matches_plain_get_train_times_when_embargo_is_zero(self):
        """embargo_pct == 0 -> get_embargo_times is identity -> the kernel
        degenerates to a direct get_train_times call, no extension."""
        train = pd.concat(
            [_kernel_train(5, 6), _kernel_train(8, 8), _kernel_train(10, 11)]
        )
        test_times = _kernel_block(4, 7)

        kernel_result = _purge_embargo(
            train, test_times, _KBARS, embargo_pct=0.0
        )
        direct_result = get_train_times(train, test_times)

        assert set(kernel_result) == set(direct_result.index)
        # Specifically: bar 8 now SURVIVES with no embargo (unlike the
        # embargo_pct=0.1 case above).
        assert _KBARS[8] in kernel_result


class TestPurgeEmbargoKernelFinalBlockNoForwardRegion:
    def test_block_ending_at_last_bar_has_no_embargo_to_apply(self):
        """Block's max t1 IS the grid's last bar -- get_embargo_times
        clamps it to itself, so embargo is a structural no-op: nothing
        exists past the grid's last bar to embargo in the first place."""
        train = pd.concat([_kernel_train(0, 2), _kernel_train(15, 16)])
        test_times = _kernel_block(17, 19)  # ends at _KBARS[19], the last bar

        kernel_result = _purge_embargo(
            train, test_times, _KBARS, embargo_pct=0.1
        )
        direct_result = get_train_times(train, test_times)

        assert set(kernel_result) == set(direct_result.index)


class TestPurgeEmbargoKernelMultiBlock:
    """Two disjoint blocks (the CPCV shape, Q6): block1 start=2/max_t1=4,
    block2 start=10/max_t1=12. embargo_pct=0.1 (step=2) -> embargoed ends
    6 and 14 respectively, applied independently per block."""

    def _two_blocks(self):
        return pd.concat([_kernel_block(2, 4), _kernel_block(10, 12)])

    def test_drops_train_in_either_blocks_embargo_zone(self):
        train = pd.concat(
            [
                _kernel_train(5, 5),  # inside block1's embargo zone (4,6]
                _kernel_train(7, 8),  # between blocks -> survives
                _kernel_train(13, 13),  # inside block2's embargo zone (12,14]
                _kernel_train(16, 17),  # after both -> survives
            ]
        )
        result = _purge_embargo(
            train, self._two_blocks(), _KBARS, embargo_pct=0.1
        )

        assert _KBARS[5] not in result
        assert _KBARS[7] in result
        assert _KBARS[13] not in result
        assert _KBARS[16] in result


class TestPurgeEmbargoKernelReturnShape:
    def test_returns_a_plain_index(self):
        train = _kernel_train(0, 2)
        test_times = _kernel_block(4, 7)
        result = _purge_embargo(train, test_times, _KBARS, embargo_pct=0.1)
        assert isinstance(result, pd.Index)