"""Purge & embargo primitives -- leak-proofing foundation for cross-validation
(AFML Ch.7, Snippets 7.1/7.2).

This module is built in three layers, each red-first and independently
testable:
  * get_embargo_times (7.2)  -- maps every bar to a timestamp `step` bars
    ahead, clamping the tail to the last bar.
  * get_train_times (7.1)    -- drops train observations whose [t0, t1]
    span overlaps a test span, under all three AFML overlap conditions.
  * _purge_embargo            -- the shared kernel both PurgedKFold and
    CombinatorialPurgedKFold consume: extends each test block forward by
    the embargo, then purges train around the embargoed span. One source
    of truth for purge+embargo semantics; the splitters are thin
    positional adapters over it (Q4/Q6).
"""
from __future__ import annotations

import pandas as pd

__all__ = ["get_embargo_times", "get_train_times"]


def get_embargo_times(times: pd.DatetimeIndex, embargo_pct: float) -> pd.Series:
    """Map every bar to a forward-embargoed timestamp (AFML Snippet 7.2).

    For ``step = int(len(times) * embargo_pct)`` bars, each timestamp is
    mapped to the timestamp ``step`` positions ahead. The trailing ``step``
    timestamps -- which have no full step of room ahead of them -- are
    clamped to the very last timestamp in ``times``, AFML's own tail
    behavior rather than a guard added on top of it.

    ``step == 0`` (either ``embargo_pct == 0`` or ``embargo_pct`` too small
    to produce a nonzero step on this grid) is the identity mapping: every
    timestamp maps to itself, i.e. no embargo.

    This is a standalone primitive operating on the *full bar grid*, not on
    event spans -- :func:`_purge_embargo` is what threads its output into
    an actual purge.

    Parameters
    ----------
    times:
        The full sample grid (UTC ``DatetimeIndex``). Must be strictly
        monotonic increasing -- "step bars ahead" is only meaningful
        chronologically if the grid is ascending.
    embargo_pct:
        Fraction of ``len(times)`` to embargo forward. Must satisfy
        ``0 <= embargo_pct < 1``.

    Returns
    -------
    pd.Series
        Named ``embargo_times``, indexed by ``times`` (same order), values
        are timestamps of the same dtype as ``times``.

    Raises
    ------
    ValueError
        If ``times`` is not strictly monotonic increasing, or if
        ``embargo_pct`` is outside ``[0, 1)``.
    """
    if not times.is_monotonic_increasing:
        raise ValueError(
            "get_embargo_times requires times to be strictly monotonic "
            "increasing -- 'step bars ahead' is only meaningful "
            "chronologically on an ascending grid."
        )
    if not (0 <= embargo_pct < 1):
        raise ValueError(
            f"get_embargo_times requires 0 <= embargo_pct < 1, got "
            f"{embargo_pct!r}."
        )

    n = len(times)
    step = int(n * embargo_pct)

    if step == 0:
        return pd.Series(times, index=times, name="embargo_times")

    shifted = pd.Series(times[step:], index=times[:-step])
    tail = pd.Series(
        pd.DatetimeIndex([times[-1]] * step), index=times[-step:]
    )
    result = pd.concat([shifted, tail])
    result.name = "embargo_times"
    return result


def get_train_times(t1: pd.Series, test_times: pd.Series) -> pd.Series:
    """Purge train observations overlapping any test span (AFML Snippet 7.1).

    Kept as AFML's naive loop in production -- a deliberate asymmetry from
    ``num_co_events``'s vectorized sweep (Q5). The outer loop here runs
    over ``test_times``' rows (a handful of test-fold spans, or CPCV
    group-combinations), each iteration doing three already-vectorized
    boolean masks against ``t1``. There is no hot path to optimize away:
    ``num_co_events`` is vectorized because its outer loop runs over every
    bar in the dataset; this one does not.

    For every ``(i, j)`` row in ``test_times`` (test span start/end), drops
    any train observation ``(t0, t1_val)`` from ``t1`` where:
      * ``i <= t0 <= j``           -- train starts inside the test span.
      * ``i <= t1_val <= j``       -- train ends inside the test span.
      * ``t0 <= i and j <= t1_val`` -- train envelops the test span.

    All three conditions use inclusive bounds -- a train span merely
    touching a test span at its boundary still overlaps it.

    ``test_times`` may carry more than one row -- a union of disjoint test
    blocks (the CPCV shape, Q6). Each row is purged against independently
    and in sequence; once a train observation is dropped for overlapping
    one block it is not reconsidered against later blocks, which is
    equivalent to (and simpler than) purging against the union directly.

    Parameters
    ----------
    t1:
        Train event end timestamps, indexed by event-start (``t0``)
        timestamps.
    test_times:
        Test span(s): a Series indexed by each test span's start
        timestamp, with the corresponding end timestamp as the value. One
        row per contiguous test block.

    Returns
    -------
    pd.Series
        The surviving subset of ``t1`` -- same dtype, unchanged values,
        filtered index. Order and values of survivors are untouched.
    """
    trn = t1.copy(deep=True)
    for i, j in test_times.items():
        starts_inside = trn[(i <= trn.index) & (trn.index <= j)].index
        ends_inside = trn[(i <= trn) & (trn <= j)].index
        envelops = trn[(trn.index <= i) & (j <= trn)].index
        drop = starts_inside.union(ends_inside).union(envelops)
        trn = trn.drop(drop)
    return trn


def _purge_embargo(
    t1: pd.Series,
    test_times: pd.Series,
    bars: pd.DatetimeIndex,
    embargo_pct: float,
) -> pd.Index:
    """Shared purge+embargo kernel (Q4/Q6) -- the single source of truth
    both ``PurgedKFold`` and ``CombinatorialPurgedKFold`` consume.

    Composes :func:`get_embargo_times` and :func:`get_train_times` rather
    than inlining AFML 7.3's positional arithmetic (Q4): extends each test
    block's right edge forward by the embargo, looked up against the
    *full* bar grid (embargo_pct is a fraction of the total sample count,
    not of any one block), then purges train around the now-embargoed
    block spans.

    ``test_times`` is one row PER CONTIGUOUS TEST BLOCK, not per
    individual test event: index = block start, value = the block's
    *max* t1 (the latest resolution time among that block's events) --
    already aggregated by the caller, since identifying which positions
    belong to a block is a splitter-level positional concern this
    timestamp-space kernel stays agnostic to. Multiple rows are a union of
    disjoint blocks (the CPCV shape, Q6); each block's embargo is computed
    and applied independently (per-block forward embargo), exactly as
    locked.

    Only the block's right edge is extended -- the left edge (block
    start) is untouched, so embargo is structurally forward-only: a train
    observation entirely before a block's start is governed by neither
    purge nor embargo, regardless of ``embargo_pct``.

    Parameters
    ----------
    t1:
        Train event end timestamps, indexed by event-start (``t0``)
        timestamps -- passed straight through to :func:`get_train_times`.
    test_times:
        One row per contiguous test block: index = block start timestamp,
        value = block's max t1 timestamp.
    bars:
        The full sample grid (UTC ``DatetimeIndex``) -- needed to compute
        the embargo step as a fraction of the *total* sample count, and
        for :func:`get_embargo_times`'s tail-clamp behavior.
    embargo_pct:
        Fraction of ``len(bars)`` to embargo forward of each block.

    Returns
    -------
    pd.Index
        Surviving train ``t0`` timestamps -- the purged-and-embargoed
        train index.
    """
    embargo_map = get_embargo_times(bars, embargo_pct)
    embargoed_test_times = test_times.map(embargo_map)
    survivors = get_train_times(t1, embargoed_test_times)
    return survivors.index