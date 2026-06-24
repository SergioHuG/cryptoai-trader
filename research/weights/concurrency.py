"""Concurrency primitives -- shared event-overlap foundation (AFML Ch.4.1-4.2).

num_co_events implements a difference-array boundary sweep -- O(events +
bars), independent of span width -- rather than AFML's naive per-bar
accumulation loop (``mpNumCoEvents``). The sweep is *exactly* equal to the
naive loop, never an approximation; see
``tests/research/weights/test_concurrency.py::TestNumCoEventsSweepMatchesNaiveOracle``
for the equivalence proof, which keeps the naive loop alive as a test
oracle only -- it never runs in production (Q6).

avg_uniqueness (AFML Snippet 4.2, ``mpSampleTW``) takes ``co_events`` as an
explicit argument and never recomputes it. This is the seam that lets
``research.weights.pipeline.build_sample_weights`` guarantee a single
shared ``num_co_events()`` call feeds every downstream consumer
(``avg_uniqueness`` AND ``return_attribution``) -- enforced there by a
call-count spy (Q3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["num_co_events", "avg_uniqueness"]


def num_co_events(close_index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """Count, for every bar in ``close_index``, how many events span it.

    AFML Snippet 4.1 (``mpNumCoEvents``), reimplemented as a boundary
    sweep: ``+1`` at each event's ``t0``, ``-1`` at the bar immediately
    after its ``t1``, then a cumulative sum over the full bar grid. This is
    mathematically identical to AFML's naive per-event ``.loc``-slice
    accumulation but runs in O(events + bars) rather than O(events *
    average_span).

    Parameters
    ----------
    close_index:
        The full bar grid (UTC ``DatetimeIndex``), typically ``close.index``.
    t1:
        Event end timestamps, indexed by event-start (``t0``) timestamps.
        Must be fully resolved (no nulls) -- this consumes the output of
        ``research.labels.barriers.get_bins``, which drops unresolved
        events by construction. A null ``t1`` reaching here means the
        caller passed an unresolved ``get_events`` frame directly, which is
        a bug, not a valid "still open" case -- a deliberate divergence
        from AFML's defensive ``t1.fillna(close_index[-1])``. (Parked: a
        future "rolling/open-events" online-weighting variant would want
        that fillna behavior; this base contract stays fail-loud.)

    Returns
    -------
    pd.Series
        Named ``num_co_events``, indexed by the full ``close_index``,
        ``int64`` dtype, 0-filled wherever no event spans that bar.

    Raises
    ------
    ValueError
        If ``t1`` contains any null values, or if any ``t0``/``t1``
        timestamp is not a member of ``close_index``.
    """
    if t1.isna().any():
        raise ValueError(
            "num_co_events requires a fully resolved t1 (no nulls). This "
            "consumes get_bins output, which drops unresolved events by "
            "construction; a null t1 here means an unresolved get_events "
            "frame was passed directly."
        )

    n = len(close_index)

    start_pos = close_index.get_indexer(t1.index)
    if (start_pos == -1).any():
        raise ValueError(
            "t1's index contains timestamps that are not members of "
            "close_index."
        )

    end_pos = close_index.get_indexer(pd.DatetimeIndex(t1.to_numpy()))
    if (end_pos == -1).any():
        raise ValueError(
            "t1's values contain timestamps that are not members of "
            "close_index."
        )

    # Boundary sweep: +1 at t0, -1 just after t1, then cumsum over the grid.
    # diff has one extra slot so a decrement landing at position == n (an
    # event ending on the very last bar) has somewhere safe to go without
    # ever being included in the cumsum below.
    diff = np.zeros(n + 1, dtype="int64")
    np.add.at(diff, start_pos, 1)
    np.add.at(diff, end_pos + 1, -1)
    counts = np.cumsum(diff[:n])

    return pd.Series(counts, index=close_index, name="num_co_events")


def avg_uniqueness(t1: pd.Series, co_events: pd.Series) -> pd.Series:
    """Average uniqueness per event (AFML Snippet 4.2, ``mpSampleTW``).

    For each event, averages the reciprocal concurrency (``1 /
    co_events``) across every bar the event spans:
    ``tW_i = mean_{t in [t0_i, t1_i]} (1 / c_t)``. Takes the precomputed
    ``co_events`` as an explicit argument and never recomputes it
    internally -- see the module docstring for why that's a load-bearing
    contract, not a convenience.

    Parameters
    ----------
    t1:
        Event end timestamps, indexed by event-start (``t0``) timestamps.
    co_events:
        Per-bar concurrency counts, typically the output of
        :func:`num_co_events`, indexed by the full bar grid.

    Returns
    -------
    pd.Series
        Named ``avg_uniqueness``, indexed by ``t1.index`` (same order,
        duplicates preserved), ``float64`` dtype.
    """
    values = [
        float((1.0 / co_events.loc[t0:t1v]).mean()) for t0, t1v in t1.items()
    ]
    return pd.Series(
        values, index=t1.index, name="avg_uniqueness", dtype="float64"
    )
