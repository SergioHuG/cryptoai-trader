"""Triple-barrier labeling primitives (AFML Ch.3).

The vertical barrier is measured in *bars forward* (Decision 1), a deliberate
deviation from AFML's wall-clock ``addVerticalBarrier`` to stay coherent with
information-driven dollar-bar sampling. Path-walk and labeling use *simple*
returns (AFML Snippets 3.2 / 3.5).
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd


class Barrier(IntEnum):
    """Which barrier bound an event first.

    Values are chosen to align with ``bin``'s sign in the symmetric case while
    remaining semantically distinct, and to store cleanly as ``int8``.
    """

    LOWER = -1      # stop-loss / lower horizontal touched first
    VERTICAL = 0    # vertical barrier (timeout) bound first
    UPPER = 1       # profit-take / upper horizontal touched first


def _vertical_barrier(
    index: pd.DatetimeIndex,
    t_events: pd.DatetimeIndex,
    max_hp: int,
) -> pd.Series:
    """Bars-forward vertical barrier (Decision 1).

    For each event timestamp, the vertical barrier falls ``max_hp`` *bars*
    later in ``index`` -- a deliberate deviation from AFML's wall-clock
    ``addVerticalBarrier`` (Snippet 3.4), chosen to stay coherent with
    information-driven dollar-bar sampling. Events whose horizon runs past
    the final bar receive ``NaT`` (the path-walk later treats these as
    "walk to end of data"; see :func:`get_events`).

    Parameters
    ----------
    index:
        The full bar index (UTC ``DatetimeIndex``), typically ``close.index``.
    t_events:
        Event timestamps, a subset of ``index`` (typically the output of
        :func:`research.labels.filters.cusum_filter`).
    max_hp:
        Vertical horizon in bars; must be >= 1.

    Returns
    -------
    pd.Series
        Named ``t1``, indexed by ``t_events``, tz-aware datetime dtype
        (tz of ``index`` preserved). ``NaT`` wherever ``position + max_hp``
        exceeds the last bar.

    Raises
    ------
    ValueError
        If ``max_hp < 1``, or if any timestamp in ``t_events`` is not a
        member of ``index`` (fail loud rather than silently misalign).
    """
    if max_hp < 1:
        raise ValueError(f"max_hp must be >= 1, got {max_hp}.")

    positions = index.get_indexer(t_events)
    if (positions == -1).any():
        raise ValueError(
            "t_events contains timestamps that are not members of index."
        )

    n = len(index)
    target = positions + max_hp
    # Clip into range so the gather is always valid, then NaT out the tail.
    clipped = np.clip(target, 0, n - 1) if n else target
    t1 = pd.Series(index[clipped], index=t_events, name="t1")
    t1[target >= n] = pd.NaT
    return t1