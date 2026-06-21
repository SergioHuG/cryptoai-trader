"""Symmetric CUSUM event filter (AFML Snippet 2.4).

Runs on log returns of close (scale-invariant). The threshold may be a scalar
constant ``h`` or a per-timestamp Series (vol-scaled ``h_t = kappa * sigma_t``
is the default usage). Each crossing registers an event and resets the breached
accumulator arm, so the filter samples activity symmetrically in both
directions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["cusum_filter"]


def cusum_filter(close: pd.Series, threshold: float | pd.Series) -> pd.DatetimeIndex:
    """Return the timestamps at which the symmetric CUSUM filter fires.

    Parameters
    ----------
    close:
        Bar close prices indexed by a (UTC) DatetimeIndex.
    threshold:
        Scalar ``h`` applied at every bar, or a Series of per-bar thresholds
        aligned to ``close``'s index.

    Returns
    -------
    pd.DatetimeIndex
        Event timestamps, a subset of ``close.index`` (tz preserved). Empty if
        no crossing occurs.
    """
    diff = np.log(close).diff().dropna()
    idx = diff.index

    if isinstance(threshold, pd.Series):
        thr = threshold.reindex(idx).to_numpy(dtype=float)
    else:
        thr = np.full(len(idx), float(threshold))

    values = diff.to_numpy(dtype=float)
    s_pos = 0.0
    s_neg = 0.0
    fired: list[int] = []

    for i in range(len(values)):
        y = values[i]
        h = thr[i]
        s_pos = max(0.0, s_pos + y)
        s_neg = min(0.0, s_neg + y)
        if s_neg < -h:
            s_neg = 0.0
            fired.append(i)
        elif s_pos > h:
            s_pos = 0.0
            fired.append(i)

    return idx[fired]
