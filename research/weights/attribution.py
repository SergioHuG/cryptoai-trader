"""Return attribution + weight normalization (AFML Ch.4.10).

``return_attribution`` uses *log* returns (``np.log(close).diff()``) -- a
deliberate, documented split from ``research.labels.barriers``, which uses
*simple* returns (AFML Snippets 3.2/3.5) for the triple-barrier path-walk
and labeling. Both are AFML-faithful; this is a different snippet (4.10)
with different intent. **Do not "harmonize" the two** -- they are
intentionally different.

Indexing matches AFML's ``mpSampleW`` exactly: ``ret.loc[t0:t1]`` is
INCLUSIVE of ``t0`` -- the same convention
``research.weights.concurrency.avg_uniqueness`` uses for
``co_events.loc[t0:t1]``. The per-event *sum* (not average) is divided by
``co_events`` over that same inclusive span, then absolute-valued.

``return_attribution`` returns the raw absolute attribution -- it does NOT
normalize. ``normalize_weights`` is a separate, explicitly-called step,
kept distinct so the locked ordering (raw attribution -> normalize -> x
decay) stays visible at the orchestrator level (Q4c, Q5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["return_attribution", "normalize_weights"]


def return_attribution(
    close: pd.Series, t1: pd.Series, co_events: pd.Series
) -> pd.Series:
    """Per-event return attribution (AFML Snippet 4.10, ``mpSampleW``).

    For each event: ``w_i = | sum_{t in [t0_i, t1_i]} (log_ret_t / c_t) |``,
    where ``log_ret = log(close).diff()``. Takes ``co_events`` as an
    explicit argument and never recomputes it -- same single-source-of-truth
    discipline as :func:`research.weights.concurrency.avg_uniqueness`.

    Parameters
    ----------
    close:
        Bar close prices, UTC ``DatetimeIndex``, ascending. This is the
        only weight-stack function that takes prices.
    t1:
        Event end timestamps, indexed by event-start (``t0``) timestamps.
    co_events:
        Per-bar concurrency counts (typically
        :func:`research.weights.concurrency.num_co_events` output), indexed
        by the full bar grid.

    Returns
    -------
    pd.Series
        Named ``return_attribution``, indexed by ``t1.index``, raw
        absolute attribution (not normalized), ``float64`` dtype.
    """
    log_ret = np.log(close).diff()

    values = [
        float((log_ret.loc[t0:t1v] / co_events.loc[t0:t1v]).sum())
        for t0, t1v in t1.items()
    ]
    attribution = pd.Series(
        values, index=t1.index, name="return_attribution", dtype="float64"
    )
    return attribution.abs()


def normalize_weights(w: pd.Series) -> pd.Series:
    """Normalize a weight Series to sum to its own length (AFML's ``I``).

    ``w_normalized = w * len(w) / w.sum()``. Guards the degenerate
    all-zero (or empty) case by returning ``w`` unchanged rather than
    dividing by zero.

    Parameters
    ----------
    w:
        Raw weight Series (e.g.
        :func:`research.weights.attribution.return_attribution` output).

    Returns
    -------
    pd.Series
        Same index as ``w``, normalized to sum to ``len(w)``, or returned
        unchanged if ``w.sum() == 0`` (including the empty case).
    """
    total = w.sum()
    if total == 0:
        return w.copy()
    return w * (len(w) / total)
