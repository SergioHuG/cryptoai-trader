"""build_triple_barrier_labels -- the triple-barrier labeling orchestrator.

A thin wiring layer on top of the AFML-faithful primitives
(``cusum_filter``/``get_events``/``get_bins``): computes ``ewma_vol`` exactly
once and feeds it into both the CUSUM threshold and the barrier target,
enforcing the single-shared-vol invariant in code rather than convention.

Pure compute, no I/O -- persistence is a separate, explicit step handled by
:mod:`research.labels.storage`.
"""
from __future__ import annotations

import pandas as pd

from research.features.volatility import ewma_vol
from research.labels.barriers import get_bins, get_events
from research.labels.config import LabelConfig
from research.labels.filters import cusum_filter

__all__ = ["build_triple_barrier_labels"]


def build_triple_barrier_labels(
    close: pd.Series,
    config: LabelConfig,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """Run the full CUSUM -> triple-barrier -> labeling pipeline.

    Computes ``vol = ewma_vol(close, config.ewma_span)`` once. Because
    ``ewma_vol`` has a warm-up region of ``NaN`` values, sampling
    :func:`cusum_filter` across that boundary uninspected produces a
    spurious event at the first valid-vol bar (accumulated drift through
    the ``NaN`` region trips the freshly-valid threshold) -- empirically
    verified, not just reasoned about. To avoid this, ``close``/``h`` are
    sliced to the first valid-vol bar *before* CUSUM sampling.

    The vertical barrier and path-walk (inside :func:`get_events`) still
    index the *full* ``close`` -- only the CUSUM event-sampling step is
    sliced.

    Parameters
    ----------
    close:
        Bar close prices, UTC ``DatetimeIndex``, ascending.
    config:
        The :class:`LabelConfig` recipe driving every knob below.
    side:
        Optional per-event bet side for meta-labeling, indexed by event
        timestamps (a subset of the post-slice CUSUM event timestamps).
        ``None`` -> symmetric labeling.

    Returns
    -------
    pd.DataFrame
        The output of :func:`get_bins` -- see that function's docstring for
        the exact column shape (symmetric vs. meta-labeling).
    """
    vol = ewma_vol(close, span=config.ewma_span)
    h = config.kappa * vol

    first_valid = vol.first_valid_index()
    if first_valid is None:
        t_events = pd.DatetimeIndex([], tz=close.index.tz)
    else:
        t_events = cusum_filter(close.loc[first_valid:], h.loc[first_valid:])

    events = get_events(
        close,
        t_events,
        (config.pt_mult, config.sl_mult),
        trgt=vol,
        max_hp=config.max_hp,
        min_ret=config.min_ret,
        side=side,
    )
    return get_bins(events, close)
