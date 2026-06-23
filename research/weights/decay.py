"""Time-decay factors over cumulative uniqueness (AFML Ch.4.11).

``time_decay`` (AFML's ``getTimeDecay``) returns DECAY FACTORS, not
pre-multiplied weights -- a pure function of ``(avg_uniqueness, decay_c)``
with no attribution input at all (Q5a). The decay axis is the CUMULATIVE
SUM of ``avg_uniqueness`` (``tW.sort_index().cumsum()``), never calendar
time -- the most easily mis-ported detail in AFML's snippet (Q5c).

``decay_c`` semantics (validated as a ``WeightConfig`` field, not here --
this module accepts any float and applies AFML's formula verbatim):
  * ``c == 1``    -> no decay (every factor == 1.0).
  * ``0 < c < 1`` -> linear decay; the factor at cumulative-uniqueness
                     ``== 0`` (an intercept, not necessarily the oldest
                     event itself) equals ``c``.
  * ``c == 0``    -> factors equal cumulative uniqueness exactly (linear
                     from 0 to 1).
  * ``c < 0``     -> truncation regime: the oldest ``(1 + c)`` fraction of
                     cumulative uniqueness is zeroed outright
                     (``clfW[clfW < 0] = 0``).

Known AFML formula limitation (not guarded here, per Q5's "no lower
bound" on ``decay_c``): at exactly ``decay_c == -1``, the negative branch
divides by ``(decay_c + 1) == 0``. This is a limitation baked into AFML's
own formula, not introduced by this port. (Parked for the throughline
write-up.)

A provable invariant from the formula (``const + slope * total == 1``
always): the NEWEST event's factor is always exactly ``1.0``, for any
``decay_c``.
"""
from __future__ import annotations

import pandas as pd

__all__ = ["time_decay"]


def time_decay(avg_uniqueness: pd.Series, decay_c: float) -> pd.Series:
    """Decay factors over cumulative average uniqueness (AFML Snippet 4.11).

    Parameters
    ----------
    avg_uniqueness:
        Per-event average uniqueness (e.g.
        :func:`research.weights.concurrency.avg_uniqueness` output),
        indexed by event-start (``t0``) timestamps.
    decay_c:
        AFML's ``clfLastW``. ``c == 1`` disables decay; ``0 <= c < 1``
        linearly decays toward ``c``; ``c < 0`` truncates the oldest
        ``(1 + c)`` fraction of cumulative uniqueness to zero. Validation
        (``c <= 1``) lives in :class:`research.weights.config.WeightConfig`,
        not here.

    Returns
    -------
    pd.Series
        Decay factors, indexed by ``avg_uniqueness.index`` sorted ascending
        (chronological), ``float64`` dtype. Empty if ``avg_uniqueness`` is
        empty.
    """
    if avg_uniqueness.empty:
        return pd.Series(
            [], index=avg_uniqueness.index, dtype="float64", name="time_decay_factor"
        )

    clf_w = avg_uniqueness.sort_index().cumsum()
    total = clf_w.iloc[-1]

    if decay_c >= 0:
        slope = (1.0 - decay_c) / total
    else:
        slope = 1.0 / ((decay_c + 1.0) * total)
    const = 1.0 - slope * total

    clf_w = const + slope * clf_w
    clf_w[clf_w < 0] = 0.0
    clf_w.name = "time_decay_factor"

    return clf_w
