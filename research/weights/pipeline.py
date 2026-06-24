"""build_sample_weights orchestrator (Step 6, Q3/Q9).

Pure-compute wiring layer: computes num_co_events exactly once and
threads it to BOTH avg_uniqueness and return_attribution -- the
single-shared-c_t invariant (Q3), the same discipline as the labels
pipeline's single-shared ewma_vol.

Locked order (Q4c, Q5d): raw attribution -> normalize_weights -> x
time_decay, with NO post-decay renormalization. Output is a three-column
frame [avg_uniqueness, weight, t1] (Q9) -- avg_uniqueness for
bootstrap/BaggingClassifier sizing, weight as the final sample_weight
deliverable, t1 carried straight through from the labels frame so the
weights artifact is self-sufficient for concurrency-aware CV (no join
back to labels needed downstream in research.validation).

No I/O -- persistence is research.weights.storage's job.
"""
from __future__ import annotations

import pandas as pd

from research.weights.attribution import normalize_weights, return_attribution
from research.weights.concurrency import avg_uniqueness, num_co_events
from research.weights.config import WeightConfig
from research.weights.decay import time_decay

__all__ = ["build_sample_weights"]


def build_sample_weights(
    labels: pd.DataFrame, close: pd.Series, config: WeightConfig
) -> pd.DataFrame:
    """Build the AFML Ch.4 sample-weight frame from a resolved labels set.

    Parameters
    ----------
    labels:
        A get_bins-style labels frame, indexed by event-start (``t0``)
        timestamps, with at least a ``t1`` column (event end timestamps).
        Any other columns (``ret``, ``bin``, ``barrier``, ``side``) are
        ignored -- weight computation depends only on ``t1`` and ``close``.
    close:
        Bar close prices, UTC ``DatetimeIndex``, ascending.
    config:
        The :class:`WeightConfig` recipe (``decay_c``) for this build.

    Returns
    -------
    pd.DataFrame
        Indexed by ``t0``, with exactly three columns:
          * ``avg_uniqueness`` -- per-event average uniqueness.
          * ``weight`` -- ``normalize_weights(return_attribution) x
            time_decay(avg_uniqueness)``, the final sample weight.
          * ``t1`` -- carried through unchanged from ``labels``.

    Raises
    ------
    ValueError
        If ``labels["t1"]`` contains any null values (propagated from
        :func:`research.weights.concurrency.num_co_events`'s fail-loud
        contract).
    """
    t1 = labels["t1"]

    co_events = num_co_events(close.index, t1)
    tw = avg_uniqueness(t1, co_events)
    attribution = return_attribution(close, t1, co_events)
    weight = normalize_weights(attribution) * time_decay(tw, config.decay_c)

    return pd.DataFrame({"avg_uniqueness": tw, "weight": weight, "t1": t1})
