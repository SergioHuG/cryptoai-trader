"""Standalone sequential bootstrap (AFML Ch.4.3-4.5).

Orthogonal to config/storage/pipeline (Q1) -- never wired into
WeightConfig or build_sample_weights. Re-rolled at training time, not a
stored artifact.

get_ind_matrix follows AFML's getIndMatrix verbatim: dense (bars x
events), INTEGER positional columns (0..n_events-1), not t0 timestamps --
this sidesteps duplicate-column ambiguity when multiple events share a
start bar, and matches AFML's own column convention exactly.

ind_matrix_avg_uniqueness (AFML's getAvgUniqueness) is a genuinely
different computation path from
research.weights.concurrency.avg_uniqueness: seq_bootstrap must recompute
average uniqueness over HYPOTHETICAL candidate subsets that have no
precomputed co_events series, so it needs the matrix form. The two paths
are proven to agree exactly on a shared fixture in
tests/research/weights/test_bootstrap.py::TestIndMatrixAvgUniquenessMatchesConcurrencyOracle
(Q10c) -- they can never silently diverge.

seq_bootstrap takes an injectable seeded rng (Q10b) -- this is the seam
the parked "fixed bootstrapped index for exact replay" feature
(-> model/training branch) will eventually record a seed against.

Dense-faithful first cut: this matrix is O(events * bars) in memory and
seq_bootstrap's per-draw avgU recompute is O(s_length * n_events) calls to
ind_matrix_avg_uniqueness, each O(bars * subset_size). A sparse/interval
representation is a parked future optimization (load profile here is
compute-bound at AFML's intended research-subsample scale, not
memory-bound, so dense-faithful is the right first cut).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["get_ind_matrix", "ind_matrix_avg_uniqueness", "seq_bootstrap"]


def get_ind_matrix(bar_index: pd.DatetimeIndex, t1: pd.Series) -> pd.DataFrame:
    """Dense event-indicator matrix (AFML Snippet 4.3, ``getIndMatrix``).

    Parameters
    ----------
    bar_index:
        The full bar grid (UTC ``DatetimeIndex``).
    t1:
        Event end timestamps, indexed by event-start (``t0``) timestamps.

    Returns
    -------
    pd.DataFrame
        Shape ``(len(bar_index), len(t1))``, indexed by ``bar_index``,
        with INTEGER positional columns ``0..len(t1)-1`` (one per event, in
        ``t1``'s iteration order). Entry is ``1`` where the column's event
        spans that bar (inclusive of both ``t0`` and ``t1``), ``0``
        elsewhere.
    """
    n_events = len(t1)
    ind_matrix = pd.DataFrame(
        0, index=bar_index, columns=range(n_events), dtype="int64"
    )
    for i, (t0, t1v) in enumerate(t1.items()):
        ind_matrix.loc[t0:t1v, i] = 1
    return ind_matrix


def ind_matrix_avg_uniqueness(ind_matrix: pd.DataFrame) -> pd.Series:
    """Per-column average uniqueness from a dense indicator matrix.

    AFML Snippet 4.4 (``getAvgUniqueness``): row-sums give per-bar
    concurrency across every column currently in ``ind_matrix``; each
    column's average uniqueness is the mean of ``1 / concurrency`` over
    just that column's own active bars.

    Parameters
    ----------
    ind_matrix:
        Dense indicator matrix, typically :func:`get_ind_matrix` output
        (or a column subset of it, as used internally by
        :func:`seq_bootstrap`).

    Returns
    -------
    pd.Series
        Named ``avg_uniqueness``, indexed by ``ind_matrix.columns``.
    """
    concurrency_counts = ind_matrix.sum(axis=1)
    uniqueness = ind_matrix.div(concurrency_counts, axis=0)
    avg_u = uniqueness[uniqueness > 0].mean()
    avg_u.name = "avg_uniqueness"
    return avg_u


def seq_bootstrap(
    ind_matrix: pd.DataFrame,
    s_length: int | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> list:
    """Sequential bootstrap draw (AFML Snippet 4.5, ``seqBootstrap``).

    Draws ``s_length`` event indices with replacement. Each draw's
    probability is proportional to the candidate's average uniqueness
    *given* everything already drawn -- so events that heavily overlap
    with the already-drawn pool become progressively less likely to be
    drawn again, favoring diverse (low-overlap) samples over uniform IID
    sampling.

    Parameters
    ----------
    ind_matrix:
        Dense indicator matrix, typically :func:`get_ind_matrix` output.
    s_length:
        Number of draws. Defaults to ``ind_matrix.shape[1]`` (AFML's
        default -- one draw per event).
    rng:
        Seeded ``numpy.random.Generator`` for reproducibility. Defaults to
        a fresh ``numpy.random.default_rng()`` if not provided.

    Returns
    -------
    list
        ``s_length`` column indices (ints), drawn with replacement.
        Empty if ``ind_matrix`` has zero columns.
    """
    if rng is None:
        rng = np.random.default_rng()
    if s_length is None:
        s_length = ind_matrix.shape[1]

    columns = list(ind_matrix.columns)
    phi: list = []

    while len(phi) < s_length:
        avg_u = pd.Series(index=columns, dtype="float64")
        for col in columns:
            subset = ind_matrix[phi + [col]]
            avg_u.loc[col] = ind_matrix_avg_uniqueness(subset).iloc[-1]
        prob = avg_u / avg_u.sum()
        draw = rng.choice(columns, p=prob.to_numpy())
        phi.append(int(draw))

    return phi
