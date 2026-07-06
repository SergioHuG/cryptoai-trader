"""backtest.ledger -- pure trial-record model and reductions (Phase 2,
Task 5).

A trial is one real-data backtest (WF or CPCV) of a distinct
:class:`~backtest.config.BacktestConfig` -- synthetic runs never log
here (Q11). This module owns the record shape and the pure statistical
reductions over a list of records (``n_trials``, ``sr_variance``,
``assemble_matrix``); ``storage.py`` owns the HDF5 append/load
mechanics. Nothing here touches disk.

Dedup is by ``config_hash``: if the ledger holds more than one record
for the same config (a deliberate re-run), the most recent
(``created_at``) survives and the rest are dropped before any reduction
runs -- ``N`` counts distinct configs, never raw record count.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["TrialRecord", "n_trials", "sr_variance", "assemble_matrix"]

_VALID_MODES = frozenset({"wf", "cpcv"})


@dataclass(frozen=True)
class TrialRecord:
    """One real-data backtest trial of a distinct config.

    Attributes:
        trial_id:    Unique identifier for this trial run (non-empty str).
        config_hash: The :class:`BacktestConfig` recipe hash this trial
                     ran -- the dedup key.
        created_at:  Timestamp the trial was recorded (coerced to
                     ``pd.Timestamp``); breaks ties on dedup (latest wins).
        mode:        ``"wf"`` or ``"cpcv"`` -- synthetic runs never
                     produce a ``TrialRecord`` at all.
        sharpe:      The trial's recorded Sharpe (WF: single-path Sharpe;
                     CPCV: median-path Sharpe per the gate composition
                     lock) -- must be finite.
        n_bets:      Number of bets in ``returns`` (>= 1); must equal
                     ``len(returns)``.
        returns:     OOS bet-return series indexed by event ``t0`` (WF:
                     the single path; CPCV: per-event mean across the phi
                     paths). Non-empty.
    """

    trial_id: str
    config_hash: str
    created_at: pd.Timestamp
    mode: str
    sharpe: float
    n_bets: int
    returns: pd.Series

    def __post_init__(self) -> None:
        if not self.trial_id:
            raise ValueError("TrialRecord.trial_id must be non-empty.")
        if not self.config_hash:
            raise ValueError("TrialRecord.config_hash must be non-empty.")
        object.__setattr__(self, "created_at", pd.Timestamp(self.created_at))
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"TrialRecord.mode must be one of {sorted(_VALID_MODES)!r}, "
                f"got {self.mode!r}."
            )
        if not np.isfinite(self.sharpe):
            raise ValueError(
                f"TrialRecord.sharpe must be finite, got {self.sharpe!r}."
            )
        if self.n_bets < 1:
            raise ValueError(
                f"TrialRecord.n_bets must be >= 1, got {self.n_bets!r}."
            )
        if len(self.returns) == 0:
            raise ValueError("TrialRecord.returns must be non-empty.")
        if len(self.returns) != self.n_bets:
            raise ValueError(
                f"TrialRecord: len(returns)={len(self.returns)!r} must "
                f"equal n_bets={self.n_bets!r}."
            )


def _dedup_by_config_hash(records: list) -> list:
    """Drop all but the most-recent (``created_at``) record per
    ``config_hash``. Order of the returned list is not significant --
    callers reduce over it, never index positionally.

    Raises
    ------
    ValueError
        If ``records`` is empty.
    """
    if not records:
        raise ValueError("ledger: records must be non-empty.")
    by_hash: dict = {}
    for record in records:
        existing = by_hash.get(record.config_hash)
        if existing is None or record.created_at > existing.created_at:
            by_hash[record.config_hash] = record
    return list(by_hash.values())


def n_trials(records: list) -> int:
    """Distinct trial-config count (``N``) -- records are deduped by
    ``config_hash`` before counting.

    Raises
    ------
    ValueError
        If ``records`` is empty.
    """
    return len(_dedup_by_config_hash(records))


def sr_variance(records: list) -> float:
    """Sample variance (``ddof=1``) of trial Sharpes across distinct
    configs (``V``) -- the DSR multiple-testing correction's variance
    term, fed into :func:`backtest.metrics.expected_max_sharpe`.

    Raises
    ------
    ValueError
        If ``records`` is empty, or fewer than 2 distinct configs remain
        after dedup (a sample variance needs at least 2 points).
    """
    deduped = _dedup_by_config_hash(records)
    if len(deduped) < 2:
        raise ValueError(
            f"sr_variance: need >= 2 distinct trial configs, got "
            f"{len(deduped)!r}."
        )
    sharpes = np.array([r.sharpe for r in deduped], dtype=float)
    return float(np.var(sharpes, ddof=1))


def assemble_matrix(records: list, min_overlap: int) -> pd.DataFrame:
    """Assemble the ``(T, N)`` PBO trial matrix: one column per distinct
    trial config (deduped, keyed by ``config_hash``), inner-joined on
    their common ``t0`` index.

    Parameters
    ----------
    records:
        Trial records (deduped internally by ``config_hash``).
    min_overlap:
        The minimum required post-join row count (``T``) -- callers pass
        the CSCV partition count ``S`` here, since :func:`backtest.metrics.pbo`
        itself requires ``T >= S``.

    Returns
    -------
    pd.DataFrame
        Shape ``(T, N)``, columns named by ``config_hash``, index the
        inner-joined ``t0`` timeline.

    Raises
    ------
    ValueError
        If ``records`` is empty, fewer than 2 distinct configs remain
        after dedup, or the inner-joined overlap is smaller than
        ``min_overlap``.
    """
    deduped = _dedup_by_config_hash(records)
    if len(deduped) < 2:
        raise ValueError(
            f"assemble_matrix: need >= 2 distinct trial configs, got "
            f"{len(deduped)!r}."
        )
    series_by_hash = {r.config_hash: r.returns for r in deduped}
    matrix = pd.concat(series_by_hash, axis=1, join="inner")
    if len(matrix) < min_overlap:
        raise ValueError(
            f"assemble_matrix: inner-joined overlap ({len(matrix)!r} rows) "
            f"is below the required minimum ({min_overlap!r})."
        )
    return matrix
