"""backtest.paths -- CPCV phi-path reconstruction (Phase 2, Task 3).

Estimator-free, pure combinatorial bookkeeping: given the per-split,
per-test-group bet-return series a caller already computed (typically
`engine.py` driving `CombinatorialPurgedKFold` + `_fit_predict_returns`),
reassembles them into `phi` full-length, dense, NaN-free out-of-sample
paths -- the `paths: pd.DataFrame` single cross-module currency.

The assignment rule (locked): "j-th test-occurrence of group g -> path
j". Group `g` is a test group in exactly `phi = C(n_groups-1,
n_test_groups-1)` of the `C(n_groups, n_test_groups)` combinatorial
splits (canonical `itertools.combinations(range(n_groups),
n_test_groups)` order -- the same order `CombinatorialPurgedKFold.split()`
enumerates, guarded here by a partition-equivalence test rather than any
import coupling). Assigning each group's occurrences to path indices
0..phi-1, in split-index order, and concatenating one occurrence per
group per path produces a path that touches every group's block exactly
once -- dense and NaN-free by construction, since the groups partition
the full event timeline.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

__all__ = ["reconstruct_paths"]


def reconstruct_paths(
    per_split_group_returns: dict, n_groups: int, n_test_groups: int
) -> pd.DataFrame:
    """Reassemble per-split group returns into `phi` dense OOS paths.

    Parameters
    ----------
    per_split_group_returns:
        ``{split_idx: {group_id: pd.Series}}``. ``split_idx`` runs
        ``0..C(n_groups, n_test_groups)-1`` in the same canonical
        ``itertools.combinations`` order the splitter enumerates.
        ``group_id`` is one of that split's ``n_test_groups`` selected
        groups; the associated ``pd.Series`` is the bet-return series for
        that group's positions under that split's fitted model.
    n_groups:
        Total group count (``N``).
    n_test_groups:
        Test-group count per combination (``k``).

    Returns
    -------
    pd.DataFrame
        Dense, NaN-free -- index is the full OOS ``t0`` timeline (the
        union of all groups' event timestamps), columns are ``path_id``
        ``0..phi-1`` where ``phi = C(n_groups-1, n_test_groups-1)``.

    Raises
    ------
    ValueError
        If ``n_groups < 2``, ``n_test_groups < 1``, or
        ``n_test_groups >= n_groups``; if ``per_split_group_returns``
        does not have exactly ``C(n_groups, n_test_groups)`` entries; or
        if any split's group-key set does not exactly match that split's
        combination.
    """
    if n_groups < 2:
        raise ValueError(
            f"reconstruct_paths: n_groups must be >= 2, got {n_groups!r}."
        )
    if n_test_groups < 1:
        raise ValueError(
            f"reconstruct_paths: n_test_groups must be >= 1, got "
            f"{n_test_groups!r}."
        )
    if n_test_groups >= n_groups:
        raise ValueError(
            f"reconstruct_paths: n_test_groups must be < n_groups, got "
            f"n_test_groups={n_test_groups!r}, n_groups={n_groups!r}."
        )

    combos = list(itertools.combinations(range(n_groups), n_test_groups))
    n_splits = len(combos)
    if len(per_split_group_returns) != n_splits:
        raise ValueError(
            f"reconstruct_paths: expected {n_splits} splits "
            f"(C({n_groups},{n_test_groups})), got "
            f"{len(per_split_group_returns)!r}."
        )
    for split_idx, combo in enumerate(combos):
        if split_idx not in per_split_group_returns:
            raise ValueError(
                f"reconstruct_paths: missing split_idx={split_idx!r} in "
                f"per_split_group_returns."
            )
        got_groups = set(per_split_group_returns[split_idx].keys())
        expected_groups = set(combo)
        if got_groups != expected_groups:
            raise ValueError(
                f"reconstruct_paths: split_idx={split_idx!r} has group "
                f"keys {sorted(got_groups)!r}, expected "
                f"{sorted(expected_groups)!r}."
            )

    phi = math.comb(n_groups - 1, n_test_groups - 1)

    group_occurrences: dict = {g: [] for g in range(n_groups)}
    for split_idx, combo in enumerate(combos):
        for g in combo:
            group_occurrences[g].append(split_idx)

    path_pieces: dict = {j: [] for j in range(phi)}
    for g in range(n_groups):
        occurrences = group_occurrences[g]
        for j, split_idx in enumerate(occurrences):
            path_pieces[j].append(per_split_group_returns[split_idx][g])

    columns = {}
    for j in range(phi):
        combined = pd.concat(path_pieces[j]).sort_index()
        columns[j] = combined

    paths = pd.DataFrame(columns)
    return paths
