"""CombinatorialPurgedKFold -- combinatorial purged cross-validation (AFML
Ch.12, Step 4).

The second thin adapter over the Step 2 kernel (Q6): contains no purge or
embargo logic of its own, only group partitioning, combination
enumeration, and the positional <-> timestamp translation
``_purge_embargo`` needs.

Subclasses ``_BaseKFold``, parallel to ``PurgedKFold`` -- a design choice
made here, not explicitly locked verbatim at Q6 (which fixed the
test-set/embargo semantics, not the class hierarchy), but a natural,
low-risk extension of it. ``n_splits = C(n_groups, n_test_groups)``
depends only on those two integers -- never on ``t1`` -- so unlike
``t1``-validation (deferred to ``split()``, mirroring ``PurgedKFold``)
there is nothing to defer here: it is computed eagerly at construction
and handed to ``super().__init__()``, giving ``get_n_splits()`` for free
via the same inherited mechanism ``PurgedKFold`` already uses.

Per Q6's lock: samples are partitioned into ``n_groups`` contiguous
blocks; every ``C(n_groups, n_test_groups)`` combination of blocks is
enumerated (canonical ``itertools.combinations`` order); each
combination's test set is the UNION of its k blocks (possibly
non-contiguous); and each block's embargo is computed and applied
independently against the FULL bar grid before a single
``_purge_embargo`` call purges train around all k embargoed blocks at
once. Path-reconstruction (the `phi` backtest-path matrix) is deferred to
Phase 2 -- this module only yields the combinatorial splits.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold

from research.validation.purge import _purge_embargo

__all__ = ["CombinatorialPurgedKFold"]


class CombinatorialPurgedKFold(_BaseKFold):
    """Combinatorial purged K-fold cross-validation (AFML Ch.12).

    Partitions samples into ``n_groups`` contiguous blocks and yields one
    fold per combination of ``n_test_groups`` blocks taken as the test
    set, purging and embargoing train around the union of those blocks.
    Like ``PurgedKFold``, validation (``t1`` set, alignment, monotonic
    index) happens lazily inside ``split()``, keeping the constructor a
    pure assignment for sklearn's ``clone()``/``get_params()`` machinery.

    Parameters
    ----------
    n_groups:
        Number of contiguous partitions of the sample range.
    n_test_groups:
        Number of groups (k) taken as the test set per combination.
        Defaults to ``2`` (AFML's canonical CPCV example).
    t1:
        Event end timestamps, indexed by event-start (``t0``) timestamps.
        Required before ``split()`` is called; stashed as-is at
        construction (no validation here -- see class docstring).
    embargo_pct:
        Fraction of the total sample count to embargo forward of EACH
        selected block independently. ``0.0`` disables embargo
        (purge-only).
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        t1: pd.Series = None,
        embargo_pct: float = 0.0,
    ):
        n_splits = math.comb(n_groups, n_test_groups)
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.t1 = t1
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        """Yield ``(train_positions, test_positions)`` for each combination.

        Validates, in order: ``self.t1`` was set; ``X.index`` exactly
        equals ``self.t1.index``; ``X.index`` is strictly monotonic
        increasing -- identical fail-loud posture to ``PurgedKFold``.

        Samples are partitioned into ``n_groups`` contiguous blocks via
        ``np.array_split``. For each of the ``C(n_groups, n_test_groups)``
        combinations: the test set is the union of the selected blocks'
        positions; each selected block contributes one row to a
        ``test_times`` Series (``block_start -> block_max_t1``, the same
        per-block aggregation ``PurgedKFold`` uses for its single fold);
        ``_purge_embargo`` is called ONCE per combination against the
        full multi-row ``test_times``, embargoing and purging around
        every selected block independently in a single pass.

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            ``(train_positions, test_positions)`` -- integer position
            arrays suitable for ``.iloc``/fancy indexing into ``X``.

        Raises
        ------
        ValueError
            If ``self.t1`` is ``None``, if ``X.index`` does not exactly
            equal ``self.t1.index``, or if ``X.index`` is not strictly
            monotonic increasing.
        """
        if self.t1 is None:
            raise ValueError(
                "CombinatorialPurgedKFold requires t1 to be set before "
                "calling split()."
            )
        if not X.index.equals(self.t1.index):
            raise ValueError(
                "X.index and t1.index must be identical -- "
                "CombinatorialPurgedKFold cannot purge correctly against "
                "a misaligned t1."
            )
        if not X.index.is_monotonic_increasing:
            raise ValueError(
                "X.index (and t1.index) must be strictly monotonic "
                "increasing -- purging by position only matches purging "
                "by time on an ascending grid."
            )

        indices = np.arange(X.shape[0])
        group_positions = np.array_split(indices, self.n_groups)

        for combo in combinations(range(self.n_groups), self.n_test_groups):
            selected_groups = [group_positions[g] for g in combo]
            test_pos = np.concatenate(selected_groups)

            blocks = {}
            for grp in selected_groups:
                block_start = self.t1.index[grp[0]]
                block_max_t1 = self.t1.iloc[grp].max()
                blocks[block_start] = block_max_t1
            test_times = pd.Series(blocks)

            train_t0_index = _purge_embargo(
                self.t1, test_times, self.t1.index, self.embargo_pct
            )
            train_pos = self.t1.index.get_indexer(train_t0_index)

            yield train_pos, test_pos
