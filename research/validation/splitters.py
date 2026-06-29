"""PurgedKFold -- leak-proof cross-validation splitter (AFML Snippet 7.3).

A thin positional adapter over the Step 2 kernel (Q3/Q4/Q6): this module
contains no purge or embargo logic of its own. It only does two things a
purge/embargo kernel can't do on its own -- enumerate contiguous,
non-shuffled test folds over the sample order, and translate between
sklearn's positional split protocol and the kernel's timestamp-space
contract.

``t1`` is stashed on the instance at construction (not passed to
``split()``), a deliberate break from sklearn's stateless-splitter
convention -- AFML's own design, and the only place ``t1`` *can* go if
``split(X, y=None, groups=None)`` is to stay sklearn-signature-compatible
(Q3).

Per Q4/Q6's composition lock: each fold collapses to a single
``(block_start, block_max_t1)`` span -- index = the fold's first test
event's ``t0``, value = the *max* ``t1`` among the fold's test events --
and ``_purge_embargo`` does the actual purge+embargo work. Passing the
FULL ``t1`` (including the fold's own test rows) into the kernel is
deliberate, not an oversight: every test event's own ``t0`` necessarily
falls inside its own fold's span, so the kernel's "starts inside" overlap
condition naturally excludes the test fold's own rows from the surviving
train set -- no separate "exclude test positions" step is needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold

from research.validation.purge import _purge_embargo

__all__ = ["PurgedKFold"]


class PurgedKFold(_BaseKFold):
    """Contiguous K-fold cross-validation with purging and embargo.

    Unlike sklearn's stock ``KFold``, test folds are never shuffled
    (shuffling is incompatible with purge, which depends on test being a
    contiguous, chronologically identifiable block -- Q3) and the
    constructor's ``t1`` is validated against ``split()``'s ``X`` lazily,
    inside ``split()`` itself, not in ``__init__`` -- keeping the
    constructor a pure assignment so sklearn's ``clone()``/``get_params()``
    machinery continues to work, the same posture AFML's own class takes.

    Parameters
    ----------
    n_splits:
        Number of contiguous folds.
    t1:
        Event end timestamps, indexed by event-start (``t0``) timestamps.
        Required before ``split()`` is called; stashed as-is at
        construction (no validation here -- see class docstring).
    embargo_pct:
        Fraction of the total sample count to embargo forward of each
        test fold. ``0.0`` disables embargo (purge-only).
    """

    def __init__(
        self,
        n_splits: int = 3,
        t1: pd.Series = None,
        embargo_pct: float = 0.0,
    ):
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        """Yield ``(train_positions, test_positions)`` for each fold.

        Validates, in order:
          1. ``self.t1`` was set at construction (fail-loud if ``None``).
          2. ``X.index`` and ``self.t1.index`` are identical (fail-loud
             alignment assert -- the load-bearing leak guard; a silently
             misaligned ``t1`` would purge the wrong rows undetectably).
          3. ``X.index`` is strictly monotonic increasing (purging by
             position only matches purging by time if the grid is
             ascending).

        Test folds are contiguous positional chunks of
        ``np.arange(X.shape[0])`` via ``np.array_split`` -- never
        shuffled. For each fold, the test span collapses to one
        ``(block_start, block_max_t1)`` row and ``_purge_embargo`` is
        called once against the FULL ``t1`` (see module docstring for why
        that's sufficient to exclude the fold's own rows from train too).

        Parameters
        ----------
        X:
            Feature frame/series whose index must exactly equal
            ``self.t1.index``.
        y, groups:
            Unused -- present only for sklearn's ``split()`` signature
            compatibility.

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
                "PurgedKFold requires t1 to be set before calling split()."
            )
        if not X.index.equals(self.t1.index):
            raise ValueError(
                "X.index and t1.index must be identical -- PurgedKFold "
                "cannot purge correctly against a misaligned t1."
            )
        if not X.index.is_monotonic_increasing:
            raise ValueError(
                "X.index (and t1.index) must be strictly monotonic "
                "increasing -- purging by position only matches purging "
                "by time on an ascending grid."
            )

        indices = np.arange(X.shape[0])
        for test_pos in np.array_split(indices, self.n_splits):
            test_t0 = self.t1.index[test_pos[0]]
            test_max_t1 = self.t1.iloc[test_pos].max()
            block = pd.Series({test_t0: test_max_t1})

            train_t0_index = _purge_embargo(
                self.t1, block, self.t1.index, self.embargo_pct
            )
            train_pos = self.t1.index.get_indexer(train_t0_index)

            yield train_pos, test_pos
