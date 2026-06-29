"""Acceptance tests for research/validation/cpcv.py -- CombinatorialPurgedKFold
(AFML Ch.12, Step 4).

The second thin adapter over the Step 2c kernel (Q6) -- this file does NOT
re-derive purge+embargo correctness (Step 2c already proved that); it
proves the combinatorial enumeration and per-block union wiring are
correct, plus a spy test confirming the kernel is actually REUSED, not
reimplemented (mirroring the weights single-shared-intermediate
discipline).

Design choice flagged explicitly (not locked verbatim at Q6, but a
natural extension of it): CombinatorialPurgedKFold also subclasses
_BaseKFold, parallel to PurgedKFold -- n_splits = C(n_groups,
n_test_groups) is computed eagerly at construction (it depends only on
n_groups/n_test_groups, never on t1, so unlike t1-validation there is
nothing to defer here) and passed to super().__init__, giving
get_n_splits() for free via the same inherited mechanism PurgedKFold uses.

Shared fixture: 12 zero-width events (t0 == t1) on an hourly grid,
n_groups=4 -> four contiguous blocks of 3 positions each: G0=[0,1,2],
G1=[3,4,5], G2=[6,7,8], G3=[9,10,11]. n_test_groups=2 -> C(4,2)=6
combinations in canonical itertools.combinations order:
(0,1),(0,2),(0,3),(1,2),(1,3),(2,3).

Every exact train_pos value below -- especially the embargo case, where
embargo from a selected group bleeds into an UNSELECTED group sitting in
the gap between two test blocks -- was independently verified against a
working draft implementation before being locked into these assertions.
"""
import math
from itertools import combinations as iter_combinations
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection._split import _BaseKFold

import research.validation.cpcv as cpcv_module
from research.validation.cpcv import CombinatorialPurgedKFold

_CBARS = pd.date_range("2024-01-01", periods=12, freq="1h", tz="UTC")
_GROUPS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]  # G0..G3


def _zero_width_t1() -> pd.Series:
    return pd.Series({b: b for b in _CBARS})


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame({"x": np.arange(len(_CBARS))}, index=_CBARS)


class TestCombinatorialPurgedKFoldValidation:
    def test_raises_if_t1_never_set(self):
        cpkf = CombinatorialPurgedKFold(n_groups=4, n_test_groups=2, t1=None)
        with pytest.raises(ValueError):
            list(cpkf.split(_feature_frame()))

    def test_alignment_mismatch_raises(self):
        t1 = _zero_width_t1()
        X = _feature_frame().iloc[:10]  # 10 rows vs t1's 12
        cpkf = CombinatorialPurgedKFold(n_groups=4, n_test_groups=2, t1=t1)
        with pytest.raises(ValueError):
            list(cpkf.split(X))

    def test_non_monotonic_index_raises(self):
        shuffled = _CBARS[[0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        t1 = _zero_width_t1().loc[shuffled]
        X = _feature_frame().loc[shuffled]
        cpkf = CombinatorialPurgedKFold(n_groups=4, n_test_groups=2, t1=t1)
        with pytest.raises(ValueError):
            list(cpkf.split(X))


class TestCombinatorialPurgedKFoldEnumeration:
    def test_get_n_splits_equals_n_choose_k(self):
        cpkf = CombinatorialPurgedKFold(
            n_groups=4, n_test_groups=2, t1=_zero_width_t1(), embargo_pct=0.0
        )
        assert cpkf.get_n_splits() == math.comb(4, 2) == 6

    def test_n_test_groups_defaults_to_two(self):
        cpkf = CombinatorialPurgedKFold(n_groups=4, t1=_zero_width_t1())
        assert cpkf.get_n_splits() == math.comb(4, 2) == 6

    def test_yields_exactly_n_choose_k_folds(self):
        cpkf = CombinatorialPurgedKFold(
            n_groups=4, n_test_groups=2, t1=_zero_width_t1(), embargo_pct=0.0
        )
        folds = list(cpkf.split(_feature_frame()))
        assert len(folds) == 6

    def test_combinations_enumerated_in_canonical_order(self):
        """Each combination's test set is exactly the union of its k
        group blocks, in itertools.combinations(range(4), 2) order."""
        cpkf = CombinatorialPurgedKFold(
            n_groups=4, n_test_groups=2, t1=_zero_width_t1(), embargo_pct=0.0
        )
        folds = list(cpkf.split(_feature_frame()))
        expected_combos = list(iter_combinations(range(4), 2))

        assert len(folds) == len(expected_combos)
        for (_, test_pos), combo in zip(folds, expected_combos):
            expected_test = sorted(
                pos for g in combo for pos in _GROUPS[g]
            )
            assert list(test_pos) == expected_test

    def test_k_equals_one_yields_one_combo_per_group(self):
        """n_test_groups=1 -> C(4,1)=4 combos, each test set is exactly
        one group's positions, no union needed."""
        cpkf = CombinatorialPurgedKFold(
            n_groups=4, n_test_groups=1, t1=_zero_width_t1(), embargo_pct=0.0
        )
        folds = list(cpkf.split(_feature_frame()))
        assert len(folds) == 4
        for (_, test_pos), group in zip(folds, _GROUPS):
            assert list(test_pos) == group


class TestCombinatorialPurgedKFoldNoEmbargo:
    """embargo_pct == 0.0 -- train is exactly the complement of test for
    every combination (pure purge, only self-overlapping test rows
    dropped -- same invariant PurgedKFold's no-embargo case satisfies)."""

    def test_train_is_exact_complement_of_test_for_every_combo(self):
        cpkf = CombinatorialPurgedKFold(
            n_groups=4, n_test_groups=2, t1=_zero_width_t1(), embargo_pct=0.0
        )
        all_positions = set(range(12))
        for train_pos, test_pos in cpkf.split(_feature_frame()):
            expected_train = sorted(all_positions - set(test_pos))
            assert list(train_pos) == expected_train

    def test_train_never_overlaps_test(self):
        cpkf = CombinatorialPurgedKFold(
            n_groups=4, n_test_groups=2, t1=_zero_width_t1(), embargo_pct=0.0
        )
        for train_pos, test_pos in cpkf.split(_feature_frame()):
            assert set(train_pos).isdisjoint(set(test_pos))


class TestCombinatorialPurgedKFoldWithEmbargo:
    """embargo_pct == 0.2 on n=12 -> step=2, per-block forward embargo
    (Q6). Verified against a working draft before being locked in -- see
    module docstring."""

    def test_embargo_bleeds_into_unselected_gap_group_and_beyond_last_block(
        self,
    ):
        """Combo (0,2): test = G0 + G2 = [0,1,2,6,7,8] -- a genuinely
        NON-contiguous union with G1=[3,4,5] sitting unselected in the
        gap, and G3=[9,10,11] sitting after the last test block.

        G0's embargo (extends to bar 4) catches positions 3,4 from the
        UNSELECTED gap group G1 -- but not position 5, which survives.
        G2's embargo (extends to bar 10) catches positions 9,10 from G3 --
        but not position 11, which survives.

        Expected survivors: exactly {5, 11}.
        """
        cpkf = CombinatorialPurgedKFold(
            n_groups=4, n_test_groups=2, t1=_zero_width_t1(), embargo_pct=0.2
        )
        folds = list(cpkf.split(_feature_frame()))
        # combo (0,2) is the 2nd yielded combination (index 1) in
        # canonical order: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        train_pos, test_pos = folds[1]
        assert list(test_pos) == [0, 1, 2, 6, 7, 8]
        assert list(train_pos) == [5, 11]

    def test_train_never_overlaps_test_with_embargo_active(self):
        cpkf = CombinatorialPurgedKFold(
            n_groups=4, n_test_groups=2, t1=_zero_width_t1(), embargo_pct=0.2
        )
        for train_pos, test_pos in cpkf.split(_feature_frame()):
            assert set(train_pos).isdisjoint(set(test_pos))


class TestCombinatorialPurgedKFoldKernelReuse:
    """Spy test mirroring the weights single-shared-intermediate
    discipline: confirms _purge_embargo is the function actually called
    -- once per combination -- not a parallel reimplementation."""

    def test_calls_purge_embargo_kernel_once_per_combination(self):
        real_kernel = cpcv_module._purge_embargo
        with patch.object(
            cpcv_module, "_purge_embargo", wraps=real_kernel
        ) as spy:
            cpkf = CombinatorialPurgedKFold(
                n_groups=4,
                n_test_groups=2,
                t1=_zero_width_t1(),
                embargo_pct=0.0,
            )
            list(cpkf.split(_feature_frame()))
        assert spy.call_count == 6  # C(4,2)


class TestCombinatorialPurgedKFoldSklearnCompatibility:
    def test_is_a_base_kfold_subclass(self):
        cpkf = CombinatorialPurgedKFold(
            n_groups=4, n_test_groups=2, t1=_zero_width_t1(), embargo_pct=0.0
        )
        assert isinstance(cpkf, _BaseKFold)

    def test_consumable_via_plain_for_loop(self):
        cpkf = CombinatorialPurgedKFold(
            n_groups=4, n_test_groups=2, t1=_zero_width_t1(), embargo_pct=0.0
        )
        count = 0
        for train_pos, test_pos in cpkf.split(_feature_frame()):
            count += 1
        assert count == 6

    def test_yields_integer_position_arrays(self):
        cpkf = CombinatorialPurgedKFold(
            n_groups=4, n_test_groups=2, t1=_zero_width_t1(), embargo_pct=0.0
        )
        train_pos, test_pos = next(cpkf.split(_feature_frame()))
        assert isinstance(train_pos, np.ndarray)
        assert isinstance(test_pos, np.ndarray)
        assert np.issubdtype(train_pos.dtype, np.integer)
        assert np.issubdtype(test_pos.dtype, np.integer)
