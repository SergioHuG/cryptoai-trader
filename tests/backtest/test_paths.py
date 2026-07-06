"""Acceptance tests for backtest.paths (Phase 2, Task 3).

Hand-built (N=4,k=2 -> phi=3) and (N=6,k=2 -> phi=5) cases assert the
exact reconstructed matrix. A partition-equivalence test guards the
assumed coupling to CombinatorialPurgedKFold's actual split() order and
group partitioning -- without importing any private internals, purely by
comparing observed behavior.
"""
import itertools
import math

import numpy as np
import pandas as pd
import pytest

from backtest.paths import reconstruct_paths
from research.validation.cpcv import CombinatorialPurgedKFold


def _build_per_split(n_groups, n_test_groups, group_dates, value_fn):
    """Hand-build a {split_idx: {group_id: pd.Series}} dict using the same
    canonical combinations order the splitter is documented to use."""
    combos = list(itertools.combinations(range(n_groups), n_test_groups))
    per_split = {}
    for split_idx, combo in enumerate(combos):
        per_split[split_idx] = {}
        for g in combo:
            dates = group_dates[g]
            vals = [value_fn(split_idx, g)] * len(dates)
            per_split[split_idx][g] = pd.Series(vals, index=dates)
    return per_split


class TestReconstructPathsExactMatrix:
    def test_n4_k2_phi3_dense_and_exact(self):
        n_groups, n_test_groups = 4, 2
        dates = pd.date_range("2026-01-01", periods=8, freq="D")
        group_dates = {g: dates[2 * g : 2 * g + 2] for g in range(n_groups)}
        per_split = _build_per_split(
            n_groups, n_test_groups, group_dates,
            value_fn=lambda split_idx, g: split_idx * 10 + g,
        )

        paths = reconstruct_paths(per_split, n_groups, n_test_groups)

        assert paths.shape == (8, 3)
        assert not paths.isna().any().any()
        assert list(paths.columns) == [0, 1, 2]

        # Hand-derived expected values (verified against the combo
        # enumeration [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]):
        # path 0: g0<-split0(val0), g1<-split0(val1), g2<-split1(val12), g3<-split2(val23)
        # path 1: g0<-split1(val10), g1<-split3(val31), g2<-split3(val32), g3<-split4(val43)
        # path 2: g0<-split2(val20), g1<-split4(val41), g2<-split5(val52), g3<-split5(val53)
        expected_col0 = [0, 0, 1, 1, 12, 12, 23, 23]
        expected_col1 = [10, 10, 31, 31, 32, 32, 43, 43]
        expected_col2 = [20, 20, 41, 41, 52, 52, 53, 53]
        np.testing.assert_array_equal(paths[0].to_numpy(), expected_col0)
        np.testing.assert_array_equal(paths[1].to_numpy(), expected_col1)
        np.testing.assert_array_equal(paths[2].to_numpy(), expected_col2)

    def test_n6_k2_phi5_dense_no_nan(self):
        n_groups, n_test_groups = 6, 2
        dates = pd.date_range("2026-02-01", periods=12, freq="D")
        group_dates = {
            g: dates[2 * g : 2 * g + 2] for g in range(n_groups)
        }
        per_split = _build_per_split(
            n_groups, n_test_groups, group_dates,
            value_fn=lambda split_idx, g: split_idx * 100 + g,
        )

        paths = reconstruct_paths(per_split, n_groups, n_test_groups)

        expected_phi = math.comb(n_groups - 1, n_test_groups - 1)
        assert expected_phi == 5
        assert paths.shape == (12, 5)
        assert not paths.isna().any().any()
        assert not paths.index.duplicated().any()
        # every event appears exactly once per column (12 rows, 12 unique dates)
        assert set(paths.index) == set(dates)

    def test_each_column_uses_every_group_exactly_once(self):
        """Every path column must touch each group's block exactly once --
        the structural property that makes the result dense/NaN-free."""
        n_groups, n_test_groups = 5, 2
        dates = pd.date_range("2026-03-01", periods=10, freq="D")
        group_dates = {g: dates[2 * g : 2 * g + 2] for g in range(n_groups)}
        per_split = _build_per_split(
            n_groups, n_test_groups, group_dates,
            value_fn=lambda split_idx, g: split_idx,
        )
        paths = reconstruct_paths(per_split, n_groups, n_test_groups)
        expected_phi = math.comb(n_groups - 1, n_test_groups - 1)
        assert paths.shape == (10, expected_phi)
        for col in paths.columns:
            # each group's 2-row block should be constant within a column
            # (all values came from a single split for that group)
            for g in range(n_groups):
                block = paths[col].loc[group_dates[g]]
                assert block.nunique() == 1


class TestReconstructPathsValidation:
    def test_n_groups_below_2_raises(self):
        with pytest.raises(ValueError):
            reconstruct_paths({}, n_groups=1, n_test_groups=1)

    def test_n_test_groups_below_1_raises(self):
        with pytest.raises(ValueError):
            reconstruct_paths({}, n_groups=4, n_test_groups=0)

    def test_n_test_groups_must_be_less_than_n_groups(self):
        with pytest.raises(ValueError):
            reconstruct_paths({}, n_groups=4, n_test_groups=4)

    def test_wrong_split_count_raises(self):
        n_groups, n_test_groups = 4, 2  # expects C(4,2)=6 splits
        dates = pd.date_range("2026-01-01", periods=8, freq="D")
        group_dates = {g: dates[2 * g : 2 * g + 2] for g in range(n_groups)}
        per_split = _build_per_split(
            n_groups, n_test_groups, group_dates, value_fn=lambda s, g: s
        )
        del per_split[5]  # drop one split -> only 5 of 6 present
        with pytest.raises(ValueError):
            reconstruct_paths(per_split, n_groups, n_test_groups)

    def test_correct_count_but_missing_split_idx_key_raises(self):
        """Same total length as expected, but a specific split_idx key is
        absent (e.g. off-by-one renaming) -- distinct from the raw count
        check above."""
        n_groups, n_test_groups = 4, 2
        dates = pd.date_range("2026-01-01", periods=8, freq="D")
        group_dates = {g: dates[2 * g : 2 * g + 2] for g in range(n_groups)}
        per_split = _build_per_split(
            n_groups, n_test_groups, group_dates, value_fn=lambda s, g: s
        )
        per_split[99] = per_split.pop(5)  # rename key 5 -> 99, count unchanged
        with pytest.raises(ValueError):
            reconstruct_paths(per_split, n_groups, n_test_groups)

    def test_mismatched_group_keys_in_a_split_raises(self):
        n_groups, n_test_groups = 4, 2
        dates = pd.date_range("2026-01-01", periods=8, freq="D")
        group_dates = {g: dates[2 * g : 2 * g + 2] for g in range(n_groups)}
        per_split = _build_per_split(
            n_groups, n_test_groups, group_dates, value_fn=lambda s, g: s
        )
        # combo for split_idx=0 is (0, 1) -- corrupt it to (0, 2)
        per_split[0] = {0: per_split[0][0], 2: pd.Series([1, 1], index=group_dates[2])}
        with pytest.raises(ValueError):
            reconstruct_paths(per_split, n_groups, n_test_groups)


class TestPartitionEquivalenceWithCombinatorialPurgedKFold:
    def test_group_partition_and_combo_order_match_the_real_splitter(self):
        """Guards the coupling: reconstruct_paths assumes group partitioning
        via np.array_split and combo order via itertools.combinations
        exactly matches what CombinatorialPurgedKFold.split() actually
        yields as test_pos, in the same order. No private import needed --
        this is a pure behavioral comparison."""
        n_groups, n_test_groups = 6, 2
        n = 60
        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        t1 = pd.Series(dates, index=dates)
        X = pd.DataFrame(index=dates)

        cpkf = CombinatorialPurgedKFold(
            n_groups=n_groups, n_test_groups=n_test_groups, t1=t1,
            embargo_pct=0.0,
        )
        actual_test_pos = [test_pos for _, test_pos in cpkf.split(X)]

        group_positions = np.array_split(np.arange(n), n_groups)
        combos = list(itertools.combinations(range(n_groups), n_test_groups))
        expected_test_pos = [
            np.concatenate([group_positions[g] for g in sorted(combo)])
            for combo in combos
        ]

        assert len(actual_test_pos) == len(expected_test_pos)
        for actual, expected in zip(actual_test_pos, expected_test_pos):
            np.testing.assert_array_equal(actual, expected)

    def test_phi_formula_matches_splitter_n_splits_times_k_over_n(self):
        """n_splits * k == N * phi (each of the N*phi group-slots across
        all splits corresponds to exactly one (split, group) pairing)."""
        for n_groups, n_test_groups in [(4, 2), (5, 2), (6, 2), (6, 3), (10, 2)]:
            n_splits = math.comb(n_groups, n_test_groups)
            phi = math.comb(n_groups - 1, n_test_groups - 1)
            assert n_splits * n_test_groups == n_groups * phi
