"""Acceptance tests for research/validation/splitters.py -- PurgedKFold
(AFML Snippet 7.3, Step 3).

A thin positional adapter over the Step 2 kernel (Q3/Q4/Q6): constructs
contiguous, non-shuffled test folds over the sample order, then for each
fold builds a single (block_start, block_max_t1) span and calls
:func:`research.validation.purge._purge_embargo` to get the surviving
train timestamps, mapping back to positions. This file does NOT re-derive
purge+embargo correctness (that's Step 2c's job, already proven) -- it
only proves the splitter wires positions <-> timestamps correctly and
produces the right fold structure.

Shared fixture: 12 zero-width events (t0 == t1) on an hourly grid, so
every test event is a single isolated point with no inherent overlap
width of its own -- isolating the embargo's exact contribution from any
incidental event-width purge effects. n_splits=3 -> three folds of 4
positions each: [0..3], [4..7], [8..11].

Every exact train_pos value below was independently verified against a
working draft implementation before being locked into these assertions
(not hand-derived and trusted blind).
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection._split import _BaseKFold

from research.validation.splitters import PurgedKFold

_SBARS = pd.date_range("2024-01-01", periods=12, freq="1h", tz="UTC")


def _zero_width_t1() -> pd.Series:
    return pd.Series({b: b for b in _SBARS})


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame({"x": np.arange(len(_SBARS))}, index=_SBARS)


class TestPurgedKFoldValidation:
    def test_raises_if_t1_never_set(self):
        pkf = PurgedKFold(n_splits=3, t1=None, embargo_pct=0.0)
        with pytest.raises(ValueError):
            list(pkf.split(_feature_frame()))

    def test_alignment_mismatch_different_length_raises(self):
        t1 = _zero_width_t1()
        X = _feature_frame().iloc[:10]  # 10 rows vs t1's 12
        pkf = PurgedKFold(n_splits=3, t1=t1, embargo_pct=0.0)
        with pytest.raises(ValueError):
            list(pkf.split(X))

    def test_alignment_mismatch_same_length_different_order_raises(self):
        t1 = _zero_width_t1()
        shuffled = _SBARS[[1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        X = _feature_frame().loc[shuffled]  # same 12 rows, reordered
        pkf = PurgedKFold(n_splits=3, t1=t1, embargo_pct=0.0)
        with pytest.raises(ValueError):
            list(pkf.split(X))

    def test_non_monotonic_index_raises(self):
        """X and t1 share the SAME shuffled order (alignment passes) but
        neither is ascending -- isolates the monotonic check specifically."""
        shuffled = _SBARS[[0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        t1 = _zero_width_t1().loc[shuffled]
        X = _feature_frame().loc[shuffled]
        pkf = PurgedKFold(n_splits=3, t1=t1, embargo_pct=0.0)
        with pytest.raises(ValueError):
            list(pkf.split(X))


class TestPurgedKFoldFoldStructure:
    def test_get_n_splits_returns_n_splits(self):
        pkf = PurgedKFold(n_splits=3, t1=_zero_width_t1(), embargo_pct=0.0)
        assert pkf.get_n_splits() == 3
        assert pkf.get_n_splits(_feature_frame()) == 3

    def test_yields_exactly_n_splits_folds(self):
        pkf = PurgedKFold(n_splits=3, t1=_zero_width_t1(), embargo_pct=0.0)
        folds = list(pkf.split(_feature_frame()))
        assert len(folds) == 3

    def test_test_folds_are_contiguous_and_cover_every_position_once(self):
        pkf = PurgedKFold(n_splits=3, t1=_zero_width_t1(), embargo_pct=0.0)
        all_test_pos = np.concatenate(
            [test_pos for _, test_pos in pkf.split(_feature_frame())]
        )
        assert list(all_test_pos) == list(range(12))

    def test_each_fold_has_four_contiguous_test_positions(self):
        pkf = PurgedKFold(n_splits=3, t1=_zero_width_t1(), embargo_pct=0.0)
        for _, test_pos in pkf.split(_feature_frame()):
            assert len(test_pos) == 4
            assert list(test_pos) == list(range(test_pos[0], test_pos[0] + 4))


class TestPurgedKFoldNoEmbargo:
    """embargo_pct == 0.0 -- train excludes only the test fold's own
    (self-overlapping) positions, nothing more."""

    def test_fold_zero_train_excludes_only_its_own_test_positions(self):
        pkf = PurgedKFold(n_splits=3, t1=_zero_width_t1(), embargo_pct=0.0)
        folds = list(pkf.split(_feature_frame()))
        train_pos, test_pos = folds[0]
        assert list(test_pos) == [0, 1, 2, 3]
        assert list(train_pos) == [4, 5, 6, 7, 8, 9, 10, 11]

    def test_train_never_overlaps_test_in_any_fold(self):
        pkf = PurgedKFold(n_splits=3, t1=_zero_width_t1(), embargo_pct=0.0)
        for train_pos, test_pos in pkf.split(_feature_frame()):
            assert set(train_pos).isdisjoint(set(test_pos))


class TestPurgedKFoldWithEmbargo:
    """embargo_pct == 0.2 on n=12 -> step=2. Verified against a working
    draft before being locked in -- see module docstring."""

    def test_fold_zero_start_embargo_extends_train_exclusion_forward(self):
        """Test span [pos0,pos3]; embargo extends the right edge to pos5.
        Positions 4,5 are embargo-governed -- excluded from train despite
        not being test positions themselves."""
        pkf = PurgedKFold(n_splits=3, t1=_zero_width_t1(), embargo_pct=0.2)
        folds = list(pkf.split(_feature_frame()))
        train_pos, test_pos = folds[0]
        assert list(test_pos) == [0, 1, 2, 3]
        assert list(train_pos) == [6, 7, 8, 9, 10, 11]

    def test_fold_one_middle_embargo_extends_on_only_the_right_side(self):
        """Test span [pos4,pos7]; embargo extends the right edge to pos9.
        Train before the span (positions 0-3) is untouched -- forward-only
        embargo, confirmed at the splitter level."""
        pkf = PurgedKFold(n_splits=3, t1=_zero_width_t1(), embargo_pct=0.2)
        folds = list(pkf.split(_feature_frame()))
        train_pos, test_pos = folds[1]
        assert list(test_pos) == [4, 5, 6, 7]
        assert list(train_pos) == [0, 1, 2, 3, 10, 11]

    def test_fold_two_final_fold_embargo_is_a_structural_no_op(self):
        """Test span ends at the grid's last position -- nothing exists
        past it to embargo, so embargo_pct=0.2 changes nothing here."""
        pkf = PurgedKFold(n_splits=3, t1=_zero_width_t1(), embargo_pct=0.2)
        folds = list(pkf.split(_feature_frame()))
        train_pos, test_pos = folds[2]
        assert list(test_pos) == [8, 9, 10, 11]
        assert list(train_pos) == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_train_never_overlaps_test_with_embargo_active(self):
        pkf = PurgedKFold(n_splits=3, t1=_zero_width_t1(), embargo_pct=0.2)
        for train_pos, test_pos in pkf.split(_feature_frame()):
            assert set(train_pos).isdisjoint(set(test_pos))


class TestPurgedKFoldSklearnCompatibility:
    def test_is_a_base_kfold_subclass(self):
        pkf = PurgedKFold(n_splits=3, t1=_zero_width_t1(), embargo_pct=0.0)
        assert isinstance(pkf, _BaseKFold)

    def test_consumable_via_plain_for_loop(self):
        pkf = PurgedKFold(n_splits=3, t1=_zero_width_t1(), embargo_pct=0.0)
        count = 0
        for train_pos, test_pos in pkf.split(_feature_frame()):
            count += 1
        assert count == 3

    def test_yields_integer_position_arrays(self):
        pkf = PurgedKFold(n_splits=3, t1=_zero_width_t1(), embargo_pct=0.0)
        train_pos, test_pos = next(pkf.split(_feature_frame()))
        assert isinstance(train_pos, np.ndarray)
        assert isinstance(test_pos, np.ndarray)
        assert np.issubdtype(train_pos.dtype, np.integer)
        assert np.issubdtype(test_pos.dtype, np.integer)
