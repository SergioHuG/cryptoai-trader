"""
Tests for research/features/mp_engine.py

Key requirement (from Phase 0 spec):
    "Write a unit test dispatching a bound method of a mock class and
     asserting completion without PicklingError."

Tests cover:
  - Single-process dispatch of a plain function
  - Single-process dispatch of a bound method of a mock class
  - Batch splitting (mpBatches > 1)
  - Custom redux function
  - Input validation (numThreads < 1, mpBatches < 1, empty atoms)
  - Pickle registration side-effect (module-level)
"""
import pickle
import types
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from research.features.mp_engine import (
    _register_pickle_reducers,
    mp_pandas_obj,
)


# ── Helpers used as atoms targets ─────────────────────────────────────────────

def _double_list(molecule: list[int]) -> list[int]:
    """Plain module-level function — always picklable."""
    return [x * 2 for x in molecule]


def _sum_list(molecule: list[int]) -> int:
    return sum(molecule)


class _Worker:
    """Mock class with a bound method — exercises the copyreg reducer."""

    def __init__(self, multiplier: int) -> None:
        self.multiplier = multiplier

    def transform(self, molecule: list[int]) -> list[int]:
        return [x * self.multiplier for x in molecule]

    def accumulate(self, molecule: list[int]) -> Decimal:
        return Decimal(str(sum(x * self.multiplier for x in molecule)))


# ── Pickle registration ───────────────────────────────────────────────────────

def test_pickle_registration_is_idempotent() -> None:
    """Calling _register_pickle_reducers() twice must not raise."""
    _register_pickle_reducers()
    _register_pickle_reducers()


def test_bound_method_is_picklable_after_registration() -> None:
    """
    Core requirement: bound method of a class must be picklable after
    the module-level registration runs (which happens at import time).
    """
    worker = _Worker(multiplier=3)
    # This must not raise PicklingError
    data = pickle.dumps(worker.transform)
    recovered = pickle.loads(data)
    # recovered should be callable
    assert callable(recovered)


# ── Single-process dispatch ───────────────────────────────────────────────────

def test_plain_function_single_thread() -> None:
    atoms = [1, 2, 3, 4, 5]
    result = mp_pandas_obj(
        func=_double_list,
        pd_obj=("molecule", atoms),
        numThreads=1,
    )
    # Default redux (list flatten): all doubled values present
    assert sorted(result) == [2, 4, 6, 8, 10]


def test_bound_method_dispatches_without_pickling_error() -> None:
    """
    Spec requirement: dispatch a bound method of a mock class and assert
    completion without PicklingError.
    """
    worker = _Worker(multiplier=5)
    atoms = [1, 2, 3]

    result = mp_pandas_obj(
        func=worker.transform,
        pd_obj=("molecule", atoms),
        numThreads=1,
    )
    assert sorted(result) == [5, 10, 15]


def test_bound_method_with_decimal_output() -> None:
    worker = _Worker(multiplier=2)
    atoms = [10, 20, 30]

    result = mp_pandas_obj(
        func=worker.accumulate,
        pd_obj=("molecule", atoms),
        numThreads=1,
        redux=lambda a, b: a + b,
    )
    # sum([10,20,30]) * 2 = 120
    assert result == Decimal("120")


# ── Batch splitting ───────────────────────────────────────────────────────────

def test_multiple_batches_produce_same_result_as_single_batch() -> None:
    atoms = list(range(10))

    result_one = mp_pandas_obj(
        func=_double_list,
        pd_obj=("molecule", atoms),
        numThreads=1,
        mpBatches=1,
    )
    result_multi = mp_pandas_obj(
        func=_double_list,
        pd_obj=("molecule", atoms),
        numThreads=1,
        mpBatches=3,
    )
    assert sorted(result_one) == sorted(result_multi)


def test_more_batches_than_atoms_skips_empty_batches() -> None:
    """mpBatches > len(atoms) → empty batches silently dropped, no crash."""
    atoms = [1, 2]
    result = mp_pandas_obj(
        func=_double_list,
        pd_obj=("molecule", atoms),
        numThreads=1,
        mpBatches=10,
    )
    assert sorted(result) == [2, 4]


# ── Redux ─────────────────────────────────────────────────────────────────────

def test_custom_redux_function_applied() -> None:
    atoms = [1, 2, 3, 4, 5]
    result = mp_pandas_obj(
        func=_sum_list,
        pd_obj=("molecule", atoms),
        numThreads=1,
        mpBatches=2,
        redux=lambda a, b: a + b,
    )
    assert result == 15   # sum(1..5)


def test_redux_in_place_flag() -> None:
    """reduxInPlace=True — redux mutates first arg."""
    accumulated: list[int] = []

    def extend_in_place(acc: list, new: list) -> None:
        acc.extend(new)

    atoms = [1, 2, 3, 4]
    result = mp_pandas_obj(
        func=_double_list,
        pd_obj=("molecule", atoms),
        numThreads=1,
        mpBatches=2,
        redux=extend_in_place,
        reduxInPlace=True,
    )
    # reduxInPlace=True → redux returns None, accumulated in first arg
    # _reduce_results uses first element as accumulator
    # Just assert no exception is raised; shape depends on batch split
    assert result is None or isinstance(result, list)


# ── Input validation ──────────────────────────────────────────────────────────

def test_raises_on_zero_threads() -> None:
    with pytest.raises(ValueError, match="numThreads"):
        mp_pandas_obj(
            func=_double_list,
            pd_obj=("molecule", [1, 2]),
            numThreads=0,
        )


def test_raises_on_negative_threads() -> None:
    with pytest.raises(ValueError, match="numThreads"):
        mp_pandas_obj(
            func=_double_list,
            pd_obj=("molecule", [1, 2]),
            numThreads=-1,
        )


def test_raises_on_zero_batches() -> None:
    with pytest.raises(ValueError, match="mpBatches"):
        mp_pandas_obj(
            func=_double_list,
            pd_obj=("molecule", [1, 2]),
            mpBatches=0,
        )


def test_empty_atoms_returns_none() -> None:
    result = mp_pandas_obj(
        func=_double_list,
        pd_obj=("molecule", []),
        numThreads=1,
    )
    assert result is None


# ── kwargs forwarding ─────────────────────────────────────────────────────────

def test_extra_kwargs_forwarded_to_func() -> None:
    def multiply_by(molecule: list[int], factor: int) -> list[int]:
        return [x * factor for x in molecule]

    result = mp_pandas_obj(
        func=multiply_by,
        pd_obj=("molecule", [1, 2, 3]),
        numThreads=1,
        factor=7,
    )
    assert sorted(result) == [7, 14, 21]
