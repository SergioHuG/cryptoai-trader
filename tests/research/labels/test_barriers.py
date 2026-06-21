"""Acceptance tests for the Barrier enum and triple-barrier helpers."""
import numpy as np

from research.labels.barriers import Barrier


class TestBarrier:
    def test_barrier_values(self):
        assert Barrier.LOWER == -1
        assert Barrier.VERTICAL == 0
        assert Barrier.UPPER == 1

    def test_barrier_is_int8_compatible(self):
        assert np.int8(Barrier.UPPER) == 1
        assert np.int8(Barrier.LOWER) == -1
