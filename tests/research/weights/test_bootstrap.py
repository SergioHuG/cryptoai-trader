"""Acceptance tests for research.weights.bootstrap (Step 5, AFML 4.3-4.5).

Standalone module -- orthogonal to config/storage/pipeline (Q1). Never
wired into WeightConfig or build_sample_weights.

get_ind_matrix follows AFML's getIndMatrix exactly: dense (bars x events),
INTEGER positional columns (0..n_events-1), not t0 timestamps -- this
sidesteps duplicate-column ambiguity when multiple events share a start
bar, and matches AFML's own column convention verbatim.

ind_matrix_avg_uniqueness (AFML's getAvgUniqueness, matrix-based) is a
genuinely different computation path from
research.weights.concurrency.avg_uniqueness (the span-based form) --
seq_bootstrap needs the matrix form because it must recompute avgU over
HYPOTHETICAL candidate subsets that have no precomputed co_events series.
TestIndMatrixAvgUniquenessMatchesConcurrencyOracle proves the two paths
can never silently diverge (Q10c).

seq_bootstrap takes an injectable seeded rng (Q10b) -- non-negotiable for
reproducibility in a risk-gated system, and pre-wires the seam the parked
"fixed bootstrapped index for exact replay" feature (-> model/training
branch) will eventually record a seed against.
"""
import numpy as np
import pandas as pd
import pytest

from research.weights.bootstrap import (
    get_ind_matrix,
    ind_matrix_avg_uniqueness,
    seq_bootstrap,
)
from research.weights.concurrency import avg_uniqueness, num_co_events


def _utc_index(n, start="2024-01-01", freq="15min"):
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC")


def _t1(idx, pairs, name="t1"):
    starts = [idx[s] for s, _ in pairs]
    ends = [idx[e] for _, e in pairs]
    return pd.Series(ends, index=pd.DatetimeIndex(starts), name=name)


class TestGetIndMatrixShape:
    def test_shape_is_bars_by_events(self):
        idx = _utc_index(5)
        t1 = _t1(idx, [(1, 3), (0, 2)])

        result = get_ind_matrix(idx, t1)

        assert result.shape == (5, 2)

    def test_is_dense_dataframe_indexed_by_bar_grid(self):
        idx = _utc_index(5)
        t1 = _t1(idx, [(1, 3)])

        result = get_ind_matrix(idx, t1)

        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_index_equal(result.index, idx)

    def test_columns_are_positional_integers(self):
        idx = _utc_index(5)
        t1 = _t1(idx, [(1, 3), (0, 2), (2, 4)])

        result = get_ind_matrix(idx, t1)

        assert list(result.columns) == [0, 1, 2]

    def test_empty_events_gives_zero_columns(self):
        idx = _utc_index(5)
        t1 = pd.Series([], dtype="datetime64[ns, UTC]", name="t1")

        result = get_ind_matrix(idx, t1)

        assert result.shape == (5, 0)


class TestGetIndMatrixIndicatorValues:
    def test_indicator_is_one_over_span_zero_elsewhere(self):
        idx = _utc_index(5)
        t1 = _t1(idx, [(1, 3)])

        result = get_ind_matrix(idx, t1)

        assert list(result[0]) == [0, 1, 1, 1, 0]

    def test_two_overlapping_events_each_get_own_column(self):
        idx = _utc_index(5)
        t1 = _t1(idx, [(0, 2), (1, 3)])

        result = get_ind_matrix(idx, t1)

        assert list(result[0]) == [1, 1, 1, 0, 0]
        assert list(result[1]) == [0, 1, 1, 1, 0]


class TestIndMatrixAvgUniquenessMatchesConcurrencyOracle:
    def test_matches_concurrency_avg_uniqueness_on_overlapping_fixture(self):
        """The matrix-based avgU (used internally by seq_bootstrap on
        hypothetical subsets) and the span-based avgU (used by the main
        weight pipeline) must agree exactly on a shared fixture -- the two
        computation paths can never silently diverge (Q10c)."""
        idx = _utc_index(8)
        t1 = _t1(idx, [(0, 3), (2, 5), (4, 7)])

        co_events = num_co_events(idx, t1)
        span_based = avg_uniqueness(t1, co_events)

        ind_matrix = get_ind_matrix(idx, t1)
        matrix_based = ind_matrix_avg_uniqueness(ind_matrix)

        assert list(matrix_based) == pytest.approx(list(span_based))

    def test_single_event_no_overlap_avg_uniqueness_one(self):
        idx = _utc_index(5)
        t1 = _t1(idx, [(1, 3)])

        result = ind_matrix_avg_uniqueness(get_ind_matrix(idx, t1))

        assert result.iloc[0] == pytest.approx(1.0)

    def test_indexed_by_matrix_columns(self):
        idx = _utc_index(5)
        t1 = _t1(idx, [(0, 2), (1, 3)])
        ind_matrix = get_ind_matrix(idx, t1)

        result = ind_matrix_avg_uniqueness(ind_matrix)

        pd.testing.assert_index_equal(result.index, ind_matrix.columns)


class TestSeqBootstrapDrawCount:
    def test_default_s_length_equals_n_events(self):
        idx = _utc_index(10)
        t1 = _t1(idx, [(0, 2), (1, 4), (3, 6), (5, 8)])
        ind_matrix = get_ind_matrix(idx, t1)

        phi = seq_bootstrap(ind_matrix, rng=np.random.default_rng(1))

        assert len(phi) == ind_matrix.shape[1]

    def test_custom_s_length_honored(self):
        idx = _utc_index(10)
        t1 = _t1(idx, [(0, 2), (1, 4), (3, 6), (5, 8)])
        ind_matrix = get_ind_matrix(idx, t1)

        phi = seq_bootstrap(ind_matrix, s_length=2, rng=np.random.default_rng(1))

        assert len(phi) == 2

    def test_empty_matrix_returns_empty_list(self):
        idx = _utc_index(5)
        ind_matrix = get_ind_matrix(idx, pd.Series([], dtype="datetime64[ns, UTC]"))

        phi = seq_bootstrap(ind_matrix, rng=np.random.default_rng(1))

        assert phi == []


class TestSeqBootstrapDrawValidity:
    def test_indices_are_valid_columns_with_replacement_allowed(self):
        idx = _utc_index(10)
        t1 = _t1(idx, [(0, 2), (1, 4), (3, 6)])
        ind_matrix = get_ind_matrix(idx, t1)

        phi = seq_bootstrap(ind_matrix, s_length=10, rng=np.random.default_rng(2))

        assert len(phi) == 10
        assert all(p in list(ind_matrix.columns) for p in phi)


class TestSeqBootstrapReproducibility:
    def test_deterministic_under_seeded_rng(self):
        idx = _utc_index(10)
        t1 = _t1(idx, [(0, 2), (1, 4), (3, 6), (5, 8)])
        ind_matrix = get_ind_matrix(idx, t1)

        phi_a = seq_bootstrap(ind_matrix, rng=np.random.default_rng(7))
        phi_b = seq_bootstrap(ind_matrix, rng=np.random.default_rng(7))

        assert phi_a == phi_b

    def test_different_seeds_are_likely_to_differ(self):
        idx = _utc_index(15)
        t1 = _t1(idx, [(0, 2), (1, 4), (3, 6), (5, 8), (7, 10), (9, 13)])
        ind_matrix = get_ind_matrix(idx, t1)

        phi_a = seq_bootstrap(ind_matrix, rng=np.random.default_rng(1))
        phi_b = seq_bootstrap(ind_matrix, rng=np.random.default_rng(2))

        assert phi_a != phi_b

    def test_rng_none_uses_fresh_default_and_runs_without_error(self):
        idx = _utc_index(8)
        t1 = _t1(idx, [(0, 2), (1, 4), (3, 6)])
        ind_matrix = get_ind_matrix(idx, t1)

        phi = seq_bootstrap(ind_matrix)

        assert len(phi) == ind_matrix.shape[1]
        assert all(p in list(ind_matrix.columns) for p in phi)


class TestSeqBootstrapFavorsUniqueSamples:
    def test_isolated_event_drawn_more_often_than_a_heavily_overlapping_cluster(
        self,
    ):
        """AFML's whole point: sequential bootstrap should draw an
        isolated, low-overlap event more often than uniform IID sampling
        would, because heavily overlapping candidates suppress each
        other's average uniqueness once any one of them is in the drawn
        pool. Statistical, not exact -- verified over many seeded runs."""
        idx = _utc_index(20)
        # column 0: isolated, no overlap with anything below.
        # columns 1-4: a tight, heavily overlapping cluster.
        t1 = _t1(
            idx,
            [(0, 1), (10, 19), (11, 19), (10, 18), (11, 18)],
        )
        ind_matrix = get_ind_matrix(idx, t1)
        n_events = ind_matrix.shape[1]
        uniform_baseline = 1.0 / n_events  # 0.2

        all_draws = []
        for seed in range(150):
            phi = seq_bootstrap(ind_matrix, rng=np.random.default_rng(seed))
            all_draws.extend(phi)

        isolated_frequency = sum(1 for p in all_draws if p == 0) / len(all_draws)

        assert isolated_frequency > uniform_baseline + 0.03
