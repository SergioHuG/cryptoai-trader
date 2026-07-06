"""Acceptance tests for backtest.synthetic (Phase 2, Task 4).

Recovery tests use a fixed seed and a generous-but-meaningful tolerance
(sampling noise at finite n means exact reproduction of the lock
session's specific numbers isn't the point -- recovering the true
parameter within a reasonable band is). The exact-phi edge cases (1.0,
1.5) use deterministic, noise-free sequences constructed to produce that
exact OLS slope by arithmetic, not by chance.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.synthetic import OUParams, fit_ou, simulate


class TestFitOURecovery:
    def test_recovers_phi_and_sigma_within_tolerance(self):
        true_phi, true_sigma, true_c = 0.35, 0.01, 0.0
        n = 5000
        rng = np.random.default_rng(2024)
        r = np.empty(n)
        r[0] = 0.0
        for t in range(1, n):
            r[t] = true_c + true_phi * r[t - 1] + true_sigma * rng.standard_normal()

        params = fit_ou(r)
        assert abs(params.phi - true_phi) < 0.05
        assert abs(params.sigma - true_sigma) < 0.001

    def test_white_noise_recovers_phi_near_zero(self):
        rng = np.random.default_rng(5)
        wn = rng.standard_normal(5000) * 0.01
        params = fit_ou(wn)
        assert abs(params.phi) < 0.05


class TestFitOUExactEdgeCases:
    def test_phi_exactly_one_gives_nan_mu_and_inf_half_life(self):
        """r = [1,2,3,4] is a perfect-fit arithmetic sequence: OLS of
        r_t on r_{t-1} gives slope==1.0, intercept==1.0 exactly (zero
        residual) -- deterministic, not seed-dependent."""
        r = np.array([1.0, 2.0, 3.0, 4.0])
        params = fit_ou(r)
        assert params.phi == pytest.approx(1.0)
        assert params.c == pytest.approx(1.0)
        assert params.sigma == pytest.approx(0.0, abs=1e-9)
        assert np.isnan(params.mu)
        assert params.half_life == float("inf")

    def test_explosive_phi_above_one_gives_inf_half_life(self):
        """r = [1, 1.5, 2.25, 3.375, 5.0625] is an exact geometric
        sequence with ratio 1.5: OLS recovers phi==1.5, c==0.0 exactly."""
        r = np.array([1.0, 1.5, 2.25, 3.375, 5.0625])
        params = fit_ou(r)
        assert params.phi == pytest.approx(1.5)
        assert params.c == pytest.approx(0.0, abs=1e-9)
        assert params.half_life == float("inf")

    def test_phi_zero_gives_zero_half_life(self):
        """r = [-2,-2,0,-1] is an exact integer sequence (found by
        construction) producing cov(x,y) == 0 exactly, so OLS recovers
        phi_hat == 0.0 -- exercised through fit_ou itself, not asserted
        by convention alone."""
        r = np.array([-2.0, -2.0, 0.0, -1.0])
        params = fit_ou(r)
        assert params.phi == 0.0
        assert params.half_life == 0.0


class TestFitOUValidation:
    def test_fewer_than_4_observations_raises(self):
        with pytest.raises(ValueError):
            fit_ou(np.array([1.0, 2.0, 3.0]))

    def test_constant_series_raises(self):
        """r_{t-1} has zero variance when the whole series is constant --
        the AR(1) slope is undefined."""
        with pytest.raises(ValueError):
            fit_ou(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))


class TestSimulateDeterminism:
    def _params(self):
        phi = 0.35
        c = 0.0002
        return OUParams(c=c, phi=phi, sigma=0.01, mu=c / (1 - phi), half_life=1.2)

    def test_same_seed_gives_byte_identical_paths(self):
        params = self._params()
        df1 = simulate(params, n_obs=100, n_sims=5, seed=42, burn_in=50)
        df2 = simulate(params, n_obs=100, n_sims=5, seed=42, burn_in=50)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_gives_different_paths(self):
        params = self._params()
        df1 = simulate(params, n_obs=100, n_sims=5, seed=42, burn_in=50)
        df3 = simulate(params, n_obs=100, n_sims=5, seed=99, burn_in=50)
        assert not df1.equals(df3)

    def test_output_shape_and_columns(self):
        params = self._params()
        df = simulate(params, n_obs=100, n_sims=7, seed=1, burn_in=20)
        assert df.shape == (100, 7)
        assert list(df.columns) == list(range(7))

    def test_mean_init_at_zero_burn_in_first_row_equals_mu(self):
        """With no burn-in, the first recorded observation is the
        deterministic mean-init point for every simulated path."""
        params = self._params()
        df = simulate(params, n_obs=5, n_sims=4, seed=1, burn_in=0)
        np.testing.assert_allclose(df.iloc[0].to_numpy(), params.mu)


class TestSimulateValidation:
    def test_non_stationary_phi_exactly_one_raises(self):
        bad = OUParams(c=0.0, phi=1.0, sigma=0.01, mu=float("nan"), half_life=float("inf"))
        with pytest.raises(ValueError):
            simulate(bad, n_obs=10, n_sims=3, seed=1, burn_in=5)

    def test_explosive_phi_above_one_raises(self):
        bad = OUParams(c=0.0, phi=1.2, sigma=0.01, mu=0.0, half_life=float("inf"))
        with pytest.raises(ValueError):
            simulate(bad, n_obs=10, n_sims=3, seed=1, burn_in=5)

    def test_n_obs_below_1_raises(self):
        params = TestSimulateDeterminism()._params()
        with pytest.raises(ValueError):
            simulate(params, n_obs=0, n_sims=3, seed=1, burn_in=5)

    def test_n_sims_below_1_raises(self):
        params = TestSimulateDeterminism()._params()
        with pytest.raises(ValueError):
            simulate(params, n_obs=10, n_sims=0, seed=1, burn_in=5)

    def test_negative_burn_in_raises(self):
        params = TestSimulateDeterminism()._params()
        with pytest.raises(ValueError):
            simulate(params, n_obs=10, n_sims=3, seed=1, burn_in=-1)
