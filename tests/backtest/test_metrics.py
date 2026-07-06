"""Acceptance tests for backtest.metrics (Phase 2, Task 2).

Every numeric oracle here was independently re-derived and verified against
this exact implementation before delivery (mirroring the lock-session
verification discipline) -- these are not aspirational targets, they are
reproducible given the fixed seeds used.
"""
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from backtest.metrics import (
    deflated_sharpe,
    drawdown_tuw,
    expected_max_sharpe,
    hhi_negative,
    hhi_positive,
    hhi_temporal,
    pbo,
    psr,
    sharpe,
    sharpe_lo,
)


class TestSharpe:
    def test_known_value(self):
        r = np.array([0.01, -0.02, 0.03, 0.015])
        expected = r.mean() / r.std(ddof=1)
        assert sharpe(r) == pytest.approx(expected)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            sharpe(np.array([]))

    def test_constant_returns_raises(self):
        """std == 0 is fail-loud, not a silent inf/nan."""
        with pytest.raises(ValueError):
            sharpe(np.array([0.01, 0.01, 0.01]))


class TestSharpeLo:
    def test_zero_autocorrelation_recovers_sqrt_q_annualization(self):
        """When returns have (near) zero autocorrelation at every lag,
        eta(q) -> sqrt(q) -- the standard square-root-of-time rule."""
        rng = np.random.default_rng(2026)
        r = rng.standard_normal(20000) * 0.01 + 0.0005
        q = 4
        sl = sharpe_lo(r, q)
        expected = sharpe(r) * np.sqrt(q)
        assert sl == pytest.approx(expected, rel=0.01)

    def test_q_equals_1_matches_plain_sharpe(self):
        """No autocorrelation terms possible at q=1 -- eta(1) == 1."""
        r = np.array([0.01, -0.02, 0.03, 0.015, -0.01])
        assert sharpe_lo(r, 1) == pytest.approx(sharpe(r))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            sharpe_lo(np.array([]), 4)

    def test_q_below_1_raises(self):
        r = np.array([0.01, -0.02, 0.03])
        with pytest.raises(ValueError):
            sharpe_lo(r, 0)

    def test_nan_autocorrelation_at_high_lag_is_treated_as_zero(self):
        """On a short series, autocorr at a high lag has too few
        overlapping pairs and returns NaN -- the NaN-guard must substitute
        0.0 rather than propagate NaN through eta(q)."""
        r = np.array([0.01, -0.02, 0.03, 0.015, -0.01])  # n=5
        result = sharpe_lo(r, q=5)  # lag=4 has a single overlapping pair
        assert np.isfinite(result)


class TestPSR:
    def test_gaussian_reduction_matches_theoretical_approx(self):
        """For a large near-Gaussian sample, the full Bailey-LdP PSR (with
        sample skew/kurtosis corrections) closely matches the pure
        theoretical-Gaussian approximation Phi(SR * sqrt(n-1))."""
        rng = np.random.default_rng(42)
        n = 2000
        r = rng.normal(0.001, 0.01, n)
        sr = sharpe(r)
        full = psr(r)
        approx = stats.norm.cdf(sr * np.sqrt(n - 1))
        assert abs(full - approx) < 1e-3

    def test_psr_increases_with_n_at_fixed_sharpe(self):
        """Holding the realized Sharpe fixed by construction (affine
        rescaling preserves skew/kurtosis exactly), more observations at
        the same edge strength should raise PSR monotonically."""

        def make_returns_with_exact_sr(rng, n, target_mean, target_std):
            z = rng.standard_normal(n)
            z = (z - z.mean()) / z.std(ddof=1)
            return target_mean + target_std * z

        rng = np.random.default_rng(123)
        target_mean, target_std = 0.0002, 0.01  # SR == 0.02 exactly
        sizes = [500, 1000, 2000, 4000]
        values = []
        for n in sizes:
            r = make_returns_with_exact_sr(rng, n, target_mean, target_std)
            values.append(psr(r))
        assert values == sorted(values)
        assert values[0] == pytest.approx(0.6722, abs=1e-3)
        assert values[-1] == pytest.approx(0.8969, abs=1e-3)

    def test_negative_skew_lowers_psr_at_equal_sharpe(self):
        """Same realized Sharpe (forced by affine rescaling, which is
        skew/kurtosis-invariant); a more negatively skewed shape should
        yield a strictly lower PSR than a near-symmetric one."""
        rng = np.random.default_rng(99)
        n = 1000
        target_mean, target_std = 0.0002, 0.01

        z = rng.standard_normal(n)
        z = (z - z.mean()) / z.std(ddof=1)
        sym = target_mean + target_std * z

        raw = stats.skewnorm.rvs(a=-8, size=n, random_state=rng)
        raw = (raw - raw.mean()) / raw.std(ddof=1)
        skewed = target_mean + target_std * raw

        assert sharpe(sym) == pytest.approx(sharpe(skewed), abs=1e-9)
        assert psr(skewed) < psr(sym)

    def test_fewer_than_2_observations_raises(self):
        with pytest.raises(ValueError):
            psr(np.array([0.01]))
        with pytest.raises(ValueError):
            psr(np.array([]))


class TestExpectedMaxSharpe:
    def test_oracle_values_at_var_0_01(self):
        assert expected_max_sharpe(0.01, 10) == pytest.approx(0.157, abs=1e-3)
        assert expected_max_sharpe(0.01, 100) == pytest.approx(0.253, abs=1e-3)

    def test_increases_with_n_trials(self):
        vals = [expected_max_sharpe(0.01, n) for n in [2, 10, 50, 100, 500]]
        assert vals == sorted(vals)

    def test_n_trials_below_1_raises(self):
        with pytest.raises(ValueError):
            expected_max_sharpe(0.01, 0)

    def test_negative_var_sr_raises(self):
        with pytest.raises(ValueError):
            expected_max_sharpe(-0.01, 10)


class TestDeflatedSharpe:
    def test_decreases_with_n_trials_at_fixed_variance(self):
        rng = np.random.default_rng(11)
        r = rng.normal(0.001, 0.01, 1000)
        vals = [deflated_sharpe(r, 0.01, n) for n in [2, 10, 50, 100, 500]]
        assert vals == sorted(vals, reverse=True)

    def test_equals_psr_with_expected_max_sharpe_as_benchmark(self):
        rng = np.random.default_rng(12)
        r = rng.normal(0.001, 0.01, 500)
        sr_star = expected_max_sharpe(0.02, 20)
        assert deflated_sharpe(r, 0.02, 20) == pytest.approx(
            psr(r, sr_star=sr_star)
        )


class TestPBO:
    def test_all_noise_regime_near_half(self):
        """Pure noise across all trials -- being IS-best is meaningless,
        OOS rank should be roughly uniform -> PBO ~ 0.45-0.55."""
        rng = np.random.default_rng(1)
        t, n, s = 80, 16, 16
        noise = rng.normal(0, 0.01, size=(t, n))
        value, lambdas = pbo(noise, n_partitions=s)
        assert 0.35 <= value <= 0.65
        assert lambdas.shape == (
            len(list(itertools.combinations(range(s), s // 2))),
        )

    def test_one_genuine_edge_regime_near_zero(self):
        """One trial has a real, consistent edge -- it should win IS and
        OOS together almost every split -> PBO ~ 0."""
        rng = np.random.default_rng(1)
        t, n, s = 80, 16, 16
        _ = rng.normal(0, 0.01, size=(t, n))  # advance stream past regime 1
        edge = rng.normal(0, 0.01, size=(t, n))
        edge[:, 0] += 0.01
        value, _ = pbo(edge, n_partitions=s)
        assert value < 0.05

    def test_partition_level_overfit_regime_near_one(self):
        """Each trial is 'tuned' to spike only during its own dedicated
        group -- whichever trial wins IS owes it entirely to a group that
        is then absent OOS -> PBO should be very high."""
        rng = np.random.default_rng(1)
        t, n, s = 80, 16, 16
        _ = rng.normal(0, 0.01, size=(t, n))
        _ = rng.normal(0, 0.01, size=(t, n))
        m = 5
        t3 = s * m
        overfit = rng.normal(-0.001, 0.005, size=(t3, s))
        groups = np.array_split(np.arange(t3), s)
        for i in range(s):
            overfit[groups[i], i] += 0.05
        value, _ = pbo(overfit, n_partitions=s)
        assert value >= 0.9

    def test_odd_partitions_raises(self):
        r = np.random.default_rng(0).normal(size=(32, 4))
        with pytest.raises(ValueError):
            pbo(r, n_partitions=15)

    def test_partitions_below_2_raises(self):
        """Distinct from the odd-S check: S=0 is even but still invalid."""
        r = np.random.default_rng(0).normal(size=(32, 4))
        with pytest.raises(ValueError):
            pbo(r, n_partitions=0)

    def test_fewer_than_2_trial_columns_raises(self):
        r = np.random.default_rng(0).normal(size=(32, 1))
        with pytest.raises(ValueError):
            pbo(r, n_partitions=16)

    def test_fewer_rows_than_partitions_raises(self):
        r = np.random.default_rng(0).normal(size=(8, 4))
        with pytest.raises(ValueError):
            pbo(r, n_partitions=16)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            pbo(np.array([1.0, 2.0, 3.0]), n_partitions=4)


class TestHHI:
    def test_uniform_is_near_zero(self):
        uniform_returns = np.full(10, 0.05)
        assert hhi_positive(uniform_returns) == pytest.approx(0.0, abs=1e-9)

    def test_concentrated_positive_matches_oracle(self):
        conc = np.array([0.001] * 9 + [0.5])
        assert hhi_positive(conc) == pytest.approx(0.961, abs=1e-3)

    def test_concentrated_negative_matches_oracle(self):
        conc = np.array([-0.001] * 9 + [-0.5])
        assert hhi_negative(conc) == pytest.approx(0.961, abs=1e-3)

    def test_hhi_positive_ignores_negative_entries(self):
        mixed = np.array([0.001] * 9 + [0.5, -100.0, -200.0])
        assert hhi_positive(mixed) == pytest.approx(0.961, abs=1e-3)

    def test_no_positive_returns_is_nan(self):
        assert np.isnan(hhi_positive(np.array([-0.01, -0.02, -0.03])))

    def test_no_negative_returns_is_nan(self):
        assert np.isnan(hhi_negative(np.array([0.01, 0.02, 0.03])))

    def test_two_or_fewer_is_nan(self):
        assert np.isnan(hhi_positive(np.array([0.1, 0.2])))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            hhi_positive(np.array([]))
        with pytest.raises(ValueError):
            hhi_negative(np.array([]))

    def test_temporal_uniform_is_near_zero(self):
        dates = pd.to_datetime([f"2026-{m:02d}-15" for m in range(1, 11)])
        assert hhi_temporal(dates) == pytest.approx(0.0, abs=1e-9)

    def test_temporal_concentrated_matches_oracle(self):
        dates = pd.to_datetime(
            [f"2026-{m:02d}-15" for m in range(1, 10)]
            + ["2026-10-15"] * 500
        )
        assert hhi_temporal(dates) == pytest.approx(0.961, abs=1e-3)

    def test_temporal_two_periods_is_nan(self):
        dates = pd.to_datetime(["2026-01-15", "2026-02-15"])
        assert np.isnan(hhi_temporal(dates))

    def test_temporal_empty_raises(self):
        with pytest.raises(ValueError):
            hhi_temporal(pd.to_datetime([]))


class TestDrawdownTuW:
    def test_known_equity_path_max_dd_is_exact(self):
        """Equity 1 -> 1.2 -> 0.9 -> 1.5: max DD == 0.25 exactly."""
        dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        returns = pd.Series([0.2, -0.25, 0.6666666666666667], index=dates)
        dd, tuw = drawdown_tuw(returns)
        assert dd.max() == pytest.approx(0.25, abs=1e-9)
        np.testing.assert_allclose(dd.to_numpy(), [0.0, 0.25, 0.0], atol=1e-9)

    def test_tuw_zero_at_new_highs_positive_underwater(self):
        dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        returns = pd.Series([0.2, -0.25, 0.6666666666666667], index=dates)
        _, tuw = drawdown_tuw(returns)
        assert tuw.iloc[0] == 0.0
        assert tuw.iloc[1] == pytest.approx(1 / 365.25, rel=1e-6)
        assert tuw.iloc[2] == 0.0

    def test_monotonically_rising_equity_has_zero_drawdown(self):
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        returns = pd.Series([0.01, 0.02, 0.01, 0.03, 0.01], index=dates)
        dd, tuw = drawdown_tuw(returns)
        np.testing.assert_allclose(dd.to_numpy(), 0.0, atol=1e-9)
        np.testing.assert_allclose(tuw.to_numpy(), 0.0, atol=1e-9)

    def test_returns_full_length_series(self):
        dates = pd.date_range("2026-01-01", periods=6, freq="D")
        returns = pd.Series(
            [0.1, -0.05, 0.02, -0.2, 0.15, 0.05], index=dates
        )
        dd, tuw = drawdown_tuw(returns)
        assert len(dd) == len(returns)
        assert len(tuw) == len(returns)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            drawdown_tuw(pd.Series([], dtype=float))