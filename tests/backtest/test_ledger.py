"""Acceptance tests for backtest.ledger (Phase 2, Task 5)."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from backtest.ledger import TrialRecord, assemble_matrix, n_trials, sr_variance


def _record(
    trial_id="t1",
    config_hash="abc123def456",
    created_at="2026-01-01",
    mode="cpcv",
    sharpe=0.5,
    returns=None,
):
    if returns is None:
        returns = pd.Series(
            [0.01, -0.02, 0.03], index=pd.date_range("2026-01-01", periods=3)
        )
    return TrialRecord(
        trial_id=trial_id,
        config_hash=config_hash,
        created_at=created_at,
        mode=mode,
        sharpe=sharpe,
        n_bets=len(returns),
        returns=returns,
    )


class TestTrialRecordConstruction:
    def test_valid_construction(self):
        rec = _record()
        assert rec.trial_id == "t1"
        assert rec.config_hash == "abc123def456"
        assert rec.mode == "cpcv"
        assert rec.sharpe == 0.5
        assert rec.n_bets == 3

    def test_created_at_coerced_to_timestamp(self):
        rec = _record(created_at="2026-03-15")
        assert isinstance(rec.created_at, pd.Timestamp)
        assert rec.created_at == pd.Timestamp("2026-03-15")

    def test_is_frozen(self):
        rec = _record()
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.sharpe = 99.0

    def test_empty_trial_id_raises(self):
        with pytest.raises(ValueError):
            _record(trial_id="")

    def test_empty_config_hash_raises(self):
        with pytest.raises(ValueError):
            _record(config_hash="")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            _record(mode="synthetic")

    def test_wf_mode_is_valid(self):
        rec = _record(mode="wf")
        assert rec.mode == "wf"

    def test_non_finite_sharpe_raises(self):
        with pytest.raises(ValueError):
            _record(sharpe=float("nan"))
        with pytest.raises(ValueError):
            _record(sharpe=float("inf"))

    def test_n_bets_below_1_raises(self):
        """Isolated from the empty-returns and length-mismatch checks:
        n_bets=0 with non-empty returns still must raise on n_bets alone
        (it's checked first)."""
        returns = pd.Series(
            [0.01, -0.02, 0.03], index=pd.date_range("2026-01-01", periods=3)
        )
        with pytest.raises(ValueError):
            TrialRecord(
                trial_id="t1", config_hash="abc", created_at="2026-01-01",
                mode="wf", sharpe=0.5, n_bets=0, returns=returns,
            )

    def test_empty_returns_raises(self):
        """Bypass the _record helper's auto n_bets=len(returns) so this
        exercises the empty-returns check itself, not the earlier
        n_bets < 1 check (which would otherwise fire first at n_bets=0)."""
        with pytest.raises(ValueError):
            TrialRecord(
                trial_id="t1", config_hash="abc", created_at="2026-01-01",
                mode="wf", sharpe=0.5, n_bets=1,
                returns=pd.Series([], dtype=float),
            )

    def test_n_bets_must_match_len_returns(self):
        returns = pd.Series(
            [0.01, -0.02, 0.03], index=pd.date_range("2026-01-01", periods=3)
        )
        with pytest.raises(ValueError):
            TrialRecord(
                trial_id="t1", config_hash="abc", created_at="2026-01-01",
                mode="wf", sharpe=0.5, n_bets=5, returns=returns,
            )


class TestNTrials:
    def test_empty_records_raises(self):
        with pytest.raises(ValueError):
            n_trials([])

    def test_counts_distinct_config_hashes(self):
        records = [
            _record(trial_id="t1", config_hash="hash_a"),
            _record(trial_id="t2", config_hash="hash_b"),
            _record(trial_id="t3", config_hash="hash_c"),
        ]
        assert n_trials(records) == 3

    def test_dedups_same_config_hash(self):
        records = [
            _record(trial_id="t1", config_hash="hash_a", created_at="2026-01-01"),
            _record(trial_id="t2", config_hash="hash_a", created_at="2026-01-02"),
            _record(trial_id="t3", config_hash="hash_b", created_at="2026-01-01"),
        ]
        assert n_trials(records) == 2


class TestSrVariance:
    def test_fewer_than_2_distinct_configs_raises(self):
        with pytest.raises(ValueError):
            sr_variance([_record(config_hash="only_one")])

    def test_raises_when_dedup_collapses_below_2(self):
        """Two records, same config_hash -- dedup leaves only 1 distinct
        config, still insufficient for a variance."""
        records = [
            _record(trial_id="t1", config_hash="hash_a", created_at="2026-01-01", sharpe=1.0),
            _record(trial_id="t2", config_hash="hash_a", created_at="2026-01-02", sharpe=2.0),
        ]
        with pytest.raises(ValueError):
            sr_variance(records)

    def test_known_variance_value(self):
        """Sharpes [1.0, 2.0, 3.0] -> sample variance (ddof=1) == 1.0
        exactly (mean=2, squared deviations [1,0,1], sum=2, /(3-1))."""
        records = [
            _record(trial_id="t1", config_hash="hash_a", sharpe=1.0),
            _record(trial_id="t2", config_hash="hash_b", sharpe=2.0),
            _record(trial_id="t3", config_hash="hash_c", sharpe=3.0),
        ]
        assert sr_variance(records) == pytest.approx(1.0)

    def test_uses_latest_sharpe_per_deduped_config(self):
        """When a config is re-run, the variance must reflect the LATEST
        trial's Sharpe, not the earlier (superseded) one."""
        records = [
            _record(trial_id="t1", config_hash="hash_a", created_at="2026-01-01", sharpe=100.0),
            _record(trial_id="t2", config_hash="hash_a", created_at="2026-01-02", sharpe=1.0),
            _record(trial_id="t3", config_hash="hash_b", sharpe=2.0),
            _record(trial_id="t4", config_hash="hash_c", sharpe=3.0),
        ]
        # after dedup: hash_a -> sharpe=1.0 (latest), hash_b -> 2.0, hash_c -> 3.0
        assert sr_variance(records) == pytest.approx(1.0)


class TestAssembleMatrix:
    def test_fewer_than_2_distinct_configs_raises(self):
        with pytest.raises(ValueError):
            assemble_matrix([_record(config_hash="only_one")], min_overlap=1)

    def test_empty_records_raises(self):
        with pytest.raises(ValueError):
            assemble_matrix([], min_overlap=1)

    def test_exact_inner_join_alignment(self):
        """Two trials with partially overlapping t0 timelines -- the
        assembled matrix must contain exactly the intersection, with the
        exact aligned values."""
        dates_a = pd.date_range("2026-01-01", periods=5, freq="D")
        dates_b = pd.date_range("2026-01-03", periods=5, freq="D")  # overlap: Jan 3-5
        rec_a = _record(
            trial_id="ta", config_hash="hash_a",
            returns=pd.Series([1, 2, 3, 4, 5], index=dates_a, dtype=float),
        )
        rec_b = _record(
            trial_id="tb", config_hash="hash_b",
            returns=pd.Series([10, 20, 30, 40, 50], index=dates_b, dtype=float),
        )
        matrix = assemble_matrix([rec_a, rec_b], min_overlap=1)

        assert matrix.shape == (3, 2)
        assert set(matrix.columns) == {"hash_a", "hash_b"}
        expected_dates = pd.date_range("2026-01-03", periods=3, freq="D")
        assert list(matrix.index) == list(expected_dates)
        np.testing.assert_array_equal(matrix["hash_a"].to_numpy(), [3, 4, 5])
        np.testing.assert_array_equal(matrix["hash_b"].to_numpy(), [10, 20, 30])

    def test_insufficient_overlap_raises(self):
        dates_a = pd.date_range("2026-01-01", periods=5, freq="D")
        dates_b = pd.date_range("2026-01-05", periods=5, freq="D")  # overlap: 1 day only
        rec_a = _record(
            trial_id="ta", config_hash="hash_a",
            returns=pd.Series([1, 2, 3, 4, 5], index=dates_a, dtype=float),
        )
        rec_b = _record(
            trial_id="tb", config_hash="hash_b",
            returns=pd.Series([10, 20, 30, 40, 50], index=dates_b, dtype=float),
        )
        with pytest.raises(ValueError):
            assemble_matrix([rec_a, rec_b], min_overlap=4)

    def test_dedup_applied_before_assembly(self):
        """A superseded (same-config, earlier) trial's returns must not
        appear in the assembled matrix -- only the latest survives."""
        dates = pd.date_range("2026-01-01", periods=4, freq="D")
        old = _record(
            trial_id="old", config_hash="hash_a", created_at="2026-01-01",
            returns=pd.Series([999, 999, 999, 999], index=dates, dtype=float),
        )
        new = _record(
            trial_id="new", config_hash="hash_a", created_at="2026-06-01",
            returns=pd.Series([1, 2, 3, 4], index=dates, dtype=float),
        )
        other = _record(
            trial_id="other", config_hash="hash_b",
            returns=pd.Series([10, 20, 30, 40], index=dates, dtype=float),
        )
        matrix = assemble_matrix([old, new, other], min_overlap=1)
        assert matrix.shape == (4, 2)
        np.testing.assert_array_equal(matrix["hash_a"].to_numpy(), [1, 2, 3, 4])
