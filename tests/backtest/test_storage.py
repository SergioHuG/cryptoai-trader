"""Acceptance tests for backtest.storage (Phase 2, Task 6).

Every test runs against a pytest ``tmp_path`` .h5 file -- never the real
``backtest/results/backtest.h5``.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.ledger import TrialRecord, n_trials, sr_variance
from backtest.storage import append_trial, load_ledger, load_result, save_result


def _record(trial_id, config_hash, created_at="2026-01-01", sharpe=0.5, mode="cpcv"):
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    returns = pd.Series([0.01, -0.02, 0.03, 0.01], index=dates)
    return TrialRecord(
        trial_id=trial_id, config_hash=config_hash, created_at=created_at,
        mode=mode, sharpe=sharpe, n_bets=len(returns), returns=returns,
    )


class TestAppendAndLoadLedger:
    def test_round_trip_single_trial(self, tmp_path):
        store_path = tmp_path / "backtest.h5"
        rec = _record("t1", "hash_a")
        append_trial(rec, store_path)

        loaded = load_ledger(store_path)
        assert len(loaded) == 1
        got = loaded[0]
        assert got.trial_id == "t1"
        assert got.config_hash == "hash_a"
        assert got.created_at == pd.Timestamp("2026-01-01")
        assert got.mode == "cpcv"
        assert got.sharpe == pytest.approx(0.5)
        assert got.n_bets == 4
        pd.testing.assert_series_equal(got.returns, rec.returns, check_names=False)

    def test_round_trip_multiple_trials(self, tmp_path):
        store_path = tmp_path / "backtest.h5"
        recs = [
            _record("t1", "hash_a", sharpe=0.1),
            _record("t2", "hash_b", sharpe=0.2),
            _record("t3", "hash_c", sharpe=0.3),
        ]
        for r in recs:
            append_trial(r, store_path)

        loaded = load_ledger(store_path)
        assert len(loaded) == 3
        loaded_by_id = {r.trial_id: r for r in loaded}
        assert set(loaded_by_id) == {"t1", "t2", "t3"}
        assert loaded_by_id["t2"].config_hash == "hash_b"
        assert loaded_by_id["t3"].sharpe == pytest.approx(0.3)

    def test_duplicate_trial_id_raises(self, tmp_path):
        store_path = tmp_path / "backtest.h5"
        rec = _record("t1", "hash_a")
        append_trial(rec, store_path)
        with pytest.raises(KeyError):
            append_trial(rec, store_path)

    def test_load_nonexistent_store_raises_file_not_found(self, tmp_path):
        store_path = tmp_path / "does_not_exist.h5"
        with pytest.raises(FileNotFoundError):
            load_ledger(store_path)

    def test_load_existing_store_with_no_trials_raises_key_error(self, tmp_path):
        """A store can exist (e.g. from save_result) with zero ledger
        entries -- that must still fail loud, not silently return []."""
        store_path = tmp_path / "backtest.h5"
        paths = pd.DataFrame({0: [1.0, 2.0], 1: [3.0, 4.0]})
        save_result(paths, "hash_x", "2026-01-01", {}, store_path)
        with pytest.raises(KeyError):
            load_ledger(store_path)

    def test_dedup_persists_across_storage_round_trip(self, tmp_path):
        """Two trials sharing a config_hash, appended at different times
        -- after a full storage round-trip, ledger.py's dedup reductions
        must still correctly resolve to the latest by created_at."""
        store_path = tmp_path / "backtest.h5"
        old = _record("old", "hash_a", created_at="2026-01-01", sharpe=100.0)
        new = _record("new", "hash_a", created_at="2026-06-01", sharpe=1.0)
        other = _record("other", "hash_b", created_at="2026-01-01", sharpe=2.0)
        for r in (old, new, other):
            append_trial(r, store_path)

        loaded = load_ledger(store_path)
        assert len(loaded) == 3  # full physical record, undeduped
        assert n_trials(loaded) == 2  # deduped by config_hash
        assert sr_variance(loaded) == pytest.approx(np.var([1.0, 2.0], ddof=1))


class TestSaveAndLoadResult:
    def test_round_trip_exact_values(self, tmp_path):
        store_path = tmp_path / "backtest.h5"
        paths = pd.DataFrame({0: [0.01, 0.02], 1: [0.03, 0.04], 2: [0.05, 0.06]})
        metadata = {"psr_gate": "pass", "dsr_gate": "provisional", "hhi_positive": 0.3}
        save_result(paths, "cfg_hash_123", "2026-03-15", metadata, store_path)

        loaded_paths, info = load_result("cfg_hash_123", store_path)
        pd.testing.assert_frame_equal(loaded_paths, paths)
        assert info["config_hash"] == "cfg_hash_123"
        assert info["created_at"] == pd.Timestamp("2026-03-15")
        assert info["psr_gate"] == "pass"
        assert info["dsr_gate"] == "provisional"
        assert info["hhi_positive"] == pytest.approx(0.3)

    def test_integer_columns_survive_round_trip(self, tmp_path):
        """Guards the pytables table-format + integer-column-label
        incompatibility discovered during this task: paths.py's
        reconstruct_paths output (int path_id columns) must round-trip
        with exact column labels and dtypes intact."""
        store_path = tmp_path / "backtest.h5"
        paths = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]})
        save_result(paths, "hash_int_cols", "2026-01-01", {}, store_path)
        loaded_paths, _ = load_result("hash_int_cols", store_path)
        assert list(loaded_paths.columns) == [0, 1]
        assert all(isinstance(c, int) for c in loaded_paths.columns)

    def test_overwrite_true_replaces_existing(self, tmp_path):
        store_path = tmp_path / "backtest.h5"
        paths_v1 = pd.DataFrame({0: [1.0, 2.0]})
        paths_v2 = pd.DataFrame({0: [9.0, 9.0]})
        save_result(paths_v1, "hash_a", "2026-01-01", {"v": 1}, store_path)
        save_result(paths_v2, "hash_a", "2026-06-01", {"v": 2}, store_path)

        loaded_paths, info = load_result("hash_a", store_path)
        pd.testing.assert_frame_equal(loaded_paths, paths_v2)
        assert info["v"] == 2

    def test_overwrite_false_raises_if_exists(self, tmp_path):
        store_path = tmp_path / "backtest.h5"
        paths = pd.DataFrame({0: [1.0, 2.0]})
        save_result(paths, "hash_a", "2026-01-01", {}, store_path)
        with pytest.raises(KeyError):
            save_result(paths, "hash_a", "2026-01-01", {}, store_path, overwrite=False)

    def test_load_nonexistent_store_raises_file_not_found(self, tmp_path):
        store_path = tmp_path / "does_not_exist.h5"
        with pytest.raises(FileNotFoundError):
            load_result("any_hash", store_path)

    def test_load_missing_config_hash_raises_key_error(self, tmp_path):
        store_path = tmp_path / "backtest.h5"
        paths = pd.DataFrame({0: [1.0, 2.0]})
        save_result(paths, "hash_a", "2026-01-01", {}, store_path)
        with pytest.raises(KeyError):
            load_result("hash_b_never_stored", store_path)


class TestLedgerAndResultsCoexistence:
    def test_ledger_and_results_coexist_in_same_store(self, tmp_path):
        """Both namespaces must be independently readable from the same
        physical .h5 file."""
        store_path = tmp_path / "backtest.h5"
        rec = _record("t1", "hash_a")
        append_trial(rec, store_path)
        paths = pd.DataFrame({0: [1.0, 2.0]})
        save_result(paths, "hash_a", "2026-01-01", {"gate": "pass"}, store_path)

        loaded_ledger = load_ledger(store_path)
        loaded_paths, info = load_result("hash_a", store_path)
        assert len(loaded_ledger) == 1
        assert info["gate"] == "pass"
