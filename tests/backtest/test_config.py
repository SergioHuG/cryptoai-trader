"""Acceptance tests for BacktestConfig (Phase 2, Task 1).

BacktestConfig is the recipe identity for a backtest run: the CPCV group
geometry (n_groups/n_test_groups), the walk-forward split count, the
embargo fraction, the O-U Monte Carlo knobs (n_sims/seed/burn_in), the PBO
partition count, the PSR benchmark, and the two gate thresholds
(gate_threshold for PSR/DSR, pbo_threshold for PBO) -- plus a schema_version
guarding against silent semantics collisions. Mirrors
research.labels.config.LabelConfig's frozen dataclass + config_hash() idiom
exactly (Q12).

No locators (symbol/threshold/config paths) live here -- this hash is the
ledger/result dedup key, keyed on the *recipe*, not on where it's applied.
"""
import dataclasses

import pytest

from backtest.config import BacktestConfig


def _valid_kwargs(**overrides):
    base = dict(
        n_groups=6,
        n_test_groups=2,
        n_wf_splits=5,
        embargo_pct=0.01,
        n_sims=1000,
        seed=42,
        burn_in=100,
        pbo_partitions=16,
        psr_benchmark=0.0,
        gate_threshold=0.95,
        pbo_threshold=0.5,
    )
    base.update(overrides)
    return base


class TestBacktestConfigConstruction:
    def test_valid_config_constructs(self):
        cfg = BacktestConfig(**_valid_kwargs())
        assert cfg.n_groups == 6
        assert cfg.n_test_groups == 2
        assert cfg.n_wf_splits == 5
        assert cfg.embargo_pct == 0.01
        assert cfg.n_sims == 1000
        assert cfg.seed == 42
        assert cfg.burn_in == 100
        assert cfg.pbo_partitions == 16
        assert cfg.psr_benchmark == 0.0
        assert cfg.gate_threshold == 0.95
        assert cfg.pbo_threshold == 0.5

    def test_schema_version_defaults_to_1(self):
        cfg = BacktestConfig(**_valid_kwargs())
        assert cfg.schema_version == 1

    def test_schema_version_can_be_set_explicitly(self):
        cfg = BacktestConfig(**_valid_kwargs(), schema_version=2)
        assert cfg.schema_version == 2

    def test_is_frozen(self):
        cfg = BacktestConfig(**_valid_kwargs())
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            cfg.n_groups = 99

    def test_gate_threshold_at_boundaries_is_valid(self):
        BacktestConfig(**_valid_kwargs(gate_threshold=0.0))
        BacktestConfig(**_valid_kwargs(gate_threshold=1.0))

    def test_pbo_threshold_at_boundaries_is_valid(self):
        BacktestConfig(**_valid_kwargs(pbo_threshold=0.0))
        BacktestConfig(**_valid_kwargs(pbo_threshold=1.0))

    def test_embargo_pct_zero_is_valid(self):
        """embargo_pct == 0 means no embargo -- a valid, if degenerate, recipe."""
        cfg = BacktestConfig(**_valid_kwargs(embargo_pct=0.0))
        assert cfg.embargo_pct == 0.0

    def test_burn_in_zero_is_valid(self):
        cfg = BacktestConfig(**_valid_kwargs(burn_in=0))
        assert cfg.burn_in == 0

    def test_no_locator_fields(self):
        """symbol/threshold/paths define storage location, not the recipe --
        they must never leak into this config (mirrors LabelConfig's own
        locator-exclusion contract, Q12)."""
        field_names = {f.name for f in dataclasses.fields(BacktestConfig)}
        assert "symbol" not in field_names
        assert "threshold" not in field_names
        assert "path" not in field_names


class TestBacktestConfigValidation:
    def test_n_groups_must_be_at_least_2(self):
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(n_groups=1))
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(n_groups=0))

    def test_n_test_groups_must_be_at_least_1(self):
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(n_test_groups=0))

    def test_n_test_groups_must_be_less_than_n_groups(self):
        """k < N -- CPCV's combinatorial split requires strictly fewer test
        groups than total groups."""
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(n_groups=4, n_test_groups=4))
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(n_groups=4, n_test_groups=5))

    def test_n_wf_splits_must_be_at_least_1(self):
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(n_wf_splits=0))
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(n_wf_splits=-1))

    def test_embargo_pct_must_be_in_zero_one_half_open_interval(self):
        """Mirrors get_embargo_times's own 0 <= embargo_pct < 1 contract."""
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(embargo_pct=-0.01))
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(embargo_pct=1.0))
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(embargo_pct=1.5))

    def test_n_sims_must_be_at_least_1(self):
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(n_sims=0))
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(n_sims=-1))

    def test_seed_must_be_non_negative(self):
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(seed=-1))

    def test_burn_in_must_be_non_negative(self):
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(burn_in=-1))

    def test_pbo_partitions_must_be_even(self):
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(pbo_partitions=15))
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(pbo_partitions=17))

    def test_pbo_partitions_must_be_at_least_2(self):
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(pbo_partitions=0))

    def test_gate_threshold_must_be_in_zero_one_closed_interval(self):
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(gate_threshold=-0.01))
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(gate_threshold=1.01))

    def test_pbo_threshold_must_be_in_zero_one_closed_interval(self):
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(pbo_threshold=-0.01))
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(pbo_threshold=1.01))

    def test_schema_version_must_be_at_least_1(self):
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(), schema_version=0)
        with pytest.raises(ValueError):
            BacktestConfig(**_valid_kwargs(), schema_version=-1)


class TestBacktestConfigHash:
    def test_hash_is_deterministic(self):
        cfg_a = BacktestConfig(**_valid_kwargs())
        cfg_b = BacktestConfig(**_valid_kwargs())
        assert cfg_a.config_hash() == cfg_b.config_hash()

    def test_hash_is_a_12_char_hex_string(self):
        cfg = BacktestConfig(**_valid_kwargs())
        h = cfg.config_hash()
        assert isinstance(h, str)
        assert len(h) == 12
        int(h, 16)  # raises ValueError if not valid hex

    def test_hash_changes_when_any_field_changes(self):
        base = BacktestConfig(**_valid_kwargs())
        variants = [
            BacktestConfig(**_valid_kwargs(n_groups=8)),
            BacktestConfig(**_valid_kwargs(n_test_groups=3)),
            BacktestConfig(**_valid_kwargs(n_wf_splits=10)),
            BacktestConfig(**_valid_kwargs(embargo_pct=0.02)),
            BacktestConfig(**_valid_kwargs(n_sims=2000)),
            BacktestConfig(**_valid_kwargs(seed=7)),
            BacktestConfig(**_valid_kwargs(burn_in=200)),
            BacktestConfig(**_valid_kwargs(pbo_partitions=20)),
            BacktestConfig(**_valid_kwargs(psr_benchmark=0.1)),
            BacktestConfig(**_valid_kwargs(gate_threshold=0.99)),
            BacktestConfig(**_valid_kwargs(pbo_threshold=0.2)),
            BacktestConfig(**_valid_kwargs(), schema_version=2),
        ]
        hashes = {base.config_hash()} | {v.config_hash() for v in variants}
        assert len(hashes) == len(variants) + 1  # all distinct

    def test_schema_version_participates_in_hash(self):
        """schema_version lives inside the hash -- protects against silent
        collisions when backtest semantics change without any knob changing."""
        cfg_v1 = BacktestConfig(**_valid_kwargs(), schema_version=1)
        cfg_v2 = BacktestConfig(**_valid_kwargs(), schema_version=2)
        assert cfg_v1.config_hash() != cfg_v2.config_hash()

    def test_hash_is_stable_across_field_order_in_kwargs(self):
        """json.dumps(..., sort_keys=True) means kwarg order at construction
        time must not affect the hash."""
        cfg_a = BacktestConfig(
            n_groups=6, n_test_groups=2, n_wf_splits=5, embargo_pct=0.01,
            n_sims=1000, seed=42, burn_in=100, pbo_partitions=16,
            psr_benchmark=0.0, gate_threshold=0.95, pbo_threshold=0.5,
        )
        cfg_b = BacktestConfig(
            pbo_threshold=0.5, gate_threshold=0.95, psr_benchmark=0.0,
            pbo_partitions=16, burn_in=100, seed=42, n_sims=1000,
            embargo_pct=0.01, n_wf_splits=5, n_test_groups=2, n_groups=6,
        )
        assert cfg_a.config_hash() == cfg_b.config_hash()
