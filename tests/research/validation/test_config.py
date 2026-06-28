"""Acceptance tests for ValidationConfig (research/validation, Step 1).

ValidationConfig is the frozen, validated recipe for the AFML Ch.7 / Ch.12
cross-validation layer: n_splits + embargo_pct (PurgedKFold) and n_groups +
n_test_groups (CombinatorialPurgedKFold), with embargo_pct shared between the
two splitters.

Deliberately UNLIKE LabelConfig / WeightConfig, it carries NO config_hash()
and NO schema_version: CV output is transient (no storage.py this
sub-package, Q2), so there is nothing for a hash to key and no reload path
for a schema version to guard. The frozen+validated half of the house
template is kept (it catches config errors at construction); the identity
half is dropped until/unless persistence is un-parked (Q9).
"""
import dataclasses

import pytest

from research.validation.config import ValidationConfig


def _valid_kwargs(**overrides):
    base = dict(n_splits=5, embargo_pct=0.01, n_groups=6, n_test_groups=2)
    base.update(overrides)
    return base


class TestValidationConfigConstruction:
    def test_valid_config_constructs(self):
        cfg = ValidationConfig(**_valid_kwargs())
        assert cfg.n_splits == 5
        assert cfg.embargo_pct == 0.01
        assert cfg.n_groups == 6
        assert cfg.n_test_groups == 2

    def test_n_test_groups_defaults_to_2(self):
        cfg = ValidationConfig(n_splits=5, embargo_pct=0.01, n_groups=6)
        assert cfg.n_test_groups == 2

    def test_is_frozen(self):
        cfg = ValidationConfig(**_valid_kwargs())
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            cfg.n_splits = 10

    def test_n_splits_is_required(self):
        with pytest.raises(TypeError):
            ValidationConfig(embargo_pct=0.01, n_groups=6)

    def test_embargo_pct_is_required(self):
        with pytest.raises(TypeError):
            ValidationConfig(n_splits=5, n_groups=6)

    def test_n_groups_is_required(self):
        with pytest.raises(TypeError):
            ValidationConfig(n_splits=5, embargo_pct=0.01)


class TestNSplitsValidation:
    def test_n_splits_two_is_valid(self):
        cfg = ValidationConfig(**_valid_kwargs(n_splits=2))
        assert cfg.n_splits == 2

    def test_n_splits_one_raises(self):
        with pytest.raises(ValueError):
            ValidationConfig(**_valid_kwargs(n_splits=1))

    def test_n_splits_zero_raises(self):
        with pytest.raises(ValueError):
            ValidationConfig(**_valid_kwargs(n_splits=0))

    def test_n_splits_negative_raises(self):
        with pytest.raises(ValueError):
            ValidationConfig(**_valid_kwargs(n_splits=-1))


class TestEmbargoPctValidation:
    def test_embargo_pct_zero_is_valid(self):
        """embargo_pct == 0 is the no-embargo (purge-only) regime (Q4 edge)."""
        cfg = ValidationConfig(**_valid_kwargs(embargo_pct=0.0))
        assert cfg.embargo_pct == 0.0

    def test_embargo_pct_near_one_is_valid(self):
        cfg = ValidationConfig(**_valid_kwargs(embargo_pct=0.999))
        assert cfg.embargo_pct == 0.999

    def test_embargo_pct_negative_raises(self):
        with pytest.raises(ValueError):
            ValidationConfig(**_valid_kwargs(embargo_pct=-0.001))

    def test_embargo_pct_one_raises(self):
        """embargo_pct == 1 would embargo the entire dataset."""
        with pytest.raises(ValueError):
            ValidationConfig(**_valid_kwargs(embargo_pct=1.0))

    def test_embargo_pct_above_one_raises(self):
        with pytest.raises(ValueError):
            ValidationConfig(**_valid_kwargs(embargo_pct=1.5))


class TestNGroupsValidation:
    def test_n_groups_two_is_valid(self):
        cfg = ValidationConfig(**_valid_kwargs(n_groups=2, n_test_groups=1))
        assert cfg.n_groups == 2

    def test_n_groups_one_raises(self):
        with pytest.raises(ValueError):
            ValidationConfig(**_valid_kwargs(n_groups=1, n_test_groups=1))

    def test_n_groups_zero_raises(self):
        with pytest.raises(ValueError):
            ValidationConfig(**_valid_kwargs(n_groups=0, n_test_groups=1))


class TestNTestGroupsValidation:
    def test_n_test_groups_one_is_valid(self):
        cfg = ValidationConfig(**_valid_kwargs(n_groups=6, n_test_groups=1))
        assert cfg.n_test_groups == 1

    def test_n_test_groups_just_below_n_groups_is_valid(self):
        cfg = ValidationConfig(**_valid_kwargs(n_groups=6, n_test_groups=5))
        assert cfg.n_test_groups == 5

    def test_n_test_groups_zero_raises(self):
        with pytest.raises(ValueError):
            ValidationConfig(**_valid_kwargs(n_groups=6, n_test_groups=0))

    def test_n_test_groups_equal_to_n_groups_raises(self):
        """k == N leaves no train group -- the cross-field rule (Q9)."""
        with pytest.raises(ValueError):
            ValidationConfig(**_valid_kwargs(n_groups=6, n_test_groups=6))

    def test_n_test_groups_above_n_groups_raises(self):
        with pytest.raises(ValueError):
            ValidationConfig(**_valid_kwargs(n_groups=6, n_test_groups=7))


class TestIdentityMachineryDeliberatelyAbsent:
    """Q9: CV output is transient -- no hash to key, no reload path to guard."""

    def test_has_no_config_hash_method(self):
        cfg = ValidationConfig(**_valid_kwargs())
        assert not hasattr(cfg, "config_hash")

    def test_has_no_schema_version_field(self):
        field_names = {f.name for f in dataclasses.fields(ValidationConfig)}
        assert "schema_version" not in field_names

    def test_exactly_four_fields(self):
        field_names = {f.name for f in dataclasses.fields(ValidationConfig)}
        assert field_names == {
            "n_splits",
            "embargo_pct",
            "n_groups",
            "n_test_groups",
        }
