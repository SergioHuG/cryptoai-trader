"""Acceptance tests for WeightConfig (research/weights, Step 1).

WeightConfig is the recipe identity for the AFML Ch.4 sample-weight build:
decay_c (AFML's clfLastW) + schema_version. The parent label-config hash is
deliberately NOT a field here -- same principle as LabelConfig's
symbol/threshold exclusion: it defines *where* (which label set) the recipe
is applied, not *what recipe* it is (Q7/Q8).
"""
import pytest

from research.weights.config import WeightConfig


def _valid_kwargs(**overrides):
    base = dict(decay_c=0.5)
    base.update(overrides)
    return base


class TestWeightConfigConstruction:
    def test_valid_config_constructs(self):
        cfg = WeightConfig(**_valid_kwargs())
        assert cfg.decay_c == 0.5

    def test_schema_version_defaults_to_1(self):
        cfg = WeightConfig(**_valid_kwargs())
        assert cfg.schema_version == 1

    def test_schema_version_can_be_set_explicitly(self):
        cfg = WeightConfig(**_valid_kwargs(), schema_version=2)
        assert cfg.schema_version == 2

    def test_is_frozen(self):
        cfg = WeightConfig(**_valid_kwargs())
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            cfg.decay_c = 0.9

    def test_decay_c_is_required(self):
        with pytest.raises(TypeError):
            WeightConfig()  # no default on decay_c

    def test_decay_c_one_is_valid(self):
        """c == 1 disables decay entirely (AFML clfLastW upper bound)."""
        cfg = WeightConfig(decay_c=1.0)
        assert cfg.decay_c == 1.0

    def test_decay_c_zero_is_valid(self):
        """c == 0 decays the oldest event's weight to zero."""
        cfg = WeightConfig(decay_c=0.0)
        assert cfg.decay_c == 0.0

    def test_decay_c_negative_is_valid(self):
        """c < 0 is the legitimate truncation regime -- no lower bound."""
        cfg = WeightConfig(decay_c=-0.5)
        assert cfg.decay_c == -0.5

    def test_label_config_hash_is_not_a_field(self):
        """The parent label recipe defines storage path, not weight identity."""
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(WeightConfig)}
        assert "label_config_hash" not in field_names
        assert "label_config" not in field_names


class TestWeightConfigValidation:
    def test_decay_c_above_one_raises(self):
        with pytest.raises(ValueError):
            WeightConfig(decay_c=1.5)

    def test_schema_version_must_be_at_least_1(self):
        with pytest.raises(ValueError):
            WeightConfig(**_valid_kwargs(), schema_version=0)
        with pytest.raises(ValueError):
            WeightConfig(**_valid_kwargs(), schema_version=-1)


class TestWeightConfigHash:
    def test_hash_is_deterministic(self):
        cfg_a = WeightConfig(**_valid_kwargs())
        cfg_b = WeightConfig(**_valid_kwargs())
        assert cfg_a.config_hash() == cfg_b.config_hash()

    def test_hash_is_a_12_char_hex_string(self):
        cfg = WeightConfig(**_valid_kwargs())
        h = cfg.config_hash()
        assert isinstance(h, str)
        assert len(h) == 12
        int(h, 16)  # raises ValueError if not valid hex

    def test_hash_changes_when_decay_c_changes(self):
        base = WeightConfig(**_valid_kwargs())
        variant = WeightConfig(**_valid_kwargs(decay_c=-0.5))
        assert base.config_hash() != variant.config_hash()

    def test_schema_version_participates_in_hash(self):
        """schema_version lives inside the hash -- protects against silent
        collisions when weight semantics change (e.g. a future
        uniqueness-only weight_scheme or exponential decay_kind) without
        decay_c itself changing."""
        cfg_v1 = WeightConfig(**_valid_kwargs(), schema_version=1)
        cfg_v2 = WeightConfig(**_valid_kwargs(), schema_version=2)
        assert cfg_v1.config_hash() != cfg_v2.config_hash()

    def test_hash_is_stable_across_field_order_in_kwargs(self):
        cfg_a = WeightConfig(decay_c=0.5, schema_version=1)
        cfg_b = WeightConfig(schema_version=1, decay_c=0.5)
        assert cfg_a.config_hash() == cfg_b.config_hash()
