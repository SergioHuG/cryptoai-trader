"""Acceptance tests for LabelConfig (Task 8a).

LabelConfig is the recipe identity for a triple-barrier labeling run:
kappa/ewma_span/pt_mult/sl_mult/max_hp/min_ret/schema_version. symbol and
threshold are deliberately NOT fields here -- they define the storage path,
not the recipe (Decision: "LabelConfig is the recipe identity, not the
partition key").
"""
import pytest

from research.labels.config import LabelConfig


def _valid_kwargs(**overrides):
    base = dict(
        kappa=2.0,
        ewma_span=20,
        pt_mult=1.0,
        sl_mult=1.0,
        max_hp=24,
        min_ret=0.0,
    )
    base.update(overrides)
    return base


class TestLabelConfigConstruction:
    def test_valid_config_constructs(self):
        cfg = LabelConfig(**_valid_kwargs())
        assert cfg.kappa == 2.0
        assert cfg.ewma_span == 20
        assert cfg.pt_mult == 1.0
        assert cfg.sl_mult == 1.0
        assert cfg.max_hp == 24
        assert cfg.min_ret == 0.0

    def test_schema_version_defaults_to_1(self):
        cfg = LabelConfig(**_valid_kwargs())
        assert cfg.schema_version == 1

    def test_schema_version_can_be_set_explicitly(self):
        cfg = LabelConfig(**_valid_kwargs(), schema_version=2)
        assert cfg.schema_version == 2

    def test_is_frozen(self):
        cfg = LabelConfig(**_valid_kwargs())
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            cfg.kappa = 99.0

    def test_vertical_only_config_is_valid(self):
        """Both multipliers == 0 is valid -- vertical-only labeling."""
        cfg = LabelConfig(**_valid_kwargs(pt_mult=0.0, sl_mult=0.0))
        assert cfg.pt_mult == 0.0
        assert cfg.sl_mult == 0.0

    def test_symbol_and_threshold_are_not_fields(self):
        """symbol/threshold define the storage path, not the recipe identity."""
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(LabelConfig)}
        assert "symbol" not in field_names
        assert "threshold" not in field_names


class TestLabelConfigValidation:
    def test_kappa_must_be_positive(self):
        with pytest.raises(ValueError):
            LabelConfig(**_valid_kwargs(kappa=0.0))
        with pytest.raises(ValueError):
            LabelConfig(**_valid_kwargs(kappa=-1.0))

    def test_ewma_span_must_be_at_least_2(self):
        with pytest.raises(ValueError):
            LabelConfig(**_valid_kwargs(ewma_span=1))
        with pytest.raises(ValueError):
            LabelConfig(**_valid_kwargs(ewma_span=0))

    def test_pt_mult_must_be_non_negative(self):
        with pytest.raises(ValueError):
            LabelConfig(**_valid_kwargs(pt_mult=-0.5))

    def test_sl_mult_must_be_non_negative(self):
        with pytest.raises(ValueError):
            LabelConfig(**_valid_kwargs(sl_mult=-0.5))

    def test_max_hp_must_be_at_least_1(self):
        with pytest.raises(ValueError):
            LabelConfig(**_valid_kwargs(max_hp=0))
        with pytest.raises(ValueError):
            LabelConfig(**_valid_kwargs(max_hp=-1))

    def test_min_ret_must_be_non_negative(self):
        with pytest.raises(ValueError):
            LabelConfig(**_valid_kwargs(min_ret=-0.01))

    def test_schema_version_must_be_at_least_1(self):
        with pytest.raises(ValueError):
            LabelConfig(**_valid_kwargs(), schema_version=0)
        with pytest.raises(ValueError):
            LabelConfig(**_valid_kwargs(), schema_version=-1)


class TestLabelConfigHash:
    def test_hash_is_deterministic(self):
        cfg_a = LabelConfig(**_valid_kwargs())
        cfg_b = LabelConfig(**_valid_kwargs())
        assert cfg_a.config_hash() == cfg_b.config_hash()

    def test_hash_is_a_12_char_hex_string(self):
        cfg = LabelConfig(**_valid_kwargs())
        h = cfg.config_hash()
        assert isinstance(h, str)
        assert len(h) == 12
        int(h, 16)  # raises ValueError if not valid hex

    def test_hash_changes_when_any_field_changes(self):
        base = LabelConfig(**_valid_kwargs())
        variants = [
            LabelConfig(**_valid_kwargs(kappa=3.0)),
            LabelConfig(**_valid_kwargs(ewma_span=30)),
            LabelConfig(**_valid_kwargs(pt_mult=2.0)),
            LabelConfig(**_valid_kwargs(sl_mult=2.0)),
            LabelConfig(**_valid_kwargs(max_hp=48)),
            LabelConfig(**_valid_kwargs(min_ret=0.001)),
            LabelConfig(**_valid_kwargs(), schema_version=2),
        ]
        hashes = {base.config_hash()} | {v.config_hash() for v in variants}
        assert len(hashes) == len(variants) + 1  # all distinct

    def test_schema_version_participates_in_hash(self):
        """schema_version lives inside the hash -- protects against silent
        collisions when label semantics change without any knob changing."""
        cfg_v1 = LabelConfig(**_valid_kwargs(), schema_version=1)
        cfg_v2 = LabelConfig(**_valid_kwargs(), schema_version=2)
        assert cfg_v1.config_hash() != cfg_v2.config_hash()

    def test_hash_is_independent_of_symbol_and_threshold(self):
        """The same recipe must hash identically regardless of what
        symbol/threshold it's later applied to -- those aren't config fields
        at all, so this is really just reaffirming there's nothing to vary."""
        cfg_a = LabelConfig(**_valid_kwargs())
        cfg_b = LabelConfig(**_valid_kwargs())
        assert cfg_a.config_hash() == cfg_b.config_hash()

    def test_hash_is_stable_across_field_order_in_kwargs(self):
        """json.dumps(..., sort_keys=True) means kwarg order at construction
        time must not affect the hash."""
        cfg_a = LabelConfig(
            kappa=2.0, ewma_span=20, pt_mult=1.0, sl_mult=1.0,
            max_hp=24, min_ret=0.0,
        )
        cfg_b = LabelConfig(
            min_ret=0.0, max_hp=24, sl_mult=1.0, pt_mult=1.0,
            ewma_span=20, kappa=2.0,
        )
        assert cfg_a.config_hash() == cfg_b.config_hash()
