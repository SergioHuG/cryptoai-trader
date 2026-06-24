"""Acceptance tests for weight storage (Step 7, Q8).

store_weights/load_weights/list_weight_configs persist sample-weight
frames to the *same* .h5 file as bars and labels, nested one level deeper
than labels to encode provenance directly in the key:

    /{SYM_KEY}/thr_{N}/weights/lbl_{label_hash}/cfg_{weight_hash}

store_weights takes the LabelConfig OBJECT (not a bare hash string) --
mirrors store_labels taking LabelConfig (Q8b). A stored weight set is
fully self-describing: attrs record BOTH configs plus label_config_hash
as a queryable provenance link (Q8c). load_weights is fail-loud, same
posture as load_labels.
"""
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from research.labels.config import LabelConfig
from research.labels.storage import store_labels
from research.weights.config import WeightConfig
from research.weights.storage import (
    list_weight_configs,
    load_weights,
    store_weights,
)


def _label_config(**overrides) -> LabelConfig:
    base = dict(
        kappa=2.0, ewma_span=20, pt_mult=1.0, sl_mult=1.0, max_hp=24, min_ret=0.0,
    )
    base.update(overrides)
    return LabelConfig(**base)


def _weight_config(**overrides) -> WeightConfig:
    base = dict(decay_c=0.5)
    base.update(overrides)
    return WeightConfig(**base)


def _labels_frame(n: int = 3) -> pd.DataFrame:
    """A synthetic frame shaped like get_bins()'s symmetric-mode output."""
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    t1 = pd.date_range("2024-01-01 06:00", periods=n, freq="15min", tz="UTC")
    cycle_ret = [0.01, -0.02, 0.0]
    cycle_bin = [1, -1, 0]
    return pd.DataFrame(
        {
            "ret": [cycle_ret[i % 3] for i in range(n)],
            "bin": pd.array([cycle_bin[i % 3] for i in range(n)], dtype="int8"),
            "t1": t1,
            "barrier": pd.array([cycle_bin[i % 3] for i in range(n)], dtype="int8"),
        },
        index=idx,
    )


def _weights_frame(n: int = 3) -> pd.DataFrame:
    """A synthetic frame shaped like build_sample_weights()'s output."""
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    t1 = pd.date_range("2024-01-01 06:00", periods=n, freq="15min", tz="UTC")
    cycle_u = [0.5, 0.7, 1.0]
    cycle_w = [1.2, 0.8, 1.0]
    return pd.DataFrame(
        {
            "avg_uniqueness": [cycle_u[i % 3] for i in range(n)],
            "weight": [cycle_w[i % 3] for i in range(n)],
            "t1": t1,
        },
        index=idx,
    )


class TestStoreAndLoadRoundTrip:
    def test_round_trips_values_and_columns(self, tmp_path: Path):
        store_path = tmp_path / "weights.h5"
        label_config = _label_config()
        weight_config = _weight_config()
        weights = _weights_frame()

        store_weights(
            weights, weight_config, label_config, "BTC/USD", Decimal("1000000"),
            store_path,
        )
        loaded = load_weights(
            weight_config, label_config, "BTC/USD", Decimal("1000000"), store_path
        )

        pd.testing.assert_frame_equal(loaded, weights, check_freq=False)

    def test_t1_column_round_trips_tz_aware(self, tmp_path: Path):
        store_path = tmp_path / "weights.h5"
        label_config = _label_config()
        weight_config = _weight_config()
        weights = _weights_frame()

        store_weights(
            weights, weight_config, label_config, "BTC/USD", Decimal("1000000"),
            store_path,
        )
        loaded = load_weights(
            weight_config, label_config, "BTC/USD", Decimal("1000000"), store_path
        )

        assert isinstance(loaded["t1"].dtype, pd.DatetimeTZDtype)
        assert str(loaded["t1"].dtype.tz) == "UTC"

    def test_different_label_configs_do_not_collide(self, tmp_path: Path):
        store_path = tmp_path / "weights.h5"
        weight_config = _weight_config()
        label_a = _label_config(kappa=2.0)
        label_b = _label_config(kappa=3.0)

        store_weights(
            _weights_frame(3), weight_config, label_a, "BTC/USD",
            Decimal("1000000"), store_path,
        )
        store_weights(
            _weights_frame(5), weight_config, label_b, "BTC/USD",
            Decimal("1000000"), store_path,
        )

        loaded_a = load_weights(
            weight_config, label_a, "BTC/USD", Decimal("1000000"), store_path
        )
        loaded_b = load_weights(
            weight_config, label_b, "BTC/USD", Decimal("1000000"), store_path
        )
        assert len(loaded_a) == 3
        assert len(loaded_b) == 5

    def test_different_weight_configs_under_same_label_do_not_collide(
        self, tmp_path: Path
    ):
        store_path = tmp_path / "weights.h5"
        label_config = _label_config()
        weight_a = _weight_config(decay_c=0.5)
        weight_b = _weight_config(decay_c=-0.5)

        store_weights(
            _weights_frame(3), weight_a, label_config, "BTC/USD",
            Decimal("1000000"), store_path,
        )
        store_weights(
            _weights_frame(4), weight_b, label_config, "BTC/USD",
            Decimal("1000000"), store_path,
        )

        loaded_a = load_weights(
            weight_a, label_config, "BTC/USD", Decimal("1000000"), store_path
        )
        loaded_b = load_weights(
            weight_b, label_config, "BTC/USD", Decimal("1000000"), store_path
        )
        assert len(loaded_a) == 3
        assert len(loaded_b) == 4


class TestKeyShapeAndProvenance:
    def test_key_matches_locked_shape_exactly(self, tmp_path: Path):
        store_path = tmp_path / "weights.h5"
        label_config = _label_config()
        weight_config = _weight_config()
        store_weights(
            _weights_frame(), weight_config, label_config, "BTC/USD",
            Decimal("1000000"), store_path,
        )

        expected_key = (
            f"/BTC_USD/thr_1000000/weights/"
            f"lbl_{label_config.config_hash()}/cfg_{weight_config.config_hash()}"
        )
        with pd.HDFStore(store_path, mode="r") as store:
            assert expected_key in store.keys()

    def test_store_signature_takes_label_config_parameter(self):
        import inspect

        params = list(inspect.signature(store_weights).parameters)
        assert "label_config" in params
        assert "weight_config" in params

    def test_attrs_record_both_configs_and_provenance_link(self, tmp_path: Path):
        store_path = tmp_path / "weights.h5"
        label_config = _label_config()
        weight_config = _weight_config()
        store_weights(
            _weights_frame(3), weight_config, label_config, "BTC/USD",
            Decimal("1000000"), store_path,
        )

        key = (
            f"/BTC_USD/thr_1000000/weights/"
            f"lbl_{label_config.config_hash()}/cfg_{weight_config.config_hash()}"
        )
        with pd.HDFStore(store_path, mode="r") as store:
            attrs = store.get_storer(key).attrs

            assert attrs["weight_config"]["decay_c"] == 0.5
            assert attrs["label_config"]["kappa"] == 2.0
            assert attrs["label_config_hash"] == label_config.config_hash()
            assert attrs["symbol"] == "BTC/USD"
            assert attrs["threshold"] == Decimal("1000000")
            assert attrs["n_weights"] == 3


class TestStoreOverwriteSemantics:
    def test_overwrite_true_by_default_replaces_existing(self, tmp_path: Path):
        store_path = tmp_path / "weights.h5"
        label_config = _label_config()
        weight_config = _weight_config()

        store_weights(
            _weights_frame(3), weight_config, label_config, "BTC/USD",
            Decimal("1000000"), store_path,
        )
        store_weights(
            _weights_frame(2), weight_config, label_config, "BTC/USD",
            Decimal("1000000"), store_path,
        )

        loaded = load_weights(
            weight_config, label_config, "BTC/USD", Decimal("1000000"), store_path
        )
        assert len(loaded) == 2

    def test_overwrite_false_raises_if_key_exists(self, tmp_path: Path):
        store_path = tmp_path / "weights.h5"
        label_config = _label_config()
        weight_config = _weight_config()
        store_weights(
            _weights_frame(3), weight_config, label_config, "BTC/USD",
            Decimal("1000000"), store_path,
        )

        with pytest.raises(KeyError):
            store_weights(
                _weights_frame(2), weight_config, label_config, "BTC/USD",
                Decimal("1000000"), store_path, overwrite=False,
            )


class TestLoadWeightsFailLoud:
    def test_missing_store_file_raises_file_not_found(self, tmp_path: Path):
        store_path = tmp_path / "does_not_exist.h5"
        with pytest.raises(FileNotFoundError):
            load_weights(
                _weight_config(), _label_config(), "BTC/USD",
                Decimal("1000000"), store_path,
            )

    def test_unstored_weight_config_raises_key_error(self, tmp_path: Path):
        store_path = tmp_path / "weights.h5"
        label_config = _label_config()
        stored_weight = _weight_config(decay_c=0.5)
        never_stored_weight = _weight_config(decay_c=-0.9)
        store_weights(
            _weights_frame(3), stored_weight, label_config, "BTC/USD",
            Decimal("1000000"), store_path,
        )

        with pytest.raises(KeyError):
            load_weights(
                never_stored_weight, label_config, "BTC/USD",
                Decimal("1000000"), store_path,
            )

    def test_same_weight_config_under_different_label_config_is_a_distinct_key(
        self, tmp_path: Path
    ):
        """The parent label hash genuinely partitions storage: storing under
        label_a does not make the (same) weight_config loadable under
        label_b -- proving Q8's provenance dimension is load-bearing, not
        cosmetic."""
        store_path = tmp_path / "weights.h5"
        weight_config = _weight_config()
        label_a = _label_config(kappa=2.0)
        label_b = _label_config(kappa=3.0)
        store_weights(
            _weights_frame(3), weight_config, label_a, "BTC/USD",
            Decimal("1000000"), store_path,
        )

        with pytest.raises(KeyError):
            load_weights(
                weight_config, label_b, "BTC/USD", Decimal("1000000"), store_path
            )


class TestListWeightConfigs:
    def test_returns_empty_frame_when_file_does_not_exist(self, tmp_path: Path):
        store_path = tmp_path / "does_not_exist.h5"
        df = list_weight_configs("BTC/USD", Decimal("1000000"), store_path)
        assert df.empty
        assert df.index.name == "weight_config_hash"

    def test_lists_recipes_across_label_sets_with_label_hash_column(
        self, tmp_path: Path
    ):
        store_path = tmp_path / "weights.h5"
        label_a = _label_config(kappa=2.0)
        label_b = _label_config(kappa=3.0)
        weight_a = _weight_config(decay_c=0.5)
        weight_b = _weight_config(decay_c=-0.5)

        store_weights(
            _weights_frame(3), weight_a, label_a, "BTC/USD",
            Decimal("1000000"), store_path,
        )
        store_weights(
            _weights_frame(5), weight_b, label_b, "BTC/USD",
            Decimal("1000000"), store_path,
        )

        df = list_weight_configs("BTC/USD", Decimal("1000000"), store_path)

        assert df.index.name == "weight_config_hash"
        assert set(df.index) == {weight_a.config_hash(), weight_b.config_hash()}
        assert df.loc[weight_a.config_hash(), "label_config_hash"] == label_a.config_hash()
        assert df.loc[weight_b.config_hash(), "label_config_hash"] == label_b.config_hash()
        assert df.loc[weight_a.config_hash(), "n_weights"] == 3
        assert df.loc[weight_b.config_hash(), "n_weights"] == 5

    def test_does_not_load_any_weight_frame(self, tmp_path: Path, monkeypatch):
        """list_weight_configs reads attrs only -- must never call store[key]."""
        store_path = tmp_path / "weights.h5"
        label_config = _label_config()
        weight_config = _weight_config()
        store_weights(
            _weights_frame(3), weight_config, label_config, "BTC/USD",
            Decimal("1000000"), store_path,
        )

        real_getitem = pd.HDFStore.__getitem__

        def _spy_getitem(self, key):
            raise AssertionError(
                "list_weight_configs must not load any weight frame via store[key]"
            )

        monkeypatch.setattr(pd.HDFStore, "__getitem__", _spy_getitem)
        try:
            df = list_weight_configs("BTC/USD", Decimal("1000000"), store_path)
        finally:
            monkeypatch.setattr(pd.HDFStore, "__getitem__", real_getitem)

        assert len(df) == 1

    def test_excludes_other_symbol_threshold_pairs(self, tmp_path: Path):
        store_path = tmp_path / "weights.h5"
        label_config = _label_config()
        weight_config = _weight_config()
        store_weights(
            _weights_frame(3), weight_config, label_config, "BTC/USD",
            Decimal("1000000"), store_path,
        )
        store_weights(
            _weights_frame(4), weight_config, label_config, "BTC/USD",
            Decimal("500000"), store_path,
        )
        store_weights(
            _weights_frame(5), weight_config, label_config, "ETH/USD",
            Decimal("1000000"), store_path,
        )

        df = list_weight_configs("BTC/USD", Decimal("1000000"), store_path)
        assert len(df) == 1
        assert df.iloc[0]["n_weights"] == 3


class TestCoexistenceWithLabelsInSameFile:
    def test_coexists_with_labels_in_same_h5(self, tmp_path: Path):
        store_path = tmp_path / "shared.h5"
        label_config = _label_config()
        weight_config = _weight_config()

        store_labels(
            _labels_frame(3), label_config, "BTC/USD", Decimal("1000000"), store_path
        )
        store_weights(
            _weights_frame(3), weight_config, label_config, "BTC/USD",
            Decimal("1000000"), store_path,
        )

        loaded_weights = load_weights(
            weight_config, label_config, "BTC/USD", Decimal("1000000"), store_path
        )
        assert len(loaded_weights) == 3

        with pd.HDFStore(store_path, mode="r") as store:
            keys = store.keys()
            assert any("/labels/" in k for k in keys)
            assert any("/weights/" in k for k in keys)
