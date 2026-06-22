"""Acceptance tests for label storage (Task 8c).

store_labels/load_labels/list_label_configs persist label frames to the
*same* .h5 file as dollar bars, in a sibling group
/{SYM_KEY}/thr_{N}/labels/cfg_{hash}. load_labels is deliberately fail-loud
(divergence from load_dollar_bars's empty-frame-on-missing): a specific-recipe
lookup where absence is almost always a typo, not a resumable checkpoint.
"""
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from research.labels.config import LabelConfig
from research.labels.storage import list_label_configs, load_labels, store_labels


def _config(**overrides) -> LabelConfig:
    base = dict(
        kappa=2.0, ewma_span=20, pt_mult=1.0, sl_mult=1.0, max_hp=24, min_ret=0.0,
    )
    base.update(overrides)
    return LabelConfig(**base)


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


class TestStoreAndLoadRoundTrip:
    def test_round_trips_values_and_columns(self, tmp_path: Path):
        store_path = tmp_path / "labels.h5"
        config = _config()
        labels = _labels_frame()

        store_labels(labels, config, "BTC/USD", Decimal("1000000"), store_path)
        loaded = load_labels(config, "BTC/USD", Decimal("1000000"), store_path)

        pd.testing.assert_frame_equal(loaded, labels, check_freq=False)

    def test_t1_column_round_trips_tz_aware(self, tmp_path: Path):
        store_path = tmp_path / "labels.h5"
        config = _config()
        labels = _labels_frame()

        store_labels(labels, config, "BTC/USD", Decimal("1000000"), store_path)
        loaded = load_labels(config, "BTC/USD", Decimal("1000000"), store_path)

        assert isinstance(loaded["t1"].dtype, pd.DatetimeTZDtype)
        assert str(loaded["t1"].dtype.tz) == "UTC"

    def test_different_symbols_do_not_collide(self, tmp_path: Path):
        store_path = tmp_path / "labels.h5"
        config = _config()
        btc_labels = _labels_frame(3)
        eth_labels = _labels_frame(2)

        store_labels(btc_labels, config, "BTC/USD", Decimal("1000000"), store_path)
        store_labels(eth_labels, config, "ETH/USD", Decimal("1000000"), store_path)

        loaded_btc = load_labels(config, "BTC/USD", Decimal("1000000"), store_path)
        loaded_eth = load_labels(config, "ETH/USD", Decimal("1000000"), store_path)
        assert len(loaded_btc) == 3
        assert len(loaded_eth) == 2

    def test_different_configs_do_not_collide(self, tmp_path: Path):
        store_path = tmp_path / "labels.h5"
        config_a = _config(kappa=2.0)
        config_b = _config(kappa=3.0)
        labels_a = _labels_frame(3)
        labels_b = _labels_frame(2)

        store_labels(labels_a, config_a, "BTC/USD", Decimal("1000000"), store_path)
        store_labels(labels_b, config_b, "BTC/USD", Decimal("1000000"), store_path)

        loaded_a = load_labels(config_a, "BTC/USD", Decimal("1000000"), store_path)
        loaded_b = load_labels(config_b, "BTC/USD", Decimal("1000000"), store_path)
        assert len(loaded_a) == 3
        assert len(loaded_b) == 2


class TestStoreOverwriteSemantics:
    def test_overwrite_true_by_default_replaces_existing(self, tmp_path: Path):
        store_path = tmp_path / "labels.h5"
        config = _config()

        store_labels(_labels_frame(3), config, "BTC/USD", Decimal("1000000"), store_path)
        store_labels(_labels_frame(2), config, "BTC/USD", Decimal("1000000"), store_path)

        loaded = load_labels(config, "BTC/USD", Decimal("1000000"), store_path)
        assert len(loaded) == 2

    def test_overwrite_false_raises_if_key_exists(self, tmp_path: Path):
        store_path = tmp_path / "labels.h5"
        config = _config()
        store_labels(_labels_frame(3), config, "BTC/USD", Decimal("1000000"), store_path)

        with pytest.raises(KeyError):
            store_labels(
                _labels_frame(2), config, "BTC/USD", Decimal("1000000"), store_path,
                overwrite=False,
            )

    def test_overwrite_false_succeeds_when_key_absent(self, tmp_path: Path):
        store_path = tmp_path / "labels.h5"
        config = _config()
        store_labels(
            _labels_frame(3), config, "BTC/USD", Decimal("1000000"), store_path,
            overwrite=False,
        )
        loaded = load_labels(config, "BTC/USD", Decimal("1000000"), store_path)
        assert len(loaded) == 3


class TestStoreAttrs:
    def test_attrs_contain_expected_keys(self, tmp_path: Path):
        store_path = tmp_path / "labels.h5"
        config = _config()
        labels = _labels_frame(3)
        store_labels(labels, config, "BTC/USD", Decimal("1000000"), store_path)

        with pd.HDFStore(store_path, mode="r") as store:
            key = f"/BTC_USD/thr_1000000/labels/cfg_{config.config_hash()}"
            attrs = store.get_storer(key).attrs

            assert attrs["symbol"] == "BTC/USD"
            assert attrs["threshold"] == Decimal("1000000")
            assert attrs["t1_encoding"] == "datetime"
            assert attrs["n_labels"] == 3
            assert attrs["label_config"]["kappa"] == 2.0

    def test_created_at_is_iso8601_utc(self, tmp_path: Path):
        import datetime as dt

        store_path = tmp_path / "labels.h5"
        config = _config()
        store_labels(_labels_frame(3), config, "BTC/USD", Decimal("1000000"), store_path)

        with pd.HDFStore(store_path, mode="r") as store:
            key = f"/BTC_USD/thr_1000000/labels/cfg_{config.config_hash()}"
            created_at = store.get_storer(key).attrs["created_at"]

        parsed = dt.datetime.fromisoformat(created_at)
        assert parsed.tzinfo is not None


class TestLoadLabelsFailLoud:
    def test_missing_store_file_raises_file_not_found(self, tmp_path: Path):
        store_path = tmp_path / "does_not_exist.h5"
        config = _config()
        with pytest.raises(FileNotFoundError):
            load_labels(config, "BTC/USD", Decimal("1000000"), store_path)

    def test_unstored_config_raises_key_error(self, tmp_path: Path):
        store_path = tmp_path / "labels.h5"
        stored_config = _config(kappa=2.0)
        never_stored_config = _config(kappa=99.0)
        store_labels(_labels_frame(3), stored_config, "BTC/USD", Decimal("1000000"), store_path)

        with pytest.raises(KeyError):
            load_labels(never_stored_config, "BTC/USD", Decimal("1000000"), store_path)


class TestListLabelConfigs:
    def test_returns_empty_frame_when_file_does_not_exist(self, tmp_path: Path):
        store_path = tmp_path / "does_not_exist.h5"
        df = list_label_configs("BTC/USD", Decimal("1000000"), store_path)
        assert df.empty
        assert df.index.name == "cfg_hash"

    def test_returns_empty_frame_when_no_configs_stored_for_pair(self, tmp_path: Path):
        store_path = tmp_path / "labels.h5"
        config = _config()
        store_labels(_labels_frame(3), config, "ETH/USD", Decimal("1000000"), store_path)

        df = list_label_configs("BTC/USD", Decimal("1000000"), store_path)
        assert df.empty

    def test_lists_multiple_configs_indexed_by_hash(self, tmp_path: Path):
        store_path = tmp_path / "labels.h5"
        config_a = _config(kappa=2.0)
        config_b = _config(kappa=3.0)
        store_labels(_labels_frame(3), config_a, "BTC/USD", Decimal("1000000"), store_path)
        store_labels(_labels_frame(5), config_b, "BTC/USD", Decimal("1000000"), store_path)

        df = list_label_configs("BTC/USD", Decimal("1000000"), store_path)

        assert df.index.name == "cfg_hash"
        assert set(df.index) == {config_a.config_hash(), config_b.config_hash()}
        assert df.loc[config_a.config_hash(), "n_labels"] == 3
        assert df.loc[config_b.config_hash(), "n_labels"] == 5
        assert df.loc[config_a.config_hash(), "kappa"] == 2.0

    def test_does_not_load_any_label_frames(self, tmp_path: Path, monkeypatch):
        """list_label_configs reads attrs only -- must never call store[key]."""
        store_path = tmp_path / "labels.h5"
        config = _config()
        store_labels(_labels_frame(3), config, "BTC/USD", Decimal("1000000"), store_path)

        real_getitem = pd.HDFStore.__getitem__

        def _spy_getitem(self, key):
            raise AssertionError(
                "list_label_configs must not load any label frame via store[key]"
            )

        monkeypatch.setattr(pd.HDFStore, "__getitem__", _spy_getitem)
        try:
            df = list_label_configs("BTC/USD", Decimal("1000000"), store_path)
        finally:
            monkeypatch.setattr(pd.HDFStore, "__getitem__", real_getitem)

        assert len(df) == 1

    def test_excludes_other_symbol_threshold_pairs(self, tmp_path: Path):
        store_path = tmp_path / "labels.h5"
        config = _config()
        store_labels(_labels_frame(3), config, "BTC/USD", Decimal("1000000"), store_path)
        store_labels(_labels_frame(4), config, "BTC/USD", Decimal("500000"), store_path)
        store_labels(_labels_frame(5), config, "ETH/USD", Decimal("1000000"), store_path)

        df = list_label_configs("BTC/USD", Decimal("1000000"), store_path)
        assert len(df) == 1
        assert df.iloc[0]["n_labels"] == 3
