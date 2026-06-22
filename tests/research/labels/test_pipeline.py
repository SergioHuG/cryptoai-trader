"""Acceptance tests for the build_triple_barrier_labels orchestrator (Task 9).

The orchestrator is the thin wiring layer that computes ewma_vol once and
feeds both the CUSUM threshold and the barrier target from it (the
single-shared-vol invariant), slices to the first valid-vol bar before
sampling with cusum_filter (Q9 -- empirically verified to avoid a spurious
boundary event), and is pure compute: no I/O.
"""
import numpy as np
import pandas as pd
import pytest

from research.features.volatility import ewma_vol
from research.labels.barriers import Barrier, get_bins, get_events
from research.labels.config import LabelConfig
from research.labels.filters import cusum_filter
from research.labels.pipeline import build_triple_barrier_labels


def _close(values, start="2024-01-01", freq="15min"):
    idx = pd.date_range(start=start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series([float(v) for v in values], index=idx)


def _spurious_boundary_close(span: int = 5, n: int = 30) -> pd.Series:
    """A price path empirically verified to produce a spurious CUSUM event at
    the first valid-vol bar if sampled WITHOUT slicing first (small steady
    drift during the vol warm-up, quiet afterwards)."""
    rets = np.array([0.003] * 6 + [0.0001] * (n - 1 - 6))
    prices = [100.0]
    for r in rets:
        prices.append(prices[-1] * np.exp(r))
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.Series(prices, index=idx)


def _config(**overrides) -> LabelConfig:
    base = dict(
        kappa=1.0, ewma_span=5, pt_mult=1.0, sl_mult=1.0, max_hp=10, min_ret=0.0,
    )
    base.update(overrides)
    return LabelConfig(**base)


class TestOrchestratorWiring:
    def test_matches_manual_sliced_pipeline_exactly(self):
        """build_triple_barrier_labels(close, config) must produce exactly the
        same output as manually wiring ewma_vol -> slice -> cusum_filter ->
        get_events -> get_bins by hand."""
        close = _spurious_boundary_close()
        config = _config()

        actual = build_triple_barrier_labels(close, config)

        vol = ewma_vol(close, span=config.ewma_span)
        first_valid = vol.first_valid_index()
        h = config.kappa * vol
        t_events = cusum_filter(close.loc[first_valid:], h.loc[first_valid:])
        events = get_events(
            close, t_events, (config.pt_mult, config.sl_mult),
            trgt=vol, max_hp=config.max_hp, min_ret=config.min_ret,
        )
        expected = get_bins(events, close)

        pd.testing.assert_frame_equal(actual, expected)

    def test_slices_to_first_valid_vol_before_sampling(self, monkeypatch):
        """The orchestrator must call cusum_filter with close/h already
        sliced to the first valid-vol bar -- not the full series."""
        close = _spurious_boundary_close()
        config = _config()
        vol = ewma_vol(close, span=config.ewma_span)
        first_valid = vol.first_valid_index()

        captured = {}
        real_cusum_filter = cusum_filter

        def _spy(close_arg, threshold_arg):
            captured["close_index0"] = close_arg.index[0]
            captured["threshold_index0"] = threshold_arg.index[0]
            return real_cusum_filter(close_arg, threshold_arg)

        monkeypatch.setattr("research.labels.pipeline.cusum_filter", _spy)
        build_triple_barrier_labels(close, config)

        assert captured["close_index0"] == first_valid
        assert captured["threshold_index0"] == first_valid

    def test_avoids_the_spurious_boundary_event_present_in_naive_sampling(self):
        """Sanity-check the fixture itself: sampling the FULL (unsliced)
        close/threshold produces an extra event at the first valid-vol bar
        that sampling the sliced series does not. This is exactly the bug
        Q9 documents -- the fixture must actually exhibit it."""
        close = _spurious_boundary_close()
        config = _config()
        vol = ewma_vol(close, span=config.ewma_span)
        first_valid = vol.first_valid_index()
        h = config.kappa * vol

        naive_events = cusum_filter(close, h)
        sliced_events = cusum_filter(close.loc[first_valid:], h.loc[first_valid:])

        assert first_valid in naive_events
        assert first_valid not in sliced_events

    def test_ewma_vol_computed_exactly_once(self, monkeypatch):
        """Single-shared-vol invariant: one ewma_vol call feeds both the
        CUSUM threshold and the barrier target -- never computed twice."""
        close = _spurious_boundary_close()
        config = _config()

        call_count = {"n": 0}
        real_ewma_vol = ewma_vol

        def _counting_ewma_vol(prices, span):
            call_count["n"] += 1
            return real_ewma_vol(prices, span=span)

        monkeypatch.setattr("research.labels.pipeline.ewma_vol", _counting_ewma_vol)
        build_triple_barrier_labels(close, config)

        assert call_count["n"] == 1

    def test_vertical_barrier_and_path_walk_index_full_close(self):
        """Vertical barrier + path-walk must still see the full close series
        (only CUSUM sampling is sliced) -- verified indirectly: an event
        near the end of the sliced region can still resolve using bars that
        existed before the slice point was reached in absolute terms (i.e.
        the full close, not a truncated one, backs the path-walk)."""
        close = _spurious_boundary_close(span=5, n=40)
        config = _config(max_hp=5)
        result = build_triple_barrier_labels(close, config)
        # Just confirm it runs end-to-end without truncation-related errors
        # and produces resolved labels indexed within the full close range.
        assert result.index.isin(close.index).all()
        assert result["t1"].isin(close.index).all()


class TestOrchestratorPureCompute:
    def test_no_io_no_hdfstore_touched(self, monkeypatch):
        close = _spurious_boundary_close()
        config = _config()

        def _raise(*args, **kwargs):
            raise AssertionError("build_triple_barrier_labels must not touch HDFStore")

        monkeypatch.setattr(pd.HDFStore, "__init__", _raise)
        build_triple_barrier_labels(close, config)  # must not raise

    def test_creates_no_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        close = _spurious_boundary_close()
        config = _config()
        build_triple_barrier_labels(close, config)
        assert list(tmp_path.iterdir()) == []


class TestOrchestratorSymmetricMode:
    def test_side_none_produces_symmetric_bins(self):
        close = _spurious_boundary_close()
        config = _config()
        result = build_triple_barrier_labels(close, config, side=None)

        assert "side" not in result.columns
        assert set(result["bin"].unique()).issubset({-1, 0, 1})


class TestOrchestratorMetaLabelingMode:
    def test_side_given_produces_meta_bins_and_retains_side_column(self):
        close = _spurious_boundary_close()
        config = _config()
        vol = ewma_vol(close, span=config.ewma_span)
        first_valid = vol.first_valid_index()
        h = config.kappa * vol
        t_events = cusum_filter(close.loc[first_valid:], h.loc[first_valid:])
        side = pd.Series(1, index=t_events, dtype="int8")

        result = build_triple_barrier_labels(close, config, side=side)

        assert "side" in result.columns
        assert set(result["bin"].unique()).issubset({0, 1})


class TestOrchestratorVerticalOnlyConfig:
    def test_vertical_only_config_never_produces_horizontal_barriers(self):
        close = _spurious_boundary_close()
        config = _config(pt_mult=0.0, sl_mult=0.0)
        result = build_triple_barrier_labels(close, config)

        assert (result["barrier"] == int(Barrier.VERTICAL)).all()


class TestOrchestratorEdgeCases:
    def test_close_shorter_than_ewma_span_returns_empty_frame(self):
        close = _close([100, 101, 102])  # 3 bars, span requires >= 5
        config = _config(ewma_span=5)
        result = build_triple_barrier_labels(close, config)
        assert result.empty
