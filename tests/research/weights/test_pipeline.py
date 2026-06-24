"""Acceptance tests for build_sample_weights (Step 6, Q3/Q9).

The orchestrator is the thin wiring layer that computes num_co_events
exactly once and threads it to BOTH avg_uniqueness and return_attribution
(the single-shared-c_t invariant, Q3) -- enforced here by a call-count spy
and an object-identity spy, mirroring the ewma_vol spy in
tests/research/labels/test_pipeline.py.

Locked weight order (Q4c, Q5d): raw attribution -> normalize_weights ->
x time_decay, with NO post-decay renormalization. Output is a three-column
frame [avg_uniqueness, weight, t1] (Q9) -- pure compute, no I/O.
"""
import pandas as pd
import pytest

from research.weights.attribution import return_attribution
from research.weights.concurrency import avg_uniqueness, num_co_events
from research.weights.config import WeightConfig
from research.weights.decay import time_decay
from research.weights.pipeline import build_sample_weights


def _close(values, start="2024-01-01", freq="15min"):
    idx = pd.date_range(start=start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series([float(v) for v in values], index=idx)


def _labels(idx, pairs, **extra_cols):
    """Build a get_bins-style labels frame from [(start_pos, end_pos), ...]."""
    starts = [idx[s] for s, _ in pairs]
    ends = [idx[e] for _, e in pairs]
    df = pd.DataFrame({"t1": ends}, index=pd.DatetimeIndex(starts))
    for col, value in extra_cols.items():
        df[col] = value
    return df


def _config(**overrides):
    base = dict(decay_c=0.5)
    base.update(overrides)
    return WeightConfig(**base)


class TestBuildSampleWeightsOutputShape:
    def test_returns_three_column_frame(self):
        close = _close([100.0, 110.0, 121.0, 133.1, 146.41])
        labels = _labels(close.index, [(1, 3)])
        config = _config()

        result = build_sample_weights(labels, close, config)

        assert list(result.columns) == ["avg_uniqueness", "weight", "t1"]
        assert list(result.index) == [close.index[1]]

    def test_t1_carried_through_from_labels(self):
        close = _close([100.0, 110.0, 121.0, 133.1, 146.41])
        labels = _labels(close.index, [(0, 2), (1, 3)])
        config = _config()

        result = build_sample_weights(labels, close, config)

        pd.testing.assert_series_equal(
            result["t1"], labels["t1"], check_names=False
        )


class TestBuildSampleWeightsLockedOrder:
    def test_weight_equals_normalize_attribution_times_decay(self):
        close = _close([100.0, 110.0, 121.0, 133.1, 146.41, 161.05])
        labels = _labels(close.index, [(0, 2), (1, 3), (3, 5)])
        config = _config(decay_c=0.3)

        result = build_sample_weights(labels, close, config)

        t1 = labels["t1"]
        co_events = num_co_events(close.index, t1)
        tw = avg_uniqueness(t1, co_events)
        attr = return_attribution(close, t1, co_events)
        from research.weights.attribution import normalize_weights

        expected_weight = normalize_weights(attr) * time_decay(tw, config.decay_c)

        pd.testing.assert_series_equal(
            result["weight"], expected_weight, check_names=False
        )

    def test_no_post_decay_renormalization(self):
        """With decay enabled (decay_c < 1), the total weight must shrink
        below len(labels) -- proving the orchestrator does NOT re-normalize
        after multiplying in the decay factor."""
        close = _close([100.0, 110.0, 121.0, 133.1, 146.41, 161.05])
        labels = _labels(close.index, [(0, 2), (1, 3), (3, 5)])
        config = _config(decay_c=0.0)

        result = build_sample_weights(labels, close, config)

        assert result["weight"].sum() < len(labels) - 1e-9


class TestBuildSampleWeightsSharedCoEvents:
    def test_num_co_events_called_exactly_once(self, monkeypatch):
        close = _close([100.0, 110.0, 121.0, 133.1, 146.41, 161.05])
        labels = _labels(close.index, [(0, 2), (1, 3), (3, 5)])
        config = _config()

        call_count = {"n": 0}
        real_num_co_events = num_co_events

        def _counting_num_co_events(close_index_arg, t1_arg):
            call_count["n"] += 1
            return real_num_co_events(close_index_arg, t1_arg)

        monkeypatch.setattr(
            "research.weights.pipeline.num_co_events", _counting_num_co_events
        )
        build_sample_weights(labels, close, config)

        assert call_count["n"] == 1

    def test_avg_uniqueness_and_attribution_receive_the_same_co_events_object(
        self, monkeypatch
    ):
        close = _close([100.0, 110.0, 121.0, 133.1, 146.41, 161.05])
        labels = _labels(close.index, [(0, 2), (1, 3), (3, 5)])
        config = _config()

        captured = {}
        real_avg_uniqueness = avg_uniqueness
        real_return_attribution = return_attribution

        def _spy_avg_uniqueness(t1_arg, co_events_arg):
            captured["avgU_co_events_id"] = id(co_events_arg)
            return real_avg_uniqueness(t1_arg, co_events_arg)

        def _spy_return_attribution(close_arg, t1_arg, co_events_arg):
            captured["attr_co_events_id"] = id(co_events_arg)
            return real_return_attribution(close_arg, t1_arg, co_events_arg)

        monkeypatch.setattr(
            "research.weights.pipeline.avg_uniqueness", _spy_avg_uniqueness
        )
        monkeypatch.setattr(
            "research.weights.pipeline.return_attribution", _spy_return_attribution
        )
        build_sample_weights(labels, close, config)

        assert captured["avgU_co_events_id"] == captured["attr_co_events_id"]


class TestBuildSampleWeightsIgnoresExtraColumns:
    def test_ignores_ret_bin_barrier_side(self):
        close = _close([100.0, 110.0, 121.0, 133.1, 146.41])
        config = _config()

        labels_bare = _labels(close.index, [(1, 3)])
        labels_with_extras = _labels(
            close.index, [(1, 3)], ret=0.42, bin=1, barrier=1, side=-1
        )

        result_bare = build_sample_weights(labels_bare, close, config)
        result_with_extras = build_sample_weights(labels_with_extras, close, config)

        pd.testing.assert_frame_equal(result_bare, result_with_extras)


class TestBuildSampleWeightsPureCompute:
    def test_no_io_no_hdfstore_touched(self, monkeypatch):
        close = _close([100.0, 110.0, 121.0, 133.1, 146.41])
        labels = _labels(close.index, [(1, 3)])
        config = _config()

        def _raise(*args, **kwargs):
            raise AssertionError("build_sample_weights must not touch HDFStore")

        monkeypatch.setattr(pd.HDFStore, "__init__", _raise)
        build_sample_weights(labels, close, config)  # must not raise

    def test_creates_no_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        close = _close([100.0, 110.0, 121.0, 133.1, 146.41])
        labels = _labels(close.index, [(1, 3)])
        config = _config()

        build_sample_weights(labels, close, config)

        assert list(tmp_path.iterdir()) == []


class TestBuildSampleWeightsEdgeCases:
    def test_empty_labels_frame_returns_empty_three_column_frame(self):
        close = _close([100.0, 110.0, 121.0])
        labels = _labels(close.index, [])
        config = _config()

        result = build_sample_weights(labels, close, config)

        assert result.empty
        assert list(result.columns) == ["avg_uniqueness", "weight", "t1"]

    def test_single_event_frame_avg_uniqueness_is_one(self):
        close = _close([100.0, 110.0, 121.0, 133.1])
        labels = _labels(close.index, [(1, 2)])
        config = _config()

        result = build_sample_weights(labels, close, config)

        assert result["avg_uniqueness"].iloc[0] == pytest.approx(1.0)
        assert result["weight"].iloc[0] > 0

    def test_null_t1_propagates_fail_loud(self):
        close = _close([100.0, 110.0, 121.0, 133.1])
        labels = pd.DataFrame(
            {"t1": [pd.NaT]}, index=pd.DatetimeIndex([close.index[1]])
        )
        config = _config()

        with pytest.raises(ValueError):
            build_sample_weights(labels, close, config)
