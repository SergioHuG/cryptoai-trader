"""Acceptance test for the research.weights package's public API surface.

The research/weights/ sub-package is considered "wired" once every public
name below is importable directly from research.weights -- not just from
its submodule.
"""
import research.weights as weights_pkg


class TestPublicAPISurface:
    def test_exports_every_public_name(self):
        expected = {
            "WeightConfig",
            "build_sample_weights",
            "store_weights",
            "load_weights",
            "list_weight_configs",
            "num_co_events",
            "avg_uniqueness",
            "return_attribution",
            "normalize_weights",
            "time_decay",
            "get_ind_matrix",
            "ind_matrix_avg_uniqueness",
            "seq_bootstrap",
        }
        assert expected.issubset(set(weights_pkg.__all__))
        for name in expected:
            assert hasattr(weights_pkg, name), (
                f"{name} not importable from research.weights"
            )

    def test_all_has_no_unexpected_extra_names(self):
        """__all__ should be exactly the public surface -- no stragglers."""
        expected = {
            "WeightConfig",
            "build_sample_weights",
            "store_weights",
            "load_weights",
            "list_weight_configs",
            "num_co_events",
            "avg_uniqueness",
            "return_attribution",
            "normalize_weights",
            "time_decay",
            "get_ind_matrix",
            "ind_matrix_avg_uniqueness",
            "seq_bootstrap",
        }
        assert set(weights_pkg.__all__) == expected
