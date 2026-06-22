"""Acceptance test for the research.labels package's public API surface.

Phase 1 (Tasks 1-9) is considered "wired" once every public name below is
importable directly from research.labels -- not just from its submodule.
"""
import research.labels as labels_pkg


class TestPublicAPISurface:
    def test_exports_every_phase1_public_name(self):
        expected = {
            "Barrier",
            "cusum_filter",
            "get_events",
            "get_bins",
            "LabelConfig",
            "build_triple_barrier_labels",
            "store_labels",
            "load_labels",
            "list_label_configs",
        }
        assert expected.issubset(set(labels_pkg.__all__))
        for name in expected:
            assert hasattr(labels_pkg, name), f"{name} not importable from research.labels"

    def test_all_has_no_unexpected_extra_names(self):
        """__all__ should be exactly the Phase-1 public surface -- no stragglers."""
        expected = {
            "Barrier",
            "cusum_filter",
            "get_events",
            "get_bins",
            "LabelConfig",
            "build_triple_barrier_labels",
            "store_labels",
            "load_labels",
            "list_label_configs",
        }
        assert set(labels_pkg.__all__) == expected
