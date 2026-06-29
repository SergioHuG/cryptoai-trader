"""Acceptance test for the research.validation package's public API surface.

The research/validation/ sub-package is considered "wired" once every
public name below is importable directly from research.validation -- not
just from its submodule. _purge_embargo stays private: it's the shared
kernel both splitters consume internally (Step 2c), not a name end users
of this sub-package are meant to call directly.
"""
import research.validation as validation_pkg


class TestPublicAPISurface:
    def test_exports_every_public_name(self):
        expected = {
            "ValidationConfig",
            "get_train_times",
            "get_embargo_times",
            "PurgedKFold",
            "CombinatorialPurgedKFold",
            "MyPipeline",
            "cv_score",
        }
        assert expected.issubset(set(validation_pkg.__all__))
        for name in expected:
            assert hasattr(validation_pkg, name), (
                f"{name} not importable from research.validation"
            )

    def test_all_has_no_unexpected_extra_names(self):
        """__all__ should be exactly the public surface -- no stragglers."""
        expected = {
            "ValidationConfig",
            "get_train_times",
            "get_embargo_times",
            "PurgedKFold",
            "CombinatorialPurgedKFold",
            "MyPipeline",
            "cv_score",
        }
        assert set(validation_pkg.__all__) == expected

    def test_purge_embargo_kernel_stays_private_not_exported(self):
        """_purge_embargo (Step 2c) is the shared kernel both splitters
        consume internally -- deliberately not part of the public
        surface (Q6 build-sequence note)."""
        assert "_purge_embargo" not in validation_pkg.__all__
        assert not hasattr(validation_pkg, "_purge_embargo")
