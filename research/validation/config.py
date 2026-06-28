"""ValidationConfig -- the recipe identity for AFML Ch.7/Ch.12 cross-validation.

A ValidationConfig fully determines the *shape* of a cross-validation run:
how many folds PurgedKFold cuts (``n_splits``), how many groups/test-groups
CombinatorialPurgedKFold combines (``n_groups``/``n_test_groups``), and the
embargo fraction shared by both splitters (``embargo_pct``).

Deliberately UNLIKE LabelConfig / WeightConfig, this config carries NO
``config_hash()`` and NO ``schema_version``. Those two siblings hash because
their outputs are persisted, config-hash-keyed artifacts
(``thr_{N}/labels/cfg_{hash}``, ``weights/lbl_{h}/cfg_{h}``) that get
reloaded later -- the hash *is* the storage identity, and schema_version
guards that reload path against silent semantic collisions.

``research/validation`` has no ``storage.py`` (locked design decision): CV
output -- a per-fold score array -- is transient. Re-running ``cv_score``
is always cheap and deterministic given the same inputs, so there is no
reload path to guard and nothing for a hash to key. Adding identity
machinery here would be cargo-culted: tested and documented for an artifact
that is never written to disk. If CV-output persistence is ever un-parked,
``config_hash()``/``schema_version`` are an additive change at that point,
not a migration.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ValidationConfig"]


@dataclass(frozen=True)
class ValidationConfig:
    """Frozen recipe shape for the cross-validation layer.

    Attributes:
        n_splits:       Number of contiguous folds for
                         :class:`research.validation.splitters.PurgedKFold`.
                         Must be ``>= 2`` -- a single fold cannot separate
                         train from test.
        embargo_pct:    Fraction of total sample count embargoed forward of
                         each test block, for BOTH PurgedKFold and
                         CombinatorialPurgedKFold (shared, not duplicated).
                         Must satisfy ``0 <= embargo_pct < 1``. ``0`` is the
                         legitimate no-embargo (purge-only) regime; ``>= 1``
                         would embargo the entire dataset.
        n_groups:       Number of contiguous partitions for
                         :class:`research.validation.cpcv.CombinatorialPurgedKFold`.
                         Must be ``>= 2`` -- CPCV needs at least two groups
                         to combine.
        n_test_groups:  Number of groups (``k``) taken as the test set per
                         CPCV combination. Must satisfy
                         ``1 <= n_test_groups < n_groups`` -- ``k`` must
                         leave at least one group for train. Defaults to
                         ``2`` (AFML's canonical CPCV example).
    """

    n_splits: int
    embargo_pct: float
    n_groups: int
    n_test_groups: int = 2

    def __post_init__(self) -> None:
        if not self.n_splits >= 2:
            raise ValueError(
                f"ValidationConfig.n_splits must be >= 2, got {self.n_splits!r}."
            )
        if not (0 <= self.embargo_pct < 1):
            raise ValueError(
                f"ValidationConfig.embargo_pct must satisfy 0 <= embargo_pct "
                f"< 1, got {self.embargo_pct!r}."
            )
        if not self.n_groups >= 2:
            raise ValueError(
                f"ValidationConfig.n_groups must be >= 2, got {self.n_groups!r}."
            )
        if not (1 <= self.n_test_groups < self.n_groups):
            raise ValueError(
                f"ValidationConfig.n_test_groups must satisfy "
                f"1 <= n_test_groups < n_groups (n_groups={self.n_groups!r}), "
                f"got {self.n_test_groups!r}."
            )
