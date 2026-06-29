"""research.validation -- AFML Ch.7 / Ch.12 leak-proof cross-validation.

Public surface:
  * ValidationConfig          -- frozen, validated CV recipe (no
    config_hash/schema_version: CV output is transient, no storage.py).
  * get_train_times,
    get_embargo_times         -- the two purge/embargo primitives
    (AFML 7.1/7.2). The shared kernel built on top of them,
    `_purge_embargo`, stays PRIVATE -- it's the internal seam both
    splitters below consume, not a name meant to be called directly.
  * PurgedKFold               -- contiguous purged K-fold splitter
    (AFML 7.3).
  * CombinatorialPurgedKFold  -- combinatorial purged splitter (AFML
    Ch.12); path-reconstruction is deferred to Phase 2.
  * MyPipeline, cv_score      -- the orchestrator (AFML 7.4 / Ch.7):
    sample_weight-aware Pipeline plus the fit-predict-score loop driving
    either splitter above.
"""
from research.validation.config import ValidationConfig
from research.validation.purge import get_embargo_times, get_train_times
from research.validation.splitters import PurgedKFold
from research.validation.cpcv import CombinatorialPurgedKFold
from research.validation.cv import MyPipeline, cv_score

__all__ = [
    "ValidationConfig",
    "get_train_times",
    "get_embargo_times",
    "PurgedKFold",
    "CombinatorialPurgedKFold",
    "MyPipeline",
    "cv_score",
]