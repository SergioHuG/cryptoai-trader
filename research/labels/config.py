"""LabelConfig — the recipe identity for a triple-barrier labeling run.

A LabelConfig fully determines the *semantics* of a labeling run: the CUSUM
threshold multiplier, the vol-estimate span, the profit-take/stop-loss
multipliers, the vertical horizon, the minimum target, and a schema version
guarding against silent label-semantics changes.

``symbol`` and ``threshold`` (the dollar-bar threshold) are deliberately NOT
fields here. They define *where* a label set is stored, not *what recipe*
produced it -- the same recipe should hash identically regardless of which
symbol/threshold it's later applied to.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

__all__ = ["LabelConfig"]


@dataclass(frozen=True)
class LabelConfig:
    """Frozen recipe for :func:`research.labels.pipeline.build_triple_barrier_labels`.

    Attributes:
        kappa:          CUSUM threshold multiplier (``h_t = kappa * ewma_vol_t``).
        ewma_span:      EWMA span (in bars) for the volatility estimate that
                         drives both the CUSUM threshold and the barrier target.
        pt_mult:        Profit-take multiplier on the per-event target.
                         ``0`` disables the profit-take barrier.
        sl_mult:        Stop-loss multiplier on the per-event target.
                         ``0`` disables the stop-loss barrier. Both ``pt_mult``
                         and ``sl_mult`` at ``0`` is valid -- vertical-only
                         labeling.
        max_hp:         Vertical horizon in bars (>= 1).
        min_ret:        Minimum per-event target required to keep an event
                         (AFML-strict ``trgt > min_ret``).
        schema_version: Version of the *labeling semantics* (not the knobs).
                         Lives inside :meth:`config_hash` so a future change
                         to label semantics (e.g. a zero-on-vertical variant)
                         cannot silently collide with an existing recipe hash.
    """

    kappa: float
    ewma_span: int
    pt_mult: float
    sl_mult: float
    max_hp: int
    min_ret: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.kappa > 0:
            raise ValueError(f"LabelConfig.kappa must be > 0, got {self.kappa!r}.")
        if not self.ewma_span >= 2:
            raise ValueError(
                f"LabelConfig.ewma_span must be >= 2, got {self.ewma_span!r}."
            )
        if not self.pt_mult >= 0:
            raise ValueError(
                f"LabelConfig.pt_mult must be >= 0, got {self.pt_mult!r}."
            )
        if not self.sl_mult >= 0:
            raise ValueError(
                f"LabelConfig.sl_mult must be >= 0, got {self.sl_mult!r}."
            )
        if not self.max_hp >= 1:
            raise ValueError(f"LabelConfig.max_hp must be >= 1, got {self.max_hp!r}.")
        if not self.min_ret >= 0:
            raise ValueError(
                f"LabelConfig.min_ret must be >= 0, got {self.min_ret!r}."
            )
        if not self.schema_version >= 1:
            raise ValueError(
                f"LabelConfig.schema_version must be >= 1, got "
                f"{self.schema_version!r}."
            )

    def config_hash(self) -> str:
        """Return a stable 12-hex-char identity hash for this recipe.

        ``sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]``.
        Deterministic regardless of constructor kwarg order. Includes
        ``schema_version``; excludes nothing else (``symbol``/``threshold``
        are not fields and so cannot leak into the hash).
        """
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]
