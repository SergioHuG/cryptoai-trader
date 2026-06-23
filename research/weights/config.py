"""WeightConfig -- the recipe identity for an AFML Ch.4 sample-weight build.

A WeightConfig fully determines the *semantics* of a weight build: the
time-decay knob (AFML's ``clfLastW``) and a schema version guarding against
silent weight-semantics changes.

The parent label-config hash is deliberately NOT a field here -- same
principle as LabelConfig's symbol/threshold exclusion. A WeightConfig
defines *what recipe* (decay only -- concurrency/avg-uniqueness/attribution
have no knobs) produced a weight set, not *which label set* it was applied
to. That provenance link is a storage-path/attrs dimension, handled in
``research.weights.storage`` (see Q8).

Documented schema_version bump triggers (none implemented in v1 -- each is
a deliberate future variant, not a free addition):
  * A uniqueness-only ``weight_scheme`` (skip return-attribution entirely).
  * An exponential ``decay_kind`` (AFML's getTimeDecay snippet is linear-only).
  * Any re-port that changes the attribution math (e.g. log- vs simple-return
    choice in research.weights.attribution).
Any of these must bump schema_version so the new semantics can't silently
collide with an existing v1 recipe hash.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

__all__ = ["WeightConfig"]


@dataclass(frozen=True)
class WeightConfig:
    """Frozen recipe for :func:`research.weights.pipeline.build_sample_weights`.

    Attributes:
        decay_c:        Time-decay multiplier (AFML's ``clfLastW``).
                         ``c == 1`` disables decay entirely. ``0 < c < 1``
                         linearly decays the oldest event's weight toward
                         ``c``. ``c == 0`` decays the oldest event to zero.
                         ``c < 0`` is the legitimate truncation regime (the
                         oldest ``(1 + c)`` fraction of cumulative uniqueness
                         is zeroed outright) -- there is no lower bound.
        schema_version:  Version of the *weight semantics* (not the knobs).
                         Lives inside :meth:`config_hash` so a future change
                         to weight semantics (see module docstring) cannot
                         silently collide with an existing recipe hash.
    """

    decay_c: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.decay_c <= 1:
            raise ValueError(
                f"WeightConfig.decay_c must be <= 1, got {self.decay_c!r}."
            )
        if not self.schema_version >= 1:
            raise ValueError(
                f"WeightConfig.schema_version must be >= 1, got "
                f"{self.schema_version!r}."
            )

    def config_hash(self) -> str:
        """Return a stable 12-hex-char identity hash for this recipe.

        ``sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]``.
        Deterministic regardless of constructor kwarg order. Includes
        ``schema_version``; excludes nothing else (the parent label-config
        hash is not a field and so cannot leak into the hash).
        """
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]
