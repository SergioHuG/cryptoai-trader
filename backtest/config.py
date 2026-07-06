"""BacktestConfig -- the recipe identity for a Phase 2 backtest run.

A BacktestConfig fully determines the *semantics* of a backtest run: the
CPCV group geometry, the walk-forward split count, the embargo fraction,
the O-U Monte Carlo knobs, the PBO partition count, the PSR benchmark, the
two gate thresholds (PSR/DSR vs PBO), and a schema version guarding
against silent semantics collisions.

No locators (symbol, threshold, storage paths) live here -- deliberately,
mirroring research.labels.config.LabelConfig (Q12). This hash is the
ledger/result dedup key, keyed on the *recipe*, not on where it is later
applied.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

__all__ = ["BacktestConfig"]


@dataclass(frozen=True)
class BacktestConfig:
    """Frozen recipe for the Phase 2 backtest/overfitting-gate layer.

    Attributes:
        n_groups:        Total CPCV group count (``N``, >= 2).
        n_test_groups:   CPCV test-group count per combination (``k``, >= 1,
                         strictly less than ``n_groups``).
        n_wf_splits:     Walk-forward split count (>= 1).
        embargo_pct:     Fraction of the full sample grid to embargo forward
                         of each test block. Must satisfy
                         ``0 <= embargo_pct < 1`` (mirrors
                         ``get_embargo_times``'s own contract).
        n_sims:          O-U Monte Carlo simulation count (>= 1).
        seed:            O-U Monte Carlo RNG seed (>= 0); feeds
                         ``numpy.random.default_rng``.
        burn_in:         O-U Monte Carlo deterministic burn-in length in
                         observations (>= 0).
        pbo_partitions:  CSCV partition count ``S`` for PBO (>= 2, even --
                         required for the ``C(S, S/2)`` combinatorial split).
        psr_benchmark:   PSR benchmark Sharpe ``SR*`` (unconstrained --
                         may be zero, positive, or negative).
        gate_threshold:  Hard-gate pass threshold for PSR/DSR,
                         ``0 <= gate_threshold <= 1``.
        pbo_threshold:   Hard-gate pass threshold for PBO,
                         ``0 <= pbo_threshold <= 1``.
        schema_version:  Version of the *backtest semantics* (not the
                         knobs). Lives inside :meth:`config_hash` so a
                         future change to backtest semantics cannot
                         silently collide with an existing recipe hash.
    """

    n_groups: int
    n_test_groups: int
    n_wf_splits: int
    embargo_pct: float
    n_sims: int
    seed: int
    burn_in: int
    pbo_partitions: int
    psr_benchmark: float
    gate_threshold: float
    pbo_threshold: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.n_groups >= 2:
            raise ValueError(
                f"BacktestConfig.n_groups must be >= 2, got {self.n_groups!r}."
            )
        if not self.n_test_groups >= 1:
            raise ValueError(
                f"BacktestConfig.n_test_groups must be >= 1, got "
                f"{self.n_test_groups!r}."
            )
        if not self.n_test_groups < self.n_groups:
            raise ValueError(
                f"BacktestConfig.n_test_groups must be < n_groups (CPCV "
                f"requires k < N), got n_test_groups={self.n_test_groups!r}, "
                f"n_groups={self.n_groups!r}."
            )
        if not self.n_wf_splits >= 1:
            raise ValueError(
                f"BacktestConfig.n_wf_splits must be >= 1, got "
                f"{self.n_wf_splits!r}."
            )
        if not (0 <= self.embargo_pct < 1):
            raise ValueError(
                f"BacktestConfig.embargo_pct must satisfy 0 <= embargo_pct "
                f"< 1, got {self.embargo_pct!r}."
            )
        if not self.n_sims >= 1:
            raise ValueError(
                f"BacktestConfig.n_sims must be >= 1, got {self.n_sims!r}."
            )
        if not self.seed >= 0:
            raise ValueError(
                f"BacktestConfig.seed must be >= 0, got {self.seed!r}."
            )
        if not self.burn_in >= 0:
            raise ValueError(
                f"BacktestConfig.burn_in must be >= 0, got {self.burn_in!r}."
            )
        if not self.pbo_partitions >= 2:
            raise ValueError(
                f"BacktestConfig.pbo_partitions must be >= 2, got "
                f"{self.pbo_partitions!r}."
            )
        if self.pbo_partitions % 2 != 0:
            raise ValueError(
                f"BacktestConfig.pbo_partitions must be even (required for "
                f"the C(S, S/2) CSCV split), got {self.pbo_partitions!r}."
            )
        if not (0 <= self.gate_threshold <= 1):
            raise ValueError(
                f"BacktestConfig.gate_threshold must satisfy "
                f"0 <= gate_threshold <= 1, got {self.gate_threshold!r}."
            )
        if not (0 <= self.pbo_threshold <= 1):
            raise ValueError(
                f"BacktestConfig.pbo_threshold must satisfy "
                f"0 <= pbo_threshold <= 1, got {self.pbo_threshold!r}."
            )
        if not self.schema_version >= 1:
            raise ValueError(
                f"BacktestConfig.schema_version must be >= 1, got "
                f"{self.schema_version!r}."
            )

    def config_hash(self) -> str:
        """Return a stable 12-hex-char identity hash for this recipe.

        ``sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]``.
        Deterministic regardless of constructor kwarg order. Includes
        ``schema_version``; excludes nothing else (no locator fields exist
        to leak into the hash).
        """
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]
