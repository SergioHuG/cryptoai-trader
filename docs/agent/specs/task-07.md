<!--
  TASK SPEC — CryptoAI Trader delegation protocol
  Governed by: docs/agent/PROTOCOL.md §3 (DEC-3, keystone) and §7 (self-verification gate manifest match)

  File naming convention: docs/agent/specs/task-NN.md
  Branch naming convention: feature/task-NN-<slug>
  The pre-push hook derives the expected spec path from the current branch name, then
  cross-checks the frontmatter's task_id/branch fields against reality — a mismatch is a STOP.

  Do not delete this comment block when instantiating a real spec; it documents the contract
  for humans reading the raw file. Everything below --- start-of-frontmatter is real content.
-->

---
task_id: task-07
branch: feature/task-07-engine-wf
module: backtest/engine.py
depends_on: [task-06]       # merged to the integration branch (feature/phase2-backtest), not main — PROTOCOL.md §9

test_manifest:
  - file: tests/backtest/test_engine.py
    functions:
      - test_fit_predict_returns_exact_bet_returns
      - test_fit_predict_returns_clones_estimator
      - test_fit_predict_returns_applies_sample_weight
      - test_fit_predict_returns_none_sample_weight
      - test_fit_predict_returns_custom_position_fn
      - test_fit_predict_returns_raises_on_index_misalignment
      - test_default_position_fn_signed_unit
      - test_walk_forward_anchoring_past_only
      - test_walk_forward_purge_equivalence
      - test_walk_forward_single_dense_path
      - test_walk_forward_forwards_sample_weight
      - test_walk_forward_thin_result
      - test_walk_forward_scalars_exact
      - test_walk_forward_raises_on_misalignment
      - test_result_dataclasses_frozen

dod:
  coverage_min_pct: 100
  docker_command: "scripts/test.sh tests/backtest/test_engine.py"

out_of_scope:
  - "run_cpcv and anything CPCV-specific (CombinatorialPurgedKFold, group-space re-split, phi-path
     reconstruction, median-path PSR gate) — that is task-08."
  - "run_synthetic, O-U null generation, and any call into backtest.synthetic — that is task-09."
  - "Tri-state hard-gate composition (PSR/DSR/PBO PASS/FAIL/UNAVAILABLE), provisional/final verdict
     logic, and populating BacktestResult.gates — that is task-09. This task leaves gates=None."
  - "Ledger integration: no trial registration, no append, no dedup, no call into backtest.ledger or
     backtest.storage. The register_trial parameter is NOT added in this task (see Behavior §7)."
  - "The single-seam spy test asserting all three modes route through _fit_predict_returns — that is
     task-09, when all three modes exist."
  - "Rolling fixed-window walk-forward (deferred, carry-forward ledger item 2). Anchored/expanding only."
  - "Probability-based position sizing and meta-labeling (deferred, carry-forward ledger item 1).
     position_fn is injectable here, but no non-default implementation ships in this task."
  - "backtest/__init__.py public API exports — that is task-10."
  - "Any change to research/validation/* — _purge_embargo and MyPipeline are imported and used as-is,
     never modified."

escalation_triggers:
  - "_purge_embargo's actual signature in research/validation/purge.py differs from the one pinned in
     the Contract below."
  - "MyPipeline's actual constructor or fit signature in research/validation/cv.py differs from the
     usage pinned in Behavior §2."
  - "A numeric oracle in the table below cannot be satisfied without changing an already-locked ADR
     (ADR-029 through ADR-044) or the Q1-Q15 design lock."
  - "A new third-party dependency would be required."
  - "A test declared in test_manifest cannot be written to assert its stated oracle without first
     changing the Contract signatures."
  - "The boundary rule in Behavior §3 produces an empty train set for any split in the designed
     fixtures (it should not, by construction — if it does, the rule or the fixture is wrong)."
---

# Task 07: `backtest/engine.py` — shared seam + walk-forward

## Identity

- **Module:** `backtest/engine.py` — the shared fit/predict seam and the walk-forward entry point
  **only**. `run_cpcv` is task-08; `run_synthetic`, gate assembly, and ledger integration are
  task-09; `__init__.py` exports are task-10. Three tasks build this one file additively.
- **Task branch:** `feature/task-07-engine-wf`
- **Depends on:** `task-06` (`storage.py`), merged to `feature/phase2-backtest`.

## Contract

Verbatim signatures the implementation must match exactly — no renaming, no signature drift, no
"improved" parameter order. If a signature here turns out to be wrong, that's a spec bug: STOP,
don't silently fix it in commit 2.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import pandas as pd


class GateState(Enum):
    """Tri-state hard-gate outcome (Q13). Defined here; populated in task-09."""
    PASS = "pass"
    FAIL = "fail"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class GateResult:
    """Assembled hard-gate verdict (Q13). Shape fixed here; populated in task-09."""
    psr: GateState
    dsr: GateState
    pbo: GateState
    verdict: str


@dataclass(frozen=True)
class BacktestResult:
    """Public verdict object returned by every run_* entry point.

    Type identity is fixed in task-07. task-08 and task-09 may ADD fields, but
    every added field must be optional with a default so task-07 construction
    sites and tests never break.
    """
    config_hash: str
    created_at: pd.Timestamp
    paths: pd.DataFrame
    sharpe: float
    psr: float
    gates: GateResult | None


def _fit_predict_returns(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    event_returns: pd.Series,
    train_pos,
    test_pos,
    position_fn: Callable,
    sample_weight: pd.Series | None = None,
) -> pd.Series:
    """Clone the estimator, fit on train_pos, predict on test_pos, map predictions
    to positions via position_fn, and return position * event_returns over test_pos
    as a bet-return Series indexed by the test events' t0."""
    ...


def _default_position_fn(sides):
    """Signed-unit primary (Q4): predicted side in {-1, 0, +1} is used directly
    as the position, unchanged."""
    ...


def run_walk_forward(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    event_returns: pd.Series,
    config: "BacktestConfig",
    position_fn: Callable | None = None,
    sample_weight: pd.Series | None = None,
) -> BacktestResult:
    """Anchored, past-only walk-forward over config.n_wf_splits test blocks.
    Returns a thin BacktestResult: dense single-column paths, the single-path
    sharpe and psr scalars, config_hash, created_at, and gates=None."""
    ...
```

**Import surface for this task** (CPCV, synthetic, ledger and storage imports arrive in task-08/09
and must NOT appear here):

```python
from sklearn.base import clone
from research.validation.purge import _purge_embargo   # private, deliberate (Q6)
from research.validation.cv import MyPipeline          # public (Q14)
from backtest.config import BacktestConfig
from backtest.metrics import sharpe, psr
```

**`_purge_embargo`'s signature, pinned** — the equivalence test calls it verbatim:

```python
def _purge_embargo(
    t1: pd.Series,
    test_times: pd.Series,
    bars: pd.DatetimeIndex,
    embargo_pct: float,
) -> pd.Index:
    ...
```

## Behavior

Written for a competent engineer with full context on the codebase's conventions (AFML grounding,
fail-loud over silent fallback) but zero context on this module. Conventions are not restated here.

**1. Index-alignment validation (fail-loud).** At both entry points (`run_walk_forward` and
`_fit_predict_returns`), the indexes of `X`, `y`, and `event_returns` must be **exactly equal** —
same labels, same order. When `sample_weight` is provided, its index must match as well. Any
mismatch raises `ValueError`. This mirrors the validation package's convention (`ValueError` for
invalid inputs); `KeyError`/`FileNotFoundError` are storage-layer postures and do not apply here.

**2. The shared seam, `_fit_predict_returns`.** The single helper all evaluation modes route
through. Per Q14:

- Wrap the bare estimator: `MyPipeline([("clf", clone(estimator))])`. The estimator is **cloned per
  fold** (`sklearn.base.clone`) — a flagged, deliberate divergence from `cv_score`'s re-fit posture,
  which unblocks the parked `mp_engine` parallelization. The caller's own estimator instance is
  never fitted.
- Slice train data by position: `X.iloc[train_pos]`, `y.iloc[train_pos]`, and when `sample_weight`
  is provided, `sample_weight.iloc[train_pos]`.
- Fit with the weight when present, without it when `sample_weight is None` (MyPipeline is a
  None-safe superset).
- Predict with **`predict()` only** — never `predict_proba`. This is a deliberate, documented
  divergence from `cv_score`'s proba posture (Q4).
- Map predictions to positions via `position_fn`, then compute
  `bet_return = position * event_returns.iloc[test_pos]`.
- Return a `pd.Series` of bet returns indexed by the test events' `t0`.

**3. Walk-forward boundary rule.** Anchored / expanding-window, past-only (Q6). The event timeline
is the positions over `X.index` in `t0` order.

- Partition those positions into **`n_wf_splits + 1` contiguous groups** with
  `numpy.array_split` semantics: group sizes differ by at most one, and earlier groups receive the
  extra event.
- **Group 0 is the initial train-only segment** — it is never a test block.
- **Split `i`** for `i = 1..n_wf_splits`: test = group `i`; train candidates = all positions
  strictly before group `i`'s start.
- `n_wf_splits` is therefore both the number of test blocks and the number of fits. Anchoring,
  expansion, and the strictly-before property hold by construction; purge applies on top.

**4. Purge and embargo.** For each split, call the kernel verbatim:
`_purge_embargo(t1, test_times, bars, embargo_pct)` where

- `test_times` is a **single-row** `pd.Series` for the split: index = group `i`'s first `t0`, value
  = the maximum `t1` over group `i`'s events.
- `bars` = the full event grid, `X.index`.
- `embargo_pct` = `config.embargo_pct`.

The kernel composes `get_embargo_times` + `get_train_times` and extends the embargo only off the
right edge. The strictly-before anchoring mask is **engine-applied and separate from the kernel** —
the engine restricts to pre-test positions first, then intersects with the kernel's result. The
final train set for split `i` is the kernel's returned index, restricted to the strictly-before
candidates.

**5. Empty train is fail-loud.** If any split's train set is empty after purge, raise `ValueError`.
There is no silent skip. By construction of §3 no split has an empty **pre-purge** train set, so an
empty post-purge set means purging consumed everything — a real condition worth surfacing, not
absorbing.

**6. Assembling the result.** The dense path is the in-order concatenation of the `n_wf_splits` test
groups' bet-return Series. Its index is the OOS `t0` timeline (groups 1..n); it has exactly **one
column**, and it contains **no NaN** — walk-forward produces a single dense path. Then:

- `paths` — the single-column `pd.DataFrame` described above.
- `sharpe` — `backtest.metrics.sharpe` applied to the dense path's values (per-bet, non-annualized,
  `mean/std(ddof=1)`).
- `psr` — `backtest.metrics.psr` applied to the same values with `sr_star=config.psr_benchmark`.
- `config_hash` — `config.config_hash()`.
- `created_at` — a `pd.Timestamp` recorded at run time.
- `gates` — **`None`**. Gate composition is task-09's deliverable. `None` means *not assembled by
  this entry point* and is semantically distinct from `GateState.UNAVAILABLE`, which is a computed
  tri-state meaning "fewer than 2 trials exist." Do not substitute a placeholder `GateResult`, and
  do not give `gates` a dataclass default — the field is explicit at every construction site.

**7. Reserved for task-09 — do not implement here.** Per Q11, `run_walk_forward` auto-appends a
trial to the ledger. That integration lands in **task-09** as an **additive** parameter,
`register_trial: bool = True`, appended to this signature. It is recorded here so the task-07
signature is designed not to fight it. This task adds no ledger parameter, imports nothing from
`backtest.ledger` or `backtest.storage`, and registers nothing.

**8. `position_fn` resolution.** The public surface accepts `position_fn=None` and resolves it
internally to `_default_position_fn` (signed-unit primary, Q4). `_fit_predict_returns` always
receives a concrete callable — `position_fn` is required there, never `None`.

## Numeric Oracles

Design-time, fixed values. These become the literal assertions in the commit-1 tests.

### Fixture A — the core walk-forward fixture

Used by every `test_walk_forward_*` test except `scalars_exact`.

```
n_events      = 9, t0 = nine ordered timestamps T0..T8, t1 non-overlapping (t1_k < t0_{k+1})
config        : n_wf_splits = 2, embargo_pct = 0.0
array_split   : 9 positions into 3 groups -> [0,1,2], [3,4,5], [6,7,8]
group 0       = train-only seed; split 1 test = [3,4,5]; split 2 test = [6,7,8]
event_returns = [0.02, -0.01, 0.03, 0.05, -0.04, 0.01, 0.02, -0.03, 0.04]
stub estimator: predict() returns +1 for every test event
```

With `embargo_pct = 0.0` and non-overlapping `t1`, purge removes nothing, so train sets are exactly
the strictly-before candidates.

| Case | Input | Expected Output | Rationale / Source |
|---|---|---|---|
| `fit_predict_returns_exact_bet_returns` | `test_pos=[2,3,4]`, `event_returns=[.02,-.01,.03,.05,-.04]`, stub predicts `[+1, 0, -1]` | Series `[0.03, 0.00, 0.04]` indexed `[T2, T3, T4]` | Hand-computed. `+1*.03=.03`, `0*.05=0`, `-1*-.04=.04`. Covers all three signed-unit positions (Q4) in one assertion. |
| `fit_predict_returns_clones_estimator` | Spy estimator recording `fit` calls on itself | The passed-in instance's `fit` is **never** called (count `0`); the fitted object is a distinct clone | Q14 per-fold `sklearn.base.clone`. |
| `fit_predict_returns_applies_sample_weight` | `sample_weight = [1,2,3,4,5]` (indexed as `X`), `train_pos=[0,1,2]` | Spy captures `sample_weight` values exactly `[1, 2, 3]` | Q14: `sample_weight.iloc[train_pos]`. |
| `fit_predict_returns_none_sample_weight` | `sample_weight=None` | `fit` is called without a weight; call succeeds and returns the same bet returns as the weighted-path fixture | Q14 (MyPipeline None-safe superset). |
| `fit_predict_returns_custom_position_fn` | `position_fn = lambda sides: sides * 2`, stub predicts `[+1, 0, -1]`, same returns as row 1 | Series `[0.06, 0.00, 0.08]` | Exactly 2x row 1. Pins that the seam is genuinely injectable (Q4). |
| `fit_predict_returns_raises_on_index_misalignment` | `event_returns` reindexed so its labels differ from `X` | raises `ValueError` | Fail-loud contract, Behavior §1. |
| `default_position_fn_signed_unit` | sides `[+1, -1, 0, +1]` | positions `[+1, -1, 0, +1]`, unchanged | Q4 signed-unit primary. |
| `walk_forward_anchoring_past_only` | Fixture A | Split 1 train positions `[0,1,2]`; split 2 train positions `[0,1,2,3,4,5]` | Behavior §3. Expanding and strictly-before, both verified in one fixture. Split 2's train strictly contains split 1's. |
| `walk_forward_purge_equivalence` | Fixture A, `embargo_pct = 0.0` | Engine's split-2 train index **equals** a direct call to `_purge_embargo(t1, test_times, bars, 0.0)` restricted to strictly-before candidates, where `test_times` is the single row `{T6: max(t1 over [6,7,8])}` and `bars = X.index` | Behavior §4. Guards the kernel coupling — the engine must not reimplement purge. |
| `walk_forward_single_dense_path` | Fixture A | `paths.shape == (6, 1)`; no NaN; index exactly `[T3..T8]` | Behavior §6. 6 = groups 1 and 2 (3 + 3). |
| `walk_forward_forwards_sample_weight` | Fixture A plus `sample_weight = [1..9]` indexed as `X` | At split 2, the spy's captured weight is exactly `[1,2,3,4,5,6]` (`sample_weight.iloc[[0..5]]`) | Q14, verified at WF level rather than only at the seam. |
| `walk_forward_thin_result` | Fixture A | `result.gates is None`; `result.config_hash == config.config_hash()`; `isinstance(result.created_at, pd.Timestamp)` | Behavior §6. Pins the thin-result decision structurally. |
| `walk_forward_scalars_exact` | Fixture A | `result.sharpe == pytest.approx(0.2279330, abs=1e-6)`; `result.psr == pytest.approx(metrics.psr(dense_path_values, sr_star=0.0), abs=1e-12)` | Dense path `[0.05,-0.04,0.01,0.02,-0.03,0.04]`; mean `0.0083333333`, `std(ddof=1) 0.0365604522`, SR `0.2279329939`. See the PSR note below. |
| `walk_forward_raises_on_misalignment` | Fixture A with `y` reindexed to differ from `X` | raises `ValueError` | Fail-loud contract, Behavior §1, at the public entry point. |
| `result_dataclasses_frozen` | Assign to `BacktestResult.sharpe` and to `GateResult.verdict` | both raise `dataclasses.FrozenInstanceError` | Frozen dataclasses, Contract above. |

### PSR oracle note

`sharpe` above is hand-computed and exact. `psr` is pinned as a **wiring assertion** — the engine's
contract is that it calls `metrics.psr` on the dense path with `sr_star = config.psr_benchmark`, not
that it re-derives the PSR formula. `metrics.psr`'s own numeric correctness is task-02's, already
built and covered.

A Gaussian-approximation oracle (`psr ~ Phi(SR * sqrt(n-1))`) was considered and **rejected for this
task**. The approximation holds only at skew 0 and non-excess kurtosis 3. Fixture A's dense path has
skew `-0.3902` and non-excess kurtosis `1.3162`, so the full PSR is `0.6870` against the
approximation's `0.6949` — a gap of `7.8e-3`, nearly eight times the `1e-3` tolerance the design
doc's appendix uses. Shortening or resymmetrizing the path does not close this: at `n = 6` the
sample kurtosis cannot reach 3, so the approximation is structurally unreachable at this scale.

The appendix's approximation oracles were computed at large `n` inside `metrics.py`'s own tests,
where they are sound. Reproducing that style here would encode a false precision and fail a correct
implementation. Pinning `psr` as a wiring assertion is the honest form of this oracle at `n = 6`.

## Definition of Done

- [ ] Commit 1: `tests/backtest/test_engine.py` with exactly the 15 functions in `test_manifest`,
      red for the right reason (`ImportError`/`AttributeError` on the not-yet-written module, not a
      logic bug in a test).
- [ ] Commit 2: `backtest/engine.py` only. No test file touched. Docker green.
- [ ] Coverage of `backtest/engine.py` >= 100%, zero pragmas — genuinely unreachable branches are
      deleted, not pragma'd.
- [ ] `scripts/test.sh tests/backtest/test_engine.py` passes clean.
- [ ] `backtest/engine.py` imports nothing from `backtest.ledger`, `backtest.storage`,
      `backtest.paths`, or `backtest.synthetic` — those arrive in task-08/09.
- [ ] Commit 3: `RUN-REPORT.md` emitted by the pre-push hook.

## Out of Scope

See `out_of_scope` in frontmatter above — the frontmatter list is authoritative for tooling. In
short: CPCV, synthetic, gate composition, ledger registration, the cross-mode spy test, rolling
walk-forward, proba sizing, and `__init__.py` exports all belong to other tasks.

## Escalation Triggers

See `escalation_triggers` in frontmatter above. Hitting any of these is an automatic STOP per
PROTOCOL.md §5 — commit the WIP state and a `*.BLOCKED.md`, do not attempt to reason around it. In
particular: if `_purge_embargo` or `MyPipeline` does not match the signature pinned in the Contract,
that is a spec bug, not something to adapt to.
