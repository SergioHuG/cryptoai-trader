# Backtest Module — Claude Code Context

## Purpose
`backtest/` is the AFML **statistical strategy-evaluation layer** — the overfitting
gate the `Backtest → Paper Trade → Live` pipeline hinges on. It is **not** a
candle/tick execution simulator (the old EMA-replay design is retired). Its core
seam is **return paths**: a strategy is scored by running it through leak-proof
cross-validation schemes, reconstructing full-length out-of-sample return paths,
and subjecting those paths to a battery of pure statistical tests (PSR, DSR, PBO,
HHI, DD/TuW) plus a synthetic null.

The engine is **price / Decimal / seam-free**: per-event realized returns
(`event_returns`) are an *input*, computed upstream in the research layer. All
pandas/numpy/sklearn stays inside `research/` and `backtest/`; nothing here
crosses the `SignalPacket` Decimal seam.

## Module layout (8 modules, built in this order)
| Module | Responsibility |
|---|---|
| `config.py` | `BacktestConfig` — frozen recipe dataclass; `config_hash()` is the ledger/result dedup key. Mirrors `LabelConfig`. |
| `metrics.py` | Pure statistics on return arrays / a (T×N) trial matrix: `sharpe`, `sharpe_lo`, `psr`, `expected_max_sharpe`, `deflated_sharpe`, `pbo`, `hhi_*`, `drawdown_tuw`. (scipy.stats) |
| `paths.py` | Pure combinatorial φ-path reconstruction for CPCV — `reconstruct_paths(...)` → dense `paths` matrix. |
| `synthetic.py` | O-U / AR(1)-on-returns null: `fit_ou` (OLS) + `simulate` (seeded `default_rng`, mean-init, fixed burn-in). |
| `ledger.py` | Pure trial-record model + reductions (`n_trials`, `sr_variance`, (T×N) matrix assembly). Dedup by `config_hash`. |
| `storage.py` | HDF5 append/load for the trial ledger and result verdicts (`backtest/results/backtest.h5`). |
| `engine.py` | Composes all of the above. `run_walk_forward` / `run_cpcv` / `run_synthetic` → `BacktestResult`. Single shared `_fit_predict_returns` seam. |
| `__init__.py` | Public API surface. |

## The return-path contract (keystone)
- **Unit** = per-event bet return, indexed by event `t0` (1:1 with the `(X, y, t1)`
  rows the labels/weights/validation layers speak — no re-indexing seam).
- **`paths: pd.DataFrame`** — index = full OOS `t0` timeline; columns = `path_id`;
  values = per-event bet returns. Dense per mode (CPCV: φ cols; WF: 1 col;
  MC: `n_sims` cols). This is the **single cross-module currency** — `metrics`
  never branches on mode.
- **Position:** default unit-sized primary — `clf.predict(X_test)` → side
  ∈ {−1,0,+1} → `bet_return = position × event_returns`. `position_fn` injectable
  (proba-sizing / meta-labeling arrive later as injected fns). Estimator contract
  = **`predict()` only**.

## Evaluation modes
- **Walk-Forward** — anchored / expanding-window, past-only (train = all events
  strictly before the test block, minus purge). Reuses the imported private
  `_purge_embargo` from `research.validation.purge`. Single dense path.
- **CPCV** — drives `CombinatorialPurgedKFold` (untouched), reconstructs
  `φ = C(N−1, k−1)` full-length OOS paths ("j-th test-occurrence of group g →
  path j"). Primary gate substrate.
- **O-U Monte Carlo** — matched-moment AR(1)-on-returns null (price-free);
  provides null-percentile context only.

## Trial ledger & gate
- **Ledger:** append-only, cross-session HDF5; a *trial* = one real-data backtest
  of a distinct config, deduped by `config_hash`. Supplies DSR `N`/`V` and the
  PBO (T×N) matrix. Synthetic runs do **not** log. "You can't hide trials."
- **Gate:** hard gates PSR / DSR / PBO, each **tri-state** PASS / FAIL /
  **UNAVAILABLE(<2 trials)**. Verdict = AND over available hard gates;
  **provisional** while any UNAVAILABLE, **final** at ≥2 trials (self-heals).
  Advisory (non-blocking): Sharpe-Lo, HHI×3, max-DD, max-TuW, synthetic
  percentile. CPCV gate = **median-path PSR ≥ 0.95**.

## Critical rules
- **Design-locked** (`/grill-me` Q1–Q15, ADR-029–ADR-044). Do not re-litigate;
  any deviation is a new ADR.
- **AFML-faithful, deviations flagged:** `predict()`-only (vs `cv_score`'s
  proba), per-fold `clone` (vs re-fit) — the latter unblocks parked CPCV
  parallelization.
- **Fail-loud, never silent fallback:** index misalignment, missing HDF5 keys,
  and PBO with `<2` trials / `S`-odd / `T<S` all raise.
- **Single shared seam:** all three modes route through `_fit_predict_returns`;
  `paths: pd.DataFrame` is the one cross-module currency.
- **No prices / no Decimal** in this package — `event_returns` is an input.
- **Docker (3.11 / pandas 2.2) authoritative.** 100% module coverage, zero pragmas.
- Results (`results/*.h5`) are never committed — only engine + metrics code.

## Testing requirements
- Red-first TDD; verified numeric oracles baked in — PSR ≈ Φ(SR·√(n−1));
  DD on `1→1.2→0.9→1.5` = 0.25; HHI uniform ≈ 0, concentrated ≈ 0.96;
  PBO noise ≈ 0.45 / edge ≈ 0 / overfit ≈ 0.99; O-U same-seed byte-identical.
- Stub estimators (`predict()` returns a known side vector) for exact
  bet-return columns.
- Partition-equivalence test guards the CPCV↔splitter coupling; a single-seam
  spy test guards the `_fit_predict_returns` invariant.
