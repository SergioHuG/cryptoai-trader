# Phase 2 — `backtest/` Build Sequence

> **Purpose:** Full context carry-over for the next session. The `/grill-me` design-lock pass is **complete** (Q1–Q15, no code written). This document is the build-actionable contract + ordered TDD cadence. Read top-to-bottom before touching code.

**Goal:** Build the AFML statistical strategy-evaluation layer — the overfitting gate the `Backtest → Paper Trade → Live` pipeline hinges on.

**Architecture:** 8-module package. Return-path-centric, price/Decimal/seam-free. Three eval modes (Walk-Forward, CPCV, O-U Monte Carlo) → a dense `paths` matrix → pure statistical metrics (Sharpe-Lo, PSR, DSR, PBO, HHI, DD/TuW) → a persisted tri-state gate verdict. A persistent, hash-deduped trial ledger supplies the multiple-testing bookkeeping (DSR `N`/`V`, PBO matrix).

**Tech stack:** Python 3.11, pandas 2.2, numpy, **scipy (new dep this phase)**, scikit-learn, HDF5/pytables. Docker (`python:3.11-slim`) is the authoritative test env. Sandbox (3.12/pandas 3.0) is exploration only.

---

## Startup checklist (run first, next session)

- [ ] Clone / pull `SergioHuG/cryptoai-trader`.
- [ ] `git checkout main && git pull && git checkout -b feature/phase2-backtest` (branch off `main` at the validation merge).
- [ ] Confirm **650-test / 100%-cov Docker baseline** (`./scripts/test.sh`), Phase 1 green.
- [ ] Confirm Phase 1 packages intact: `data/bars.py`, `research/labels/`, `research/weights/`, `research/validation/`.
- [ ] `backtest/` currently holds only `__init__.py`, `CLAUDE.md` (to be rewritten), `results/.gitkeep`.

---

## The locked contract (Q1–Q15 reference)

**Identity (Q1).** `backtest/` = AFML *statistical evaluation layer*, NOT a candle/tick execution simulator. Core seam = return paths. A thin `(splits, estimator, X, y, event_returns) → paths` adapter feeds them. Old `CLAUDE.md` (EMA-replay) is **superseded**.

**Return unit & path shape (Q3).**
- Unit = **per-event bet return, indexed by event `t0`** (1:1 with the `(X, y, t1)` rows the labels/weights/validation layers speak — no re-indexing seam).
- **`paths: pd.DataFrame`** — index = full OOS `t0` timeline, columns = `path_id`, values = per-event bet returns. Dense per mode (CPCV: φ cols, no NaN; WF: 1 col; MC: `n_sims` cols). This is the **single cross-module currency**; `metrics` never branches on mode.
- **`event_returns: pd.Series` is an engine INPUT** (return realized over each event's `[t0,t1]`, computed upstream). Engine never touches prices/Decimal/the seam. Fail-loud on index misalignment.

**Position mapping (Q4).**
- Default = **unit-sized primary**: `clf.predict(X_test)` → side ∈ {−1,0,+1} → `position`; `bet_return = position × event_returns`. Predicted `0` → flat/zero-return event.
- `position_fn` **injectable** (default = "predict → signed unit"); proba-sizing (Ch.10) + meta-labeling become injected `position_fn`s later.
- Estimator contract = **`predict()` only** (deliberate, documented divergence from `cv_score`'s `predict_proba`).

**Modes.**
- **WF (Q6)** = **anchored / expanding-window, past-only** (train = all events strictly before the test block, minus purge). Single dense path. Reuses **`_purge_embargo`** (imported private from `research.validation.purge`) + engine-applied "strictly-before" anchoring mask. Rolling fixed-window WF **deferred**.
- **CPCV (Q5)** = `paths.py` reconstructs φ full-length OOS paths. `φ = (k/N)·C(N,k) = C(N−1,k−1)`. Assignment: **"j-th test-occurrence of group g → path j"** (verified dense, NaN-free). `CombinatorialPurgedKFold` **untouched**; a partition-equivalence test guards the coupling. Engine re-splits each split's combo-ordered `test_pos` into k group blocks → `{split_idx: {group_id: bet_return_series}}`; `paths.py` does pure path-id bookkeeping. `np.array_split` partition helper stays inline.
- **O-U MC (Q7)** = **matched-moment AR(1)-on-returns null** (price-free). O-U discretizes to `r_t = c + φ·r_{t-1} + σ·ε_t`; fit via **OLS** to the strategy's realized return series; generate `n_sims` synthetic paths via `numpy.random.default_rng(seed)` + **mean-init + fixed deterministic burn-in**. Ch.13 OTR (price-level, PT/SL mesh) **deferred**.

**Metrics (`metrics.py`, pure; Q8–Q10).**
- **Two Sharpes:** per-bet **non-annualized** `mean/std(ddof=1)` feeds PSR/DSR; **Sharpe-Lo** (autocorrelation-adjusted `η(q) = q/√(q + 2Σ(q−k)ρ_k)`, `q` = realized bets/year) is a **flagged-approximate annualized diagnostic** (autocorr over bet index, irregular spacing).
- **PSR** = `Φ((SR − SR*)·√(n−1) / √(1 − γ₃·SR + ((γ₄−1)/4)·SR²))`; `SR*` default 0; `γ₃` skew, `γ₄` **non-excess** kurtosis (normal=3); bias-corrected sample moments. Gate `PSR(0) ≥ 0.95`.
- **DSR** = PSR with `SR*_deflated = √V · [(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]`, γ = Euler–Mascheroni (0.5772…), `V` = variance of trial SRs, `N` = trial count. **`V`/`N` injected** (from ledger). Gate `DSR ≥ 0.95`.
- **PBO via CSCV (Q9)** = pure fn on a **(T×N) trial matrix**. Partition rows into `S` groups (`np.array_split`), for each `C(S,S/2)` split: IS-best-by-Sharpe → its OOS relative rank `ω = rank/(N+1)` → logit `λ = ln(ω/(1−ω))`; **PBO = mean(λ ≤ 0)**. Returns PBO + λ array. `S=16`/even/`N≥2`/`T≥S` **fail-loud**. Built now, **live at ≥2 trials** (fail-loud until then; engine wraps as UNAVAILABLE).
- **HHI ×3 (Q10)** = normalized `(Σwᵢ² − 1/n)/(1 − 1/n)`, `wᵢ = rᵢ/Σr`: `hhi_positive` (`r>0`), `hhi_negative` (`r<0`), `hhi_temporal` (bet counts per calendar period, default **monthly**, over `t0`). NaN when `n ≤ 2`; raise on empty.
- **DD / TuW (Q10)** = **compounded** equity `(1+r).cumprod()`; `dd` = HWM→trough decline fractions; `tuw` = years between HWM and recovery. Return full series; engine reduces to max/percentiles.
- All metrics **per-path pure**; engine aggregates across φ paths.

**Ledger & state (Q11).**
- Append-only, cross-session, **HDF5** persistent (the "you can't hide trials" ethic).
- `ledger.py` = pure trial-record model + reductions (`N`, `V`, (T×N) matrix assembly). `storage.py` = HDF5 append/load only.
- **Trial** = one real-data backtest (WF or CPCV) of a **distinct config**. Record: `trial_id`, `config_hash`, `created_at`, `mode`, `sharpe`, `n_bets`, OOS **bet-return series** (WF: the path; CPCV: **per-event mean across φ paths**). Synthetic runs **do not log**.
- **Dedup by `config_hash`** → `N` = distinct configs. PBO matrix **inner-joins** trial series on common `t0`, fail-loud if overlap `< S`.
- **Auto-append** on WF/CPCV runs, `register_trial=True` opt-out.

**Config & verdict (Q12).**
- `BacktestConfig` = frozen dataclass mirroring `LabelConfig` exactly: `__post_init__` validation + `config_hash()` = `sha256(json.dumps(asdict, sort_keys=True))[:12]` with `schema_version` folded into the hash. **Recipe-knob fields only** (locators excluded). This hash **is** the ledger dedup key.
  - Fields: `n_groups`, `n_test_groups`, `n_wf_splits`, `embargo_pct`, `n_sims`, `seed`, `burn_in`, `pbo_partitions` (S), `psr_benchmark` (0), `gate_threshold` (0.95), `pbo_threshold` (0.5), `schema_version`.
- **Verdict persists** to the backtest-owned HDF5 store keyed by `config_hash`. `BacktestResult` carries scalars + tri-states + per-mode breakdown + `config_hash` + `created_at` + the **dense `paths`**.

**Gate composition (Q13).**
- **Hard gates:** PSR / DSR / PBO — each **tri-state** PASS / FAIL / **UNAVAILABLE(<2 trials)**. Verdict = AND over *available* hard gates; **provisional** while any UNAVAILABLE, **final** at ≥2 trials (self-heals, no code change).
- **Advisory (non-blocking):** Sharpe-Lo, HHI ×3, max-DD, max-TuW, synthetic-null percentile.
- **CPCV = primary gate substrate:** evaluate PSR **per φ-path**, gate on **median-path PSR ≥ 0.95** (report min-path as conservative view). Trial `sharpe` recorded = **median path Sharpe** (feeds DSR `V`); stored series = per-event mean (PBO alignment).
- **WF** = corroborating single-path PSR. **Synthetic** = null-percentile context only.
- Verdict top-level: `{"provisional_pass" | "provisional_fail" | "pass" | "fail"}`.

**Fit contract & determinism (Q14).**
- `_fit_predict_returns(estimator, X, y, event_returns, train_pos, test_pos, position_fn, sample_weight=None)` — the single shared helper all three modes call.
- Reuse **public `MyPipeline`** from `research.validation.cv` (import, no privacy issue): wrap bare estimator in `MyPipeline([("clf", clf)])`, slice `sample_weight.iloc[train_pos]`, `clf.fit(X_train, y_train, sample_weight=w_train)`, then **`clf.predict(X_test)`** → `position_fn` → `× event_returns.iloc[test_pos]`.
- `sample_weight` **optional** (default None; MyPipeline is None-safe superset).
- **Clone estimator per fold** (`sklearn.base.clone`) — flagged divergence from `cv_score`'s re-fit; unblocks parked `mp_engine` parallelization.
- **Determinism:** verdict reproducible given `(data, seeded estimator, config.seed)`. Engine seeds only the synthetic RNG; estimator seeding is the caller's.

---

## Module dependency graph

```
config.py      (no deps — frozen dataclass + hash)
metrics.py     (scipy.stats; pure functions on returns arrays / (T×N) matrix)
paths.py       (numpy; pure φ combinatorial assembly)
synthetic.py   (numpy default_rng; O-U/AR(1) fit + generate)
      │
ledger.py      (pandas; pure N/V/matrix reductions over records)
storage.py     (pytables; HDF5 append/load for ledger + results)
      │
engine.py      (composes ALL above + imports:
                  research.validation.purge._purge_embargo,
                  research.validation.cv.MyPipeline,
                  research.validation.cpcv.CombinatorialPurgedKFold)
      │
__init__.py    (public API surface)
```

**Build order:** `config → metrics → paths → synthetic → ledger → storage → engine → __init__`.
Each module: **red tests → Docker red confirm (Brooke runs, reports count) → implement → Docker green confirm → git commands (Claude) → commit/push (Brooke) → next.**

---

## Step 0 — Infra (do before any module)

**0a. scipy — two independent mechanisms (both required):**
- **Docker gate** (`scripts/test.sh`): append `scipy` to the pip-install line. This is what the authoritative env actually installs; without it, red/green runs can't import scipy. (Optional side-cleanup: the line currently duplicates `pandas numpy tables  pandas numpy tables` — leave or tidy, Brooke's call.)
- **Local env:** `poetry lock --no-update` (heals the pre-existing scikit-learn drift — it's in `pyproject.toml` but absent from `poetry.lock`), then add `scipy` to `[tool.poetry.dependencies]`, then `poetry lock --no-update` again. *Local reproducibility only — does not affect the Docker gate.*

**0b. `.coveragerc`:** remove the `backtest/*` line from `[run] omit`. Phase 2 held to the **100% module-coverage bar**. Existing `[report] exclude_lines` already covers `__repr__` / `raise NotImplementedError`; research hits 100% with **zero pragmas** — target genuine 100%, add no new `# pragma: no cover`.

**0c. Backtest-owned HDF5 store:** `backtest/results/backtest.h5` (the `.gitkeep` is already there). **Local key constants in `backtest/storage.py`** — NOT `data/hdf5_keys.py` (those are symbol/threshold→group for the bars/labels file; backtest is keyed by `config_hash`, a different axis). Two namespaces: `/ledger/...` and `/results/cfg_{hash}`.

**0d. Rewrite `backtest/CLAUDE.md`** to the locked spec above (retire the EMA-replay description).

---

## Build sequence — per module

### Task 1 — `config.py`
**Files:** Create `backtest/config.py`; Test `tests/backtest/test_config.py`.
**Contract:** Frozen `BacktestConfig` dataclass, fields per Q12 above. `__post_init__` validates each (mirror `LabelConfig`'s `> 0` / `>= n` guards; `n_test_groups < n_groups`; `pbo_partitions` even; `0 ≤ *_threshold ≤ 1`; `n_sims ≥ 1`; `burn_in ≥ 0`). `config_hash()` = `sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]`, `schema_version` inside.
**Test targets:** valid construction; each validation raises `ValueError` with a clear message; `config_hash` stable & 12-hex; two configs differing only in a knob hash differently; `schema_version` bump changes the hash; frozen (assignment raises `FrozenInstanceError`).

### Task 2 — `metrics.py`
**Files:** Create `backtest/metrics.py`; Test `tests/backtest/test_metrics.py`.
**Contract (pure functions, scipy.stats):**
- `sharpe(returns) -> float` (per-bet, `mean/std(ddof=1)`).
- `sharpe_lo(returns, q) -> float` (autocorrelation-adjusted annualized; `q` = bets/year).
- `psr(returns, sr_star=0.0) -> float` (Bailey–LdP; `skew`/`kurtosis` `bias=False`, kurtosis `fisher=False`; `norm.cdf`).
- `expected_max_sharpe(var_sr, n_trials) -> float` (Euler–Mascheroni; `norm.ppf`).
- `deflated_sharpe(returns, sr_variance_across_trials, n_trials) -> float`.
- `pbo(returns_matrix, n_partitions=16) -> (float, np.ndarray)` (CSCV; fail-loud S-even/N≥2/T≥S).
- `hhi_positive/negative/temporal(...)`; `drawdown_tuw(returns) -> (dd, tuw)`.
**Test oracles (verified in the lock session — use as exact expected values):**
- PSR Gaussian: full PSR ≈ `Φ(SR·√(n−1))` (match to ~1e-3).
- PSR monotonic ↑ in n; negative skew lowers PSR vs symmetric at equal SR.
- `expected_max_sharpe` ↑ in N (e.g. N=10 → 0.157, N=100 → 0.253 at var_sr=0.01); DSR ↓ in N.
- PBO regimes: all-noise ≈ 0.45–0.5; one genuine edge ≈ 0.0; partition-level overfit ≈ 0.99.
- HHI: uniform ≈ 0.0; concentrated (`[0.001]×9 + [0.5]`) ≈ 0.96.
- DD: equity `1→1.2→0.9→1.5` → max DD = **0.25** exactly.
- AR(1)-context (for synthetic, Task 4) recovery/determinism (see appendix).
**Edge cases:** empty input raises; constant returns (std 0) handled fail-loud; PBO S-odd/N<2/T<S raise.

### Task 3 — `paths.py`
**Files:** Create `backtest/paths.py`; Test `tests/backtest/test_paths.py`.
**Contract:** `reconstruct_paths(per_split_group_returns: dict[int, dict[int, pd.Series]], n_groups, n_test_groups) -> pd.DataFrame` (dense φ-column matrix, index = OOS `t0` timeline). Estimator-free; pure combinatorial. "j-th test-occurrence of group g → path j".
**Test targets:** hand-built `{split:{group:series}}` for small (N=4,k=2 → φ=3; N=6,k=2 → φ=5) → assert exact φ-column matrix, dense (no NaN), each event once per column. **Partition-equivalence test:** the `np.array_split(np.arange(n), n_groups)` + `itertools.combinations` order reproduces exactly the `test_pos` unions `CombinatorialPurgedKFold.split()` yields.

### Task 4 — `synthetic.py`
**Files:** Create `backtest/synthetic.py`; Test `tests/backtest/test_synthetic.py`.
**Contract:** `fit_ou(returns) -> OUParams(c, phi, sigma, mu, half_life)` (OLS of `r_t` on `r_{t−1}`); `simulate(params, n_obs, n_sims, seed, burn_in) -> pd.DataFrame` (`default_rng(seed)`, mean-init, fixed burn-in; cols = `sim_id`).
**Test oracles (verified):** param recovery (φ 0.35 → ~0.366, σ exact at n=5000); **same seed → byte-identical paths**, diff seed → differ; white-noise → φ ≈ 0.
**Edge:** `|φ| ≥ 1` (non-stationary) → half_life inf / fail-loud as appropriate; too-short series raises.

### Task 5 — `ledger.py`
**Files:** Create `backtest/ledger.py`; Test `tests/backtest/test_ledger.py`.
**Contract:** `TrialRecord` dataclass (fields per Q11). Pure reductions over `list[TrialRecord]`: `n_trials`, `sr_variance`, `assemble_matrix() -> (T×N) DataFrame` (inner-join on `t0`, fail-loud if overlap `< S`). Dedup by `config_hash`.
**Test targets:** N/V over known records; matrix assembly aligns & inner-joins; dedup drops same-hash; insufficient-overlap raises.

### Task 6 — `storage.py`
**Files:** Create `backtest/storage.py`; Test `tests/backtest/test_storage.py`.
**Contract:** Local HDF5 key constants; `append_trial`, `load_ledger`, `save_result`, `load_result` against `backtest/results/backtest.h5`. Fail-loud on missing keys (mirror `load_labels` `KeyError`/`FileNotFoundError` posture).
**Test targets:** round-trip trial append/load and result save/load against a **temp .h5**; missing-key raises; dedup persists.

### Task 7 — `engine.py`
**Files:** Create `backtest/engine.py`; Test `tests/backtest/test_engine.py`.
**Contract:** `_fit_predict_returns(...)` (Q14). `run_walk_forward(...)`, `run_cpcv(...)`, `run_synthetic(...)` → `BacktestResult`. WF anchoring mask + `_purge_embargo` reuse. CPCV: drive `CombinatorialPurgedKFold`, re-split `test_pos` into group blocks, call `paths.reconstruct_paths`, per-path PSR, median gate. Auto-append trials (dedup). Tri-state gate assembly, provisional/final.
**Test targets:** **stub estimator** (`predict()` returns known side vector) → exact bet-return column; WF anchoring correctness; **`_purge_embargo` equivalence** (WF purge == direct kernel call); CPCV group-space re-split equivalence; **single-seam spy** — all three modes route through `_fit_predict_returns`; tri-state UNAVAILABLE at <2 trials → provisional verdict; median-path PSR gate; determinism (same seed → same synthetic verdict).

### Task 8 — `__init__.py`
**Files:** Modify `backtest/__init__.py`; Test `tests/backtest/test_import_boundary.py` (or extend existing boundary test).
**Contract:** Export `BacktestConfig`, `BacktestResult`, `run_walk_forward`, `run_cpcv`, `run_synthetic`, ledger read/append surface. `__all__` set. Metrics importable via `backtest.metrics` (not all hoisted).
**Test targets:** public names importable; `__all__` matches; no seam/Decimal leakage across the research boundary (import-boundary test stays green).

---

## Verified numeric-oracle appendix (exact expected values from the lock session)

```
φ path assignment (dense, NaN-free) across (N,k):
  (6,2)→φ5  (5,2)→φ4  (6,3)→φ10  (4,2)→φ3  (10,2)→φ9   [n_splits·k = N·φ ✓]

PSR/DSR:
  Gaussian PSR ≈ Φ(SR·√(n−1))              (full 0.9993 vs approx 0.9994)
  PSR ↑ in n                                (n=500 0.654 → n=4000 0.869)
  expected_max_sharpe(var=0.01): N=10 0.157, N=100 0.253   (↑ in N)
  DSR ↓ in N; negative skew lowers PSR

PBO / CSCV (S=16):
  all-noise ≈ 0.45      one-genuine-edge ≈ 0.00      partition-overfit ≈ 0.998

HHI (normalized):
  uniform ≈ 0.000       concentrated([0.001]×9+[0.5]) ≈ 0.961

DD/TuW (compounded):
  equity 1→1.2→0.9→1.5  →  max DD = 0.250

O-U / AR(1) (OLS fit, default_rng):
  recovery: φ 0.35→~0.366, σ 0.01→0.0100 (n=5000)
  determinism: same seed → identical; diff seed → differ
  white-noise → φ ≈ 0
```

---

## Carry-forward ledgers

**Deferral ledger (→ ADRs at `/throughline`):**
1. Probability-based position sizing (AFML Ch.10) + meta-labeling via injectable `position_fn`.
2. Rolling fixed-window walk-forward (config flag, additive).
3. AFML Ch.13 OTR / price-level O-U + PT/SL mesh.
4. `mp_engine` parallelization of the CPCV fit loop (now unblocked by per-fold cloning).

**Revisit-when-data-exists:**
- PBO scope-reality flag: live threshold, default `S`, ranking metric — retune once real strategies/data accumulate. Provisional→final gate flips automatically at ≥2 trials.
- `pbo_threshold` 0.5 → tighter (e.g. 0.2) once a trial population exists.

**Still pending (non-blocking, pre-existing):** ADR-018 (labels orchestrator), ADR-019 (OHLC high/low touch) write-ups.

**House cadence (unchanged):**
- Never commit to `main`; always `feature/phase2-backtest`.
- Red tests → Docker red confirm (Brooke) → implement → Docker green confirm → Claude gives git commands → Brooke commits/pushes.
- Docker (3.11/pandas 2.2) authoritative for counts/coverage; sandbox is exploration only.
- Claude states target file path with every artifact; all diffs shown before writes; Brooke applies local changes.
