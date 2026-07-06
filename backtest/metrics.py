"""backtest.metrics -- pure statistical metrics on return arrays / a (T x N)
trial matrix (Phase 2, Task 2).

No mode-awareness lives here: every function operates on already-realized
per-bet returns (or, for :func:`pbo`, a trial matrix of them) and never
branches on WF/CPCV/synthetic. All functions are pure -- no I/O, no
persistence, no engine-level aggregation across CPCV's phi paths (that is
`engine.py`'s job).

Formulas (locked, Q8-Q10):
  * PSR (Bailey-Lopez de Prado):
      PSR = Phi((SR - SR*) * sqrt(n-1) / sqrt(1 - skew*SR + ((kurt-1)/4)*SR^2))
    with `skew`/`kurt` bias-corrected sample moments (scipy `bias=False`),
    `kurt` NON-EXCESS (scipy `fisher=False`, normal == 3).
  * DSR = PSR with SR* replaced by the expected max Sharpe across N trials:
      SR*_deflated = sqrt(V) * [(1-gamma)*Phi^-1(1-1/N) + gamma*Phi^-1(1-1/(N*e))]
    gamma = Euler-Mascheroni constant, V = variance of trial Sharpes, N =
    trial count (both injected from the ledger).
  * PBO via CSCV: partition the (T x N) trial matrix's rows into S groups,
    for every C(S, S/2) IS/OOS split find the IS-best trial by Sharpe, look
    up its OOS *relative rank* omega = rank/(N+1), logit lambda =
    ln(omega/(1-omega)); PBO = mean(lambda <= 0).
  * HHI (normalized): (sum(w_i^2) - 1/n) / (1 - 1/n), w_i = r_i / sum(r) --
    computed over positive returns, negative returns, or per-calendar-period
    bet counts (temporal, default monthly).
  * Drawdown / TuW: dense per-observation series over the compounded
    equity curve (1+r).cumprod() -- `dd` = fractional decline from the
    running high-water mark at every point; `tuw` = years elapsed since the
    most recent high-water mark (0 exactly at a new high). Engine reduces
    both to max/percentiles.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "sharpe",
    "sharpe_lo",
    "psr",
    "expected_max_sharpe",
    "deflated_sharpe",
    "pbo",
    "hhi_positive",
    "hhi_negative",
    "hhi_temporal",
    "drawdown_tuw",
]

_EULER_GAMMA = np.euler_gamma
_SECONDS_PER_YEAR = 365.25 * 24 * 3600.0


def sharpe(returns) -> float:
    """Per-bet, non-annualized Sharpe ratio: ``mean / std(ddof=1)``.

    Raises
    ------
    ValueError
        If ``returns`` is empty, or has zero variance (a constant series --
        fail-loud rather than dividing by zero).
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        raise ValueError("sharpe: returns must be non-empty.")
    std = r.std(ddof=1)
    if std == 0:
        raise ValueError(
            "sharpe: returns have zero variance (a constant series) -- "
            "Sharpe is undefined."
        )
    return float(r.mean() / std)


def sharpe_lo(returns, q: float) -> float:
    """Autocorrelation-adjusted, annualized Sharpe (Lo 2002).

    ``eta(q) = q / sqrt(q + 2 * sum_{k=1}^{q-1} (q-k) * rho_k)``, where
    ``rho_k`` is the lag-``k`` autocorrelation of ``returns`` and ``q`` is
    the realized bet count per year. Returns ``sharpe(returns) * eta(q)``.
    This is a flagged-APPROXIMATE annualized diagnostic (advisory only,
    never a hard gate): autocorrelation is computed over the bet index,
    not real time, so irregular bet spacing is not corrected for.

    When ``returns`` has (near) zero autocorrelation at every lag,
    ``eta(q) -> sqrt(q)``, recovering the standard square-root-of-time
    annualization.

    Raises
    ------
    ValueError
        If ``returns`` is empty or ``q < 1``.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        raise ValueError("sharpe_lo: returns must be non-empty.")
    if q < 1:
        raise ValueError(f"sharpe_lo: q must be >= 1, got {q!r}.")
    sr = sharpe(r)
    s = pd.Series(r)
    rho_sum = 0.0
    for k in range(1, int(q)):
        rho_k = s.autocorr(lag=k)
        if np.isnan(rho_k):
            rho_k = 0.0
        rho_sum += (q - k) * rho_k
    eta = q / np.sqrt(q + 2 * rho_sum)
    return float(sr * eta)


def psr(returns, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey-Lopez de Prado).

    ``Phi((SR - sr_star) * sqrt(n-1) / sqrt(1 - skew*SR + ((kurt-1)/4)*SR^2))``.
    ``skew``/``kurt`` are bias-corrected sample moments (``bias=False``);
    ``kurt`` is NON-EXCESS (``fisher=False`` -- a normal distribution gives
    ``kurt == 3``, not 0).

    Raises
    ------
    ValueError
        If ``returns`` has fewer than 2 observations (propagated from
        :func:`sharpe`'s empty/constant-series guards, plus an explicit
        ``n >= 2`` check here).
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 2:
        raise ValueError(f"psr: need at least 2 observations, got n={n!r}.")
    sr = sharpe(r)
    skew = stats.skew(r, bias=False)
    kurt = stats.kurtosis(r, bias=False, fisher=False)
    numerator = (sr - sr_star) * np.sqrt(n - 1)
    denom = np.sqrt(1 - skew * sr + ((kurt - 1) / 4) * sr**2)
    return float(stats.norm.cdf(numerator / denom))


def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """Expected maximum Sharpe ratio across ``n_trials`` independent trials.

    ``sqrt(var_sr) * [(1-gamma)*Phi^-1(1-1/N) + gamma*Phi^-1(1-1/(N*e))]``,
    ``gamma`` = Euler-Mascheroni constant. This is the deflated-Sharpe
    benchmark ``SR*`` fed into :func:`psr` by :func:`deflated_sharpe`.
    Increasing in ``N`` -- more trials raise the bar a genuine strategy
    must clear before it is credited as skill rather than luck.

    Raises
    ------
    ValueError
        If ``n_trials < 1`` or ``var_sr < 0``.
    """
    if n_trials < 1:
        raise ValueError(
            f"expected_max_sharpe: n_trials must be >= 1, got {n_trials!r}."
        )
    if var_sr < 0:
        raise ValueError(
            f"expected_max_sharpe: var_sr must be >= 0, got {var_sr!r}."
        )
    n = n_trials
    term1 = (1 - _EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
    term2 = _EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e))
    return float(np.sqrt(var_sr) * (term1 + term2))


def deflated_sharpe(
    returns, sr_variance_across_trials: float, n_trials: int
) -> float:
    """Deflated Sharpe Ratio: PSR benchmarked against the expected max
    Sharpe across ``n_trials`` (the multiple-testing correction).

    ``deflated_sharpe(r, V, N) == psr(r, sr_star=expected_max_sharpe(V, N))``.
    Decreasing in ``N`` at fixed ``V`` -- more trials make the same
    realized Sharpe less convincing.
    """
    sr_star = expected_max_sharpe(sr_variance_across_trials, n_trials)
    return psr(returns, sr_star=sr_star)


def pbo(returns_matrix, n_partitions: int = 16) -> tuple:
    """Probability of Backtest Overfitting via CSCV (Bailey et al.).

    Partitions the ``(T, N)`` trial-return matrix's rows into
    ``n_partitions`` (``S``) contiguous groups. For every combinatorial
    split choosing ``S/2`` groups as in-sample (IS) and the complementary
    ``S/2`` as out-of-sample (OOS): finds the IS-best trial by Sharpe,
    looks up that trial's OOS relative rank ``omega = rank / (N+1)``, and
    computes the logit ``lambda = ln(omega / (1 - omega))``.
    ``PBO = mean(lambda <= 0)`` -- the fraction of splits in which the
    IS-best trial performed at or below the OOS median (the overfitting
    signature).

    Parameters
    ----------
    returns_matrix:
        ``(T, N)`` array-like -- ``T`` time periods (rows), ``N`` distinct
        trial configs (columns).
    n_partitions:
        ``S``, the CSCV partition count. Must be even and ``>= 2``.

    Returns
    -------
    tuple[float, np.ndarray]
        ``(pbo_value, lambdas)`` -- the PBO estimate and the per-split
        logit array (length ``C(S, S/2)``).

    Raises
    ------
    ValueError
        If ``returns_matrix`` is not 2D, ``n_partitions`` is odd or < 2,
        the matrix has fewer than 2 trial columns (``N < 2``), or fewer
        rows than partitions (``T < S``).
    """
    r = np.asarray(returns_matrix, dtype=float)
    if r.ndim != 2:
        raise ValueError(
            f"pbo: returns_matrix must be 2D (T x N), got ndim={r.ndim!r}."
        )
    t, n = r.shape
    s = n_partitions
    if s < 2:
        raise ValueError(f"pbo: n_partitions (S) must be >= 2, got {s!r}.")
    if s % 2 != 0:
        raise ValueError(f"pbo: n_partitions (S) must be even, got {s!r}.")
    if n < 2:
        raise ValueError(
            f"pbo: returns_matrix must have >= 2 trial columns, got N={n!r}."
        )
    if t < s:
        raise ValueError(
            f"pbo: returns_matrix must have >= S rows, got T={t!r}, S={s!r}."
        )

    groups = np.array_split(np.arange(t), s)
    half = s // 2
    lambdas = []

    for is_groups in itertools.combinations(range(s), half):
        is_set = set(is_groups)
        oos_groups = [g for g in range(s) if g not in is_set]
        is_rows = np.concatenate([groups[g] for g in sorted(is_set)])
        oos_rows = np.concatenate([groups[g] for g in oos_groups])

        is_ret = r[is_rows, :]
        oos_ret = r[oos_rows, :]

        is_std = is_ret.std(axis=0, ddof=1)
        is_std_safe = np.where(is_std == 0, np.nan, is_std)
        is_sharpe = is_ret.mean(axis=0) / is_std_safe
        best_idx = int(np.nanargmax(is_sharpe))

        oos_std = oos_ret.std(axis=0, ddof=1)
        oos_std_safe = np.where(oos_std == 0, np.nan, oos_std)
        oos_sharpe = oos_ret.mean(axis=0) / oos_std_safe

        order = np.argsort(oos_sharpe)
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        rank = ranks[best_idx]
        omega = rank / (n + 1)
        lam = np.log(omega / (1 - omega))
        lambdas.append(lam)

    lambdas_arr = np.array(lambdas)
    pbo_value = float(np.mean(lambdas_arr <= 0))
    return pbo_value, lambdas_arr


def _normalized_hhi(weights: np.ndarray) -> float:
    """Shared normalized-HHI kernel: ``(sum(w^2) - 1/n) / (1 - 1/n)``.

    ``NaN`` when ``n <= 2`` (the normalization is undefined at ``n == 1``
    and uninformative at ``n == 2``).
    """
    n = weights.size
    if n <= 2:
        return float("nan")
    sq = float(np.sum(weights**2))
    return (sq - 1.0 / n) / (1.0 - 1.0 / n)


def hhi_positive(returns) -> float:
    """Normalized HHI of concentration among positive returns.

    ``w_i = r_i / sum(r_i)`` over ``r_i > 0``. ``NaN`` if fewer than 3
    positive returns exist (including zero positive returns).

    Raises
    ------
    ValueError
        If ``returns`` overall is empty.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        raise ValueError("hhi_positive: returns must be non-empty.")
    pos = r[r > 0]
    if pos.size == 0:
        return float("nan")
    weights = pos / pos.sum()
    return _normalized_hhi(weights)


def hhi_negative(returns) -> float:
    """Normalized HHI of concentration among negative returns.

    ``w_i = r_i / sum(r_i)`` over ``r_i < 0`` (sign cancels: both
    numerator and denominator are negative, so weights are positive
    fractions summing to 1). ``NaN`` if fewer than 3 negative returns
    exist.

    Raises
    ------
    ValueError
        If ``returns`` overall is empty.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        raise ValueError("hhi_negative: returns must be non-empty.")
    neg = r[r < 0]
    if neg.size == 0:
        return float("nan")
    weights = neg / neg.sum()
    return _normalized_hhi(weights)


def hhi_temporal(t0, freq: str = "ME") -> float:
    """Normalized HHI of bet-count concentration across calendar periods.

    Groups ``t0`` (event-start timestamps) by ``freq`` (default monthly,
    ``"ME"`` -- pandas' non-deprecated month-end alias, valid on both the
    pandas 2.2 Docker gate and newer pandas), counts bets per period, and
    applies the same normalized-HHI kernel to the period weights. ``NaN``
    if fewer than 3 non-empty periods exist.

    Raises
    ------
    ValueError
        If ``t0`` is empty.
    """
    idx = pd.DatetimeIndex(t0)
    if idx.size == 0:
        raise ValueError("hhi_temporal: t0 must be non-empty.")
    counts = pd.Series(1, index=idx).groupby(pd.Grouper(freq=freq)).sum()
    # pd.Grouper fills gap periods (e.g. a month with no bets) as 0 -- drop
    # them before weighting. At least one period is guaranteed > 0 here
    # since idx is non-empty, so no further empty-check is reachable.
    counts = counts[counts > 0]
    weights = counts.to_numpy(dtype=float)
    weights = weights / weights.sum()
    return _normalized_hhi(weights)


def drawdown_tuw(returns: pd.Series) -> tuple:
    """Compounded drawdown and time-under-water, dense per observation.

    ``equity = (1 + returns).cumprod()``; ``dd`` = fractional decline from
    the running high-water mark at every point (``0`` at a new high);
    ``tuw`` = years elapsed since the most recent high-water mark (``0``
    exactly at a new high). Both are dense, full-length series aligned to
    ``returns.index`` -- the engine reduces them to max/percentiles.

    Parameters
    ----------
    returns:
        Per-event bet returns indexed by a ``DatetimeIndex`` (event
        ``t0``).

    Returns
    -------
    tuple[pd.Series, pd.Series]
        ``(dd, tuw)``.

    Raises
    ------
    ValueError
        If ``returns`` is empty.
    """
    r = pd.Series(returns)
    if r.size == 0:
        raise ValueError("drawdown_tuw: returns must be non-empty.")
    equity = (1 + r).cumprod()
    hwm = equity.cummax()
    dd = (hwm - equity) / hwm
    dd.name = "dd"

    is_new_hwm = equity.to_numpy() == hwm.to_numpy()
    last_hwm_t = None
    tuw_vals = []
    for t, is_hwm in zip(r.index, is_new_hwm):
        if is_hwm:
            last_hwm_t = t
            tuw_vals.append(0.0)
        else:
            years = (t - last_hwm_t).total_seconds() / _SECONDS_PER_YEAR
            tuw_vals.append(years)
    tuw = pd.Series(tuw_vals, index=r.index, name="tuw")
    return dd, tuw