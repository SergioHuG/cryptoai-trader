"""backtest.synthetic -- O-U / AR(1)-on-returns matched-moment null
(Phase 2, Task 4).

Price-free synthetic null distribution (Q7): the O-U process discretizes
to the AR(1) recursion ``r_t = c + phi * r_{t-1} + sigma * eps_t``, fit
via OLS directly against a strategy's realized per-bet return series.
``simulate`` then draws ``n_sims`` independent synthetic paths from the
fitted process via a seeded ``numpy.random.default_rng`` -- mean-init
(every path starts at the fitted stationary mean) plus a fixed
deterministic burn-in before recording -- giving the engine's synthetic
evaluation mode a null-percentile context, never a hard gate.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd

__all__ = ["OUParams", "fit_ou", "simulate"]

OUParams = namedtuple("OUParams", ["c", "phi", "sigma", "mu", "half_life"])
"""Fitted AR(1)-on-returns parameters.

Attributes:
    c:         OLS intercept.
    phi:       OLS slope (AR(1) coefficient / persistence).
    sigma:     Residual standard deviation (``ddof = n_pairs - 2``).
    mu:        Stationary mean ``c / (1 - phi)``. Only meaningful when
               ``|phi| < 1``; ``NaN`` when ``phi == 1`` exactly (the
               algebraic expression is undefined).
    half_life: Mean-reversion half-life in bet-index units,
               ``ln(2) / -ln(|phi|)``. ``0.0`` at ``phi == 0`` (instant
               reversion); ``inf`` whenever ``|phi| >= 1`` (non-stationary
               -- no mean reversion to speak of).
"""


def fit_ou(returns) -> OUParams:
    """Fit an AR(1)-on-returns O-U discretization via OLS.

    ``r_t`` regressed on ``r_{t-1}`` across the full series (``n - 1``
    pairs from ``n`` observations). Does not raise on a non-stationary
    fit (``|phi| >= 1``) -- that is a valid, if unusual, empirical
    outcome; it is reflected in ``half_life == inf`` rather than a raise.
    :func:`simulate` is what refuses to consume a non-stationary fit.

    Raises
    ------
    ValueError
        If ``returns`` has fewer than 4 observations (too few pairs to
        leave any residual degrees of freedom), or if ``r_{t-1}`` has
        zero variance (a constant lagged series -- AR(1) slope is
        undefined).
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 4:
        raise ValueError(
            f"fit_ou: need at least 4 observations to fit AR(1) with "
            f"residual degrees of freedom, got n={n!r}."
        )
    y = r[1:]
    x = r[:-1]
    n_pairs = y.size

    x_mean, y_mean = x.mean(), y.mean()
    var_x = np.sum((x - x_mean) ** 2)
    if var_x == 0:
        raise ValueError(
            "fit_ou: r_{t-1} has zero variance (a constant lagged series) "
            "-- the AR(1) slope is undefined."
        )
    cov_xy = np.sum((x - x_mean) * (y - y_mean))
    phi_hat = float(cov_xy / var_x)
    c_hat = float(y_mean - phi_hat * x_mean)

    residuals = y - (c_hat + phi_hat * x)
    dof = n_pairs - 2
    # dof >= 1 is guaranteed by the n >= 4 precondition above
    # (n_pairs = n - 1 >= 3, dof = n_pairs - 2 >= 1) -- no further guard
    # is reachable here.
    sigma_hat = float(np.sqrt(np.sum(residuals**2) / dof))

    mu = float("nan") if phi_hat == 1.0 else c_hat / (1 - phi_hat)

    if abs(phi_hat) >= 1:
        half_life = float("inf")
    elif phi_hat == 0:
        half_life = 0.0
    else:
        half_life = float(np.log(2) / (-np.log(abs(phi_hat))))

    return OUParams(c=c_hat, phi=phi_hat, sigma=sigma_hat, mu=mu, half_life=half_life)


def simulate(
    params: OUParams, n_obs: int, n_sims: int, seed: int, burn_in: int
) -> pd.DataFrame:
    """Draw ``n_sims`` synthetic AR(1)-on-returns paths from a fitted O-U.

    Every path is mean-initialized at ``params.mu``, advanced
    ``burn_in + n_obs - 1`` steps via
    ``r_t = c + phi * r_{t-1} + sigma * eps_t`` with
    ``eps_t ~ N(0, 1)`` drawn from ``numpy.random.default_rng(seed)``,
    then the first ``burn_in`` steps are discarded. Fully deterministic
    given ``(params, n_obs, n_sims, seed, burn_in)`` -- the same seed
    reproduces byte-identical output; a different seed does not.

    Parameters
    ----------
    params:
        A fitted, stationary (``|phi| < 1``) :class:`OUParams`.
    n_obs:
        Recorded path length (>= 1).
    n_sims:
        Number of independent simulated paths (>= 1).
    seed:
        RNG seed (>= 0).
    burn_in:
        Deterministic burn-in length in steps (>= 0), discarded before
        recording.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_obs, n_sims)``; integer index ``0..n_obs-1``; columns
        ``sim_id`` ``0..n_sims-1``.

    Raises
    ------
    ValueError
        If ``params`` is not stationary (``|phi| >= 1`` or non-finite
        ``mu``), or if ``n_obs < 1``, ``n_sims < 1``, ``burn_in < 0``.
    """
    if abs(params.phi) >= 1 or not np.isfinite(params.mu):
        raise ValueError(
            "simulate: params must represent a stationary fit "
            f"(|phi| < 1 and finite mu), got phi={params.phi!r}, "
            f"mu={params.mu!r}."
        )
    if n_obs < 1:
        raise ValueError(f"simulate: n_obs must be >= 1, got {n_obs!r}.")
    if n_sims < 1:
        raise ValueError(f"simulate: n_sims must be >= 1, got {n_sims!r}.")
    if burn_in < 0:
        raise ValueError(f"simulate: burn_in must be >= 0, got {burn_in!r}.")

    rng = np.random.default_rng(seed)
    total_len = burn_in + n_obs
    eps = rng.standard_normal(size=(n_sims, total_len - 1))

    paths = np.empty((n_sims, total_len))
    paths[:, 0] = params.mu
    for t in range(1, total_len):
        paths[:, t] = params.c + params.phi * paths[:, t - 1] + params.sigma * eps[:, t - 1]

    recorded = paths[:, burn_in:]
    return pd.DataFrame(recorded.T, columns=range(n_sims))
