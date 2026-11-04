"""Portfolio strategy engine: cleaned inputs + walk-forward backtest.

The canonical implementation of the machinery the daily briefs run
(their ``maxsharpe-alloc.py`` pipeline), built for arbitrary
universes of ETFs / stocks:

- **Covariance cleaning** — the sample covariance is one choice, not
  the default truth. Cleaners: BBP (the feasible rotationally
  invariant estimator of Bun–Bouchaud–Potters, via the hybrid
  Stieltjes transform — the briefs' cleaner), Marčenko–Pastur
  eigenvalue clipping (Laloux et al.), Ledoit–Wolf linear shrinkage
  to a scaled identity, and Ledoit–Péché nonlinear shrinkage —
  the same RIE shrinkage function as BBP but with the Stieltjes
  transform estimated from the *covariance* spectrum itself, so
  it works in covariance space where the MP law does not apply.
- **Mean cleaning** — sample means are the noisiest input in
  Markowitz (Merton 1980). Cleaners: James–Stein shrinkage toward
  the grand mean (the briefs' cleaner), Jorion's Bayes–Stein, and
  the grand mean itself (full shrinkage).
- **Strategies** — long-only, fully invested (``w >= 0``,
  ``sum w = 1``), solved from *cleaned* inputs: 1/N, minimum
  variance, maximum diversification, NCO (nested clustered
  optimisation), HRP, risk parity (ERC), minimum CVaR (95%),
  maximum Sharpe (tangency), and the ensemble Average of the
  selected rules. Two rules read returns alone, with no cleaned
  inputs: the RS(n, w) relative-strength rules and SPMO, the
  S&P 500 Momentum Index rule (see :func:`spmo_weights`).
  A strategy may also be named as a *combination*
  (:data:`COMBO_STRATEGY_NAMES`) that carries its own cleaning
  pair — the briefs' allocation-table convention — in which case
  the caller's pair is only the fallback for unqualified names.
- **Backtest** — walk-forward with an expanding estimation window,
  monthly-style rebalancing every ``hold`` days, weights estimated
  on log returns up to each rebalance close only (no lookahead),
  weights drifting with simple returns between rebalances, Sharpe
  on excess returns over a contemporaneous risk-free series, and
  optional one-way transaction costs.

Everything is estimated on daily **log** returns; means are handled
in excess of the risk-free rate. No display, fetching-session, or
application coupling: callers pass price frames (see
:mod:`toolbox.DataZoo.market.yahoo_prices` for the Yahoo fetcher).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform

TRADING_DAYS = 252

COV_METHODS = ("sample", "bbp", "clip", "ledoit_wolf", "nlshrink")
COV_LABELS = {
    "sample": "Sample (raw)",
    "bbp": "BBP — RIE cleaned",
    "clip": "MP clipping",
    "ledoit_wolf": "Ledoit–Wolf",
    "nlshrink": "Nonlinear shrinkage (covariance)",
}
MEAN_METHODS = ("sample", "james_stein", "bayes_stein", "grand_mean")
MEAN_LABELS = {
    "sample": "Sample",
    "james_stein": "James–Stein",
    "bayes_stein": "Bayes–Stein (Jorion)",
    "grand_mean": "Grand mean",
}
STRATEGY_NAMES = (
    "1/N",
    "Min variance",
    "Max diversification",
    "NCO",
    "HRP",
    "Risk parity",
    "Min CVaR",
    "Max Sharpe",
    "SPMO",
    "Average",
)

# Combination strategies (user direction 2026-10-31): a strategy may
# carry its own cleaning pair in its name, exactly as the briefs'
# allocation tables name their rows — "Max Sharpe (BBP × BS)" is the
# tangency portfolio on the BBP covariance with Bayes–Stein means.
# The pair in the name overrides the caller's (cov_method,
# mean_method); an unqualified name — and HRP / Risk parity, which
# are named bare — falls back to the caller's pair for the missing
# component.
_COV_SHORT = {"sample": "raw", "bbp": "BBP", "ledoit_wolf": "LW"}
_MEAN_SHORT = {"sample": "sample", "james_stein": "JS",
               "bayes_stein": "BS"}
_COV_FROM_SHORT = {v: k for k, v in _COV_SHORT.items()}
_MEAN_FROM_SHORT = {v: k for k, v in _MEAN_SHORT.items()}
# Ledoit–Wolf joined the grid 2026-10-31 at the user's direction
# ("Add lw — this will increase strategy list"): 17 → 23 names.
_GRID_COVS = ("sample", "bbp", "ledoit_wolf")
_GRID_MEANS = ("sample", "james_stein", "bayes_stein")
_COV_ONLY_RULES = ("Min variance", "Max diversification", "NCO")

# Relative-strength rules (user direction 2026-11-02):
# "RS(n, w)" holds the n tickers with the highest trailing
# w-day return at equal weight 1/n, re-chosen at every
# rebalance — no covariance or mean inputs. The pair rides
# in the name like the cleaning pairs do; any well-formed
# RS(n, w) resolves, and the grid offers this curated set —
# winner-take-all at a month, a half-year and a year, then
# widening baskets as the window lengthens.
RS_PAIRS = ((1, 21), (1, 126), (1, 252), (2, 63), (3, 126), (5, 252))
RS_STRATEGY_NAMES = tuple(f"RS({n}, {w})" for n, w in RS_PAIRS)

# Estimation windows as strategy variants (user direction
# 2026-11-02): there is no global estimation-window parameter —
# each window creates a new strategy, exactly as RS(n, w)
# carries its lookback in its name. The suffix is " @ 252" for
# a rolling window of that many trading days, " @ All" for the
# expanding window. Every window-using combination base appears
# at each of the four windows; 1/N never uses a window (its
# weights cannot depend on one) and the RS rules carry their
# own lookback, so both stay unsuffixed.
STRATEGY_WINDOWS = (126, 252, 504, "All")


def parse_window(name: str) -> tuple[str, int | None]:
    """``(clean_name, window)`` for a strategy name: the window
    is an int for an ``" @ <days>"`` suffix, None for
    ``" @ All"`` and for a name with no suffix (both are the
    expanding window). Raises ValueError on a malformed
    suffix."""
    if " @ " not in name:
        return name, None
    clean, _, tail = name.rpartition(" @ ")
    if tail == "All":
        return clean, None
    try:
        w = int(tail)
    except ValueError:
        raise ValueError(f"Unknown strategy: {name}") from None
    if w < 1:
        raise ValueError(f"Unknown strategy: {name}")
    return clean, w


_WINDOWED_BASES = (
    tuple(f"Max Sharpe ({_COV_SHORT[c]} × {_MEAN_SHORT[m]})"
          for c in _GRID_COVS for m in _GRID_MEANS)
    + tuple(f"{rule} ({_COV_SHORT[c]})"
            for rule in _COV_ONLY_RULES for c in _GRID_COVS)
    + ("HRP", "Risk parity", "Min CVaR")
)

COMBO_STRATEGY_NAMES = (
    tuple(f"{base} @ {w}"
          for base in _WINDOWED_BASES for w in STRATEGY_WINDOWS)
    + ("1/N",)
    + RS_STRATEGY_NAMES
    + ("SPMO",)
    + ("Average",)
)


def _parse_rs(name: str) -> tuple[int, int] | None:
    """``(n, w)`` for a well-formed ``RS(n, w)`` name with
    n, w >= 1, else None."""
    if name.startswith("RS(") and name.endswith(")"):
        parts = name[3:-1].split(", ")
        if len(parts) == 2:
            try:
                n, w = int(parts[0]), int(parts[1])
            except ValueError:
                return None
            if n >= 1 and w >= 1:
                return n, w
    return None


def resolve_strategy(name: str) -> tuple[str, str | None, str | None]:
    """``(base_rule, cov_method, mean_method)`` for a strategy name.

    A ``None`` component falls back to the caller's pair. Bare base
    names resolve to ``(name, None, None)`` — ``"Average"`` likewise
    (the ensemble layer handles it). An estimation-window suffix
    (see :func:`parse_window`) is stripped first. Raises ValueError
    on an unknown name or qualifier.
    """
    name, _win = parse_window(name)
    if name in STRATEGY_NAMES:
        return name, None, None
    if _parse_rs(name) is not None:
        return name, None, None
    if " (" in name and name.endswith(")"):
        base, qual = name[:-1].split(" (", 1)
        if base == "Max Sharpe" and " × " in qual:
            cov_s, mean_s = qual.split(" × ")
            if cov_s in _COV_FROM_SHORT and mean_s in _MEAN_FROM_SHORT:
                return (base, _COV_FROM_SHORT[cov_s],
                        _MEAN_FROM_SHORT[mean_s])
        if base in _COV_ONLY_RULES and qual in _COV_FROM_SHORT:
            return base, _COV_FROM_SHORT[qual], None
    raise ValueError(f"Unknown strategy: {name}")


_COMBO_RANK = {s: i for i, s in enumerate(COMBO_STRATEGY_NAMES)}
_WINDOWED_BASE_SET = frozenset(_WINDOWED_BASES)


def _combo_rank(name: str) -> float | None:
    """A name's rank in the combination grid, or None when the
    name is not a combination at all. An unsuffixed windowed
    base ranks just ahead of its own windowed variants."""
    if name in _COMBO_RANK:
        return float(_COMBO_RANK[name])
    clean, _w = parse_window(name)
    if clean in _WINDOWED_BASE_SET:
        return _COMBO_RANK[f"{clean} @ {STRATEGY_WINDOWS[0]}"] - 0.5
    return None


def _strategy_order(names: list[str]) -> list[str]:
    """Display order for a set of strategy names.

    A pure combination set follows :data:`COMBO_STRATEGY_NAMES`.
    Otherwise bare-name order rules, with each combination placed
    at its base rule's position (Average, a bare name, stays last).
    """
    ranks = [_combo_rank(s) for s in names]
    if all(r is not None for r in ranks):
        return [s for _r, s in sorted(zip(ranks, names),
                                      key=lambda t: t[0])]
    base_rank = {s: i for i, s in enumerate(STRATEGY_NAMES)}

    def key(s: str) -> tuple[int, float]:
        base = s if s in STRATEGY_NAMES else resolve_strategy(s)[0]
        r = _combo_rank(s)
        return (base_rank.get(base, len(base_rank)),
                r if r is not None else -1.0)

    return sorted(names, key=key)

# ---------------------------------------------------------------------------
# Returns and risk-free
# ---------------------------------------------------------------------------


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns of a price frame (rows with any NaN dropped)."""
    px = prices.dropna().sort_index()
    return np.log(px / px.shift(1)).dropna()


def rf_daily_series(index: pd.Index, annual_yields_pct: pd.Series | None,
                    fallback_daily: float = 0.0) -> np.ndarray:
    """Daily risk-free rates aligned to ``index`` from annual yields (%).

    ``annual_yields_pct`` is a dated series of annualised yields in
    percent (e.g. ^IRX); it is forward-filled onto ``index``. Falls
    back to a constant daily rate when no history is available.
    """
    if annual_yields_pct is not None and len(annual_yields_pct):
        y = (annual_yields_pct.reindex(index, method="ffill")
             .ffill().bfill().values / 100.0)
        return (1.0 + y) ** (1.0 / TRADING_DAYS) - 1.0
    return np.full(len(index), float(fallback_daily))


# ---------------------------------------------------------------------------
# Covariance cleaning
# ---------------------------------------------------------------------------


def _stieltjes_hybrid(lam: float, q: float) -> complex:
    """Stieltjes transform of the Marčenko–Pastur law at ``lam``.

    Closed form outside the bulk; inside the bulk, the defining
    integral against the MP density, trapezoided on a fine grid
    (the briefs' hybrid — the closed form's branch is numerically
    unreliable just inside the edges).
    """
    lm = (1 - np.sqrt(q)) ** 2
    lp = (1 + np.sqrt(q)) ** 2
    if lam >= lm:
        z = lam - 1j * 1e-9
        disc = (z + q - 1) ** 2 - 4 * q * z
        return ((z + q - 1) - np.sqrt(disc)) / (2 * q * z)
    xs = np.linspace(lm, lp, 2001)
    rho = np.sqrt(np.maximum(0, (lp - xs) * (xs - lm))) / (2 * np.pi * q * xs)
    f = rho / (lam - xs)
    dx = xs[1] - xs[0]
    return (0.5 * f[0] + f[1:-1].sum() + 0.5 * f[-1]) * dx


def bbp_clean_corr(C: np.ndarray, q: float) -> np.ndarray:
    """BBP/RIE-cleaned correlation matrix.

    Each sample eigenvalue ``λ`` is replaced by the oracle shrinkage
    ``ξ = λ / |1 − q + qλ·m(λ)|²`` with ``m`` the MP Stieltjes
    transform — the feasible rotationally invariant estimator of
    Bun–Bouchaud–Potters. Requires ``q = p/n < 1``.
    """
    vals, vecs = np.linalg.eigh(np.asarray(C, dtype=float))
    o = np.argsort(vals)[::-1]
    vals, vecs = vals[o], vecs[:, o]
    xi = np.array([l / abs(1 - q + q * l * _stieltjes_hybrid(l, q)) ** 2
                   for l in vals])
    out = (vecs * xi) @ vecs.T
    return 0.5 * (out + out.T)


def clip_clean_corr(C: np.ndarray, q: float) -> np.ndarray:
    """Marčenko–Pastur eigenvalue clipping (Laloux et al. 1999).

    Eigenvalues at or below the MP upper edge ``(1+√q)²`` are
    replaced by their own mean (trace preserved); the result is
    renormalised to unit diagonal, as a correlation must be.
    """
    C = np.asarray(C, dtype=float)
    vals, vecs = np.linalg.eigh(C)
    lam_plus = (1 + np.sqrt(q)) ** 2
    noise = vals <= lam_plus
    if noise.any() and not noise.all():
        vals = vals.copy()
        vals[noise] = vals[noise].mean()
    out = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(out))
    out = out / np.outer(d, d)
    return 0.5 * (out + out.T)


_SQRT5 = np.sqrt(5.0)


def _epanechnikov(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel ``(3/(4√5))(1 − u²/5)`` on ``|u| ≤ √5``."""
    k = (3.0 / (4.0 * _SQRT5)) * (1.0 - u ** 2 / 5.0)
    return np.where(np.abs(u) <= _SQRT5, k, 0.0)


def nlshrink_eigs(lam: np.ndarray, n: int) -> np.ndarray:
    """Ledoit–Péché nonlinear shrinkage of covariance eigenvalues.

    Each sample eigenvalue ``λ`` of the covariance matrix is replaced
    by the rotationally invariant oracle shrinkage
    ``ξ = λ / |1 − q + qλ·g(λ)|²``, ``q = p/n``, with ``g`` the
    Stieltjes (resolvent) transform of the *limiting spectral
    distribution of the covariance eigenvalues themselves* — the
    same shrinkage function as :func:`bbp_clean_corr`, differing
    only in the plug-in for ``g``: BBP uses the analytic Marčenko–
    Pastur transform, valid only for a correlation spectrum (unit
    population variances), while here ``g`` is estimated from the
    sample spectrum by Epanechnikov kernel smoothing (bandwidth
    ``h = n^(−1/3)`` in units of the mean eigenvalue, so the map is
    exactly scale-equivariant), with the principal-value part by
    trapezoid integration of the smoothed density against the
    singularity-subtracted kernel — the numerical scheme of
    :func:`_stieltjes_hybrid`. This is the Ledoit–Wolf (2017, 2020)
    nonlinear shrinkage estimator; it shrinks each eigenvalue by its
    own amount, unlike the single-intensity linear shrinkage of
    :func:`ledoit_wolf_cov`. Requires ``q < 1``.
    """
    lam = np.sort(np.maximum(np.asarray(lam, dtype=float), 0.0))
    p = len(lam)
    q = p / n
    scale = lam.mean()
    if scale <= 0:
        return lam
    x = lam / scale
    h = n ** (-1.0 / 3.0)
    # Smoothed spectral density at the sample eigenvalues.
    f = _epanechnikov((x[:, None] - x[None, :]) / h).mean(axis=1) / h
    # Principal value of g on a fine grid over the kernel support.
    a = x.min() - _SQRT5 * h - 1e-9
    b = x.max() + _SQRT5 * h + 1e-9
    grid = np.linspace(a, b, 4097)
    dx = grid[1] - grid[0]
    fg = _epanechnikov((grid[:, None] - x[None, :]) / h).mean(axis=1) / h
    g = np.empty(p)
    for i in range(p):
        denom = x[i] - grid
        near = np.abs(denom) < 1e-12
        integrand = np.where(
            near, 0.0, (fg - f[i]) / np.where(near, 1.0, denom))
        g[i] = ((0.5 * integrand[0] + integrand[1:-1].sum()
                 + 0.5 * integrand[-1]) * dx
                + f[i] * np.log((x[i] - a) / (b - x[i])))
    xi = x / ((1.0 - q + q * x * g) ** 2 + (np.pi * q * x * f) ** 2)
    return xi * scale


def nlshrink_cov(R: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Nonlinear-shrinkage covariance of log returns ``R``.

    Returns ``(covariance, bandwidth)`` with the bandwidth in units
    of the mean sample eigenvalue. Uses the population (1/n) sample
    covariance, as in the original estimator.
    """
    S0 = sample_cov(R)
    vals, vecs = np.linalg.eigh(S0)
    d = nlshrink_eigs(vals, len(R))
    S = (vecs * d) @ vecs.T
    return 0.5 * (S + S.T), float(len(R) ** (-1.0 / 3.0))


def ledoit_wolf_cov(R: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Ledoit–Wolf (2004) covariance: sample shrunk to a scaled identity.

    Returns ``(covariance, shrinkage_intensity)`` with the intensity
    in ``[0, 1]``. Uses the population (1/n) sample covariance, as in
    the original estimator.
    """
    X = R.values - R.values.mean(axis=0)
    n, p = X.shape
    S = (X.T @ X) / n
    m = np.trace(S) / p
    target = m * np.eye(p)
    d2 = float(np.sum((S - target) ** 2)) / p
    if d2 <= 0:
        return S, 0.0
    # b̄² = (1/(n²p)) Σ_t ‖x_t x_t' − S‖²_F, expanded so no n×p×p
    # tensor is ever formed: ‖xx'−S‖² = (x'x)² − 2x'Sx + ‖S‖²_F.
    xx = np.sum(X ** 2, axis=1)               # x_t' x_t
    xSx = np.einsum("ij,jk,ik->i", X, S, X)   # x_t' S x_t
    s_fro = float(np.sum(S ** 2))
    bbar2 = float(np.sum(xx ** 2 - 2.0 * xSx + s_fro)) / (n ** 2 * p)
    b2 = min(bbar2, d2)
    shrink = b2 / d2
    return (1.0 - shrink) * S + shrink * target, float(shrink)


def sample_cov(R: pd.DataFrame) -> np.ndarray:
    """Population sample covariance of log returns (the briefs' S_raw)."""
    return np.cov(R.values.T) * (len(R) - 1) / len(R)


def clean_covariance(R: pd.DataFrame, method: str) -> tuple[np.ndarray, dict]:
    """Covariance of log returns ``R`` under ``method`` + diagnostics.

    Returns ``(S, info)`` where ``info`` carries ``q``, the MP edges,
    the correlation eigenvalue spectrum, the signal count (eigenvalues
    above the MP upper edge), both condition numbers, and the
    method's own parameter (LW shrinkage) when it has one.
    """
    if method not in COV_METHODS:
        raise ValueError(f"Unknown covariance method: {method}")
    n, p = len(R), R.shape[1]
    q = p / n
    vols = R.std().values * np.sqrt((n - 1) / n)  # population vols, match S_raw
    C = np.corrcoef(R.values.T)
    S_raw = np.outer(vols, vols) * C
    eig = np.linalg.eigvalsh(C)[::-1]
    info = {
        "n": n, "p": p, "q": float(q),
        "mp_lo": float((1 - np.sqrt(q)) ** 2),
        "mp_hi": float((1 + np.sqrt(q)) ** 2),
        "eigenvalues": eig,
        "n_signal": int((eig > (1 + np.sqrt(q)) ** 2).sum()),
        "cond_raw": float(np.linalg.cond(S_raw)),
        "method": method,
        "shrinkage": None,
        "bandwidth": None,
    }
    if method == "sample" or q >= 1.0:
        # RIE/clipping/NLS need q < 1; with q >= 1 the sample matrix
        # is singular and the honest fallbacks are shrinkage or
        # sample.
        S = S_raw if method == "sample" else ledoit_wolf_cov(R)[0]
        if method == "ledoit_wolf":
            S, info["shrinkage"] = ledoit_wolf_cov(R)
    elif method == "bbp":
        S = np.outer(vols, vols) * bbp_clean_corr(C, q)
    elif method == "clip":
        S = np.outer(vols, vols) * clip_clean_corr(C, q)
    elif method == "nlshrink":
        S, info["bandwidth"] = nlshrink_cov(R)
    else:  # ledoit_wolf
        S, info["shrinkage"] = ledoit_wolf_cov(R)
    S = 0.5 * (S + S.T)
    info["cond_clean"] = float(np.linalg.cond(S))
    info["corr_clean"] = S / np.outer(np.sqrt(np.diag(S)), np.sqrt(np.diag(S)))
    info["corr_raw"] = C
    return S, info


# ---------------------------------------------------------------------------
# Mean cleaning
# ---------------------------------------------------------------------------


def james_stein(mu_hat: np.ndarray, samp_var: float) -> tuple[np.ndarray, float]:
    """James–Stein shrinkage of ``mu_hat`` toward its grand mean.

    ``samp_var`` is the mean sampling variance of the components
    (the briefs use ``mean(σ²/n)``). Returns ``(mu_js, c)`` where
    ``c`` is the weight kept on the sample deviations.
    """
    p = len(mu_hat)
    m = float(mu_hat.mean())
    dev = mu_hat - m
    denom = max(float((dev ** 2).sum()), 1e-18)
    c = max(0.0, 1.0 - (p - 3) * samp_var / denom) if p > 3 else 1.0
    return m + c * dev, float(c)


def bayes_stein(mu_hat: np.ndarray, S: np.ndarray,
                n: int) -> tuple[np.ndarray, float]:
    """Jorion (1986) Bayes–Stein shrinkage toward the grand mean.

    Weight on the grand mean: ``w = (p+2) / ((p+2) + n·D)`` with
    ``D = (μ̂−μ̂₀)' Σ⁻¹ (μ̂−μ̂₀)`` the Mahalanobis distance of the
    sample means from their common value.
    """
    p = len(mu_hat)
    m = float(mu_hat.mean())
    dev = mu_hat - m
    try:
        D = float(dev @ np.linalg.pinv(S) @ dev)
    except np.linalg.LinAlgError:
        return mu_hat.copy(), 0.0
    w = (p + 2.0) / ((p + 2.0) + n * max(D, 0.0))
    w = float(min(1.0, max(0.0, w)))
    return (1.0 - w) * mu_hat + w * m, w


def clean_mean(R: pd.DataFrame, S: np.ndarray, method: str,
               rf_d: float) -> tuple[np.ndarray, dict]:
    """Excess mean of log returns ``R`` under ``method`` + diagnostics.

    All variants start from the sample mean in excess of the daily
    risk-free rate ``rf_d`` (the briefs' convention).
    """
    if method not in MEAN_METHODS:
        raise ValueError(f"Unknown mean method: {method}")
    n = len(R)
    mu_hat = R.mean().values - rf_d
    vols = R.std().values
    info = {"method": method, "js_c": None, "bs_w": None,
            "mu_sample": mu_hat.copy()}
    if method == "sample":
        return mu_hat, info
    if method == "grand_mean":
        return np.full_like(mu_hat, mu_hat.mean()), info
    if method == "james_stein":
        mu, c = james_stein(mu_hat, float((vols ** 2 / n).mean()))
        info["js_c"] = c
        return mu, info
    mu, w = bayes_stein(mu_hat, S, n)
    info["bs_w"] = w
    return mu, info


def estimate_inputs(R: pd.DataFrame, cov_method: str, mean_method: str,
                    rf_d: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(mu_excess, S, info)`` for one estimation window of log returns."""
    S, cov_info = clean_covariance(R, cov_method)
    mu, mean_info = clean_mean(R, S, mean_method, rf_d)
    info = {**cov_info, **mean_info}
    return mu, S, info


# ---------------------------------------------------------------------------
# Long-only solvers (all: w >= 0, sum w = 1)
# ---------------------------------------------------------------------------


def _clean_w(w: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=float), 0, None)
    total = w.sum()
    return w / total if total > 0 else np.full(len(w), 1.0 / len(w))


def minvar_weights(S: np.ndarray) -> np.ndarray:
    """Long-only global minimum-variance weights.

    The problem is convex (PSD quadratic, simplex constraints), so
    one SLSQP run from the equal-weight start finds the global
    optimum; the vertex starts remain as a fallback for solver
    failure only. (The briefs' batch code multi-starts every run —
    insurance an interactive engine does not need to buy 54 times
    per backtest.)
    """
    p = S.shape[0]

    def solve(s):
        r = minimize(lambda w: w @ S @ w, s, method="SLSQP",
                     bounds=[(0, None)] * p,
                     constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                     options={"maxiter": 1000, "ftol": 1e-14})
        return r if r.success else None

    best = solve(np.ones(p) / p)
    if best is None:
        for i in range(p):
            r = solve(np.eye(p)[i])
            if r is not None and (best is None or r.fun < best.fun):
                best = r
    return _clean_w(best.x) if best is not None else np.ones(p) / p


def max_sharpe_weights(S: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Long-only tangency weights via the convex reformulation.

    Minimise ``w'Σw`` subject to ``w'μ = 1, w ≥ 0`` (then normalise);
    falls back to direct Sharpe maximisation when the QP is
    infeasible (e.g. all excess means ≤ 0).
    """
    p = len(mu)
    # Rescale so the QP constraint w'μ = 1 is well conditioned. When
    # every excess mean is ≤ 0 the QP is infeasible by construction
    # and control passes to the direct Sharpe maximisation below,
    # which prices the least-bad tangency correctly.
    scale = float(mu.mean()) if mu.mean() > 1e-12 else float(np.abs(mu).max() or 1.0)
    mu_s = mu / scale
    # Convex QP: one run from equal weight, vertex starts only as a
    # failure fallback (see minvar_weights).
    starts = [np.ones(p) / p] + [np.eye(p)[i] for i in range(p)]
    best = None
    for s in starts:
        r = minimize(lambda w: w @ S @ w, s, method="SLSQP",
                     bounds=[(0, None)] * p,
                     constraints={"type": "eq", "fun": lambda w: w @ mu_s - 1},
                     options={"maxiter": 5000, "ftol": 1e-14})
        if r.success and (best is None or r.fun < best.fun):
            best = r
        if best is not None:
            break
    if best is not None and np.clip(best.x, 0, None).sum() > 0:
        return _clean_w(best.x)

    def neg_sharpe(w):
        return -(w @ mu) / np.sqrt(w @ S @ w + 1e-18)

    best = None
    for s in starts:
        r = minimize(neg_sharpe, s, method="SLSQP", bounds=[(0, None)] * p,
                     constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                     options={"maxiter": 5000, "ftol": 1e-14})
        if r.success and (best is None or r.fun < best.fun):
            best = r
    return _clean_w(best.x) if best is not None else np.ones(p) / p


def maxdiv_weights(S: np.ndarray) -> np.ndarray:
    """Maximum diversification (Choueifaty): tangency with σ as the mean."""
    return max_sharpe_weights(S, np.sqrt(np.diag(S)))


def _cluster_var(S: np.ndarray, idx: list[int]) -> float:
    sub = S[np.ix_(idx, idx)]
    iv = 1.0 / np.diag(sub)
    w = iv / iv.sum()
    return float(w @ sub @ w)


def hrp_weights(S: np.ndarray) -> np.ndarray:
    """Hierarchical risk parity (López de Prado 2016) on covariance ``S``.

    Single-linkage clustering on the correlation distance
    ``√((1−ρ)/2)``, quasi-diagonal ordering, recursive bisection with
    inverse-variance cluster allocation. Uses the covariance actually
    passed in — cleaned in, cleaned out.
    """
    S = np.asarray(S, dtype=float)
    p = S.shape[0]
    if p == 1:
        return np.ones(1)
    sd = np.sqrt(np.diag(S))
    C = S / np.outer(sd, sd)
    dist = np.sqrt(np.clip(0.5 * (1.0 - C), 0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")

    def leaves(node):
        if node < p:
            return [node]
        return leaves(int(link[node - p, 0])) + leaves(int(link[node - p, 1]))

    order = leaves(2 * p - 2)

    def bisection(items):
        if len(items) <= 1:
            return {items[0]: 1.0} if items else {}
        mid = len(items) // 2
        left, right = items[:mid], items[mid:]
        v_l, v_r = _cluster_var(S, left), _cluster_var(S, right)
        alpha = 1.0 - v_l / (v_l + v_r) if (v_l + v_r) > 0 else 0.5
        out = {}
        for k, v in bisection(left).items():
            out[k] = out.get(k, 0.0) + v * alpha
        for k, v in bisection(right).items():
            out[k] = out.get(k, 0.0) + v * (1.0 - alpha)
        return out

    w = np.zeros(p)
    for k, v in bisection(order).items():
        w[k] = v
    return _clean_w(w)


def nco_weights(R: pd.DataFrame, S: np.ndarray, k: int | None = None) -> np.ndarray:
    """Nested clustered optimisation (López de Prado).

    Single-linkage clusters on the correlation distance into
    ``k = round(√p)`` clusters (≥ 2); long-only min-variance on ``S``
    within each cluster, then min-variance across cluster portfolios.
    """
    p = R.shape[1]
    if p == 1:
        return np.ones(1)
    C = np.corrcoef(R.values.T)
    dist = np.sqrt(np.clip(0.5 * (1.0 - C), 0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    if k is None:
        k = max(2, int(round(np.sqrt(p))))
    k = min(k, p)
    cl = fcluster(link, t=k, criterion="maxclust")
    clusters = [np.where(cl == j)[0] for j in range(1, k + 1)]
    clusters = [c for c in clusters if len(c)]
    Wmat = np.zeros((p, len(clusters)))
    for j, idx in enumerate(clusters):
        Wmat[idx, j] = minvar_weights(S[np.ix_(idx, idx)])
    S_red = Wmat.T @ S @ Wmat
    w = Wmat @ minvar_weights(S_red)
    return _clean_w(w)


def erc_weights(S: np.ndarray) -> np.ndarray:
    """Equal risk contribution (risk parity) weights.

    Solved in the Spinu / Griveau-Billion convex form: minimise
    ``½ y'Σy − Σᵢ log yᵢ`` over ``y > 0`` (equal budgets), then
    normalise ``w = y / Σy``. The direct least-squares form in ``w``
    is badly scaled at daily-return magnitudes (objectives ~1e-11)
    and stalls SLSQP at its start point; the log-barrier form does
    not. ERC weights are invariant to a common rescaling of ``Σ``.
    """
    p = S.shape[0]
    if p == 1:
        return np.ones(1)
    Ss = S / float(np.mean(np.diag(S)))

    def obj(y):
        return 0.5 * float(y @ Ss @ y) - float(np.sum(np.log(y)))

    def grad(y):
        return Ss @ y - 1.0 / y

    y0 = 1.0 / np.sqrt(np.diag(Ss))
    y0 = y0 / np.exp(np.mean(np.log(y0)))
    best = None
    for s in (y0, np.ones(p)):
        r = minimize(obj, s, jac=grad, method="SLSQP",
                     bounds=[(1e-10, None)] * p,
                     options={"maxiter": 2000, "ftol": 1e-16})
        if r.success and (best is None or r.fun < best.fun):
            best = r
    if best is None:
        return np.ones(p) / p
    return _clean_w(best.x)


def cvar_weights(R_exc: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    """Minimum CVaR_α of scenario excess returns (Rockafellar–Uryasev LP).

    Variables (w, a, u): minimise ``a + (1/(n(1−α))) Σuₜ`` subject to
    ``uₜ ≥ −rₜ'w − a``, ``uₜ ≥ 0``, ``Σw = 1``, ``w ≥ 0`` (HiGHS).
    """
    n, p = R_exc.shape
    c = np.zeros(p + 1 + n)
    c[p] = 1.0
    c[p + 1:] = 1.0 / (n * (1.0 - alpha))
    A = np.zeros((n, p + 1 + n))
    A[:, :p] = -R_exc
    A[:, p] = -1.0
    A[np.arange(n), p + 1 + np.arange(n)] = -1.0
    A_eq = np.zeros((1, p + 1 + n))
    A_eq[0, :p] = 1.0
    bounds = [(0, None)] * p + [(None, None)] + [(0, None)] * n
    res = linprog(c, A_ub=A, b_ub=np.zeros(n), A_eq=A_eq, b_eq=[1.0],
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("CVaR LP failed: " + str(res.message))
    return _clean_w(res.x[:p])


def rs_weights(R: pd.DataFrame, n: int, w: int) -> np.ndarray:
    """Equal weights on the n strongest tickers over the last
    w rows of ``R`` (log returns): strength is the trailing
    sum of log returns — the log price ratio over the window.
    A window longer than the history uses all of it. Ties
    keep universe order (the sort is stable); n at or above
    the universe size is 1/N."""
    p = R.shape[1]
    k = min(n, p)
    strength = R.tail(w).sum(axis=0).to_numpy()
    top = np.argsort(-strength, kind="stable")[:k]
    out = np.zeros(p)
    out[top] = 1.0 / k
    return out


# SPMO (user direction 2026-11-04): the S&P 500 Momentum Index
# rule (the benchmark of Invesco's SPMO), from the S&P DJI
# momentum appendix. On each rebalance, a ticker's momentum
# value is its price change over the 12 months ending one
# month before the rebalance — the most recent month is
# skipped against short-term reversal — divided by the
# standard deviation of its daily price returns over the same
# window. The risk-adjusted values are z-scored across the
# universe, the z-scores capped at +/-3, and mapped to a
# strictly positive score: 1 + z above zero, 1 / (1 - z)
# below it, so scores live in [0.25, 4] and negative momentum
# down-weights rather than excludes. The index then holds
# the top 20% by score, weighted by float-adjusted market
# cap x score. A ticker without 12 months of history uses a
# 9-month window (and under 10 months it is not eligible).
# Months are trading days here, as everywhere in this engine:
# 252 + a 21-day skip, 189 for the fallback.
#
# Adaptations of a prices-only engine, stated plainly: there
# are no market caps, so the selected names are weighted by
# score alone; the index's 20% turnover buffer needs
# incumbency state a weights function does not carry, so
# selection is the plain top quintile; and the single-name
# cap (the lower of 9% or 3x the parent-index weight) and
# the sector constraint are defined relative to the parent
# index, so they do not apply. The semiannual March /
# September cadence is the caller's rebalance schedule.
SPMO_SKIP_DAYS = 21
SPMO_WINDOW_DAYS = 252
SPMO_FALLBACK_DAYS = 189
SPMO_FRACTION = 0.20
SPMO_Z_CAP = 3.0


def spmo_scores(R: pd.DataFrame) -> pd.Series:
    """S&P momentum scores on the estimation window ``R``
    (log returns), indexed by the tickers that carry one.

    Each ticker is measured on its own valid history: the
    252-day window ending 21 days before the window's end,
    or the 189-day fallback when 252 are not available;
    a ticker with neither is absent from the result (and
    from the cross-sectional mean and standard deviation,
    as the methodology excludes missing values there).
    """
    values = {}
    for col in R.columns:
        r = R[col].dropna()
        for span in (SPMO_WINDOW_DAYS, SPMO_FALLBACK_DAYS):
            if len(r) < span + SPMO_SKIP_DAYS:
                continue
            window = r.iloc[-(span + SPMO_SKIP_DAYS):-SPMO_SKIP_DAYS]
            simple = np.expm1(window.to_numpy())
            sd = float(np.std(simple, ddof=1))
            if sd > 0:
                values[col] = float(np.expm1(window.sum())) / sd
                break
    if not values:
        return pd.Series(dtype=float)
    x = pd.Series(values)
    sd_cs = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    if sd_cs > 0:
        z = ((x - float(x.mean())) / sd_cs).clip(
            -SPMO_Z_CAP, SPMO_Z_CAP)
    else:
        z = pd.Series(0.0, index=x.index)
    return pd.Series(
        np.where(z > 0, 1.0 + z,
                 np.where(z < 0, 1.0 / (1.0 - z), 1.0)),
        index=x.index)


def spmo_weights(R: pd.DataFrame) -> np.ndarray:
    """SPMO weights on the estimation window ``R``: the top
    quintile of :func:`spmo_scores` (at least one name),
    weighted in proportion to score. With no scorable
    ticker — a window shorter than the 189-day fallback
    plus the skip — the rule is 1/N."""
    p = R.shape[1]
    scores = spmo_scores(R)
    if scores.empty:
        return np.ones(p) / p
    k = max(1, math.ceil(SPMO_FRACTION * len(scores)))
    top = scores.sort_values(ascending=False, kind="stable").index[:k]
    chosen = scores[top]
    out = pd.Series(0.0, index=R.columns)
    out[top] = chosen / chosen.sum()
    return out.to_numpy()


def strategy_weights(name: str, R: pd.DataFrame, S: np.ndarray,
                     mu: np.ndarray, rf_d: float) -> np.ndarray:
    """Weights of one strategy on one estimation window.

    ``R`` log returns, ``S`` the (possibly cleaned) covariance,
    ``mu`` the (possibly cleaned) excess mean, ``rf_d`` the window's
    daily risk-free rate (for the CVaR scenarios). ``Average`` is not
    a single-window rule here — the backtest/snapshot layers build it
    as the mean of the other selected rules. An ``RS(n, w)`` name
    resolves to itself and is solved by :func:`rs_weights` from
    ``R`` alone.
    """
    p = R.shape[1]
    if name == "1/N":
        return np.ones(p) / p
    if name == "Min variance":
        return minvar_weights(S)
    if name == "Max diversification":
        return maxdiv_weights(S)
    if name == "NCO":
        return nco_weights(R, S)
    if name == "HRP":
        return hrp_weights(S)
    if name == "Risk parity":
        return erc_weights(S)
    if name == "Min CVaR":
        return cvar_weights(R.values - rf_d)
    if name == "Max Sharpe":
        return max_sharpe_weights(S, mu)
    if name == "SPMO":
        return spmo_weights(R)
    rs = _parse_rs(name)
    if rs is not None:
        return rs_weights(R, rs[0], rs[1])
    raise ValueError(f"Unknown or ensemble strategy: {name}")


def snapshot_weights(R: pd.DataFrame, cov_method: str, mean_method: str,
                     rf_d: float, strategies: list[str]) -> tuple[pd.DataFrame, dict]:
    """Current weights of ``strategies`` on the history ``R``.

    Combination names (see :data:`COMBO_STRATEGY_NAMES`) are solved
    on their own cleaning pair; ``(cov_method, mean_method)`` is the
    fallback pair for unqualified names. Each name's own estimation
    window (its ``@`` suffix; no suffix and ``@ All`` = all of
    ``R``) slices ``R`` for that strategy, and inputs are estimated
    once per distinct (window, pair) in play. Returns
    ``(weights, info)``: weights indexed by strategy with one
    column per ticker (Average = mean of the other selected
    rules), and the estimation diagnostics of the fallback pair
    on the full ``R``.
    """
    mu0, S0, info = estimate_inputs(R, cov_method, mean_method, rf_d)
    tickers = list(R.columns)
    rows = {}
    base = [s for s in strategies if s != "Average"]
    cache = {((cov_method, mean_method), None): (mu0, S0)}
    slices = {((cov_method, mean_method), None): R}
    for name in base:
        clean, win = parse_window(name)
        rule, cov_m, mean_m = resolve_strategy(clean)
        pair = (cov_m or cov_method, mean_m or mean_method)
        key = (pair, win)
        if key not in cache:
            R_w = R if win is None else R.tail(win)
            cache[key] = estimate_inputs(R_w, pair[0], pair[1],
                                         rf_d)[:2]
            slices[key] = R_w
        mu, S = cache[key]
        rows[name] = strategy_weights(rule, slices[key], S, mu, rf_d)
    if "Average" in strategies and base:
        rows["Average"] = np.mean([rows[s] for s in base], axis=0)
    order = _strategy_order(list(rows))
    W = pd.DataFrame({t: [rows[s][i] for s in order]
                      for i, t in enumerate(tickers)}, index=order)
    return W, info


# ---------------------------------------------------------------------------
# Performance statistics
# ---------------------------------------------------------------------------


def performance_stats(daily: np.ndarray, rf_daily: np.ndarray,
                      turnover: list[float]) -> dict:
    """Summary statistics of an out-of-sample daily return series.

    Sharpe/Sortino are on excess returns over ``rf_daily`` (aligned,
    same length). Max drawdown and the ulcer index are on the gross
    equity curve.
    """
    r = np.asarray(daily, dtype=float)
    rf = np.asarray(rf_daily, dtype=float)[:len(r)]
    exc = r - rf
    eq = np.cumprod(1.0 + r)
    dd = eq / np.maximum.accumulate(eq) - 1.0
    years = len(r) / TRADING_DAYS
    total = float(eq[-1] - 1.0) if len(r) else 0.0
    cagr = float(eq[-1] ** (1.0 / years) - 1.0) if years > 0 and eq[-1] > 0 else 0.0
    sd = float(exc.std(ddof=1)) if len(exc) > 1 else 0.0
    dsd = float(np.sqrt(np.mean(np.minimum(exc, 0.0) ** 2))) if len(exc) else 0.0
    max_dd = float(dd.min()) if len(dd) else 0.0
    return {
        "total_return": total,
        "cagr": cagr,
        "vol": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else 0.0,
        "sharpe": float(exc.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
        "sortino": float(exc.mean() / dsd * np.sqrt(TRADING_DAYS)) if dsd > 0 else 0.0,
        "max_dd": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else 0.0,
        "ulcer": float(np.sqrt(np.mean((dd * 100.0) ** 2))) if len(dd) else 0.0,
        "win_rate": float(np.mean(r > 0)) if len(r) else 0.0,
        "turnover": float(np.mean(turnover)) if turnover else 0.0,
        "n_days": int(len(r)),
    }


def asset_stats(prices: pd.DataFrame, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Per-asset statistics on the full price history.

    Annualised mean/volatility from daily log returns, Sharpe at a
    zero risk-free rate (a cross-asset comparison, not an excess
    claim), 12-1 momentum, max drawdown, last price, and beta to the
    benchmark's simple returns when one is supplied.
    """
    R = log_returns(prices)
    rows = {}
    bench_ret = (benchmark.pct_change() if benchmark is not None else None)
    for t in prices.columns:
        px = prices[t].dropna()
        if t not in R.columns or len(px) < 2:
            continue
        r = R[t]
        eq = px / px.iloc[0]
        dd = eq / eq.cummax() - 1.0
        mom = (px.iloc[-22] / px.iloc[-253] - 1.0) if len(px) >= 253 else np.nan
        beta = np.nan
        if bench_ret is not None:
            sr = px.pct_change()
            common = sr.index.intersection(bench_ret.index)
            if len(common) > 20 and bench_ret.loc[common].var() > 0:
                beta = float(sr.loc[common].cov(bench_ret.loc[common])
                             / bench_ret.loc[common].var())
        mu = float(r.mean() * TRADING_DAYS)
        vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
        rows[t] = {
            "last": float(px.iloc[-1]),
            "mu_ann": mu,
            "vol_ann": vol,
            "sharpe0": mu / vol if vol > 0 else 0.0,
            "momentum_12_1": float(mom) if pd.notna(mom) else np.nan,
            "max_dd": float(dd.min()),
            "beta": beta,
            "n_obs": int(len(px)),
        }
    return pd.DataFrame(rows).T


def _ema_state(s: pd.Series, fast: int, slow: int | None) -> bool | None:
    """One EMA trend state at a series' last close — the briefs'
    convention (movement-tracker's px_ema20 / ema_signal /
    ema_regime): pandas ewm(span=N, adjust=False) final values on
    the full history handed in; ``slow=None`` compares the last
    close itself against the fast EMA. None when the series holds
    fewer sessions than the slower span requires."""
    s = s.dropna()
    need = fast if slow is None else slow
    if len(s) < need:
        return None
    e_fast = s.ewm(span=fast, adjust=False).mean().iloc[-1]
    if slow is None:
        return bool(s.iloc[-1] > e_fast)
    e_slow = s.ewm(span=slow, adjust=False).mean().iloc[-1]
    return bool(e_fast > e_slow)


def _weekly_win_rate(w: pd.Series, bench: pd.Series) -> float:
    """Share of calendar weeks (W-FRI, per-period return from the
    available trading days, edges partial — winrate.py's
    period_returns) in which ``w``'s return beat ``bench``'s, over
    the weeks both series span inside ``w``'s window."""
    b = bench.dropna()
    b = b[(b.index >= w.index[0]) & (b.index <= w.index[-1])]
    if len(b) < 2 or len(w) < 2:
        return float("nan")

    def _period(s: pd.Series) -> pd.Series:
        g = s.groupby(pd.Grouper(freq="W-FRI"))
        return g.last() / g.first() - 1.0

    a, v = _period(w).align(_period(b), join="inner")
    valid = a.notna() & v.notna()
    n = int(valid.sum())
    return float((a[valid] > v[valid]).mean()) if n else float("nan")


def merged_stats(prices: pd.DataFrame,
                 benchmark: pd.Series | None = None,
                 win_benchmarks: pd.DataFrame | None = None,
                 rf_annual_pct: pd.Series | None = None) -> pd.DataFrame:
    """The one merged per-asset stats table (user direction
    2026-11-02): the union of the Strategies page's table
    (:func:`asset_stats`, plus the page's Rebased column) and the
    briefs' panel stats table (panel-stats.py / movement-tracker),
    ALL columns from both, computed once here so the dashboard and
    the briefs can never drift apart. Removal passes (user
    direction 2026-11-02): ``last`` and ``momentum_12_1`` dropped
    in pass 1, ``n_obs`` in pass 2 — Rebased carries the price
    level on the chart's own basis.

    Bases — where the two source tables disagreed, both statistics
    are kept under their source names rather than choosing one:

    - Full passed frame, per ticker (asset_stats basis):
      ``mu_ann``, ``vol_ann`` (annualised log-return mean/vol),
      ``sharpe0`` (Sharpe of log returns at rf = 0), ``max_dd``,
      ``beta`` (vs ``benchmark``'s simple returns, when supplied).
      (asset_stats also computes ``last``, ``momentum_12_1`` and
      ``n_obs``; the merged table drops all three, above.)
    - ``rebased`` follows the Strategies page's arithmetic (user
      direction 2026-11-02, test-pinned): 100 × last / first valid
      print of the passed frame. (The briefs' rebased is the same
      formula on their ~1y window frame — pass a 1y frame and the
      two agree.)
    - Trailing min(252, available) observations per ticker (the
      briefs' ~1y window basis): ``day_chg``, ``wk_chg`` (5),
      ``mo_chg`` (21), ``q_chg`` (63) trailing-session changes,
      ``ytd_chg`` (vs the last close of the prior calendar year
      inside the window; NaN when the window holds none),
      ``yr_chg`` (the trailing 252-session change on the FULL
      passed history — the one-year price change, added at the
      user's direction 2026-11-02 for the Strategies Prices
      table; NaN under 253 closes), ``vol``
      (annualised log-return vol over the window), ``sharpe``
      (daily SIMPLE returns in excess of the contemporaneous
      ^IRX-style yield — ``rf_annual_pct`` converted daily exactly
      as the briefs do, (1 + y/100)**(1/252) − 1, forward-filled;
      rf = 0 when no yield series is supplied), ``ulcer`` (sqrt of
      the mean squared drawdown from the running peak), ``dd``
      (current drawdown from the window maximum), ``dd_q``
      (quantile of the current drawdown within the window's
      running-drawdown history — a deeper current drawdown reads
      a higher quantile; the briefs' (hist >= last).mean() on the
      non-positive drawdown series).
    - Trend states on the FULL passed history (the briefs'
      convention): ``px_ema20`` (close > EMA20), ``ema_long``
      (EMA20 > EMA50), ``ema_regime`` (EMA50 > EMA200); True = L,
      False = S, None when the history is too short.
    - ``win_rate``: mean over VT / VEA / VWO of the weekly win
      rate against that benchmark over the trailing window (the
      briefs' definition), from the ``win_benchmarks`` closes;
      NaN when no benchmark frame is supplied.

    Units: every rate-like column is a FRACTION (the briefs store
    percents — divide-by-100 is the only unit change); ``rebased``
    is index points (start = 100), the trend states are bool /
    None.
    """
    base = asset_stats(prices, benchmark)
    rows = {}
    for t in base.index:
        px = prices[t].dropna().sort_index()
        w = px.iloc[-TRADING_DAYS:]
        row = {k: base.loc[t, k] for k in base.columns}
        row["rebased"] = float(px.iloc[-1] / px.iloc[0] * 100.0)
        v = w.values
        row["day_chg"] = float(v[-1] / v[-2] - 1.0) if len(v) >= 2 \
            else float("nan")

        def _trail(k: int) -> float:
            return float(v[-1] / v[-k - 1] - 1.0) if len(v) >= k + 1 \
                else float("nan")
        row["wk_chg"], row["mo_chg"], row["q_chg"] = (
            _trail(5), _trail(21), _trail(63))
        ytd_cut = pd.Timestamp(year=w.index[-1].year - 1, month=12, day=31)
        prior = w[w.index <= ytd_cut]
        row["ytd_chg"] = (float(w.iloc[-1] / prior.iloc[-1] - 1.0)
                          if len(prior) else float("nan"))
        row["yr_chg"] = (float(px.iloc[-1]
                               / px.iloc[-TRADING_DAYS - 1] - 1.0)
                         if len(px) >= TRADING_DAYS + 1
                         else float("nan"))
        row["px_ema20"] = _ema_state(px, 20, None)
        row["ema_long"] = _ema_state(px, 20, 50)
        row["ema_regime"] = _ema_state(px, 50, 200)
        lr = np.log(w / w.shift(1)).dropna()
        row["vol"] = (float(lr.std() * np.sqrt(TRADING_DAYS))
                      if len(lr) >= 2 else 0.0)
        dd_hist = w / w.cummax() - 1.0
        row["dd"] = float(dd_hist.iloc[-1])
        row["dd_q"] = float((dd_hist >= dd_hist.iloc[-1]).mean())
        if len(w) >= 3:
            row["ulcer"] = float(np.sqrt(np.mean(dd_hist.values ** 2)))
            rets = (w / w.shift(1) - 1.0).dropna()
            if rf_annual_pct is not None and len(rf_annual_pct):
                ya = (rf_annual_pct.reindex(
                    rf_annual_pct.index.union(rets.index))
                    .ffill().reindex(rets.index))
                excess = rets - ((1.0 + ya / 100.0) ** (1.0 / 252.0)
                                 - 1.0).fillna(0.0)
            else:
                excess = rets
            sd = float(excess.std())
            row["sharpe"] = (float(excess.mean() / sd
                                   * np.sqrt(TRADING_DAYS))
                             if sd > 0 else 0.0)
        else:
            row.update({"ulcer": 0.0, "sharpe": 0.0})
        if win_benchmarks is not None:
            rates = [_weekly_win_rate(w, win_benchmarks[b])
                     for b in ("VT", "VEA", "VWO")
                     if b in win_benchmarks.columns]
            rates = [r for r in rates if pd.notna(r)]
            row["win_rate"] = float(np.mean(rates)) if rates \
                else float("nan")
        else:
            row["win_rate"] = float("nan")
        rows[t] = row
    cols = ["rebased", "day_chg", "wk_chg", "mo_chg", "q_chg",
            "ytd_chg", "yr_chg", "px_ema20", "ema_long", "ema_regime",
            "mu_ann",
            "vol_ann", "vol", "sharpe0", "sharpe",
            "ulcer", "win_rate", "dd", "dd_q", "max_dd", "beta"]
    out = pd.DataFrame(rows).T
    return out.reindex(columns=cols)


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------


def walk_forward_backtest(
    prices: pd.DataFrame,
    *,
    strategies: list[str],
    cov_method: str = "bbp",
    mean_method: str = "james_stein",
    seed: int = 126,
    hold: int = 21,
    cost_bps: float = 0.0,
    window: int | None = None,
    rf_annual_pct: pd.Series | None = None,
) -> dict | None:
    """Walk-forward backtest of ``strategies`` on ``prices``.

    Expanding estimation window seeded after ``seed`` log returns;
    rebalances every ``hold`` trading days. At each rebalance the
    inputs are estimated on all log returns up to that close (no
    lookahead) — once per distinct cleaning pair in play, since
    combination strategy names carry their own pair and
    ``(cov_method, mean_method)`` is only the fallback for
    unqualified names — the snapshot cost of trading from the drifted
    weights to the new weights — ``Σ|Δw| × cost_bps`` — is charged
    to the first day of the new holding period, and weights drift
    with daily simple returns in between. ``Average`` (when
    selected) holds the mean of the other selected rules' weights
    at each rebalance and is scored exactly like a rule.

    ``window`` (user direction 2026-10-31) switches the estimation
    to a rolling window: each rebalance then estimates on the
    trailing ``window`` log returns — fewer while less history
    exists — and the window-mean risk-free rate follows the same
    slice. ``None`` (the default) is the expanding window and is
    bit-identical to the pre-``window`` behavior. ``seed`` only
    sets where out-of-sample scoring starts, so every
    ``(seed, window)`` pair is valid.

    Per-strategy windows (user direction 2026-11-02): a strategy
    name may carry its own window as an ``@`` suffix (see
    :func:`parse_window`), which overrides ``window`` for that
    strategy — the estimation slice and its risk-free mean follow
    the strategy's own window. Unsuffixed names use ``window``.

    Returns ``None`` when the common history cannot seed plus hold.
    Otherwise a dict with the OOS daily ``returns`` and ``equity``
    frames, the ``stats`` frame (one row per strategy), the
    ``weights_history`` (rebalance date → strategy → weight vector),
    and run metadata.
    """
    px = prices.dropna().sort_index()
    if len(px) < seed + hold + 1:
        return None
    if window is not None and window < 1:
        raise ValueError(f"window must be positive, got {window}")
    simple = px.pct_change().dropna()
    logret = np.log(px / px.shift(1)).dropna()
    simple = simple.loc[logret.index]
    rf_all = rf_daily_series(logret.index, rf_annual_pct)
    tickers = list(px.columns)
    base = [s for s in strategies if s != "Average"]
    resolved = {}
    eff_window = {}
    for name in base:
        clean, w_suf = parse_window(name)
        resolved[name] = resolve_strategy(clean)
        # A name's own @ window wins; an unsuffixed name falls
        # back to the call's ``window`` (None = expanding).
        eff_window[name] = w_suf if " @ " in name else window
    labels = list(base) + (["Average"] if "Average" in strategies and base else [])
    if not labels:
        return None
    port: dict[str, list[float]] = {lab: [] for lab in labels}
    turnover: dict[str, list[float]] = {lab: [] for lab in labels}
    drift: dict[str, np.ndarray | None] = {lab: None for lab in labels}
    weights_history: list[dict] = []
    n_reb = 0
    for start in range(seed, len(simple), hold):
        end = min(start + hold, len(simple))
        cache: dict = {}
        slices: dict = {}
        W = {}
        for name in base:
            rule, cov_m, mean_m = resolved[name]
            pair = (cov_m or cov_method, mean_m or mean_method)
            win = eff_window[name]
            key = (win, pair)
            if key not in cache:
                lo = 0 if win is None else max(0, start - win)
                R_w = logret.iloc[lo:start]
                rf_w = float(rf_all[lo:start].mean())
                cache[key] = estimate_inputs(R_w, pair[0], pair[1],
                                             rf_w)[:2]
                slices[key] = (R_w, rf_w)
            mu, S = cache[key]
            R_w, rf_w = slices[key]
            W[name] = strategy_weights(rule, R_w, S, mu, rf_w)
        if "Average" in labels:
            W["Average"] = np.mean([W[s] for s in base], axis=0)
        n_reb += 1
        weights_history.append({
            "date": simple.index[start],
            "weights": {lab: dict(zip(tickers, W[lab])) for lab in labels},
        })
        for lab in labels:
            w = W[lab]
            trade = 0.0
            if drift[lab] is not None:
                delta = float(np.abs(w - drift[lab]).sum())
                turnover[lab].append(0.5 * delta)
                trade = delta * cost_bps / 1e4
            wcur = w.copy()
            for t in range(start, end):
                r = simple.iloc[t].values
                day = float(wcur @ r)
                if t == start:
                    day -= trade
                port[lab].append(day)
                wcur = wcur * (1.0 + r)
                s = wcur.sum()
                wcur = wcur / s if s > 0 else np.ones_like(wcur) / len(wcur)
            drift[lab] = wcur
    dates = simple.index[seed:]
    rf_oos = rf_all[seed:]
    returns = pd.DataFrame(port, index=dates)
    equity = (1.0 + returns).cumprod()
    stats = pd.DataFrame({lab: performance_stats(port[lab], rf_oos, turnover[lab])
                          for lab in labels}).T
    stats = stats.sort_values("sharpe", ascending=False)
    return {
        "returns": returns,
        "equity": equity,
        "stats": stats,
        "weights_history": weights_history,
        "oos_start": dates[0], "oos_end": dates[-1],
        "n_rebalances": n_reb, "seed": seed, "hold": hold,
        "window": window,
        "cov_method": cov_method, "mean_method": mean_method,
        "cost_bps": cost_bps, "tickers": tickers,
    }
