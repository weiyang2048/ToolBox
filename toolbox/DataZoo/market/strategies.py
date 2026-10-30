"""Portfolio strategy engine: cleaned inputs + walk-forward backtest.

The canonical implementation of the machinery the daily briefs run
(their ``maxsharpe-alloc.py`` pipeline), built for arbitrary
universes of ETFs / stocks:

- **Covariance cleaning** — the sample covariance is one choice, not
  the default truth. Cleaners: BBP (the feasible rotationally
  invariant estimator of Bun–Bouchaud–Potters, via the hybrid
  Stieltjes transform — the briefs' cleaner), Marčenko–Pastur
  eigenvalue clipping (Laloux et al.), and Ledoit–Wolf linear
  shrinkage to a scaled identity.
- **Mean cleaning** — sample means are the noisiest input in
  Markowitz (Merton 1980). Cleaners: James–Stein shrinkage toward
  the grand mean (the briefs' cleaner), Jorion's Bayes–Stein, and
  the grand mean itself (full shrinkage).
- **Strategies** — long-only, fully invested (``w >= 0``,
  ``sum w = 1``), solved from the *cleaned* inputs the caller chose:
  1/N, minimum variance, maximum diversification, NCO (nested
  clustered optimisation), HRP, risk parity (ERC), minimum CVaR
  (95%), maximum Sharpe (tangency), and the ensemble Average of the
  selected rules.
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

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform

TRADING_DAYS = 252

COV_METHODS = ("sample", "bbp", "clip", "ledoit_wolf")
COV_LABELS = {
    "sample": "Sample (raw)",
    "bbp": "BBP — RIE cleaned",
    "clip": "MP clipping",
    "ledoit_wolf": "Ledoit–Wolf",
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
    "Average",
)

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
    }
    if method == "sample" or q >= 1.0:
        # RIE/clipping need q < 1; with q >= 1 the sample matrix is
        # singular and the honest fallbacks are shrinkage or sample.
        S = S_raw if method == "sample" else ledoit_wolf_cov(R)[0]
        if method == "ledoit_wolf":
            S, info["shrinkage"] = ledoit_wolf_cov(R)
    elif method == "bbp":
        S = np.outer(vols, vols) * bbp_clean_corr(C, q)
    elif method == "clip":
        S = np.outer(vols, vols) * clip_clean_corr(C, q)
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


def strategy_weights(name: str, R: pd.DataFrame, S: np.ndarray,
                     mu: np.ndarray, rf_d: float) -> np.ndarray:
    """Weights of one strategy on one estimation window.

    ``R`` log returns, ``S`` the (possibly cleaned) covariance,
    ``mu`` the (possibly cleaned) excess mean, ``rf_d`` the window's
    daily risk-free rate (for the CVaR scenarios). ``Average`` is not
    a single-window rule here — the backtest/snapshot layers build it
    as the mean of the other selected rules.
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
    raise ValueError(f"Unknown or ensemble strategy: {name}")


def snapshot_weights(R: pd.DataFrame, cov_method: str, mean_method: str,
                     rf_d: float, strategies: list[str]) -> tuple[pd.DataFrame, dict]:
    """Current weights of ``strategies`` on the window ``R``.

    Returns ``(weights, info)``: weights indexed by strategy with one
    column per ticker (Average = mean of the other selected rules),
    and the estimation diagnostics from :func:`estimate_inputs`.
    """
    mu, S, info = estimate_inputs(R, cov_method, mean_method, rf_d)
    tickers = list(R.columns)
    rows = {}
    base = [s for s in strategies if s != "Average"]
    for name in base:
        rows[name] = strategy_weights(name, R, S, mu, rf_d)
    if "Average" in strategies and base:
        rows["Average"] = np.mean([rows[s] for s in base], axis=0)
    order = [s for s in STRATEGY_NAMES if s in rows]
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
    downside = exc[exc < 0]
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
    rf_annual_pct: pd.Series | None = None,
) -> dict | None:
    """Walk-forward backtest of ``strategies`` on ``prices``.

    Expanding estimation window seeded after ``seed`` log returns;
    rebalances every ``hold`` trading days. At each rebalance the
    inputs are estimated on all log returns up to that close (no
    lookahead), the snapshot cost of trading from the drifted
    weights to the new weights — ``Σ|Δw| × cost_bps`` — is charged
    to the first day of the new holding period, and weights drift
    with daily simple returns in between. ``Average`` (when
    selected) holds the mean of the other selected rules' weights
    at each rebalance and is scored exactly like a rule.

    Returns ``None`` when the common history cannot seed plus hold.
    Otherwise a dict with the OOS daily ``returns`` and ``equity``
    frames, the ``stats`` frame (one row per strategy), the
    ``weights_history`` (rebalance date → strategy → weight vector),
    and run metadata.
    """
    px = prices.dropna().sort_index()
    if len(px) < seed + hold + 1:
        return None
    simple = px.pct_change().dropna()
    logret = np.log(px / px.shift(1)).dropna()
    simple = simple.loc[logret.index]
    rf_all = rf_daily_series(logret.index, rf_annual_pct)
    tickers = list(px.columns)
    base = [s for s in strategies if s != "Average"]
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
        R = logret.iloc[:start]
        rf_win = float(rf_all[:start].mean())
        mu, S, _info = estimate_inputs(R, cov_method, mean_method, rf_win)
        W = {name: strategy_weights(name, R, S, mu, rf_win) for name in base}
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
        "cov_method": cov_method, "mean_method": mean_method,
        "cost_bps": cost_bps, "tickers": tickers,
    }
