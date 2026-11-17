"""Bitcoin power law + 63-day BTC correlations, for the dashboard.

Moved here from the morning brief's Flaco script
(btc-powerlaw-correlations.py) at the user's direction 2026-11-01:
the chart now lives on the Markets page, in the Commodities section
under the board. The port is faithful — same data, same fits, same
figure; the brief-only EMA strategy sidecar was dropped (nothing
has consumed it since the brief's EMA box was removed in 2026-10).

Fit: power law ln(P) = lnA + b*ln(days since 2009-01-03) on the
extended blockchain.info history (Mt. Gox era onward, 2010-07-17+).
Price line: Yahoo BTC-USD daily closes (2014-09-17+) plus
blockchain.info points for 2010-2014. Correlations: 63-day rolling
BTC×gold (GLD) and BTC×SPX (SPY) of daily log returns from
2014-09-17, extended back to 2010 on the sparse blockchain.info
series (63-calendar-day windows, plotted thinner).
BTC market cap comes from blockchain.info; gold market cap =
Yahoo GC=F ($/ozt) × WGC above-ground stock (end-2024 216,265 t,
walked with ~3,738 t/yr mine production). The BTC÷gold ratio gets
its own power-law fit, whose 100% crossing is the model-implied
"flippening" date. Both fits also carry slope standard errors
(fit_power_law_full) — classical OLS and a prewhitened
Newey-West (HAC) variant for the residuals' strong persistence —
surfaced in the stats payload and the figure's legend labels.
"""
import datetime
import io
import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from scipy.stats import t as _student_t

from toolbox.DataZoo.market.eastern import (
    ET,
    asof_stamp,
    utc_index_to_et_dates,
)
from toolbox.DataZoo.market.yahoo_prices import fetch_json, with_retry, yahoo_daily

UA = "toolbox-datazoo"
GENESIS = datetime.date(2009, 1, 3)
OZT_PER_TONNE = 32150.7
# WGC end-2024 anchor: 216,265 t above-ground gold.
_ANCHOR_DAYS = (pd.Timestamp("2024-12-31") - pd.Timestamp(GENESIS)).days


def _blockchain_chart(endpoint, start, col):
    """Fetch a blockchain.info time-series chart into a DataFrame."""
    url = (f"https://api.blockchain.info/charts/{endpoint}"
           "?timespan=all&format=json")
    d = fetch_json(url, 60, user_agent=UA)
    cutoff = pd.Timestamp(start)
    ys = pd.Series([v["y"] for v in d["values"]], dtype=float)
    ts = pd.to_datetime([v["x"] for v in d["values"]], unit="s",
                        utc=True).tz_convert(None).normalize()
    mask = (ys > 0) & (ts >= cutoff)
    return pd.DataFrame(
        {"t": ts[mask], col: ys[mask].to_numpy()}).set_index("t")


def blockchain_btc():
    """Full BTC price history back to the Mt. Gox era (~4-day spacing)."""
    return _blockchain_chart("market-price", "2010-07-17", "p")


def blockchain_marketcap():
    """BTC market cap history (sparse, ~6-day spacing)."""
    return _blockchain_chart("market-cap", "2013-01-01", "mc")


def days_since_genesis(idx):
    """Days since 2009-01-03 for each timestamp in idx (vectorized)."""
    return (idx.normalize() - pd.Timestamp(GENESIS)).days.to_numpy(float)


def fit_power_law(days_arr, values):
    """OLS fit of ln(values) = lnA + b*ln(days); (slope b, lnA, sigma)."""
    lnV = np.log(values)
    ln_d = np.log(days_arr)
    b, lnA = np.polyfit(ln_d, lnV, 1)
    sigma = float((lnV - (lnA + b * ln_d)).std())
    return b, lnA, sigma


# --- Fixed-b (Kiefer-Vogelsang) critical values, simulated ---
# The fixed-b t-limit for the slope, partialling out the design
# F = (1, ln-days)', is G(1) / sqrt(D_1) with G the partial-sum
# process of the demeaned regressor against Brownian motion,
# Gp its residual projection, and D_1 the Bartlett-kernel
# double integral of dGp dGp at b = 1 — evaluated through the
# kernel's window identity, D_1 = int_{-1}^{1} [Gp(min(u+1,1))
# - Gp(max(u,0))]^2 du (evaluated in the split form
# sum_i Gp_i^2 + sum_{j>=1} (Gp_m - Gp_j)^2 — the same terms,
# no window gathers). The limit depends on the design only
# through the regressor's limit function, here the empirical
# quantile function of the fit's own ln-days, so the critical
# value follows whatever design it is given (measured: 5.996
# on the note's span design against its recorded 6.005 at
# finer resolution — inside that study's own MC error; KV's
# location-model table value 4.78 would under-cover on this
# design even at rho = 0). Fixed seed and settings make the
# value a deterministic constant per design; the simulation
# costs ~13 s at the shipped settings, so results are
# cached — in-process and on disk, matched by design
# tolerance (see _fb_store_match) — and repeat fits, new
# processes and one-observation design updates reuse the
# stored value instead of paying it again.
_FB_GRID = 2000        # limit-process grid resolution m
_FB_REPS = 60000       # simulated |t| draws per design
_FB_SEED = 20261115


def _fb_t_draws(ln_d, m, reps, seed):
    """Draws from the design-projected fixed-b t-limit (b = 1)."""
    xs = np.sort(np.asarray(ln_d, dtype=float))
    n = len(xs)
    r = (np.arange(m) + 0.5) / m
    idx = np.minimum(n - 1, (r * n).astype(int))
    x = xs[idx]
    f_mat = np.column_stack([np.ones(m), x])
    q = x - x.mean()
    minv = np.linalg.inv((f_mat.T @ f_mat) / m)
    h_mat = np.cumsum(q[:, None] * f_mat, axis=0) / m
    rng = np.random.default_rng(seed)
    sq = math.sqrt(1.0 / m)
    ts = np.empty(reps)
    chunk = 1000
    for s in range(0, reps, chunk):
        p = min(chunk, reps - s)
        dw = rng.standard_normal((p, m)) * sq
        theta = (dw @ f_mat) @ minv.T
        g = np.cumsum(q[None, :] * dw, axis=1)
        gp = np.empty((p, m + 1))
        gp[:, 0] = 0.0
        gp[:, 1:] = g - (h_mat @ theta.T).T
        d1 = ((gp * gp).sum(axis=1)
              + ((gp[:, -1:] - gp) ** 2).sum(axis=1)
              - gp[:, -1] ** 2) / m
        ts[s:s + p] = g[:, -1] / np.sqrt(d1)
    return ts


# --- CV cache: tolerance-matched, persistent ---
# The cache this replaces keyed entries by the design's
# exact bytes, so the daily case — the same design plus one
# observation — missed completely and paid the ~13 s
# simulation again for an identical value. Matching is by
# tolerance instead: a stored entry serves any design whose
# size is within _FB_KEY_N_TOL and whose ln-quantile
# function (at _FB_KEY_POINTS probability points) is
# everywhere within _FB_KEY_Q_TOL ln-days. The tolerances
# sit far inside the value's own uncertainty: appending one
# observation moves a design's quantiles by ~0.0006 and its
# freshly simulated CV by at most 0.002 (both live designs,
# measured 2026-11-15), against ~0.01-0.02 of simulation
# resolution noise — while a materially different design
# still misses (the note's span design vs the live price
# design: quantile functions up to 0.07 apart). Entries
# persist as JSON under ~/.cache/toolbox/ (the
# news_cloud_tone cache-dir precedent), written atomically
# (temp file + rename, so a concurrent reader never sees a
# torn file); a missing, corrupt or unwritable store falls
# back to simulation — the store is a cache, never a
# source of truth.
_FB_KEY_POINTS = 41
_FB_KEY_Q_TOL = 0.01
_FB_KEY_N_TOL = 50
_FB_STORE_MAX = 64
_FB_STORE = None       # lazily loaded [{n, q, cv}, ...]


def _fb_design_fingerprint(ln_d):
    """(n, ln-quantiles at _FB_KEY_POINTS probability points)."""
    xs = np.sort(np.asarray(ln_d, dtype=float))
    qs = np.quantile(xs, np.linspace(0.0, 1.0, _FB_KEY_POINTS))
    return len(xs), tuple(np.round(qs, 4).tolist())


def _fb_store_path():
    return Path.home() / ".cache" / "toolbox" / "fixed_b_cv.json"


def _fb_store_load():
    """The persisted CV entries; [] on any read problem."""
    global _FB_STORE
    if _FB_STORE is not None:
        return _FB_STORE
    entries = []
    try:
        raw = json.loads(_fb_store_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = []
    if isinstance(raw, list):
        for item in raw:
            try:
                n = int(item["n"])
                q = [float(v) for v in item["q"]]
                cv = float(item["cv"])
            except (KeyError, TypeError, ValueError):
                continue
            if (len(q) == _FB_KEY_POINTS and math.isfinite(cv)
                    and 1.0 < cv < 20.0
                    and all(math.isfinite(v) for v in q)):
                entries.append({"n": n, "q": q, "cv": cv})
    _FB_STORE = entries
    return entries


def _fb_store_save():
    """Persist the entries atomically; failure is silent —
    a lost write costs one re-simulation, never a value."""
    tmp = None
    try:
        path = _fb_store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent,
                                   prefix=path.name + ".",
                                   suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(_FB_STORE, fh)
        os.replace(tmp, path)
        tmp = None
    except OSError:
        if tmp is not None:
            with suppress(OSError):
                os.unlink(tmp)


def _fb_store_match(n, qs):
    """The stored CV for a design within tolerance, or None."""
    for entry in reversed(_fb_store_load()):
        if abs(entry["n"] - n) > _FB_KEY_N_TOL:
            continue
        gap = max(abs(a - b) for a, b in zip(entry["q"], qs, strict=True))
        if gap <= _FB_KEY_Q_TOL:
            return entry["cv"]
    return None


def _fixed_b_cv(ln_d):
    """The 95% fixed-b critical value for the design ln_d.

    Deterministic per design (fixed seed and settings);
    served from the tolerance-matched persistent cache
    (see _fb_store_match) when a stored design is close
    enough, simulated otherwise and then stored.
    """
    n, qs = _fb_design_fingerprint(ln_d)
    cv = _fb_store_match(n, qs)
    if cv is not None:
        return cv
    draws = _fb_t_draws(ln_d, _FB_GRID, _FB_REPS, _FB_SEED)
    cv = float(np.quantile(np.abs(draws), 0.95))
    entries = _fb_store_load()
    entries.append({"n": n, "q": list(qs), "cv": cv})
    del entries[:-_FB_STORE_MAX]     # keep the newest entries
    _fb_store_save()
    return cv


def fit_power_law_full(days_arr, values, stage=None):
    """The fit_power_law fit with slope uncertainty, as a dict.

    Keys: b, lnA, sigma — the identical values fit_power_law
    returns (sigma the ddof=0 residual std) — plus n, the
    classical OLS standard errors b_se / lnA_se and interval
    b_ci95, and the Newey-West (HAC) counterparts b_se_nw /
    lnA_se_nw / b_ci95_nw with the bandwidth nw_bandwidth and
    the recoloring coefficient nw_rho actually used. The
    fixed-b extension adds b_ci95_fb / lnA_ci95_fb with the
    simulated critical value fb_cv at bandwidth fraction
    fb_b = 1.0.

    Classical convention: s^2 = RSS/(n - 2) with t critical
    values at n - 2 dof, settled by Monte Carlo (2026-11-14;
    20,000 reps per cell on this board's own design —
    blockchain.info ln-days, n = 1484, residual sigma 0.69).
    At n = 1484 every dof convention covers at 95.0-95.1%, but
    the ddof=0 / normal variant sags as n falls (94.6% at
    n = 200, 94.2% at n = 60) while this one holds 95.1%; it
    is the only variant calibrated at every n, and its mean SE
    matched the empirical sd of the slope to 0.3%. Intervals
    also cover at 94.9% under iid resamples of the real
    residuals (same marginal shape, no dependence).

    CAVEAT — the classical interval is a precision statement
    against iid log-residuals, and the real ones are anything
    but: the price fit's residuals have lag-1 autocorrelation
    0.99. In an AR(1) Monte Carlo matched to that persistence
    (same design, same marginal sigma) the classical interval
    covers the true slope only ~10% of the time and the true
    sampling sd runs ~15x b_se. Read b_se as the floor on the
    exponent's uncertainty; b_se_nw is the persistence-aware
    figure.

    Newey-West design (Monte Carlo record, 2026-11-14, same
    harness): plain Bartlett NW on the residual scores fails
    at EVERY bandwidth here — coverage peaks at 68% near
    L = 200 — because OLS projection attenuates the residual
    autocovariances at exactly the long lags that carry the
    long-run variance (the population autocovariance at lag
    100 is 1.7e-7; the residual sample analogue is ~0). The
    shipped variant therefore prewhitens (Andrews-Monahan):
    the scores are filtered by the residuals' AR(1)
    coefficient, a Bartlett long-run variance with a
    Newey-West-1994 plug-in bandwidth (nw_bandwidth, small by
    construction) is computed on the whitened scores, and the
    sandwich is recolored by (1 - nw_rho)^-2 with nw_rho the
    Kendall bias-corrected AR(1) coefficient — the correction
    is first-order here because (1 - rho)^-2 amplifies the
    coefficient's -0.0045 small-sample bias (measured) into a
    ~40% variance understatement if left raw. Measured
    coverage of b_ci95_nw: ~92% under the matched AR(1), 95%
    under iid (where it agrees with the classical interval).
    92 is not 95, and no residual-based variant in the
    harness did better without inflating the mean SE to ~2x
    the true sd: at rho = 0.99 the long-run variance rests on
    ~n(1 - rho) ~ 15 effective observations, so treat the NW
    interval as the honest scale of the uncertainty, not an
    exact 95%.

    Operating boundary, in local-to-unity terms (measured
    2026-11-15 in "The HAC Grid in Local-to-Unity Terms",
    extending "When Newey-West Fails" and "The Persistence
    Grid"): in the persistent tail the coordinate that
    governs HAC coverage is c = n(rho - 1), not rho alone.
    At fixed c this prewhitened interval's coverage is
    n-invariant (spread at most 0.89pp across n = 371 to
    2968, 20,000 reps per cell) and crosses 90% at
    c* ~ -9.6: trustworthy at or above c*, i.e. while
    n(1 - rho) stays under ~9.6, and degrading beyond it by
    standard-error dispersion, not bias (the median SE
    stays at the true sd; the left tail of SE draws
    fattens). Plain NW has no 90% crossing anywhere in
    this regime at any n (its crossing sits at c ~ -255) —
    broken throughout the tail, as recorded above. The
    classical interval admits no c boundary at all: its
    coverage is a function of (rho, n) jointly, with up to
    31.6pp spread at fixed c, so b_se remains a floor,
    never an interval. This board's price fit sits at
    c ~ -13.7 (n = 1484, rho-hat = 0.9908), on the safe
    side of c* by a factor ~1.4; the pooled collapse curve
    predicts 91.4% coverage there, against 92.0% measured
    directly at that cell.

    The companion ratio fit (BTC/gold market cap, the same
    shipped function) measures on the same scale
    (2026-11-15): n = 1168, rho-hat = 0.9923, so c ~ -9.0 —
    just inside the degrading side of c*, where the price
    fit clears the boundary by a factor ~1.4 in |c| the
    ratio reaches only ~0.94x |c*|. The pooled collapse
    curve predicts ~90% coverage (0.897) for the ratio's
    NW interval, against 91.4% predicted and 92.0%
    measured for the price fit: ratio_slope_se_nw carries
    no margin past the boundary, and its interval should
    be read as already at the edge of the degrading zone.

    ``stage`` (2026-11-16) is an optional progress hook:
    when given, it is called once with a label just before
    the fixed-b block below — the fit's one opaque,
    potentially slow step (the simulated critical value) —
    so a caller's progress bar can name it instead of
    sitting silent. It changes no value.

    Fixed-b interval (shipped 2026-11-15; measured in
    "Fixed-b Critical Values on the Power-Law Design" on
    this same harness, 20,000 reps per cell): the
    prewhitened statistic with its whitened-score Bartlett
    bandwidth fixed at M = n - 1 (b = 1), judged against
    fb_cv — the design-projected Kiefer-Vogelsang critical
    value simulated for this fit's own ln-days design (see
    _fixed_b_cv; the projection raises it ~26-28% above
    KV's location-model table, so no table value is
    hardcoded). Measured coverage of b_ci95_fb: 0.950 to
    0.982 across the whole local-to-unity tail, including
    0.950 at the random-walk boundary, where b_ci95_nw
    covers 0.753. The pairing is a heuristic — plain
    Bartlett fixed-b values on a prewhitened statistic,
    with no prewhitened limit theory behind it — and
    measured conservative (1.4-3.2pp of over-coverage
    mid-tail; at the live cells the note interpolates
    coverage ~0.98 for both fits, against ~0.92 / ~0.90
    for b_ci95_nw). The premium is width: computed
    directly on the live data (2026-11-15) the price
    fit's fb half-width is 2.98 against the NW 1.76, and
    the ratio fit's fb interval is [-2.97, 12.54] —
    wider than the note's c-interpolated estimates (3.69
    and [-0.61, 10.18]), which scaled the shipped
    half-widths by the MC width ratios rather than
    computing the fb SE on the live residuals; in the
    tail the SE draw itself is dispersed, so the direct
    and interpolated figures need not agree. Either way
    the ratio interval includes a zero slope: at the
    coverage that holds in the tail, the ratio's power
    law is not distinguishable from flat. b_ci95_fb is
    the interval to quote when coverage in the tail is
    the question; b_ci95_nw remains the narrower,
    boundary-limited one.
    """
    days_arr = np.asarray(days_arr, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(days_arr)
    if n < 3:
        msg = f"fit_power_law_full needs >= 3 points, got {n}"
        raise ValueError(msg)
    b, lnA, sigma = fit_power_law(days_arr, values)
    ln_d = np.log(days_arr)
    resid = np.log(values) - (lnA + b * ln_d)
    s2 = float(resid @ resid) / (n - 2)
    xm = float(ln_d.mean())
    sxx = float(((ln_d - xm) ** 2).sum())
    b_se = math.sqrt(s2 / sxx)
    lnA_se = math.sqrt(s2 * (1.0 / n + xm * xm / sxx))
    half = float(_student_t.ppf(0.975, n - 2)) * b_se

    # --- Newey-West, AR(1)-prewhitened (see docstring) ---
    xtx_inv = np.array([[1.0 / n + xm * xm / sxx, -xm / sxx],
                        [-xm / sxx, 1.0 / sxx]])
    scores = np.column_stack([resid, resid * ln_d])
    rho = float(resid[:-1] @ resid[1:] / (resid[:-1] @ resid[:-1]))
    rho = min(max(rho, -0.999), 0.999)
    white = scores[1:] - rho * scores[:-1]
    nw_n = n - 1
    v_w = white @ xtx_inv[1]
    vv = float(v_w @ v_w)
    r_w = float(v_w[:-1] @ v_w[1:] / vv) if vv > 0 else 0.0
    r_w = min(max(r_w, -0.999), 0.999)
    alpha1 = (2.0 * r_w / (1.0 - r_w ** 2)) ** 2
    nw_lag = int(min(25, max(0, math.floor(
        1.1447 * (alpha1 * nw_n) ** (1.0 / 3.0)))))
    s_mat = (white.T @ white) / nw_n
    for j in range(1, nw_lag + 1):
        gj = (white[:-j].T @ white[j:]) / nw_n
        s_mat = s_mat + (1.0 - j / (nw_lag + 1.0)) * (gj + gj.T)
    rho_t = min(max(rho + (1.0 + 3.0 * rho) / n, -0.9995), 0.9995)
    cov_nw = nw_n * xtx_inv @ (s_mat / (1.0 - rho_t) ** 2) @ xtx_inv
    lnA_se_nw = math.sqrt(max(float(cov_nw[0, 0]), 0.0))
    b_se_nw = math.sqrt(max(float(cov_nw[1, 1]), 0.0))
    half_nw = float(_student_t.ppf(0.975, n - 2)) * b_se_nw

    if stage is not None:
        stage("fixed-b interval (simulated critical value; "
              "the cold wait's dominant cost)")
    # --- Fixed-b (b = 1) on the prewhitened statistic ---
    # The same whitened scores, recolor and sandwich as the
    # NW block, with the Bartlett truncation fixed at
    # M = nw_n (weights 1 - j/M) instead of the plug-in
    # bandwidth, and the simulated design-projected fixed-b
    # critical value instead of the t value. Autocovariances
    # of the whitened scores via FFT (the direct-lag sums
    # would run to lag nw_n - 2).
    m_fb = nw_n
    nfft = 1 << int(np.ceil(np.log2(2 * nw_n)))
    fw1 = np.fft.rfft(white[:, 0], nfft)
    fw2 = np.fft.rfft(white[:, 1], nfft)
    c11 = np.fft.irfft(np.conj(fw1) * fw1, nfft)[:nw_n] / nw_n
    c22 = np.fft.irfft(np.conj(fw2) * fw2, nfft)[:nw_n] / nw_n
    c12 = np.fft.irfft(np.conj(fw1) * fw2, nfft)[:nw_n] / nw_n
    c21 = np.fft.irfft(np.conj(fw2) * fw1, nfft)[:nw_n] / nw_n
    jj = np.arange(1, m_fb)
    w_fb = 1.0 - jj / m_fb
    cross_fb = c12[0] + float(w_fb @ (c12[1:m_fb] + c21[1:m_fb]))
    s_fb = np.array([
        [c11[0] + 2.0 * float(w_fb @ c11[1:m_fb]), cross_fb],
        [cross_fb, c22[0] + 2.0 * float(w_fb @ c22[1:m_fb])]])
    cov_fb = nw_n * xtx_inv @ (s_fb / (1.0 - rho_t) ** 2) @ xtx_inv
    lnA_se_fb = math.sqrt(max(float(cov_fb[0, 0]), 0.0))
    b_se_fb = math.sqrt(max(float(cov_fb[1, 1]), 0.0))
    fb_cv = _fixed_b_cv(ln_d)
    half_b_fb = fb_cv * b_se_fb
    half_a_fb = fb_cv * lnA_se_fb
    return {
        "b": float(b), "lnA": float(lnA), "sigma": sigma, "n": n,
        "b_se": b_se, "lnA_se": lnA_se,
        "b_ci95": (float(b) - half, float(b) + half),
        "b_se_nw": b_se_nw, "lnA_se_nw": lnA_se_nw,
        "b_ci95_nw": (float(b) - half_nw, float(b) + half_nw),
        "nw_bandwidth": nw_lag, "nw_rho": rho_t,
        "b_ci95_fb": (float(b) - half_b_fb, float(b) + half_b_fb),
        "lnA_ci95_fb": (float(lnA) - half_a_fb,
                        float(lnA) + half_a_fb),
        "fb_cv": fb_cv, "fb_b": 1.0,
    }


def trend_envelope(lnA, b, sigma, days_arr):
    """Power-law trend with +/-1 sigma envelope over days_arr."""
    ln_d = np.log(days_arr)
    trend = np.exp(lnA + b * ln_d)
    lo = np.exp(lnA - sigma + b * ln_d)
    hi = np.exp(lnA + sigma + b * ln_d)
    return trend, lo, hi


def roll_corr(a, b_, win=63, minp=8):
    """63-calendar-day rolling Pearson correlation (right-closed)."""
    return a.rolling(f"{win}D", min_periods=minp).corr(b_).dropna()


# The legs' retry is the shared with_retry in yahoo_prices
# (canonical there since 2026-11-01, when the dollar and
# US vs ex-US fetchers adopted it). It began as this module's
# private copy the same day: the deployed dashboard's
# power-law tab had failed with a bare "HTTP Error 429" — a
# rate limiter on the cloud host's shared egress IP — and no
# leg identity. Kept under the original private name: the
# call sites below and this module's tests use it unchanged.
_with_retry = with_retry


def _utcnow():
    """The current instant in UTC — the module's one clock seam.

    fetch_powerlaw stamps its fetch window from it and the
    in-progress test in _btc_display_date reads it, so a test
    can freeze both through a single monkeypatch.
    """
    return datetime.datetime.now(datetime.timezone.utc)


def _btc_display_date(last_stamp):
    """The date the dashboard shows for the last BTC bar.

    One policy (user directions 2026-11-01 and 2026-11-02): a
    completed bar carries its Eastern date — Yahoo stamps bars
    at 00:00 UTC, the prior Eastern evening (19:00 EST), so the
    Eastern conversion is the bar's own day. The in-progress
    bar — stamped with today's UTC date, still trading —
    instead carries today's Eastern date: its Eastern
    conversion reads yesterday all day, which mislabels a
    live price. Display only: the series index keeps Yahoo's
    bar dates (see fetch_powerlaw's dating note), so no
    value, fit or correlation moves with the label.
    """
    now = _utcnow()
    if last_stamp.date() == now.date():
        return now.astimezone(ET).date().isoformat()
    return utc_index_to_et_dates(
        pd.DatetimeIndex([last_stamp]))[0].date().isoformat()


def fetch_powerlaw(progress=None, stage=None):
    """Fetch all six sources and compute the chart payload + stats.

    Returns a dict consumed by chart_png() and by the page's stats
    chips. Each leg retries transient HTTP errors with backoff;
    a final failure raises a RuntimeError naming the leg — the
    caller isolates it.

    Progress (2026-11-16, the dashboard's progress standard):
    ``progress(done, total, detail)`` reports each source leg
    as it completes (the futures board's _fetch_all idiom),
    then re-baselines to the fit phase's three opaque units —
    the price fit, the ratio fit, and a third the caller
    completes (the chart draw). ``stage(label)`` names the
    opaque steps as they start: the price fit's classical +
    Newey-West pass, its fixed-b step (via fit_power_law_full's
    own hook), and the ratio fit. Both hooks default to None —
    today's behaviour, byte-identical outputs.
    """
    p2 = int(_utcnow().timestamp())
    # BTC trades 24/7: keep Yahoo's in-progress UTC-day bar
    # (the shared equity-session truncation would end the
    # series a day early, and two days early on weekends).
    legs = [("yahoo BTC-USD",
             lambda: yahoo_daily("BTC-USD", p2=p2,
                                 include_partial=True)),
            ("yahoo GLD", lambda: yahoo_daily("GLD", p2=p2)),
            ("yahoo SPY", lambda: yahoo_daily("SPY", p2=p2)),
            ("yahoo GC=F", lambda: yahoo_daily("GC=F", p2=p2)),
            ("blockchain.info market-price", blockchain_btc),
            ("blockchain.info market-cap", blockchain_marketcap)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(_with_retry, fn, leg)
                for leg, fn in legs]
        if progress is not None:
            progress(0, len(jobs), "")
        results = [None] * len(jobs)
        by_fut = {f: i for i, f in enumerate(jobs)}
        done = 0
        for fut in as_completed(jobs):
            i = by_fut[fut]
            results[i] = fut.result()
            done += 1
            if progress is not None:
                progress(done, len(jobs), legs[i][0])
        btc, gld, spy, gold_oz, bdf, bmc = results

    # Dating (user directions 2026-11-01 and 2026-11-02:
    # everything in the app in Eastern): Yahoo stamps BTC-USD
    # bars at 00:00 UTC, and the Nov 2 UTC bar opens Nov 1,
    # 19:00 EST. DISPLAYED dates are Eastern, assigned at the
    # surface by _btc_display_date (stats["btc_date"], the
    # figure annotation): a completed bar carries its Eastern
    # conversion, while the in-progress bar carries today's
    # Eastern date — its conversion would read yesterday all
    # day (user decision 2026-11-02). The series index itself
    # deliberately keeps Yahoo's bar dates: a bar's
    # close-to-close return ends on its stamp's Eastern day
    # (19:00 ET), which is exactly what pairs it with the right
    # GLD/SPY session in the correlations below — re-dating the
    # index by a day moved corr_gold from +0.50 to +0.07 in
    # testing, a values change, not a dating change. The
    # blockchain.info series likewise keep their sampled dates:
    # the fits and the ratio's "latest" date hang off those,
    # exactly as served.

    # --- extended fit on 2010+ history ---
    # The fit phase re-baselines the bar to its three opaque
    # units (price fit, ratio fit, the caller's chart draw);
    # three is below the bar's ETA floor, so the phase reads
    # elapsed-only rather than carrying a fabricated to-go.
    if progress is not None:
        progress(0, 3, "")
    if stage is not None:
        stage("Fitting the price power law — classical "
              "+ Newey–West…")
    t = days_since_genesis(bdf.index)
    fit = fit_power_law_full(
        t, bdf["p"].values,
        stage=(lambda label: stage(
            "Fitting the price power law — " + label + "…"))
        if stage is not None else None)
    b, lnA, sigma = fit["b"], fit["lnA"], fit["sigma"]
    n_fit = fit["n"]
    if progress is not None:
        progress(1, 3, f"price fit — slope {b:.2f}, fixed-b CI "
                       f"[{fit['b_ci95_fb'][0]:.2f}, "
                       f"{fit['b_ci95_fb'][1]:.2f}]")
    end = datetime.date(2030, 12, 31)
    days = pd.date_range("2010-01-01", end, freq="D")
    trend, lo, hi = trend_envelope(
        lnA, b, sigma, days_since_genesis(days))
    p2029 = float(np.exp(
        lnA + b * math.log((datetime.date(2029, 12, 31) - GENESIS).days)))
    early = bdf[bdf.index < pd.Timestamp("2014-09-17")]

    # --- correlations ---
    rets = pd.DataFrame({"BTC": np.log(btc).diff(),
                         "GOLD": np.log(gld).diff(),
                         "SPX": np.log(spy).diff()}).dropna()
    c_gold = rets["BTC"].rolling(63).corr(rets["GOLD"]).dropna()
    c_spx = rets["BTC"].rolling(63).corr(rets["SPX"]).dropna()
    bdf = bdf.copy()
    bdf["lr"] = np.log(bdf["p"]).diff()
    slr = pd.DataFrame({"BTC": bdf["lr"]})
    slr["GOLD"] = np.log(gld).diff().reindex(
        slr.index, method="ffill").values
    slr["SPX"] = np.log(spy).diff().reindex(
        slr.index, method="ffill").values
    slr = slr.dropna()
    pre = slr[slr.index < pd.Timestamp("2014-09-17")]
    eg = roll_corr(pre["BTC"], pre["GOLD"])
    es = roll_corr(pre["BTC"], pre["SPX"])

    # --- BTC-to-gold market-cap ratio. The series is the
    # sampled cap exactly as served and is dated by its last
    # sample (user direction 2026-11-01: keep the date honest
    # — no synthetic extension wearing a measured date). ---
    cap = bmc["mc"]
    g_on_b = gold_oz.reindex(cap.index, method="ffill")
    gold_days = days_since_genesis(cap.index)
    gold_mc = g_on_b * (216265.0 + (gold_days - _ANCHOR_DAYS)
                        / 365.25 * 3738.0) * OZT_PER_TONNE
    ratio = (cap / gold_mc).replace(
        [np.inf, -np.inf], np.nan).dropna()
    ratio = ratio[ratio > 0]
    ratio_now = float(ratio.iloc[-1])
    rt = days_since_genesis(ratio.index)
    if stage is not None:
        stage("Fitting the BTC ÷ gold-cap ratio…")
    fit_r = fit_power_law_full(rt, ratio.values)
    b_r, lnA_r, sigma_r = fit_r["b"], fit_r["lnA"], fit_r["sigma"]
    if progress is not None:
        progress(2, 3, f"ratio fit — slope {b_r:.2f}")
    r_days = pd.date_range("2013-01-01", end, freq="D")
    r_trend, r_lo, r_hi = trend_envelope(
        lnA_r, b_r, sigma_r, days_since_genesis(r_days))
    flip_date = None
    if b_r > 0:
        t_flip = math.exp(-lnA_r / b_r)
        flip_date = GENESIS + datetime.timedelta(days=t_flip)

    last = c_gold.index[-1]
    floor = math.floor(
        min(c_gold.min(), c_spx.min(), eg.min(), es.min()) * 20) / 20
    return {
        "asof": asof_stamp(),
        "btc": btc, "early": early, "days": days, "trend": trend,
        "lo": lo, "hi": hi, "ratio": ratio, "r_days": r_days,
        "r_trend": r_trend, "r_lo": r_lo, "r_hi": r_hi,
        "c_gold": c_gold, "c_spx": c_spx, "eg": eg, "es": es,
        "floor": floor, "last": last,
        "stats": {
            "slope": float(b), "sigma": float(sigma), "n_fit": n_fit,
            "slope_se": fit["b_se"],
            "slope_ci_lo": fit["b_ci95"][0],
            "slope_ci_hi": fit["b_ci95"][1],
            "slope_se_nw": fit["b_se_nw"],
            "slope_ci_nw_lo": fit["b_ci95_nw"][0],
            "slope_ci_nw_hi": fit["b_ci95_nw"][1],
            "slope_ci_fb_lo": fit["b_ci95_fb"][0],
            "slope_ci_fb_hi": fit["b_ci95_fb"][1],
            "end2029": p2029,
            "btc_last": float(btc.iloc[-1]),
            "btc_date": _btc_display_date(btc.index[-1]),
            "ratio_pct": ratio_now * 100,
            "ratio_date": ratio.index[-1].date().isoformat(),
            "ratio_slope": float(b_r),
            "ratio_slope_se": fit_r["b_se"],
            "ratio_slope_se_nw": fit_r["b_se_nw"],
            "ratio_slope_ci_fb_lo": fit_r["b_ci95_fb"][0],
            "ratio_slope_ci_fb_hi": fit_r["b_ci95_fb"][1],
            "flippening": (flip_date.isoformat()
                           if flip_date else None),
            "corr_gold": float(c_gold.iloc[-1]),
            "corr_spx": float(c_spx.iloc[-1]),
            "data_end": last.date().isoformat(),
        },
    }


def chart_png(data):
    """Render the three-panel power-law figure to PNG bytes.

    Layout (user request 2026-10-23, from the brief era): top-left
    the log-scale price with its power law and ±1σ band, top-right
    the BTC÷gold market-cap ratio with its own fit and the
    flippening line, bottom the two 63-day correlations full width.

    Rendered transparent with a dark-page palette (user
    direction 2026-11-04): snow ink, brightened series
    hues, dark glass label boxes — the figure blends into
    the app's themed page.
    """
    btc, early = data["btc"], data["early"]
    days, trend, lo, hi = (data["days"], data["trend"],
                           data["lo"], data["hi"])
    ratio, r_days = data["ratio"], data["r_days"]
    r_trend, r_lo, r_hi = data["r_trend"], data["r_lo"], data["r_hi"]
    c_gold, c_spx, eg, es = (data["c_gold"], data["c_spx"],
                             data["eg"], data["es"])
    last, floor, st = data["last"], data["floor"], data["stats"]
    b, b_r = st["slope"], st["ratio_slope"]
    # The legend quotes each slope with its uncertainty when
    # the payload carries it (fetch_powerlaw always does;
    # older/hand-built payloads render the bare slope as
    # before). The fixed-b 95% CI headlines on its own line
    # when present — it is the interval whose coverage holds
    # in the tail (see fit_power_law_full) — with the
    # Newey-West SE after it as the secondary figure. Without
    # it, the NW SE is the quoted figure: the residuals are
    # strongly persistent (lag-1 autocorrelation 0.99), under
    # which the classical SE understates the slope's sampling
    # sd ~15x.
    # When the payload carries the fixed-b interval it
    # headlines the label on its own line, with the NW SE
    # after it as the secondary figure; the fallbacks below
    # are unchanged.
    slope_lab = f"power law (slope {b:.2f})"
    if st.get("slope_ci_fb_lo") is not None:
        slope_lab = (f"power law (slope {b:.2f})\n"
                     f"fixed-b 95% CI "
                     f"[{st['slope_ci_fb_lo']:.2f}, "
                     f"{st['slope_ci_fb_hi']:.2f}]")
        if st.get("slope_se_nw"):
            slope_lab += f"; NW SE \u00b1{st['slope_se_nw']:.2f}"
    elif st.get("slope_se_nw"):
        slope_lab = (f"power law (slope {b:.2f} "
                     f"\u00b1 {st['slope_se_nw']:.2f} NW SE)")
    elif st.get("slope_se"):
        slope_lab = (f"power law (slope {b:.2f} "
                     f"\u00b1 {st['slope_se']:.2f})")
    ratio_lab = f"ratio power law (slope {b_r:.2f})"
    if st.get("ratio_slope_ci_fb_lo") is not None:
        ratio_lab = (f"ratio power law (slope {b_r:.2f})\n"
                     f"fixed-b 95% CI "
                     f"[{st['ratio_slope_ci_fb_lo']:.2f}, "
                     f"{st['ratio_slope_ci_fb_hi']:.2f}]")
        if st.get("ratio_slope_se_nw"):
            ratio_lab += (f"; NW SE "
                          f"\u00b1{st['ratio_slope_se_nw']:.2f}")
    elif st.get("ratio_slope_se_nw"):
        ratio_lab = (f"ratio power law (slope {b_r:.2f} "
                     f"\u00b1 {st['ratio_slope_se_nw']:.2f} NW SE)")
    elif st.get("ratio_slope_se"):
        ratio_lab = (f"ratio power law (slope {b_r:.2f} "
                     f"\u00b1 {st['ratio_slope_se']:.2f})")

    fig = plt.figure(figsize=(11, 7.4))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.6, 1],
                           hspace=0.08, wspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1r = fig.add_subplot(gs[0, 1], sharex=ax1)
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)

    # --- Top-left: BTC price power law only ---
    ax1.fill_between(days, lo, hi, color="#e4dcc9", alpha=0.22, zorder=1)
    ax1.semilogy(days, trend, color="#7FA8D9", lw=1.4, alpha=0.85,
                 zorder=3, label=slope_lab)
    if len(early):
        ax1.semilogy(early.index, early["p"], ".", color="#B4A98F",
                     ms=3, alpha=0.6,
                     label="BTC price (2010-2014, sparse)", zorder=2)
    ax1.semilogy(btc.index, btc, color="#F5F2EA", lw=2.2, alpha=1.0,
                 zorder=5, label="BTC price")
    _lp_x, _lp_y = btc.index[-1], float(btc.iloc[-1])
    ax1.plot([_lp_x], [_lp_y], marker="o", ms=6, color="#E08A70",
             markeredgecolor="white", markeredgewidth=1.0, zorder=7)
    # The annotation's date TEXT is stats["btc_date"] (the
    # displayed Eastern date, _btc_display_date) — only the
    # text rides on it: the point's coordinate and the "today"
    # line below both sit on the series' own last index value,
    # which keeps Yahoo's bar date; at this axis span the day
    # is sub-pixel.
    ax1.annotate(f"${_lp_y:,.0f}\n{st['btc_date']}",
                 xy=(_lp_x, _lp_y), xytext=(-8, -30),
                 textcoords="offset points", ha="right", va="top",
                 fontsize=8.5, color="#E08A70", fontweight="bold",
                 bbox={"boxstyle": "round,pad=0.3", "facecolor": "#0B1118",
                           "alpha": 0.9, "edgecolor": "#E08A70", "lw": 0.7},
                 arrowprops={"arrowstyle": "-", "color": "#E08A70", "lw": 0.9})
    ax1.axhline(300000, color="#4CAF6E", ls=":", lw=1.3)
    ax1.text(datetime.datetime(2030, 12, 31), 345000, "$300,000 ",  # noqa: DTZ001 - chart axis coordinate; module dates are naive by construction
             color="#4CAF6E", fontsize=9, ha="right")
    # The "today" line marks the annotated last point — the shared
    # `last` (the correlations' data end) sat days away from the
    # point it labelled.
    ax1.axvline(_lp_x, color="#E08A70", ls="--", lw=1.2)
    ax1.text(_lp_x + pd.Timedelta(days=8), 40000, "today",
             color="#E08A70", fontsize=9, va="top")
    ax1.set_ylabel("BTC-USD (log scale)", color="#7FA8D9")
    ax1.tick_params(axis="y", colors="#7FA8D9")
    ax1.set_title("BTC price — power law", fontsize=10,
                  color="#E8ECEC")
    ax1.legend(loc="lower left", fontsize=8,
               facecolor="#0B1118", framealpha=0.85,
               edgecolor="#3A4646", labelcolor="#E8ECEC")
    ax1.grid(True, which="major", alpha=0.25)
    ax1.text(0.03, 0.96,
             f"fit 2010\u2013{last.date().year}, n={st['n_fit']}\n"
             f"end-2029 \u2248 ${st['end2029']:,.0f}",
             transform=ax1.transAxes, fontsize=8, va="top", ha="left",
             color="#E8ECEC",
             bbox={"boxstyle": "round", "facecolor": "#0B1118", "alpha": 0.85})
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_xlim(datetime.datetime(2010, 1, 1),  # noqa: DTZ001 - chart axis coordinate; module dates are naive by construction
                 datetime.datetime(2030, 12, 31))  # noqa: DTZ001 - chart axis coordinate; module dates are naive by construction

    # --- Top-right: BTC/gold market-cap ratio power law only ---
    ax1r.fill_between(r_days, r_lo * 100, r_hi * 100, color="#e8d9a8",
                      alpha=0.22, zorder=1)
    ax1r.axhline(100, color="#4CAF6E", ls=":", lw=1.3, zorder=2)
    ax1r.text(datetime.datetime(2010, 6, 1), 118,  # noqa: DTZ001 - chart axis coordinate; module dates are naive by construction
              "flippening \u2014 100%\nmodel \u2248 "
              f"{st['flippening'] or 'n/a'}",
              color="#4CAF6E", fontsize=8.5, va="bottom")
    ax1r.set_ylim(1e-3, 400)
    ax1r.semilogy(r_days, r_trend * 100, color="#C9A227", lw=1.8,
                  ls="--", zorder=3, label=ratio_lab)
    ax1r.semilogy(ratio.index, ratio * 100, color="#E3B341", lw=1.4,
                  alpha=0.9, zorder=4,
                  label="BTC/gold market cap "
                        f"(latest {st['ratio_pct']:.2f}%)")
    _lr_x, _lr_y = ratio.index[-1], float(ratio.iloc[-1] * 100)
    ax1r.plot([_lr_x], [_lr_y], marker="o", ms=6, color="#E08A70",
              markeredgecolor="white", markeredgewidth=1.0, zorder=6)
    ax1r.annotate(f"{_lr_y:.2f}%\n{_lr_x.date().isoformat()}",
                  xy=(_lr_x, _lr_y), xytext=(-8, 14),
                  textcoords="offset points", ha="right", va="bottom",
                  fontsize=8.5, color="#E08A70", fontweight="bold",
                  bbox={"boxstyle": "round,pad=0.3", "facecolor": "#0B1118",
                            "alpha": 0.9, "edgecolor": "#E08A70", "lw": 0.7},
                  arrowprops={"arrowstyle": "-", "color": "#E08A70", "lw": 0.9})
    # The marker line sits on the ratio's own last (sampled)
    # date and says "latest" — the shared `last` line used
    # before sat days away from the point and called it
    # "today", which the sampled date is not.
    ax1r.axvline(_lr_x, color="#E08A70", ls="--", lw=1.2)
    ax1r.text(_lr_x + pd.Timedelta(days=8), 20, "latest",
              color="#E08A70", fontsize=9, va="top")
    # Source note, bottom-right in short lines that stay clear
    # of the legend (which ends near x=0.58 of the panel).
    ax1r.text(0.98, 0.02,
              "BTC cap: blockchain.info samples,\n"
              f"latest {st['ratio_date']} (~4\u20136d sampling)\n"
              "Gold: GC=F \u00d7 WGC above-ground stock",
              transform=ax1r.transAxes, fontsize=7.5, va="bottom",
              ha="right", color="#9AA5A5")
    ax1r.set_ylabel("BTC \u00f7 gold market cap (%)", color="#E3B341")
    ax1r.tick_params(axis="y", colors="#E3B341")
    ax1r.set_title("BTC / gold market cap — power law", fontsize=10,
                   color="#E8ECEC")
    ax1r.legend(loc="lower left", fontsize=8,
                facecolor="#0B1118", framealpha=0.85,
                edgecolor="#3A4646", labelcolor="#E8ECEC")
    ax1r.grid(True, which="major", alpha=0.25)
    plt.setp(ax1r.get_xticklabels(), visible=False)

    fig.suptitle("Bitcoin power law and 63-day BTC\u00d7gold / "
                 "BTC\u00d7SPX correlations", fontsize=11, y=0.98,
                 color="#E8ECEC")

    # --- Bottom row: correlations, spanning both columns ---
    ax2.plot(eg.index, eg, color="#E3B341", lw=1.0, alpha=0.65)
    ax2.plot(c_gold.index, c_gold, color="#E3B341", lw=1.5,
             label=f"BTC \u00d7 gold (latest {st['corr_gold']:+.2f})")
    ax2.plot(es.index, es, color="#6FA8D8", lw=1.0, alpha=0.65)
    ax2.plot(c_spx.index, c_spx, color="#6FA8D8", lw=1.5,
             label=f"BTC \u00d7 SPX (latest {st['corr_spx']:+.2f})")
    ax2.set_ylim(floor, 1.2)
    ax2.axhline(0, color="#E8ECEC", lw=0.8, alpha=0.5)
    ax2.set_ylabel("63-day rolling correlation", color="#E8ECEC")
    ax2.tick_params(colors="#E8ECEC")
    ax2.legend(loc="upper left", fontsize=9,
               facecolor="#0B1118", framealpha=0.85,
               edgecolor="#3A4646", labelcolor="#E8ECEC")
    ax2.text(0.02, 0.06, "pre-2014: sparse data, thinner lines",
             transform=ax2.transAxes, fontsize=8, va="bottom",
             color="#9AA5A5")
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(datetime.datetime(2010, 1, 1),  # noqa: DTZ001 - chart axis coordinate; module dates are naive by construction
                 datetime.datetime(2030, 12, 31))  # noqa: DTZ001 - chart axis coordinate; module dates are naive by construction
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # Geometry is pinned by the explicit subplots_adjust
    # below, never tight_layout: the figure's twin axes are
    # incompatible with tight_layout, which bailed out to
    # the default subplot params and emitted a UserWarning
    # on every render (the app-log warning the user flagged
    # 2026-11-06; removal verified byte-identical on the
    # same data — the call was a measured no-op). The
    # starting canvas carried ~120px of white on the right
    # and ~65px at the bottom of the 1210x814 canvas
    # (measured); pull the outer margins in (user request
    # 2026-11-01).
    # Ink margins measured after the change: left 81->17,
    # right 120->11, bottom 67->34; nothing clipped.
    fig.subplots_adjust(left=0.072, right=0.99, bottom=0.07,
                        top=0.93)
    for _ax in (ax1, ax1r, ax2):
        for _sp in _ax.spines.values():
            _sp.set_color("#7A8585")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, transparent=True)
    plt.close(fig)
    return buf.getvalue()
