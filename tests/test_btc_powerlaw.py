"""Offline tests for the Bitcoin power-law math (no network)."""
import datetime

import numpy as np
import pandas as pd
import pytest

from toolbox.DataZoo.market import btc_powerlaw as bp


@pytest.fixture
def stub_fb_cv(monkeypatch):
    """Replace the fixed-b CV simulation with a constant 6.0.

    The CV is a ~13 s simulation per design, pinned on its
    own by test_fixed_b_cv_* (known value, determinism, cache
    discipline). The fit/fetch tests using this fixture assert
    classical/NW values, payload shape, or that the fb
    interval brackets its slope — none of which depends on
    the CV's value, only on it being a plausible ~6. Before
    this fixture, ten designs' simulations were ~125 s of
    this file's ~150 s runtime.
    """
    monkeypatch.setattr(bp, "_fixed_b_cv", lambda ln_d: 6.0)


def test_fit_power_law_recovers_known_law():
    days = np.geomspace(100, 20000, 400)
    values = np.exp(1.5 + 2.5 * np.log(days))
    b, lnA, sigma = bp.fit_power_law(days, values)
    assert abs(b - 2.5) < 1e-8
    assert abs(lnA - 1.5) < 1e-6
    assert sigma < 1e-8


def test_fit_power_law_full_hand_computed_se(stub_fb_cv):
    # x = ln(days) = [0, 1, 2, 3], ln(values) = [0, 1, 1, 3].
    # By hand: xbar = 1.5, Sxx = 5, Sxy = 4.5 -> b = 0.9,
    # lnA = -0.1; residuals (0.1, 0.2, -0.7, 0.4), RSS = 0.70,
    # s^2 = RSS/(n-2) = 0.35. So b_se = sqrt(0.35/5) = sqrt(0.07),
    # lnA_se = sqrt(0.35*(1/4 + 1.5^2/5)) = sqrt(0.245), the
    # ddof=0 sigma = sqrt(0.70/4) = sqrt(0.175), and the 95% CI
    # uses t(0.975, 2) = 4.302652729749275.
    days = np.exp([0.0, 1.0, 2.0, 3.0])
    values = np.exp([0.0, 1.0, 1.0, 3.0])
    f = bp.fit_power_law_full(days, values)
    assert f["n"] == 4
    assert abs(f["b"] - 0.9) < 1e-9
    assert abs(f["lnA"] + 0.1) < 1e-9
    assert abs(f["sigma"] - np.sqrt(0.175)) < 1e-9
    assert abs(f["b_se"] - np.sqrt(0.07)) < 1e-9
    assert abs(f["lnA_se"] - np.sqrt(0.245)) < 1e-9
    half = 4.302652729749275 * np.sqrt(0.07)
    assert abs(f["b_ci95"][0] - (0.9 - half)) < 1e-9
    assert abs(f["b_ci95"][1] - (0.9 + half)) < 1e-9


def test_fit_power_law_full_extends_tuple_api(stub_fb_cv):
    # The dict's point estimates are the tuple API's values,
    # bit for bit (consumer-shape pin: nothing existing moves).
    rng = np.random.default_rng(3)
    days = np.geomspace(200, 9000, 300)
    values = np.exp(-20.0 + 4.2 * np.log(days)
                    + rng.normal(0, 0.5, len(days)))
    b, lnA, sigma = bp.fit_power_law(days, values)
    f = bp.fit_power_law_full(days, values)
    assert f["b"] == float(b) and f["lnA"] == float(lnA)
    assert f["sigma"] == float(sigma) and f["n"] == 300
    assert f["b_se"] > 0 and f["lnA_se"] > 0
    assert f["b_ci95"][0] < f["b"] < f["b_ci95"][1]
    import pytest

    with pytest.raises(ValueError, match=">= 3 points"):
        bp.fit_power_law_full(days[:2], values[:2])


def test_fit_power_law_full_ci_coverage_seeded(stub_fb_cv):
    # Reduced-rep coverage pin (seeded, deterministic): the
    # sharp convention discrimination lives in the 20,000-rep
    # study recorded in fit_power_law_full's docstring; this
    # pins the wiring at CI scale. n = 40 is where the dof /
    # critical-value choice bites: the ddof=0 + normal variant
    # covers at ~94% there, the shipped n-2 + t variant at 95%.
    # Band [0.93, 0.97] is ~4 binomial sd at 2,000 reps; the
    # SE-vs-empirical-sd ratio (asserted within 10%, a ~6 sd
    # check) catches scale errors the band cannot.
    rng = np.random.default_rng(2026)
    days = np.geomspace(300, 6000, 40)
    x = np.log(days)
    reps, hits, ses, ests = 2000, 0, [], []
    for _ in range(reps):
        values = np.exp(-37.0 + 5.6 * x + rng.normal(0, 0.69, 40))
        f = bp.fit_power_law_full(days, values)
        hits += f["b_ci95"][0] <= 5.6 <= f["b_ci95"][1]
        ses.append(f["b_se"])
        ests.append(f["b"])
    cov = hits / reps
    assert 0.93 <= cov <= 0.97
    ratio = float(np.std(ests, ddof=1) / np.mean(ses))
    assert 0.9 < ratio < 1.1


def _nw_independent(days, values):
    """fit_power_law_full's Newey-West block, recomputed from the
    definitions with plain loops (no shared code): AR(1) score
    filter, Bartlett LRV of the whitened scores with the NW94
    plug-in bandwidth, Kendall-corrected recolor, sandwich."""
    import math

    n = len(days)
    x = [math.log(float(dv)) for dv in days]
    y = [math.log(float(vv)) for vv in values]
    xm = math.fsum(x) / n
    ym = math.fsum(y) / n
    sxx = math.fsum((v - xm) ** 2 for v in x)
    b = math.fsum((x[i] - xm) * (y[i] - ym) for i in range(n)) / sxx
    lnA = ym - b * xm
    e = [y[i] - lnA - b * x[i] for i in range(n)]
    i00 = 1.0 / n + xm * xm / sxx
    i01 = -xm / sxx
    i11 = 1.0 / sxx
    rho = max(-0.999, min(0.999,
                          math.fsum(e[i] * e[i + 1] for i in range(n - 1))
                          / math.fsum(e[i] ** 2 for i in range(n - 1))))
    u = [(e[i] - rho * e[i - 1],
          x[i] * e[i] - rho * x[i - 1] * e[i - 1])
         for i in range(1, n)]
    nn = n - 1
    v = [i01 * a + i11 * bb for a, bb in u]
    vv = math.fsum(t * t for t in v)
    rw = (math.fsum(v[i] * v[i + 1] for i in range(len(v) - 1)) / vv
          if vv > 0 else 0.0)
    rw = max(-0.999, min(0.999, rw))
    alpha1 = (2.0 * rw / (1.0 - rw ** 2)) ** 2
    lag = int(min(25, max(0, math.floor(
        1.1447 * (alpha1 * nn) ** (1.0 / 3.0)))))
    S = [[0.0, 0.0], [0.0, 0.0]]
    for j in range(lag + 1):
        wgt = 1.0 if j == 0 else 1.0 - j / (lag + 1.0)
        for a_ in range(2):
            for b_ in range(2):
                gj = math.fsum(u[t][a_] * u[t + j][b_]
                               for t in range(nn - j)) / nn
                S[a_][b_] += wgt * (gj if j == 0 else gj + math.fsum(
                    u[t][b_] * u[t + j][a_] for t in range(nn - j)) / nn)
    rho_t = max(-0.9995, min(0.9995, rho + (1.0 + 3.0 * rho) / n))
    fac = 1.0 / (1.0 - rho_t) ** 2
    var_b = nn * fac * (i01 ** 2 * S[0][0] + 2 * i01 * i11 * S[0][1]
                        + i11 ** 2 * S[1][1])
    var_a = nn * fac * (i00 ** 2 * S[0][0] + 2 * i00 * i01 * S[0][1]
                        + i01 ** 2 * S[1][1])
    return math.sqrt(var_b), math.sqrt(var_a), lag, rho_t


def test_fit_power_law_full_nw_hand_computed(stub_fb_cv):
    # Same hand-computed dataset as the classical-SE test:
    # x = [0, 1, 2, 3], residuals (0.1, 0.2, -0.7, 0.4). By hand
    # the residuals' lag-1 OLS coefficient is
    # (0.02 - 0.14 - 0.28) / (0.01 + 0.04 + 0.49) = -0.40/0.54
    # = -20/27, and the Kendall-corrected recolor coefficient
    # is -20/27 + (1 - 60/27)/4 = -113/108 < -0.9995, so the
    # clip binds and nw_rho is exactly -0.9995. The SEs
    # themselves are pinned against the independent loop
    # implementation above (agreement to 1e-12).
    days = np.exp([0.0, 1.0, 2.0, 3.0])
    values = np.exp([0.0, 1.0, 1.0, 3.0])
    f = bp.fit_power_law_full(days, values)
    b_nw, a_nw, lag, rho_t = _nw_independent(days, values)
    assert f["nw_rho"] == -0.9995 and rho_t == -0.9995
    assert f["nw_bandwidth"] == lag
    assert abs(f["b_se_nw"] - b_nw) < 1e-12
    assert abs(f["lnA_se_nw"] - a_nw) < 1e-12
    assert f["b_ci95_nw"][0] < f["b"] < f["b_ci95_nw"][1]


def test_fit_power_law_full_nw_iid_matches_classical(stub_fb_cv):
    # iid control (seeded): with no persistence to correct, the
    # NW SE tracks the classical one — the prewhitening must
    # not wreck the case that already worked. Across seeds the
    # ratio concentrates near 1; the band is ~3 sd of it.
    rng = np.random.default_rng(11)
    days = np.geomspace(200, 9000, 400)
    x = np.log(days)
    ratios = []
    for _ in range(25):
        values = np.exp(-20.0 + 4.2 * x + rng.normal(0, 0.5, len(days)))
        f = bp.fit_power_law_full(days, values)
        ratios.append(f["b_se_nw"] / f["b_se"])
    assert 0.8 < float(np.mean(ratios)) < 1.25


def test_fit_power_law_full_nw_ar1_coverage_seeded(stub_fb_cv):
    # Seeded AR(1) coverage at the board's own scale (n = 1484,
    # rho = 0.99, sigma = 0.69 — the live fit's persistence).
    # The 20,000-rep study in fit_power_law_full's docstring
    # measures the shipped interval at ~92% here while the
    # classical interval covers ~11%; 300 reps keeps CI time
    # ~2 s, and the bands are ~4 binomial sd around those
    # rates. The mean-SE / empirical-sd ratio pins the scale.
    rng = np.random.default_rng(4242)
    n = 1484
    days = np.geomspace(550, 6400, n)
    x = np.log(days)
    rho, sigma = 0.99, 0.69
    reps, hits_nw, hits_cl = 300, 0, 0
    ses, ests = [], []
    for _ in range(reps):
        u = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), n)
        e = np.empty(n)
        e[0] = rng.normal(0, sigma)
        for t in range(1, n):
            e[t] = rho * e[t - 1] + u[t]
        f = bp.fit_power_law_full(days, np.exp(-37.3 + 5.6 * x + e))
        hits_nw += f["b_ci95_nw"][0] <= 5.6 <= f["b_ci95_nw"][1]
        hits_cl += f["b_ci95"][0] <= 5.6 <= f["b_ci95"][1]
        ses.append(f["b_se_nw"])
        ests.append(f["b"])
    assert 0.84 <= hits_nw / reps <= 0.99
    assert hits_cl / reps < 0.30
    ratio = float(np.mean(ses) / np.std(ests, ddof=1))
    assert 0.8 < ratio < 1.9


def test_fixed_b_cv_span_design_matches_note():
    # Known-value pin against "Fixed-b Critical Values on the
    # Power-Law Design": its span design (ln-days linear in r
    # over [592, 6524], the BTC design's span) simulates to
    # 6.005 at m = 4000 / 100,000 reps. A dense equally spaced
    # sample's empirical quantiles trace that continuum
    # design; the shipped settings (m = 2000, 60,000 reps,
    # fixed seed) read 5.996 in pre-ship benchmarking. The
    # 0.06 tolerance is ~2.4x the quantile MC se at 60,000
    # draws — a wiring/scale pin, with the sharp value in the
    # note. KV's location-model table (4.78) sits far outside
    # the band: the design projection is doing real work.
    days = np.linspace(592.0, 6524.0, 20001)
    cv = bp._fixed_b_cv(np.log(days))
    assert abs(cv - 6.005) < 0.06


def test_fixed_b_cv_deterministic():
    # Fixed seed + fixed settings: the simulated draws are
    # bit-identical across calls, so the cached CV is a
    # constant of the design, not a draw of the day.
    x = np.log(np.geomspace(100, 5000, 60))
    t1 = bp._fb_t_draws(x, m=500, reps=4000, seed=bp._FB_SEED)
    t2 = bp._fb_t_draws(x, m=500, reps=4000, seed=bp._FB_SEED)
    assert np.array_equal(t1, t2)
    assert bp._fixed_b_cv(x) == bp._fixed_b_cv(x)
    assert bp._fixed_b_cv(x) > 3.0


def test_fixed_b_cv_cache_tolerance_and_fallback(monkeypatch, tmp_path):
    # The cache discipline, at reduced simulation settings
    # (the value itself is pinned by the span-design test):
    # a stored entry serves a one-observation successor
    # design without re-simulating (tolerance match), and a
    # corrupt store falls back to simulation instead of
    # raising. The store path is redirected to tmp_path so
    # the test never touches the real ~/.cache store.
    monkeypatch.setattr(bp, "_FB_GRID", 200)
    monkeypatch.setattr(bp, "_FB_REPS", 2000)
    store = tmp_path / "fixed_b_cv.json"
    monkeypatch.setattr(bp, "_fb_store_path", lambda: store)
    monkeypatch.setattr(bp, "_FB_STORE", None)

    days = np.geomspace(200, 9000, 400)
    cv = bp._fixed_b_cv(np.log(days))
    assert cv > 1.0 and store.exists()

    real_draws = bp._fb_t_draws

    def _no_sim(*args, **kwargs):
        msg = "cache miss: the successor design re-simulated"
        raise AssertionError(msg)

    monkeypatch.setattr(bp, "_fb_t_draws", _no_sim)
    nxt = np.append(days, days[-1] * 1.001)
    assert bp._fixed_b_cv(np.log(nxt)) == cv

    monkeypatch.setattr(bp, "_fb_t_draws", real_draws)
    store.write_text("garbage{{{", encoding="utf-8")
    monkeypatch.setattr(bp, "_FB_STORE", None)
    other = np.log(np.geomspace(300, 6000, 200))
    assert bp._fixed_b_cv(other) > 1.0


def test_fixed_b_interval_contains_nw_interval():
    # Seeded persistent fixture (AR(1) rho = 0.99, n = 400):
    # the fixed bandwidth's SE runs ~0.8x the plug-in's, but
    # the ~6x critical value dominates the t value's 1.96x,
    # so the fb interval strictly contains the NW interval —
    # the width premium the note measures at the live cells
    # (2.10x / 2.45x), reproduced here at fixture scale.
    rng = np.random.default_rng(7)
    n = 400
    days = np.geomspace(200, 9000, n)
    x = np.log(days)
    rho, sigma = 0.99, 0.69
    u = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), n)
    e = np.empty(n)
    e[0] = rng.normal(0, sigma)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + u[t]
    f = bp.fit_power_law_full(days, np.exp(-20.0 + 4.2 * x + e))
    assert f["fb_b"] == 1.0 and f["fb_cv"] > 3.0
    assert f["b_ci95_fb"][0] < f["b_ci95_nw"][0]
    assert f["b_ci95_fb"][1] > f["b_ci95_nw"][1]
    assert f["lnA_ci95_fb"][0] < f["lnA"] < f["lnA_ci95_fb"][1]


def test_fixed_b_interval_iid_coverage_smoke():
    # iid smoke (seeded, deterministic): the note's machinery
    # validation at rho = 0 covers 0.951-0.953 at n = 1484 /
    # 20,000 reps; at this smoke's n = 200 the fixed-b limit
    # is cruder and the seeded run covers 0.935. 200 reps on
    # one design (one cached CV simulation) pins the wiring;
    # the band is ~+/-2.6 binomial sd around the observed
    # rate. The sharp coverage claims live in the note.
    rng = np.random.default_rng(99)
    n = 200
    days = np.geomspace(300, 6000, n)
    x = np.log(days)
    reps, hits = 200, 0
    for _ in range(reps):
        values = np.exp(-37.0 + 5.6 * x + rng.normal(0, 0.69, n))
        f = bp.fit_power_law_full(days, values)
        hits += f["b_ci95_fb"][0] <= 5.6 <= f["b_ci95_fb"][1]
    assert 0.88 <= hits / reps <= 0.99


def test_days_since_genesis():
    idx = pd.DatetimeIndex(["2009-01-03", "2009-01-13", "2010-01-03"])
    out = bp.days_since_genesis(idx)
    assert list(out) == [0.0, 10.0, 365.0]


def test_trend_envelope_zero_sigma_collapses():
    days = np.array([100.0, 1000.0, 10000.0])
    trend, lo, hi = bp.trend_envelope(0.5, 2.0, 0.0, days)
    assert np.allclose(lo, trend) and np.allclose(hi, trend)
    assert np.allclose(trend, np.exp(0.5) * days ** 2)


def test_roll_corr_of_identical_series_is_one():
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    s = pd.Series(np.sin(np.arange(120) / 7.0), index=idx)
    out = bp.roll_corr(s, s.copy())
    assert len(out) > 0
    assert abs(out.iloc[-1] - 1.0) < 1e-9



def test_with_retry_recovers_from_transient_429():
    from urllib.error import HTTPError

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            url = "u"
            raise HTTPError(url, 429, "Too Many Requests",
                            None, None)
        return "ok"

    assert bp._with_retry(flaky, "leg-x") == "ok"
    assert calls["n"] == 3


def test_with_retry_names_the_leg_on_final_failure():
    import pytest

    def bad():
        msg = "boom"
        raise ValueError(msg)

    with pytest.raises(RuntimeError, match="leg-y: boom"):
        bp._with_retry(bad, "leg-y")


def test_with_retry_does_not_retry_permanent_http_errors():
    from urllib.error import HTTPError

    import pytest

    calls = {"n": 0}

    def gone():
        calls["n"] += 1
        url = "u"
        raise HTTPError(url, 404, "Not Found", None, None)

    with pytest.raises(RuntimeError, match="leg-z"):
        bp._with_retry(gone, "leg-z")
    assert calls["n"] == 1


def _powerlaw_fixture(monkeypatch):
    """Fake the six sources; the BTC series' last bar is stamped
    2026-11-02 (00:00 UTC). Returns the fake BTC series."""
    btc_idx = pd.date_range("2026-05-01", "2026-11-02", freq="D")
    wig = 1.0 + 0.05 * np.sin(np.arange(len(btc_idx)) / 3.0)
    btc = pd.Series(np.exp(np.linspace(9.0, 11.3, len(btc_idx))) * wig,
                    index=btc_idx)
    bus = pd.bdate_range("2010-01-04", "2026-10-30")
    wig_b = 1.0 + 0.03 * np.sin(np.arange(len(bus)) / 5.0)
    gld = pd.Series(np.exp(np.linspace(4.5, 5.6, len(bus))) * wig_b,
                    index=bus)
    spy = pd.Series(
        np.exp(np.linspace(5.0, 6.4, len(bus)))
        * (1.0 + 0.04 * np.cos(np.arange(len(bus)) / 4.0)), index=bus)
    gold_idx = pd.bdate_range("2013-01-01", "2026-10-30")
    gold = pd.Series(np.linspace(1200.0, 3900.0, len(gold_idx)),
                     index=gold_idx)

    def fake_yahoo(ticker, p1=0, p2=None, include_partial=False):
        return {"BTC-USD": btc, "GLD": gld, "SPY": spy,
                "GC=F": gold}[ticker]

    bdf_idx = pd.date_range("2010-07-17", "2026-10-31", freq="4D")
    bdf = pd.DataFrame(
        {"p": np.exp(np.linspace(0.0, 11.3, len(bdf_idx)))
         * (1.0 + 0.2 * np.sin(np.arange(len(bdf_idx)) / 2.0))},
        index=bdf_idx)
    cap_idx = pd.date_range("2013-01-01", "2026-10-27", freq="6D")
    bmc = pd.DataFrame(
        {"mc": np.exp(np.linspace(15.0, 21.0, len(cap_idx)))},
        index=cap_idx)
    monkeypatch.setattr(bp, "yahoo_daily", fake_yahoo)
    monkeypatch.setattr(bp, "blockchain_btc", lambda: bdf)
    monkeypatch.setattr(bp, "blockchain_marketcap", lambda: bmc)
    return btc


def _freeze_clock(monkeypatch, when_utc):
    monkeypatch.setattr(bp, "_utcnow", lambda: when_utc)


def test_fetch_powerlaw_dates_btc_in_eastern(monkeypatch, stub_fb_cv):
    # Yahoo stamps BTC-USD bars at 00:00 UTC: the fixture's last
    # bar is stamped 2026-11-02 but starts 2026-11-01 19:00 EST.
    # With the clock frozen three days past it, that bar is
    # COMPLETED, so the surfaced date is its Eastern conversion,
    # 2026-11-01 (user direction 2026-11-01: everything in the
    # app in Eastern). The conversion happens at the surface:
    # the internal index keeps Yahoo's bar dates, because a
    # bar's close-to-close return ends on its stamp's Eastern
    # day, which is what pairs it with the right equity session
    # in the correlations — the values must not move.
    btc = _powerlaw_fixture(monkeypatch)
    _freeze_clock(monkeypatch, datetime.datetime(
        2026, 11, 5, 12, 0, tzinfo=datetime.timezone.utc))

    out = bp.fetch_powerlaw()
    assert str(out["btc"].index[-1].date()) == "2026-11-02"
    assert out["stats"]["btc_date"] == "2026-11-01"
    assert out["stats"]["btc_last"] == float(btc.iloc[-1])
    assert out["asof"].endswith((" EST", " EDT"))


def test_fetch_powerlaw_in_progress_bar_gets_et_today(monkeypatch,
                                                      stub_fb_cv):
    # The same fixture, but the clock frozen inside the last
    # bar's UTC day (2026-11-02 19:45 UTC = 14:45 EST): the bar
    # is still trading, so it carries today's Eastern date,
    # 2026-11-02 (user decision 2026-11-02 — the Eastern
    # conversion would read 2026-11-01 all day). The relabel is
    # display-only: same index, same last price, and the fit —
    # computed off the untouched series — is bit-identical to
    # the completed-bar run.
    _powerlaw_fixture(monkeypatch)
    _freeze_clock(monkeypatch, datetime.datetime(
        2026, 11, 5, 12, 0, tzinfo=datetime.timezone.utc))
    done = bp.fetch_powerlaw()
    _freeze_clock(monkeypatch, datetime.datetime(
        2026, 11, 2, 19, 45, tzinfo=datetime.timezone.utc))
    live = bp.fetch_powerlaw()

    assert done["stats"]["btc_date"] == "2026-11-01"
    assert live["stats"]["btc_date"] == "2026-11-02"
    assert live["btc"].index.equals(done["btc"].index)
    assert live["stats"]["btc_last"] == done["stats"]["btc_last"]
    assert live["stats"]["slope"] == done["stats"]["slope"]


def test_fetch_powerlaw_surfaces_slope_uncertainty(monkeypatch,
                                                   stub_fb_cv):
    # The stats payload gains the slopes' standard errors and
    # the price slope's CI alongside the pre-existing keys
    # (which keep their values — the SE is the same fit's).
    _powerlaw_fixture(monkeypatch)
    _freeze_clock(monkeypatch, datetime.datetime(
        2026, 11, 5, 12, 0, tzinfo=datetime.timezone.utc))
    out = bp.fetch_powerlaw()
    st = out["stats"]
    assert st["slope_se"] > 0 and st["ratio_slope_se"] > 0
    assert st["slope_ci_lo"] < st["slope"] < st["slope_ci_hi"]
    # The figure renders through the new +-SE legend labels.
    png = bp.chart_png(out)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_fetch_powerlaw_surfaces_nw_uncertainty(monkeypatch, stub_fb_cv):
    # The payload gains the Newey-West SEs alongside the
    # classical ones, the NW interval brackets its slope, and
    # the figure renders through the label fallbacks below
    # the fixed-b headline: with the fb and NW keys dropped,
    # classical only, and then no SEs at all (hand-built
    # payloads keep rendering exactly as before).
    _powerlaw_fixture(monkeypatch)
    _freeze_clock(monkeypatch, datetime.datetime(
        2026, 11, 5, 12, 0, tzinfo=datetime.timezone.utc))
    out = bp.fetch_powerlaw()
    st = out["stats"]
    assert st["slope_se_nw"] > 0 and st["ratio_slope_se_nw"] > 0
    assert st["slope_ci_nw_lo"] < st["slope"] < st["slope_ci_nw_hi"]
    assert bp.chart_png(out)[:8] == b"\x89PNG\r\n\x1a\n"
    for drop in (["slope_se_nw", "slope_ci_nw_lo", "slope_ci_nw_hi",
                  "ratio_slope_se_nw", "slope_ci_fb_lo",
                  "slope_ci_fb_hi", "ratio_slope_ci_fb_lo",
                  "ratio_slope_ci_fb_hi"],
                 ["slope_se_nw", "slope_ci_nw_lo", "slope_ci_nw_hi",
                  "ratio_slope_se_nw", "slope_ci_fb_lo",
                  "slope_ci_fb_hi", "ratio_slope_ci_fb_lo",
                  "ratio_slope_ci_fb_hi", "slope_se", "slope_ci_lo",
                  "slope_ci_hi", "ratio_slope_se"]):
        legacy = dict(out)
        legacy["stats"] = {k: v for k, v in st.items() if k not in drop}
        assert bp.chart_png(legacy)[:8] == b"\x89PNG\r\n\x1a\n"


def test_fetch_powerlaw_surfaces_fb_uncertainty(monkeypatch, stub_fb_cv):
    # The payload gains the fixed-b intervals for both fits
    # alongside the NW figures; each brackets its own slope.
    # chart_png headlines the fixed-b interval in its legend
    # labels when the payload carries it — dropping the fb
    # keys changes the render — and falls back to the NW
    # labels when it doesn't.
    _powerlaw_fixture(monkeypatch)
    _freeze_clock(monkeypatch, datetime.datetime(
        2026, 11, 5, 12, 0, tzinfo=datetime.timezone.utc))
    out = bp.fetch_powerlaw()
    st = out["stats"]
    assert st["slope_ci_fb_lo"] < st["slope"] < st["slope_ci_fb_hi"]
    assert (st["ratio_slope_ci_fb_lo"] < st["ratio_slope"]
            < st["ratio_slope_ci_fb_hi"])
    png = bp.chart_png(out)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    legacy = dict(out)
    legacy["stats"] = {k: v for k, v in st.items()
                       if "ci_fb" not in k}
    legacy_png = bp.chart_png(legacy)
    assert legacy_png[:8] == png[:8]
    assert legacy_png != png


def test_chart_png_does_not_call_tight_layout():
    # Regression pin (2026-11-06): chart_png's twin axes are
    # incompatible with tight_layout — the call bailed to
    # the default subplot params and emitted a UserWarning
    # on every render (the app-log warning). The explicit
    # subplots_adjust pins the geometry; the call must not
    # come back.
    import inspect

    assert "tight_layout(" not in inspect.getsource(bp.chart_png)


def test_fit_power_law_full_stage_hook_names_fixed_b(stub_fb_cv):
    """The stage hook (2026-11-16) fires exactly once, naming
    the fixed-b step, and changes no value."""
    rng = np.random.default_rng(7)
    days = np.exp(np.linspace(6.0, 8.5, 80))
    values = np.exp(1.0 + 2.0 * np.log(days)
                    + rng.normal(0, 0.05, len(days)))
    seen = []
    f = bp.fit_power_law_full(days, values, stage=seen.append)
    assert len(seen) == 1
    assert "fixed-b" in seen[0]
    assert f == bp.fit_power_law_full(days, values)


def test_fetch_powerlaw_reports_legs_fits_and_stages(
        monkeypatch, stub_fb_cv):
    """Progress (2026-11-16): the six legs report as they
    complete; the bar then re-baselines to the fit phase's
    three units (price fit, ratio fit, the caller's chart
    draw), and the stage labels name the opaque steps in
    order. The payload is the plain fetch's."""
    _powerlaw_fixture(monkeypatch)
    _freeze_clock(monkeypatch, datetime.datetime(
        2026, 11, 5, 12, 0, tzinfo=datetime.timezone.utc))
    reports, stages = [], []
    out = bp.fetch_powerlaw(
        progress=lambda d, t, s: reports.append((d, t, s)),
        stage=stages.append)
    assert out["stats"] == bp.fetch_powerlaw()["stats"]
    assert reports[0] == (0, 6, "")
    legs = reports[1:7]
    assert [r[0] for r in legs] == [1, 2, 3, 4, 5, 6]
    assert {r[2] for r in legs} == {
        "yahoo BTC-USD", "yahoo GLD", "yahoo SPY", "yahoo GC=F",
        "blockchain.info market-price",
        "blockchain.info market-cap"}
    assert reports[7] == (0, 3, "")
    assert reports[8][:2] == (1, 3)
    assert reports[8][2].startswith("price fit — slope")
    assert reports[9][:2] == (2, 3)
    assert reports[9][2].startswith("ratio fit — slope")
    assert len(reports) == 10
    assert stages[0].startswith(
        "Fitting the price power law — classical")
    assert "fixed-b" in stages[1]
    assert stages[2].startswith("Fitting the BTC ÷ gold-cap ratio")
    assert len(stages) == 3
