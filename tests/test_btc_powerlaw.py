"""Offline tests for the Bitcoin power-law math (no network)."""
import datetime

import numpy as np
import pandas as pd

from toolbox.DataZoo.market import btc_powerlaw as bp


def test_fit_power_law_recovers_known_law():
    days = np.geomspace(100, 20000, 400)
    values = np.exp(1.5 + 2.5 * np.log(days))
    b, lnA, sigma = bp.fit_power_law(days, values)
    assert abs(b - 2.5) < 1e-8
    assert abs(lnA - 1.5) < 1e-6
    assert sigma < 1e-8


def test_fit_power_law_full_hand_computed_se():
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


def test_fit_power_law_full_extends_tuple_api():
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


def test_fit_power_law_full_ci_coverage_seeded():
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
            raise HTTPError("u", 429, "Too Many Requests",
                            None, None)
        return "ok"

    assert bp._with_retry(flaky, "leg-x") == "ok"
    assert calls["n"] == 3


def test_with_retry_names_the_leg_on_final_failure():
    import pytest

    def bad():
        raise ValueError("boom")

    with pytest.raises(RuntimeError, match="leg-y: boom"):
        bp._with_retry(bad, "leg-y")


def test_with_retry_does_not_retry_permanent_http_errors():
    from urllib.error import HTTPError

    import pytest

    calls = {"n": 0}

    def gone():
        calls["n"] += 1
        raise HTTPError("u", 404, "Not Found", None, None)

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


def test_fetch_powerlaw_dates_btc_in_eastern(monkeypatch):
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


def test_fetch_powerlaw_in_progress_bar_gets_et_today(monkeypatch):
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


def test_fetch_powerlaw_surfaces_slope_uncertainty(monkeypatch):
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


def test_chart_png_does_not_call_tight_layout():
    # Regression pin (2026-11-06): chart_png's twin axes are
    # incompatible with tight_layout — the call bailed to
    # the default subplot params and emitted a UserWarning
    # on every render (the app-log warning). The explicit
    # subplots_adjust pins the geometry; the call must not
    # come back.
    import inspect

    assert "tight_layout(" not in inspect.getsource(bp.chart_png)
