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


def test_chart_png_does_not_call_tight_layout():
    # Regression pin (2026-11-06): chart_png's twin axes are
    # incompatible with tight_layout — the call bailed to
    # the default subplot params and emitted a UserWarning
    # on every render (the app-log warning). The explicit
    # subplots_adjust pins the geometry; the call must not
    # come back.
    import inspect

    assert "tight_layout(" not in inspect.getsource(bp.chart_png)
