"""The dollar tab: quote conventions on synthetic series.
Every currency appreciates exactly 10%/yr against the dollar
on a USD-per-currency basis; DXY is flat at 100. The bars
must all read +10.0 at 1Y, breadth 11 of 11, DXY vs its
200-day exactly 0.0; the head must carry the breadth chip."""
import time
from urllib.error import HTTPError

import numpy as np
import pandas as pd

from toolbox.DataZoo.market import the_dollar
from toolbox.DataZoo.market.the_dollar import (
    _compute,
    chart_figs,
    dollar_bars_html,
    dollar_foot_html,
    dollar_head_html,
)

N = 2900
IDX = pd.bdate_range("2015-01-01", periods=N)
NAMES = ["Euro", "British pound", "Australian dollar",
         "New Zealand dollar", "Japanese yen", "Swiss franc",
         "Canadian dollar", "Swedish krona", "Norwegian krone",
         "Danish krone", "Chinese renminbi"]


def test_dollar_math_on_synthetic():
    g = 1.10 ** (1 / 252)
    usd = pd.DataFrame({n: 0.5 * g ** np.arange(N)
                        for n in NAMES}, index=IDX)
    dxy = pd.Series(100.0, index=IDX)
    p = _compute(dxy, usd)
    assert all(b["1Y"] == 10.0 for b in p["bars"])
    assert p["breadth"] == 11
    assert p["dxy_vs200"] == 0.0
    assert p["dxy_from_high"] == 0.0
    assert p["vol_now"] >= 0.0
    # The tab's render split (2026-11-10): head chips, two
    # Plotly line panels on the shared warped axis (main
    # 560 tall, vol 280), the categorical bars panel still
    # SVG, and the footnote — each block centered on the
    # 1120px measure (2026-11-02).
    head = dollar_head_html(p)
    assert "11 of 11" in head
    assert head.startswith(
        '<div style="max-width:1120px;margin:0 auto">')
    assert head.endswith("</div>")
    figs = chart_figs(p)
    assert [f.layout.height for _t, f in figs] == [560, 280]
    assert dollar_bars_html(p).count("<svg") == 1
    assert "renminbi" in dollar_foot_html()
    # Freshness (2026-11-02): per-panel last dates are exposed
    # and the note takes the lag pill for a later board date.
    assert p["last_date"] == IDX[-1].strftime("%Y-%m-%d")
    assert p["dxy_last_date"] == p["last_date"]
    assert p["vol_last_date"] <= p["last_date"]
    nxt = IDX[-1] + pd.tseries.offsets.BDay(1)
    out = dollar_head_html(p, board_date=nxt.date())
    assert "1 session behind the board" in out


def test_fetch_survives_a_transient_429_on_one_leg(monkeypatch):
    """One leg (the euro) 429s once, then answers: the shared
    retry absorbs it and the full payload still computes.
    Yahoo is stubbed with the same synthetic series as above,
    so the payload must carry the same +10.0 bars."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    g = 1.10 ** (1 / 252)
    calls = {}

    def fake(sym, p1=0, **kw):
        calls[sym] = calls.get(sym, 0) + 1
        if sym == "EURUSD=X" and calls[sym] == 1:
            raise HTTPError("u", 429, "Too Many Requests",
                            None, None)
        if sym == "DX-Y.NYB":
            return pd.Series(100.0, index=IDX)
        return pd.Series(0.5 * g ** np.arange(N), index=IDX)

    monkeypatch.setattr(the_dollar, "yahoo_daily", fake)
    p = the_dollar.fetch_the_dollar()
    assert calls["EURUSD=X"] == 2
    assert all(b["1Y"] == 10.0 for b in p["bars"])
    assert p["breadth"] == 11
