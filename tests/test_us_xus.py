"""US vs ex-US: the shift math on synthetic total-return
series. US grows 10%/yr, developed ex-US 5%/yr, EM flat,
ACWX 6%/yr, dollar +2%/yr — so over the 5y window the ratios must fall,
the Dev 1y ratio change is (1.05/1.10 - 1), the spread equals
the ACWX minus SPY 12-month spread exactly (6% - 10% on the
final window, in points), and every ex-US ratio must sit
BELOW its 200-day (a steady relative downtrend)."""
import time
from urllib.error import HTTPError

import numpy as np
import pandas as pd

from toolbox.DataZoo.market import us_xus
from toolbox.DataZoo.market.us_xus import (
    _compute,
    chart_figs,
    us_xus_foot_html,
    us_xus_head_html,
)

N = 2900
IDX = pd.bdate_range("2015-01-01", periods=N)


def series(annual):
    g = (1 + annual) ** (1 / 252)
    return pd.Series(100 * g ** np.arange(N), index=IDX)


def test_shift_math_on_synthetic():
    df = pd.DataFrame({
        "SPY": series(0.10), "EFA": series(0.05),
        "EEM": series(0.0), "ACWX": series(0.06),
        "DX-Y.NYB": series(0.02)})
    p = _compute(df)
    assert p["snap"]["dev"]["chg1y"] == -4.5  # 1.05/1.10 - 1
    assert p["snap"]["em"]["chg1y"] == -9.1   # 1.00/1.10 - 1
    assert p["snap"]["xus"]["vs200"] < 0
    assert p["snap"]["dev"]["chgwin"] == -21.4
    # 5-year display window (2026-11-10): the frame starts
    # exactly 5 years before the last date, rebased to 100.
    first = (IDX[-1] - pd.DateOffset(years=5)).strftime(
        "%Y-%m-%d")
    assert p["frame"]["dates"][0] == first
    assert [p["frame"][k][0] for k in ("dev", "em", "xus")] == [
        100.0, 100.0, 100.0]
    # Dollar correlation pinned to that same 5y window:
    # recompute it over exactly the display rows and compare.
    _start = df.index.get_indexer(
        [df.index[-1] - pd.DateOffset(years=5)],
        method="bfill")[0]
    _d0 = df.index[_start]
    _both = pd.DataFrame({
        "dxy": df["DX-Y.NYB"] / df["DX-Y.NYB"].loc[_d0] * 100,
        "ratio": (df["ACWX"] / df["SPY"])
        / (df["ACWX"] / df["SPY"]).loc[_d0] * 100}).dropna()
    _both = _both.loc[_d0:]
    assert p["corr_dxy"] == round(float(
        _both["dxy"].pct_change().corr(
            _both["ratio"].pct_change())), 2)
    assert p["spread_now"] == -4.0           # 6% - 10% window
    assert p["ret1y"]["SPY"] == 10.0
    # corr itself is degenerate on two perfectly geometric
    # series (constant daily changes) — boundedness only here;
    # the sign is covered by the live smoke (-0.41 real data).
    assert -1.0 <= p["corr_dxy"] <= 1.0
    # The tab's render split (2026-11-10): head chips, three
    # Plotly panels on the shared warped axis (main panel
    # 560 tall, tape and dollar panels 280), footnote — each
    # block centered on the 1120px measure (2026-11-02).
    head = us_xus_head_html(p)
    assert "Shift tape" in head and "-4.0 pts" in head
    assert head.startswith(
        '<div style="max-width:1120px;margin:0 auto">')
    assert head.endswith("</div>")
    figs = chart_figs(p)
    assert [f.layout.height for _t, f in figs] == [560, 280, 280]
    assert figs[0][0] == ("Five years of US dominance — "
                          "ex-US ÷ US, relative strength ratios")
    _ann = [a.text for a in figs[0][1].layout.annotations]
    assert "level with the US" in _ann
    assert "-21.4% vs US" in _ann
    assert "200-day average" in us_xus_foot_html()
    assert "five-year panel" in us_xus_foot_html()
    # Freshness (2026-11-02): the payload carries its own last
    # date; a board date one session later adds the lag pill.
    assert p["last_date"] == IDX[-1].strftime("%Y-%m-%d")
    nxt = IDX[-1] + pd.tseries.offsets.BDay(1)
    out = us_xus_head_html(p, board_date=nxt.date())
    assert "1 session behind the board" in out
    out = us_xus_head_html(p, board_date=IDX[-1].date())
    assert "current" in out


def test_fetch_survives_a_transient_429_on_one_leg(monkeypatch):
    """One leg (EEM) 429s once, then answers: the shared retry
    absorbs it and the full payload still computes, with the
    same values as the direct _compute test."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    annual = {"SPY": 0.10, "EFA": 0.05, "EEM": 0.0,
              "ACWX": 0.06, "DX-Y.NYB": 0.02}
    calls = {}

    def fake(sym, p1=0, **kw):
        calls[sym] = calls.get(sym, 0) + 1
        if sym == "EEM" and calls[sym] == 1:
            raise HTTPError("u", 429, "Too Many Requests",
                            None, None)
        return series(annual[sym])

    monkeypatch.setattr(us_xus, "yahoo_daily", fake)
    p = us_xus.fetch_us_xus()
    assert calls["EEM"] == 2
    assert p["ret1y"]["SPY"] == 10.0
    assert p["spread_now"] == -4.0
