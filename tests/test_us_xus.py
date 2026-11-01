"""US vs ex-US: the shift math on synthetic total-return
series. US grows 10%/yr, developed ex-US 5%/yr, EM flat,
ACWX 6%/yr, dollar +2%/yr — so over 10y the ratios must fall,
the Dev 1y ratio change is (1.05/1.10 - 1), the spread equals
the ACWX minus SPY 12-month spread exactly (6% - 10% on the
final window, in points), and every ex-US ratio must sit
BELOW its 200-day (a steady relative downtrend)."""
import time
from urllib.error import HTTPError

import numpy as np
import pandas as pd

from toolbox.DataZoo.market import us_xus
from toolbox.DataZoo.market.us_xus import _compute, render_us_xus

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
    assert p["snap"]["dev"]["chg10y"] < -30
    assert p["spread_now"] == -4.0           # 6% - 10% window
    assert p["ret1y"]["SPY"] == 10.0
    # corr itself is degenerate on two perfectly geometric
    # series (constant daily changes) — boundedness only here;
    # the sign is covered by the live smoke (-0.41 real data).
    assert -1.0 <= p["corr_dxy"] <= 1.0
    out = render_us_xus(p)
    assert "US vs ex-US" not in out  # title lives in the tab
    assert "Shift tape" in out and "-4.0 pts" in out
    assert out.count("<svg") == 3


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
