"""The dollar tab: quote conventions on synthetic series.
Every currency appreciates exactly 10%/yr against the dollar
on a USD-per-currency basis; DXY is flat at 100. The bars
must all read +10.0 at 1Y, breadth 11 of 11, DXY vs its
200-day exactly 0.0, and the render must carry three SVGs."""
import numpy as np
import pandas as pd

from toolbox.DataZoo.market.the_dollar import _compute, \
    render_the_dollar

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
    out = render_the_dollar(p)
    assert out.count("<svg") == 3
    assert "11 of 11" in out
