"""Offline tests for the Bitcoin power-law math (no network)."""
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

