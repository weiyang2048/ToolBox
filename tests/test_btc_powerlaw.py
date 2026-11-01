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


def test_extend_cap_tail_scales_last_sample_by_price_growth():
    mc_idx = pd.date_range("2020-01-01", periods=3, freq="6D")
    mc = pd.Series([100.0, 110.0, 121.0], index=mc_idx)
    px_idx = pd.date_range("2020-01-01", periods=20, freq="D")
    price = pd.Series(10.0 * 1.1 ** np.arange(20), index=px_idx)
    out = bp.extend_cap_tail(mc, price)
    # history untouched; tail = last sample scaled by price
    # growth since the last sample date (day 12)
    assert list(out.iloc[:3]) == [100.0, 110.0, 121.0]
    assert out.index[-1] == px_idx[-1] and len(out) == 3 + 7
    expected = 121.0 * price.iloc[-1] / price.loc[mc_idx[-1]]
    assert abs(out.iloc[-1] / expected - 1) < 1e-12


def test_extend_cap_tail_no_tail_when_price_ends_at_sample():
    mc_idx = pd.date_range("2020-01-01", periods=3, freq="6D")
    mc = pd.Series([100.0, 110.0, 121.0], index=mc_idx)
    price = pd.Series([10.0, 11.0, 12.1], index=mc_idx)
    out = bp.extend_cap_tail(mc, price)
    assert list(out) == [100.0, 110.0, 121.0]
