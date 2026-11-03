"""Tests for the 12-1 momentum functions in
toolbox.DataZoo.market.indicators — ported with the module from
the Investments app (user-approved migration 2026-11-03).

Config defaults are always passed explicitly, so the values
under test never depend on any app's configuration.
"""
import numpy as np
import pandas as pd
import pytest

from toolbox.DataZoo.market.indicators import (
    _compute_raw_momentum,
    compute_momentum_universe,
)

PARAMS = dict(skip_days=21, lookback_days=252, min_lookback_days=30)


def _growth(days=300, rate=0.01):
    return pd.Series(100.0 * (1.0 + rate) ** np.arange(days))


def test_growth_series_known_value():
    mom = _compute_raw_momentum(_growth(), **PARAMS)
    # end = P(-22), start = P(-253): ratio spans lookback - skip days.
    assert mom == pytest.approx(1.01 ** (252 - 21) - 1, rel=1e-9)


def test_constant_prices_zero_momentum():
    s = pd.Series(np.full(300, 50.0))
    assert _compute_raw_momentum(s, **PARAMS) == 0.0


def test_lookback_clamps_to_available_history():
    # 40 bars: effective lookback 39, span 39 - 21 = 18 >= min 10.
    mom = _compute_raw_momentum(
        _growth(days=40), skip_days=21, lookback_days=252,
        min_lookback_days=10)
    assert mom == pytest.approx(1.01 ** 18 - 1, rel=1e-9)


def test_too_short_history_returns_none():
    assert _compute_raw_momentum(_growth(days=40), **PARAMS) is None


def test_skip_not_less_than_lookback_returns_none():
    assert _compute_raw_momentum(
        _growth(), skip_days=252, lookback_days=252,
        min_lookback_days=30) is None


def test_empty_and_none_return_none():
    assert _compute_raw_momentum(pd.Series(dtype=float), **PARAMS) is None
    assert _compute_raw_momentum(None, **PARAMS) is None


def test_nonpositive_price_raises():
    s = _growth()
    s.iloc[10] = -1.0
    with pytest.raises(ValueError):
        _compute_raw_momentum(s, **PARAMS)


def test_universe_scores_each_column_and_skips_short():
    prices = pd.DataFrame({
        "UP": _growth().to_numpy(),
        "FLAT": np.full(300, 25.0),
        "SHORT": np.concatenate([_growth(days=40).to_numpy(),
                                 np.full(260, np.nan)]),
    })
    out = compute_momentum_universe(prices, **PARAMS)
    assert set(out.index) == {"UP", "FLAT"}
    assert out["UP"] == pytest.approx(1.01 ** 231 - 1, rel=1e-9)
    assert out["FLAT"] == 0.0
