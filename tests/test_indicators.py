"""Tests for toolbox.DataZoo.market.indicators (QuantStats-backed)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("quantstats")

from toolbox.DataZoo.market.indicators import alpha_beta, drawdown_stats


def test_drawdown_stats_handles_missing_and_short_inputs():
    for prices in (None, pd.Series([100.0]), pd.Series([], dtype=float)):
        result = drawdown_stats(prices)
        assert np.isnan(result["current"])
        assert np.isnan(result["max"])
        assert np.isnan(result["ulcer_index"])
        assert np.isnan(result["pctile"])
        assert result["history"] == []


def test_drawdown_stats_known_path():
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    prices = pd.Series([100.0, 120.0, 60.0, 90.0], index=index)

    result = drawdown_stats(prices)

    assert result["current"] == pytest.approx(25.0)
    assert result["max"] == pytest.approx(50.0)
    # QuantStats' ulcer index divides the squared drawdowns by n - 1.
    assert result["ulcer_index"] == pytest.approx(39.53)
    assert result["history"] == [0.0, -50.0, -25.0]
    assert 0.0 <= result["pctile"] <= 100.0


def test_drawdown_stats_rising_prices_have_zero_drawdown():
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    prices = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)

    result = drawdown_stats(prices)

    assert result["current"] == pytest.approx(0.0)
    assert result["max"] == pytest.approx(0.0)
    assert result["ulcer_index"] == pytest.approx(0.0)
    assert result["pctile"] == pytest.approx(100.0)
    assert result["history"] == [0.0, 0.0, 0.0]


def test_alpha_beta_handles_missing_and_unaligned_inputs():
    index = pd.date_range("2024-01-01", periods=3, freq="B")
    returns = pd.Series([0.01, -0.02, 0.03], index=index)

    for pair in (
        (None, returns),
        (returns, None),
        (returns.iloc[:1], returns.iloc[:1]),
        (returns, pd.Series([0.01], index=[index[-1] + pd.Timedelta(days=7)])),
    ):
        beta, alpha = alpha_beta(*pair)
        assert np.isnan(beta)
        assert np.isnan(alpha)


def test_alpha_beta_known_linear_relationship():
    index = pd.date_range("2024-01-01", periods=8, freq="B")
    benchmark = pd.Series(
        [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, -0.015, 0.004],
        index=index,
    )
    strategy = 1.5 * benchmark + 0.001

    beta, alpha = alpha_beta(strategy, benchmark, periods=252)

    assert beta == pytest.approx(1.5)
    # QuantStats annualizes the 0.001 daily intercept at 252 periods.
    assert alpha == pytest.approx(0.252)


def test_alpha_beta_identical_series():
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    returns = pd.Series([0.01, -0.02, 0.03, 0.005], index=index)

    beta, alpha = alpha_beta(returns, returns)

    assert beta == pytest.approx(1.0)
    assert alpha == pytest.approx(0.0)
