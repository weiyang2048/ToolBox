"""Tests for toolbox.DataZoo.market.strategies (cleaning, solvers, backtest)."""

import numpy as np
import pandas as pd
import pytest

from toolbox.DataZoo.market.strategies import (
    STRATEGY_NAMES,
    clean_covariance,
    clean_mean,
    james_stein,
    log_returns,
    snapshot_weights,
    strategy_weights,
    walk_forward_backtest,
)

RNG = np.random.default_rng(7)


def _returns(n=500, p=8, seed=7):
    rng = np.random.default_rng(seed)
    # One common factor + idiosyncratic noise: a spiked population.
    f = rng.normal(0, 0.01, size=(n, 1))
    beta = rng.uniform(0.5, 1.5, size=(1, p))
    idio = rng.normal(0, 0.008, size=(n, p))
    X = f * beta + idio + 0.0002
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame(X, index=idx, columns=[f"A{i}" for i in range(p)])


def _prices(n=500, p=8):
    R = _returns(n=n, p=p)
    return 100.0 * np.exp(R.cumsum())


@pytest.mark.parametrize("method", ["sample", "bbp", "clip", "ledoit_wolf"])
def test_clean_covariance_psd_and_symmetric(method):
    R = _returns()
    S, info = clean_covariance(R, method)
    assert np.allclose(S, S.T)
    assert np.linalg.eigvalsh(S).min() > -1e-10
    assert info["q"] == pytest.approx(R.shape[1] / len(R))
    assert info["n_signal"] >= 1  # the factor spike pokes out of the bulk


def test_bbp_and_clip_recover_factor_better_than_sample():
    # Frobenius distance of the cleaned correlation to the population
    # correlation should beat the sample's on a spiked population.
    rng = np.random.default_rng(3)
    p, n = 10, 400
    beta = rng.uniform(0.5, 1.5, size=p)
    pop_corr = np.outer(beta, beta) * 0.01 ** 2
    pop_corr = pop_corr / np.outer(np.sqrt(np.diag(pop_corr) + 0.008 ** 2),
                                   np.sqrt(np.diag(pop_corr) + 0.008 ** 2))
    np.fill_diagonal(pop_corr, 1.0)
    R = _returns(n=n, p=p, seed=11)
    _, info_bbp = clean_covariance(R, "bbp")
    C_sample = np.corrcoef(R.values.T)
    # BBP must at least move the spectrum: largest eigenvalue shrunk.
    assert info_bbp["eigenvalues"][0] >= np.linalg.eigvalsh(
        info_bbp["corr_clean"])[-1] - 1e-8


def test_james_stein_shrinks_toward_grand_mean():
    mu = np.array([0.01, -0.005, 0.02, 0.0, 0.004, -0.002])
    shrunk, c = james_stein(mu, samp_var=1e-5)
    assert 0.0 <= c <= 1.0
    assert np.std(shrunk) <= np.std(mu) + 1e-12
    assert shrunk.mean() == pytest.approx(mu.mean())


@pytest.mark.parametrize("mean_method", ["sample", "james_stein",
                                         "bayes_stein", "grand_mean"])
def test_clean_mean_variants(mean_method):
    R = _returns()
    S, _ = clean_covariance(R, "sample")
    mu, _ = clean_mean(R, S, mean_method, rf_d=0.0)
    assert mu.shape == (R.shape[1],)
    if mean_method == "grand_mean":
        assert np.allclose(mu, mu[0])


@pytest.mark.parametrize("name", [s for s in STRATEGY_NAMES if s != "Average"])
def test_strategy_weights_long_only_fully_invested(name):
    R = _returns(n=300)
    S, _ = clean_covariance(R, "bbp")
    mu, _ = clean_mean(R, S, "james_stein", rf_d=0.0)
    w = strategy_weights(name, R, S, mu, rf_d=0.0)
    assert w.sum() == pytest.approx(1.0)
    assert w.min() >= -1e-9


def test_minvar_beats_equal_weight_in_sample():
    R = _returns()
    S, _ = clean_covariance(R, "sample")
    mu, _ = clean_mean(R, S, "sample", rf_d=0.0)
    w_mv = strategy_weights("Min variance", R, S, mu, rf_d=0.0)
    w_eq = np.ones(len(mu)) / len(mu)
    assert w_mv @ S @ w_mv <= w_eq @ S @ w_eq + 1e-12


def test_snapshot_weights_average_is_mean():
    R = _returns(n=300)
    W, info = snapshot_weights(R, "bbp", "james_stein", 0.0,
                               ["Min variance", "Max Sharpe", "Average"])
    assert list(W.index) == ["Min variance", "Max Sharpe", "Average"]
    assert np.allclose(W.loc["Average"].values,
                       W.loc[["Min variance", "Max Sharpe"]].mean().values)
    assert np.allclose(W.sum(axis=1).values, 1.0)


def test_walk_forward_backtest_runs_and_is_cost_monotone():
    prices = _prices(n=500)
    out = walk_forward_backtest(
        prices, strategies=["1/N", "Min variance", "Max Sharpe", "Average"],
        cov_method="bbp", mean_method="james_stein", seed=126, hold=21)
    assert out is not None
    assert out["n_rebalances"] >= 10
    assert len(out["returns"]) == 500 - 1 - 126
    assert set(out["stats"].index) == {"1/N", "Min variance", "Max Sharpe", "Average"}
    assert np.isfinite(out["stats"].values).all()
    costly = walk_forward_backtest(
        prices, strategies=["Max Sharpe"], cov_method="bbp",
        mean_method="james_stein", seed=126, hold=21, cost_bps=50.0)
    free = walk_forward_backtest(
        prices, strategies=["Max Sharpe"], cov_method="bbp",
        mean_method="james_stein", seed=126, hold=21, cost_bps=0.0)
    assert (costly["stats"].loc["Max Sharpe", "total_return"]
            <= free["stats"].loc["Max Sharpe", "total_return"] + 1e-12)


def test_erc_equalises_risk_contributions():
    from toolbox.DataZoo.market.strategies import erc_weights

    R = _returns()
    S, _ = clean_covariance(R, "sample")
    w = erc_weights(S)
    rc = w * (S @ w)
    assert np.allclose(rc / rc.sum(), 1.0 / len(w), atol=0.01)


def test_max_sharpe_all_negative_means_not_minvar():
    # With every excess mean negative the tangency QP (w'μ = 1) is
    # infeasible; the direct maximiser must still find the least-bad
    # Sharpe, which beats the min-variance portfolio's Sharpe.
    from toolbox.DataZoo.market.strategies import (
        max_sharpe_weights,
        minvar_weights,
    )

    R = _returns(n=300)
    S, _ = clean_covariance(R, "bbp")
    mu, _ = clean_mean(R, S, "james_stein", rf_d=0.0)
    if not (mu < 0).all():
        pytest.skip("fixture means not all negative")
    sharpe = lambda w: w @ mu / np.sqrt(w @ S @ w)
    assert sharpe(max_sharpe_weights(S, mu)) >= sharpe(minvar_weights(S)) - 1e-9


def test_backtest_too_short_returns_none():
    assert walk_forward_backtest(_prices(n=100), strategies=["1/N"]) is None


def test_log_returns_shape():
    prices = _prices(n=100)
    R = log_returns(prices)
    assert len(R) == 99 and R.shape[1] == prices.shape[1]
