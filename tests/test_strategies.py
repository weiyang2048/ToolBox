"""Tests for toolbox.strategies (cleaning, solvers, backtest)."""

import numpy as np
import pandas as pd
import pytest

from toolbox.strategies import (
    COMBO_STRATEGY_NAMES,
    STRATEGY_NAMES,
    bbp_clean_corr,
    clean_covariance,
    clean_mean,
    estimate_inputs,
    james_stein,
    log_returns,
    nlshrink_eigs,
    resolve_strategy,
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


@pytest.mark.parametrize("method", ["sample", "bbp", "clip", "ledoit_wolf",
                                    "nlshrink"])
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
    # BBP must at least move the spectrum: largest eigenvalue shrunk.
    assert info_bbp["eigenvalues"][0] >= np.linalg.eigvalsh(
        info_bbp["corr_clean"])[-1] - 1e-8


def test_nlshrink_matches_rie_oracle_on_identity_population():
    # With population = identity the covariance LSD *is* Marčenko–
    # Pastur, so the analytic BBP/RIE map applied to the covariance
    # eigenvalues is the exact oracle; the kernel plug-in must land
    # close to it (gap = kernel smoothing bias, shrinking in n).
    rng = np.random.default_rng(42)
    n, p = 1000, 100
    X = rng.normal(size=(n, p))
    S0 = np.cov(X.T) * (n - 1) / n
    lam = np.linalg.eigvalsh(S0)
    d_nls = np.sort(nlshrink_eigs(lam, n))
    d_oracle = np.sort(np.linalg.eigvalsh(bbp_clean_corr(S0, p / n)))
    rel = np.linalg.norm(d_nls - d_oracle) / np.linalg.norm(d_oracle)
    assert rel < 0.06
    # Trace is (asymptotically) preserved by the RIE map.
    assert d_nls.sum() == pytest.approx(lam.sum(), rel=0.02)


def test_nlshrink_scale_equivariance_and_compression():
    lam = np.array([0.02, 0.05, 0.1, 0.3, 0.8, 1.5, 3.0, 6.0])
    a = nlshrink_eigs(lam, 500)
    b = nlshrink_eigs(37.0 * lam, 500)
    assert np.allclose(b, 37.0 * a, rtol=1e-8)
    # Nonlinear shrinkage compresses: top eigenvalue down, bottom up.
    rng = np.random.default_rng(5)
    X = rng.normal(0, 0.01, size=(800, 12))
    S0 = np.cov(X.T) * (799 / 800)
    raw = np.linalg.eigvalsh(S0)
    d = np.sort(nlshrink_eigs(raw, 800))
    assert d[-1] < raw[-1]
    assert d[0] > raw[0]


def test_nlshrink_q_ge_1_falls_back_to_ledoit_wolf():
    R = _returns(n=30, p=40)
    S_nls, _ = clean_covariance(R, "nlshrink")
    S_lw, _ = clean_covariance(R, "ledoit_wolf")
    assert np.allclose(S_nls, S_lw)


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
    from toolbox.strategies import erc_weights

    R = _returns()
    S, _ = clean_covariance(R, "sample")
    w = erc_weights(S)
    rc = w * (S @ w)
    assert np.allclose(rc / rc.sum(), 1.0 / len(w), atol=0.01)


def test_max_sharpe_all_negative_means_not_minvar():
    # With every excess mean negative the tangency QP (w'μ = 1) is
    # infeasible; the direct maximiser must still find the least-bad
    # Sharpe, which beats the min-variance portfolio's Sharpe.
    from toolbox.strategies import (
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


def test_resolve_strategy_names():
    assert resolve_strategy("1/N") == ("1/N", None, None)
    assert resolve_strategy("Average") == ("Average", None, None)
    assert resolve_strategy("Max Sharpe (raw × sample)") == (
        "Max Sharpe", "sample", "sample")
    assert resolve_strategy("Max Sharpe (BBP × JS)") == (
        "Max Sharpe", "bbp", "james_stein")
    assert resolve_strategy("Max Sharpe (BBP × BS)") == (
        "Max Sharpe", "bbp", "bayes_stein")
    assert resolve_strategy("Min variance (raw)") == (
        "Min variance", "sample", None)
    assert resolve_strategy("NCO (BBP)") == ("NCO", "bbp", None)
    assert resolve_strategy("Max Sharpe (LW × BS)") == (
        "Max Sharpe", "ledoit_wolf", "bayes_stein")
    assert resolve_strategy("Min variance (LW)") == (
        "Min variance", "ledoit_wolf", None)
    with pytest.raises(ValueError):
        resolve_strategy("Max Sharpe (XX × BS)")
    with pytest.raises(ValueError):
        resolve_strategy("Bogus")


def test_combo_names_cover_the_demoed_grid():
    # 92 since the estimation windows became strategy variants
    # (user direction 2026-11-02): the 21 window-using bases
    # (Max Sharpe 3 covs x 3 means, the three cov-only rules x 3
    # covs, HRP, risk parity, Min CVaR) at four windows each,
    # plus 1/N, the six RS pairs and Average.
    assert len(COMBO_STRATEGY_NAMES) == 92
    assert len(set(COMBO_STRATEGY_NAMES)) == 92
    assert "RS(1, 126)" in COMBO_STRATEGY_NAMES
    for name in COMBO_STRATEGY_NAMES:
        resolve_strategy(name)  # every listed name must resolve


def test_snapshot_combo_rows_match_their_own_pair():
    R = _returns(n=300)
    names = ["Max Sharpe (BBP × BS)", "Min variance (raw)",
             "Min variance (LW)", "Average"]
    W, _info = snapshot_weights(R, "sample", "sample", 0.0, names)
    mu, S, _i = estimate_inputs(R, "bbp", "bayes_stein", 0.0)
    want = strategy_weights("Max Sharpe", R, S, mu, 0.0)
    np.testing.assert_allclose(W.loc["Max Sharpe (BBP × BS)"].values,
                               want, atol=1e-12)
    mu_lw, S_lw, _i_lw = estimate_inputs(R, "ledoit_wolf", "sample", 0.0)
    want_lw = strategy_weights("Min variance", R, S_lw, mu_lw, 0.0)
    np.testing.assert_allclose(W.loc["Min variance (LW)"].values,
                               want_lw, atol=1e-12)
    mu0, S0, _i0 = estimate_inputs(R, "sample", "sample", 0.0)
    want0 = strategy_weights("Min variance", R, S0, mu0, 0.0)
    np.testing.assert_allclose(W.loc["Min variance (raw)"].values,
                               want0, atol=1e-12)
    np.testing.assert_allclose(
        W.loc["Average"].values,
        W.drop(index="Average").mean().values, atol=1e-12)
    assert list(W.index) == names  # combo display order, Average last


def test_backtest_combo_strategies_run():
    prices = _prices(n=400)
    names = ["Max Sharpe (raw × JS)", "Max Sharpe (BBP × BS)",
             "Min variance (BBP)", "Average"]
    out = walk_forward_backtest(prices, strategies=names,
                                cov_method="bbp", mean_method="bayes_stein",
                                seed=126, hold=21)
    assert out is not None
    assert set(out["returns"].columns) == set(names)
    assert set(out["stats"].index) == set(names)


def test_backtest_window_huge_equals_expanding():
    prices = _prices(n=400)
    names = ["Max Sharpe (BBP × BS)", "Min variance (raw)", "Average"]
    kw = dict(strategies=names, cov_method="bbp",
              mean_method="bayes_stein", seed=126, hold=21)
    exp = walk_forward_backtest(prices, **kw)
    huge = walk_forward_backtest(prices, window=10 ** 9, **kw)
    pd.testing.assert_frame_equal(exp["returns"], huge["returns"])
    assert exp["window"] is None and huge["window"] == 10 ** 9


def test_backtest_window_rolling_keeps_oos_span():
    prices = _prices(n=400)
    names = ["Max Sharpe (BBP × BS)", "Average"]
    kw = dict(strategies=names, cov_method="bbp",
              mean_method="bayes_stein", seed=126, hold=21)
    exp = walk_forward_backtest(prices, **kw)
    roll = walk_forward_backtest(prices, window=126, **kw)
    assert roll["oos_start"] == exp["oos_start"]
    assert roll["oos_end"] == exp["oos_end"]
    assert roll["n_rebalances"] == exp["n_rebalances"]
    assert not np.allclose(roll["returns"].values, exp["returns"].values)
    with pytest.raises(ValueError):
        walk_forward_backtest(prices, window=0, **kw)


def _rs_returns():
    """Four tickers with known trailing sums: over the full
    frame A > B > C > D; over the last 2 rows C is strongest."""
    data = {
        "A": [0.06, 0.06, 0.06, 0.01, 0.01],
        "B": [0.05, 0.05, 0.05, 0.01, 0.01],
        "C": [0.01, 0.01, 0.01, 0.06, 0.06],
        "D": [0.00, 0.00, 0.00, 0.00, 0.00],
    }
    idx = pd.bdate_range("2026-01-05", periods=5)
    return pd.DataFrame(data, index=idx)


def test_rs_weights_top_n_equal_weight():
    from toolbox.strategies import rs_weights

    R = _rs_returns()
    # Full window: A (0.20) and B (0.17) lead.
    assert list(rs_weights(R, 2, 5)) == [0.5, 0.5, 0.0, 0.0]
    # Winner-take-all at w=2: C's last two days dominate.
    assert list(rs_weights(R, 1, 2)) == [0.0, 0.0, 1.0, 0.0]
    # A window longer than the history uses all of it.
    assert list(rs_weights(R, 1, 252)) == [1.0, 0.0, 0.0, 0.0]
    # n at the universe size is 1/N.
    assert list(rs_weights(R, 4, 5)) == [0.25] * 4
    assert list(rs_weights(R, 9, 5)) == [0.25] * 4


def test_rs_ties_keep_universe_order():
    from toolbox.strategies import rs_weights

    R = _rs_returns()
    R["B"] = R["A"]  # exact tie for the lead
    assert list(rs_weights(R, 1, 5)) == [1.0, 0.0, 0.0, 0.0]


def test_rs_names_resolve_and_rank_before_average():
    from toolbox.strategies import (
        COMBO_STRATEGY_NAMES,
        RS_STRATEGY_NAMES,
        _strategy_order,
        resolve_strategy,
    )

    for name in RS_STRATEGY_NAMES:
        assert name in COMBO_STRATEGY_NAMES
        assert resolve_strategy(name) == (name, None, None)
    # Any well-formed pair resolves, grid membership or not.
    assert resolve_strategy("RS(4, 40)") == ("RS(4, 40)", None, None)
    with pytest.raises(ValueError):
        resolve_strategy("RS(0, 40)")
    order = _strategy_order(["Average", "RS(1, 252)", "1/N"])
    assert order == ["1/N", "RS(1, 252)", "Average"]


def test_rs_snapshot_and_backtest():
    R = _returns(n=300)
    W, _info = snapshot_weights(R, "ledoit_wolf", "sample", 0.0,
                                ["RS(2, 63)", "Average"])
    w = W.loc["RS(2, 63)"].values
    assert np.isclose(w.sum(), 1.0)
    assert sorted(w)[-2:] == [0.5, 0.5]
    assert np.allclose(W.loc["Average"].values, w)

    prices = _prices(n=500)
    out = walk_forward_backtest(
        prices, strategies=["RS(1, 21)"], cov_method="bbp",
        mean_method="sample", seed=126, hold=21)
    assert out is not None
    assert set(out["stats"].index) == {"RS(1, 21)"}
    assert np.isfinite(out["stats"].values).all()


def test_parse_window_suffix():
    from toolbox.strategies import parse_window

    assert parse_window("Max Sharpe (LW × sample) @ 252") == (
        "Max Sharpe (LW × sample)", 252)
    assert parse_window("HRP @ All") == ("HRP", None)
    assert parse_window("Min variance (raw)") == (
        "Min variance (raw)", None)
    assert parse_window("RS(1, 252)") == ("RS(1, 252)", None)
    with pytest.raises(ValueError):
        parse_window("HRP @ soon")
    with pytest.raises(ValueError):
        parse_window("HRP @ 0")


def test_windowed_grid_shape():
    # 21 window-using bases x 4 windows + 1/N + 6 RS + Average.
    assert len(COMBO_STRATEGY_NAMES) == 92
    assert "Max Sharpe (LW × sample) @ 252" in COMBO_STRATEGY_NAMES
    assert "Max Sharpe (LW × sample) @ All" in COMBO_STRATEGY_NAMES
    assert "Min variance (LW) @ 126" in COMBO_STRATEGY_NAMES
    assert "HRP @ 504" in COMBO_STRATEGY_NAMES
    assert "Min CVaR @ All" in COMBO_STRATEGY_NAMES
    assert "HRP" not in COMBO_STRATEGY_NAMES
    assert "1/N" in COMBO_STRATEGY_NAMES
    for name in COMBO_STRATEGY_NAMES:
        resolve_strategy(name)  # every grid name resolves


def test_snapshot_window_suffix_slices_per_strategy():
    R = _returns()
    names = ["Min variance (raw) @ 63", "Min variance (raw) @ All",
             "Min variance (raw)"]
    W, _info = snapshot_weights(R, "sample", "sample", 0.0, names)
    # "@ All" is the expanding window: identical to no suffix.
    assert np.allclose(W.loc["Min variance (raw) @ All"],
                       W.loc["Min variance (raw)"])
    # A short rolling window estimates on different data.
    assert not np.allclose(W.loc["Min variance (raw) @ 63"],
                           W.loc["Min variance (raw)"])


def test_backtest_name_window_overrides_global():
    prices = _prices(n=400)
    kw = dict(cov_method="bbp", mean_method="bayes_stein",
              seed=126, hold=21)
    suffixed = walk_forward_backtest(
        prices, strategies=["Min variance (raw) @ 126"], **kw)
    globaled = walk_forward_backtest(
        prices, strategies=["Min variance (raw)"], window=126, **kw)
    assert np.allclose(suffixed["returns"].values,
                       globaled["returns"].values)
    expanding = walk_forward_backtest(
        prices, strategies=["Min variance (raw) @ All"], window=126,
        **kw)
    plain = walk_forward_backtest(
        prices, strategies=["Min variance (raw)"], **kw)
    assert np.allclose(expanding["returns"].values,
                       plain["returns"].values)
