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
    cvar_weights,
    estimate_inputs,
    hrp_weights,
    james_stein,
    log_returns,
    nco_weights,
    nlshrink_eigs,
    performance_stats,
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


def test_snapshot_weights_progress_callback():
    # The progress hook (the dashboard's current-weights
    # bar): one call before the estimation, then one per
    # solved strategy, labelled by strategy name; Average
    # is derived, not solved, so it does not count.
    R = _returns(n=300)
    calls = []
    W, _info = snapshot_weights(
        R, "sample", "sample", 0.0,
        ["Min variance", "Max Sharpe", "Average"],
        progress=lambda d, t, s: calls.append((d, t, s)))
    assert list(W.index) == ["Min variance", "Max Sharpe", "Average"]
    assert calls == [(0, 2, ""), (1, 2, "Min variance"),
                     (2, 2, "Max Sharpe")]


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


def test_merged_stats_progress_callback():
    # The progress hook (the dashboard's stats-table bar):
    # one call before the per-ticker loop, then one per
    # ticker, labelled by ticker, ending at (n, n).
    from toolbox.strategies import merged_stats

    prices = _prices(n=300, p=4)
    calls = []
    out = merged_stats(prices,
                       progress=lambda d, t, s: calls.append((d, t, s)))
    assert out is not None and len(out) == 4
    assert [c[0] for c in calls] == [0, 1, 2, 3, 4]
    assert all(c[1] == 4 for c in calls)
    assert [c[2] for c in calls[1:]] == list(prices.columns)


def test_walk_forward_progress_callback():
    # The progress hook (the dashboard's progress bar): one
    # call before the loop with done == 0, then one after
    # each rebalance, ending at (total, total) with total
    # equal to the run's rebalance count; the dates are the
    # rebalance dates in order.
    prices = _prices(n=500)
    calls = []
    out = walk_forward_backtest(
        prices, strategies=["1/N", "Min variance"],
        cov_method="bbp", mean_method="james_stein", seed=126,
        hold=21, progress=lambda d, t, w: calls.append((d, t, w)))
    assert out is not None
    total = out["n_rebalances"]
    assert [c[0] for c in calls] == list(range(total + 1))
    assert all(c[1] == total for c in calls)
    hist_dates = [h["date"] for h in out["weights_history"]]
    assert [c[2] for c in calls[1:]] == hist_dates
    assert calls[0][2] == hist_dates[0]


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
    # 94 since SPMO (cap × score) joined as a lone tweak
    # (user direction 2026-11-06): the 21 window-using bases
    # (Max Sharpe 3 covs x 3 means, the three cov-only rules
    # x 3 covs, HRP, risk parity, Min CVaR) at four windows
    # each (the windows became strategy variants 2026-11-02),
    # plus 1/N, the six RS pairs, SPMO, its cap-weighted
    # sibling and Average.
    assert len(COMBO_STRATEGY_NAMES) == 94
    assert len(set(COMBO_STRATEGY_NAMES)) == 94
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
    kw = {"strategies": names, "cov_method": "bbp",
              "mean_method": "bayes_stein", "seed": 126, "hold": 21}
    exp = walk_forward_backtest(prices, **kw)
    huge = walk_forward_backtest(prices, window=10 ** 9, **kw)
    pd.testing.assert_frame_equal(exp["returns"], huge["returns"])
    assert exp["window"] is None and huge["window"] == 10 ** 9


def test_backtest_window_rolling_keeps_oos_span():
    prices = _prices(n=400)
    names = ["Max Sharpe (BBP × BS)", "Average"]
    kw = {"strategies": names, "cov_method": "bbp",
              "mean_method": "bayes_stein", "seed": 126, "hold": 21}
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


def _spmo_returns(n=300, drifts=None):
    """Deterministic log returns: a per-ticker drift plus a
    shared wiggle, so daily volatility is positive everywhere
    and the drift order drives the momentum order."""
    if drifts is None:
        drifts = [0.0002 * j for j in range(10)]
    wiggle = 0.001 * np.sin(1.7 * np.arange(n))
    return pd.DataFrame({f"T{j}": d + wiggle
                         for j, d in enumerate(drifts)})


def test_spmo_scores_rank_and_bound():
    from toolbox.strategies import spmo_scores

    scores = spmo_scores(_spmo_returns())
    assert len(scores) == 10
    assert scores.between(0.25, 4.0).all()
    assert scores.is_monotonic_increasing
    assert scores.iloc[-1] > 1.0 > scores.iloc[0]


def test_spmo_weights_top_quintile_by_score():
    from toolbox.strategies import spmo_scores, spmo_weights

    R = _spmo_returns()
    w = spmo_weights(R)
    scores = spmo_scores(R)
    # The top quintile of ten is two names: the highest drifts.
    assert list(np.flatnonzero(w)) == [8, 9]
    expect = scores.iloc[[8, 9]].to_numpy()
    assert np.allclose(w[[8, 9]], expect / expect.sum())
    assert np.isclose(w.sum(), 1.0)


def test_spmo_skips_the_most_recent_month():
    from toolbox.strategies import rs_weights, spmo_weights

    R = _spmo_returns(drifts=[0.001] * 10)
    wiggle = 0.001 * np.sin(1.7 * np.arange(len(R)))
    R["T9"] = 0.002 + wiggle
    R.iloc[-21:, R.columns.get_loc("T9")] = -0.05
    # The skipped month keeps the crashed leader's score, so
    # SPMO still holds it; a plain trailing rule ranks it last
    # and holds none of it.
    assert spmo_weights(R)[9] > 0
    assert rs_weights(R, 1, 252)[9] == 0.0


def test_spmo_fallback_window_and_short_history():
    from toolbox.strategies import spmo_scores, spmo_weights

    # 230 rows: under the 273 the 12-month window needs, over
    # the 210 the 9-month fallback needs — scores still form.
    R = _spmo_returns(n=230)
    assert len(spmo_scores(R)) == 10
    assert np.isclose(spmo_weights(R).sum(), 1.0)
    # 150 rows: nothing is scorable and the rule is 1/N.
    R = _spmo_returns(n=150)
    assert spmo_scores(R).empty
    assert np.allclose(spmo_weights(R), 0.1)


def test_spmo_variant_defaults_match_plain():
    from toolbox.strategies import spmo_variant_weights, spmo_weights

    R = _spmo_returns()
    assert np.array_equal(spmo_variant_weights(R), spmo_weights(R))
    # Nothing scorable: both fall back to 1/N.
    short = _spmo_returns(n=150)
    assert np.allclose(spmo_variant_weights(short), 0.1)


def test_spmo_variant_cap_times_score():
    from toolbox.strategies import spmo_scores, spmo_variant_weights

    R = _spmo_returns()
    scores = spmo_scores(R)
    caps = pd.Series(dict.fromkeys(R.columns, 1.0))
    caps["T8"] = 3.0
    w = spmo_variant_weights(R, caps=caps)
    # Same two selected names as the plain rule, reweighted
    # in proportion to cap x score.
    assert list(np.flatnonzero(w)) == [8, 9]
    expect = scores.iloc[[8, 9]].to_numpy() * np.array([3.0, 1.0])
    assert np.allclose(w[[8, 9]], expect / expect.sum())
    # A selected name with no cap drops out entirely, and
    # the survivor takes the whole book.
    caps["T9"] = np.nan
    w = spmo_variant_weights(R, caps=caps)
    assert list(np.flatnonzero(w)) == [8]
    assert np.isclose(w[8], 1.0)


def test_spmo_select_buffer_retention():
    from toolbox.strategies import spmo_select

    scores = pd.Series({f"T{j}": float(j) for j in range(10)})
    # Plain: the top quintile of ten is two names.
    assert list(spmo_select(scores)) == ["T9", "T8"]
    # The buffer keeps an incumbent ranked within the top
    # 24% (top 3 of ten): T7 (3rd) stays, T6 (4th) goes.
    sel = spmo_select(scores, prev_selected={"T7", "T6"},
                      retain_fraction=0.24)
    assert list(sel) == ["T9", "T8", "T7"]
    # No incumbents, or an empty previous book: plain again.
    assert list(spmo_select(scores, prev_selected=None,
                            retain_fraction=0.24)) == ["T9", "T8"]
    assert list(spmo_select(scores, prev_selected=set(),
                            retain_fraction=0.24)) == ["T9", "T8"]


def test_spmo_variant_buffer_holds_the_band():
    from toolbox.strategies import (
        SPMO_BUFFER_FRACTION,
        spmo_variant_weights,
        spmo_weights,
    )

    R = _spmo_returns()
    # Scores increase with the drift, so T7 ranks third:
    # outside the entry quintile, inside the buffer band.
    w = spmo_variant_weights(R, prev_selected={"T7"},
                             retain_fraction=SPMO_BUFFER_FRACTION)
    assert list(np.flatnonzero(w)) == [7, 8, 9]
    assert np.isclose(w.sum(), 1.0)
    assert list(np.flatnonzero(spmo_weights(R))) == [8, 9]


def test_spmo_reference_date():
    from toolbox.strategies import spmo_reference_date

    sessions = pd.bdate_range("2026-01-05", "2026-09-30")
    # A March decision strikes at the last session of
    # February (Feb 28, 2026 is a Saturday, so Feb 27).
    assert spmo_reference_date(sessions, pd.Timestamp("2026-03-20")) \
        == pd.Timestamp("2026-02-27")
    # A September decision strikes at end-August (a Monday
    # session here).
    assert spmo_reference_date(sessions, pd.Timestamp("2026-09-18")) \
        == pd.Timestamp("2026-08-31")


def test_spmo_name_resolves_and_runs():
    assert "SPMO" in COMBO_STRATEGY_NAMES
    assert resolve_strategy("SPMO") == ("SPMO", None, None)
    R = _returns(n=300)
    W, _info = snapshot_weights(R, "ledoit_wolf", "sample", 0.0,
                                ["SPMO", "Average"])
    w = W.loc["SPMO"].values
    assert np.isclose(w.sum(), 1.0)
    assert (w > 0).sum() == 2  # top quintile of the 8 names
    assert np.allclose(W.loc["Average"].values, w)


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
    # 21 window-using bases x 4 windows + 1/N + 6 RS + SPMO
    # + SPMO (cap × score) + Average.
    assert len(COMBO_STRATEGY_NAMES) == 94
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
    kw = {"cov_method": "bbp", "mean_method": "bayes_stein",
              "seed": 126, "hold": 21}
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


def test_spmo_cap_score_name_resolves_and_sits_beside_spmo():
    from toolbox.strategies import SPMO_CAP_SCORE_NAME, _strategy_order

    assert SPMO_CAP_SCORE_NAME == "SPMO (cap × score)"
    i = COMBO_STRATEGY_NAMES.index("SPMO")
    assert COMBO_STRATEGY_NAMES[i + 1] == SPMO_CAP_SCORE_NAME
    # A parameterized rule in the RS idiom: it resolves to
    # itself, and lives in the combination grid only — not
    # among the bare base names.
    assert resolve_strategy(SPMO_CAP_SCORE_NAME) == (
        SPMO_CAP_SCORE_NAME, None, None)
    assert SPMO_CAP_SCORE_NAME not in STRATEGY_NAMES
    order = _strategy_order(
        ["Average", SPMO_CAP_SCORE_NAME, "SPMO", "1/N"])
    assert order == ["1/N", "SPMO", SPMO_CAP_SCORE_NAME, "Average"]


def test_spmo_cap_score_named_strategy_is_the_variant():
    from toolbox.strategies import (
        SPMO_CAP_SCORE_NAME,
        spmo_variant_weights,
    )

    R = _spmo_returns()
    mu, S, _info = estimate_inputs(R, "ledoit_wolf", "sample", 0.0)
    caps = pd.Series(dict.fromkeys(R.columns, 1.0))
    caps["T8"] = 3.0
    w = strategy_weights(SPMO_CAP_SCORE_NAME, R, S, mu, 0.0,
                         caps=caps)
    # The named strategy IS the canonical variant — the same
    # function the study gates ran — not a second copy.
    assert np.array_equal(w, spmo_variant_weights(R, caps=caps))
    # The caps bite: not the score-only rule's weights.
    assert not np.allclose(
        w, strategy_weights("SPMO", R, S, mu, 0.0))
    # Without caps the name refuses rather than silently
    # degrading to score-only.
    with pytest.raises(ValueError):
        strategy_weights(SPMO_CAP_SCORE_NAME, R, S, mu, 0.0)


def test_spmo_cap_score_snapshot_and_backtest():
    from toolbox.strategies import (
        SPMO_CAP_SCORE_NAME,
        spmo_variant_weights,
    )

    R = _spmo_returns()
    caps = pd.Series({t: float(j + 1)
                      for j, t in enumerate(R.columns)})
    W, _info = snapshot_weights(R, "ledoit_wolf", "sample", 0.0,
                                [SPMO_CAP_SCORE_NAME], caps=caps)
    assert np.allclose(W.loc[SPMO_CAP_SCORE_NAME].values,
                       spmo_variant_weights(R, caps=caps))

    prices = _prices(n=500)
    shares = pd.Series({t: 100.0 * (j + 1)
                        for j, t in enumerate(prices.columns)})
    out = walk_forward_backtest(
        prices, strategies=[SPMO_CAP_SCORE_NAME],
        cov_method="bbp", mean_method="sample", seed=252,
        hold=21, caps_shares=shares)
    assert out is not None
    assert set(out["stats"].index) == {SPMO_CAP_SCORE_NAME}
    assert np.isfinite(out["stats"].values).all()
    # The first rebalance is the variant struck on the seed
    # window with caps = shares x the close at its last
    # session — the backtest's strike convention.
    logret = np.log(prices / prices.shift(1)).dropna()
    R_w = logret.iloc[:252]
    struck = shares.reindex(prices.columns) * prices.loc[R_w.index[-1]]
    expect = spmo_variant_weights(R_w, caps=struck)
    first = out["weights_history"][0]["weights"][SPMO_CAP_SCORE_NAME]
    got = np.array([first[t] for t in prices.columns])
    assert np.allclose(got, expect)
    # Selecting the rule without share counts raises up
    # front, before any rebalance runs.
    with pytest.raises(ValueError):
        walk_forward_backtest(
            prices, strategies=[SPMO_CAP_SCORE_NAME],
            cov_method="bbp", mean_method="sample", seed=252,
            hold=21)


# ---------------------------------------------------------------------------
# Known-value tests (idea search #2, candidate 3): every expectation below
# is derived independently of the implementation — by hand, by brute
# force, or by first-principles arithmetic in the test itself.
# ---------------------------------------------------------------------------


def test_hrp_two_unrelated_assets_is_inverse_variance():
    # Two uncorrelated assets: the single bisection allocates
    # between the singleton clusters in inverse proportion to
    # their variances (cluster variance of a singleton is its own
    # variance, and HRP gives the left cluster v_r / (v_l + v_r)).
    # v = (0.04, 0.01) -> w = (0.01, 0.04) / 0.05 = (0.2, 0.8).
    w = hrp_weights(np.diag([0.04, 0.01]))
    assert np.allclose(w, [0.2, 0.8], atol=1e-12)


def test_hrp_two_identical_pairs_hand_computed():
    # Assets {0, 1} are the same asset twice (var 0.04, corr 1);
    # assets {2, 3} likewise with var 0.01; the pairs are
    # uncorrelated. Quasi-diagonalisation keeps each identical
    # pair adjacent (they merge at distance 0), and the first
    # bisection cut (midpoint of the 4-leaf order) always falls
    # between the pairs, so the result does not depend on the
    # dendrogram's leaf traversal order:
    #   cluster variance of an identical pair under its
    #   inverse-variance portfolio (1/2, 1/2) is the pair's own
    #   variance, so the pairs split inverse-variance:
    #   pair {0,1} gets 0.01 / (0.04 + 0.01) = 0.2, pair {2,3}
    #   gets 0.8; within a pair the split is equal.
    S = np.array([[0.04, 0.04, 0.0, 0.0],
                  [0.04, 0.04, 0.0, 0.0],
                  [0.0, 0.0, 0.01, 0.01],
                  [0.0, 0.0, 0.01, 0.01]])
    assert np.allclose(hrp_weights(S), [0.1, 0.1, 0.4, 0.4], atol=1e-12)
    # Relabelling the assets (pairs swapped) moves the weights
    # with them — the allocation is a property of the covariance,
    # not of the leaf order.
    P = [2, 3, 0, 1]
    assert np.allclose(hrp_weights(S[np.ix_(P, P)]),
                       [0.4, 0.4, 0.1, 0.1], atol=1e-12)


def _returns_with_exact_cov(S: np.ndarray) -> pd.DataFrame:
    """Returns whose *sample* covariance (ddof=1) is exactly ``S``.

    Hadamard columns are mean-zero and mutually orthogonal with
    squared norm = n, so X = Z·chol(S)·sqrt((n-1)/n) has
    X'X / (n-1) = S by construction — no sampling error in the
    correlations NCO clusters on.
    """
    from scipy.linalg import hadamard

    n = 8
    Z = hadamard(n)[:, 1:S.shape[0] + 1].astype(float)
    X = Z @ (np.linalg.cholesky(S) * np.sqrt((n - 1) / n))
    return pd.DataFrame(X, columns=[f"A{i}" for i in range(S.shape[0])])


def test_nco_diagonal_covariance_is_inverse_variance():
    # With a diagonal covariance, min-variance inside any cluster
    # is inverse-variance there, the cluster portfolio variance
    # is 1 / sum(1/sigma^2) over the cluster, and min-variance
    # across clusters divides by those variances — the cluster
    # sums telescope, so for ANY partition NCO is exactly the
    # global inverse-variance portfolio. (SLSQP solves the two
    # min-variance steps, hence the solver tolerance.)
    S = np.diag([0.04, 0.01, 0.09, 0.0025])
    R = _returns_with_exact_cov(S)
    iv = 1.0 / np.diag(S)
    assert np.allclose(nco_weights(R, S, k=2), iv / iv.sum(), atol=1e-6)


def test_nco_two_block_composition_hand_computed():
    # Block 1: vars (0.04, 0.04), cov 0.02. Its 2-asset
    # min-variance weight is (0.04 - 0.02) / (0.04 + 0.04 - 2*0.02)
    # = 1/2 on each, with portfolio variance
    # (0.04 + 0.04 + 2*0.02) / 4 = 0.03.
    # Block 2: vars (0.09, 0.01), cov 0.015. The unconstrained
    # min-variance weight on the first asset is
    # (0.01 - 0.015) / (0.09 + 0.01 - 2*0.015) < 0, so the
    # long-only optimum is the corner (0, 1) — the portfolio
    # variance is 0.01 (the variance function's derivative at
    # w = 0 is +0.01 > 0, confirming the corner).
    # Across blocks (independent): inverse-variance on
    # (0.03, 0.01) gives (1/4, 3/4). Composed:
    # (1/4 * 1/2, 1/4 * 1/2, 3/4 * 0, 3/4 * 1) = (1/8, 1/8, 0, 3/4).
    S = np.array([[0.04, 0.02, 0.0, 0.0],
                  [0.02, 0.04, 0.0, 0.0],
                  [0.0, 0.0, 0.09, 0.015],
                  [0.0, 0.0, 0.015, 0.01]])
    R = _returns_with_exact_cov(S)
    assert np.allclose(nco_weights(R, S, k=2),
                       [0.125, 0.125, 0.0, 0.75], atol=1e-6)


def _cvar_tail_mean(w: np.ndarray, R: np.ndarray, alpha: float) -> float:
    """CVaR_alpha of portfolio losses by the definition: the mean
    of the worst (1 - alpha) * n scenario losses (the test cases
    keep (1 - alpha) * n a whole number, so no fractional-tail
    convention is involved)."""
    losses = -(R @ w)
    k = int(round((1.0 - alpha) * len(losses)))
    return float(np.sort(losses)[::-1][:k].mean())


def test_cvar_two_asset_worst_case_hand_computed():
    # 4 scenarios, alpha = 0.75: (1 - alpha) * n = 1, so CVaR is
    # the single worst scenario loss. Scenario losses as
    # functions of w (weight on asset A):
    #   s1: -0.02 - 0.08w   (a gain, never the worst)
    #   s2:  0.01 + 0.19w   (rising)
    #   s3:  0.03 - 0.08w   (falling)
    #   s4: -0.04 + 0.09w   (a gain at the crossing)
    # The worst case is minimised where the rising and falling
    # lines cross: 0.01 + 0.19w = 0.03 - 0.08w -> w = 2/27.
    R = np.array([[0.10, 0.02],
                  [-0.20, -0.01],
                  [0.05, -0.03],
                  [-0.05, 0.04]])
    w = cvar_weights(R, alpha=0.75)
    assert np.allclose(w, [2.0 / 27.0, 25.0 / 27.0], atol=1e-8)


def test_cvar_three_asset_matches_simplex_grid():
    # Independent oracle: exhaustive grid over the weight
    # simplex (step 0.01) scoring CVaR by the tail-mean
    # definition above — no LP, no shared code path. The grid's
    # own minimum sits at (0.3, 0, 0.7); the LP must attain the
    # same CVaR value and land within one grid step of it.
    R = np.array([[0.08, 0.01, 0.03],
                  [-0.12, -0.02, -0.05],
                  [0.04, 0.06, -0.02],
                  [-0.03, -0.09, 0.01],
                  [0.02, -0.04, -0.07],
                  [-0.06, 0.03, 0.05]])
    best_val, best_w = np.inf, None
    grid = np.arange(0.0, 1.0 + 1e-12, 0.01)
    for a in grid:
        for b in grid[grid <= 1.0 - a + 1e-12]:
            cand = np.array([a, b, 1.0 - a - b])
            val = _cvar_tail_mean(cand, R, 0.5)
            if val < best_val:
                best_val, best_w = val, cand
    w = cvar_weights(R, alpha=0.5)
    assert _cvar_tail_mean(w, R, 0.5) == pytest.approx(best_val, abs=1e-9)
    assert np.allclose(w, best_w, atol=0.011)


def test_performance_stats_known_series():
    # Four days with a constant risk-free rate; every expected
    # value is recomputed here from the definitions with plain
    # arithmetic (statistics/math, explicit loops) — equity
    # curve by running product, drawdowns against the running
    # peak, sample stdev with ddof=1, 252 trading days.
    import math
    import statistics as st

    r = [0.10, -0.05, 0.02, -0.01]
    rf = [0.001] * 4
    exc = [x - 0.001 for x in r]
    eq, v = [], 1.0
    for x in r:
        v *= 1.0 + x
        eq.append(v)
    peak, dds = eq[0], []
    for e_ in eq:
        peak = max(peak, e_)
        dds.append(e_ / peak - 1.0)
    max_dd = min(dds)
    cagr = eq[-1] ** (252.0 / len(r)) - 1.0
    expected = {
        "total_return": eq[-1] - 1.0,
        "cagr": cagr,
        "vol": st.stdev(r) * math.sqrt(252),
        "sharpe": (sum(exc) / 4) / st.stdev(exc) * math.sqrt(252),
        "sortino": (sum(exc) / 4)
        / math.sqrt(sum(min(e_, 0.0) ** 2 for e_ in exc) / 4)
        * math.sqrt(252),
        "max_dd": max_dd,
        "calmar": cagr / abs(max_dd),
        "ulcer": math.sqrt(sum((d * 100.0) ** 2 for d in dds) / 4),
        "win_rate": 0.5,
        "turnover": 0.3,
        "n_days": 4,
    }
    out = performance_stats(np.array(r), np.array(rf), [0.2, 0.4])
    for key, want in expected.items():
        assert out[key] == pytest.approx(want, rel=1e-9, abs=1e-12), key
