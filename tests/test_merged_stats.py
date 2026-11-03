"""merged_stats: the one stats table for the Strategies page and
the briefs (user direction 2026-11-02).

(a) On the real Global-sectors 10y frame, the asset_stats columns
of merged_stats equal asset_stats' own outputs and rebased equals
the Strategies page's arithmetic.
(b) The briefs' columns cross-check against panel-stats.py's
stats_for (Flaco repo) on the sector universe — windowed columns
against movement-tracker's unrounded _end_label_stats on the same
window and against stats_for's rounded rows, trend states against
stats_for on the full history, win_rate against winrate.py's
period_returns machinery.
Plus an offline synthetic shape test. Network tests skip when the
Yahoo pull fails.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from toolbox.DataZoo.market.strategies import asset_stats, merged_stats
from toolbox.DataZoo.market.yahoo_prices import download_close_prices

SECTORS = ["EXI", "IXC", "IXG", "IXJ", "IXN", "IXP",
           "JXI", "KXI", "MXI", "REET", "RXI"]
A_COLS = ["last", "mu_ann", "vol_ann", "sharpe0", "momentum_12_1",
          "max_dd", "beta", "n_obs"]
ALL_COLS = ["rebased", "day_chg", "wk_chg", "mo_chg", "q_chg",
            "ytd_chg", "yr_chg", "px_ema20", "ema_long", "ema_regime", "mu_ann",
            "vol_ann", "vol", "sharpe0", "sharpe",
            "ulcer", "win_rate", "dd", "dd_q", "max_dd", "beta"]
# asset_stats columns the merged table drops (removal passes 1-2,
# user direction 2026-11-02).
DROPPED = ["last", "momentum_12_1", "n_obs"]

_FLACO_MB = (Path.home() / "workspace" / "flaco-repo" / "scripts"
             / "morning-briefing")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, _FLACO_MB / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_merged_stats_shape_offline():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2023-01-02", periods=300)
    px = pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))
         for t in ("AAA", "BBB")}, index=idx)
    out = merged_stats(px)
    assert list(out.columns) == ALL_COLS
    for dropped in DROPPED:
        assert dropped not in out.columns
    assert list(out.index) == ["AAA", "BBB"]
    assert pd.isna(out.loc["AAA", "win_rate"])  # no benchmark frame
    assert pd.isna(out.loc["AAA", "beta"])      # no benchmark series
    assert out.loc["AAA", "ema_regime"] is not None  # 300 sessions


def test_merged_stats_yr_chg_offline():
    # The one-year change: trailing 252 sessions on the full
    # history; NaN under 253 closes.
    idx = pd.bdate_range("2023-01-02", periods=300)
    px = pd.DataFrame(
        {"CCC": 100.0 * 1.001 ** np.arange(len(idx))}, index=idx)
    out = merged_stats(px)
    assert out.loc["CCC", "yr_chg"] == pytest.approx(
        1.001 ** 252 - 1.0)
    short = merged_stats(px.iloc[:100])
    assert pd.isna(short.loc["CCC", "yr_chg"])


@pytest.fixture(scope="module")
def sectors_10y():
    px = download_close_prices(SECTORS + ["SPY"], period="10y")
    if px.empty or len(px) < 500:
        pytest.skip("Yahoo 10y pull unavailable")
    return px


def test_a_columns_match_asset_stats(sectors_10y):
    frame = sectors_10y[SECTORS]
    bench = sectors_10y["SPY"]
    merged = merged_stats(frame, benchmark=bench)
    base = asset_stats(frame, bench)
    for col in [c for c in A_COLS if c not in DROPPED]:
        np.testing.assert_allclose(
            merged[col].astype(float), base[col].astype(float),
            rtol=0, atol=1e-9, err_msg=col)
    # Rebased anchor (canonical, shared with merged_stats): first
    # VALID print per column. The raw frame's first row is NaN for
    # tickers whose history starts a row late (8 of the 11 sector
    # ETFs on the current 10y pull), so a frame.iloc[0] anchor makes
    # one side NaN and the comparison flip between runs.
    page_rebased = frame.apply(
        lambda s: s.dropna().iloc[-1] / s.dropna().iloc[0] * 100.0)
    for t in SECTORS:
        assert abs(float(merged.loc[t, "rebased"])
                   - float(page_rebased[t])) < 1e-9


@pytest.fixture(scope="module")
def panel():
    if not (_FLACO_MB / "panel-stats.py").exists():
        pytest.skip("Flaco repo not present (cross-repo check, "
                    "local only)")
    ps = _load("panel_stats", "panel-stats.py")
    frame = ps.fetch_all()
    if frame.empty:
        pytest.skip("panel-stats 3y pull unavailable")
    series = ps.universe_series(frame, ps.SECTORS)
    stats, ws, we = ps.stats_for(series)
    if not stats:
        pytest.skip("panel-stats produced no sector stats")
    return ps, series, stats, ws, we


def test_b_window_columns_match_panel_stats(panel):
    ps, series, stats, ws, we = panel
    mt = ps._mt
    windowed = pd.DataFrame(
        {t: s[(s.index >= ws) & (s.index <= we)]
         for t, s in series.items()})
    rf = mt._irx_yield_history()
    merged = merged_stats(windowed, rf_annual_pct=rf)
    # stats_for returns PERCENTS, rounded; merged returns fractions.
    rounded_tol = {"rebased": 0.05 + 1e-9, "day_chg": 0.005 + 1e-9,
                   "wk_chg": 0.005 + 1e-9, "mo_chg": 0.005 + 1e-9,
                   "q_chg": 0.005 + 1e-9, "ytd_chg": 0.005 + 1e-9,
                   "vol": 0.05 + 1e-9, "sharpe": 0.005 + 1e-9,
                   "ulcer": 0.05 + 1e-9, "dd": 0.05 + 1e-9,
                   "dd_q": 0.005 + 1e-9}
    pct_cols = {"day_chg", "wk_chg", "mo_chg", "q_chg", "ytd_chg",
                "vol", "ulcer", "dd"}  # stats_for stores these in %
    for t, row in stats.items():
        w = windowed[t].dropna()
        (_r, day, wk, mo, q, vol, dd, dd_q, ytd, sharpe, ulcer) = \
            mt._end_label_stats(w)
        exact = {"day_chg": day / 100, "wk_chg": wk / 100,
                 "mo_chg": mo / 100, "q_chg": q / 100,
                 "ytd_chg": ytd / 100, "vol": vol / 100,
                 "sharpe": sharpe, "ulcer": ulcer / 100,
                 "dd": dd / 100, "dd_q": dd_q}
        for col, want in exact.items():
            got = float(merged.loc[t, col])
            if np.isnan(want):
                assert np.isnan(got), (t, col, got, want)
            else:
                assert abs(got - want) < 1e-9, (t, col, got, want)
        for col, tol in rounded_tol.items():
            want = row[col]
            got = float(merged.loc[t, col]) * (100.0 if col in pct_cols
                                               else 1.0)
            if np.isnan(want):
                assert np.isnan(got), (t, col, got, want)
            else:
                assert abs(got - want) <= tol, (t, col, got, want)


def test_b_trend_states_match_panel_stats(panel):
    ps, series, stats, ws, we = panel
    full = pd.DataFrame({t: s[s.index <= we] for t, s in series.items()})
    merged = merged_stats(full)
    for t, row in stats.items():
        for col in ("px_ema20", "ema_long", "ema_regime"):
            assert merged.loc[t, col] == row[col], (t, col)


def test_b_win_rate_matches_winrate_machinery(panel):
    ps, series, stats, ws, we = panel
    wr = _load("winrate", "winrate.py")
    # panel-stats' own pull carries VEA/VWO (region universe) but
    # not VT; the briefs' win-rate benchmarks come from winrate.py's
    # own fetches — pull the three directly, cleaned the same way.
    bframe = download_close_prices(["VT", "VEA", "VWO"], period="3y")
    if bframe.empty:
        pytest.skip("win-rate benchmark pull unavailable")
    benches = {}
    for t in ("VT", "VEA", "VWO"):
        s = bframe[t].dropna()
        s.index = pd.to_datetime(s.index).normalize()
        benches[t] = s
    bench_frame = pd.DataFrame(benches)
    windowed = pd.DataFrame(
        {t: s[(s.index >= ws) & (s.index <= we)]
         for t, s in series.items()})
    merged = merged_stats(windowed, win_benchmarks=bench_frame)
    for t in stats:
        w = windowed[t].dropna()
        rates = []
        for b in ("VT", "VEA", "VWO"):
            bw = benches[b]
            bw = bw[(bw.index >= ws) & (bw.index <= we)].dropna()
            a, v = wr.period_returns(w, "weekly").align(
                wr.period_returns(bw, "weekly"), join="inner")
            valid = a.notna() & v.notna()
            rates.append(float((a[valid] > v[valid]).mean()))
        want = float(np.mean(rates))
        got = float(merged.loc[t, "win_rate"])
        assert abs(got - want) < 1e-9, (t, got, want)
