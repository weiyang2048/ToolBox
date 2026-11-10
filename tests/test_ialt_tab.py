"""IALT tab: since-inception compute, render hygiene, and the
flaky-leg fetch discipline (mirrors the other tabs' tests)."""
import numpy as np
import pandas as pd
import pytest

from toolbox.DataZoo.market import ialt_tab
from toolbox.DataZoo.market.ialt_tab import (
    _compute,
    chart_figs,
    fetch_ialt_tab,
    ialt_foot_html,
    ialt_head_html,
)


def _frame(n=400, seed=11):
    """Synthetic aligned history: IALT shares half of SPY's
    daily move plus its own drift; AGG and the peers wander
    independently."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-02", periods=n)
    spy_r = rng.normal(0.0004, 0.010, n)
    cols = {
        "SPY": 100 * np.cumprod(1 + spy_r),
        "VOO": 100 * np.cumprod(
            1 + spy_r + rng.normal(0.0, 0.0005, n)),
        "IALT": 100 * np.cumprod(
            1 + 0.5 * spy_r + rng.normal(0.0006, 0.004, n)),
        "AGG": 100 * np.cumprod(1 + rng.normal(0.0, 0.002, n)),
    }
    for sym in ialt_tab.PEERS:
        cols[sym] = 100 * np.cumprod(
            1 + rng.normal(0.0002, 0.006, n))
    return pd.DataFrame(cols, index=idx)


def test_compute_since_inception_stats():
    p = _compute(_frame())
    assert p["ret"]["ialt"] > p["ret"]["spy"] > 0
    assert 0.3 < p["corr_spy"] < 0.8
    assert abs(p["corr_agg"]) < 0.3
    assert p["max_dd"] <= 0 and p["cur_dd"] <= 0
    assert p["vol"] > 0 and p["sharpe"] > 0
    assert p["inception"] == "Jan 2, 2025"
    assert p["last_date"] == p["dates"][-1]
    assert len(p["paths"]["ialt"]) == 400
    assert p["paths"]["ialt"][0] == 100.0
    assert len(p["roll"]["spy"]) == len(p["roll_dates"]) > 300
    assert abs(p["roll_last"]["spy"] - round(p["roll"]["spy"][-1], 2)) < 1e-9
    # The VOO comparison (user direction 2026-11-07): a
    # full-length drawdown series, deeper than IALT's on
    # this frame (VOO takes SPY's full move, IALT half).
    assert len(p["dd_voo"]) == 400
    assert max(p["dd_voo"]) <= 0
    assert p["voo_max_dd"] == round(min(p["dd_voo"]), 1)
    assert p["voo_cur_dd"] == round(p["dd_voo"][-1], 1)
    assert p["voo_max_dd"] < p["max_dd"]


def test_head_chips_and_pill_and_foot():
    p = _compute(_frame())
    out = ialt_head_html(p)
    assert out.startswith(
        '<div style="max-width:1120px;margin:0 auto">')
    assert "Since inception" in out
    assert "1 session behind" not in out  # no board date, no pill
    lag = ialt_head_html(p, board_date="2026-12-31")
    assert "behind the board" in lag
    assert "opening chapter" in ialt_foot_html()


def test_chart_figs_layout_and_warp():
    # The Plotly panels (user direction 2026-11-10): the main
    # panel stands tallest, legends are the short names, the
    # warped seam sits at 1 - TAIL_SHARE, and the last 30
    # days stretch — the final month of x positions spans
    # ~TAIL_SHARE of the axis while covering ~30/400 of the
    # window's days.
    p = _compute(_frame())
    figs = chart_figs(p)
    assert [t.split(" — ")[0] for t, _f in figs] == [
        "Since inception (Jan 2, 2025)",
        "The diversifier's ride",
        "Rolling 63-day correlation"]
    heights = [f.layout.height for _t, f in figs]
    assert heights == [560, 280, 280]
    main = figs[0][1]
    names = [tr.name for tr in main.data if tr.name]
    assert names == ["IALT", "SPY", "AGG", "Peers"]
    seams = [s for s in main.layout.shapes
             if s.x0 == 1 - ialt_tab.TAIL_SHARE]
    assert len(seams) == 1
    xs = main.data[0].x
    assert xs[0] == 0.0 and xs[-1] == 1.0
    # ~30 of ~280 business days in the tail, at 25% width.
    assert xs[-22] < 0.78 < xs[-18]
    dd_names = [tr.name for tr in figs[1][1].data if tr.name]
    assert dd_names == ["IALT", "VOO"]


def test_chart_figs_tolerates_a_pre_voo_payload():
    # A last-good payload cached before the VOO series
    # existed draws the drawdown panel IALT-only.
    p = _compute(_frame())
    for k in ("dd_voo", "voo_max_dd", "voo_cur_dd"):
        del p[k]
    figs = chart_figs(p)
    dd_names = [tr.name for tr in figs[1][1].data if tr.name]
    assert dd_names == ["IALT"]


def test_fetch_retries_a_flaky_leg(monkeypatch):
    frame = _frame()
    calls = {}

    def fake_daily(sym, p1=0, p2=None, include_partial=False):
        calls[sym] = calls.get(sym, 0) + 1
        if sym == "BTAL" and calls[sym] == 1:
            from urllib.error import HTTPError
            raise HTTPError("u", 429, "Too Many Requests", None, None)
        return frame[sym]

    monkeypatch.setattr(ialt_tab, "yahoo_daily", fake_daily)
    monkeypatch.setattr(ialt_tab.time, "sleep", lambda s: None)
    p = fetch_ialt_tab()
    assert calls["BTAL"] == 2
    assert p["ret"]["ialt"] > 0


def test_fetch_names_a_failing_leg(monkeypatch):
    frame = _frame()

    def dead(sym, p1=0, p2=None, include_partial=False):
        if sym == "AGG":
            raise RuntimeError("boom")
        return frame[sym]

    monkeypatch.setattr(ialt_tab, "yahoo_daily", dead)
    with pytest.raises(Exception):
        fetch_ialt_tab()
