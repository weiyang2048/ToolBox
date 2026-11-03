"""IALT tab: since-inception compute, render hygiene, and the
flaky-leg fetch discipline (mirrors the other tabs' tests)."""
import numpy as np
import pandas as pd
import pytest

from toolbox.DataZoo.market import ialt_tab
from toolbox.DataZoo.market.ialt_tab import (
    _compute,
    fetch_ialt_tab,
    render_ialt_tab,
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


def test_render_panels_chips_and_pill():
    p = _compute(_frame())
    out = render_ialt_tab(p)
    assert out.startswith(
        '<div style="max-width:1120px;margin:0 auto">')
    assert out.count("<svg") == 3
    assert "Since inception" in out and "opening chapter" in out
    assert "1 session behind" not in out  # no board date, no pill
    lag = render_ialt_tab(p, board_date="2026-12-31")
    assert "behind the board" in lag


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
