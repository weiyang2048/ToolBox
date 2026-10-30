"""Tests for toolbox.DataZoo.market.futures_board (network-free)."""

import datetime

from toolbox.DataZoo.market import futures_board as fb


def test_board_layout():
    groups = fb.all_groups()
    assert [g for g, _c, _k in groups] == [
        "US Equity",
        "International equity — Americas",
        "International equity — Europe",
        "International equity — Asia-Pacific",
        "Energy", "Crypto", "Metals", "Agriculture", "Rates", "FX",
    ]
    assert sum(len(c) for _g, c, _k in groups) == 48
    kinds = {k for _g, _c, k in groups}
    assert kinds == {"future", "cash"}


def test_cash_dec():
    assert fb.cash_dec(None) == 2
    assert fb.cash_dec(204_000) == 0
    assert fb.cash_dec(5_400) == 1
    assert fb.cash_dec(950) == 2


def test_shade_zero_endpoints_and_midpoints():
    assert fb.shade_zero(None, 10) == "FFFFFF"
    assert fb.shade_zero(5, 0) == "FFFFFF"
    assert fb.shade_zero(0, 10) == "FFFFFF"
    assert fb.shade_zero(10, 10) == "63BE7B"     # full green
    assert fb.shade_zero(-10, 10) == "F8696B"    # full red
    assert fb.shade_zero(99, 10) == "63BE7B"     # clamped
    assert fb.shade_zero(-99, 10) == "F8696B"
    # Half-scale: exact midpoint of white->green / white->red.
    assert fb.shade_zero(5, 10) == "B1DEBD"
    assert fb.shade_zero(-5, 10) == "FCB4B5"
    # The negative branch must lighten toward white as |v| shrinks —
    # the regression a drifted copy once introduced ran it backwards.
    small = fb.shade_zero(-1, 10)
    assert small > fb.shade_zero(-5, 10) > fb.shade_zero(-10, 10)


def _fake_fetch_board(monkeypatch, closes_by_sym):
    def fake_fetch(sym):
        closes = closes_by_sym.get(sym)
        if closes is None:
            return None
        return closes[-1] * 1.01, closes[-1], 1_700_000_000

    def fake_history(sym):
        return closes_by_sym.get(sym) or []

    monkeypatch.setattr(fb, "fetch", fake_fetch)
    monkeypatch.setattr(fb, "history", fake_history)
    return fb.fetch_board(now=datetime.datetime(2026, 10, 29, 12, 0))


def test_fetch_board_assembly(monkeypatch):
    closes = {sym: [100.0] * 300
              for _g, contracts, _k in fb.all_groups()
              for sym, _l, _d in contracts}
    closes["ES=F"] = [100.0] * 299 + [110.0]   # +10% week, +10% day base
    del closes["ETH=F"]                        # unavailable -> null row
    rep = _fake_fetch_board(monkeypatch, closes)
    assert rep["available"] == 47
    assert list(rep["data"]) == [g for g, _c, _k in fb.all_groups()]
    es = rep["data"]["US Equity"][0]
    assert es["label"] == "S&P 500 (ES)"
    assert abs(es["pct"] - 1.0) < 1e-9         # last = settle * 1.01
    assert abs(es["week"] - 10.0) < 1e-9       # close[-1]/close[-6]
    assert es["y10"] is None                   # 300 closes < 2520
    eth = rep["data"]["Crypto"][1]
    assert eth["last"] is None and eth["pct"] is None
    assert eth["label"] == "Ether CME (ETH)"   # null row keeps its slot


def test_board_scales():
    data = {"G": [{"pct": -4.0, "week": 8.0, "month": None,
                   "quarter": -2.0, "year": 1.0, "y3": None,
                   "y5": None, "y10": None},
                  {"pct": 2.0, "week": -3.0, "month": 5.0,
                   "quarter": 1.0, "year": -6.0, "y3": None,
                   "y5": None, "y10": None}]}
    day, scales = fb.board_scales(data)
    assert day == 4.0
    assert scales == {"week": 8.0, "month": 5.0, "quarter": 2.0,
                      "year": 6.0, "y3": 0, "y5": 0, "y10": 0}


def test_multiyear_horizons_are_cagr(monkeypatch):
    """Year-and-longer horizons annualize (user request 2026-10-29):
    a series up exactly 1.1**10 over its full span reads y10 = +10%
    CAGR (not +159% total), y5 = +21%, and year (the 1-year span,
    CAGR == total) at the full total change."""
    closes = {sym: [100.0] * 2521
              for _g, contracts, _k in fb.all_groups()
              for sym, _label, _dec in contracts}
    closes["ES=F"] = [100.0] * 2520 + [100.0 * 1.1 ** 10]
    rep = _fake_fetch_board(monkeypatch, closes)
    es = rep["data"]["US Equity"][0]
    assert abs(es["y10"] - 10.0) < 1e-6
    assert abs(es["y5"] - 21.0) < 1e-6
    assert abs(es["year"] - 100.0 * (1.1 ** 10 - 1.0)) < 1e-6
    assert abs(es["week"] - 100.0 * (1.1 ** 10 - 1.0)) < 1e-6


def test_ls_states_rising_series_all_long():
    from toolbox.DataZoo.market.futures_board import ls_states
    closes = [100.0 * 1.001 ** i for i in range(300)]
    assert ls_states(closes) == (True, True, True)


def test_ls_states_falling_series_all_short():
    from toolbox.DataZoo.market.futures_board import ls_states
    closes = [100.0 * 0.999 ** i for i in range(300)]
    assert ls_states(closes) == (False, False, False)


def test_ls_states_length_gates():
    from toolbox.DataZoo.market.futures_board import ls_states
    rising = [100.0 * 1.001 ** i for i in range(300)]
    assert ls_states(rising[:10]) == (None, None, None)
    assert ls_states(rising[:30]) == (True, None, None)
    assert ls_states(rising[:120]) == (True, True, None)
    assert ls_states([]) == (None, None, None)
