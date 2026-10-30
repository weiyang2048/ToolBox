"""Tests for toolbox.DataZoo.market.futures_board (network-free)."""

import datetime

from toolbox.DataZoo.market import futures_board as fb


def test_board_layout():
    groups = fb.all_groups()
    assert [g for g, _c, _k in groups] == [
        "Equity index",
        "International equity — Asia-Pacific",
        "International equity — Europe",
        "International equity — Americas",
        "Energy", "Metals", "Agriculture", "Rates", "FX", "Crypto",
    ]
    assert sum(len(c) for _g, c, _k in groups) == 39
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
    assert rep["available"] == 38
    assert list(rep["data"]) == [g for g, _c, _k in fb.all_groups()]
    es = rep["data"]["Equity index"][0]
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
