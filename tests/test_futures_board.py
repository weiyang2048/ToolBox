"""Tests for toolbox.DataZoo.market.futures_board (network-free)."""

import datetime

from toolbox.DataZoo.market import futures_board as fb


def test_board_layout():
    groups = fb.all_groups()
    assert [g for g, _c in groups] == [
        "US Equity",
        "Global sectors",
        "Asia-Pacific", "Europe", "Americas",
        "Crypto", "Metals", "Energy", "Grains & oilseeds",
        "Livestock", "Softs", "Rates", "FX",
    ]
    assert sum(len(c) for _g, c in groups) == 67
    kinds = {k for _g, c in groups for _s, _l, _d, k in c}
    assert kinds == {"future", "cash"}
    us = dict(groups)["US Equity"]
    assert len(us) == 4
    assert all(k == "future" for *_x, k in us)
    gs = dict(groups)["Global sectors"]
    assert len(gs) == 11
    assert all(k == "cash" for *_x, k in gs)
    gd = dict(groups)
    assert [len(gd[g]) for g in ("Asia-Pacific", "Europe",
                                 "Americas")] == [8, 4, 3]
    assert gd["Asia-Pacific"][-1][0] == "^AXJO"
    assert all(k == "cash" for g in ("Asia-Pacific", "Europe",
                                     "Americas")
               for *_x, k in gd[g])


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
              for _g, contracts in fb.all_groups()
              for sym, _l, _d, _k in contracts}
    closes["ES=F"] = [100.0] * 299 + [110.0]   # +10% week, +10% day base
    del closes["ETH=F"]                        # unavailable -> null row
    rep = _fake_fetch_board(monkeypatch, closes)
    assert rep["available"] == 66
    assert list(rep["data"]) == [g for g, _c in fb.all_groups()]
    es = rep["data"]["US Equity"][0]
    assert es["label"] == "S&P 500 (ES)"
    assert abs(es["pct"] - 1.0) < 1e-9         # last = settle * 1.01
    assert abs(es["week"] - 10.0) < 1e-9       # close[-1]/close[-6]
    assert es["y10"] is None                   # 300 closes < 2520
    eth = next(r for r in rep["data"]["Crypto"]
               if r["sym"] == "ETH=F")   # null rows sort last
    assert eth["last"] is None and eth["pct"] is None
    assert eth["label"] == "Ether CME (ETH)"   # null row keeps its slot


def test_fetch_board_quote_fallback(monkeypatch):
    """A contract with history but no live quote (Yahoo ALTSYMBOL,
    e.g. RMB=F) takes last/day from its last two completed closes
    instead of rendering a null row."""
    closes = {sym: [100.0] * 300
              for _g, contracts in fb.all_groups()
              for sym, _l, _d, _k in contracts}
    closes["RMB=F"] = [100.0] * 299 + [101.0]

    def fake_fetch(sym):
        if sym == "RMB=F":
            return None
        c = closes[sym]
        return c[-1] * 1.01, c[-1], 1_700_000_000

    monkeypatch.setattr(fb, "fetch", fake_fetch)
    monkeypatch.setattr(fb, "history", lambda sym: closes.get(sym) or [])
    rep = fb.fetch_board(now=datetime.datetime(2026, 10, 29, 12, 0))
    rmb = next(r for r in rep["data"]["FX"] if r["sym"] == "RMB=F")
    assert rmb["last"] == 101.0
    assert abs(rmb["pct"] - 1.0) < 1e-9     # 101 vs prior close 100
    assert rep["available"] == 67
    assert any("RMB=F" in n for n in rep["notes"])


def test_fetch_board_subset_groups(monkeypatch):
    """groups= restricts the fetch and the returned data to the
    named groups (the dashboard's per-board progressive loads)."""
    closes = {sym: [100.0] * 300
              for _g, contracts in fb.all_groups()
              for sym, _l, _d, _k in contracts}
    _fake_fetch_board(monkeypatch, closes)
    rep = fb.fetch_board(now=datetime.datetime(2026, 10, 29, 12, 0),
                         groups=("FX",))
    assert list(rep["data"]) == ["FX"]
    assert rep["available"] == 8
    assert [r["sym"] for r in rep["data"]["FX"]] == [
        sym for _g, contracts in fb.all_groups() if _g == "FX"
        for sym, _l, _d, _k in contracts]


def test_rows_sorted_by_year_within_group(monkeypatch):
    """Within each group, rows sort by Year % CAGR descending and a
    row without a year value goes last (user request 2026-10-30)."""
    closes = {sym: [100.0] * 300
              for _g, contracts in fb.all_groups()
              for sym, _l, _d, _k in contracts}
    closes["GC=F"] = [100.0 + i for i in range(300)]      # year > 0
    closes["SI=F"] = [400.0 - i for i in range(300)]      # year < 0
    closes["PL=F"] = [100.0] * 100                        # no year value
    rep = _fake_fetch_board(monkeypatch, closes)
    labels = [r["label"] for r in rep["data"]["Metals"]]
    assert labels == ["Gold (GC)", "Copper (HG)",
                      "Palladium (PA)", "Silver (SI)",
                      "Platinum (PL)"]


def test_groups_ordered_by_average_year(monkeypatch):
    """Groups come back ordered by their average Year % CAGR
    descending (user request 2026-10-31); a group with no year
    values keeps its configured place among the year-less,
    last. Flat series everywhere except Metals (rising) and
    Energy (falling): averages Metals > flat 0 > Energy."""
    closes = {sym: [100.0] * 300
              for _g, contracts in fb.all_groups()
              for sym, _l, _d, _k in contracts}
    for sym in ("GC=F", "SI=F", "HG=F", "PL=F", "PA=F"):
        closes[sym] = [100.0 + i for i in range(300)]
    for sym in ("CL=F", "BZ=F", "NG=F", "HO=F", "RB=F"):
        closes[sym] = [400.0 - i for i in range(300)]
    rep = _fake_fetch_board(monkeypatch, closes)
    order = list(rep["data"])
    assert order[0] == "Metals"
    assert order[-1] == "Energy"
    assert fb.group_year_average(rep["data"]["Metals"]) > 0
    assert fb.group_year_average(rep["data"]["Energy"]) < 0
    assert fb.group_year_average([{"year": None}]) is None
    assert fb.group_year_average(
        [{"year": 10.0}, {"year": None}, {"year": 20.0}]) == 15.0


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
              for _g, contracts in fb.all_groups()
              for sym, _label, _dec, _k in contracts}
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
