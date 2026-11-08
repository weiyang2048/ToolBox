"""Tests for toolbox.DataZoo.market.futures_board (network-free)."""

import datetime

from toolbox.DataZoo.market import futures_board as fb


def test_board_layout():
    groups = fb.all_groups()
    assert [g for g, _c in groups] == [
        "US Equity",
        "Global sectors",
        "Asia-Pacific", "Europe", "Americas", "Spot FX",
        "Crypto", "Metals", "Energy", "Grains & oilseeds",
        "Livestock", "Softs", "Rates", "FX",
    ]
    assert sum(len(c) for _g, c in groups) == 79
    kinds = {k for _g, c in groups for _s, _l, _d, k in c}
    assert kinds == {"future", "cash"}
    us = dict(groups)["US Equity"]
    assert len(us) == 4
    assert all(k == "future" for *_x, k in us)
    fx = dict(groups)["Spot FX"]
    assert len(fx) == 12
    assert fx[0][0] == "DX-Y.NYB"
    assert all(k == "cash" for *_x, k in fx)
    assert all(s.endswith("USD=X") or s == "DX-Y.NYB"
               for s, *_x in fx)
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

    def fake_bars(sym):
        # Completed bars under one fixed session date; fetch_board
        # reads the closes and the last bar's date off them.
        return [(datetime.date(2026, 10, 28), c)
                for c in closes_by_sym.get(sym) or []]

    monkeypatch.setattr(fb, "fetch", fake_fetch)
    monkeypatch.setattr(fb, "_history_bars", fake_bars)
    return fb.fetch_board(now=datetime.datetime(2026, 10, 29, 12, 0))


def test_fetch_board_progress_callback(monkeypatch):
    # The progress hook (the dashboard's board bars): one
    # call before the fetches complete, then one per
    # completed line, the counter running 0..total where
    # total is the board's unique-symbol count.
    closes = {"ES=F": [100.0, 101.0, 102.0]}

    def fake_fetch(sym):
        return 103.0, 102.0, 1_700_000_000

    def fake_bars(sym):
        return [(datetime.date(2026, 10, 28), c)
                for c in closes.get(sym, [100.0, 101.0])]

    monkeypatch.setattr(fb, "fetch", fake_fetch)
    monkeypatch.setattr(fb, "_history_bars", fake_bars)
    calls = []
    rep = fb.fetch_board(
        now=datetime.datetime(2026, 10, 29, 12, 0),
        progress=lambda d, t, s: calls.append((d, t, s)))
    total = len({sym for _g, contracts in fb.all_groups()
                 for sym, *_ in contracts})
    assert rep["available"] > 0
    assert [c[0] for c in calls] == list(range(total + 1))
    assert all(c[1] == total for c in calls)
    assert calls[0][2] == "" and calls[-1][0] == total


def test_fetch_board_assembly(monkeypatch):
    closes = {sym: [100.0] * 300
              for _g, contracts in fb.all_groups()
              for sym, _l, _d, _k in contracts}
    closes["ES=F"] = [100.0] * 299 + [110.0]   # +10% week, +10% day base
    del closes["ETH=F"]                        # unavailable -> null row
    rep = _fake_fetch_board(monkeypatch, closes)
    assert rep["available"] == 78
    assert list(rep["data"]) == [g for g, _c in fb.all_groups()]
    es = rep["data"]["US Equity"][0]
    assert es["label"] == "S&P 500 (ES)"
    assert abs(es["pct"] - 1.0) < 1e-9         # last = settle * 1.01
    assert abs(es["week"] - 10.0) < 1e-9       # close[-1]/close[-6]
    assert es["y10"] is None                   # 300 closes < 2520
    eth = next(r for r in rep["data"]["Crypto"]
               if r["sym"] == "ETH=F")   # null rows sort last
    assert eth["last"] is None and eth["pct"] is None
    assert eth["label"] == "Ether (ETH)"   # null row keeps its slot


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
    monkeypatch.setattr(fb, "_history_bars",
                        lambda sym: [(datetime.date(2026, 10, 28), c)
                                     for c in closes.get(sym) or []])
    rep = fb.fetch_board(now=datetime.datetime(2026, 10, 29, 12, 0))
    rmb = next(r for r in rep["data"]["FX"] if r["sym"] == "RMB=F")
    assert rmb["last"] == 101.0
    assert abs(rmb["pct"] - 1.0) < 1e-9     # 101 vs prior close 100
    assert rep["available"] == 79
    assert any("RMB=F" in n for n in rep["notes"])


def test_history_is_the_closes_view_of_history_bars(monkeypatch):
    """history() keeps its public contract (completed closes only)
    as the closes view of _history_bars."""
    bars = [(datetime.date(2026, 10, 27), 1.0),
            (datetime.date(2026, 10, 28), 2.0)]
    monkeypatch.setattr(fb, "_history_bars", lambda sym: bars)
    assert fb.history("ES=F") == [1.0, 2.0]


def test_row_date_is_the_market_times_eastern_date(monkeypatch):
    """A quoted row's "date" is the Eastern calendar date of its
    market time — the same instant the mtime string renders, in
    machine-usable form (user direction 2026-11-01: renderers
    gray a day % whose data is not from today). The probe instant
    is Fri 2026-10-30 16:00 EDT: Oct 30 in Eastern, already
    Oct 31 in Tokyo."""
    closes = {sym: [100.0] * 300
              for _g, contracts in fb.all_groups()
              for sym, _l, _d, _k in contracts}
    epoch = int(datetime.datetime(
        2026, 10, 30, 16, 0,
        tzinfo=datetime.timezone(datetime.timedelta(hours=-4)),
    ).timestamp())

    def fake_fetch(sym):
        c = closes[sym]
        return c[-1] * 1.01, c[-1], epoch

    monkeypatch.setattr(fb, "fetch", fake_fetch)
    monkeypatch.setattr(fb, "_history_bars",
                        lambda sym: [(datetime.date(2026, 10, 29), c)
                                     for c in closes.get(sym) or []])
    rep = fb.fetch_board(now=datetime.datetime(2026, 11, 1, 12, 0))
    es = rep["data"]["US Equity"][0]
    assert es["date"] == "2026-10-30"
    assert es["mtime"].startswith("10-30")


def test_row_date_fallback_and_null(monkeypatch):
    """A quote-less fallback row (the RMB=F pattern) takes its
    data date from its last completed bar — the only date its
    data has; a fully null row's date stays None."""
    closes = {sym: [100.0] * 300
              for _g, contracts in fb.all_groups()
              for sym, _l, _d, _k in contracts}

    def fake_fetch(sym):
        if sym in ("RMB=F", "ETH=F"):
            return None
        c = closes[sym]
        return c[-1] * 1.01, c[-1], 1_700_000_000

    def fake_bars(sym):
        if sym == "ETH=F":
            return []
        last = (datetime.date(2026, 10, 27) if sym == "RMB=F"
                else datetime.date(2026, 10, 28))
        return [(last, c) for c in closes.get(sym) or []]

    monkeypatch.setattr(fb, "fetch", fake_fetch)
    monkeypatch.setattr(fb, "_history_bars", fake_bars)
    rep = fb.fetch_board(now=datetime.datetime(2026, 10, 29, 12, 0))
    rmb = next(r for r in rep["data"]["FX"] if r["sym"] == "RMB=F")
    assert rmb["pct"] is not None
    assert rmb["date"] == "2026-10-27"
    eth = next(r for r in rep["data"]["Crypto"] if r["sym"] == "ETH=F")
    assert eth["pct"] is None and eth["date"] is None


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


def test_etf_labels_are_bare_tickers():
    # User direction 2026-11-07: on the board, ETFs use only the
    # ticker — the alternatives family and the momentum groups.
    # The global sector suite is the exception since 2026-11-08
    # (next test): its names came back, abbreviated.
    for contracts in (fb.ALTERNATIVES,
                      fb.MOMENTUM_US, fb.MOMENTUM_EUROPE,
                      fb.MOMENTUM_ASIA_PACIFIC):
        for sym, label, _dec in contracts:
            assert label == sym, (sym, label)


def test_global_sector_labels_carry_abbreviated_names():
    # User direction 2026-11-08: the global sector rows carry
    # the sector name again, in its standard short form (a
    # mechanical four-characters-per-word cut was tried first
    # that morning and replaced the same day). The pairing is
    # pinned exactly — a label change is a user decision.
    short = {"IXN": "Tech", "IXG": "Fin", "IXJ": "Health",
             "IXC": "Energy", "MXI": "Matls", "EXI": "Inds",
             "RXI": "Cons Disc", "KXI": "Cons Stap",
             "IXP": "Comm Svcs", "JXI": "Utils",
             "REET": "Real Est"}
    assert {s for s, _l, _d in fb.GLOBAL_SECTORS} == set(short)
    for sym, label, _dec in fb.GLOBAL_SECTORS:
        assert label == f"{short[sym]} ({sym})", (sym, label)


def test_shade_centered_white_at_five():
    # The CAGR shading (user direction 2026-11-07): white AT +5%,
    # red below it, green above — shade_zero of the shifted value.
    assert fb.shade_centered(None, 10) == "FFFFFF"
    assert fb.shade_centered(5.0, 10) == "FFFFFF"
    assert fb.shade_centered(15.0, 10) == "63BE7B"    # full green
    assert fb.shade_centered(-5.0, 10) == "F8696B"    # full red
    assert fb.shade_centered(0.0, 10) == "FCB4B5"     # half red
    assert fb.shade_centered(10.0, 10) == "B1DEBD"    # half green
    assert fb.CAGR_WHITE_PCT == 5.0
    assert fb.CAGR_KEYS == ("year", "y3", "y5", "y10")


def test_board_scales_centered_only_cagr():
    data = {"G": [{"pct": -4.0, "week": 8.0, "month": None,
                   "quarter": -2.0, "year": 21.0, "y3": None,
                   "y5": None, "y10": None},
                  {"pct": 2.0, "week": -3.0, "month": 5.0,
                   "quarter": 1.0, "year": -6.0, "y3": None,
                   "y5": None, "y10": None}]}
    day, scales = fb.board_scales(data, center=fb.CAGR_WHITE_PCT)
    assert day == 4.0
    # Non-CAGR columns keep the zero-centered scale...
    assert scales["week"] == 8.0 and scales["quarter"] == 2.0
    # ...the year column rescales around +5: max(|21-5|, |-6-5|).
    assert scales["year"] == 16.0


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


def test_alternatives_is_a_named_group_only(monkeypatch):
    # The Alternatives roster (2026-11-02) is selectable by name
    # for the dashboard section, and never joins the unfiltered
    # board the briefs' futures report fetches.
    closes = {sym: [50.0] * 300 for sym, _l, _d in fb.ALTERNATIVES}

    def fake_fetch(sym):
        c = closes.get(sym)
        return None if not c else (c[-1] * 1.01, c[-1], 1_700_000_000)

    def fake_bars(sym):
        return [(datetime.date(2026, 10, 28), c)
                for c in closes.get(sym) or []]

    monkeypatch.setattr(fb, "fetch", fake_fetch)
    monkeypatch.setattr(fb, "_history_bars", fake_bars)
    now = datetime.datetime(2026, 10, 29, 12, 0)
    rep = fb.fetch_board(now=now, groups=["Alternatives"])
    assert list(rep["data"]) == ["Alternatives"]
    rows = rep["data"]["Alternatives"]
    assert [r["sym"] for r in rows] == [
        s for s, _l, _d in fb.ALTERNATIVES]
    assert all(r["kind"] == "cash" for r in rows)
    assert rep["available"] == len(fb.ALTERNATIVES)
    full = fb.fetch_board(now=now)
    assert "Alternatives" not in full["data"]
    assert list(full["data"]) == [g for g, _c in fb.all_groups()]


def test_momentum_groups_are_named_only(monkeypatch):
    # The momentum-sleeve rosters (2026-11-07) are selectable
    # by name for the dashboard's equity board — SPMO/FMTM
    # for the US lines, IDMO/XUSM for Europe, EEMO for
    # Asia-Pacific — and never join the unfiltered board.
    rosters = {"Momentum US": fb.MOMENTUM_US,
               "Momentum Europe": fb.MOMENTUM_EUROPE,
               "Momentum Asia-Pacific": fb.MOMENTUM_ASIA_PACIFIC}
    closes = {sym: [50.0] * 300
              for roster in rosters.values()
              for sym, _l, _d in roster}

    def fake_fetch(sym):
        c = closes.get(sym)
        return None if not c else (c[-1] * 1.01, c[-1], 1_700_000_000)

    def fake_bars(sym):
        return [(datetime.date(2026, 10, 28), c)
                for c in closes.get(sym) or []]

    monkeypatch.setattr(fb, "fetch", fake_fetch)
    monkeypatch.setattr(fb, "_history_bars", fake_bars)
    now = datetime.datetime(2026, 10, 29, 12, 0)
    for name, roster in rosters.items():
        rep = fb.fetch_board(now=now, groups=[name])
        assert list(rep["data"]) == [name]
        rows = rep["data"][name]
        assert [r["sym"] for r in rows] == [
            s for s, _l, _d in roster]
        assert all(r["kind"] == "cash" for r in rows)
        assert rep["available"] == len(roster)
    full = fb.fetch_board(now=now)
    for name in rosters:
        assert name not in full["data"]
    assert list(full["data"]) == [g for g, _c in fb.all_groups()]


def test_grad_hex_endpoints_and_midpoint():
    # The briefs' stats-table ramp: red at 0, white at 0.5,
    # green at 1, on the shade_zero stops.
    assert fb.grad_hex(0.0) == "#F8696B"
    assert fb.grad_hex(0.5) == "#FFFFFF"
    assert fb.grad_hex(1.0) == "#63BE7B"
    assert fb.grad_hex(-1.0) == "#F8696B"  # clamped
    assert fb.grad_hex(2.0) == "#63BE7B"


def test_norm_good_directions():
    # 1 = better: min-max, flipped when higher is worse.
    assert fb.norm_good([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]
    assert fb.norm_good([0.0, 5.0, 10.0],
                        higher_good=False) == [1.0, 0.5, 0.0]
    # A constant column sits at the white midpoint.
    assert fb.norm_good([3.0, 3.0]) == [0.5, 0.5]
    # Missing values pass through as None, others unaffected.
    out = fb.norm_good([0.0, None, float("nan"), 10.0])
    assert out[0] == 0.0 and out[1] is None and out[2] is None
    assert out[3] == 1.0


def test_norm_good_center_zero():
    # Divergent around 0: the largest magnitude sets the
    # scale, sign picks the side, 0 is white.
    assert fb.norm_good([-10.0, 0.0, 5.0],
                        center_zero=True) == [0.0, 0.5, 0.75]
