"""A futures + international cash-index board, with cell shading.

:func:`fetch_board` pulls the latest quotes from Yahoo Finance (v8
chart API) for 34 futures contracts in five groups — US Equity,
Commodities, Crypto, Rates, FX — plus 14 cash indices
in a single International equity group (rendered immediately
after US Equity: Yahoo serves no usable non-US equity futures, so
the international lines are the cash markets themselves), and
returns the board as plain
data. This module is the single implementation shared by every
consumer (report generators and dashboards alike); renderers format
the returned rows for their own medium and shade percentage cells
with :func:`shade_zero` on the scales from :func:`board_scales`.

Change convention: for each line, last = meta.regularMarketPrice and
the baseline is the prior completed settlement (for a cash index, its
prior close), meta.chartPreviousClose on a range=1d chart (a longer
range makes chartPreviousClose a stale close days back). Each row
carries a "kind" ("future" or "cash") so renderers and footnotes
treat the international lines as cash indices, not futures.

Horizon columns: each row also carries trailing % changes of the
front-month continuous daily close vs the close N sessions before
the last completed session — week 5, month 21, quarter 63, year 252,
3Y 756, 5Y 1260, 10Y 2520 (fields week/month/quarter/year/y3/y5/y10).
The year-and-longer horizons (year, y3, y5, y10) are expressed as
CAGR — the total change annualized over sessions/252 years, exactly
1, 3, 5 and 10 (user request 2026-10-29); week, month and quarter
stay total % changes. A horizon is None (renders n/a) when the
contract's history is shorter than the lookback — 10Y for RTY=F
(series starts 2017), BTC=F (2017) and ETH=F (2021).

Degrade rules: a contract Yahoo cannot serve (a symbol that 404s) is
emitted as a null row so renderers show n/a; a failed history pull
leaves that contract's horizons null, never a crash. If NO contract
returns data, ``available`` is 0 and callers decide what that means
(a CLI wrapper exits nonzero; a dashboard shows an error).
"""

from __future__ import annotations

import datetime
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

CT = ZoneInfo("America/Chicago")
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/120.0 Safari/537.36"),
      "Accept": "application/json"}

# (group, [(yahoo symbol, display label, decimals for Last)])
GROUPS = [
    ("US Equity", [
        ("ES=F", "S&P 500 (ES)", 2),
        ("NQ=F", "Nasdaq 100 (NQ)", 2),
        ("YM=F", "Dow (YM)", 2),
        ("RTY=F", "Russell 2000 (RTY)", 2),
        # CME E-mini Select Sector futures (user request 2026-10-30):
        # the sector lines Yahoo carries with a usable daily history.
        # Deliberately absent — Yahoo has no usable series for them:
        # Deliberately absent (probed 2026-10-30, retried the same
        # day at the user's request): XAY=F — only ~31 non-null bars
        # scattered across the decade, so its horizons would span
        # multi-year gaps; XAR=F / SOX=F / RX=F and SXB/SXT/SXO=F —
        # a live quote but zero history bars; SXI=F / SXR=F — one
        # bar each. XAK=F joined the board on the retry: Yahoo had
        # resolved it to an MSCI Norway contract in the morning and
        # to the Technology future, with a clean 2016- history, by
        # the afternoon.
        ("XAZ=F", "Communication services (XAZ)", 1),
        ("XAP=F", "Consumer staples (XAP)", 1),
        ("XAE=F", "Energy sector (XAE)", 1),
        ("XAF=F", "Financials (XAF)", 1),
        ("XAV=F", "Health care (XAV)", 1),
        ("XAI=F", "Industrials (XAI)", 1),
        ("XAB=F", "Materials (XAB)", 1),
        ("XAU=F", "Utilities (XAU)", 1),
        ("XAK=F", "Technology (XAK)", 1),
        ("BIO=F", "Nasdaq biotech (BIO)", 1),
    ]),
    ("Commodities", [
        ("CL=F", "WTI crude (CL)", 2),
        ("BZ=F", "Brent (BZ)", 2),
        ("NG=F", "Natural gas (NG)", 3),
        ("RB=F", "RBOB gasoline (RB)", 4),
        ("GC=F", "Gold (GC)", 2),
        ("SI=F", "Silver (SI)", 2),
        ("HG=F", "Copper (HG)", 4),
        ("PL=F", "Platinum (PL)", 2),
        ("ZC=F", "Corn (ZC)", 2),
        ("ZS=F", "Soybeans (ZS)", 2),
        ("ZW=F", "Wheat (ZW)", 2),
    ]),
    ("Crypto", [
        ("BTC=F", "Bitcoin CME (BTC)", 0),
        ("ETH=F", "Ether CME (ETH)", 1),
    ]),
    ("Rates", [
        ("ZN=F", "10Y note (ZN)", 3),
        ("ZB=F", "30Y bond (ZB)", 3),
        ("ZF=F", "5Y note (ZF)", 3),
        ("ZT=F", "2Y note (ZT)", 3),
    ]),
    ("FX", [
        ("DX-Y.NYB", "Dollar index (DXY)", 3),
        ("6E=F", "Euro (6E)", 4),
        ("6J=F", "Yen (6J)", 6),
        ("6B=F", "Pound (6B)", 4),
    ]),
]

# Cash indices, grouped by continent. Display names carry the country
# in parentheses; the Euro Stoxx 50 is a region — the Eurozone — and
# is labelled as such. dec is None: these indices span ~5,400 to
# ~204,000, so decimals are chosen from the index level (cash_dec).
CASH_GROUPS = [
    ("International equity", [
        ("^GSPTSE", "TSX Composite (Canada)", None),
        ("^BVSP", "Bovespa (Brazil)", None),
        ("^MXX", "IPC (Mexico)", None),
        ("^STOXX50E", "Euro Stoxx 50 (Eurozone)", None),
        ("^GDAXI", "DAX (Germany)", None),
        ("^FTSE", "FTSE 100 (United Kingdom)", None),
        ("^FCHI", "CAC 40 (France)", None),
        ("^N225", "Nikkei 225 (Japan)", None),
        ("^HSI", "Hang Seng (Hong Kong)", None),
        ("^KS11", "KOSPI (South Korea)", None),
        ("^TWII", "Taiwan Weighted (Taiwan)", None),
        ("^NSEI", "Nifty 50 (India)", None),
        ("^STI", "Straits Times (Singapore)", None),
        ("^AXJO", "S&P/ASX 200 (Australia)", None),
    ]),
]

# Trailing-horizon lookbacks, in sessions before the last completed one.
HORIZONS = [("week", 5), ("month", 21), ("quarter", 63), ("year", 252),
            ("y3", 756), ("y5", 1260), ("y10", 2520)]
HORIZON_KEYS = [key for key, _n in HORIZONS]


def cash_dec(last):
    """Decimals for a cash index's Last, from its level."""
    if last is None:
        return 2
    if last >= 10000:
        return 0
    if last >= 1000:
        return 1
    return 2


def all_groups():
    """Futures groups with the cash continent groups after US Equity.

    Returns [(group name, [(symbol, label, dec)], kind)] in board
    order — 6 groups, 48 lines.
    """
    out = []
    for gname, contracts in GROUPS:
        out.append((gname, contracts, "future"))
        if gname == "US Equity":
            out.extend((g, c, "cash") for g, c in CASH_GROUPS)
    return out


def fetch(sym):
    """(last, settle, market_time) from Yahoo v8 chart meta, or None."""
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
           + urllib.parse.quote(sym, safe="")
           + "?range=1d&interval=1d")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.load(resp)
    result = payload.get("chart", {}).get("result") or []
    if not result:
        return None
    meta = result[0].get("meta", {})
    last = meta.get("regularMarketPrice")
    settle = meta.get("chartPreviousClose")
    if last is None or settle in (None, 0):
        return None
    return last, settle, meta.get("regularMarketTime")


def history(sym):
    """Completed daily closes (oldest..newest), in-progress bar dropped.

    Explicit period1/period2 is required: range=max makes Yahoo
    silently downgrade to weekly/monthly bars, while a period window
    with interval=1d returns true daily bars. 12y covers the
    2520-session 10Y lookback plus slack.
    """
    now = datetime.datetime.now(CT)
    p2 = int(now.timestamp())
    p1 = int((now - datetime.timedelta(days=12 * 366)).timestamp())
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
           + urllib.parse.quote(sym, safe="")
           + "?period1=%d&period2=%d&interval=1d" % (p1, p2))
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)
    result = payload.get("chart", {}).get("result") or []
    if not result:
        return []
    ts = result[0].get("timestamp") or []
    closes = (result[0].get("indicators", {}).get("quote") or [{}])[0] \
        .get("close") or []
    today = datetime.datetime.now(CT).date()
    out = []
    for t, c in zip(ts, closes):
        if c is None:
            continue
        d = datetime.datetime.fromtimestamp(t, CT).date()
        if d >= today:  # today's bar is still forming
            continue
        out.append(float(c))
    return out


def _ema_last(closes, span):
    """Final value of the adjust=False EMA recursion over closes —
    pandas ewm(span=span, adjust=False) semantics (alpha =
    2/(span+1), seeded at the first close), without the pandas
    dependency."""
    alpha = 2.0 / (span + 1.0)
    e = closes[0]
    for c in closes[1:]:
        e = alpha * c + (1.0 - alpha) * e
    return e


def ls_states(closes):
    """The (0x20, 20x50, 50x200) trend states on completed daily
    closes — the same three states the briefs' stats tables carry
    (combined into one L/S column on the dashboard board at the
    user's request 2026-10-30). True = L, False = S, None when the
    history is shorter than the state's lookback (20 / 50 / 200
    sessions). 0x20: last close above its EMA20; 20x50: EMA20 above
    EMA50; 50x200: EMA50 above EMA200. EMAs are the adjust=False
    recursion, identical to pandas ewm(span=N, adjust=False) final
    values on the same closes."""
    n = len(closes)
    if n < 20:
        return (None, None, None)
    e20 = _ema_last(closes, 20)
    px = closes[-1] > e20
    if n < 50:
        return (px, None, None)
    e50 = _ema_last(closes, 50)
    cross = e20 > e50
    if n < 200:
        return (px, cross, None)
    return (px, cross, e50 > _ema_last(closes, 200))


def _fetch_row(sym):
    """(fetch(sym), history(sym)) for one contract.

    Each call's exceptions are returned in its slot (not raised), so
    the caller degrades the same way for every consumer: a failed
    quote is a null row, a failed history leaves the horizons null.
    """
    try:
        got = fetch(sym)
    except Exception as exc:  # 404, timeout, bad JSON...
        got = exc
    try:
        closes = history(sym)
    except Exception as exc:  # horizons degrade to null, no crash
        closes = exc
    return got, closes


def fetch_board(now=None):
    """Pull the whole board.

    Returns {asof, source, available, data, notes}: data maps each
    group name to its rows ({sym, label, dec, kind, last, chg, pct,
    mtime, week..y10, ls}); rows within each group are sorted by the
    Year % column (CAGR) descending, rows without a year value last
    (user request 2026-10-30); available counts the lines
    with a live quote; notes carries one human-readable line per
    degraded contract, for callers that log. Each row's "ls" is the
    (0x20, 20x50, 50x200) trend-state triple from ls_states() on the
    same completed closes the horizons use.
    """
    now = now or datetime.datetime.now(CT)
    asof = now.strftime("%A %B %d, %Y, %H:%M %Z")
    data, notes, available = {}, [], 0
    # The two HTTP requests per line (quote + 12y history) are
    # I/O-bound; fetch all 48 lines concurrently. Futures are awaited
    # in symbol order, so notes and data come out in board order.
    order = [(gname, sym, label, dec, kind)
             for gname, contracts, kind in all_groups()
             for sym, label, dec in contracts]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {sym: pool.submit(_fetch_row, sym)
                for _, sym, _, _, _ in order}
        for gname, sym, label, dec, kind in order:
            rows = data.setdefault(gname, [])
            row = {"sym": sym, "label": label, "dec": dec, "kind": kind,
                   "last": None, "chg": None, "pct": None, "mtime": None,
                   "ls": (None, None, None)}
            for key, _n in HORIZONS:
                row[key] = None
            got, closes = futs[sym].result()
            if isinstance(got, Exception):  # 404, timeout, bad JSON...
                notes.append("futures: %s unavailable (%s)" % (sym, got))
                got = None
            if got is not None:
                last, settle, mtime = got
                row["last"] = last
                row["chg"] = last - settle
                row["pct"] = 100.0 * (last - settle) / settle
                if mtime:
                    row["mtime"] = datetime.datetime.fromtimestamp(
                        mtime, CT).strftime("%m-%d %H:%M CT")
                if dec is None:  # cash index: decimals from its level
                    row["dec"] = cash_dec(last)
                available += 1
            elif dec is None:
                row["dec"] = cash_dec(None)
            if isinstance(closes, Exception):  # horizons degrade to null
                notes.append("futures: %s history unavailable (%s)"
                             % (sym, closes))
                closes = []
            row["ls"] = ls_states(closes)
            for key, n in HORIZONS:
                if len(closes) > n:
                    total = closes[-1] / closes[-1 - n] - 1.0
                    if n >= 252:
                        # Multi-year columns are CAGR (user request
                        # 2026-10-29): annualize at sessions/252
                        # years — exactly 1, 3, 5, 10 for the
                        # year-and-longer lookbacks (n = 252 is the
                        # identity).
                        total = (1.0 + total) ** (252.0 / n) - 1.0
                    row[key] = 100.0 * total
            rows.append(row)
    for rows in data.values():
        # Within-group order is by Year % CAGR, best first; the sort
        # is stable, so ties (and the year-less tail) keep the
        # configured contract order.
        rows.sort(key=lambda r: (r["year"] is None,
                                 -(r["year"] or 0.0)))
    return {"asof": asof, "source": "Yahoo Finance",
            "available": available, "data": data, "notes": notes}


# --- Shading -------------------------------------------------------
# Gradient stops: #F8696B (red) -> #FFFFFF (white) -> #63BE7B (green),
# linear in v/scale around zero, clamped to the scale. This is the
# one canonical implementation; a hand-copied variant of it once
# inverted the negative branch in one consumer and drifted unnoticed.

_WHITE = (255, 255, 255)
_GREEN = (99, 190, 123)
_RED = (248, 105, 107)


def _lerp(c1, c2, t):
    return tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))


def shade_zero(v, scale):
    """Cell-color hex (RRGGBB, no #) for a % value on a zero-centered
    scale: white at 0, full green/red at ±scale, white for missing
    values or a zero scale."""
    if v is None or scale in (None, 0):
        return "FFFFFF"
    t = min(abs(v) / scale, 1.0)
    return "%02X%02X%02X" % _lerp(_WHITE, _GREEN if v >= 0 else _RED, t)


def board_scales(data):
    """(day_scale, {horizon: scale}) for shading a fetched board:
    the board-wide max |day %| and, per horizon column, the max
    |value| in that column (0 when a column is entirely missing)."""
    day = [abs(r["pct"]) for rows in data.values() for r in rows
           if r.get("pct") is not None]
    scales = {}
    for key in HORIZON_KEYS:
        vals = [abs(r[key]) for rows in data.values() for r in rows
                if r.get(key) is not None]
        scales[key] = max(vals) if vals else 0
    return (max(day) if day else 0), scales
