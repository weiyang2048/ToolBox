"""IALT — the Markets page Alternatives section's second tab
(user-approved demo 2026-11-02).

    fetch_ialt_tab()   IALT + SPY + AGG + VOO + the section's
                       seven peer funds (Yahoo adjusted closes)
    render_ialt_tab()  HTML for the page (house dark theme)

IALT's own listed history begins 2025-12-11 — under a year —
so the tab is since-inception by construction: total-return
paths against stocks (SPY), bonds (AGG) and the section's
peer basket, the drawdown ride (with VOO's drawdown on the
same panel as the equity comparison, user direction
2026-11-07), and rolling 63-day correlations. Nothing here wears a 1Y/3Y/5Y label it has not
lived; the footnote says so. The chips carry the fund's
opening-chapter stats (return, vol, Sharpe rf=0, max
drawdown, correlations) with the short window named.
"""
import time
from concurrent.futures import ThreadPoolExecutor

from toolbox.DataZoo.market.eastern import (
    format_day,
    format_month_day,
)
from toolbox.DataZoo.market.svg_charts import line_path, scale, svg_elems
from toolbox.DataZoo.market.yahoo_prices import with_retry, yahoo_daily

_MUTE = "#9AA0A6"
_ACCENT = "#FFB066"
PEERS = ("QAI", "HDG", "MNA", "DBMF", "CTA", "KMLM", "BTAL")
_SYMS = ("IALT", "SPY", "AGG", "VOO") + PEERS


def fetch_ialt_tab():
    """The IALT payload. Each leg retries transient HTTP
    errors with backoff (the shared with_retry, as in the
    other tabs); a final failure raises a RuntimeError naming
    the leg. The frame starts at IALT's first bar; the other
    legs are cut to that window and forward-filled across
    their own holidays."""
    import pandas as pd

    p1 = int(time.time()) - (3 * 365 + 90) * 86400

    def get(sym):
        return sym, with_retry(
            lambda: yahoo_daily(sym, p1=p1), "yahoo " + sym)

    with ThreadPoolExecutor(max_workers=6) as pool:
        got = dict(pool.map(get, _SYMS))
    df = pd.DataFrame(got).dropna(subset=["IALT"])
    df = df[df.index >= df.index[0]].ffill().dropna()
    return _compute(df)


def _compute(df):
    import numpy as np

    basket = df[list(PEERS)].pct_change().mean(axis=1).fillna(0)
    paths = {
        "ialt": df["IALT"] / df["IALT"].iloc[0] * 100,
        "spy": df["SPY"] / df["SPY"].iloc[0] * 100,
        "agg": df["AGG"] / df["AGG"].iloc[0] * 100,
        "peers": (1 + basket).cumprod() * 100}
    dd = df["IALT"] / df["IALT"].cummax() - 1
    dd_voo = df["VOO"] / df["VOO"].cummax() - 1
    rets = df[["IALT", "SPY", "AGG"]].pct_change().dropna()
    roll = None
    if len(rets) > 70:
        roll = {
            "spy": rets["IALT"].rolling(63).corr(rets["SPY"]),
            "agg": rets["IALT"].rolling(63).corr(rets["AGG"])}
    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    out = {
        "dates": dates,
        "paths": {k: [round(float(v), 2) for v in s]
                  for k, s in paths.items()},
        "dd": [round(float(v) * 100, 2) for v in dd],
        "dd_voo": [round(float(v) * 100, 2) for v in dd_voo],
        "voo_max_dd": round(float(dd_voo.min() * 100), 1),
        "voo_cur_dd": round(float(dd_voo.iloc[-1] * 100), 1),
        "ret": {k: round(float(s.iloc[-1] - 100), 1)
                for k, s in paths.items()},
        "vol": round(float(rets["IALT"].std()
                           * np.sqrt(252) * 100), 1),
        "sharpe": round(float(rets["IALT"].mean()
                              / rets["IALT"].std()
                              * np.sqrt(252)), 2),
        "max_dd": round(float(dd.min() * 100), 1),
        "cur_dd": round(float(dd.iloc[-1] * 100), 1),
        "corr_spy": round(float(rets["IALT"].corr(
            rets["SPY"])), 2),
        "corr_agg": round(float(rets["IALT"].corr(
            rets["AGG"])), 2),
        "asof": format_day(df.index[-1]),
        "last_date": df.index[-1].strftime("%Y-%m-%d"),
        "inception": format_month_day(df.index[0])}
    if roll is not None:
        both = roll["spy"].dropna()
        out["roll_dates"] = [d.strftime("%Y-%m-%d")
                             for d in both.index]
        out["roll"] = {
            k: [round(float(v), 3) for v in s.loc[both.index]]
            for k, s in roll.items()}
        out["roll_last"] = {k: round(float(s.iloc[-1]), 2)
                            for k, s in roll.items()}
    return out


_FULL = {"ialt": "IALT", "spy": "US stocks (SPY)",
         "agg": "US bonds (AGG)",
         "peers": "Alternatives peers (eq-wtd)",
         "dd": "IALT drawdown", "dd_voo": "VOO drawdown",
         "cspy": "vs SPY", "cagg": "vs AGG"}


def _panel(dates, series, colors, labels, title, ylabel,
           zero=None, bold=(), fill=None, h=250):
    """One line panel as SVG, month ticks (the tab's window
    is under a year — the decade tabs' year ticks would leave
    it nearly bare)."""
    import datetime as _dt

    import numpy as np

    w, L, R, T, B = 1120, 52, 64, 46, 24
    xs_d = [_dt.date.fromisoformat(d) for d in dates]
    allv = [v for s in series.values() for v in s if v is not None]
    lo, hi = min(allv), max(allv)
    if zero is not None:
        lo, hi = min(lo, zero), max(hi, zero)
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    step = 10 ** np.floor(np.log10((hi - lo) / 4))
    for m in (1, 2, 2.5, 5, 10):
        if (hi - lo) / (step * m) <= 5:
            step *= m
            break
    x0, x1 = xs_d[0].toordinal(), xs_d[-1].toordinal()
    body = ['<text x="%d" y="15" font-size="13.5" font-weight="700" '
            'fill="#FAFAFA">%s</text>' % (L, title)]
    body.append('<text x="14" y="%d" font-size="10.5" fill="%s" '
                'transform="rotate(-90 14 %d)">%s</text>'
                % ((T + h - B) // 2 + 30, _MUTE,
                   (T + h - B) // 2 + 30, ylabel))
    t = np.ceil(lo / step) * step
    while t <= hi:
        y = scale(t, lo, hi, h - B, T)
        body.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" '
                    'stroke="#26221C"/>' % (L, w - R, y, y))
        body.append('<text x="%d" y="%.1f" font-size="10.5" '
                    'fill="%s" text-anchor="end">%g</text>'
                    % (L - 6, y + 3.5, _MUTE, t))
        t += step
    if zero is not None:
        yz = scale(zero, lo, hi, h - B, T)
        body.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" '
                    'stroke="#8A8378" stroke-width="1.1" '
                    'stroke-dasharray="5 4"/>' % (L, w - R, yz, yz))
    seen = set()
    for d in xs_d:
        if (d.year, d.month) in seen:
            continue
        seen.add((d.year, d.month))
        if len(seen) % 2 == 1:
            body.append('<text x="%.1f" y="%d" font-size="10.5" '
                        'fill="%s" text-anchor="middle">%s</text>'
                        % (scale(d.toordinal(), x0, x1, L, w - R),
                           h - 6, _MUTE, d.strftime("%b %y")))
    ends = []
    for key, vals in series.items():
        pts = [(scale(d.toordinal(), x0, x1, L, w - R),
                scale(v, lo, hi, h - B, T))
               for d, v in zip(xs_d, vals) if v is not None]
        if fill == key:
            yz = scale(zero, lo, hi, h - B, T)
            dpath = ("M %.1f %.1f " % (pts[0][0], yz)
                     + " ".join("L %.1f %.1f" % p for p in pts)
                     + " L %.1f %.1f Z" % (pts[-1][0], yz))
            body.append('<path d="%s" fill="#C0392B" '
                        'fill-opacity="0.35"/>' % dpath)
        body.append('<path d="%s" fill="none" stroke="%s" '
                    'stroke-width="%s" stroke-linejoin="round"/>'
                    % (line_path([p[0] for p in pts],
                                 [p[1] for p in pts]),
                       colors[key],
                       "2.2" if key in bold else "1.4"))
        if labels.get(key):
            ends.append([pts[-1][0], pts[-1][1], colors[key],
                         labels[key]])
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + 13)
    for ex, ey, c, txt in ends:
        body.append('<text x="%.1f" y="%.1f" font-size="11" '
                    'font-weight="700" fill="%s">%s</text>'
                    % (ex + 6, ey + 4, c, txt))
    lx = L
    for key in series:
        body.append('<circle cx="%d" cy="33" r="4" fill="%s"/>'
                    % (lx + 4, colors[key]))
        body.append('<text x="%d" y="36.5" font-size="10.5" '
                    'fill="#D8D8D8">%s</text>'
                    % (lx + 13, _FULL.get(key, key)))
        lx += 13 + 7.8 * len(_FULL.get(key, key)) + 22
    return svg_elems(w, h, "".join(body))


def _dd_series(p):
    """The drawdown panel's series: IALT always, VOO's
    comparison when the payload carries it — a last-good
    payload cached by the pre-VOO code has no dd_voo and
    must still render."""
    out = {"dd": p["dd"]}
    if p.get("dd_voo"):
        out["dd_voo"] = p["dd_voo"]
    return out


def _dd_colors(p):
    out = {"dd": "#E23B3E"}
    if p.get("dd_voo"):
        out["dd_voo"] = "#7FB3D5"
    return out


def _dd_labels(p):
    out = {"dd": "%.1f%%" % p["cur_dd"]}
    if p.get("dd_voo"):
        out["dd_voo"] = "%.1f%%" % p["voo_cur_dd"]
    return out


def render_ialt_tab(p, board_date=None):
    """Chips + the three panels + the honest footnote, in the
    centered 1120px block the page's tabs share. With
    `board_date`, the as-of note carries the freshness pill
    (canonical toolbox.DataZoo.market.freshness) marking any
    lag behind the board date."""
    import html

    from toolbox.DataZoo.market.freshness import lag_pill_html

    pill = lag_pill_html(board_date, p.get("last_date"))
    r = p["ret"]
    chips = [
        "Since inception &nbsp;<b>%+.1f%%</b> · SPY <b>%+.1f%%</b>"
        " · AGG <b>%+.1f%%</b>"
        % (r["ialt"], r["spy"], r["agg"]),
        "Vol (ann.) &nbsp;<b>%.1f%%</b> · Sharpe (rf=0) <b>%.2f</b>"
        % (p["vol"], p["sharpe"]),
        "Max drawdown &nbsp;<b>%.1f%%</b> · now <b>%.1f%%</b>"
        % (p["max_dd"], p["cur_dd"]),
        "Correlation &nbsp;SPY <b>%+.2f</b> · AGG <b>%+.2f</b>"
        % (p["corr_spy"], p["corr_agg"]),
    ]
    chip_html = "".join(
        '<span style="background:#171512;border:1px solid '
        '#2E2A24;border-radius:999px;padding:5px 12px;'
        'font-size:13px;color:#D8D8D8">%s</span>' % c
        for c in chips)
    panels = [_panel(
        p["dates"], p["paths"],
        {"ialt": _ACCENT, "spy": "#7FB3D5", "agg": _MUTE,
         "peers": "#2E8B57"},
        {k: "%+.1f%%" % r[k] for k in r},
        "Since inception (%s) — total return, rebased to 100"
        % p["inception"], "Rebased", bold=("ialt",), h=270),
        _panel(p["dates"], _dd_series(p), _dd_colors(p),
               _dd_labels(p),
               "The diversifier's ride — IALT drawdown since "
               "inception", "Drawdown (%)", zero=0, fill="dd",
               bold=("dd",), h=220)]
    if p.get("roll"):
        panels.append(_panel(
            p["roll_dates"],
            {"cspy": p["roll"]["spy"], "cagg": p["roll"]["agg"]},
            {"cspy": "#7FB3D5", "cagg": _MUTE},
            {k: "%+.2f" % v for k, v in (
                ("cspy", p["roll_last"]["spy"]),
                ("cagg", p["roll_last"]["agg"]))},
            "Rolling 63-day correlation — IALT vs stocks and "
            "bonds", "Correlation", zero=0, h=240))
    note = (
        '<div style="font-size:12px;color:%s;margin:2px 0 10px">'
        'As of <span style="color:%s;font-weight:700">%s close'
        '</span>%s &middot; IALT history begins %s — every '
        'figure is since inception</div>'
        % (_MUTE, _ACCENT, html.escape(p["asof"]), pill,
           html.escape(p["inception"])))
    foot = (
        '<div style="font-size:11.5px;color:#8A8378;margin-top:8px">'
        'Adjusted closes (dividends reinvested), Yahoo. IALT is '
        'BlackRock\'s systematic multi-strategy alternatives '
        'fund (long/short equity, macro, event-driven on a '
        'T-bill collateral base). Its own listed history is '
        'under a year, so nothing here is a 1Y/3Y/5Y figure and '
        'the Sharpe and correlations carry a short window — '
        'read them as the fund\'s opening chapter, not its '
        'record. Peers: the section\'s other seven funds (QAI, '
        'HDG, MNA, DBMF, CTA, KMLM, BTAL), equal-weighted '
        'daily.</div>')
    return ('<div style="max-width:1120px;margin:0 auto">'
            + note
            + '<div style="display:flex;gap:8px;flex-wrap:wrap;'
              'margin-bottom:12px">' + chip_html + "</div>"
            + "".join(panels) + foot + "</div>")
