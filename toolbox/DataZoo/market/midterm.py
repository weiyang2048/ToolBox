"""Midterm elections — the average market path around US
midterms (Markets page, Equities tab 3).

    midterm_series()       the shipped historical series
                           (package data, midterm_paths.csv)
    current_cycle_paths()  the 2026 cycle's path so far, live
                           from Yahoo; {} on any failure
    render_midterms()      HTML for the page (house dark theme)

Definitions (computed 2026-11-03 from Yahoo daily closes):
US is the S&P 500 price index around the 19 midterms
1950-2022; ex-US is a per-midterm equal-weight composite of
five local-currency price indices (Nikkei 225, Hang Seng,
FTSE 100, DAX, CAC 40) over the 15 midterms 1966-2022 —
with the honest wrinkle that 1966-82 rests on the Nikkei
alone (1 index), 1986 on 2, 1990 on 4, and 1994+ on all 5.
Every path is indexed to 100 at the election-day close and
aligned by trading days relative to election day
(-252..+252); the shipped series carries the cross-midterm
mean and the middle half (q25-q75). Price returns
throughout — dividends excluded; ex-US in local currency,
so no FX moves are in it either.
"""

import html
import os
import time
from concurrent.futures import ThreadPoolExecutor

from toolbox.DataZoo.market.svg_charts import line_path, scale, svg_elems
from toolbox.DataZoo.market.yahoo_prices import with_retry, yahoo_daily

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES_CSV = os.path.join(HERE, "midterm_paths.csv")

_MUTE = "#9AA0A6"
_ACCENT = "#FFB066"
_OVERLAY = "#7FB3D5"
_INTL = ("^N225", "^HSI", "^FTSE", "^GDAXI", "^FCHI")
_DAYS = 252  # trading days on each side of election day


def midterm_series():
    """The historical average midterm paths as a DataFrame:
    one row per relative trading day (-252..+252) with
    us_mean/us_q25/us_q75 and xus_mean/xus_q25/xus_q75,
    all indexed to 100 at day 0 (see the module docstring
    for definitions and samples)."""
    import pandas as pd

    return pd.read_csv(SERIES_CSV)


def _indexed(sym, p1):
    """One ticker's closes over the window, indexed to 100
    at its last close; returns (start_day, [values]) with
    values running start_day..0, or None on failure."""
    try:
        closes = with_retry(lambda: yahoo_daily(sym, p1=p1),
                            "yahoo " + sym).dropna()
    except Exception:
        return None
    if len(closes) < 60:
        return None
    closes = closes.iloc[-(_DAYS + 1):]
    vals = (closes / closes.iloc[-1] * 100).tolist()
    return -len(vals) + 1, vals


def current_cycle_paths():
    """The 2026 cycle's path into today's vote (Nov 3, 2026
    IS election day), live from Yahoo via the canonical
    yahoo_prices helpers: ^GSPC for the US and the five
    ex-US indices, each indexed to 100 at its last available
    close, which is aligned to day 0. Returns {"us": [...],
    "xus": [...], "us_start": int, "xus_start": int} — the
    ex-US array is the equal-weight mean over whichever
    indices cover each day, kept only where at least 2 do.
    Any failure degrades: a missing side is simply absent,
    and a total failure returns {} rather than raising, so
    the tab falls back to the historical series alone."""
    try:
        p1 = int(time.time()) - 445 * 86400  # trailing ~14 months
        syms = ("^GSPC",) + _INTL
        with ThreadPoolExecutor(max_workers=6) as pool:
            got = dict(zip(syms, pool.map(
                lambda s: _indexed(s, p1), syms)))
        out = {}
        if got.get("^GSPC") is not None:
            start, vals = got["^GSPC"]
            out["us"], out["us_start"] = vals, start
        legs = [got[s] for s in _INTL if got.get(s) is not None]
        if len(legs) >= 2:
            comp = []
            start = 0
            for day in range(-_DAYS, 1):
                pts = [vals[day - s] for s, vals in legs
                       if s <= day]
                if len(pts) >= 2:
                    if not comp:
                        start = day
                    comp.append(sum(pts) / len(pts))
                elif comp:
                    break
            if comp:
                out["xus"], out["xus_start"] = comp, start
        return out
    except Exception:
        return {}


def _panel(days, mean, q25, q75, title, overlay, overlay_start):
    """One average-path panel as SVG: IQR band, mean line,
    the 100 reference, a dashed marker at election day, and
    the optional 2026 overlay (dashed, days overlay_start..0)."""
    w, h = 1120, 250
    L, R, T, B = 52, 118, 46, 24
    allv = list(q75) + list(q25)
    if overlay:
        allv += list(overlay)
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    import numpy as np

    step = 10 ** np.floor(np.log10((hi - lo) / 4))
    for m in (1, 2, 2.5, 5, 10):
        if (hi - lo) / (step * m) <= 5:
            step *= m
            break
    ticks = []
    t = np.ceil(lo / step) * step
    while t <= hi:
        ticks.append(float(t))
        t += step

    def X(d):
        return scale(d, -_DAYS, _DAYS, L, w - R)

    def Y(v):
        return scale(v, lo, hi, h - B, T)

    body = [f'<text x="{L}" y="15" font-size="13.5" '
            f'font-weight="700" fill="#FAFAFA">'
            f'{html.escape(title)}</text>']
    # Legend, top-right inside the panel.
    lx = w - R - 300
    body.append(f'<line x1="{lx}" x2="{lx + 26}" y1="11" y2="11" '
                f'stroke="{_ACCENT}" stroke-width="2.4"/>'
                f'<text x="{lx + 32}" y="15" font-size="10.5" '
                f'fill="{_MUTE}">Average</text>')
    body.append(f'<rect x="{lx + 108}" y="5" width="26" height="11" '
                f'fill="{_ACCENT}" fill-opacity="0.16"/>'
                f'<text x="{lx + 140}" y="15" font-size="10.5" '
                f'fill="{_MUTE}">Middle half</text>')
    if overlay:
        body.append(
            f'<line x1="{lx + 222}" x2="{lx + 248}" y1="11" y2="11" '
            f'stroke="{_OVERLAY}" stroke-width="1.8" '
            f'stroke-dasharray="6 4"/>'
            f'<text x="{lx + 254}" y="15" font-size="10.5" '
            f'fill="{_MUTE}">2026</text>')
    for tv in ticks:
        y = Y(tv)
        body.append(f'<line x1="{L}" x2="{w - R}" y1="{y:.1f}" '
                    f'y2="{y:.1f}" stroke="#26221C" '
                    f'stroke-width="1"/>')
        body.append(f'<text x="{L - 6}" y="{y + 3.5:.1f}" '
                    f'font-size="10.5" fill="{_MUTE}" '
                    f'text-anchor="end">{tv:g}</text>')
    y100 = Y(100)
    body.append(f'<line x1="{L}" x2="{w - R}" y1="{y100:.1f}" '
                f'y2="{y100:.1f}" stroke="#8A8378" '
                f'stroke-width="1.1" stroke-dasharray="5 4"/>')
    x0 = X(0)
    body.append(f'<line x1="{x0:.1f}" x2="{x0:.1f}" y1="{T}" '
                f'y2="{h - B}" stroke="#8A8378" stroke-width="1.1" '
                f'stroke-dasharray="5 4"/>')
    for d, lab in ((-252, "-12m"), (-126, "-6m"), (0, "Election day"),
                   (126, "+6m"), (252, "+12m")):
        body.append(f'<text x="{X(d):.1f}" y="{h - 6}" '
                    f'font-size="10.5" fill="{_MUTE}" '
                    f'text-anchor="middle">{lab}</text>')
    band = ([(X(d), Y(v)) for d, v in zip(days, q75)]
            + [(X(d), Y(v)) for d, v in
               zip(days[::-1], q25[::-1])])
    body.append('<path d="'
                + line_path([p[0] for p in band],
                            [p[1] for p in band])
                + f' Z" fill="{_ACCENT}" fill-opacity="0.16" '
                  'stroke="none"/>')
    body.append('<path d="'
                + line_path([X(d) for d in days],
                            [Y(v) for v in mean])
                + f'" fill="none" stroke="{_ACCENT}" '
                  'stroke-width="2.2" stroke-linejoin="round"/>')
    end = f"{mean[-1]:.1f}"
    body.append(f'<text x="{w - R + 8}" y="{Y(mean[-1]) + 3.5:.1f}" '
                f'font-size="11" font-weight="700" '
                f'fill="{_ACCENT}">{end}</text>')
    if overlay:
        odays = list(range(overlay_start, overlay_start
                           + len(overlay)))
        body.append('<path d="'
                    + line_path([X(d) for d in odays],
                                [Y(v) for v in overlay])
                    + f'" fill="none" stroke="{_OVERLAY}" '
                      'stroke-width="1.8" stroke-dasharray="6 4" '
                      'stroke-linejoin="round"/>')
        body.append(f'<text x="{X(odays[0]) + 2:.1f}" '
                    f'y="{Y(overlay[0]) - 7:.1f}" font-size="10.5" '
                    f'fill="{_OVERLAY}">2026 (into today\'s '
                    'vote)</text>')
    return svg_elems(w, h, "".join(body))


def _chips(items):
    chip_html = "".join(
        '<span style="background:#171512;border:1px solid '
        '#2E2A24;border-radius:999px;padding:5px 12px;'
        f'font-size:13px;color:#D8D8D8">{c}</span>'
        for c in items)
    return ('<div style="display:flex;gap:8px;flex-wrap:wrap;'
            f'margin:10px 0 16px">{chip_html}</div>')


def render_midterms(df, current=None):
    """Lede + the two average-path panels + stat chips + the
    source note, wrapped in one centered block (max-width
    1120px, the panels' own width, margin auto — the house
    tab convention). `df` is midterm_series(); `current` is
    current_cycle_paths() ({} or None drops the 2026
    overlay). The US chips carry the study's window stats
    (mean of the 19 elections' own returns — the mean path's
    own pre-vote reading differs, since each election is
    indexed to its own day 0); the ex-US chips are computed
    from the composite series itself."""
    current = current or {}
    days = df["day"].tolist()
    xus = df.set_index("day")["xus_mean"]
    xus_after = float(xus.loc[_DAYS] - 100)
    xus_before = float(100 / xus.loc[-126] * 100 - 100)
    lede = ('<div style="font-size:12.5px;color:' + _MUTE
            + ';margin:2px 0 10px">The average stock-market '
            'path around US midterm elections — price returns, '
            'indexed to <span style="color:' + _ACCENT
            + ';font-weight:700">100 at the election-day '
            'close</span>; the band is the middle half of '
            'elections, and the dashed blue line is 2026 into '
            'today\'s vote.</div>')
    us = _panel(days, df["us_mean"].tolist(),
                df["us_q25"].tolist(), df["us_q75"].tolist(),
                "United States — S&P 500, 19 midterms, 1950–2022",
                current.get("us"), current.get("us_start", 0))
    us_chips = _chips([
        "12m after &nbsp;<b>+14.6%</b> mean",
        "12m after &nbsp;positive <b>19 of 19</b>",
        "6m before &nbsp;<b>+1.3%</b> mean",
    ])
    xu = _panel(days, df["xus_mean"].tolist(),
                df["xus_q25"].tolist(), df["xus_q75"].tolist(),
                "Ex-US — five-market composite, 15 midterms, "
                "1966–2022",
                current.get("xus"), current.get("xus_start", 0))
    xu_chips = _chips([
        f"12m after &nbsp;<b>{xus_after:+.1f}%</b> composite",
        f"6m before &nbsp;<b>{xus_before:+.1f}%</b> composite",
        "Sample &nbsp;<b>15 midterms</b>, 1966–2022",
    ])
    note = ('<div style="font-size:11.5px;color:#8A8378;'
            'margin-top:8px">Definitions: US is the S&P 500 '
            'price index around the 19 midterms 1950–2022. '
            'Ex-US is a per-midterm equal-weight composite of '
            'the Nikkei 225, Hang Seng, FTSE 100, DAX and CAC '
            '40 price indices (local currency) over the 15 '
            'midterms 1966–2022 — the early years rest on '
            'fewer markets (1966–82 the Nikkei alone, 1986 '
            'two, 1990 four, 1994 on all five). Price returns '
            'throughout: dividends excluded, and no FX moves '
            'in the ex-US composite. The 2026 overlay is the '
            'current cycle\'s path into today\'s vote, each '
            'market indexed to 100 at its last close. '
            'Computed 2026-11-03 from Yahoo daily closes.'
            '</div>')
    return ('<div style="max-width:1120px;margin:0 auto">'
            + lede + us + us_chips + xu + xu_chips + note
            + "</div>")
