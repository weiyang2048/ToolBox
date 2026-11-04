"""Election cycle — the average market path around US
elections, split by election type (Markets page,
Equities tab 3).

    election_series()       the shipped historical series
                            (package data,
                            election_cycle_paths.csv)
    current_cycle_paths()   the 2026 midterm cycle's path
                            so far, live from Yahoo; {}
                            on any failure
    render_election_cycle() HTML for the page (house dark
                            theme)

Definitions (computed 2026-11-03 from Yahoo daily
closes): US elections split into two sets — midterms
and presidential elections — each aligned by trading
days around election day (±12 months, -252..+252),
each market indexed to 100 at the election-day close;
the shipped series carries, per set, the cross-election
mean and the middle half (q25-q75). Each set has a US
series (S&P 500 price index) and one ex-US series: a
per-election equal-weight composite of the Nikkei 225,
Hang Seng, FTSE 100, DAX and CAC 40 price indices, over
whichever have data for that election. Samples: midterm
US 19 elections (1950-2022), midterm ex-US 15
(1966-2022); presidential US 20 (1948-2024),
presidential ex-US 15 (1968-2024). Price returns
throughout — dividends excluded; non-US markets in
local currency, so no FX moves are in them either. The
2026 overlay belongs to the midterm set only: 2026 is
a midterm year. The overlay is anchored the same way as
the historical paths — each market indexed to 100 at
its own election-day close (Nov 3, 2026) — and extends
past election day as new sessions complete (user
direction 2026-11-04: the current cycle continues into
the future side of the panel).
"""

import datetime
import html
import os
from concurrent.futures import ThreadPoolExecutor

from toolbox.DataZoo.market.svg_charts import line_path, scale, svg_elems
from toolbox.DataZoo.market.yahoo_prices import with_retry, yahoo_daily

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES_CSV = os.path.join(HERE, "election_cycle_paths.csv")

_MUTE = "#9AA0A6"
_ACCENT = "#FFB066"
_OVERLAY = "#7FB3D5"
_XUS = ("^N225", "^HSI", "^FTSE", "^GDAXI", "^FCHI")
_DAYS = 252  # trading days on each side of election day
_ELECTION_DAY = "2026-11-03"  # the 2026 midterm vote (a Tuesday)
# The overlay's fetch window opens 400 days before the
# vote — enough for the full 252-session run-up — and is
# anchored to the election rather than to today, so the
# window never slides forward and eats the run-up.
_FETCH_P1 = int(datetime.datetime(
    2025, 9, 29, tzinfo=datetime.timezone.utc).timestamp())

# Tab structure: (subhead, panels). Each panel: (series
# key, overlay key or None, title, the 12m-before mean,
# the 6m-before mean, the 6m-after mean, how many
# elections were positive at +6m, the 12m-after mean,
# how many were positive at +12m) — the headline stats
# are means over elections of each election's own
# returns, and the chips print them in chronological
# order: the two "before" chips first, then the two
# "after" chips. Only the midterm panels carry an
# overlay key: 2026 is a midterm year.
SECTIONS = (
    ("Midterm elections", (
        ("mid_us", "us", "United States — S&P 500, "
         "19 midterms, 1950–2022",
         4.0, 1.3, 13.4, "19 of 19", 14.6, "19 of 19"),
        ("mid_xus", "xus", "Ex-US — five-market composite, "
         "15 midterms, 1966–2022",
         1.1, -6.1, 11.7, "13 of 15", 10.8, "13 of 15"),
    )),
    ("Presidential elections", (
        ("pres_us", None, "United States — S&P 500, "
         "20 elections, 1948–2024",
         9.5, 4.6, 4.0, "13 of 20", 8.1, "13 of 20"),
        ("pres_xus", None, "Ex-US — five-market composite, "
         "15 elections, 1968–2024",
         13.0, 3.6, 9.8, "13 of 15", 13.4, "13 of 15"),
    )),
)


def election_series():
    """The historical average election-cycle paths as a
    DataFrame: one row per relative trading day
    (-252..+252) with <set>_<market>_mean/_q25/_q75 for
    set in {mid, pres} and market in {us, xus}, all
    indexed to 100 at day 0 (see the module docstring
    for definitions and samples)."""
    import pandas as pd

    return pd.read_csv(SERIES_CSV)


def _anchor_cycle(closes):
    """Anchor one market's daily closes to the 2026
    election: (start_day, [values]) with trading days
    counted in the market's own sessions from its
    election-day session — the last session on or before
    Nov 3, 2026, at most 4 calendar days earlier (a
    market closed on the day itself anchors to its
    previous session; a series ending well before the
    vote is stale, not anchored) — values indexed to 100
    at that session's close, clipped to ±_DAYS. Sessions
    after the vote extend the path past day 0 as they
    complete. None when the series never reaches the
    election or holds fewer than 60 sessions in the
    window."""
    import pandas as pd

    pos = closes.index.searchsorted(
        pd.Timestamp(_ELECTION_DAY), side="right") - 1
    if pos < 0:
        return None
    if (pd.Timestamp(_ELECTION_DAY)
            - closes.index[pos]).days > 4:
        return None
    base = closes.iloc[pos]
    if not base:
        return None
    lo = max(0, pos - _DAYS)
    hi = min(len(closes), pos + _DAYS + 1)
    vals = (closes.iloc[lo:hi] / base * 100).tolist()
    if len(vals) < 60:
        return None
    return lo - pos, vals


def _indexed(sym, p1):
    """One ticker's closes anchored to the 2026 election
    (see _anchor_cycle); returns (start_day, [values])
    with values running start_day..start_day+len-1, or
    None on failure."""
    try:
        closes = with_retry(lambda: yahoo_daily(sym, p1=p1),
                            "yahoo " + sym).dropna()
    except Exception:
        return None
    return _anchor_cycle(closes)


def _composite(legs):
    """Equal-weight composite of (start, values) legs over
    days -_DAYS..+_DAYS, kept only where at least 2 legs
    cover the day; returns (start_day, [values]) or None.
    The kept span is contiguous from its first covered
    day, so the composite ends where the second-latest
    leg ends."""
    comp = []
    start = None
    for day in range(-_DAYS, _DAYS + 1):
        pts = [vals[day - s] for s, vals in legs
               if s <= day < s + len(vals)]
        if len(pts) >= 2:
            if start is None:
                start = day
            comp.append(sum(pts) / len(pts))
        elif start is not None:
            break
    if start is None:
        return None
    return start, comp


def current_cycle_paths():
    """The 2026 midterm cycle's path, live from Yahoo via
    the canonical yahoo_prices helpers: ^GSPC for the
    US, and the five ex-US indices (^N225, ^HSI, ^FTSE,
    ^GDAXI, ^FCHI) composited equal-weight over
    whichever fetched (at least two required). Each
    market is anchored to its own election-day session
    and indexed to 100 at that close (see
    _anchor_cycle), so the path runs from up to 252
    trading days before the vote through the latest
    completed session — past election day as post-vote
    sessions accrue. Returns {"us": [...],
    "us_start": int, "xus": [...], "xus_start": int}
    for whichever sides fetched. Any failure degrades:
    a missing side is simply absent, and a total
    failure returns {} rather than raising, so the tab
    falls back to the historical series alone."""
    try:
        p1 = _FETCH_P1  # anchored to the vote, see above
        syms = ("^GSPC",) + _XUS
        with ThreadPoolExecutor(max_workers=6) as pool:
            got = dict(zip(syms, pool.map(
                lambda s: _indexed(s, p1), syms)))
        out = {}

        def put(key, leg):
            if leg is not None:
                out[key], out[key + "_start"] = leg[1], leg[0]

        put("us", got.get("^GSPC"))
        legs = [got[s] for s in _XUS if got.get(s) is not None]
        if len(legs) >= 2:
            put("xus", _composite(legs))
        return out
    except Exception:
        return {}


def _panel(days, mean, q25, q75, title, overlay, overlay_start):
    """One average-path panel as SVG: IQR band, mean line,
    the 100 reference, a dashed marker at election day,
    and the optional 2026 overlay (dashed, days
    overlay_start..its end — day 0 on election day,
    later as post-vote sessions complete — labelled in
    the sky above its end; the plate clears both the
    line and the band, see ceil_v)."""
    w, h = 1120, 250
    L, R, T, B = 52, 118, 46, 24
    allv = list(q75) + list(q25)
    if overlay:
        allv += list(overlay)
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    import numpy as np

    ceil_v = None
    if overlay:
        # The overlay label's ceiling: the higher of the
        # overlay's own values and the band's top (q75)
        # over the plate's day span (the plate is
        # right-aligned to just left of the overlay's
        # end; its width
        # in days follows from the text length). The
        # plate is placed with its bottom edge a margin
        # above this ceiling, in the empty sky, so it
        # never covers the line or the band — the first
        # plate sat on the endpoint's height (~100) and
        # hid the overlay's last ~90 days, which run at
        # 97-104, under its fill. Headroom: if the
        # ceiling sits so high in the range that plate +
        # margin would cross the plot's top, extend hi
        # (the plate is 20px and the margin 5px, so the
        # ceiling may use at most (180-27)/180 of the
        # range — see the placement below).
        end_day = overlay_start + len(overlay) - 1
        # (Not `lab`: the axis-label loop below reuses
        # that name — the plate reads this one.)
        olab = ("2026 (into today's vote)" if end_day <= 0
                else "2026 (live)")
        lab_w = len(olab) * 6.3 + 16
        px_day = (w - R - L) / (2 * _DAYS)
        d_lo = end_day + int(np.ceil(-6 / px_day
                                     - lab_w / px_day))
        pos = {d: i for i, d in enumerate(days)}
        ceil_v = max(
            [overlay[d - overlay_start]
             for d in range(d_lo, end_day + 1)
             if 0 <= d - overlay_start < len(overlay)]
            + [q75[pos[d]] for d in range(d_lo, end_day + 1)
               if d in pos])
        hi = max(hi, lo + (ceil_v - lo) / 0.85)

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
    for d, lab in ((-252, "-12m"), (-126, "-6m"),
                   (0, "Election day"), (126, "+6m"),
                   (252, "+12m")):
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
        # Label in the empty sky above the overlay's
        # end: right-aligned to just left of the end
        # day, its plate's bottom edge 5px above
        # ceil_v — the
        # higher of the overlay and the band top over
        # the plate's span, computed with the scale
        # above — so the dashed line runs visibly all
        # the way into the election-day marker under a
        # label that covers neither it nor the band.
        # The plate is in the chips' colors (#171512 on
        # #2E2A24) at near-full opacity, sized to the
        # text with real padding, and the glyphs carry
        # a paint-order halo in the plate color.
        lx2 = X(end_day) - 6
        plate_bot = Y(ceil_v) - 5
        tw = len(olab) * 6.3  # ~0.60em/char: bold-safe
        body.append(f'<rect x="{lx2 - tw - 8:.1f}" '
                    f'y="{plate_bot - 20:.1f}" '
                    f'width="{tw + 16:.1f}" height="20" '
                    f'rx="4" fill="#171512" '
                    f'fill-opacity="0.94" stroke="#2E2A24" '
                    f'stroke-width="1"/>')
        body.append(f'<text x="{lx2:.1f}" '
                    f'y="{plate_bot - 6.5:.1f}" '
                    f'font-size="10.5" font-weight="600" '
                    f'text-anchor="end" fill="{_OVERLAY}" '
                    f'stroke="#171512" stroke-width="3" '
                    f'paint-order="stroke" '
                    f'stroke-linejoin="round">{olab}</text>')
    return svg_elems(w, h, "".join(body))


def _chips(items):
    chip_html = "".join(
        '<span style="background:#171512;border:1px solid '
        '#2E2A24;border-radius:999px;padding:5px 12px;'
        f'font-size:13px;color:#D8D8D8">{c}</span>'
        for c in items)
    return ('<div style="display:flex;gap:8px;flex-wrap:wrap;'
            f'margin:10px 0 16px">{chip_html}</div>')


def render_election_cycle(df, current=None):
    """Lede + two sections — Midterm elections, then
    Presidential elections — each with a US and an ex-US
    average-path panel and its stat chips, + the source
    note, wrapped in one centered block (max-width
    1120px, the panels' own width, margin auto — the
    house tab convention). `df` is election_series();
    `current` is current_cycle_paths() ({} or None drops
    the 2026 overlays, which only the midterm panels
    draw — 2026 is a midterm year). The chips carry the
    study's headline stats (means over elections of each
    election's own returns); panel endpoint labels read
    off the mean paths themselves."""
    current = current or {}
    days = df["day"].tolist()
    lede = ('<div style="font-size:12.5px;color:' + _MUTE
            + ';margin:2px 0 10px">The average stock-market '
            'path around US elections, ±12 months — '
            'midterms and presidential elections in '
            'separate panels — price returns, indexed to '
            '<span style="color:' + _ACCENT
            + ';font-weight:700">100 at the election-day '
            'close</span>; the band is the middle half of '
            'elections, and the dashed blue line on the '
            'midterm panels is the 2026 cycle, live '
            'through the latest completed session.</div>')
    parts = [lede]
    for subhead, panels in SECTIONS:
        parts.append(
            '<div style="font-size:15px;font-weight:700;'
            'color:#FAFAFA;margin:20px 0 2px">'
            + subhead + '</div>')
        for key, okey, title, b12, b6, a6, a6pos, a12, \
                a12pos in panels:
            overlay = current.get(okey) if okey else None
            ostart = (current.get(okey + "_start", 0)
                      if okey else 0)
            parts.append(_panel(
                days, df[key + "_mean"].tolist(),
                df[key + "_q25"].tolist(),
                df[key + "_q75"].tolist(), title,
                overlay, ostart))
            parts.append(_chips([
                f"12m before &nbsp;<b>{b12:+.1f}%</b> mean",
                f"6m before &nbsp;<b>{b6:+.1f}%</b> mean",
                f"6m after &nbsp;<b>{a6:+.1f}%</b> mean "
                f"&nbsp;·&nbsp; positive <b>{a6pos}</b>",
                f"12m after &nbsp;<b>{a12:+.1f}%</b> mean "
                f"&nbsp;·&nbsp; positive <b>{a12pos}</b>",
            ]))
    note = ('<div style="font-size:11.5px;color:#8A8378;'
            'margin-top:8px">Definitions: US elections '
            'split by type — midterms and presidential '
            'elections — aligned by trading days around '
            'election day (±12 months), each market indexed '
            'to 100 at the election-day close. US is the '
            'S&amp;P 500 (midterms: 19 elections, '
            '1950–2022; presidential: 20 elections, '
            '1948–2024). Ex-US is a per-election '
            'equal-weight composite of the Nikkei 225, '
            'Hang Seng, FTSE 100, DAX and CAC 40 over '
            'whichever have data (midterms: 15 elections, '
            '1966–2022; presidential: 15 elections, '
            '1968–2024). Price returns throughout: '
            'dividends excluded; non-US markets in local '
            'currency, so no FX moves are in them. The '
            '2026 overlay on the midterm panels is the '
            'current cycle\'s path, each market indexed '
            'to 100 at its own Nov 3, 2026 election-day '
            'close and extending past election day as '
            'new sessions complete, fetched live from '
            'Yahoo daily closes. The historical series '
            'were computed 2026-11-03 from Yahoo daily '
            'closes.'
            '</div>')
    return ('<div style="max-width:1120px;margin:0 auto">'
            + "".join(parts) + note + "</div>")
