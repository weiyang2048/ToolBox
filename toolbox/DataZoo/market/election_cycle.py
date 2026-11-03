"""Election cycle — the average market path around US
elections (Markets page, Equities tab 3).

    election_series()       the shipped historical series
                            (package data,
                            election_cycle_paths.csv)
    current_cycle_paths()   the 2026 cycle's path so far,
                            live from Yahoo; {} on any
                            failure
    render_election_cycle() HTML for the page (house dark
                            theme)

Definitions (computed 2026-11-03 from Yahoo daily closes):
every US federal election — midterms and presidential
elections pooled, every even year — aligned by trading
days around election day (±24 months, -504..+504), each
market indexed to 100 at the election-day close; the
shipped series carries the cross-election mean and the
middle half (q25-q75). US is the S&P 500 price index
(37 elections, 1950-2022); Europe is a per-election
equal-weight composite of the FTSE 100, DAX and CAC 40
price indices, over whichever of the three have data for
that election (19 elections, 1986-2022); Japan is the
Nikkei 225 (28 elections, 1968-2022); Hong Kong is the
Hang Seng (17 elections, 1990-2022). Price returns
throughout — dividends excluded; non-US markets in local
currency, so no FX moves are in them either. Adjacent
elections' windows overlap by construction: the curve
is the average four-year cycle centered on election day.
"""

import html
import os
import time
from concurrent.futures import ThreadPoolExecutor

from toolbox.DataZoo.market.svg_charts import line_path, scale, svg_elems
from toolbox.DataZoo.market.yahoo_prices import with_retry, yahoo_daily

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES_CSV = os.path.join(HERE, "election_cycle_paths.csv")

_MUTE = "#9AA0A6"
_ACCENT = "#FFB066"
_OVERLAY = "#7FB3D5"
_EU = ("^FTSE", "^GDAXI", "^FCHI")
_DAYS = 504  # trading days on each side of election day

# Region key, panel title, and the study's headline stats
# (means over elections): the +24m-after mean, how many
# elections were positive at +24m, and the 6m-before mean.
REGIONS = (
    ("us", "United States — S&P 500, 37 elections, "
           "1950–2022", 19.4, "31 of 37", 2.8),
    ("eu", "Europe — FTSE 100 · DAX · CAC 40 composite, "
           "19 elections, 1986–2022", 14.3, "16 of 19", -2.9),
    ("jp", "Japan — Nikkei 225, 28 elections, 1968–2022",
     18.7, "20 of 28", -1.8),
    ("hk", "Hong Kong — Hang Seng, 17 elections, "
           "1990–2022", 18.1, "11 of 17", 1.1),
)


def election_series():
    """The historical average election-cycle paths as a
    DataFrame: one row per relative trading day
    (-504..+504) with <region>_mean/<region>_q25/
    <region>_q75 for us/eu/jp/hk, all indexed to 100 at
    day 0 (see the module docstring for definitions and
    samples)."""
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


def _composite(legs):
    """Equal-weight composite of (start, values) legs over
    days -_DAYS..0, kept only where at least 2 legs cover
    the day; returns (start_day, [values]) or None."""
    comp = []
    start = 0
    for day in range(-_DAYS, 1):
        pts = [vals[day - s] for s, vals in legs if s <= day]
        if len(pts) >= 2:
            if not comp:
                start = day
            comp.append(sum(pts) / len(pts))
        elif comp:
            break
    if not comp:
        return None
    return start, comp


def current_cycle_paths():
    """The 2026 cycle's path into today's vote (Nov 3,
    2026 IS election day), live from Yahoo via the
    canonical yahoo_prices helpers: ^GSPC for the US,
    ^FTSE/^GDAXI/^FCHI composited for Europe, ^N225 for
    Japan and ^HSI for Hong Kong, each indexed to 100 at
    its last available close, which is aligned to day 0.
    Returns {"us": [...], "us_start": int, "eu": [...],
    ...} for whichever regions fetched. Any failure
    degrades: a missing region is simply absent, and a
    total failure returns {} rather than raising, so the
    tab falls back to the historical series alone."""
    try:
        p1 = int(time.time()) - 920 * 86400  # trailing ~30 months
        syms = ("^GSPC",) + _EU + ("^N225", "^HSI")
        with ThreadPoolExecutor(max_workers=6) as pool:
            got = dict(zip(syms, pool.map(
                lambda s: _indexed(s, p1), syms)))
        out = {}

        def put(key, leg):
            if leg is not None:
                out[key], out[key + "_start"] = leg[1], leg[0]

        put("us", got.get("^GSPC"))
        put("jp", got.get("^N225"))
        put("hk", got.get("^HSI"))
        legs = [got[s] for s in _EU if got.get(s) is not None]
        if len(legs) >= 2:
            put("eu", _composite(legs))
        return out
    except Exception:
        return {}


def _panel(days, mean, q25, q75, title, overlay, overlay_start):
    """One average-path panel as SVG: IQR band, mean line,
    the 100 reference, a dashed marker at election day,
    and the optional 2026 overlay (dashed, days
    overlay_start..0, labelled at its end near day 0)."""
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
    for d, lab in ((-504, "-24m"), (-252, "-12m"),
                   (0, "Election day"), (252, "+12m"),
                   (504, "+24m")):
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
        # Label at the END of the overlay, just above its
        # day-0 endpoint (the midterm version labelled the
        # line's start, where the band is widest, and
        # crowded it). All three lines converge at the
        # endpoint and the band passes under the label, so
        # it sits on a backing plate in the chips' colors
        # (#171512 on #2E2A24) at near-full opacity, sized
        # to the text with real padding, and the glyphs
        # carry a paint-order halo in the plate color —
        # together they keep the label crisp where it
        # crosses the band (a 0.55-black rect with ~1px
        # of padding did not).
        lx2, ly2 = X(0) - 6, Y(overlay[-1]) - 8
        lab = "2026 (into today's vote)"
        tw = len(lab) * 6.3  # ~0.60em/char: bold-safe
        body.append(f'<rect x="{lx2 - tw - 8:.1f}" '
                    f'y="{ly2 - 13.5:.1f}" '
                    f'width="{tw + 16:.1f}" height="20" '
                    f'rx="4" fill="#171512" '
                    f'fill-opacity="0.94" stroke="#2E2A24" '
                    f'stroke-width="1"/>')
        body.append(f'<text x="{lx2:.1f}" y="{ly2:.1f}" '
                    f'font-size="10.5" font-weight="600" '
                    f'text-anchor="end" fill="{_OVERLAY}" '
                    f'stroke="#171512" stroke-width="3" '
                    f'paint-order="stroke" '
                    f'stroke-linejoin="round">{lab}</text>')
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
    """Lede + the four regional average-path panels, each
    with its stat chips, + the source note, wrapped in one
    centered block (max-width 1120px, the panels' own
    width, margin auto — the house tab convention). `df`
    is election_series(); `current` is
    current_cycle_paths() ({} or None drops the 2026
    overlays). The chips carry the study's headline stats
    (means over elections of each election's own returns);
    panel endpoint labels read off the mean paths
    themselves."""
    current = current or {}
    days = df["day"].tolist()
    lede = ('<div style="font-size:12.5px;color:' + _MUTE
            + ';margin:2px 0 10px">The average stock-market '
            'path around US elections — midterms and '
            'presidential pooled, ±24 months — price '
            'returns, indexed to <span style="color:'
            + _ACCENT + ';font-weight:700">100 at the '
            'election-day close</span>; the band is the '
            'middle half of elections, and the dashed blue '
            'line is the 2026 cycle into today\'s vote.'
            '</div>')
    parts = [lede]
    for key, title, after, positive, before in REGIONS:
        parts.append(_panel(
            days, df[key + "_mean"].tolist(),
            df[key + "_q25"].tolist(),
            df[key + "_q75"].tolist(), title,
            current.get(key), current.get(key + "_start", 0)))
        parts.append(_chips([
            f"24m after &nbsp;<b>{after:+.1f}%</b> mean "
            f"&nbsp;·&nbsp; positive <b>{positive}</b>",
            f"6m before &nbsp;<b>{before:+.1f}%</b> mean",
        ]))
    note = ('<div style="font-size:11.5px;color:#8A8378;'
            'margin-top:8px">Definitions: every US federal '
            'election — midterms and presidential elections '
            'pooled — aligned by trading days around '
            'election day (±24 months), each market indexed '
            'to 100 at the election-day close. US is the '
            'S&amp;P 500 (37 elections, 1950–2022); Europe '
            'is a per-election equal-weight composite of '
            'the FTSE 100, DAX and CAC 40 over whichever '
            'have data (19 elections, 1986–2022); Japan is '
            'the Nikkei 225 (28 elections, 1968–2022); Hong '
            'Kong is the Hang Seng (17 elections, '
            '1990–2022). Price returns throughout: '
            'dividends excluded; non-US markets in local '
            'currency, so no FX moves are in them. Adjacent '
            'elections\' windows overlap by construction '
            '— the curve is the average four-year cycle '
            'centered on election day. The 2026 overlay is '
            'the current cycle\'s path into today\'s vote, '
            'each market indexed to 100 at its last close. '
            'Computed 2026-11-03 from Yahoo daily closes.'
            '</div>')
    return ('<div style="max-width:1120px;margin:0 auto">'
            + "".join(parts) + note + "</div>")
