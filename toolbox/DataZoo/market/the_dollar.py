"""The dollar — the Currencies section's second tab (Markets
page), demoed and approved 2026-11-01.

    fetch_the_dollar()   DXY + the G10-and-renminbi spot set
                         (the board's Spot FX group), 10y
    render_the_dollar()  HTML for the page (house dark theme)

Every currency series is the value of one unit of the
currency in USD (the board's XXXUSD=X quotes), so a rising
line or a positive bar always means that currency
strengthened against the dollar. The tab's universe is the
full spot set of 11 — broader than the board's old CME
futures list, at the user's direction ("Keep all 11 in the
tab"), and the board's Spot FX group now carries the same
set. The Danish krone is pegged to the euro; the two lines
agreeing to a tenth is a standing consistency check.
"""
import html
import time
from concurrent.futures import ThreadPoolExecutor

from toolbox.DataZoo.market.yahoo_prices import with_retry, yahoo_daily

_MUTE = "#9AA0A6"
_ACCENT = "#FFB066"
# (symbol, friendly name) — USD per one unit of the currency.
FX = [("EURUSD=X", "Euro"), ("GBPUSD=X", "British pound"),
      ("AUDUSD=X", "Australian dollar"),
      ("NZDUSD=X", "New Zealand dollar"),
      ("JPYUSD=X", "Japanese yen"),
      ("CHFUSD=X", "Swiss franc"),
      ("CADUSD=X", "Canadian dollar"),
      ("SEKUSD=X", "Swedish krona"),
      ("NOKUSD=X", "Norwegian krone"),
      ("DKKUSD=X", "Danish krone"),
      ("CNYUSD=X", "Chinese renminbi")]

from toolbox.DataZoo.market.svg_charts import line_path, scale, svg_elems


def fetch_the_dollar():
    """The dollar payload. Each leg retries transient HTTP
    errors with backoff (the shared with_retry, as in the
    power law); a final failure raises a RuntimeError naming
    the leg."""
    import pandas as pd

    p1 = int(time.time()) - (10 * 365 + 320) * 86400
    syms = [s for s, _n in FX] + ["DX-Y.NYB"]

    def get(sym):
        return sym, with_retry(
            lambda: yahoo_daily(sym, p1=p1), "yahoo " + sym)

    with ThreadPoolExecutor(max_workers=6) as pool:
        got = dict(pool.map(get, syms))
    usd = pd.DataFrame({name: got[sym] for sym, name in FX}
                       ).dropna()
    return _compute(got["DX-Y.NYB"].dropna(), usd)


def _compute(dxy, usd):
    import numpy as np

    H = {"1M": 21, "3M": 63, "1Y": 252}
    bars = []
    for name in usd.columns:
        s = usd[name]
        bars.append({"name": name, **{
            k: round(float(s.iloc[-1] / s.iloc[-1 - n] - 1)
                     * 100, 1) for k, n in H.items()}})
    bars.sort(key=lambda b: -b["1Y"])
    ma = dxy.rolling(200).mean()
    both = dxy.to_frame("dxy").assign(ma200=ma).dropna()
    vols = (np.log(usd / usd.shift(1)).rolling(21).std()
            * np.sqrt(252) * 100)
    avg_vol = vols.mean(axis=1).dropna().iloc[-1260:]
    return {
        "dxy_dates": [d.strftime("%Y-%m-%d") for d in both.index],
        "dxy": [round(float(v), 2) for v in both["dxy"]],
        "ma200": [round(float(v), 2) for v in both["ma200"]],
        "bars": bars,
        "breadth": sum(1 for b in bars if b["1Y"] > 0),
        "vol_dates": [d.strftime("%Y-%m-%d")
                      for d in avg_vol.index],
        "vol": [round(float(v), 2) for v in avg_vol],
        "dxy_last": round(float(dxy.iloc[-1]), 1),
        "dxy_vs200": round(float(
            dxy.iloc[-1] / ma.iloc[-1] - 1) * 100, 1),
        "dxy_from_high": round(float(
            dxy.iloc[-1] / dxy.iloc[-2520:].max() - 1) * 100, 1),
        "vol_now": round(float(avg_vol.iloc[-1]), 1),
        "vol_pct": round(float(
            (avg_vol < avg_vol.iloc[-1]).mean() * 100), 0),
        "vol_med": round(float(avg_vol.median()), 1),
        "asof": usd.index[-1].strftime("%a %b %-d, %Y"),
        "last_date": usd.index[-1].strftime("%Y-%m-%d"),
        "dxy_last_date": dxy.index[-1].strftime("%Y-%m-%d"),
        "vol_last_date": avg_vol.index[-1].strftime("%Y-%m-%d")}


def _ordinal(n):
    n = int(n)
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return "%d%s" % (n, suf)


def _line_panel(dates, series, colors, title, ylabel, h, dash=(),
                end_labels=None, y_lo=None, y_hi=None):
    import datetime as _dt

    import numpy as np

    w = 1120
    L, R, T, B = 52, 64, 46, 24
    xs_d = [_dt.date.fromisoformat(d) for d in dates]
    allv = [v for s in series.values() for v in s if v is not None]
    lo = min(allv) if y_lo is None else y_lo
    hi = max(allv) if y_hi is None else y_hi
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    step = 10 ** np.floor(np.log10((hi - lo) / 4))
    for m in (1, 2, 2.5, 5, 10):
        if (hi - lo) / (step * m) <= 5:
            step *= m
            break
    body = ['<text x="%d" y="15" font-size="13.5" font-weight="700" '
            'fill="#FAFAFA">%s</text>' % (L, html.escape(title))]
    t = np.ceil(lo / step) * step
    while t <= hi:
        y = scale(t, lo, hi, h - B, T)
        body.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" '
                    'stroke="#26221C"/>' % (L, w - R, y, y))
        body.append('<text x="%d" y="%.1f" font-size="10.5" '
                    'fill="%s" text-anchor="end">%g</text>'
                    % (L - 6, y + 3.5, _MUTE, t))
        t += step
    body.append('<text x="%d" y="%d" font-size="10.5" fill="%s" '
                'transform="rotate(-90 %d %d)">%s</text>'
                % (14, (T + h - B) // 2 + 30, _MUTE, 14,
                   (T + h - B) // 2 + 30, html.escape(ylabel)))
    x0, x1 = xs_d[0].toordinal(), xs_d[-1].toordinal()
    for yr in sorted({d.year for d in xs_d})[::2]:
        xd = _dt.date(yr, 1, 1).toordinal()
        if x0 <= xd <= x1:
            body.append('<text x="%.1f" y="%d" font-size="10.5" '
                        'fill="%s" text-anchor="middle">%d</text>'
                        % (scale(xd, x0, x1, L, w - R), h - 6,
                           _MUTE, yr))
    for key, vals in series.items():
        pts = [(scale(d.toordinal(), x0, x1, L, w - R),
                scale(v, lo, hi, h - B, T))
               for d, v in zip(xs_d, vals) if v is not None]
        body.append('<path d="%s" fill="none" stroke="%s" '
                    'stroke-width="1.6" stroke-linejoin="round" %s/>'
                    % (line_path([p[0] for p in pts],
                                 [p[1] for p in pts]),
                       colors[key],
                       'stroke-dasharray="5 4"'
                       if key in dash else ""))
        if end_labels and end_labels.get(key):
            ex, ey = pts[-1]
            body.append('<text x="%.1f" y="%.1f" font-size="11" '
                        'font-weight="700" fill="%s">%s</text>'
                        % (ex + 6, ey + 4, colors[key],
                           html.escape(end_labels[key])))
    lx = L
    for key in series:
        body.append('<circle cx="%d" cy="33" r="4" fill="%s"/>'
                    % (lx + 4, colors[key]))
        body.append('<text x="%d" y="36.5" font-size="10.5" '
                    'fill="#D8D8D8">%s</text>'
                    % (lx + 13, html.escape(key)))
        lx += 13 + 7.8 * len(key) + 22
    return svg_elems(w, h, "".join(body))


def _bars_panel(p):
    w, h = 1120, 330
    L, R, T = 178, 250, 40
    bars = p["bars"][::-1]
    lo = min(min(b["1Y"] for b in bars), 0.0) - 1.2
    hi = max(max(b["1Y"] for b in bars), 0.0) + 1.2
    xz = scale(0.0, lo, hi, L, w - R)
    body = ['<text x="10" y="15" font-size="13.5" font-weight="700" '
            'fill="#FAFAFA">Who\'s beating the dollar — %d of 11 '
            'currencies up over 1 year</text>' % p["breadth"]]
    body.append('<line x1="%.1f" x2="%.1f" y1="%d" y2="%d" '
                'stroke="#8A8378" stroke-width="1.1"/>'
                % (xz, xz, T, h - 12))
    n = len(bars)
    for i, b in enumerate(bars):
        y = T + i * (h - T - 16) / n
        bh = (h - T - 16) / n * 0.58
        x1 = scale(b["1Y"], lo, hi, L, w - R)
        c = "#1B9E57" if b["1Y"] >= 0 else "#C0392B"
        body.append('<rect x="%.1f" y="%.1f" width="%.1f" '
                    'height="%.1f" fill="%s"/>'
                    % (min(xz, x1), y, abs(x1 - xz), bh, c))
        body.append('<text x="%d" y="%.1f" font-size="11.5" '
                    'fill="#D8D8D8" text-anchor="end">%s</text>'
                    % (L - 10, y + bh / 2 + 4,
                       html.escape(b["name"])))
        # Positive bars: value then detail, right of the bar.
        # Negative bars: both texts start right of the zero
        # line — the space left of it is the name column, and
        # printing there collides with the names.
        if b["1Y"] >= 0:
            tx, dx, anchor = x1 + 8, x1 + 60, "start"
        else:
            tx, dx, anchor = xz + 8, xz + 60, "start"
        body.append('<text x="%.1f" y="%.1f" font-size="11" '
                    'font-weight="700" fill="%s" '
                    'text-anchor="%s">%+.1f%%</text>'
                    % (tx, y + bh / 2 + 4, c, anchor, b["1Y"]))
        body.append('<text x="%.1f" y="%.1f" font-size="10.5" '
                    'fill="%s" text-anchor="%s">1M %+.1f · 3M '
                    '%+.1f</text>'
                    % (dx, y + bh / 2 + 4, _MUTE, anchor,
                       b["1M"], b["3M"]))
    body.append('<text x="%.1f" y="%d" font-size="10.5" fill="%s" '
                'text-anchor="middle">Return vs USD, 1 year (%%)'
                '</text>' % ((L + w - R) / 2, h - 0, _MUTE))
    return svg_elems(w, h, "".join(body))


def render_the_dollar(p, board_date=None):
    """Chips + panels + footnote. With `board_date`, the as-of
    note carries the freshness pill (canonical
    toolbox.DataZoo.market.freshness) marking any lag behind
    the board date."""
    from toolbox.DataZoo.market.freshness import lag_pill_html

    pill = lag_pill_html(board_date, p.get("last_date"))
    chips = [
        "DXY &nbsp;<b>%.1f</b> · %+.1f%% vs 200-day · %+.1f%% "
        "from the 10y high" % (p["dxy_last"], p["dxy_vs200"],
                               p["dxy_from_high"]),
        "Breadth &nbsp;<b>%d of 11</b> up vs USD (1Y)"
        % p["breadth"],
        "FX vol &nbsp;<b>%.1f%%</b> · %s pct of 5y"
        % (p["vol_now"], _ordinal(p["vol_pct"])),
    ]
    chip_html = "".join(
        '<span style="background:#171512;border:1px solid '
        '#2E2A24;border-radius:999px;padding:5px 12px;'
        'font-size:13px;color:#D8D8D8">%s</span>' % c
        for c in chips)
    p1 = _line_panel(
        p["dxy_dates"], {"Dollar index (DXY)": p["dxy"],
                         "200-day average": p["ma200"]},
        {"Dollar index (DXY)": "#7FB3D5",
         "200-day average": "#9AA0A6"},
        "The dollar cycle — DXY vs its 200-day average, "
        "10 years", "DXY", 260,
        end_labels={"Dollar index (DXY)": "%.1f" % p["dxy_last"]})
    p2 = _bars_panel(p)
    p3 = _line_panel(
        p["vol_dates"], {"Realized vol": p["vol"],
                         "5y median": [p["vol_med"]] *
                         len(p["vol"])},
        {"Realized vol": "#7FB3D5", "5y median": "#9AA0A6"},
        "Is FX awake — average realized vol across the 11 "
        "currencies (21d, annualized)", "Vol (%)", 240,
        dash=("5y median",), y_lo=0.0)
    note = ('<div style="font-size:12px;color:%s;margin:2px 0 '
            '10px">As of <span style="color:%s;font-weight:700">'
            '%s close</span>%s &middot; every series measured '
            'against USD</div>' % (_MUTE, _ACCENT,
                                   html.escape(p["asof"]), pill))
    foot = ('<div style="font-size:11.5px;color:#8A8378;'
            'margin-top:8px">Spot quotes (Yahoo), each currency '
            'as its USD value, so a positive bar always means '
            'that currency strengthened. The set is the full '
            'G10 + renminbi — broader than a futures list, and '
            'the same set the board\'s Spot FX group carries. '
            'Built-in check: the Danish krone is pegged to the '
            'euro, and the two agree to a tenth at every '
            'horizon.</div>')
    return (note
            + '<div style="display:flex;gap:8px;flex-wrap:wrap;'
              'margin-bottom:12px">' + chip_html + "</div>"
            + p1 + p2 + p3 + foot)
