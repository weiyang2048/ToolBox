"""US vs ex-US — the leadership-shift view (Markets page,
Equities tab 2).

    fetch_us_xus()   total-return closes for SPY / EFA / EEM /
                     ACWX + DXY (Yahoo), a decade of ratios,
                     the rolling 12-month spread, the dollar
                     panel and the snapshot stats
    render_us_xus()  HTML for the page (house dark theme)

The demo this was approved from (demos/equities-tabs/us-xus)
computed, as of Fri Oct 30, 2026: 10y ratios Dev -40.5 /
EM -47.7 / all ex-US -41.3; 1y returns US +11.1 / Dev +12.4 /
EM +11.1 / ex-US +12.8; spread +1.6 pts; every ex-US ratio
below its 200-day; DXY corr -0.42. The module recomputes the
same quantities live; the verdict text in the footnote states
the METHOD, and the chips carry the live numbers — a shift
call is made only by the tape and the 200-day check together,
never hard-coded.
"""
import html
import time
from concurrent.futures import ThreadPoolExecutor

from toolbox.DataZoo.market.svg_charts import line_path, scale, svg_elems
from toolbox.DataZoo.market.yahoo_prices import with_retry, yahoo_daily

_MUTE = "#9AA0A6"
_ACCENT = "#FFB066"
_SYMS = ("SPY", "EFA", "EEM", "ACWX", "DX-Y.NYB")
_BLOCS = (("dev", "EFA", "Developed ex-US", "#2E8B57"),
          ("em", "EEM", "Emerging", "#E67300"),
          ("xus", "ACWX", "All ex-US", "#173F6E"))


def fetch_us_xus():
    """The shift payload. Each leg retries transient HTTP
    errors with backoff (the shared with_retry, as in the
    power law); a final failure raises a RuntimeError naming
    the leg."""
    import pandas as pd

    p1 = int(time.time()) - (10 * 365 + 320) * 86400

    def get(sym):
        return sym, with_retry(
            lambda: yahoo_daily(sym, p1=p1), "yahoo " + sym)

    with ThreadPoolExecutor(max_workers=5) as pool:
        got = dict(pool.map(get, _SYMS))
    return _compute(pd.DataFrame(got).dropna())


def _compute(df):
    import pandas as pd

    ratios = pd.DataFrame({
        "dev": df["EFA"] / df["SPY"] * 100,
        "em": df["EEM"] / df["SPY"] * 100,
        "xus": df["ACWX"] / df["SPY"] * 100})
    for c in ratios:
        ratios[c] = ratios[c] / ratios[c].iloc[260] * 100
    disp = ratios.iloc[260:]
    spread = (df["ACWX"].pct_change(252)
              - df["SPY"].pct_change(252)).dropna() * 100
    spread = spread[spread.index >= disp.index[0]]
    dxy = df["DX-Y.NYB"] / df["DX-Y.NYB"].loc[disp.index[0]] * 100
    ratio = df["ACWX"] / df["SPY"]
    ratio_r = ratio / ratio.loc[disp.index[0]] * 100
    both = pd.DataFrame({"dxy": dxy, "ratio": ratio_r}).dropna()
    corr = float(both["dxy"].pct_change()
                 .corr(both["ratio"].pct_change()))

    def frame_of(series_map):
        return {"dates": [d.strftime("%Y-%m-%d")
                          for d in disp.index],
                **{k: [round(float(v), 2) for v in s]
                   for k, s in series_map.items()}}

    def ret(s, n):
        return round(float(s.iloc[-1] / s.iloc[-1 - n] - 1)
                     * 100, 1)

    snap = {}
    for key, col, _label, _c in _BLOCS:
        r = df[col] / df["SPY"]
        ma = r.rolling(200).mean()
        snap[key] = {
            "vs200": round(float(r.iloc[-1] / ma.iloc[-1] - 1)
                           * 100, 1),
            "chg10y": round(float(disp[key].iloc[-1] - 100), 1),
            "chg1y": round(float(disp[key].iloc[-1]
                                 / disp[key].iloc[-252] - 1)
                           * 100, 1)}
    frame = frame_of({"dev": disp["dev"], "em": disp["em"],
                      "xus": disp["xus"]})
    frame["spread"] = [round(float(v), 2) for v in
                       spread.reindex(disp.index)]
    frame["dxy"] = [round(float(v), 2) for v in
                    both["dxy"].reindex(disp.index)]
    frame["ratio"] = [round(float(v), 2) for v in
                      both["ratio"].reindex(disp.index)]
    return {
        "frame": frame, "snap": snap,
        "ret1y": {k: ret(df[k], 252)
                  for k in ("SPY", "EFA", "EEM", "ACWX")},
        "ret1m": {k: ret(df[k], 21)
                  for k in ("SPY", "EFA", "EEM", "ACWX")},
        "spread_now": round(float(spread.iloc[-1]), 1),
        "corr_dxy": round(corr, 2),
        "asof": df.index[-1].strftime("%a %b %-d, %Y"),
        "last_date": df.index[-1].strftime("%Y-%m-%d")}


def _panel(dates, series, colors, labels, title, ylabel,
           zero=None, bold=(), h=250):
    """One line panel as SVG. series: {key: [values or None]}."""
    import datetime as _dt

    import numpy as np

    w = 1120
    L, R, T, B = 52, 64, 46, 24
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
    ticks = []
    t = np.ceil(lo / step) * step
    while t <= hi:
        ticks.append(float(t))
        t += step
    x0, x1 = xs_d[0].toordinal(), xs_d[-1].toordinal()
    body = ['<text x="%d" y="15" font-size="13.5" font-weight="700" '
            'fill="#FAFAFA">%s</text>' % (L, html.escape(title))]
    body.append('<text x="%d" y="%d" font-size="10.5" '
                'fill="%s" transform="rotate(-90 %d %d)">%s</text>'
                % (14, (T + h - B) // 2 + 30, _MUTE, 14,
                   (T + h - B) // 2 + 30, html.escape(ylabel)))
    for tv in ticks:
        y = scale(tv, lo, hi, h - B, T)
        body.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" '
                    'stroke="#26221C" stroke-width="1"/>'
                    % (L, w - R, y, y))
        body.append('<text x="%d" y="%.1f" font-size="10.5" '
                    'fill="%s" text-anchor="end">%g</text>'
                    % (L - 6, y + 3.5, _MUTE, tv))
    if zero is not None:
        yz = scale(zero, lo, hi, h - B, T)
        body.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" '
                    'stroke="#8A8378" stroke-width="1.1" '
                    'stroke-dasharray="5 4"/>' % (L, w - R, yz, yz))
    years = sorted({d.year for d in xs_d})
    for yr in years[::2]:
        xd = _dt.date(yr, 1, 1).toordinal()
        if x0 <= xd <= x1:
            body.append('<text x="%.1f" y="%d" font-size="10.5" '
                        'fill="%s" text-anchor="middle">%d</text>'
                        % (scale(xd, x0, x1, L, w - R), h - 6,
                           _MUTE, yr))
    ends = []
    for key, vals in series.items():
        pts = [(scale(d.toordinal(), x0, x1, L, w - R),
                scale(v, lo, hi, h - B, T))
               for d, v in zip(xs_d, vals) if v is not None]
        body.append('<path d="%s" fill="none" stroke="%s" '
                    'stroke-width="%s" stroke-linejoin="round"/>'
                    % (line_path([p[0] for p in pts],
                                 [p[1] for p in pts]),
                       colors[key],
                       "2.2" if key in bold else "1.4"))
        if labels.get(key):
            ends.append([pts[-1][0], pts[-1][1], colors[key],
                         labels[key]])
    # Endpoint labels: enforce a 13px minimum vertical gap so
    # near-tied endpoints never print over each other.
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + 13)
    for ex, ey, c, txt in ends:
        body.append('<text x="%.1f" y="%.1f" font-size="11" '
                    'font-weight="700" fill="%s">%s</text>'
                    % (ex + 6, ey + 4, c, html.escape(txt)))
    # Legend in the strip between the title and the plot (the
    # bottom strip belongs to the year labels).
    lx = L
    for key in series:
        body.append('<circle cx="%d" cy="%d" r="4" fill="%s"/>'
                    % (lx + 4, 33, colors[key]))
        body.append('<text x="%d" y="%d" font-size="10.5" '
                    'fill="#D8D8D8">%s</text>'
                    % (lx + 13, 36.5,
                       html.escape(_FULL.get(key, key))))
        lx += 13 + 7.8 * len(_FULL.get(key, key)) + 22
    return svg_elems(w, h, "".join(body))


_FULL = {"dev": "Developed ex-US (EFA/SPY)",
         "em": "Emerging (EEM/SPY)",
         "xus": "All ex-US (ACWX/SPY)",
         "spread": "Spread (pts)",
         "ratio": "ACWX / SPY ratio",
         "dxy": "Dollar index (DXY)"}


def _spread_panel(frame):
    import datetime as _dt

    import numpy as np

    w, h = 1120, 240
    L, R, T, B = 52, 30, 26, 24
    dates = frame["dates"]
    vals = frame["spread"]
    xs_d = [_dt.date.fromisoformat(d) for d in dates]
    clean = [(d, v) for d, v in zip(xs_d, vals) if v is not None]
    lo = min(min(v for _d, v in clean), 0.0)
    hi = max(max(v for _d, v in clean), 0.0)
    pad = (hi - lo) * 0.1 or 1.0
    lo, hi = lo - pad, hi + pad
    x0, x1 = xs_d[0].toordinal(), xs_d[-1].toordinal()
    yz = scale(0.0, lo, hi, h - B, T)
    body = ['<text x="%d" y="15" font-size="13.5" font-weight="700" '
            'fill="#FAFAFA">The shift tape — rolling 12-month '
            'return, all ex-US minus US</text>' % L]
    for tv in range(int(np.ceil(lo / 10)) * 10,
                    int(hi) + 1, 10):
        y = scale(tv, lo, hi, h - B, T)
        body.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" '
                    'stroke="#26221C"/>' % (L, w - R, y, y))
        body.append('<text x="%d" y="%.1f" font-size="10.5" '
                    'fill="%s" text-anchor="end">%+d</text>'
                    % (L - 6, y + 3.5, _MUTE, tv))

    def areas():
        """Both sign fills in one pass, split at interpolated
        zero crossings."""
        out = {1: [], -1: []}
        runs = {1: [], -1: []}

        def flush(sign):
            run = runs[sign]
            if len(run) >= 2:
                d = ["M %.1f %.1f" % (run[0][0], yz)]
                d += ["L %.1f %.1f" % p for p in run]
                d.append("L %.1f %.1f Z" % (run[-1][0], yz))
                out[sign].append("".join(d))
            runs[sign] = []

        prev = None
        for d, v in clean:
            x = scale(d.toordinal(), x0, x1, L, w - R)
            y = scale(v, lo, hi, h - B, T)
            sign = 1 if v > 0 else -1
            if prev is not None and prev[2] != sign:
                pv, px, _py = prev[1], prev[0], prev[3]
                xc = px + (x - px) * (0.0 - pv) / (v - pv)
                closing = -sign
                runs[closing].append((xc, yz))
                flush(closing)
                runs[sign] = [(xc, yz)]
            runs[sign].append((x, y))
            prev = (x, v, sign, y)
        flush(1)
        flush(-1)
        return "".join(out[1]), "".join(out[-1])

    pos, neg = areas()
    body.append('<path d="%s" fill="#1B9E57" fill-opacity="0.4"/>'
                % pos)
    body.append('<path d="%s" fill="#C0392B" fill-opacity="0.3"/>'
                % neg)
    xs = [scale(d.toordinal(), x0, x1, L, w - R) for d, _v in clean]
    ys = [scale(v, lo, hi, h - B, T) for _d, v in clean]
    body.append('<path d="%s" fill="none" stroke="#7FB3D5" '
                'stroke-width="1.4" stroke-linejoin="round"/>'
                % line_path(xs, ys))
    body.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" '
                'stroke="#8A8378" stroke-width="1.1"/>'
                % (L, w - R, yz, yz))
    years = sorted({d.year for d in xs_d})
    for yr in years[::2]:
        xd = _dt.date(yr, 1, 1).toordinal()
        if x0 <= xd <= x1:
            body.append('<text x="%.1f" y="%d" font-size="10.5" '
                        'fill="%s" text-anchor="middle">%d</text>'
                        % (scale(xd, x0, x1, L, w - R), h - 6,
                           _MUTE, yr))
    return svg_elems(w, h, "".join(body))


def render_us_xus(p, board_date=None):
    """Chips + the three panels + the honest footnote, wrapped
    in one centered block (max-width 1120px, the panels' own
    width, margin auto — user direction 2026-11-02: the tab
    sat left on the wide page while the page's other blocks
    center). With `board_date`, the as-of note carries the
    freshness pill (canonical
    toolbox.DataZoo.market.freshness) marking any lag behind
    the board date."""
    from toolbox.DataZoo.market.freshness import lag_pill_html

    pill = lag_pill_html(board_date, p.get("last_date"))
    f, s = p["frame"], p["snap"]
    r1, r1m = p["ret1y"], p["ret1m"]
    chips = [
        "1Y &nbsp;US <b>%+.1f%%</b> · Dev <b>%+.1f%%</b> · EM "
        "<b>%+.1f%%</b> · ex-US <b>%+.1f%%</b>"
        % (r1["SPY"], r1["EFA"], r1["EEM"], r1["ACWX"]),
        "Shift tape &nbsp;<b>%+.1f pts</b>" % p["spread_now"],
        "Ratio vs its 200-day &nbsp;<b>%+.1f%%</b>"
        % s["xus"]["vs200"],
        "Last month &nbsp;US <b>%+.1f%%</b> · Dev <b>%+.1f%%</b>"
        " · EM <b>%+.1f%%</b> · ex-US <b>%+.1f%%</b>"
        % (r1m["SPY"], r1m["EFA"], r1m["EEM"], r1m["ACWX"]),
    ]
    chip_html = "".join(
        '<span style="background:#171512;border:1px solid '
        '#2E2A24;border-radius:999px;padding:5px 12px;'
        'font-size:13px;color:#D8D8D8">%s</span>' % c
        for c in chips)
    ratio_labels = {k: "%+.1f%%" % s[k]["chg10y"]
                    for k, _c, _l, _col in _BLOCS}
    colors = {k: col for k, _c, _l, col in _BLOCS}
    p1 = _panel(f["dates"],
                {"xus": f["xus"], "dev": f["dev"], "em": f["em"]},
                colors, ratio_labels,
                "A decade of US dominance — relative strength of "
                "ex-US vs the US, 10 years",
                "Ratio (start = 100)", zero=100, bold=("xus",),
                h=270)
    p2 = _spread_panel(f)
    p3 = _panel(f["dates"],
                {"ratio": f["ratio"], "dxy": f["dxy"]},
                {"ratio": "#7FB3D5", "dxy": "#9AA0A6"},
                {"ratio": "", "dxy": ""},
                "The shift rides the dollar — ex-US relative "
                "strength vs the dollar (corr %+.2f)"
                % p["corr_dxy"],
                "Start = 100", h=250)
    note = (
        '<div style="font-size:12px;color:%s;margin:2px 0 10px">'
        'As of <span style="color:%s;font-weight:700">%s close'
        '</span>%s &middot; total-return ETF proxies</div>'
        % (_MUTE, _ACCENT, html.escape(p["asof"]), pill))
    foot = (
        '<div style="font-size:11.5px;color:#8A8378;margin-top:8px">'
        'Proxies: SPY (US), EFA (developed ex-US), EEM '
        '(emerging), ACWX (all ex-US), on adjusted closes '
        '(dividends reinvested). Read: the decade panel shows '
        'what the shift is shifting FROM; the tape turns '
        'positive when ex-US outruns the US over a rolling '
        'year; the claim "a shift is under way" needs the tape '
        'positive AND the ratio above its 200-day average — '
        'the chips report both, live.</div>')
    return ('<div style="max-width:1120px;margin:0 auto">'
            + note
            + '<div style="display:flex;gap:8px;flex-wrap:wrap;'
              'margin-bottom:12px">' + chip_html + "</div>"
            + p1 + p2 + p3 + foot + "</div>")
