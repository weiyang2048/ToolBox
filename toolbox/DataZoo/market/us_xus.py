"""US vs ex-US — the leadership-shift view (Markets page,
Equities tab 2).

    fetch_us_xus()   total-return closes for SPY / EFA / EEM /
                     ACWX + DXY (Yahoo), a decade of ratios,
                     the rolling 12-month spread, the dollar
                     panel and the snapshot stats
    us_xus_head_html()  the as-of note + chips
    chart_figs()        the three panels as Plotly figures
                        on the shared warped axis
                        (plotly_panels)
    us_xus_foot_html()  the footnote

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

from toolbox.DataZoo.market.eastern import format_day
from toolbox.DataZoo.market.plotly_panels import (
    add_seam,
    add_y_chips,
    nice_ticks,
    panel_layout,
    warped_panel,
    x_axis,
    y_axis_dict,
)
from toolbox.DataZoo.market.yahoo_prices import with_retry, yahoo_daily

_MUTE = "#9AA0A6"
_ACCENT = "#FFB066"
_SYMS = ("SPY", "EFA", "EEM", "ACWX", "DX-Y.NYB")
_BLOCS = (("dev", "EFA", "Developed ex-US", "#2E8B57"),
          ("em", "EEM", "Emerging", "#E67300"),
          ("xus", "ACWX", "All ex-US", "#90EE90"))


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
        "asof": format_day(df.index[-1]),
        "last_date": df.index[-1].strftime("%Y-%m-%d")}


_FULL = {"dev": "Developed ex-US (EFA/SPY)",
         "em": "Emerging (EEM/SPY)",
         "xus": "All ex-US (ACWX/SPY)",
         "spread": "Spread (pts)",
         "ratio": "ACWX / SPY ratio",
         "dxy": "Dollar index (DXY)"}
_SHORT = {"dev": "Developed ex-US", "em": "Emerging",
          "xus": "All ex-US", "ratio": "ex-US ratio",
          "dxy": "DXY"}
# The blocs' SVG hues, with dev brightened for the dark
# Plotly skin (the IALT peers precedent).
_COLORS = {"dev": "#35B26F", "em": "#E67300",
           "xus": "#90EE90", "ratio": "#7FB3D5",
           "dxy": "#9AA0A6"}


def _spread_fig(p):
    """The shift tape as a Plotly area panel on the shared
    warped axis: the rolling 12-month spread (ex-US minus
    US, points), filled green above zero and red below —
    the sign split the SVG panel drew — with the spread
    line over it and a dashed zero rule."""
    import plotly.graph_objects as go

    f = p["frame"]
    dates, vals = f["dates"], f["spread"]
    axis, warp, warped = x_axis(dates)
    import pandas as pd

    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    pts = [(d, v) for d, v in zip(idx, vals) if v is not None]
    runs = []
    run = [pts[0]]
    for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
        if (v0 > 0) != (v1 > 0) and v0 != v1:
            frac = (0.0 - v0) / (v1 - v0)
            dc = d0 + (d1 - d0) * frac
            run.append((dc, 0.0))
            runs.append(run)
            run = [(dc, 0.0)]
        run.append((d1, v1))
    runs.append(run)
    fig = go.Figure()
    for r in runs:
        if len(r) < 2:
            continue
        pos = r[1][1] > 0 or (len(r) > 2 and r[2][1] > 0)
        fig.add_trace(go.Scatter(
            x=warp([q[0] for q in r]),
            y=[q[1] for q in r], mode="lines",
            fill="tozeroy", showlegend=False,
            hoverinfo="skip",
            fillcolor="rgba(27,158,87,0.40)" if pos
            else "rgba(192,57,43,0.30)",
            line=dict(color="rgba(0,0,0,0)", width=0)))
    fig.add_trace(go.Scatter(
        x=warp([q[0] for q in pts]),
        y=[q[1] for q in pts], mode="lines",
        name="Spread (pts)",
        line=dict(color="#7FB3D5", width=1.5),
        customdata=[q[0].strftime("%b %d, %Y") for q in pts],
        hovertemplate="%{customdata}<br>Spread (pts): "
                      "%{y:+.2f}<extra></extra>"))
    fig.add_shape(type="line", xref="x", yref="y",
                  x0=0.0, x1=1.0, y0=0, y1=0,
                  line=dict(color="rgba(232,236,236,0.5)",
                            width=1, dash="dash"))
    lo = min(0.0, min(v for _d, v in pts))
    hi = max(0.0, max(v for _d, v in pts))
    pad = (hi - lo) * 0.1 or 1.0
    lo, hi = lo - pad, hi + pad
    ticks = nice_ticks(lo, hi, 10)
    y_range = (lo, hi)
    if warped:
        add_seam(fig, y_range)
    add_y_chips(fig, ticks, lambda v: "%+g" % v)
    layout = panel_layout(280, axis)
    layout["yaxis"] = y_axis_dict(ticks, y_range)
    fig.update_layout(**layout)
    return fig


def chart_figs(p):
    """The tab's three panels as (title, figure) pairs
    (user direction 2026-11-10 — the IALT treatment
    extended on the user's "do the same for other plots"):
    the decade panel stands 560 tall so the main plot
    claims the space, the tape and dollar panels 280 each;
    all three ride the shared warped axis (last 30 days
    at 25%)."""
    f, s = p["frame"], p["snap"]
    ratios = {"xus": f["xus"], "dev": f["dev"], "em": f["em"]}
    allv = [v for vals in ratios.values() for v in vals]
    lo, hi = min(allv) - 2.0, max(allv) + 3.0
    main = warped_panel(
        f["dates"], ratios, _COLORS, _FULL, _SHORT,
        height=560, y_ticks=nice_ticks(lo, hi, 10),
        y_range=(lo, hi), fmt_y=lambda v: "%g" % v,
        end_labels={k: "%+.1f%%" % s[k]["chg10y"]
                    for k in ratios},
        bold=("xus",), hline=100, min_sep=5.0)
    both = {"ratio": f["ratio"], "dxy": f["dxy"]}
    allv = [v for vals in both.values() for v in vals
            if v is not None]
    lo3, hi3 = min(allv) - 2.0, max(allv) + 3.0
    third = warped_panel(
        f["dates"], both, _COLORS, _FULL, _SHORT,
        height=280, y_ticks=nice_ticks(lo3, hi3, 10),
        y_range=(lo3, hi3), fmt_y=lambda v: "%g" % v,
        end_labels={}, min_sep=5.0)
    return [("A decade of US dominance — relative strength "
             "of ex-US vs the US, 10 years", main),
            ("The shift tape — rolling 12-month return, "
             "all ex-US minus US", _spread_fig(p)),
            ("The shift rides the dollar — ex-US relative "
             "strength vs the dollar (corr %+.2f)"
             % p["corr_dxy"], third)]


def us_xus_head_html(p, board_date=None):
    """The as-of note + chips, in the centered 1120px block
    the page's tabs share. With `board_date`, the as-of
    note carries the freshness pill (canonical
    toolbox.DataZoo.market.freshness). The Plotly panels
    render separately (chart_figs), centered on the same
    measure by the caller, and us_xus_foot_html closes
    the tab."""
    from toolbox.DataZoo.market.freshness import lag_pill_html

    pill = lag_pill_html(board_date, p.get("last_date"))
    r1, r1m = p["ret1y"], p["ret1m"]
    s = p["snap"]
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
    note = (
        '<div style="font-size:12px;color:%s;margin:2px 0 10px">'
        'As of <span style="color:%s;font-weight:700">%s close'
        '</span>%s &middot; total-return ETF proxies</div>'
        % (_MUTE, _ACCENT, html.escape(p["asof"]), pill))
    return ('<div style="max-width:1120px;margin:0 auto">'
            + note
            + '<div style="display:flex;gap:8px;flex-wrap:wrap;'
              'margin-bottom:12px">' + chip_html
            + "</div></div>")


def us_xus_foot_html():
    """The honest-method footnote, in the tab's centered
    block."""
    return (
        '<div style="max-width:1120px;margin:0 auto">'
        '<div style="font-size:11.5px;color:#8A8378;'
        'margin-top:8px">'
        'Proxies: SPY (US), EFA (developed ex-US), EEM '
        '(emerging), ACWX (all ex-US), on adjusted closes '
        '(dividends reinvested). Read: the decade panel '
        'shows what the shift is shifting FROM; the tape '
        'turns positive when ex-US outruns the US over a '
        'rolling year; the claim "a shift is under way" '
        'needs the tape positive AND the ratio above its '
        '200-day average — the chips report both, live.'
        '</div></div>')
