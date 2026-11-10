"""IALT — the Markets page Alternatives section's second tab
(user-approved demo 2026-11-02).

    fetch_ialt_tab()   IALT + SPY + AGG + VOO + the section's
                       seven peer funds (Yahoo adjusted closes)
    ialt_head_html()   the as-of note + stat chips (HTML)
    chart_figs()       the three panels as Plotly figures
                       (user direction 2026-11-10, demoed and
                       approved the same day — replacing the
                       SVG panels this module rendered before)
    ialt_foot_html()   the short-history footnote (HTML)

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
from toolbox.DataZoo.market.yahoo_prices import with_retry, yahoo_daily

_MUTE = "#9AA0A6"
_ACCENT = "#FFB066"
PEERS = ("QAI", "HDG", "MNA", "DBMF", "CTA", "KMLM", "BTAL")
_SYMS = ("IALT", "SPY", "AGG", "VOO") + PEERS

# Warped time axis for the Plotly panels (user direction
# 2026-11-10, demoed and approved — the sentiment chart's
# treatment, same constants): the last TAIL_DAYS of each
# panel's window take TAIL_SHARE of the axis width and
# everything earlier compresses into the rest, with a
# dotted seam where the stretch begins.
TAIL_DAYS = 30
TAIL_SHARE = 0.25


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
# Short legend names: the full names would wrap the legend
# into one row per series on a phone and eat the plot height
# the chart is meant to claim. Hovers carry the full names.
_SHORT = {"ialt": "IALT", "spy": "SPY", "agg": "AGG",
          "peers": "Peers", "dd": "IALT", "dd_voo": "VOO",
          "cspy": "vs SPY", "cagg": "vs AGG"}
# Plotly dark-skin palette: the figures render transparent
# and blend into the themed page, so series hues are the
# SVG panels' hues brightened where the dark green peer
# line would sink (the sentiment chart's approach).
_PLOT_COLORS = {"ialt": _ACCENT, "spy": "#7FB3D5",
                "agg": "#9AA0A6", "peers": "#35B26F",
                "dd": "#E23B3E", "dd_voo": "#7FB3D5",
                "cspy": "#7FB3D5", "cagg": "#9AA0A6"}
_INK = "#E8ECEC"
_PLOT_MUTE = "#A7B3B3"
_BOX = "#0B1118"


def _warp_setup(dates):
    """The panel's warped time coordinate: day offsets from
    the first date mapped piecewise — [start, cut] onto
    [0, 1 - TAIL_SHARE], [cut, end] onto [1 - TAIL_SHARE, 1],
    cut = end - TAIL_DAYS (the sentiment chart's warp, same
    constants). Windows no longer than the tail skip the
    warp. Returns (index, cut, warp_fn, warped)."""
    import pandas as pd

    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    x0, x1 = idx[0], idx[-1]
    span = (x1 - x0).total_seconds() / 86400.0
    cut = x1 - pd.Timedelta(days=TAIL_DAYS)
    cut_day = (cut - x0).total_seconds() / 86400.0
    warped = span > TAIL_DAYS

    def _warp(ix):
        days = (pd.DatetimeIndex(ix)
                - x0).total_seconds() / 86400.0
        if span <= 0:
            return [0.0] * len(days)
        if not warped:
            return [float(d) / span for d in days]
        return [float(d) / cut_day * (1.0 - TAIL_SHARE)
                if d <= cut_day else
                (1.0 - TAIL_SHARE) + (float(d) - cut_day)
                / (span - cut_day) * TAIL_SHARE
                for d in days]

    return idx, cut, _warp, warped


def _x_axis(dates):
    """The warped axis dict + warp function: month starts in
    the compressed head ("%b %y"), skipping any within 15
    days of the seam so its label cannot collide, then the
    seam date and the last date as "%b %d". The axis spans
    exactly the data (range [0, 1] IS first-to-last
    observation under the warp)."""
    import pandas as pd

    idx, cut, _warp, warped = _warp_setup(dates)
    x0, x1 = idx[0], idx[-1]

    def one(ts):
        return _warp(pd.DatetimeIndex([ts]))[0]

    if warped:
        vals, text = [], []
        m = pd.Timestamp(x0.year, x0.month, 1)
        while m <= x1:
            if x0 <= m < cut - pd.Timedelta(days=15):
                vals.append(one(m))
                text.append(m.strftime("%b %y"))
            m += pd.DateOffset(months=1)
        vals += [1 - TAIL_SHARE, 1.0]
        text += [cut.strftime("%b %d"), x1.strftime("%b %d")]
    else:
        vals = [0.0, 0.5, 1.0]
        text = [d.strftime("%b %d")
                for d in (x0, x0 + (x1 - x0) / 2, x1)]
    axis = dict(range=[0.0, 1.0], tickmode="array",
                tickvals=vals, ticktext=text,
                tickfont=dict(color=_INK, size=10),
                ticklabeloverflow="allow", showgrid=False,
                automargin=False)
    return axis, _warp, warped


def _fig_panel(dates, series, height, y_ticks, y_range, fmt_y,
               end_labels, zero=False, fill_key=None, bold=(),
               min_sep=None, note_at_top=True):
    """One Plotly panel on the warped axis, in the sentiment
    chart's space-claiming layout (user direction
    2026-11-10): margins 10/26/10/10 with automargin off (the
    right 26 fits the last date label's right half, which
    otherwise clips), no axis titles, the y scale printed
    inside on dark chips, the legend horizontal under the
    plot, endpoint labels as dark chips spread so converged
    endpoints never stack (the canonical spreader in
    sentiment), and the dotted warp seam with its note.
    ``series`` maps key -> values; ``end_labels`` maps key ->
    endpoint chip text."""
    import pandas as pd
    import plotly.graph_objects as go

    from toolbox.DataZoo.market.sentiment import _spread_labels

    xaxis, _warp, warped = _x_axis(dates)
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    fig = go.Figure()
    for key, vals in series.items():
        pts = [(d, v) for d, v in zip(idx, vals)
               if v is not None]
        kw = {}
        if fill_key == key:
            kw = dict(fill="tozeroy",
                      fillcolor="rgba(226,59,62,0.30)")
        fig.add_trace(go.Scatter(
            x=_warp([q[0] for q in pts]),
            y=[q[1] for q in pts],
            mode="lines", name=_SHORT[key], **kw,
            line=dict(color=_PLOT_COLORS[key],
                      width=2.2 if key in bold else 1.5),
            customdata=[q[0].strftime("%b %d, %Y")
                        for q in pts],
            hovertemplate="%{customdata}<br>" + _FULL[key]
                          + ": %{y:.2f}<extra></extra>"))
    if zero:
        fig.add_shape(type="line", xref="x", yref="y",
                      x0=0.0, x1=1.0, y0=0, y1=0,
                      line=dict(color="rgba(232,236,236,0.5)",
                                width=1, dash="dash"))
    if warped:
        fig.add_shape(type="line", xref="x", yref="paper",
                      x0=1 - TAIL_SHARE, x1=1 - TAIL_SHARE,
                      y0=0, y1=1,
                      line=dict(
                          color="rgba(232,236,236,0.45)",
                          width=1, dash="dot"))
        fig.add_annotation(
            x=1 - TAIL_SHARE,
            y=y_range[1] if note_at_top else y_range[0],
            xref="x", yref="y",
            text="last 30 days \u2192", showarrow=False,
            xanchor="left",
            yanchor="top" if note_at_top else "bottom",
            xshift=4, yshift=-2 if note_at_top else 2,
            font=dict(size=9.5, color=_PLOT_MUTE))
    ends = [(k, series[k][-1]) for k in end_labels
            if series[k][-1] is not None]
    if ends:
        sep = min_sep or (y_range[1] - y_range[0]) * 0.07
        placed = _spread_labels(
            [float(v) for _k, v in ends],
            lo=y_range[0] + sep * 0.4,
            hi=y_range[1] - sep * 0.4, min_sep=sep)
        xw = _warp([idx[-1]])[0]
        xw_lbl = _warp([idx[-1] - pd.Timedelta(days=3)])[0]
        for (key, v), ly in zip(ends, placed):
            fig.add_trace(go.Scatter(
                x=[xw], y=[v], mode="markers",
                showlegend=False, hoverinfo="skip",
                marker=dict(color=_PLOT_COLORS[key], size=8,
                            line=dict(color="white",
                                      width=1))))
            fig.add_annotation(
                x=xw_lbl, y=ly, text=end_labels[key],
                showarrow=False, xanchor="right",
                yanchor="middle",
                font=dict(size=11, color=_PLOT_COLORS[key]),
                bgcolor=_BOX, bordercolor=_PLOT_COLORS[key],
                borderwidth=1, borderpad=3)
    for v in y_ticks:
        fig.add_annotation(
            x=0.0, y=v, xref="x", yref="y", text=fmt_y(v),
            showarrow=False, xanchor="left",
            yanchor="middle", xshift=4,
            font=dict(size=10, color=_INK),
            bgcolor="rgba(11,17,24,0.72)", borderpad=2)
    fig.update_layout(
        height=height, margin=dict(l=10, r=26, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_INK), hovermode="x unified",
        legend=dict(orientation="h", x=0, y=-0.09,
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(color=_INK, size=10.5)),
        xaxis=xaxis,
        yaxis=dict(range=list(y_range),
                   tickvals=list(y_ticks),
                   showticklabels=False, zeroline=False,
                   showline=False, automargin=False,
                   tickfont=dict(color=_INK),
                   gridcolor="rgba(232,236,236,0.25)",
                   griddash="dash"))
    return fig


def _nice_ticks(lo, hi, step):
    import math

    t = math.ceil(lo / step) * step
    out = []
    while t <= hi:
        out.append(round(t, 6))
        t += step
    return out


def chart_figs(p):
    """The tab's three panels as (title, figure) pairs in
    display order (user direction 2026-11-10, demoed and
    approved): the total-return panel stands 560 tall so the
    main plot claims the space, the drawdown and correlation
    panels 280 each. A payload without a VOO leg (a last-good
    cached by the pre-VOO code) draws the drawdown panel
    IALT-only, and a payload without rolling correlations
    omits that panel — the tolerances the SVG render carried.
    """
    paths = p["paths"]
    allv = [v for s in paths.values() for v in s]
    # Tight y bounds (ticks stay on the 5-grid inside them):
    # the range hugs the data so no strip is wasted above
    # the peak or below the trough.
    lo, hi = min(allv) - 1.0, max(allv) + 1.8
    figs = [(
        "Since inception (%s) — total return, rebased to 100"
        % p["inception"],
        _fig_panel(
            p["dates"], paths, height=560,
            y_ticks=_nice_ticks(lo, hi, 5), y_range=(lo, hi),
            fmt_y=lambda v: "%g" % v,
            end_labels={k: "%+.1f%%" % p["ret"][k]
                        for k in paths},
            bold=("ialt",), min_sep=4.0))]
    dd_series = {"dd": p["dd"]}
    dd_labels = {"dd": "%.1f%%" % p["cur_dd"]}
    if p.get("dd_voo"):
        dd_series["dd_voo"] = p["dd_voo"]
        dd_labels["dd_voo"] = "%.1f%%" % p["voo_cur_dd"]
    dlo = min(p["dd"] + (p.get("dd_voo") or [])) - 0.5
    figs.append((
        "The diversifier's ride — IALT drawdown since "
        "inception",
        _fig_panel(
            p["dates"], dd_series, height=280,
            y_ticks=_nice_ticks(dlo, 0, 2.5),
            y_range=(dlo, 0.6),
            fmt_y=lambda v: "%g%%" % v,
            end_labels=dd_labels, zero=True, fill_key="dd",
            bold=("dd",), min_sep=1.6)))
    if p.get("roll"):
        figs.append((
            "Rolling 63-day correlation — IALT vs stocks "
            "and bonds",
            _fig_panel(
                p["roll_dates"],
                {"cspy": p["roll"]["spy"],
                 "cagg": p["roll"]["agg"]},
                height=280, y_ticks=[0, 0.25, 0.5, 0.75],
                y_range=(-0.18, 0.92),
                fmt_y=lambda v: "%g" % v,
                end_labels={
                    k: "%+.2f" % v for k, v in (
                        ("cspy", p["roll_last"]["spy"]),
                        ("cagg", p["roll_last"]["agg"]))},
                zero=True, min_sep=0.14, note_at_top=False)))
    return figs


def ialt_head_html(p, board_date=None):
    """The as-of note + stat chips, in the centered 1120px
    block the page's tabs share. With `board_date`, the
    as-of note carries the freshness pill (canonical
    toolbox.DataZoo.market.freshness) marking any lag
    behind the board date. The Plotly panels render
    separately (chart_figs), centered on the same measure
    by the caller, and ialt_foot_html closes the tab."""
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
    note = (
        '<div style="font-size:12px;color:%s;margin:2px 0 10px">'
        'As of <span style="color:%s;font-weight:700">%s close'
        '</span>%s &middot; IALT history begins %s — every '
        'figure is since inception</div>'
        % (_MUTE, _ACCENT, html.escape(p["asof"]), pill,
           html.escape(p["inception"])))
    return ('<div style="max-width:1120px;margin:0 auto">'
            + note
            + '<div style="display:flex;gap:8px;flex-wrap:wrap;'
              'margin-bottom:12px">' + chip_html
            + "</div></div>")


def ialt_foot_html():
    """The honest short-history footnote, in the tab's
    centered 1120px block."""
    return (
        '<div style="max-width:1120px;margin:0 auto">'
        '<div style="font-size:11.5px;color:#8A8378;'
        'margin-top:8px">'
        'Adjusted closes (dividends reinvested), Yahoo. IALT is '
        'BlackRock\'s systematic multi-strategy alternatives '
        'fund (long/short equity, macro, event-driven on a '
        'T-bill collateral base). Its own listed history is '
        'under a year, so nothing here is a 1Y/3Y/5Y figure and '
        'the Sharpe and correlations carry a short window — '
        'read them as the fund\'s opening chapter, not its '
        'record. Peers: the section\'s other seven funds (QAI, '
        'HDG, MNA, DBMF, CTA, KMLM, BTAL), equal-weighted '
        'daily.</div></div>')
