"""Shared warped-axis Plotly line panels for the Markets
tabs (user directions 2026-11-10, demoed and approved):
IALT first, then the dollar and US vs ex-US on the user's
"do the same for other plots" — one canonical builder so
every tab's charts share the treatment instead of drifting
copies of it.

The treatment: the x axis is time warped so the last
TAIL_DAYS days take TAIL_SHARE of the width (a dotted seam
marks the join, real dates ride in hover via customdata);
the layout claims the maximum space for the plot itself —
margins 10/26/10/10 with automargin off (the right 26 fits
the last date label's right half, which otherwise clips),
no axis titles, the y scale printed inside the plot on
dark chips, the legend horizontal under the plot with
short names, and endpoint values as dark chips spread so
converged endpoints never stack (the canonical spreader
in sentiment). Tick labels adapt to the window (demo
finding 2026-11-10): month starts on windows up to two
years, Jan 1 year labels beyond that — ~120 month labels
crammed into a 10-year window's compressed head garble
into scribble.
"""
import pandas as pd

TAIL_DAYS = 30
TAIL_SHARE = 0.25

_INK = "#E8ECEC"
_PLOT_MUTE = "#9AA0A6"
_BOX = "rgba(11,17,24,0.85)"


def nice_ticks(lo, hi, step):
    import math

    t = math.ceil(lo / step) * step
    out = []
    while t <= hi:
        out.append(round(t, 6))
        t += step
    return out


def warp_setup(dates):
    """(index, cut, warp, warped): the piecewise-linear
    day-count warp — [start, cut] maps onto [0, 1 -
    TAIL_SHARE] and [cut, end] onto [1 - TAIL_SHARE, 1],
    cut = end - TAIL_DAYS. Windows of TAIL_DAYS or less
    stay linear (warped False)."""
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


def x_axis(dates):
    """The warped axis dict + warp function. Head ticks:
    month starts ("%b %y") on windows up to two years,
    skipping any within 15 days of the seam so its label
    cannot collide; Jan 1 year labels ("%Y") beyond two
    years. Then the seam date and the last date as
    "%b %d". The axis spans exactly the data (range [0, 1]
    IS first-to-last observation under the warp)."""
    idx, cut, _warp, warped = warp_setup(dates)
    x0, x1 = idx[0], idx[-1]
    span_days = (x1 - x0).total_seconds() / 86400.0

    def one(ts):
        return _warp(pd.DatetimeIndex([ts]))[0]

    if warped and span_days > 730:
        vals, text = [], []
        y = pd.Timestamp(x0.year + 1, 1, 1)
        while y <= x1:
            if y < cut - pd.Timedelta(days=30):
                vals.append(one(y))
                text.append(y.strftime("%Y"))
            y += pd.DateOffset(years=1)
        vals += [1 - TAIL_SHARE, 1.0]
        text += [cut.strftime("%b %d"), x1.strftime("%b %d")]
    elif warped:
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


def panel_layout(height, xaxis):
    """The shared layout kwargs (margins, transparent
    paper, unified hover, the under-plot legend), for
    panels built by hand on the same axis — e.g. an area
    panel whose traces the line builder does not cover."""
    return dict(
        height=height, margin=dict(l=10, r=26, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_INK), hovermode="x unified",
        legend=dict(orientation="h", x=0, y=-0.09,
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(color=_INK, size=10.5)),
        xaxis=xaxis)


def y_axis_dict(y_ticks, y_range):
    return dict(range=list(y_range), tickvals=list(y_ticks),
                showticklabels=False, zeroline=False,
                showline=False, automargin=False,
                tickfont=dict(color=_INK),
                gridcolor="rgba(232,236,236,0.25)",
                griddash="dash")


def add_seam(fig, y_range, note_at_top=True):
    """The dotted warp seam + its note (top, or bottom
    when a panel's top right belongs to a chip)."""
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=1 - TAIL_SHARE, x1=1 - TAIL_SHARE,
                  y0=0, y1=1,
                  line=dict(color="rgba(232,236,236,0.45)",
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


def add_y_chips(fig, y_ticks, fmt_y, x_at=0.0):
    """The y scale printed inside the plot on dark chips
    (the native tick labels are off — they would cost the
    plot its left strip). x_at is the data x the chips
    anchor at — 0.0 on the warped unit axis; a panel on
    another axis (the election cycle's -252..252) passes
    its left edge."""
    for v in y_ticks:
        fig.add_annotation(
            x=x_at, y=v, xref="x", yref="y", text=fmt_y(v),
            showarrow=False, xanchor="left",
            yanchor="middle", xshift=4,
            font=dict(size=10, color=_INK),
            bgcolor="rgba(11,17,24,0.72)", borderpad=2)


def warped_panel(dates, series, colors, full_names,
                 short_names, *, height, y_ticks, y_range, fmt_y,
                 end_labels, zero=False, hline=None,
                 fill_key=None, fillcolor=None, bold=(),
                 dash=(), min_sep=None, note_at_top=True):
    """One Plotly line panel on the warped axis (see the
    module docstring for the treatment). ``series`` maps
    key -> values aligned to ``dates``; ``colors`` /
    ``full_names`` / ``short_names`` map the same keys;
    ``end_labels`` maps key -> endpoint chip text (keys
    absent or with a None endpoint get no chip). Only the
    five data arguments are positional — every panel knob
    after ``short_names`` is keyword-only, so a knob can
    never land in the wrong slot positionally."""
    import plotly.graph_objects as go

    from toolbox.DataZoo.market.sentiment import _spread_labels

    xaxis, _warp, warped = x_axis(dates)
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    fig = go.Figure()
    for key, vals in series.items():
        pts = [(d, v) for d, v in zip(idx, vals)
               if v is not None]
        kw = {}
        if fill_key == key:
            kw = dict(fill="tozeroy", fillcolor=fillcolor)
        line = dict(color=colors[key],
                    width=2.2 if key in bold else 1.5)
        if key in dash:
            line["dash"] = "dash"
        fig.add_trace(go.Scatter(
            x=_warp([q[0] for q in pts]),
            y=[q[1] for q in pts],
            mode="lines", name=short_names[key], **kw,
            line=line,
            customdata=[q[0].strftime("%b %d, %Y")
                        for q in pts],
            hovertemplate="%{customdata}<br>"
                          + full_names[key]
                          + ": %{y:.2f}<extra></extra>"))
    ref = hline if hline is not None else (0 if zero else None)
    if ref is not None:
        fig.add_shape(type="line", xref="x", yref="y",
                      x0=0.0, x1=1.0, y0=ref, y1=ref,
                      line=dict(color="rgba(232,236,236,0.5)",
                                width=1, dash="dash"))
    if warped:
        add_seam(fig, y_range, note_at_top)
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
                marker=dict(color=colors[key], size=8,
                            line=dict(color="white",
                                      width=1))))
            fig.add_annotation(
                x=xw_lbl, y=ly, text=end_labels[key],
                showarrow=False, xanchor="right",
                yanchor="middle",
                font=dict(size=11, color=colors[key]),
                bgcolor=_BOX, bordercolor=colors[key],
                borderwidth=1, borderpad=3)
    add_y_chips(fig, y_ticks, fmt_y)
    layout = panel_layout(height, xaxis)
    layout["yaxis"] = y_axis_dict(y_ticks, y_range)
    fig.update_layout(**layout)
    return fig
