"""Market sentiment: Fear & Greed (US stocks + crypto) and gold.

The canonical home of the sentiment block that lived in the
morning brief's Sentiment subsection until 2026-10-31, when the
user moved it to the Investments dashboard's Markets page (a
"Market sentiments" section after Market talk) and deleted it
from the briefings.

Sources (all public, no auth; fetched fresh on each call):
  CNN    https://production.dataviz.cnn.io/index/fearandgreed/graphdata
  Crypto https://api.alternative.me/fng/?limit=365
  Gold   GC=F gold futures via the Yahoo v8 chart API

The CNN and crypto series build on the canonical raw fetchers in
:mod:`toolbox.DataZoo.market.fear_greed` (``fetch_cnn_graph`` /
``fetch_crypto_entries``) — one fetch layer for both modules; only
the gold fetch and the chart rendering live here.

:func:`fetch_sentiment` returns the three one-year series plus
the latest readings; any source that fails is omitted (the
caller degrades), and the result is None only when nothing at
all could be fetched. :func:`chart_fig` renders the one-year
banded chart (zone bands, both Fear & Greed lines, gold
min-max normalized with its peak at GOLD_PEAK on a right axis
labeled in $/oz) as a Plotly figure (user direction
2026-11-10: the chart in Plotly, replacing the matplotlib
PNG this module rendered before).
"""
import datetime
import json
import urllib.request

import pandas as pd

from toolbox.DataZoo.market.fear_greed import (
    fetch_cnn_graph,
    fetch_crypto_entries,
)

UA = {"User-Agent": "toolbox-sentiment"}

CNN_COLOR = "#1a3c6e"
CRYPTO_COLOR = "#8e44ad"
GOLD_COLOR = "#b8860b"
BANDS = [(0, 25, "Extreme Fear", "#e74c3c"), (25, 45, "Fear", "#e67e22"),
         (45, 55, "Neutral", "#f1c40f"), (55, 75, "Greed", "#82c46a"),
         (75, 100, "Extreme Greed", "#27ae60")]

# Chart geometry (see chart_fig): the index axis runs 0-115 so
# the 0-100 scale ceiling carries 15% headroom, and the gold
# series is normalized with its window peak at GOLD_PEAK on its
# own 0-100 axis — the axis top then sits at the gold max plus
# (100/GOLD_PEAK - 1) ≈ 17.6% of the gold range, and the peak
# itself renders below the band ceiling instead of on it.
GOLD_PEAK = 85.0
# Endpoint labels: minimum vertical separation (index units)
# between two label centers, and the band they must stay in.
LABEL_MIN_SEP = 8.0
LABEL_LO, LABEL_HI = 4.0, 112.0


def _spread_labels(desired, lo=LABEL_LO, hi=LABEL_HI,
                   min_sep=LABEL_MIN_SEP):
    """Adjust endpoint-label center positions (index units) so
    no two sit closer than ``min_sep``, preserving their order,
    clamped into [lo, hi]. Converged endpoints (e.g. crypto 26
    vs CNN 23) would otherwise draw their labels on top of each
    other at identical offsets."""
    order = sorted(range(len(desired)), key=lambda i: desired[i])
    pos = [float(desired[i]) for i in order]
    for j in range(1, len(pos)):
        pos[j] = max(pos[j], pos[j - 1] + min_sep)
    if pos and pos[-1] > hi:
        pos[-1] = float(hi)
        for j in range(len(pos) - 2, -1, -1):
            pos[j] = min(pos[j], pos[j + 1] - min_sep)
    if pos and pos[0] < lo:
        pos[0] = float(lo)
        for j in range(1, len(pos)):
            pos[j] = max(pos[j], pos[j - 1] + min_sep)
    out = [0.0] * len(desired)
    for j, i in enumerate(order):
        out[i] = pos[j]
    return out


def zone(v):
    """The Fear & Greed zone label for a 0-100 value, read
    straight from BANDS so the two can never drift apart."""
    return next((label for _lo, hi, label, _c in BANDS if v <= hi),
                BANDS[-1][2])


def _get_json(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers=headers or UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _daily_series(vals, epochs, unit, tz=None):
    """Daily series from epoch stamps. With ``tz``, the stamps are
    instants and are dated in that zone; without it, the stamps are
    day-keys whose UTC date is the day itself (see fetch_cnn)."""
    dates = pd.to_datetime(epochs, unit=unit, utc=True)
    if tz is None:
        dates = dates.tz_convert(None)
    else:
        dates = dates.tz_convert(tz).tz_localize(None)
    return pd.Series(vals, index=dates.normalize())


def fetch_cnn():
    """CNN Fear & Greed (US stocks), last 365 days, trading days.

    CNN stamps each daily rating with its session date (historical
    points at 00:00 UTC of the session date, the live point late
    that same UTC day), so the UTC date IS the Eastern session
    date — converting zones would misdate every point a day early.
    The series therefore keeps its UTC day-keys."""
    d = fetch_cnn_graph(timeout=30)
    pts = d["fear_and_greed_historical"]["data"]
    cut = (datetime.datetime.now(datetime.timezone.utc)
           - datetime.timedelta(days=365)).timestamp() * 1000
    pts = [p for p in pts if p["x"] >= cut]
    return _daily_series([int(round(p["y"])) for p in pts],
                         [p["x"] for p in pts], "ms")


def _cap_to_eastern_today(s):
    """Re-date a final point that is future-dated in Eastern
    terms to today's Eastern date (user direction 2026-11-02:
    "let's use EST"). A UTC-day-keyed index is published at
    19:00 Eastern on the evening BEFORE its UTC day, and a
    futures bar trades this Eastern evening under tomorrow's
    session date — left alone, both wear tomorrow's date all
    evening. The newest reading is tonight's, and it is dated
    tonight. Only a final point AHEAD of the Eastern date
    moves (merging keep-last into today's point when one
    exists); history is untouched, and the source day returns
    on its own tomorrow, when the Eastern date reaches it.
    The companion rule stands (same day, user-caught): a
    day-keyed series is never dated BEHIND its source day
    either — see fetch_crypto."""
    if not len(s):
        return s
    from toolbox.DataZoo.market.eastern import ET

    today = datetime.datetime.now(ET).date()
    if s.index[-1].date() <= today:
        return s
    idx = list(s.index)
    idx[-1] = pd.Timestamp(today)
    out = pd.Series(list(s.values), index=idx)
    return out[~out.index.duplicated(keep="last")].sort_index()


def fetch_crypto():
    """Crypto Fear & Greed (alternative.me), 365 calendar days.

    alternative.me stamps each value at 00:00 UTC of the day the
    index belongs to — a day-key, not an instant: the source
    itself presents the 2026-11-02 stamp as the Nov 2 (current)
    reading. The series therefore keeps its UTC day-keys, like
    CNN's. (From 2026-11-01 to 2026-11-02 the stamps were dated
    in Eastern as instants, which labelled every value a day
    early — the current reading showed as yesterday's; the user
    caught it. Day-keyed daily series keep their source day;
    the app's Eastern rule governs true instants.) The final
    point is Eastern-capped (see _cap_to_eastern_today): the
    index for tomorrow's UTC day is published at 19:00 Eastern
    tonight and is tonight's reading.
    """
    rows = fetch_crypto_entries(365, timeout=30)
    return _cap_to_eastern_today(_daily_series(
        [int(r["value"]) for r in rows],
        [int(r["timestamp"]) for r in rows], "s"))


def fetch_gold():
    """GC=F gold futures $/oz, last year of daily closes.

    Through the canonical yahoo_daily since 2026-11-02 (this
    function's private chart fetch was the last copy of the
    lagging-bar disease: Yahoo leaves a completed session's
    bar null for hours, and a private dropna read a day
    stale). Yahoo stamps futures bars at midnight Eastern,
    so yahoo_daily's UTC-normalized dates are the exchange
    dates; include_partial keeps the live session's bar, as
    the old range fetch did. The final point is Eastern-
    capped (see _cap_to_eastern_today): the live bar trades
    this Eastern evening under tomorrow's session date."""
    import time as _time

    from toolbox.DataZoo.market.yahoo_prices import yahoo_daily

    s = yahoo_daily("GC=F",
                    p1=int(_time.time()) - 400 * 86400,
                    include_partial=True)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return _cap_to_eastern_today(s)


def fetch_sentiment(progress=None):
    """All three series + latest readings, or None if every
    source failed. Each present key of ``latest`` is a dict with
    the value (and zone, for the two indices).

    ``progress`` (optional) is a callable invoked as
    ``progress(done, total, source_label)``: once before the
    sources with ``done == 0``, then after each source attempt
    (failures included — the attempt completed). For callers
    that surface progress; the default ``None`` reports
    nothing."""
    series = {}
    sources = (("cnn", "CNN Fear & Greed", fetch_cnn),
               ("crypto", "Crypto Fear & Greed", fetch_crypto),
               ("gold", "Gold", fetch_gold))
    if progress is not None:
        progress(0, len(sources), "")
    for i, (name, label, fn) in enumerate(sources, start=1):
        try:
            s = fn().dropna()
            if len(s) >= 30:
                series[name] = s
        except Exception:  # noqa: BLE001 - degrade per source
            pass
        if progress is not None:
            progress(i, len(sources), label)
    if not series:
        return None
    latest = {}
    for name in ("cnn", "crypto"):
        if name in series:
            v = float(series[name].iloc[-1])
            latest[name] = {"value": v, "zone": zone(v)}
    if "gold" in series:
        latest["gold"] = {"value": float(series["gold"].iloc[-1])}
    # No fetch-time "asof" (removed 2026-11-02): the tab dates
    # each series by its own last measurement, like the other
    # tabs — a pull-time stamp here invited exactly the
    # mismatch the freshness stamps exist to prevent.
    return {"series": series, "latest": latest}


def chart_fig(series):
    """The one-year banded sentiment chart as a Plotly figure
    (user direction 2026-11-10, replacing the matplotlib PNG
    this module rendered before). Same construction as the
    PNG drew: zone bands with their labels, both Fear & Greed
    lines, gold min-max normalized with its peak at GOLD_PEAK
    on a right axis labeled in $/oz, and endpoint labels with
    value \u00b7 zone only \u2014 no dates in the plot (user
    direction 2026-11-02): each series' date lives in the
    caller's freshness stamp line above the chart."""
    import plotly.graph_objects as go

    # Dark-skin palette (user direction 2026-11-04): the
    # chart renders transparent and blends into the app's
    # themed page, so its ink is the page's ink \u2014 snow text,
    # brightened series hues (the module constants stay the
    # HTML tab's darker ink), dark glass label boxes.
    _INK = "#E8ECEC"
    _MUTE = "#A7B3B3"
    _BOX = "#0B1118"
    _CNN = "#5B8DEF"
    _CRYPTO = "#B678D8"
    _GOLD = "#E3B341"

    fig = go.Figure()
    x0 = min(s.index[0] for s in series.values())
    x1 = max(s.index[-1] for s in series.values())
    for lo, hi, label, color in BANDS:
        fig.add_shape(
            type="rect", xref="x", yref="y", x0=x0, x1=x1,
            y0=lo, y1=hi, fillcolor=color, opacity=0.14,
            line=dict(width=0), layer="below")
        fig.add_annotation(
            x=x0, y=(lo + hi) / 2, text=label, showarrow=False,
            xanchor="left", yanchor="middle", xshift=4,
            font=dict(size=9, color=_MUTE))
    if "cnn" in series:
        s = series["cnn"]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            name="CNN (US stocks)",
            line=dict(color=_CNN, width=1.6)))
    if "crypto" in series:
        s = series["crypto"]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            name="Crypto (alternative.me)", opacity=0.9,
            line=dict(color=_CRYPTO, width=1.6)))
    if "gold" in series:
        g = series["gold"]
        gmin, gmax = float(g.min()), float(g.max())
        # Headroom repair (2026-11-02, second round, carried
        # from the PNG): normalizing the peak to exactly 100
        # put it on the band ceiling, the top gridline and the
        # top gold tick at once, so the Jan\u2013Feb spikes read as
        # truncated even with the axis extended to 115. The
        # peak maps to GOLD_PEAK on a 0-100 gold axis: the
        # axis top is the max plus ~17.6% of the range, and
        # the peak renders below the band ceiling.
        gn = ((g - gmin) / (gmax - gmin) * GOLD_PEAK
              if gmax > gmin else g * 0 + GOLD_PEAK / 2)
        fig.add_trace(go.Scatter(
            x=g.index, y=gn.values, mode="lines", yaxis="y2",
            name="Gold futures (GC=F, $/oz)", opacity=0.95,
            line=dict(color=_GOLD, width=1.6),
            customdata=g.values,
            hovertemplate="%{customdata:$,.0f}/oz<extra>"
                          "Gold</extra>"))
        fig.add_trace(go.Scatter(
            x=[g.index[-1]], y=[gn.iloc[-1]], mode="markers",
            yaxis="y2", showlegend=False, hoverinfo="skip",
            marker=dict(color=_GOLD, size=9,
                        line=dict(color="white", width=1))))
        fig.add_annotation(
            x=g.index[-1], y=float(gn.iloc[-1]), yref="y2",
            text=f"${g.iloc[-1]:,.0f}/oz", showarrow=False,
            xanchor="right", yanchor="middle", ax=-8, ay=-14,
            font=dict(size=11, color=_GOLD),
            bgcolor=_BOX, bordercolor=_GOLD, borderwidth=1,
            borderpad=3)
    endpoints = [(color, series[name])
                 for name, color in (("cnn", _CNN),
                                     ("crypto", _CRYPTO))
                 if name in series]
    if endpoints:
        # Label collision handling (2026-11-02, second round,
        # carried from the PNG): identical offsets put
        # converged endpoints' labels on top of each other
        # (crypto 26 vs CNN 23 live). Labels anchor inside the
        # plot (right-aligned, ending just before each series'
        # last date) at _spread_labels positions \u2014 order
        # preserved, LABEL_MIN_SEP apart.
        placed = _spread_labels(
            [float(s.iloc[-1]) + 3.0 for _c, s in endpoints])
        for (color, s), label_y in zip(endpoints, placed):
            v = float(s.iloc[-1])
            fig.add_trace(go.Scatter(
                x=[s.index[-1]], y=[v], mode="markers",
                showlegend=False, hoverinfo="skip",
                marker=dict(color=color, size=9,
                            line=dict(color="white", width=1))))
            fig.add_annotation(
                x=s.index[-1] - pd.Timedelta(days=3), y=label_y,
                text=f"{v:.0f} \u00b7 {zone(v)}", showarrow=False,
                xanchor="right", yanchor="middle",
                font=dict(size=11, color=color),
                bgcolor=_BOX, bordercolor=color, borderwidth=1,
                borderpad=3)
    # Headroom (2026-11-02 live fix, carried from the PNG):
    # the index axis runs to 115 so the 0-100 scale ceiling
    # carries ~15% headroom and endpoint labels sit clear of
    # the axes; gold carries its own 0-100 axis with the peak
    # at GOLD_PEAK (see above). The zone bands stay 0-100.
    yaxis = dict(range=[0, 115], title=dict(
                     text="Index (0\u2013100)", font=dict(color=_INK)),
                 tickvals=[0, 20, 40, 60, 80, 100],
                 tickfont=dict(color=_INK), zeroline=False,
                 gridcolor="rgba(232,236,236,0.25)",
                 griddash="dash")
    layout = dict(
        height=460, margin=dict(l=46, r=54, t=8, b=26),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_INK), hovermode="x unified",
        legend=dict(x=0.005, y=0.995, xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(11,17,24,0.85)",
                    bordercolor="#3A4646", borderwidth=1,
                    font=dict(color=_INK, size=10)),
        xaxis=dict(tickformat="%b %Y", dtick="M1",
                   tickfont=dict(color=_INK, size=10),
                   showgrid=False),
        yaxis=yaxis)
    if "gold" in series:
        g = series["gold"]
        gmin, gmax = float(g.min()), float(g.max())
        fracs = (0, 0.25, 0.5, 0.75, 1.0)
        layout["yaxis2"] = dict(
            range=[0, 100], overlaying="y", side="right",
            tickvals=[GOLD_PEAK * f for f in fracs],
            ticktext=[f"${gmin + (gmax - gmin) * f:,.0f}"
                      for f in fracs],
            title=dict(text="Gold price ($/oz)",
                       font=dict(color=_GOLD)),
            tickfont=dict(color=_GOLD), showgrid=False,
            zeroline=False)
    fig.update_layout(**layout)
    return fig
