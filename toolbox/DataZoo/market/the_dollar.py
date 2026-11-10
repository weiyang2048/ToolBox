"""The dollar — the Currencies section's second tab (Markets
page), demoed and approved 2026-11-01.

    fetch_the_dollar()   DXY + the G10-and-renminbi spot set
                         (the board's Spot FX group), 10y
    dollar_head_html()   the as-of note + chips
    chart_figs()         the two line panels as Plotly
                         figures on the shared warped axis
                         (plotly_panels); the currency bars
                         panel between them stays SVG
                         (dollar_bars_html) — it is
                         categorical, not a time series
    dollar_foot_html()   the footnote

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

from toolbox.DataZoo.market.eastern import format_day
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

from toolbox.DataZoo.market.plotly_panels import nice_ticks, warped_panel
from toolbox.DataZoo.market.svg_charts import scale, svg_elems


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
        "asof": format_day(usd.index[-1]),
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


_COLORS = {"dxy": "#7FB3D5", "ma": "#9AA0A6",
           "vol": "#7FB3D5", "med": "#9AA0A6"}
_FULL = {"dxy": "Dollar index (DXY)", "ma": "200-day average",
         "vol": "Realized vol", "med": "5y median"}
_SHORT = {"dxy": "DXY", "ma": "200-day avg",
          "vol": "Realized vol", "med": "5y median"}


def chart_figs(p):
    """The tab's two line panels as (title, figure) pairs
    (user direction 2026-11-10, demoed and approved — the
    IALT treatment extended on the user's "do the same for
    other plots"): the DXY panel stands 560 tall so the main
    plot claims the space, the vol panel 280; both ride the
    shared warped axis (last 30 days at 25%). The bars
    panel between them is categorical and stays SVG
    (dollar_bars_html)."""
    allv = [v for s in (p["dxy"], p["ma200"]) for v in s
            if v is not None]
    lo, hi = min(allv) - 1.0, max(allv) + 1.8
    main = warped_panel(
        p["dxy_dates"], {"dxy": p["dxy"], "ma": p["ma200"]},
        _COLORS, _FULL, _SHORT, height=560,
        y_ticks=nice_ticks(lo, hi, 5), y_range=(lo, hi),
        fmt_y=lambda v: "%g" % v,
        end_labels={"dxy": "%.1f" % p["dxy_last"]},
        bold=("dxy",), min_sep=4.0)
    vhi = max(p["vol"]) + 1.0
    vol = warped_panel(
        p["vol_dates"],
        {"vol": p["vol"],
         "med": [p["vol_med"]] * len(p["vol"])},
        _COLORS, _FULL, _SHORT, height=280,
        y_ticks=nice_ticks(0, vhi, 2.5), y_range=(0, vhi),
        fmt_y=lambda v: "%g%%" % v,
        end_labels={"vol": "%.1f%%" % p["vol_now"]},
        bold=("vol",), dash=("med",), min_sep=1.2)
    return [("The dollar cycle — DXY vs its 200-day "
             "average, 10 years", main),
            ("Is FX awake — average realized vol across "
             "the 11 currencies (21d, annualized)", vol)]


def dollar_head_html(p, board_date=None):
    """The as-of note + chips, in the centered 1120px block
    the page's tabs share. With `board_date`, the as-of
    note carries the freshness pill (canonical
    toolbox.DataZoo.market.freshness). The Plotly panels
    render separately (chart_figs), centered on the same
    measure by the caller, with dollar_bars_html between
    them and dollar_foot_html closing the tab."""
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
    note = ('<div style="font-size:12px;color:%s;margin:2px 0 '
            '10px">As of <span style="color:%s;font-weight:700">'
            '%s close</span>%s &middot; every series measured '
            'against USD</div>' % (_MUTE, _ACCENT,
                                   html.escape(p["asof"]), pill))
    return ('<div style="max-width:1120px;margin:0 auto">'
            + note
            + '<div style="display:flex;gap:8px;flex-wrap:wrap;'
              'margin-bottom:12px">' + chip_html
            + "</div></div>")


def dollar_bars_html(p):
    """The currency bars panel (SVG, unchanged) in the
    tab's centered block — it sits between the two Plotly
    panels in display order."""
    return ('<div style="max-width:1120px;margin:0 auto">'
            + _bars_panel(p) + "</div>")


def dollar_foot_html():
    """The footnote, in the tab's centered block."""
    return ('<div style="max-width:1120px;margin:0 auto">'
            '<div style="font-size:11.5px;color:#8A8378;'
            'margin-top:8px">Spot quotes (Yahoo), each '
            'currency as its USD value, so a positive bar '
            'always means that currency strengthened. The '
            'set is the full G10 + renminbi — broader than '
            'a futures list, and the same set the board\'s '
            'Spot FX group carries. Built-in check: the '
            'Danish krone is pegged to the euro, and the two '
            'agree to a tenth at every horizon.</div></div>')
