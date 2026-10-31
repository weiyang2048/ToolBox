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

:func:`fetch_sentiment` returns the three one-year series plus
the latest readings; any source that fails is omitted (the
caller degrades), and the result is None only when nothing at
all could be fetched. :func:`chart_png` renders the one-year
banded chart (zone bands, both Fear & Greed lines, gold
min-max normalized to 0-100 on a right axis labeled in $/oz)
to PNG bytes.
"""
import datetime
import io
import json
import urllib.request
from zoneinfo import ZoneInfo

import pandas as pd

CT = ZoneInfo("America/Chicago")
UA = {"User-Agent": "toolbox-sentiment"}
CNN_UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/120.0 Safari/537.36"),
          "Accept": "application/json",
          "Accept-Language": "en-US,en;q=0.9",
          "Referer": "https://www.cnn.com/markets/fear-and-greed",
          "Origin": "https://www.cnn.com"}

CNN_COLOR = "#1a3c6e"
CRYPTO_COLOR = "#8e44ad"
GOLD_COLOR = "#b8860b"
BANDS = [(0, 25, "Extreme Fear", "#e74c3c"), (25, 45, "Fear", "#e67e22"),
         (45, 55, "Neutral", "#f1c40f"), (55, 75, "Greed", "#82c46a"),
         (75, 100, "Extreme Greed", "#27ae60")]


def zone(v):
    """The Fear & Greed zone label for a 0-100 value, read
    straight from BANDS so the two can never drift apart."""
    return next((label for _lo, hi, label, _c in BANDS if v <= hi),
                BANDS[-1][2])


def _get_json(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers=headers or UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _daily_series(vals, epochs, unit):
    dates = pd.to_datetime(epochs, unit=unit, utc=True).tz_convert(None)
    return pd.Series(vals, index=dates.normalize())


def fetch_cnn():
    """CNN Fear & Greed (US stocks), last 365 days, trading days."""
    d = _get_json("https://production.dataviz.cnn.io/index/"
                  "fearandgreed/graphdata", headers=CNN_UA)
    pts = d["fear_and_greed_historical"]["data"]
    cut = (datetime.datetime.now(datetime.timezone.utc)
           - datetime.timedelta(days=365)).timestamp() * 1000
    pts = [p for p in pts if p["x"] >= cut]
    return _daily_series([int(round(p["y"])) for p in pts],
                         [p["x"] for p in pts], "ms")


def fetch_crypto():
    """Crypto Fear & Greed (alternative.me), 365 calendar days."""
    d = _get_json("https://api.alternative.me/fng/?limit=365")
    rows = sorted(d["data"], key=lambda r: int(r["timestamp"]))
    return _daily_series([int(r["value"]) for r in rows],
                         [int(r["timestamp"]) for r in rows], "s")


def fetch_gold():
    """GC=F gold futures $/oz, last year of daily closes."""
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
           "?range=1y&interval=1d")
    d = _get_json(url, headers={"User-Agent": "Mozilla/5.0"})
    res = d["chart"]["result"][0]
    closes = res["indicators"]["quote"][0]["close"]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True)
    idx = idx.tz_convert("America/Chicago").tz_localize(None).normalize()
    s = pd.Series(closes, index=idx).dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def fetch_sentiment():
    """All three series + latest readings, or None if every
    source failed. Each present key of ``latest`` is a dict with
    the value (and zone, for the two indices)."""
    series = {}
    for name, fn in (("cnn", fetch_cnn), ("crypto", fetch_crypto),
                     ("gold", fetch_gold)):
        try:
            s = fn().dropna()
            if len(s) < 30:
                continue
            series[name] = s
        except Exception:  # noqa: BLE001 - degrade per source
            continue
    if not series:
        return None
    latest = {}
    for name in ("cnn", "crypto"):
        if name in series:
            v = float(series[name].iloc[-1])
            latest[name] = {"value": v, "zone": zone(v)}
    if "gold" in series:
        latest["gold"] = {"value": float(series["gold"].iloc[-1])}
    now = datetime.datetime.now(CT)
    return {"series": series, "latest": latest,
            "asof": now.strftime("%a %b %-d, %-I:%M %p CT")}


def chart_png(series):
    """The one-year banded sentiment chart as PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(12, 5.6))
    x0 = min(s.index[0] for s in series.values())
    for lo, hi, label, color in BANDS:
        ax.axhspan(lo, hi, color=color, alpha=0.14, linewidth=0)
        ax.text(x0, (lo + hi) / 2, label, fontsize=8, color="#555555",
                va="center", ha="left", alpha=0.9)
    if "cnn" in series:
        s = series["cnn"]
        ax.plot(s.index, s.values, color=CNN_COLOR, linewidth=1.6,
                label="CNN (US stocks)")
    if "crypto" in series:
        s = series["crypto"]
        ax.plot(s.index, s.values, color=CRYPTO_COLOR, linewidth=1.6,
                label="Crypto (alternative.me)", alpha=0.9)
    ax2 = None
    if "gold" in series:
        g = series["gold"]
        gmin, gmax = g.min(), g.max()
        gn = ((g - gmin) / (gmax - gmin) * 100 if gmax > gmin
              else g * 0 + 50)
        ax2 = ax.twinx()
        ax2.plot(g.index, gn.values, color=GOLD_COLOR, linewidth=1.6,
                 label="Gold futures (GC=F, $/oz)", alpha=0.95)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Gold price ($/oz)", fontsize=11,
                       color=GOLD_COLOR)
        ax2.set_yticks([0, 25, 50, 75, 100])
        ax2.set_yticklabels(
            [f"${gmin + (gmax - gmin) * t / 100:,.0f}"
             for t in (0, 25, 50, 75, 100)])
        ax2.tick_params(colors=GOLD_COLOR)
        ax2.spines["top"].set_visible(False)
        ax2.scatter([g.index[-1]], [gn.iloc[-1]], color=GOLD_COLOR,
                    s=40, zorder=5, edgecolors="white", linewidths=1)
        ax2.annotate(f"${g.iloc[-1]:,.0f}/oz",
                     xy=(g.index[-1], gn.iloc[-1]),
                     xytext=(8, -18), textcoords="offset points",
                     fontsize=10, fontweight="bold", color=GOLD_COLOR,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white",
                               ec=GOLD_COLOR, alpha=0.9))
    for name, color in (("cnn", CNN_COLOR), ("crypto", CRYPTO_COLOR)):
        if name in series:
            s = series[name]
            v = s.iloc[-1]
            ax.scatter([s.index[-1]], [v], color=color, s=40, zorder=5,
                       edgecolors="white", linewidths=1)
            ax.annotate(f"{v:.0f} · {zone(v)}", xy=(s.index[-1], v),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=10, fontweight="bold", color=color,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec=color, alpha=0.9))
    ax.set_ylim(0, 100)
    ax.set_ylabel("Index (0–100)", fontsize=11)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        h1, l1 = h1 + h2, l1 + l2
    ax.legend(h1, l1, frameon=True, loc="upper left", fontsize=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
