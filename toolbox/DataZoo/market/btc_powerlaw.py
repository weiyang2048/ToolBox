"""Bitcoin power law + 63-day BTC correlations, for the dashboard.

Moved here from the morning brief's Flaco script
(btc-powerlaw-correlations.py) at the user's direction 2026-11-01:
the chart now lives on the Markets page, in the Commodities section
under the board. The port is faithful — same data, same fits, same
figure; the brief-only EMA strategy sidecar was dropped (nothing
has consumed it since the brief's EMA box was removed in 2026-10).

Fit: power law ln(P) = lnA + b*ln(days since 2009-01-03) on the
extended blockchain.info history (Mt. Gox era onward, 2010-07-17+).
Price line: Yahoo BTC-USD daily closes (2014-09-17+) plus
blockchain.info points for 2010-2014. Correlations: 63-day rolling
BTC×gold (GLD) and BTC×SPX (SPY) of daily log returns from
2014-09-17, extended back to 2010 on the sparse blockchain.info
series (63-calendar-day windows, plotted thinner).
BTC market cap comes from blockchain.info; gold market cap =
Yahoo GC=F ($/ozt) × WGC above-ground stock (end-2024 216,265 t,
walked with ~3,738 t/yr mine production). The BTC÷gold ratio gets
its own power-law fit, whose 100% crossing is the model-implied
"flippening" date.
"""
import datetime
import io
import math
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolbox.DataZoo.market.yahoo_prices import fetch_json, yahoo_daily

UA = "toolbox-datazoo"
CT = ZoneInfo("America/Chicago")
GENESIS = datetime.date(2009, 1, 3)
OZT_PER_TONNE = 32150.7
# WGC end-2024 anchor: 216,265 t above-ground gold.
_ANCHOR_DAYS = (pd.Timestamp("2024-12-31") - pd.Timestamp(GENESIS)).days


def _blockchain_chart(endpoint, start, col):
    """Fetch a blockchain.info time-series chart into a DataFrame."""
    url = (f"https://api.blockchain.info/charts/{endpoint}"
           "?timespan=all&format=json")
    d = fetch_json(url, 60, user_agent=UA)
    cutoff = pd.Timestamp(start)
    ys = pd.Series([v["y"] for v in d["values"]], dtype=float)
    ts = pd.to_datetime([v["x"] for v in d["values"]], unit="s",
                        utc=True).tz_convert(None).normalize()
    mask = (ys > 0) & (ts >= cutoff)
    return pd.DataFrame(
        {"t": ts[mask], col: ys[mask].to_numpy()}).set_index("t")


def blockchain_btc():
    """Full BTC price history back to the Mt. Gox era (~4-day spacing)."""
    return _blockchain_chart("market-price", "2010-07-17", "p")


def blockchain_marketcap():
    """BTC market cap history (sparse, ~4-day spacing)."""
    return _blockchain_chart("market-cap", "2013-01-01", "mc")


def days_since_genesis(idx):
    """Days since 2009-01-03 for each timestamp in idx (vectorized)."""
    return (idx.normalize() - pd.Timestamp(GENESIS)).days.to_numpy(float)


def fit_power_law(days_arr, values):
    """OLS fit of ln(values) = lnA + b*ln(days); (slope b, lnA, sigma)."""
    lnV = np.log(values)
    ln_d = np.log(days_arr)
    b, lnA = np.polyfit(ln_d, lnV, 1)
    sigma = float((lnV - (lnA + b * ln_d)).std())
    return b, lnA, sigma


def trend_envelope(lnA, b, sigma, days_arr):
    """Power-law trend with +/-1 sigma envelope over days_arr."""
    ln_d = np.log(days_arr)
    trend = np.exp(lnA + b * ln_d)
    lo = np.exp(lnA - sigma + b * ln_d)
    hi = np.exp(lnA + sigma + b * ln_d)
    return trend, lo, hi


def roll_corr(a, b_, win=63, minp=8):
    """63-calendar-day rolling Pearson correlation (right-closed)."""
    return a.rolling(f"{win}D", min_periods=minp).corr(b_).dropna()


def fetch_powerlaw():
    """Fetch all six sources and compute the chart payload + stats.

    Returns a dict consumed by chart_png() and by the page's stats
    chips. Raises on any fetch failure — the caller isolates it.
    """
    p2 = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(yahoo_daily, t, p2=p2)
                for t in ("BTC-USD", "GLD", "SPY", "GC=F")]
        jobs += [pool.submit(blockchain_btc),
                 pool.submit(blockchain_marketcap)]
        btc, gld, spy, gold_oz, bdf, bmc = [f.result() for f in jobs]

    # --- extended fit on 2010+ history ---
    t = days_since_genesis(bdf.index)
    b, lnA, sigma = fit_power_law(t, bdf["p"].values)
    n_fit = len(bdf)
    end = datetime.date(2030, 12, 31)
    days = pd.date_range("2010-01-01", end, freq="D")
    trend, lo, hi = trend_envelope(
        lnA, b, sigma, days_since_genesis(days))
    p2029 = float(np.exp(
        lnA + b * math.log((datetime.date(2029, 12, 31) - GENESIS).days)))
    early = bdf[bdf.index < pd.Timestamp("2014-09-17")]

    # --- correlations ---
    rets = pd.DataFrame({"BTC": np.log(btc).diff(),
                         "GOLD": np.log(gld).diff(),
                         "SPX": np.log(spy).diff()}).dropna()
    c_gold = rets["BTC"].rolling(63).corr(rets["GOLD"]).dropna()
    c_spx = rets["BTC"].rolling(63).corr(rets["SPX"]).dropna()
    bdf = bdf.copy()
    bdf["lr"] = np.log(bdf["p"]).diff()
    slr = pd.DataFrame({"BTC": bdf["lr"]})
    slr["GOLD"] = np.log(gld).diff().reindex(
        slr.index, method="ffill").values
    slr["SPX"] = np.log(spy).diff().reindex(
        slr.index, method="ffill").values
    slr = slr.dropna()
    pre = slr[slr.index < pd.Timestamp("2014-09-17")]
    eg = roll_corr(pre["BTC"], pre["GOLD"])
    es = roll_corr(pre["BTC"], pre["SPX"])

    # --- BTC-to-gold market-cap ratio ---
    g_on_b = gold_oz.reindex(bmc.index, method="ffill")
    gold_days = days_since_genesis(bmc.index)
    gold_mc = g_on_b * (216265.0 + (gold_days - _ANCHOR_DAYS)
                        / 365.25 * 3738.0) * OZT_PER_TONNE
    ratio = (bmc["mc"] / gold_mc).replace(
        [np.inf, -np.inf], np.nan).dropna()
    ratio = ratio[ratio > 0]
    ratio_now = float(ratio.iloc[-1])
    rt = days_since_genesis(ratio.index)
    b_r, lnA_r, sigma_r = fit_power_law(rt, ratio.values)
    r_days = pd.date_range("2013-01-01", end, freq="D")
    r_trend, r_lo, r_hi = trend_envelope(
        lnA_r, b_r, sigma_r, days_since_genesis(r_days))
    flip_date = None
    if b_r > 0:
        t_flip = math.exp(-lnA_r / b_r)
        flip_date = GENESIS + datetime.timedelta(days=t_flip)

    last = c_gold.index[-1]
    floor = math.floor(
        min(c_gold.min(), c_spx.min(), eg.min(), es.min()) * 20) / 20
    now_ct = datetime.datetime.now(CT)
    return {
        "asof": now_ct.strftime("%a %b %-d, %-I:%M %p CT"),
        "btc": btc, "early": early, "days": days, "trend": trend,
        "lo": lo, "hi": hi, "ratio": ratio, "r_days": r_days,
        "r_trend": r_trend, "r_lo": r_lo, "r_hi": r_hi,
        "c_gold": c_gold, "c_spx": c_spx, "eg": eg, "es": es,
        "floor": floor, "last": last,
        "stats": {
            "slope": float(b), "sigma": float(sigma), "n_fit": n_fit,
            "end2029": p2029,
            "btc_last": float(btc.iloc[-1]),
            "btc_date": btc.index[-1].date().isoformat(),
            "ratio_pct": ratio_now * 100,
            "ratio_date": ratio.index[-1].date().isoformat(),
            "ratio_slope": float(b_r),
            "flippening": (flip_date.isoformat()
                           if flip_date else None),
            "corr_gold": float(c_gold.iloc[-1]),
            "corr_spx": float(c_spx.iloc[-1]),
            "data_end": last.date().isoformat(),
        },
    }


def chart_png(data):
    """Render the three-panel power-law figure to PNG bytes.

    Layout (user request 2026-10-23, from the brief era): top-left
    the log-scale price with its power law and ±1σ band, top-right
    the BTC÷gold market-cap ratio with its own fit and the
    flippening line, bottom the two 63-day correlations full width.
    """
    btc, early = data["btc"], data["early"]
    days, trend, lo, hi = (data["days"], data["trend"],
                           data["lo"], data["hi"])
    ratio, r_days = data["ratio"], data["r_days"]
    r_trend, r_lo, r_hi = data["r_trend"], data["r_lo"], data["r_hi"]
    c_gold, c_spx, eg, es = (data["c_gold"], data["c_spx"],
                             data["eg"], data["es"])
    last, floor, st = data["last"], data["floor"], data["stats"]
    b, b_r = st["slope"], st["ratio_slope"]

    fig = plt.figure(figsize=(11, 7.4))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.6, 1],
                           hspace=0.08, wspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1r = fig.add_subplot(gs[0, 1], sharex=ax1)
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)

    # --- Top-left: BTC price power law only ---
    ax1.fill_between(days, lo, hi, color="#e4dcc9", alpha=0.35, zorder=1)
    ax1.semilogy(days, trend, color="#1f3a5f", lw=1.4, alpha=0.85,
                 zorder=3, label=f"power law (slope {b:.2f})")
    if len(early):
        ax1.semilogy(early.index, early["p"], ".", color="#8a7f6a",
                     ms=3, alpha=0.6,
                     label="BTC price (2010-2014, sparse)", zorder=2)
    ax1.semilogy(btc.index, btc, color="#26221c", lw=2.2, alpha=1.0,
                 zorder=5, label="BTC price")
    _lp_x, _lp_y = btc.index[-1], float(btc.iloc[-1])
    ax1.plot([_lp_x], [_lp_y], marker="o", ms=6, color="#b3543f",
             markeredgecolor="white", markeredgewidth=1.0, zorder=7)
    ax1.annotate(f"${_lp_y:,.0f}\n{_lp_x.date().isoformat()}",
                 xy=(_lp_x, _lp_y), xytext=(-8, -30),
                 textcoords="offset points", ha="right", va="top",
                 fontsize=8.5, color="#b3543f", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           alpha=0.9, edgecolor="#b3543f", lw=0.7),
                 arrowprops=dict(arrowstyle="-", color="#b3543f", lw=0.9))
    ax1.axhline(300000, color="#2e7d32", ls=":", lw=1.3)
    ax1.text(datetime.datetime(2030, 12, 31), 345000, "$300,000 ",
             color="#2e7d32", fontsize=9, ha="right")
    ax1.axvline(last, color="#b3543f", ls="--", lw=1.2)
    ax1.text(last + pd.Timedelta(days=8), 40000, "today",
             color="#b3543f", fontsize=9, va="top")
    ax1.set_ylabel("BTC-USD (log scale)", color="#1f3a5f")
    ax1.tick_params(axis="y", colors="#1f3a5f")
    ax1.set_title("BTC price — power law", fontsize=10)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(True, which="major", alpha=0.25)
    ax1.text(0.03, 0.96,
             f"fit 2010\u2013{last.date().year}, n={st['n_fit']}\n"
             f"end-2029 \u2248 ${st['end2029']:,.0f}",
             transform=ax1.transAxes, fontsize=8, va="top", ha="left",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_xlim(datetime.datetime(2010, 1, 1),
                 datetime.datetime(2030, 12, 31))

    # --- Top-right: BTC/gold market-cap ratio power law only ---
    ax1r.fill_between(r_days, r_lo * 100, r_hi * 100, color="#e8d9a8",
                      alpha=0.45, zorder=1)
    ax1r.axhline(100, color="#2e7d32", ls=":", lw=1.3, zorder=2)
    ax1r.text(datetime.datetime(2010, 6, 1), 118,
              "flippening \u2014 100%\nmodel \u2248 "
              f"{st['flippening'] or 'n/a'}",
              color="#2e7d32", fontsize=8.5, va="bottom")
    ax1r.set_ylim(1e-3, 400)
    ax1r.semilogy(r_days, r_trend * 100, color="#8B6914", lw=1.8,
                  ls="--", zorder=3,
                  label=f"ratio power law (slope {b_r:.2f})")
    ax1r.semilogy(ratio.index, ratio * 100, color="#B8860B", lw=1.4,
                  alpha=0.9, zorder=4,
                  label="BTC/gold market cap "
                        f"(now {st['ratio_pct']:.2f}%)")
    _lr_x, _lr_y = ratio.index[-1], float(ratio.iloc[-1] * 100)
    ax1r.plot([_lr_x], [_lr_y], marker="o", ms=6, color="#b3543f",
              markeredgecolor="white", markeredgewidth=1.0, zorder=6)
    ax1r.annotate(f"{_lr_y:.2f}%\n{_lr_x.date().isoformat()}",
                  xy=(_lr_x, _lr_y), xytext=(-8, 14),
                  textcoords="offset points", ha="right", va="bottom",
                  fontsize=8.5, color="#b3543f", fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            alpha=0.9, edgecolor="#b3543f", lw=0.7),
                  arrowprops=dict(arrowstyle="-", color="#b3543f", lw=0.9))
    ax1r.axvline(last, color="#b3543f", ls="--", lw=1.2)
    ax1r.text(last + pd.Timedelta(days=8), 20, "today",
              color="#b3543f", fontsize=9, va="top")
    ax1r.set_ylabel("BTC \u00f7 gold market cap (%)", color="#B8860B")
    ax1r.tick_params(axis="y", colors="#B8860B")
    ax1r.set_title("BTC / gold market cap — power law", fontsize=10)
    ax1r.legend(loc="lower left", fontsize=8)
    ax1r.grid(True, which="major", alpha=0.25)
    plt.setp(ax1r.get_xticklabels(), visible=False)

    fig.suptitle("Bitcoin power law and 63-day BTC\u00d7gold / "
                 "BTC\u00d7SPX correlations", fontsize=11, y=0.98)

    # --- Bottom row: correlations, spanning both columns ---
    ax2.plot(eg.index, eg, color="#B8860B", lw=1.0, alpha=0.65)
    ax2.plot(c_gold.index, c_gold, color="#B8860B", lw=1.5,
             label=f"BTC \u00d7 gold (latest {st['corr_gold']:+.2f})")
    ax2.plot(es.index, es, color="#4682B4", lw=1.0, alpha=0.65)
    ax2.plot(c_spx.index, c_spx, color="#4682B4", lw=1.5,
             label=f"BTC \u00d7 SPX (latest {st['corr_spx']:+.2f})")
    ax2.set_ylim(floor, 1.2)
    ax2.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax2.set_ylabel("63-day rolling correlation")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.text(0.02, 0.06, "pre-2014: sparse data, thinner lines",
             transform=ax2.transAxes, fontsize=8, va="bottom",
             color="#777777")
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(datetime.datetime(2010, 1, 1),
                 datetime.datetime(2030, 12, 31))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()
