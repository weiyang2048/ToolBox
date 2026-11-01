"""Yahoo Finance price access — two paths in one canonical module.

:func:`download_close_prices` batches a yfinance history request, keeps Close,
and drops timezone info. Pass ``normalize=True`` to scale each series to its
first valid print. Session caching and universe filters stay with the caller.

:func:`yahoo_daily` is the single-ticker sibling on the raw v8 chart API
(adjclose events), with a complete-trading-day rule: the in-progress
session's partial bar is dropped, so a series always ends at the latest
complete trading day. :func:`fetch_json` is its JSON-over-HTTP helper.
"""

from __future__ import annotations

import datetime
import json
import urllib.request
from zoneinfo import ZoneInfo

import pandas as pd


def normalize_close_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Divide each column by its first valid price. Empty frames pass through."""
    if prices.empty:
        return prices
    def _first(col: pd.Series):
        start = col.first_valid_index()
        return 1 if start is None else col[start]

    bases = prices.apply(_first)
    return prices.div(bases, axis=1)


def download_close_prices(
    tickers: list[str],
    period: str = "5y",
    *,
    normalize: bool = False,
) -> pd.DataFrame:
    """Adjusted close prices for ``tickers`` over ``period``.

    Columns are ticker symbols, in first-seen order. A timezone-aware index is
    converted to naive. ``normalize`` scales each column by its first valid price.
    """
    symbols = list(dict.fromkeys(tickers))
    if not symbols:
        return pd.DataFrame()

    prices = _close_frame(_history(symbols, period))
    prices = _strip_timezone(prices)
    if normalize and not prices.empty:
        prices = normalize_close_prices(prices)
    return prices


def _history(tickers: list[str], period: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.Tickers(tickers).history(period=period, auto_adjust=True)


def _close_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Keep Close columns and name them by ticker (MultiIndex level 1)."""
    if history.empty:
        return pd.DataFrame()
    close_cols = [col for col in history.columns if "Close" in col]
    out = history.loc[:, close_cols].copy()
    if isinstance(history.columns, pd.MultiIndex):
        out.columns = [col[1] for col in close_cols]
    return out


def _strip_timezone(prices: pd.DataFrame) -> pd.DataFrame:
    index = prices.index
    if hasattr(index, "tz") and index.tz is not None:
        prices = prices.copy()
        prices.index = index.tz_convert(None)
    return prices


_USER_AGENT = "toolbox-datazoo"


def fetch_json(url, timeout=30, user_agent=None):
    """GET ``url`` and parse the JSON body. Closes the response handle."""
    req = urllib.request.Request(
        url, headers={"User-Agent": user_agent or _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def yahoo_daily(ticker, p1=0, p2=None, include_partial=False):
    """Daily adjusted closes for one ticker from Yahoo's v8 chart API.

    Returns a tz-naive, midnight-normalized Series of adjclose values,
    NaNs dropped. The in-progress session's partial bar is dropped:
    the series ends at the latest COMPLETE trading day
    (America/New_York); after the 16:00 ET close on a weekday, that
    day's bar is complete and is kept.

    ``include_partial=True`` skips that equity-session rule and
    keeps every bar Yahoo serves, including the in-progress one —
    the right choice for 24/7 instruments (crypto), whose current
    UTC-day bar is the standard "today" bar on every crypto chart.
    """
    if p2 is None:  # fetch through "now"; a fixed stamp would silently go stale
        p2 = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1={p1}&period2={p2}&interval=1d&events=adjclose")
    d = fetch_json(url, 30)
    r = d["chart"]["result"][0]
    ts, closes = r["timestamp"], r["indicators"]["adjclose"][0]["adjclose"]
    s = pd.Series(
        closes,
        index=pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize(),
    ).dropna()
    if include_partial:
        return s
    now_ny = datetime.datetime.now(datetime.timezone.utc).astimezone(
        ZoneInfo("America/New_York"))
    today_ny = pd.Timestamp(now_ny.date())
    if now_ny.weekday() < 5 and now_ny.hour >= 16:
        return s[s.index <= today_ny]
    return s[s.index < today_ny]
