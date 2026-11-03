"""Yahoo Finance price access — two paths in one canonical module.

:func:`download_close_prices` batches a yfinance history request, keeps Close,
and drops timezone info. Pass ``normalize=True`` to scale each series to its
first valid print. Session caching and universe filters stay with the caller.

:func:`yahoo_daily` is the single-ticker sibling on the raw v8 chart API
(adjclose events), with a complete-trading-day rule: the in-progress
session's partial bar is dropped, so a series always ends at the latest
complete trading day. :func:`fetch_json` is its JSON-over-HTTP helper.

:func:`with_retry` is the one retry wrapper for fetch legs built on
this layer: the dashboard's multi-leg fetchers (power law, the dollar,
US vs ex-US) wrap each leg in it, so a transient HTTP 429/5xx from
Yahoo's rate limiter is absorbed with backoff instead of failing a
whole tab on a single leg.

:func:`chart_payload` is the shared raw-response cache under both
price paths (and under futures_board's history bars): one chart
fetch per symbol per TTL serves every consumer in the process, so
a dashboard section's board and tabs render from a single pull of
each history instead of each fetching its own copy.
"""

from __future__ import annotations

import datetime
import json
import threading
import time
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

# The browser user agent the chart cache fetches with — the same
# string futures_board has always sent (its UA dict now builds on
# this constant). Yahoo's chart endpoint rate-limits by IP and
# fingerprint in bursts (the dashboard's 2026-11-01/02 tab
# outages); the plain API agent above drew 429s on a morning when
# this browser agent was served normally, so the shared chart
# fetches go out under it.
YAHOO_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36")


def fetch_json(url, timeout=30, user_agent=None):
    """GET ``url`` and parse the JSON body. Closes the response handle."""
    req = urllib.request.Request(
        url, headers={"User-Agent": user_agent or _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def with_retry(fn, leg, attempts=3):
    """Run one fetch leg, retrying transient HTTP errors (429 /
    5xx) with backoff, and name the leg in any final failure so
    a caller surfacing the error can say WHICH source failed.
    The canonical retry for this layer — moved here 2026-11-01
    from btc_powerlaw's private copy so every multi-leg fetcher
    (power law, the dollar, US vs ex-US) shares the one
    implementation: 3 attempts, delays 1.5s then 3.75s, and
    only HTTP 429/500/502/503/504 are retried; anything else
    fails at once, wrapped in a RuntimeError naming the leg."""
    import time as _time
    from urllib.error import HTTPError

    delay = 1.5
    for k in range(attempts):
        try:
            return fn()
        except HTTPError as e:
            if (e.code not in (429, 500, 502, 503, 504)
                    or k == attempts - 1):
                raise RuntimeError("{}: {}".format(leg, e)) from e
            _time.sleep(delay)
            delay *= 2.5
        except Exception as e:
            raise RuntimeError("{}: {}".format(leg, e)) from e


# The shared chart cache: ticker -> (window start fetched,
# monotonic fetch time, raw payload). One entry per symbol — the
# widest fresh window serves every narrower request, because
# consumers slice the raw bars to their own [p1, p2] exactly.
_CHART_TTL_SECONDS = 900  # matches the dashboard's page cache TTL
_chart_cache = {}
_chart_inflight = {}
_chart_lock = threading.Lock()


def chart_payload(ticker, p1=0, p2=None):
    """The raw v8 chart payload for ``ticker`` over [p1, p2]
    (interval=1d, adjclose events), shared process-wide.

    A fresh entry (at most _CHART_TTL_SECONDS old) whose window
    starts at or before ``p1`` is served as-is — its p2 was "now"
    when fetched, so the TTL bounds its staleness the same way
    the dashboard's own page cache does. Otherwise the payload
    is fetched once: concurrent callers for the same symbol
    wait for the in-flight fetch instead of stampeding Yahoo
    (single-flight), and a failed fetch is never cached — the
    error propagates, and one waiter takes over as the next
    fetcher. The response carries both the raw quote closes and
    the adjclose series, so daily-close readers and adjusted-
    close readers share the one pull.
    """
    if p2 is None:  # fetch through "now", as in yahoo_daily
        p2 = int(datetime.datetime.now(
            datetime.timezone.utc).timestamp())
    while True:
        with _chart_lock:
            entry = _chart_cache.get(ticker)
            if (entry is not None and entry[0] <= p1
                    and time.monotonic() - entry[1]
                    <= _CHART_TTL_SECONDS):
                return entry[2]
            event = _chart_inflight.get(ticker)
            if event is None:
                event = threading.Event()
                _chart_inflight[ticker] = event
                owner = True
            else:
                owner = False
        if not owner:
            event.wait()
            continue
        try:
            url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                   f"?period1={p1}&period2={p2}&interval=1d&events=adjclose")
            payload = fetch_json(url, 30, user_agent=YAHOO_UA)
            with _chart_lock:
                _chart_cache[ticker] = (
                    p1, time.monotonic(), payload)
            return payload
        finally:
            with _chart_lock:
                _chart_inflight.pop(ticker, None)
            event.set()


def _repaired_adjclose(r):
    """The payload result's adjclose list with Yahoo's unfilled
    bars repaired from the same payload (observed 2026-11-02,
    hours after the Monday close: the Nov 2 bar sat null on
    both chart hosts for every equity while meta already
    carried the close, and BTC-USD's whole Nov 2 bar sat null
    behind its filled Nov 3 bar — series read a session stale
    and the BTC correlations lost Monday entirely).

    Two fills, both scaled by the standing adjustment ratio
    (a valid bar's adjclose / raw close — Yahoo anchors
    adjclose so the ratio is ~1 at the recent end):

    - A TRAILING null bar fills from meta.regularMarketPrice.
      Skipped when a dividend or split event lands after the
      last valid bar (the ratio changes on exactly those
      days), when the meta price is missing, or when its New
      York date is not the null bar's session date.
    - An INTERIOR null bar fills from the next bar's open —
      only for CRYPTOCURRENCY payloads, whose bars are
      contiguous (the payload itself shows each bar's open
      equal to the prior close); an equity's next open gaps
      overnight and fills nothing."""
    ts = r.get("timestamp") or []
    adj = list(r["indicators"]["adjclose"][0]["adjclose"])
    if not ts or not adj or all(a is not None for a in adj):
        return adj
    closes = ((r["indicators"].get("quote") or [{}])[0]
              .get("close") or [])

    def _ratio(i):
        raw = closes[i] if i < len(closes) else None
        return adj[i] / raw if raw else None

    if adj[-1] is None:
        valid = [i for i, a in enumerate(adj) if a is not None]
        meta = r.get("meta") or {}
        price, mtime = (meta.get("regularMarketPrice"),
                        meta.get("regularMarketTime"))
        if valid and price and mtime:
            ny = ZoneInfo("America/New_York")

            def _ny_day(stamp):
                return datetime.datetime.fromtimestamp(
                    stamp, datetime.timezone.utc).astimezone(ny).date()

            last = valid[-1]
            events = r.get("events") or {}
            blocked = any(
                (ev.get("date") or 0) > ts[last]
                for kind in ("dividends", "splits")
                for ev in (events.get(kind) or {}).values())
            ratio = _ratio(last)
            if (not blocked and ratio is not None
                    and _ny_day(ts[-1]) == _ny_day(mtime)):
                adj[-1] = price * ratio
    if (r.get("meta") or {}).get("instrumentType") \
            == "CRYPTOCURRENCY":
        opens = ((r["indicators"].get("quote") or [{}])[0]
                 .get("open") or [])
        for i in range(len(adj) - 1):
            if adj[i] is not None or i + 1 >= len(opens) \
                    or not opens[i + 1]:
                continue
            prior = [j for j in range(i) if adj[j] is not None]
            ratio = _ratio(prior[-1]) if prior else None
            if ratio is not None:
                adj[i] = opens[i + 1] * ratio
    return adj


def yahoo_daily(ticker, p1=0, p2=None, include_partial=False):
    """Daily adjusted closes for one ticker from Yahoo's v8 chart API.

    Returns a tz-naive, midnight-normalized Series of adjclose values,
    NaNs dropped. A trailing bar Yahoo has not filled yet is
    repaired from the payload meta (see _repaired_adjclose).
    The in-progress session's partial bar is dropped:
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
    d = chart_payload(ticker, p1, p2)
    r = d["chart"]["result"][0]
    ts_all, closes_all = r["timestamp"], _repaired_adjclose(r)
    # A cache hit may hold a wider window than requested; slice
    # the raw bars to exactly the requested [p1, p2], as a direct
    # fetch of this window would have returned.
    pairs = [(t, c) for t, c in zip(ts_all, closes_all)
             if p1 <= t <= p2]
    s = pd.Series(
        [c for _t, c in pairs],
        index=pd.to_datetime([t for t, _c in pairs], unit="s",
                             utc=True).tz_convert(None).normalize(),
    ).dropna()
    if include_partial:
        return s
    now_ny = datetime.datetime.now(datetime.timezone.utc).astimezone(
        ZoneInfo("America/New_York"))
    today_ny = pd.Timestamp(now_ny.date())
    if now_ny.weekday() < 5 and now_ny.hour >= 16:
        return s[s.index <= today_ny]
    return s[s.index < today_ny]
