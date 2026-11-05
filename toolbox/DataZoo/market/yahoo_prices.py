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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


# Tickers per batch call when a download reports progress:
# the batch is split into chunks of this size so the caller
# sees real per-chunk counts (user request 2026-11-05 — a
# 100-ticker download otherwise sat at its opening label
# for the whole fetch). Without a progress callback the
# download stays the one batch call it has always been.
_PROGRESS_CHUNK = 20


def download_close_prices(
    tickers: list[str],
    period: str = "5y",
    *,
    normalize: bool = False,
    progress=None,
) -> pd.DataFrame:
    """Adjusted close prices for ``tickers`` over ``period``.

    Columns are ticker symbols, in first-seen order. A timezone-aware index is
    converted to naive. ``normalize`` scales each column by its first valid price.

    A ticker the batch download returns as a stub (missing or
    nearly empty — Yahoo throttling the batch) is refilled
    whole from the repaired chart path; see _refill_stubs.

    ``progress`` is the shared callback contract —
    ``progress(done, total, label)`` with counts in tickers:
    one call at 0 before the first fetch, one after each
    completed chunk, and a label-only beat before any stub
    refills. Passing it splits the batch into chunks of
    ``_PROGRESS_CHUNK``, downloaded in parallel (counts
    report in completion order); the assembled frame,
    refills and tail patch are exactly the no-progress
    path's.
    """
    symbols = list(dict.fromkeys(tickers))
    if not symbols:
        return pd.DataFrame()

    if progress is None:
        prices = _close_frame(_history(symbols, period))
    else:
        progress(0, len(symbols), "downloading prices")
        if len(symbols) <= _PROGRESS_CHUNK:
            prices = _close_frame(_history(symbols, period))
            progress(len(symbols), len(symbols), symbols[-1])
        else:
            chunks = [symbols[i:i + _PROGRESS_CHUNK]
                      for i in range(0, len(symbols), _PROGRESS_CHUNK)]
            frames = [None] * len(chunks)
            done = 0
            # Chunks download in parallel (speed pass
            # 2026-11-05: sequential chunks cost ~6 s each —
            # a 505-ticker pull spent ~2.5 min in this loop).
            # Counts still report as each chunk lands, and
            # the frames concat in chunk order, so the
            # assembled frame is the sequential one's; a
            # chunk that raises fails the download, exactly
            # as the sequential loop did.
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(_history, chunk, period): i
                           for i, chunk in enumerate(chunks)}
                for fut in as_completed(futures):
                    i = futures[fut]
                    frames[i] = _close_frame(fut.result())
                    done += len(chunks[i])
                    progress(done, len(symbols), chunks[i][-1])
            prices = pd.concat(frames, axis=1)
            prices = prices.reindex(
                columns=[s for s in symbols if s in prices.columns])
        stubs = [s for s in symbols
                 if s not in prices.columns
                 or int(prices[s].notna().sum()) < STUB_MIN_OBS]
        if stubs:
            progress(len(symbols), len(symbols),
                     "repairing throttled tickers")
    prices = _strip_timezone(prices)
    prices = _refill_stubs(prices, symbols, period)
    prices = _patch_tail(prices)
    if normalize and not prices.empty:
        prices = normalize_close_prices(prices)
    return prices


# A batch column shorter than this is a stub, not a history:
# the batch download was throttled or dropped the leg (on the
# dashboard's shared egress IP this cost the Strategies page
# its IXN column on 2026-11-02 — the default universe silently
# rendered, weighted and backtested on 10 of its 11 sectors).
STUB_MIN_OBS = 60


def _period_start(period: str) -> int:
    """Epoch seconds for the start of a yfinance period string
    ("5y", "6mo"); unrecognized strings fall back to 10y back —
    over-fetching is safe here because a refill is sliced to
    the batch frame's own window before it is used."""
    import re
    import time as _time

    now = int(_time.time())
    m = re.fullmatch(r"(\d+)(y|mo)", period or "")
    if not m:
        return now - int(10 * 365.25 * 86400)
    n, unit = int(m.group(1)), m.group(2)
    days = n * 365.25 if unit == "y" else n * 30.44
    return now - int(days * 86400)


def _refill_stubs(prices: pd.DataFrame, symbols, period: str
                  ) -> pd.DataFrame:
    """Refill stub columns from yahoo_daily (the repaired chart
    path) — user direction 2026-11-02 ("refill stub tickers in
    the price path").

    yfinance's batch history can return one ticker as a stub
    (missing, all-NaN, or a handful of rows) when Yahoo
    throttles the batch, while the same ticker's chart payload
    is served normally. A column with fewer than STUB_MIN_OBS
    valid points is refetched whole through the chart path and
    replaces the stub only when it brings MORE valid points —
    a genuinely young ticker (whose refill is the same short
    history) is untouched. Refilled values are sliced to the
    batch frame's window and the frame's index grows to their
    union, so a fully missing ticker returns with its dates.
    Best-effort per ticker: a failed refill keeps the stub."""
    stubs = [s for s in symbols
             if s not in prices.columns
             or int(prices[s].notna().sum()) < STUB_MIN_OBS]
    if not stubs:
        return prices
    p1 = _period_start(period)
    lo = prices.index.min() if len(prices) else None
    hi = prices.index.max() if len(prices) else None
    out = prices.copy()
    for sym in stubs:
        try:
            s = yahoo_daily(sym, p1=p1, include_partial=True)
        except Exception:  # noqa: BLE001 - best-effort refill
            continue
        if not len(s):
            continue
        if lo is not None:
            s = s[(s.index >= lo) & (s.index <= hi)]
        have = (int(out[sym].notna().sum())
                if sym in out.columns else 0)
        if int(s.notna().sum()) <= have:
            continue
        out = out.reindex(out.index.union(s.index)).sort_index()
        out[sym] = s.reindex(out.index)
    return out


def _patch_tail(prices: pd.DataFrame) -> pd.DataFrame:
    """Fill recent NaN cells from yahoo_daily (the repaired
    chart path). yfinance assembles the same lagging daily
    bars, so a completed session can sit NaN at a column's
    tail for hours after the close while the chart path's
    repairs already carry it. Best-effort and bounded: only
    NaN cells in the last three rows, only where yahoo_daily
    has a value for that date; a failed leg keeps its NaNs."""
    if prices.empty:
        return prices
    import datetime as _dt

    p1 = int(_dt.datetime.now(_dt.timezone.utc).timestamp()) \
        - 45 * 86400
    tail = prices.iloc[-3:]
    for col in prices.columns:
        if not tail[col].isna().any():
            continue
        try:
            s = yahoo_daily(str(col), p1=p1, include_partial=True)
        except Exception:  # noqa: BLE001 - best-effort patch
            continue
        for d in tail.index[tail[col].isna()]:
            v = s.get(d) if len(s) else None
            if v is not None and pd.notna(v):
                prices.loc[d, col] = float(v)
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
                raise RuntimeError(f"{leg}: {e}") from e
            _time.sleep(delay)
            delay *= 2.5
        except Exception as e:
            raise RuntimeError(f"{leg}: {e}") from e


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


# The intraday fallback cache: ticker -> (monotonic fetch
# time, {Eastern date: raw close at that session's end}).
_INTRADAY_CACHE = {}
_INTRADAY_LOCK = threading.Lock()


def _intraday_session_closes(ticker, instrument_type=None):
    """Each recent session's closing RAW price, reconstructed
    from Yahoo's 60-minute bars: the last intraday close at or
    before the session's end (16:00 Eastern; 17:00 for futures,
    whose Globex day ends then). The daily bars lag their own
    fill for hours after a close (see _repaired_adjclose); the
    intraday bars never do, so a still-null daily bar can be
    filled from its own session's tape. Cached per process at
    the chart TTL; any failure returns {} — the caller keeps
    the unfilled series rather than failing."""
    now_m = time.monotonic()
    with _INTRADAY_LOCK:
        entry = _INTRADAY_CACHE.get(ticker)
        if entry is not None \
                and now_m - entry[0] <= _CHART_TTL_SECONDS:
            return entry[1]
    out = {}
    try:
        p2 = int(datetime.datetime.now(
            datetime.timezone.utc).timestamp())
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
               f"{ticker}?period1={p2 - 12 * 86400}&period2={p2}"
               f"&interval=60m")
        d = with_retry(lambda: fetch_json(url, 30,
                                          user_agent=YAHOO_UA),
                       "yahoo intraday " + ticker)
        r = d["chart"]["result"][0]
        closes = r["indicators"]["quote"][0]["close"]
        cutoff = 17 if instrument_type == "FUTURE" else 16
        ny = ZoneInfo("America/New_York")
        for t, c in zip(r["timestamp"], closes):
            if c is None:
                continue
            et = datetime.datetime.fromtimestamp(
                t, datetime.timezone.utc).astimezone(ny)
            if et.hour < cutoff:  # the bar ends by the close
                out[et.date()] = float(c)  # later bars overwrite
    except Exception:  # noqa: BLE001 - a fallback must not fail
        out = {}
    with _INTRADAY_LOCK:
        _INTRADAY_CACHE[ticker] = (now_m, out)
    return out


def yahoo_daily(ticker, p1=0, p2=None, include_partial=False):
    """Daily adjusted closes for one ticker from Yahoo's v8 chart API.

    Returns a tz-naive, midnight-normalized Series of adjclose values,
    NaNs dropped. Bars Yahoo has not filled yet are repaired
    from the payload meta (see _repaired_adjclose) and, for
    any recent bar still null, from that session's intraday
    tape (see _intraday_session_closes) — one ladder, so every
    consumer of this layer stops a session early never.
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
    # Any recent bar still null (the index/futures shape: the
    # completed session's bar sits unfilled behind the live
    # session's) fills from that session's intraday tape.
    if any(c is None for _t, c in pairs[-5:]) \
            and (r.get("meta") or {}).get("instrumentType") \
            != "CRYPTOCURRENCY":
        fills = _intraday_session_closes(
            ticker, (r.get("meta") or {}).get("instrumentType"))
        if fills:
            ny = ZoneInfo("America/New_York")
            valid = [(t, c) for t, c in pairs if c is not None]
            ratio = 1.0
            if valid:
                raw = ((r["indicators"].get("quote") or [{}])[0]
                       .get("close") or [])
                t_last, c_last = valid[-1]
                i_last = ts_all.index(t_last)
                if i_last < len(raw) and raw[i_last]:
                    ratio = c_last / raw[i_last]

            def _filled(t, c):
                if c is not None:
                    return c
                day = datetime.datetime.fromtimestamp(
                    t, datetime.timezone.utc).astimezone(ny).date()
                v = fills.get(day)
                return v * ratio if v is not None else c

            pairs = [(t, _filled(t, c)) for t, c in pairs]
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
