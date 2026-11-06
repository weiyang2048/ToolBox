"""Fear & Greed index data sources.

Exposes :class:`FearGreed`, a small wrapper around two sentiment sources:

- **Traditional markets** (S&P 500) via CNN's undocumented Fear & Greed data API.
- **Crypto** via the public `alternative.me <https://alternative.me/crypto/fear-and-greed-index/>`_ API.

Both history methods return series shaped for plotting (see :meth:`FearGreed.cnn_history`).

The module also carries the package's one canonical raw fetch layer for
these two sources — :func:`fetch_cnn_graph` and :func:`fetch_crypto_entries` —
on which both :class:`FearGreed` and the dashboard sentiment block
(``toolbox.DataZoo.market.sentiment``) build, so a URL or header fix ever
lands in exactly one place.
"""

from __future__ import annotations

import datetime
from typing import Literal, overload

import pandas as pd
import requests

#: Timeout (seconds) for outbound HTTP requests.
_REQUEST_TIMEOUT = 10

#: CNN's data API rejects requests without a browser-like User-Agent and
#: the CNN page context; this full header set is the canonical one.
CNN_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.cnn.com/markets/fear-and-greed",
    "Origin": "https://www.cnn.com",
}

#: CNN Fear & Greed graph data endpoint (headline index + all components).
_CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

#: alternative.me crypto Fear & Greed endpoint (supports a ``limit`` for history).
_CRYPTO_URL = "https://api.alternative.me/fng/"

#: CNN component keys mapped to human-readable labels, in display order. The
#: first entry is the headline 0-100 index; the rest are its raw sub-indicators.
CNN_SERIES_LABELS = {
    "fear_and_greed_historical": "Fear & Greed Index (0-100)",
    "market_momentum_sp500": "Market Momentum - S&P 500",
    "market_momentum_sp125": "Market Momentum - S&P 500 (125-day MA)",
    "stock_price_strength": "Stock Price Strength (net new 52w highs)",
    "stock_price_breadth": "Stock Price Breadth (McClellan volume)",
    "put_call_options": "Put / Call Options ratio",
    "market_volatility_vix": "Market Volatility - VIX",
    "market_volatility_vix_50": "Market Volatility - VIX (50-day MA)",
    "junk_bond_demand": "Junk Bond Demand (yield spread)",
    "safe_haven_demand": "Safe Haven Demand (stocks - bonds 20d)",
}


def fetch_cnn_graph(timeout: int = _REQUEST_TIMEOUT) -> dict:
    """Raw CNN Fear & Greed graph payload (headline index + components).

    The package's one canonical CNN fetch. Raises on transport or HTTP
    failure; callers apply their own failure policy.
    """
    response = requests.get(_CNN_URL, headers=CNN_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_crypto_entries(limit: int = 365, timeout: int = _REQUEST_TIMEOUT) -> list:
    """Raw alternative.me Fear & Greed entries, oldest first.

    The package's one canonical crypto fetch. Raises on transport or
    HTTP failure; callers apply their own failure policy.
    """
    response = requests.get(_CRYPTO_URL, params={"limit": limit}, timeout=timeout)
    response.raise_for_status()
    return sorted(response.json()["data"], key=lambda e: int(e["timestamp"]))


class FearGreed:
    """Fetch Fear & Greed history for traditional and crypto markets.

    Each method performs a live fetch and returns series ready for plotting; on
    failure it returns an empty result rather than raising.
    """

    @overload
    def cnn_history(self, return_df: Literal[False] = False) -> list[dict]: ...

    @overload
    def cnn_history(self, return_df: Literal[True]) -> pd.DataFrame: ...

    def cnn_history(self, return_df: bool = False) -> list[dict] | pd.DataFrame:
        """Full CNN Fear & Greed history: the index plus all sub-indicators.

        At most one observation is kept per UTC calendar day (the last if CNN
        emits an intraday update).

        Args:
            return_df: If ``True``, return a :class:`~pandas.DataFrame` indexed by
                ``YYYY-MM-DD`` UTC day strings with one column per series key. If
                ``False`` (default), return the list-of-dicts shape described below.

        Returns:
            A list of series (ordered per :data:`CNN_SERIES_LABELS`), each a dict::

                {
                    "key": str,                     # CNN component key
                    "label": str,                   # human-readable label
                    "x": list[datetime.datetime],   # daily timestamps (UTC)
                    "y": list[float],               # raw indicator values
                    "score": float,                 # current 0-100 sub-score
                    "rating": str,                  # current classification
                }

            Or, when ``return_df`` is ``True``, a DataFrame indexed by UTC calendar
            day strings (``YYYY-MM-DD``) with one column per series key. Components
            without a historical ``data`` array are skipped. Returns an empty list
            / empty DataFrame on failure.
        """
        try:
            graph = fetch_cnn_graph()
        except Exception:  # noqa: BLE001 - documented empty-on-failure contract
            return pd.DataFrame() if return_df else []
        history = []
        for key, label in CNN_SERIES_LABELS.items():
            block = graph.get(key)
            if not block or not block.get("data"):
                continue
            # Keep the last observation per UTC calendar day (dict overwrites,
            # points arrive oldest-first).
            by_day = {}
            for pt in block["data"]:
                day = datetime.datetime.fromtimestamp(pt["x"] / 1000, tz=datetime.timezone.utc).date()
                by_day[day] = pt["y"]
            history.append(
                {
                    "key": key,
                    "label": label,
                    "x": [datetime.datetime.combine(d, datetime.time(), datetime.timezone.utc) for d in by_day],
                    "y": list(by_day.values()),
                    "score": block.get("score"),
                    "rating": block.get("rating"),
                }
            )
        if not return_df:
            return history
        if not history:
            return pd.DataFrame()
        series = {
            s["key"]: pd.Series(s["y"], index=pd.DatetimeIndex(s["x"], name="time"))
            for s in history
        }
        df = pd.DataFrame(series).sort_index()
        df.index = pd.Index(df.index.strftime("%Y-%m-%d"), name="time")
        return df

    def crypto_history(self, limit: int = 365) -> dict:
        """Historical crypto Fear & Greed index (0-100) from alternative.me.

        Args:
            limit: Number of most-recent daily values to fetch.

        Returns:
            A series dict (same shape as :meth:`cnn_history` entries) with ``x``
            ordered oldest-to-newest, or an empty dict on failure.
        """
        try:
            entries = fetch_crypto_entries(limit=limit)
            return {
                "key": "crypto_fear_and_greed",
                "label": "Crypto Fear & Greed Index (0-100)",
                "x": [datetime.datetime.fromtimestamp(int(e["timestamp"]), tz=datetime.timezone.utc) for e in entries],
                "y": [int(e["value"]) for e in entries],
                "score": int(entries[-1]["value"]),
                "rating": entries[-1]["value_classification"],
            }
        except Exception:  # noqa: BLE001 - documented empty-on-failure contract
            return {}

    def fear_greed_history(self, limit: int = 365) -> pd.DataFrame:
        """Combine CNN and crypto Fear & Greed histories by UTC calendar day.

        Args:
            limit: Number of recent crypto daily values to fetch.

        Returns:
            A DataFrame indexed by ``YYYY-MM-DD`` strings containing all CNN
            series and a ``crypto_fear_and_greed`` column.
        """
        cnn = self.cnn_history(return_df=True)
        crypto = self.crypto_history(limit=limit)
        if not crypto:
            return cnn

        crypto_by_day = {
            timestamp.strftime("%Y-%m-%d"): value
            for timestamp, value in zip(crypto["x"], crypto["y"])
        }
        crypto_df = pd.DataFrame.from_dict(
            crypto_by_day,
            orient="index",
            columns=[crypto["key"]],
        )
        crypto_df.index.name = "time"
        return cnn.join(crypto_df, how="outer").sort_index().dropna()
