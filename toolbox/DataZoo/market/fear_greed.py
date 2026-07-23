"""Fear & Greed index data sources.

Exposes :class:`FearGreed`, a small wrapper around two sentiment sources:

- **Traditional markets** (S&P 500) via CNN's undocumented Fear & Greed data API.
- **Crypto** via the public `alternative.me <https://alternative.me/crypto/fear-and-greed-index/>`_ API.

Both history methods return series shaped for plotting (see :meth:`FearGreed.cnn_history`).
"""

from __future__ import annotations

import datetime

import requests

#: Timeout (seconds) for outbound HTTP requests.
_REQUEST_TIMEOUT = 10

#: CNN's data API rejects requests without a browser-like User-Agent.
_BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

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


class FearGreed:
    """Fetch Fear & Greed history for traditional and crypto markets.

    Each method performs a live fetch and returns series ready for plotting; on
    failure it returns an empty result rather than raising.
    """

    @staticmethod
    def _cnn_graphdata() -> dict:
        """Fetch and return the raw CNN Fear & Greed graph data JSON."""
        response = requests.get(_CNN_URL, headers=_BROWSER_HEADERS, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()

    def cnn_history(self) -> list[dict]:
        """Full CNN Fear & Greed history: the index plus all sub-indicators.

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

            Components without a historical ``data`` array are skipped. Returns an
            empty list on failure.
        """
        try:
            graph = self._cnn_graphdata()
        except Exception:
            return []
        history = []
        for key, label in CNN_SERIES_LABELS.items():
            block = graph.get(key)
            if not block or not block.get("data"):
                continue
            points = block["data"]
            history.append(
                {
                    "key": key,
                    "label": label,
                    "x": [datetime.datetime.fromtimestamp(pt["x"] / 1000, tz=datetime.timezone.utc) for pt in points],
                    "y": [pt["y"] for pt in points],
                    "score": block.get("score"),
                    "rating": block.get("rating"),
                }
            )
        return history

    def crypto_history(self, limit: int = 365) -> dict:
        """Historical crypto Fear & Greed index (0-100) from alternative.me.

        Args:
            limit: Number of most-recent daily values to fetch.

        Returns:
            A series dict (same shape as :meth:`cnn_history` entries) with ``x``
            ordered oldest-to-newest, or an empty dict on failure.
        """
        try:
            response = requests.get(_CRYPTO_URL, params={"limit": limit}, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            entries = sorted(response.json()["data"], key=lambda e: int(e["timestamp"]))
            return {
                "key": "crypto_fear_and_greed",
                "label": "Crypto Fear & Greed Index (0-100)",
                "x": [datetime.datetime.fromtimestamp(int(e["timestamp"]), tz=datetime.timezone.utc) for e in entries],
                "y": [int(e["value"]) for e in entries],
                "score": int(entries[-1]["value"]),
                "rating": entries[-1]["value_classification"],
            }
        except Exception:
            return {}
