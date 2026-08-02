"""Stock index constituent lists from Wikipedia."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

#: Timeout (seconds) for outbound HTTP requests.
_REQUEST_TIMEOUT = 30

#: Wikipedia rejects requests without a browser-like User-Agent.
_BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

#: Index name -> Wikipedia source. ``column`` is the ticker field; ``min_rows``
#: picks the constituent table when a page has several similar ones.
_INDICES: dict[str, dict] = {
    "sp500": {
        "label": "S&P 500",
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "column": "Symbol",
        "min_rows": 400,
    },
    "sp100": {
        "label": "S&P 100",
        "url": "https://en.wikipedia.org/wiki/S%26P_100",
        "column": "Symbol",
        "min_rows": 80,
    },
    "sp400": {
        "label": "S&P MidCap 400",
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "column": "Symbol",
        "min_rows": 300,
    },
    "sp600": {
        "label": "S&P SmallCap 600",
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
        "column": "Symbol",
        "min_rows": 500,
    },
    "dow": {
        "label": "Dow Jones Industrial Average",
        "url": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
        "column": "Symbol",
        "min_rows": 25,
    },
    "nasdaq100": {
        "label": "NASDAQ-100",
        "url": "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
        "column": "Ticker",
        "min_rows": 80,
    },
    "russell1000": {
        "label": "Russell 1000",
        "url": "https://en.wikipedia.org/wiki/Russell_1000_Index",
        "column": "Symbol",
        "min_rows": 800,
    },
    "ftse100": {
        "label": "FTSE 100",
        "url": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "column": "Ticker",
        "min_rows": 80,
    },
    "dax": {
        "label": "DAX",
        "url": "https://en.wikipedia.org/wiki/DAX",
        "column": "Ticker",
        "min_rows": 30,
    },
    "cac40": {
        "label": "CAC 40",
        "url": "https://en.wikipedia.org/wiki/CAC_40",
        "column": "Ticker",
        "min_rows": 30,
    },
    "eurostoxx50": {
        "label": "EURO STOXX 50",
        "url": "https://en.wikipedia.org/wiki/EURO_STOXX_50",
        "column": "Ticker",
        "min_rows": 40,
    },
    "hangseng": {
        "label": "Hang Seng",
        "url": "https://en.wikipedia.org/wiki/Hang_Seng_Index",
        "column": "Ticker",
        "min_rows": 50,
    },
    "tsx60": {
        "label": "S&P/TSX 60",
        "url": "https://en.wikipedia.org/wiki/S%26P/TSX_60",
        "column": "Symbol",
        "min_rows": 50,
    },
    "asx50": {
        "label": "S&P/ASX 50",
        "url": "https://en.wikipedia.org/wiki/S%26P/ASX_50",
        "column": "Symbol",
        "min_rows": 40,
    },
    "nifty50": {
        "label": "Nifty 50",
        "url": "https://en.wikipedia.org/wiki/NIFTY_50",
        "column": "Symbol",
        "min_rows": 40,
    },
    "sensex": {
        "label": "BSE Sensex",
        "url": "https://en.wikipedia.org/wiki/BSE_SENSEX",
        "column": "Symbol",
        "min_rows": 25,
    },
    "ibex35": {
        "label": "IBEX 35",
        "url": "https://en.wikipedia.org/wiki/IBEX_35",
        "column": "Ticker",
        "min_rows": 30,
    },
    "smi": {
        "label": "Swiss Market Index",
        "url": "https://en.wikipedia.org/wiki/Swiss_Market_Index",
        "column": "Ticker",
        "min_rows": 15,
    },
    "omxs30": {
        "label": "OMX Stockholm 30",
        "url": "https://en.wikipedia.org/wiki/OMX_Stockholm_30",
        "column": "Ticker",
        "min_rows": 25,
    },
}


#: Prefer these Wikipedia columns for company / security names.
_NAME_COLUMNS = ("Security", "Company", "Name", "Company name")


class StockIndices:
    """Fetch stock-index constituents from Wikipedia.

    ``available`` lists supported index names. Use :meth:`get` to pull tickers,
    or ``get(name, as_frame=True)`` for tickers + company names in one fetch.

    Example::

        si = StockIndices()
        si.available                      # ('sp500', 'dow', ...)
        si.get("sp500")                   # ['MMM', 'AOS', ...]
        si.get("sp500", as_frame=True)    # DataFrame: ticker, name
    """

    available: tuple[str, ...] = tuple(_INDICES)

    def _table(self, name: str) -> pd.DataFrame | None:
        """Download and select the constituent table for ``name``."""
        key = name.strip().lower()
        meta = _INDICES.get(key)
        if meta is None:
            return None
        try:
            response = requests.get(
                meta["url"], headers=_BROWSER_HEADERS, timeout=_REQUEST_TIMEOUT
            )
            response.raise_for_status()
            tables = pd.read_html(StringIO(response.text))
            column = meta["column"]
            min_rows = meta["min_rows"]
            return next(
                t for t in tables if column in t.columns and len(t) >= min_rows
            )
        except Exception:
            return None

    def get(self, name: str, *, as_frame: bool = False) -> list[str] | pd.DataFrame:
        """Return constituent tickers, or a ``ticker``/``name`` frame.

        One Wikipedia fetch either way. Dots in symbols become hyphens for
        yfinance (e.g. ``BRK.B`` → ``BRK-B``). Returns ``[]`` / empty frame
        on unknown name or fetch failure.
        """
        key = name.strip().lower()
        meta = _INDICES.get(key)
        if meta is None:
            return pd.DataFrame(columns=["ticker", "name"]) if as_frame else []

        table = self._table(key)
        if table is None or table.empty:
            return pd.DataFrame(columns=["ticker", "name"]) if as_frame else []

        ticker_col = meta["column"]
        tickers = (
            table[ticker_col]
            .astype(str)
            .str.replace(".", "-", regex=False)
            .str.strip()
        )
        if not as_frame:
            return tickers.tolist()

        name_col = next((c for c in _NAME_COLUMNS if c in table.columns), None)
        names = (
            table[name_col].astype(str).str.strip()
            if name_col is not None
            else tickers.copy()
        )
        return pd.DataFrame({"ticker": tickers.to_numpy(), "name": names.to_numpy()})

    def label(self, name: str) -> str:
        """Human-readable label for an index name, or ``name`` if unknown."""
        meta = _INDICES.get(name.strip().lower())
        return meta["label"] if meta else name
