"""Yahoo Finance adjusted close prices.

:func:`download_close_prices` batches a yfinance history request, keeps Close,
and drops timezone info. Pass ``normalize=True`` to scale each series to its
first valid print. Session caching and universe filters stay with the caller.
"""

from __future__ import annotations

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
