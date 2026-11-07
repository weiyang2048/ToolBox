"""Market sentiment and financial data sources.

The re-exports load lazily (PEP 562, user direction
2026-11-06): importing this package stays cheap and
holds the package import lock only momentarily.
Eager submodule imports held it long enough for
concurrent cold-start sessions (Streamlit Cloud) to
trip the import deadlock detector (_DeadlockError
on toolbox.DataZoo.market).
"""
import importlib

_EXPORTS = {
    "CNN_SERIES_LABELS": "fear_greed",
    "FearGreed": "fear_greed",
    "board_scales": "futures_board",
    "fetch_board": "futures_board",
    "shade_zero": "futures_board",
    "StockIndices": "stock_indices",
    "download_close_prices": "yahoo_prices",
    "fetch_json": "yahoo_prices",
    "normalize_close_prices": "yahoo_prices",
    "yahoo_daily": "yahoo_prices",
    "ZillowZipData": "zillow_data",
}

__all__ = [
    "FearGreed",
    "CNN_SERIES_LABELS",
    "ZillowZipData",
    "StockIndices",
    "download_close_prices",
    "normalize_close_prices",
    "fetch_json",
    "yahoo_daily",
    "fetch_board",
    "board_scales",
    "shade_zero",
]


def __getattr__(name):
    if name in _EXPORTS:
        module = importlib.import_module(
            f".{_EXPORTS[name]}", __name__)
        return getattr(module, name)
    try:
        return importlib.import_module(
            f".{name}", __name__)
    except ModuleNotFoundError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None


def __dir__():
    return sorted(set(globals()) | set(__all__))
