"""Portfolio risk indicators built on QuantStats.

The functions in this module operate on plain pandas price or return
series and have no display, data-fetching, or application coupling.
QuantStats is an optional dependency: install ``toolbox[quantstats]``
to use them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import quantstats as qs

__all__ = ["alpha_beta", "drawdown_stats"]

_NAN = float("nan")


def drawdown_stats(prices: pd.Series | None) -> dict:
    """Drawdown metrics from QuantStats.

    ``max_drawdown`` expects prices; ``to_drawdown_series`` /
    ``ulcer_index`` expect returns.

    All numeric outputs are floats rounded to 2 decimals (display may
    round further). Returns keys:
      current — positive current depth (%)
      max — ``|qs.stats.max_drawdown|`` as positive %
      ulcer_index — ``qs.stats.ulcer_index`` as %
      pctile — percentile of current depth vs full history
      history — full-series negated % depths (no sparkline baseline)
    """
    empty = {
        "current": _NAN,
        "max": _NAN,
        "ulcer_index": _NAN,
        "pctile": _NAN,
        "history": [],
    }
    if prices is None:
        return empty
    price_s = prices.dropna()
    if len(price_s) < 2:
        return empty
    returns = price_s.pct_change().dropna()
    if returns.empty:
        return empty
    try:
        dd = qs.stats.to_drawdown_series(returns)
        max_dd = float(qs.stats.max_drawdown(price_s))
        ulcer = float(qs.stats.ulcer_index(returns))
    except (TypeError, ValueError, KeyError):
        return empty
    if dd.empty:
        return empty

    last = dd.iloc[-1]
    current = (
        float(np.round(-last * 100.0, 2)) if pd.notna(last) else _NAN
    )
    mx = float(np.round(-max_dd * 100.0, 2)) if np.isfinite(max_dd) else _NAN
    ulcer_pct = float(np.round(ulcer * 100.0, 2)) if np.isfinite(ulcer) else _NAN

    dd_pct = [round(float(v) * 100.0, 2) for v in dd.tolist() if pd.notna(v)]
    history = dd_pct

    if dd_pct and pd.notna(current):
        depths = np.asarray([-v for v in dd_pct], dtype=float)
        pctile = float(np.round(100.0 * float(np.mean(depths <= current)), 2))
    else:
        pctile = _NAN

    return {
        "current": current,
        "max": mx,
        "ulcer_index": ulcer_pct,
        "pctile": pctile,
        "history": history,
    }


def alpha_beta(
    strategy_returns: pd.Series | None,
    benchmark_returns: pd.Series | None,
    periods: float = 252,
) -> tuple[float, float]:
    """``(beta, alpha)`` from ``qs.stats.greeks`` on intersecting dates.

    Returns ``(nan, nan)`` when either input is missing or the two
    series share fewer than two observations.
    """
    if strategy_returns is None or benchmark_returns is None:
        return _NAN, _NAN
    s = strategy_returns.dropna()
    b = benchmark_returns.dropna()
    common = s.index.intersection(b.index)
    if len(common) < 2:
        return _NAN, _NAN
    try:
        greeks = qs.stats.greeks(
            s.loc[common],
            benchmark=b.loc[common],
            periods=periods,
            prepare_returns=False,
        )
        beta = float(greeks["beta"])
        alpha = float(greeks["alpha"])
        return (
            beta if np.isfinite(beta) else _NAN,
            alpha if np.isfinite(alpha) else _NAN,
        )
    except (TypeError, ValueError, KeyError, ZeroDivisionError):
        return _NAN, _NAN
