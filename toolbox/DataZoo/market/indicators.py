"""Portfolio risk indicators built on QuantStats, plus the pure
12-1 momentum math.

The functions in this module operate on plain pandas price or return
series and have no display, data-fetching, or application coupling.
QuantStats is an optional dependency: install ``toolbox[quantstats]``
to use the drawdown / greeks functions; the momentum functions are
pure pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import quantstats as qs

__all__ = [
    "DEFAULT_LOOKBACK_DAYS",
    "DEFAULT_SKIP_DAYS",
    "MIN_LOOKBACK_DAYS",
    "alpha_beta",
    "compute_momentum_universe",
    "drawdown_stats",
]

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


# ---------------------------------------------------------------------------
# 12-1 momentum (moved from the Investments app's
# src/indicators/momemtum.py at the user's direction 2026-11-03 —
# the filename typo dies in the move). Pure math on pandas: the
# defaults are the canonical 12-1 values (skip the most recent
# month, look back a year, require a month of span); app callers
# pass their own configured values explicitly at every call site.
# ---------------------------------------------------------------------------

DEFAULT_SKIP_DAYS = 21
DEFAULT_LOOKBACK_DAYS = 252
MIN_LOOKBACK_DAYS = 30


def _compute_raw_momentum(
    prices: pd.Series,
    *,
    skip_days: int,
    lookback_days: int,
    min_lookback_days: int = MIN_LOOKBACK_DAYS,
) -> float | None:
    """
    12-1 momentum over available history:
        mom_t = P(t - skip) / P(t - lookback) - 1
    Lookback is clamped to available data length (min span enforced).
    """
    if prices is None or prices.empty:
        return None

    prices = prices.dropna()
    if prices.empty or (prices <= 0).any():
        raise ValueError("All prices must be positive for momentum calculation")
    if skip_days < 0 or lookback_days <= skip_days:
        return None
    # Use whatever history is available: clamp the lookback window to the data,
    # as long as the resulting span still meets the minimum required length.
    effective_lookback = min(lookback_days, len(prices) - 1)
    if effective_lookback - skip_days < min_lookback_days:
        return None

    end_idx = -(skip_days + 1)
    start_idx = -(effective_lookback + 1)
    end_price = prices.iloc[end_idx]
    start_price = prices.iloc[start_idx]
    if pd.isna(end_price) or pd.isna(start_price) or start_price <= 0:
        return None

    return float(end_price / start_price - 1)


def compute_momentum_universe(
    prices: pd.DataFrame,
    *,
    skip_days: int = DEFAULT_SKIP_DAYS,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_lookback_days: int = MIN_LOOKBACK_DAYS,
) -> pd.Series:
    """12-1 momentum per symbol; symbols whose history cannot
    support the span are skipped."""
    raw_scores: dict[str, float] = {}
    for symbol in prices.columns:
        try:
            score = _compute_raw_momentum(
                prices[symbol],
                skip_days=skip_days,
                lookback_days=lookback_days,
                min_lookback_days=min_lookback_days,
            )
            if score is not None:
                raw_scores[symbol] = float(score)
        except (ValueError, IndexError):
            continue

    if not raw_scores:
        return pd.Series(dtype=float)
    return pd.Series(raw_scores, dtype=float)
