"""Causal feature variant for the registered AlphaEx/MaxDD search.

This module is deliberately separate from ``alpha_dd_search.make_features``.
It keeps the original 13 feature names and order, but lets a 99.5% observed
window contribute to rolling statistics and exposes the observed coverage that
was used.  Missing observations are never interpolated or replaced by zero.

Feature row ``t`` is shifted by one bar: every value is computed from bars no
later than ``t - 1``.  The 90-day drawdown is the maximum of observed prices in
the window; it is not an estimate of an unobserved intrabar or missing-bar
extremum.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .alpha_dd_search import FEATURE_NAMES as BASE_FEATURE_NAMES


BARS_DAY = 96
BARS_YEAR = BARS_DAY * 365
OBSERVED_COVERAGE_THRESHOLD = 0.995
COVERAGE_FEATURE_NAMES = (
    "return_coverage_7",
    "price_coverage_90",
    "flow_coverage_7",
)
FEATURE_NAMES = tuple(BASE_FEATURE_NAMES) + COVERAGE_FEATURE_NAMES

_REQUIRED_COLUMNS = ("close", "quote_volume", "taker_buy_quote")


def _minimum_observations(window: int) -> int:
    """Return the registered 99.5%-coverage rolling threshold."""
    return int(math.ceil(OBSERVED_COVERAGE_THRESHOLD * window))


def _validate_bars(bars: pd.DataFrame) -> None:
    if not isinstance(bars, pd.DataFrame):
        raise TypeError("bars must be a pandas DataFrame")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars must have a DatetimeIndex")
    if bars.index.has_duplicates or not bars.index.is_monotonic_increasing:
        raise ValueError("bars index must be unique and monotonic increasing")
    missing = [column for column in _REQUIRED_COLUMNS if column not in bars.columns]
    if missing:
        raise ValueError(f"missing feature input columns: {missing}")


def _numeric_series(bars: pd.DataFrame, column: str) -> pd.Series:
    try:
        values = bars[column].astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be numeric") from exc
    return pd.Series(values.to_numpy(copy=False), index=bars.index, dtype=float)


def _history_ready(index: pd.Index, window: int, *, extra_bars: int = 0) -> pd.Series:
    """Require elapsed row history, independent of observed-value coverage."""
    positions = np.arange(len(index))
    return pd.Series(
        positions >= window - 1 + extra_bars,
        index=index,
        dtype=bool,
    )


def _rolling_stat(
    values: pd.Series,
    window: int,
    *,
    statistic: str,
    extra_history_bars: int = 0,
) -> pd.Series:
    """Compute an observed rolling statistic without shortening its window."""
    minimum = _minimum_observations(window)
    rolling = values.rolling(window=window, min_periods=minimum)
    if statistic == "std":
        result = rolling.std()
    elif statistic == "mean":
        result = rolling.mean()
    elif statistic == "max":
        result = rolling.max()
    elif statistic == "count":
        result = rolling.count()
    else:
        raise ValueError(f"unsupported rolling statistic: {statistic}")
    # min_periods allows sparse observations before a complete nominal window
    # has elapsed.  Keep those rows unavailable rather than changing the
    # experiment's initial-history semantics.
    ready = _history_ready(values.index, window, extra_bars=extra_history_bars)
    return result.where(ready)


def _coverage(
    values: pd.Series,
    window: int,
    *,
    extra_history_bars: int = 0,
) -> pd.Series:
    # Coverage itself remains observable below the eligibility threshold; the
    # associated statistic is what becomes NaN when too few observations exist.
    # This lets the model see the measured gap fraction instead of confusing it
    # with an unobserved coverage value.
    count = values.rolling(window=window, min_periods=1).count()
    count = count.where(
        _history_ready(values.index, window, extra_bars=extra_history_bars)
    )
    return count / float(window)


def make_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Build the causal 16-column feature frame.

    Rolling statistics use ``ceil(0.995 * window)`` finite observations, but
    retain a full nominal elapsed history before becoming eligible.  A missing
    close or a non-positive close is unavailable; a zero quote volume makes
    the flow imbalance unavailable.  No missing value is backfilled or turned
    into a numerical zero.
    """
    _validate_bars(bars)
    close = _numeric_series(bars, "close")
    quote_volume = _numeric_series(bars, "quote_volume")
    taker_buy_quote = _numeric_series(bars, "taker_buy_quote")

    valid_close = close.notna() & np.isfinite(close) & (close > 0.0)
    log_price = np.log(close.where(valid_close))
    returns = log_price.diff()

    out = pd.DataFrame(index=bars.index)
    for days in (1, 7, 30, 90):
        window = days * BARS_DAY
        # Endpoint-to-endpoint momentum intentionally does not sum or
        # interpolate over interior gaps.  Coverage is exposed separately.
        out[f"momentum_{days}"] = log_price.diff(window)

    for days in (1, 7, 30):
        window = days * BARS_DAY
        out[f"vol_{days}"] = (
            _rolling_stat(
                returns,
                window,
                statistic="std",
                extra_history_bars=1,
            )
            * np.sqrt(BARS_YEAR)
        )

    for days in (7, 30, 90):
        window = days * BARS_DAY
        observed_max = _rolling_stat(log_price, window, statistic="max")
        # The max is explicitly observed-only.  Missing prices are not
        # reconstructed as a hidden high or low.
        out[f"drawdown_{days}"] = np.expm1(log_price - observed_max)

    vol_1 = out["vol_1"]
    vol_30 = out["vol_30"]
    out["vol_ratio"] = vol_1 / vol_30.where(vol_30 > 0.0)

    positive_quote_volume = quote_volume.where(
        quote_volume.notna() & np.isfinite(quote_volume) & (quote_volume > 0.0)
    )
    flow = (2.0 * taker_buy_quote / positive_quote_volume - 1.0).where(
        taker_buy_quote.notna() & np.isfinite(taker_buy_quote)
    )
    flow = flow.where(np.isfinite(flow))
    out["flow_1"] = _rolling_stat(flow, BARS_DAY, statistic="mean")
    out["flow_7"] = _rolling_stat(flow, 7 * BARS_DAY, statistic="mean")

    # Coverage is computed on the same observed series used by the associated
    # statistic and shifted with every other feature.  Return coverage needs
    # one extra historical bar because the first log return is undefined.
    out["return_coverage_7"] = _coverage(
        returns,
        7 * BARS_DAY,
        extra_history_bars=1,
    )
    out["price_coverage_90"] = _coverage(
        close.where(valid_close),
        90 * BARS_DAY,
    )
    out["flow_coverage_7"] = _coverage(
        flow,
        7 * BARS_DAY,
    )

    # All feature values, including explicit coverage, describe information
    # available before the decision bar.
    return out.loc[:, FEATURE_NAMES].shift(1)


__all__ = [
    "BASE_FEATURE_NAMES",
    "BARS_DAY",
    "BARS_YEAR",
    "COVERAGE_FEATURE_NAMES",
    "FEATURE_NAMES",
    "OBSERVED_COVERAGE_THRESHOLD",
    "make_features",
]
