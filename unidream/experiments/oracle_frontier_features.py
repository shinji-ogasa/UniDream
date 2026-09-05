"""Small, causal feature families for the exploratory oracle frontier.

Every output at decision time ``t`` uses bars through ``t - 1``.  Missing grid
rows remain missing observations: rolling windows require their full nominal
history and at least 99.5% finite observations, without interpolation.  The
base16 family is the existing AlphaEx/MaxDD feature frame.

RSI uses simple trailing gain/loss means, ATR uses a simple trailing true-range
mean, and channel position uses the observed high/low channel.  Constant-price
windows have RSI 50, channel position 0.5, price z-score 0, and efficiency 0.
The downside/upside feature is the log ratio of return RMS values with a 1e-8
return floor added to both sides; this keeps one-sided and flat paths finite.
No missing input is assigned one of these constant-window values.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .alpha_dd_features import OBSERVED_COVERAGE_THRESHOLD, make_features


TECHNICAL_FEATURE_NAMES = (
    "rsi14", "rsi96", "atr14_relative", "atr96_relative",
    "channel_position96", "channel_position672",
    "price_zscore96", "price_zscore672", "downside_upside_log_vol_ratio96",
    "efficiency_ratio96", "efficiency_ratio672", "weighted_flow96", "weighted_flow672",
)
FLOW_FEATURE_NAMES = (
    "weighted_flow96", "weighted_flow672",
    "quote_volume_intensity96", "quote_volume_intensity672",
)
TRADE_FEATURE_NAMES = (
    "trade_intensity96", "trade_intensity672",
    "mean_trade_size_relative96", "mean_trade_size_relative672",
)
RETURN_RMS_FLOOR = 1e-8


def _complete_grid(bars: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(bars, pd.DataFrame):
        raise TypeError("bars must be a pandas DataFrame")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars must have a DatetimeIndex")
    if bars.empty or bars.index.hasnans:
        raise ValueError("bars must have a nonempty, finite timestamp index")
    if bars.index.has_duplicates or not bars.index.is_monotonic_increasing:
        raise ValueError("bars index must be unique and monotonic increasing")
    if np.any(bars.index.asi8 % pd.Timedelta(minutes=15).value):
        raise ValueError("bars timestamps must be aligned to the 15-minute grid")
    required = {"close", "high", "low", "quote_volume", "taker_buy_quote"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"missing feature input columns: {missing}")
    grid = pd.date_range(bars.index[0], bars.index[-1], freq="15min", name=bars.index.name)
    return bars.reindex(grid)


def _finite(bars: pd.DataFrame, column: str, *, positive: bool = False) -> pd.Series:
    try:
        values = bars[column].astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be numeric") from exc
    valid = np.isfinite(values)
    if positive:
        valid &= values > 0.0
    return values.where(valid)


def _rolling(values: pd.Series, window: int, statistic: str, *, difference: bool = False) -> pd.Series:
    minimum = math.ceil(OBSERVED_COVERAGE_THRESHOLD * window)
    result = getattr(values.rolling(window, min_periods=minimum), statistic)()
    ready = np.arange(len(values)) >= window - 1 + int(difference)
    return result.where(ready)


def _constant_ratio(numerator: pd.Series, denominator: pd.Series, neutral: float) -> pd.Series:
    """Divide measured values, with an explicit finite zero/zero convention."""
    result = numerator / denominator.where(denominator > 0.0)
    measured_flat = numerator.notna() & denominator.eq(0.0) & numerator.eq(0.0)
    return result.mask(measured_flat, neutral)


def make_feature_groups(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return base16, technical (29), and flow (20 or 24) feature frames.

    Absent timestamps are inserted on a complete 15-minute grid.  A trades
    column is optional: ``n_trades`` is preferred, with ``trades`` as an alias.
    Schema depends only on column availability, never on future observations.
    Zero quote volume or zero trades is unavailable for the corresponding
    intensity/flow feature.  Trade size is quote volume per observed trade.
    """
    bars = _complete_grid(bars)
    base = make_features(bars)
    close = _finite(bars, "close", positive=True)
    high = _finite(bars, "high", positive=True)
    low = _finite(bars, "low", positive=True)
    valid_candle = high.ge(low) & high.ge(close) & low.le(close)
    high = high.where(valid_candle)
    low = low.where(valid_candle)
    log_price = np.log(close)
    returns = log_price.diff()
    price_change = close.diff()
    gain = price_change.clip(lower=0.0)
    loss = -price_change.clip(upper=0.0)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1,
    ).max(axis=1, skipna=False)

    technical = pd.DataFrame(index=bars.index)
    for window in (14, 96):
        avg_gain = _rolling(gain, window, "mean", difference=True)
        avg_loss = _rolling(loss, window, "mean", difference=True)
        technical[f"rsi{window}"] = 100.0 * _constant_ratio(avg_gain, avg_gain + avg_loss, 0.5)
        technical[f"atr{window}_relative"] = _rolling(true_range, window, "mean", difference=True) / close

    for window in (96, 672):
        observed_high = _rolling(high, window, "max")
        observed_low = _rolling(low, window, "min")
        technical[f"channel_position{window}"] = _constant_ratio(
            close - observed_low, observed_high - observed_low, 0.5,
        )
        mean = _rolling(close, window, "mean")
        std = _rolling(close, window, "std")
        technical[f"price_zscore{window}"] = _constant_ratio(close - mean, std, 0.0)
        total_movement = _rolling(returns.abs(), window, "sum", difference=True)
        technical[f"efficiency_ratio{window}"] = _constant_ratio(
            log_price.diff(window).abs(), total_movement, 0.0,
        )

    downside = np.sqrt(_rolling(returns.clip(upper=0.0).pow(2), 96, "mean", difference=True))
    upside = np.sqrt(_rolling(returns.clip(lower=0.0).pow(2), 96, "mean", difference=True))
    technical["downside_upside_log_vol_ratio96"] = np.log(
        (downside + RETURN_RMS_FLOOR) / (upside + RETURN_RMS_FLOOR)
    )

    quote = _finite(bars, "quote_volume", positive=True)
    buy_quote = _finite(bars, "taker_buy_quote")
    valid_flow = buy_quote.ge(0.0) & buy_quote.le(quote)
    paired_quote = quote.where(valid_flow)
    signed_quote = (2.0 * buy_quote - quote).where(valid_flow)
    flow = pd.DataFrame(index=bars.index)
    for window in (96, 672):
        weighted_flow = _rolling(signed_quote, window, "sum") / _rolling(paired_quote, window, "sum")
        technical[f"weighted_flow{window}"] = weighted_flow
        flow[f"weighted_flow{window}"] = weighted_flow
        flow[f"quote_volume_intensity{window}"] = quote / _rolling(quote, window, "mean") - 1.0

    trade_column = next((name for name in ("n_trades", "trades") if name in bars.columns), None)
    flow_names = FLOW_FEATURE_NAMES
    if trade_column is not None:
        trades = _finite(bars, trade_column, positive=True)
        trade_size = quote / trades
        for window in (96, 672):
            flow[f"trade_intensity{window}"] = trades / _rolling(trades, window, "mean") - 1.0
            flow[f"mean_trade_size_relative{window}"] = trade_size / _rolling(trade_size, window, "mean") - 1.0
        flow_names += TRADE_FEATURE_NAMES

    # The existing base frame already applies this same one-bar shift.
    technical = technical.loc[:, TECHNICAL_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).shift(1)
    flow = flow.loc[:, flow_names].replace([np.inf, -np.inf], np.nan).shift(1)
    return {
        "base16": base,
        "technical": pd.concat([base, technical], axis=1),
        "flow": pd.concat([base, flow], axis=1),
    }


__all__ = ["FLOW_FEATURE_NAMES", "TECHNICAL_FEATURE_NAMES", "TRADE_FEATURE_NAMES", "make_feature_groups"]
