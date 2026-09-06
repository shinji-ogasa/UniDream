"""Fixed short-price/flow blocks on the unchanged technical29 feature frame.

Raw inputs use UTC bar-open time. Every new output at t uses bars through
t-15min, with one final full-grid shift and no filling or interpolation.
This archive event-time contract does not establish actual receipt-time access.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .oracle_derivative_features import _raw_frame, make_derivative_groups

PRICE_FEATURE_NAMES = (
    "spot_log_return4", "spot_log_return16", "spot_log_return48",
    "spot_body_sign1", "spot_close_location1",
)
FLOW_FEATURE_NAMES = (
    "spot_weighted_flow4", "perp_weighted_flow4", "spot_quote_activity24_672",
)


def _numeric(frame, name, *, positive=False):
    try:
        source = frame[name]
    except KeyError as exc:
        raise ValueError(f"missing raw input column: {name}") from exc
    if np.iscomplexobj(source):
        raise ValueError(f"{name} must be real numeric data")
    try:
        values = source.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be real numeric data") from exc
    valid = np.isfinite(values)
    if positive:
        valid &= values > 0
    return values.where(valid)


def _rolling(values, window, statistic):
    result = getattr(values.rolling(window, min_periods=math.ceil(.995 * window)), statistic)()
    return result.where(np.arange(len(values)) >= window - 1)


def _weighted_flow4(frame):
    quote = _numeric(frame, "quote_volume", positive=True)
    buy = _numeric(frame, "taker_buy_quote")
    paired = buy.ge(0) & buy.le(quote)
    return (_rolling((2 * buy - quote).where(paired), 4, "sum")
            / _rolling(quote.where(paired), 4, "sum"))


def make_short_feature_groups(spot, um):
    """Return technical29 and its price5, flow3, and combined8 augmentations.

    Return k requires all k+1 positive finite closes, including both endpoints;
    it never jumps across a missing close. Body sign and close location require
    positive finite OHLC with low <= open,close <= high. A measured zero-range
    candle has open=close=high=low: body sign and close location are zero.
    Missing or inconsistent candles do not receive that neutral value.

    Four-bar signed flow is sum(2*taker_buy_quote-quote)/sum(quote), requiring
    four valid positive quotes and taker values in [0,quote]. Spot quote activity
    is log(mean_quote24/mean_quote672), using independent valid positive quotes,
    thresholds 24/24 and 669/672, and a full nominal 672-bar history.

    The inherited guard validates timing metadata when present and reindexes
    sparse UM onto the complete Spot grid. Technical29 is returned exactly from
    the inherited helper. Only the new columns receive the final shift; UM gets
    no additional delay. No outcomes, coefficients or evaluation masks enter.
    """
    spot = _raw_frame(spot, "Spot", complete=True)
    um = _raw_frame(um, "UM", complete=False).reindex(spot.index)
    technical = make_derivative_groups(spot, um)["technical"]
    close = _numeric(spot, "close", positive=True)
    opening = _numeric(spot, "open", positive=True)
    high = _numeric(spot, "high", positive=True)
    low = _numeric(spot, "low", positive=True)
    price = pd.DataFrame(index=spot.index)
    log_close = np.log(close)
    for window in (4, 16, 48):
        complete = close.notna().astype(int).rolling(window + 1, min_periods=window + 1).sum().eq(window + 1)
        price[f"spot_log_return{window}"] = log_close.diff(window).where(complete)
    candle = (opening.notna() & close.notna() & high.notna() & low.notna()
              & high.ge(low) & high.ge(opening) & high.ge(close)
              & low.le(opening) & low.le(close))
    price["spot_body_sign1"] = np.sign(np.log(close / opening)).where(candle)
    span = (high - low).where(candle)
    location = ((2 * close - high - low) / span.where(span > 0)).where(candle)
    price["spot_close_location1"] = location.mask(candle & span.eq(0), 0.)
    flow = pd.DataFrame(index=spot.index)
    flow["spot_weighted_flow4"] = _weighted_flow4(spot)
    flow["perp_weighted_flow4"] = _weighted_flow4(um)
    quote = _numeric(spot, "quote_volume", positive=True)
    flow["spot_quote_activity24_672"] = np.log(
        _rolling(quote, 24, "mean") / _rolling(quote, 672, "mean"))
    price = price.loc[:, PRICE_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).shift(1)
    flow = flow.loc[:, FLOW_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).shift(1)
    return {"technical": technical,
            "technical_short_price": pd.concat([technical, price], axis=1),
            "technical_short_flow": pd.concat([technical, flow], axis=1),
            "technical_short_both": pd.concat([technical, price, flow], axis=1)}


__all__ = ["PRICE_FEATURE_NAMES", "FLOW_FEATURE_NAMES", "make_short_feature_groups"]
