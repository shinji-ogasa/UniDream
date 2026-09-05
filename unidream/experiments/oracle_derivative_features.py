"""The eight raw UM/Spot features fixed by the derivative data preflight.

Inputs are raw bar-OPEN frames. Calculations use the Spot 15-minute grid, then
shift exactly once, so a decision at t uses bars through t-15min. Stored
decision_ts is validated as open+15min; it must not replace the raw input index.
This module performs no data acquisition, fitting, target construction or scoring.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .oracle_frontier_features import make_feature_groups


INTERVAL = pd.Timedelta(minutes=15)
OBSERVED_COVERAGE = .995
REALIZED_VARIANCE_EPSILON = 1e-12
PERP_FLOW_FEATURE_NAMES = ("perp_weighted_flow24", "perp_weighted_flow96")
DERIVATIVE_FEATURE_NAMES = (
    "perp_weighted_flow24", "perp_minus_spot_flow24",
    "perp_weighted_flow96", "perp_minus_spot_flow96",
    "relative_quote_activity24_672", "traded_close_basis",
    "traded_close_basis_change24", "relative_realized_variance24",
)
_RAW_INPUTS = ("quote_volume", "taker_buy_quote", "close")


def _raw_frame(bars: pd.DataFrame, name: str, *, complete: bool) -> pd.DataFrame:
    if not isinstance(bars, pd.DataFrame):
        raise TypeError(f"{name} must be a DataFrame")
    if bars.columns.has_duplicates:
        raise ValueError(f"{name} columns must be unique")
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError(f"{name} requires timezone-aware raw bar-open timestamps")
    if bars.index.name in ("decision_ts", "bar_close_ts"):
        raise ValueError(f"{name} index must be raw bar-open time, not {bars.index.name}")
    index = bars.index.tz_convert("UTC")
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{name} timestamps must be finite, unique and increasing")
    if np.any(index.asi8 % INTERVAL.value):
        raise ValueError(f"{name} timestamps must align to the 15-minute grid")
    if complete and (not len(index) or np.any(np.diff(index.asi8) != INTERVAL.value)):
        raise ValueError("Spot must retain the complete nonempty 15-minute grid")
    missing = [column for column in _RAW_INPUTS if column not in bars.columns]
    if missing:
        raise ValueError(f"{name} missing raw input columns: {missing}")
    out = bars.copy(deep=False)
    out.index = index
    observed = out.loc[:, _RAW_INPUTS].notna().any(axis=1).to_numpy()
    expected_times = {
        "bar_open_ts": index,
        "bar_close_ts": index + INTERVAL - pd.Timedelta(milliseconds=1),
        "decision_ts": index + INTERVAL,
    }
    for column, expected in expected_times.items():
        if column not in out.columns:
            continue
        supplied = out[column].notna().to_numpy()
        if np.any(observed & ~supplied):
            raise ValueError(f"{name} observed raw rows have missing {column}")
        if not supplied.any():
            continue
        try:
            actual = pd.DatetimeIndex(out.loc[supplied, column])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} {column} must contain timezone-aware timestamps") from exc
        if actual.tz is None:
            raise ValueError(f"{name} {column} must be timezone-aware")
        if not actual.tz_convert("UTC").equals(expected[supplied]):
            raise ValueError(f"{name} {column} differs from raw bar-open timing contract")
    return out


def _inputs(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    try:
        quote = bars.quote_volume.astype(float)
        buy = bars.taker_buy_quote.astype(float)
        close = bars.close.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("raw close and quote-volume inputs must be numeric") from exc
    quote = quote.where(np.isfinite(quote) & quote.gt(0))
    buy = buy.where(np.isfinite(buy))
    valid_flow = buy.ge(0) & buy.le(quote)
    close = close.where(np.isfinite(close) & close.gt(0))
    return quote, quote.where(valid_flow), (2 * buy - quote).where(valid_flow), close


def _rolling(values: pd.Series, window: int, statistic: str = "mean", *, difference: bool = False) -> pd.Series:
    value = getattr(values.rolling(window, min_periods=math.ceil(OBSERVED_COVERAGE * window)), statistic)()
    return value.where(np.arange(len(values)) >= window - 1 + int(difference))


def make_derivative_groups(spot: pd.DataFrame, um: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return exact base16/technical, technical+UM flow, and technical+all eight.

    Output names/dimensions are base16/16, technical/29, perp_flow/31 and
    derivative/37. A timezone-aware representation is converted to UTC. Spot
    must retain its complete grid; sparse UM rows are reindexed onto that grid
    as NaN. Timing columns, when supplied, must describe the raw open index.

    Coverage thresholds are 24/24, 96/96 and 669/672. Quote activity deliberately
    uses each market's independent valid-volume rolling means, as registered in
    the preflight. It does not restrict both markets to their joint-valid rows,
    nor restrict volume activity to rows with valid taker flow. Nominal history
    is counted on the Spot grid, including before the first observed UM row.

    Basis means traded-close divergence, not mark-price basis, funding or OI.
    The 1e-12 realized-variance floor regularizes measured zero volatility;
    missing returns remain missing. No field is filled or interpolated.
    """
    spot = _raw_frame(spot, "Spot", complete=True)
    um = _raw_frame(um, "UM", complete=False).reindex(spot.index)
    old = make_feature_groups(spot)
    qs, paired_qs, signed_s, close_s = _inputs(spot)
    qp, paired_qp, signed_p, close_p = _inputs(um)
    new = pd.DataFrame(index=spot.index)
    for window in (24, 96):
        flow_s = _rolling(signed_s, window, "sum") / _rolling(paired_qs, window, "sum")
        flow_p = _rolling(signed_p, window, "sum") / _rolling(paired_qp, window, "sum")
        new[f"perp_weighted_flow{window}"] = flow_p
        new[f"perp_minus_spot_flow{window}"] = flow_p - flow_s
    new["relative_quote_activity24_672"] = np.log(
        (_rolling(qp, 24) / _rolling(qp, 672)) / (_rolling(qs, 24) / _rolling(qs, 672)))
    basis = np.log(close_p / close_s)
    new["traded_close_basis"] = basis
    new["traded_close_basis_change24"] = basis.diff(24)
    return_p, return_s = np.log(close_p).diff(), np.log(close_s).diff()
    new["relative_realized_variance24"] = np.log(
        (_rolling(return_p.pow(2), 24, "sum", difference=True) + REALIZED_VARIANCE_EPSILON)
        / (_rolling(return_s.pow(2), 24, "sum", difference=True) + REALIZED_VARIANCE_EPSILON))
    new = new.loc[:, DERIVATIVE_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).shift(1)
    return {
        "base16": old["base16"], "technical": old["technical"],
        "perp_flow": pd.concat([old["technical"], new.loc[:, PERP_FLOW_FEATURE_NAMES]], axis=1),
        "derivative": pd.concat([old["technical"], new], axis=1),
    }


__all__ = ["DERIVATIVE_FEATURE_NAMES", "PERP_FLOW_FEATURE_NAMES", "make_derivative_groups"]
