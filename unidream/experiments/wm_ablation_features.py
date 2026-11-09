"""Window-local OHLCV features for the Transformer WM ablation.

The unit of computation is the *whole model input window*, not the complete
dataset.  Each window contains completed bars only.  If a raw timestamp is a
15-minute bar's opening time, a decision at ``t`` may include bars through
``t - 15 minutes``.  Every indicator, including EMA, Wilder averages and OBV,
starts again at the beginning of that same window.  Thus technical features
cannot expose raw observations preceding the no-indicator model's context.

Rolling estimates use the available prefix during warm-up, with no padding
from earlier observations, interpolation, learned parameters, or fitting.
EMA periods are smoothing parameters; their finite support is the supplied
window.  Zero denominators have an explicitly neutral value.  Invalid raw
inputs are rejected rather than silently imputed.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


FEATURE_VERSION = "window_local_ohlcv_ta44_v1"
RAW_COLUMNS = ("open", "high", "low", "close", "volume")
BASE_COLUMNS = (
    "log_open_previous_close",
    "log_high_previous_close",
    "log_low_previous_close",
    "log_close_previous_close",
    "log1p_volume",
)
TECHNICAL_COLUMNS = (
    "momentum_4", "momentum_16", "momentum_64", "momentum_96",
    "sma_gap_8", "sma_gap_16", "sma_gap_64",
    "ema_gap_12", "ema_gap_26", "ema_gap_64",
    "macd_12_26", "macd_signal_9", "macd_histogram",
    "rsi_14", "rsi_28",
    "stochastic_14", "stochastic_signal_3", "williams_r_14",
    "cci_20",
    "realized_vol_4", "realized_vol_16", "realized_vol_64",
    "atr_ratio_14", "atr_ratio_28",
    "bollinger_position_20", "bollinger_width_20",
    "downside_vol_16", "downside_vol_64",
    "obv_flow_4", "obv_flow_16", "obv_flow_64",
    "cmf_20", "mfi_14",
    "vwap_gap_16", "vwap_gap_64",
    "relative_volume_16", "relative_volume_64",
    "adx_14", "plus_di_14", "minus_di_14",
    "donchian_position_16", "donchian_position_64",
    "donchian_width_16", "donchian_width_64",
)
# Short alias used by the experiment runner's common-width channel masks.
TA_COLUMNS = TECHNICAL_COLUMNS
GROUPS = ("A", "B", "C", "D")
BAR_DURATION = pd.Timedelta(minutes=15)


def feature_columns(
    group: str, market_columns: Sequence[str] = ()
) -> tuple[str, ...]:
    """Return the exact registered order; C/D require named market inputs."""
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}, got {group!r}")
    market_columns = tuple(market_columns)
    if group in ("C", "D") and not market_columns:
        raise ValueError("groups C and D require market_columns")
    if len(set(market_columns)) != len(market_columns):
        raise ValueError("market_columns must be unique")
    if set(market_columns).intersection(BASE_COLUMNS + TECHNICAL_COLUMNS):
        raise ValueError("market_columns collide with registered OHLCV features")
    return (
        BASE_COLUMNS
        + (market_columns if group in ("C", "D") else ())
        + (TECHNICAL_COLUMNS if group in ("B", "D") else ())
    )


def _divide(
    numerator: np.ndarray, denominator: np.ndarray, *, neutral: float = 0.0
) -> np.ndarray:
    shape = np.broadcast_shapes(numerator.shape, denominator.shape)
    result = np.full(shape, neutral, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0.0)
    return result


def _lag(values: np.ndarray, periods: int = 1) -> np.ndarray:
    positions = np.maximum(np.arange(values.shape[1]) - periods, 0)
    return values[:, positions]


def _rolling_sum(values: np.ndarray, period: int) -> np.ndarray:
    cumulative = np.cumsum(values, axis=1, dtype=np.float64)
    result = cumulative.copy()
    result[:, period:] -= cumulative[:, :-period]
    return result


def _rolling_mean(values: np.ndarray, period: int) -> np.ndarray:
    counts = np.minimum(np.arange(1, values.shape[1] + 1), period)
    return _rolling_sum(values, period) / counts


def _rolling_extreme(values: np.ndarray, period: int, *, maximum: bool) -> np.ndarray:
    # Repeating the first value preserves the exact prefix extrema.
    padded = np.pad(values, ((0, 0), (period - 1, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, period, axis=1)
    return np.max(windows, axis=-1) if maximum else np.min(windows, axis=-1)


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    result[:, 0] = values[:, 0]
    for position in range(1, values.shape[1]):
        result[:, position] = (
            alpha * values[:, position] + (1.0 - alpha) * result[:, position - 1]
        )
    return result


def _technical_features(raw: np.ndarray) -> np.ndarray:
    open_, high, low, close, volume = (raw[:, :, i] for i in range(5))
    previous_close = _lag(close)
    # The first return is unknown outside the supplied context and is zero.
    returns = np.log(close / previous_close)
    changes = close - previous_close
    gains, losses = np.maximum(changes, 0.0), np.maximum(-changes, 0.0)
    typical = (high + low + close) / 3.0
    out: dict[str, np.ndarray] = {}

    for period in (4, 16, 64, 96):
        out[f"momentum_{period}"] = np.log(close / _lag(close, period))
    for period in (8, 16, 64):
        out[f"sma_gap_{period}"] = np.log(close / _rolling_mean(close, period))
    emas = {period: _ema(close, 2.0 / (period + 1.0)) for period in (12, 26, 64)}
    for period, ema in emas.items():
        out[f"ema_gap_{period}"] = np.log(close / ema)
    macd_price = emas[12] - emas[26]
    signal_price = _ema(macd_price, 2.0 / 10.0)
    out["macd_12_26"] = macd_price / close
    out["macd_signal_9"] = signal_price / close
    out["macd_histogram"] = (macd_price - signal_price) / close

    for period in (14, 28):
        avg_gain = _ema(gains, 1.0 / period)
        avg_loss = _ema(losses, 1.0 / period)
        out[f"rsi_{period}"] = _divide(
            avg_gain, avg_gain + avg_loss, neutral=0.5
        )
    low14 = _rolling_extreme(low, 14, maximum=False)
    high14 = _rolling_extreme(high, 14, maximum=True)
    stochastic = _divide(close - low14, high14 - low14, neutral=0.5)
    out["stochastic_14"] = stochastic
    out["stochastic_signal_3"] = _rolling_mean(stochastic, 3)
    out["williams_r_14"] = stochastic - 1.0

    typical_mean = _rolling_mean(typical, 20)
    # CCI uses mean absolute deviation about each *current* window mean,
    # not a rolling mean of previously computed absolute deviations.
    deviation = np.zeros_like(typical)
    for lag in range(min(20, raw.shape[1])):
        valid = np.arange(raw.shape[1]) >= lag
        deviation[:, valid] += np.abs(
            typical[:, np.arange(raw.shape[1])[valid] - lag]
            - typical_mean[:, valid]
        )
    deviation /= np.minimum(np.arange(1, raw.shape[1] + 1), 20)
    out["cci_20"] = _divide(typical - typical_mean, 0.015 * deviation)

    for period in (4, 16, 64):
        out[f"realized_vol_{period}"] = np.sqrt(
            np.maximum(_rolling_mean(returns * returns, period), 0.0)
        )
    true_range = np.maximum(high - low, np.maximum(
        np.abs(high - previous_close), np.abs(low - previous_close)
    ))
    for period in (14, 28):
        out[f"atr_ratio_{period}"] = _ema(true_range, 1.0 / period) / close

    mean20 = _rolling_mean(close, 20)
    # Center prices first to avoid cancellation when close is nearly flat.
    centered = close - close[:, :1]
    variance20 = np.maximum(
        _rolling_mean(centered * centered, 20)
        - _rolling_mean(centered, 20) ** 2,
        0.0,
    )
    std20 = np.sqrt(variance20)
    out["bollinger_position_20"] = _divide(close - mean20, 2.0 * std20)
    out["bollinger_width_20"] = 4.0 * std20 / mean20
    for period in (16, 64):
        downside = np.minimum(returns, 0.0)
        out[f"downside_vol_{period}"] = np.sqrt(
            np.maximum(_rolling_mean(downside * downside, period), 0.0)
        )
    signed_volume = np.sign(changes) * volume
    for period in (4, 16, 64):
        # Equivalent to local OBV increments, made independent of volume units.
        out[f"obv_flow_{period}"] = _divide(
            _rolling_sum(signed_volume, period), _rolling_sum(volume, period)
        )
    money_flow_multiplier = _divide(2.0 * close - high - low, high - low)
    out["cmf_20"] = _divide(
        _rolling_sum(money_flow_multiplier * volume, 20), _rolling_sum(volume, 20)
    )
    raw_money_flow = typical * volume
    typical_direction = np.sign(typical - _lag(typical))
    positive_money = _rolling_sum(raw_money_flow * (typical_direction > 0), 14)
    negative_money = _rolling_sum(raw_money_flow * (typical_direction < 0), 14)
    out["mfi_14"] = _divide(
        positive_money, positive_money + negative_money, neutral=0.5
    )
    for period in (16, 64):
        vwap = _divide(
            _rolling_sum(typical * volume, period),
            _rolling_sum(volume, period),
        )
        # If a whole interval has no trades, a neutral gap is explicit.
        out[f"vwap_gap_{period}"] = np.log(_divide(close, vwap, neutral=1.0))
        out[f"relative_volume_{period}"] = _divide(
            volume, _rolling_mean(volume, period), neutral=1.0
        ) - 1.0

    upward = high - _lag(high)
    downward = _lag(low) - low
    plus_dm = np.where((upward > downward) & (upward > 0), upward, 0.0)
    minus_dm = np.where((downward > upward) & (downward > 0), downward, 0.0)
    smoothed_range = _ema(true_range, 1.0 / 14.0)
    plus_di = _divide(_ema(plus_dm, 1.0 / 14.0), smoothed_range)
    minus_di = _divide(_ema(minus_dm, 1.0 / 14.0), smoothed_range)
    dx = _divide(np.abs(plus_di - minus_di), plus_di + minus_di)
    out["adx_14"] = _ema(dx, 1.0 / 14.0)
    out["plus_di_14"] = plus_di
    out["minus_di_14"] = minus_di
    for period in (16, 64):
        highest = _rolling_extreme(high, period, maximum=True)
        lowest = _rolling_extreme(low, period, maximum=False)
        out[f"donchian_position_{period}"] = _divide(
            close - lowest, highest - lowest, neutral=0.5
        )
        out[f"donchian_width_{period}"] = (highest - lowest) / close
    return np.stack([out[name] for name in TECHNICAL_COLUMNS], axis=-1)


def transform_windows(
    raw_ohlcv: np.ndarray,
    group: str = "A",
    market_windows: np.ndarray | None = None,
    market_columns: Sequence[str] = (),
) -> np.ndarray:
    """Transform ``[batch, time, OHLCV]`` into finite float32 input features.

    A/B contain identical five base columns, with B adding TA44.  C adds the
    supplied market columns to A; D adds the *identical* TA44 to C.  Market
    values must already be causally aligned, with published values available
    by the decision time.  This function performs no cross-window scaling.
    """
    columns = feature_columns(group, market_columns)
    raw = np.asarray(raw_ohlcv, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1] < 1 or raw.shape[2] != 5:
        raise ValueError("raw_ohlcv must have shape [batch, nonempty time, 5]")
    if not np.isfinite(raw).all():
        raise ValueError("raw_ohlcv contains missing or nonfinite values")
    if np.any(raw[:, :, :4] <= 0.0) or np.any(raw[:, :, 4] < 0.0):
        raise ValueError("OHLC prices must be positive and volume nonnegative")
    open_, high, low, close, volume = (raw[:, :, i] for i in range(5))
    if np.any(high < np.maximum(open_, close)) or np.any(low > np.minimum(open_, close)):
        raise ValueError("OHLC high/low do not bracket open and close")
    previous_close = _lag(close).copy()
    previous_close[:, 0] = open_[:, 0]
    base = np.stack((
        np.log(open_ / previous_close),
        np.log(high / previous_close),
        np.log(low / previous_close),
        np.log(close / previous_close),
        np.log1p(volume),
    ), axis=-1)
    parts = [base]
    if group in ("C", "D"):
        market = np.asarray(market_windows, dtype=np.float64)
        if market.shape != (*raw.shape[:2], len(tuple(market_columns))):
            raise ValueError("market_windows shape does not match raw windows/market_columns")
        if not np.isfinite(market).all():
            raise ValueError("market_windows contains missing or nonfinite values")
        parts.append(market)
    if group in ("B", "D"):
        parts.append(_technical_features(raw))
    result = np.concatenate(parts, axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("feature values exceed finite float32 representation")
    if result.shape[-1] != len(columns):
        raise RuntimeError("feature implementation/registry mismatch")
    return result


def make_window_features(
    raw_window: pd.DataFrame,
    group: str = "A",
    market_columns: Sequence[str] = (),
    *,
    decision_time: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """DataFrame convenience wrapper for a single strictly contiguous window.

    Passing ``decision_time`` checks that the final supplied bar is precisely
    the last completed 15-minute bar.  Timestamps denote bar opens in UTC.
    Additional raw rows must be sliced away *before* calling this function.
    """
    if not isinstance(raw_window.index, pd.DatetimeIndex) or raw_window.empty:
        raise ValueError("raw_window requires a nonempty DatetimeIndex")
    index = raw_window.index
    if index.tz is None or str(index.tz) != "UTC":
        raise ValueError("raw_window timestamps must be timezone-aware UTC")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("raw_window timestamps must be unique and increasing")
    if len(index) > 1 and np.any(np.diff(index.asi8) != BAR_DURATION.value):
        raise ValueError("raw_window must be contiguous 15-minute bars")
    if decision_time is not None:
        decision = pd.Timestamp(decision_time)
        if decision.tzinfo is None:
            raise ValueError("decision_time must be timezone-aware")
        if index[-1] + BAR_DURATION != decision.tz_convert("UTC"):
            raise ValueError("raw_window must end at the last completed decision bar")
    missing = set(RAW_COLUMNS).difference(raw_window.columns)
    if group in ("C", "D"):
        missing.update(set(market_columns).difference(raw_window.columns))
    if missing:
        raise ValueError(f"missing input columns: {sorted(missing)}")
    raw = raw_window.loc[:, RAW_COLUMNS].to_numpy(dtype=np.float64)[None, :, :]
    market = (
        raw_window.loc[:, list(market_columns)].to_numpy(dtype=np.float64)[None, :, :]
        if group in ("C", "D") else None
    )
    features = transform_windows(raw, group, market, market_columns)
    return pd.DataFrame(features[0], index=index, columns=feature_columns(group, market_columns))
