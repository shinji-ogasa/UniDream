"""特徴量エンジニアリングモジュール.

OHLCV → 対数リターン + RSI + MACD + ATR + rolling z-score 正規化。

リーク防止方針:
  特徴量は「バー t の開始時に既知の情報のみ」で構成する。
  - 対数リターン: log(P_{t-1}/P_{t-2}) — バー t-1 の確定済みリターン（shift(1)）
  - TA 指標: バー t-1 までの close を使って計算（shift(1)）
  - get_raw_returns: バー t のリターン log(P_t/P_{t-1}) — エージェントが獲得する報酬（shift 無し）

zscore_window は **日数** で統一する（config: zscore_window_days: 60）。
バー数への変換は BARS_PER_DAY[interval] を使って自動計算する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

# 足種別の 1 日あたりバー数
BARS_PER_DAY: dict[str, int] = {
    "1m": 1440,
    "5m": 288,
    "15m": 96,
    "30m": 48,
    "1h": 24,
    "4h": 6,
    "1d": 1,
}


def days_to_bars(days: int, interval: str = "15m") -> int:
    """日数をバー数に変換する."""
    return days * BARS_PER_DAY.get(interval, 96)


# --- 個別指標計算 ---

def log_returns(series: pd.Series) -> pd.Series:
    """対数リターンを計算する（shift(1) 済み）.

    特徴量としてバー t に格納される値は log(P_{t-1}/P_{t-2})。
    バー t 開始時点で既知の情報のみを使う。
    """
    return np.log(series / series.shift(1)).shift(1)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI を計算する. shift(1) 済み."""
    if HAS_PANDAS_TA:
        rsi = ta.rsi(close, length=period)
    else:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
    return rsi.shift(1)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD と MACD シグナルを計算する. shift(1) 済み.

    Returns:
        DataFrame with columns ['macd', 'macd_signal']
    """
    if HAS_PANDAS_TA:
        result = ta.macd(close, fast=fast, slow=slow, signal=signal)
        macd_col = f"MACD_{fast}_{slow}_{signal}"
        signal_col = f"MACDs_{fast}_{slow}_{signal}"
        df = pd.DataFrame({
            "macd": result[macd_col],
            "macd_signal": result[signal_col],
        })
    else:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        df = pd.DataFrame({"macd": macd_line, "macd_signal": signal_line})
    return df.shift(1)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR（平均真値域）を計算する. shift(1) 済み."""
    if HAS_PANDAS_TA:
        atr = ta.atr(high, low, close, length=period)
    else:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
    return atr.shift(1)


# --- 正規化 ---

def rolling_zscore(series: pd.Series, window_bars: int) -> pd.Series:
    """rolling z-score 正規化.

    Args:
        series: 正規化対象の時系列
        window_bars: ウィンドウサイズ（バー数）。days_to_bars() で変換済みの値を渡す。
    """
    mean = series.rolling(window_bars, min_periods=window_bars // 4).mean()
    std = series.rolling(window_bars, min_periods=window_bars // 4).std()
    return (series - mean) / (std + 1e-8)


def atr_normalize_returns(returns: pd.Series, atr: pd.Series) -> pd.Series:
    """リターンを ATR で割ってボラ正規化する."""
    return returns / (atr + 1e-8)


def rolling_zscore_df(df: pd.DataFrame, window_bars: int) -> pd.DataFrame:
    """DataFrame 全カラムに rolling z-score を適用する."""
    return df.apply(lambda col: rolling_zscore(col, window_bars))


# --- Realized Volatility（マルチスケール）---

def compute_realized_vol(
    close: pd.Series,
    windows_bars: list[int],
    interval: str = "15m",
) -> pd.DataFrame:
    """マルチスケール Realized Volatility を計算する（shift(1) 済み）.

    RV = rolling std of log returns over window.
    各ウィンドウは **バー数** で指定する。

    Args:
        close: 終値系列
        windows_bars: ウィンドウサイズのリスト（バー数）
        interval: 足種（列名生成用）

    Returns:
        DataFrame with columns like ['rv_4', 'rv_16', 'rv_96']
    """
    log_ret = np.log(close / close.shift(1))
    cols = {}
    for w in windows_bars:
        cols[f"rv_{w}"] = log_ret.rolling(w, min_periods=w // 2).std()
    df = pd.DataFrame(cols, index=close.index)
    return df.shift(1)


def augment_with_rebound_features(
    features_df: pd.DataFrame,
    raw_returns: pd.Series,
    zscore_window_days: int = 60,
    interval: str = "15m",
    windows_hours: list[int] | None = None,
) -> pd.DataFrame:
    """既存特徴量に drawdown / rebound 系の補助特徴量を追加する.

    raw_returns から close の対数価格 proxy を再構成し、
    decision time t で既知の情報のみを使うため shift(1) したうえで
    recent range 内の位置や直近高値/安値からの距離を計算する。
    """
    if windows_hours is None:
        windows_hours = [24, 72]
    bars_per_hour = max(1, BARS_PER_DAY.get(interval, 96) // 24)
    window_bars = days_to_bars(zscore_window_days, interval)

    raw_returns = raw_returns.reindex(features_df.index).astype(np.float64)
    known_log_price = raw_returns.cumsum().shift(1)

    cols: dict[str, pd.Series] = {}
    for hours in windows_hours:
        w = max(2, hours * bars_per_hour)
        rolling_max = known_log_price.rolling(w, min_periods=w // 3).max()
        rolling_min = known_log_price.rolling(w, min_periods=w // 3).min()
        price_range = (rolling_max - rolling_min).clip(lower=1e-8)
        cols[f"dd_{w}"] = (known_log_price - rolling_max)
        cols[f"rebound_{w}"] = (known_log_price - rolling_min)
        cols[f"range_pos_{w}"] = (known_log_price - rolling_min) / price_range
        cols[f"mom_{w}"] = known_log_price - known_log_price.shift(w)

    derived = pd.DataFrame(cols, index=features_df.index)
    derived_z = rolling_zscore_df(derived, window_bars=window_bars)
    return pd.concat([features_df, derived_z], axis=1).dropna()


def augment_with_context_features(
    features_df: pd.DataFrame,
    raw_returns: pd.Series,
    zscore_window_days: int = 60,
    interval: str = "15m",
    windows_hours: list[int] | None = None,
) -> pd.DataFrame:
    """低頻度の trend / vol / funding context を追加する."""
    if windows_hours is None:
        windows_hours = [4, 24, 72]
    bars_per_hour = max(1, BARS_PER_DAY.get(interval, 96) // 24)
    window_bars = days_to_bars(zscore_window_days, interval)

    raw_returns = raw_returns.reindex(features_df.index).astype(np.float64)
    known_log_price = raw_returns.cumsum().shift(1)
    cols: dict[str, pd.Series] = {}
    for hours in windows_hours:
        w = max(2, hours * bars_per_hour)
        cols[f"ctx_mom_{w}"] = known_log_price - known_log_price.shift(w)
        cols[f"ctx_vol_{w}"] = raw_returns.rolling(w, min_periods=w // 3).std().shift(1)
        cols[f"ctx_downsemi_{w}"] = (
            raw_returns.clip(upper=0.0).pow(2).rolling(w, min_periods=w // 3).mean().pow(0.5).shift(1)
        )
        if "funding_rate" in features_df.columns:
            fr = features_df["funding_rate"].astype(np.float64)
            cols[f"ctx_funding_{w}"] = fr.rolling(w, min_periods=w // 3).mean().shift(1)
        if "oi_change" in features_df.columns:
            oi = features_df["oi_change"].astype(np.float64)
            cols[f"ctx_oi_{w}"] = oi.rolling(w, min_periods=w // 3).mean().shift(1)

    derived = pd.DataFrame(cols, index=features_df.index)
    derived_z = rolling_zscore_df(derived, window_bars=window_bars)
    return pd.concat([features_df, derived_z], axis=1).dropna()


def augment_with_funding_context_features(
    features_df: pd.DataFrame,
    zscore_window_days: int = 60,
    interval: str = "15m",
    windows_hours: list[int] | None = None,
) -> pd.DataFrame:
    """funding_rate から極端値・持続性・反転圧力を作る."""
    if "funding_rate" not in features_df.columns:
        return features_df
    if windows_hours is None:
        windows_hours = [8, 24, 72]
    bars_per_hour = max(1, BARS_PER_DAY.get(interval, 96) // 24)
    window_bars = days_to_bars(zscore_window_days, interval)

    fr = features_df["funding_rate"].astype(np.float64)
    cols: dict[str, pd.Series] = {}
    for hours in windows_hours:
        w = max(2, hours * bars_per_hour)
        mean = fr.rolling(w, min_periods=w // 3).mean()
        std = fr.rolling(w, min_periods=w // 3).std()
        cols[f"fund_mean_{w}"] = mean.shift(1)
        cols[f"fund_std_{w}"] = std.shift(1)
        cols[f"fund_z_{w}"] = ((fr - mean) / (std + 1e-8)).shift(1)
        cols[f"fund_skew_{w}"] = fr.rolling(w, min_periods=w // 3).skew().shift(1)
        cols[f"fund_posratio_{w}"] = (fr.gt(0.0).rolling(w, min_periods=w // 3).mean()).shift(1)
        cols[f"fund_impulse_{w}"] = (fr - mean).abs().shift(1)

    derived = pd.DataFrame(cols, index=features_df.index)
    derived_z = rolling_zscore_df(derived, window_bars=window_bars)
    return pd.concat([features_df, derived_z], axis=1).dropna()


def compute_basis_features(
    spot_close: pd.Series,
    mark_price_df: pd.DataFrame,
) -> pd.DataFrame:
    """spot と futures mark の乖離特徴を作る."""
    if mark_price_df is None or mark_price_df.empty:
        return pd.DataFrame(index=spot_close.index)
    mark_close = mark_price_df["mark_close"].reindex(spot_close.index, method="ffill")
    mark_close = mark_close.bfill().ffill()
    basis = np.log(mark_close / spot_close).shift(1)
    basis_mom = basis.diff().shift(1)
    basis_abs = basis.abs().shift(1)
    return pd.DataFrame(
        {
            "basis": basis,
            "basis_mom": basis_mom,
            "basis_abs": basis_abs,
        },
        index=spot_close.index,
    )


def align_extra_series(
    extra_series: dict[str, pd.Series | pd.DataFrame] | None,
    target_index: pd.DatetimeIndex,
) -> list[pd.Series]:
    if not extra_series:
        return []
    aligned: list[pd.Series] = []
    for name, value in extra_series.items():
        if isinstance(value, pd.DataFrame):
            if value.shape[1] != 1:
                continue
            series = value.iloc[:, 0]
        else:
            series = value
        series = series.rename(name)
        series = series.reindex(target_index, method="ffill")
        series = series.bfill().ffill().shift(1)
        aligned.append(series)
    return aligned


def compute_extra_series_features(
    extra_series: dict[str, pd.Series | pd.DataFrame] | None,
    target_index: pd.DatetimeIndex,
    interval: str = "15m",
    windows_hours: list[int] | None = None,
) -> list[pd.Series | pd.DataFrame]:
    """Align generic external series and derive simple regime/context features.

    Each source contributes:
    - raw shifted level
    - one-step delta
    - absolute impulse
    - rolling mean / std / z-score over a few context windows
    """
    if not extra_series:
        return []
    if windows_hours is None:
        windows_hours = [4, 24, 72]

    bars_per_hour = max(1, BARS_PER_DAY.get(interval, 96) // 24)
    aligned = align_extra_series(extra_series, target_index)
    parts: list[pd.Series | pd.DataFrame] = []
    for series in aligned:
        name = str(series.name)
        base = series.astype(np.float64)
        parts.append(base)
        parts.append(base.diff().rename(f"{name}_delta1"))
        parts.append(base.abs().rename(f"{name}_abs"))
        for hours in windows_hours:
            w = max(2, hours * bars_per_hour)
            mean = base.rolling(w, min_periods=max(2, w // 3)).mean()
            std = base.rolling(w, min_periods=max(2, w // 3)).std()
            parts.append(mean.rename(f"{name}_mean_{w}"))
            parts.append(std.rename(f"{name}_std_{w}"))
            parts.append(((base - mean) / (std + 1e-8)).rename(f"{name}_z_{w}"))
            parts.append((base - mean).abs().rename(f"{name}_impulse_{w}"))
    return parts


# --- Funding Rate / OI 前処理 ---

def align_funding_rate(
    funding_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Funding rate を target 周期に forward fill する（shift(1) 済み）.

    Funding rate は 8h ごとに更新。15m 足に対して最新既知の値を使う。
    """
    if funding_df is None or funding_df.empty:
        return pd.Series(0.0, index=target_index, name="funding_rate")
    fr = funding_df["funding_rate"].reindex(target_index, method="ffill")
    fr = fr.fillna(0.0)
    return fr.shift(1).rename("funding_rate")


def compute_oi_change(
    oi_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """OI 変化率を計算し target 周期にアラインする（shift(1) 済み）.

    OI の対数変化率を使用（大きな値でも安定）。
    """
    if oi_df is None or oi_df.empty:
        return pd.Series(0.0, index=target_index, name="oi_change")
    oi = oi_df["open_interest"].reindex(target_index, method="ffill")
    oi = oi.fillna(method="bfill").fillna(method="ffill")
    oi_change = np.log(oi / oi.shift(1)).fillna(0.0)
    return oi_change.shift(1).rename("oi_change")


# --- メインの特徴量生成 ---

def compute_features(
    df: pd.DataFrame,
    zscore_window_days: int = 60,
    interval: str = "15m",
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    atr_period: int = 14,
    funding_df: "pd.DataFrame | None" = None,
    oi_df: "pd.DataFrame | None" = None,
    mark_price_df: "pd.DataFrame | None" = None,
    extra_series: "dict[str, pd.Series | pd.DataFrame] | None" = None,
    extra_series_windows_hours: list[int] | None = None,
    rv_windows_hours: list[int] | None = None,
) -> pd.DataFrame:
    """OHLCV DataFrame から特徴量を計算して結合する.

    すべての特徴量は shift(1) 済み（未来情報リーク防止）。
    その後 rolling z-score 正規化を適用する。

    Args:
        df: columns=[open, high, low, close, volume]
        zscore_window_days: rolling z-score の窓サイズ（**日数**）
        interval: 足種（バー数変換に使用）
        funding_df: Funding rate DataFrame（省略時はスキップ）
        oi_df: Open Interest DataFrame（省略時はスキップ）
        rv_windows_hours: Realized Vol のウィンドウ（時間単位、デフォルト [1, 4, 24]）

    Returns:
        特徴量 DataFrame（NaN 行は dropna で除去済み）
    """
    # 日数 → バー数変換（単位を明確化）
    window_bars = days_to_bars(zscore_window_days, interval)
    bars_per_hour = BARS_PER_DAY.get(interval, 96) // 24

    # --- 対数リターン（shift(1) 込み）---
    open_ret = log_returns(df["open"]).rename("open_ret")
    high_ret = log_returns(df["high"]).rename("high_ret")
    low_ret = log_returns(df["low"]).rename("low_ret")
    close_ret = log_returns(df["close"]).rename("close_ret")
    vol_ret = log_returns(df["volume"]).rename("vol_ret")

    # --- テクニカル指標（shift(1) 込み）---
    rsi = compute_rsi(df["close"], period=rsi_period)
    macd_df = compute_macd(df["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    atr = compute_atr(df["high"], df["low"], df["close"], period=atr_period)

    # ATR 正規化リターン（close_ret / ATR）
    close_ret_raw = np.log(df["close"] / df["close"].shift(1)).shift(1)
    atr_norm_ret = atr_normalize_returns(close_ret_raw, atr / df["close"].shift(1)).rename("atr_norm_ret")

    parts = [
        open_ret, high_ret, low_ret, close_ret, vol_ret,
        rsi, macd_df["macd"], macd_df["macd_signal"],
        atr_norm_ret,
        atr.rename("atr"),
    ]

    # --- Realized Volatility（マルチスケール、shift(1) 込み）---
    if rv_windows_hours is None:
        rv_windows_hours = [1, 4, 24]
    rv_windows_bars = [max(2, h * bars_per_hour) for h in rv_windows_hours]
    rv_df = compute_realized_vol(df["close"], rv_windows_bars, interval=interval)
    parts.append(rv_df)

    # --- Funding Rate（shift(1) 込み）---
    if funding_df is not None and not funding_df.empty:
        fr = align_funding_rate(funding_df, df.index)
        parts.append(fr)

    # --- OI 変化率（shift(1) 込み）---
    if oi_df is not None and not oi_df.empty:
        oi_chg = compute_oi_change(oi_df, df.index)
        parts.append(oi_chg)

    # --- spot / mark basis --- 
    if mark_price_df is not None and not mark_price_df.empty:
        basis_df = compute_basis_features(df["close"], mark_price_df)
        parts.append(basis_df)

    extra_parts = compute_extra_series_features(
        extra_series,
        df.index,
        interval=interval,
        windows_hours=extra_series_windows_hours,
    )
    if extra_parts:
        parts.extend(extra_parts)

    # --- 結合 ---
    feat = pd.concat(parts, axis=1)

    # --- rolling z-score 正規化（バー数で統一）---
    feat_normalized = rolling_zscore_df(feat, window_bars=window_bars)
    feat_normalized = feat_normalized.dropna()
    return feat_normalized


def get_raw_returns(df: pd.DataFrame) -> pd.Series:
    """close の対数リターン（正規化なし）を返す.

    Oracle / バックテストで使う「バー t で獲得するリターン」。
    returns[t] = log(close_t / close_{t-1})。
    特徴量は shift(1) されているため、features[t] は t-1 までの情報のみ。
    returns[t] は t の実現リターン → ルックアヘッドにならない。
    """
    return np.log(df["close"] / df["close"].shift(1)).dropna()
