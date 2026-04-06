"""Binance OHLCV データ取得モジュール.

公開 REST API を使用（認証不要）。ページネーション対応。
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import pandas as pd  # pd.Timestamp を _parse_timestamp で使用
import requests

BINANCE_BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"
MAX_LIMIT = 1000  # Binance の 1 リクエスト上限
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # 秒（指数バックオフの基数）


def _request_with_retry(url: str, params: dict, timeout: int = 30) -> requests.Response:
    """リトライ付き GET リクエスト（指数バックオフ）."""
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = _RETRY_BACKOFF * (2 ** attempt)
            print(f"  [download] Retry {attempt + 1}/{_MAX_RETRIES} after {wait:.1f}s: {e}")
            time.sleep(wait)


def _parse_timestamp(ts: str | datetime) -> int:
    """日時文字列または datetime を Unix ミリ秒に変換する（UTC 基準）.

    naive datetime/文字列は UTC として解釈する。
    ローカルタイムゾーン依存を避けるため pd.Timestamp を使用。
    """
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


def _klines_to_df(raw: list, include_taker_fields: bool = False) -> pd.DataFrame:
    """Binance klines レスポンスを DataFrame に変換する."""
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    if include_taker_fields:
        numeric_cols.extend(["quote_volume", "taker_buy_base", "taker_buy_quote"])
    for col in numeric_cols:
        df[col] = df[col].astype(float)
    keep_cols = ["open", "high", "low", "close", "volume"]
    if include_taker_fields:
        keep_cols.extend(["quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote"])
    return df[keep_cols]


def fetch_binance_ohlcv(
    symbol: str,
    interval: str,
    start: str | datetime,
    end: str | datetime,
    base_url: str = BINANCE_BASE_URL,
    sleep_sec: float = 0.1,
) -> pd.DataFrame:
    """Binance から OHLCV データを取得する.

    Args:
        symbol: ティッカー記号（例: "BTCUSDT"）
        interval: 足種（例: "15m", "1h", "1d"）
        start: 開始日時（ISO 文字列 or datetime）
        end: 終了日時（ISO 文字列 or datetime）
        base_url: Binance API ベース URL
        sleep_sec: リクエスト間のスリープ秒数（レート制限対策）

    Returns:
        open/high/low/close/volume の DataFrame（インデックス: datetime）
    """
    start_ms = _parse_timestamp(start)
    end_ms = _parse_timestamp(end)
    all_rows = []

    current_start = start_ms
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }
        resp = _request_with_retry(base_url + KLINES_ENDPOINT, params)
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)
        last_close_time = data[-1][6]  # close_time
        current_start = last_close_time + 1

        if len(data) < MAX_LIMIT:
            break

        time.sleep(sleep_sec)

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = _klines_to_df(all_rows)
    # end より後のデータを除外
    end_dt = pd.Timestamp(end_ms, unit="ms")
    df = df[df.index < end_dt]
    return df


def fetch_binance_spot_taker_proxy(
    symbol: str,
    interval: str,
    start: str | datetime,
    end: str | datetime,
    base_url: str = BINANCE_BASE_URL,
    sleep_sec: float = 0.1,
) -> pd.DataFrame:
    """Build order-flow proxy series from spot klines.

    Spot klines include taker buy base volume, so this provides a long-history
    free fallback when futures taker-flow endpoints do not expose the period.
    """
    start_ms = _parse_timestamp(start)
    end_ms = _parse_timestamp(end)
    all_rows = []

    current_start = start_ms
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }
        resp = _request_with_retry(base_url + KLINES_ENDPOINT, params)
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)
        last_close_time = data[-1][6]
        current_start = last_close_time + 1

        if len(data) < MAX_LIMIT:
            break

        time.sleep(sleep_sec)

    if not all_rows:
        return pd.DataFrame(columns=["signed_order_flow", "taker_imbalance", "buy_sell_ratio"])

    df = _klines_to_df(all_rows, include_taker_fields=True)
    end_dt = pd.Timestamp(end_ms, unit="ms")
    df = df[df.index < end_dt]

    buy_vol = df["taker_buy_base"].astype(float)
    total = df["volume"].astype(float)
    sell_vol = (total - buy_vol).clip(lower=0.0)
    signed = buy_vol - sell_vol
    out = pd.DataFrame(
        {
            "signed_order_flow": signed,
            "taker_imbalance": signed / (total + 1e-8),
            "buy_sell_ratio": buy_vol / (sell_vol + 1e-8),
        },
        index=df.index,
    )
    return out


BINANCE_FUTURES_URL = "https://fapi.binance.com"


def fetch_mark_price_klines(
    symbol: str,
    interval: str,
    start: str | datetime,
    end: str | datetime,
    base_url: str = BINANCE_FUTURES_URL,
    sleep_sec: float = 0.1,
) -> pd.DataFrame:
    """Binance Futures の mark price kline を取得する.

    Returns:
        DataFrame (index=datetime, columns=['mark_close'])
    """
    start_ms = _parse_timestamp(start)
    end_ms = _parse_timestamp(end)
    all_rows = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }
        resp = _request_with_retry(base_url + "/fapi/v1/markPriceKlines", params)
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        last_close_time = data[-1][6]
        current_start = last_close_time + 1
        if len(data) < MAX_LIMIT:
            break
        time.sleep(sleep_sec)

    if not all_rows:
        return pd.DataFrame(columns=["mark_close"])

    df = _klines_to_df(all_rows)
    end_dt = pd.Timestamp(end_ms, unit="ms")
    df = df[df.index < end_dt]
    return df[["close"]].rename(columns={"close": "mark_close"})


def fetch_funding_rate(
    symbol: str,
    start: str | datetime,
    end: str | datetime,
    base_url: str = BINANCE_FUTURES_URL,
    sleep_sec: float = 0.1,
) -> pd.DataFrame:
    """Binance Futures から過去の funding rate を取得する.

    8 時間ごとのデータを返す。15m 足への forward fill は呼び出し側で行う。

    Returns:
        DataFrame (index=datetime, columns=['funding_rate'])
    """
    start_ms = _parse_timestamp(start)
    end_ms = _parse_timestamp(end)
    all_rows: list[dict] = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }
        resp = _request_with_retry(base_url + "/fapi/v1/fundingRate", params)
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        current_start = data[-1]["fundingTime"] + 1
        if len(data) < MAX_LIMIT:
            break
        time.sleep(sleep_sec)

    if not all_rows:
        return pd.DataFrame(columns=["funding_rate"])

    df = pd.DataFrame(all_rows)
    df["time"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df.set_index("time")[["funding_rate"]]
    df = df[df.index < pd.Timestamp(end_ms, unit="ms")]
    return df


def fetch_open_interest_hist(
    symbol: str,
    period: str,
    start: str | datetime,
    end: str | datetime,
    base_url: str = BINANCE_FUTURES_URL,
    sleep_sec: float = 0.5,
) -> pd.DataFrame:
    """Binance Futures から OI 履歴を取得する.

    Args:
        period: 集計間隔 ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d")

    Returns:
        DataFrame (index=datetime, columns=['open_interest'])
    """
    start_ms = _parse_timestamp(start)
    end_ms = _parse_timestamp(end)
    all_rows: list[dict] = []
    current_start = start_ms
    limit = 500  # this endpoint max

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }
        resp = _request_with_retry(base_url + "/futures/data/openInterestHist", params)
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        current_start = data[-1]["timestamp"] + 1
        if len(data) < limit:
            break
        time.sleep(sleep_sec)

    if not all_rows:
        return pd.DataFrame(columns=["open_interest"])

    df = pd.DataFrame(all_rows)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["open_interest"] = df["sumOpenInterest"].astype(float)
    df = df.set_index("time")[["open_interest"]]
    df = df[df.index < pd.Timestamp(end_ms, unit="ms")]
    return df


def _taker_long_short_to_df(raw: list[dict]) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["signed_order_flow", "taker_imbalance", "buy_sell_ratio"])
    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    buy_vol = df["buyVol"].astype(float)
    sell_vol = df["sellVol"].astype(float)
    total = buy_vol + sell_vol
    out = pd.DataFrame(
        {
            "signed_order_flow": (buy_vol - sell_vol).to_numpy(),
            "taker_imbalance": (((buy_vol - sell_vol) / (total + 1e-8))).to_numpy(),
            "buy_sell_ratio": df["buySellRatio"].astype(float).to_numpy(),
        },
        index=df["time"],
    )
    return out.sort_index()


def fetch_taker_buy_sell_volume(
    symbol: str,
    period: str,
    start: str | datetime,
    end: str | datetime,
    base_url: str = BINANCE_FUTURES_URL,
    sleep_sec: float = 0.5,
) -> pd.DataFrame:
    """Fetch taker buy/sell flow proxy from Binance Futures.

    Endpoint:
        /futures/data/takerlongshortRatio

    Returns:
        DataFrame (index=datetime, columns=['signed_order_flow', 'taker_imbalance', 'buy_sell_ratio'])
    """
    start_ms = _parse_timestamp(start)
    end_ms = _parse_timestamp(end)
    all_rows: list[dict] = []
    current_start = start_ms
    limit = 500

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }
        resp = _request_with_retry(base_url + "/futures/data/takerlongshortRatio", params)
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        current_start = int(data[-1]["timestamp"]) + 1
        if len(data) < limit:
            break
        time.sleep(sleep_sec)

    df = _taker_long_short_to_df(all_rows)
    if df.empty:
        return df
    return df[df.index < pd.Timestamp(end_ms, unit="ms")]


def fetch_multi_symbol(
    symbols: list[str],
    interval: str,
    start: str | datetime,
    end: str | datetime,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """複数シンボルの OHLCV を取得する.

    Returns:
        {symbol: DataFrame} の辞書
    """
    result = {}
    for sym in symbols:
        result[sym] = fetch_binance_ohlcv(sym, interval, start, end, **kwargs)
    return result
