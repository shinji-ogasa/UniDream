"""Binance OHLCV データ取得モジュール.

公開 REST API を使用（認証不要）。ページネーション対応。
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

BINANCE_BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"
MAX_LIMIT = 1000  # Binance の 1 リクエスト上限


def _parse_timestamp(ts: str | datetime) -> int:
    """日時文字列または datetime を Unix ミリ秒に変換する."""
    if isinstance(ts, datetime):
        return int(ts.timestamp() * 1000)
    return int(datetime.fromisoformat(ts).timestamp() * 1000)


def _klines_to_df(raw: list) -> pd.DataFrame:
    """Binance klines レスポンスを DataFrame に変換する."""
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


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
        resp = requests.get(base_url + KLINES_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
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
