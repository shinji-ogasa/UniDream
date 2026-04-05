from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests


_BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    cur = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    out: list[pd.Timestamp] = []
    while cur < end:
        out.append(cur)
        if cur.month == 12:
            cur = pd.Timestamp(year=cur.year + 1, month=1, day=1, tz="UTC")
        else:
            cur = pd.Timestamp(year=cur.year, month=cur.month + 1, day=1, tz="UTC")
    return out


def _download_month(symbol: str, interval: str, month: pd.Timestamp) -> pd.DataFrame:
    month_str = month.strftime("%Y-%m")
    url = f"{_BASE_URL}/{symbol}/{interval}/{symbol}-{interval}-{month_str}.zip"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        inner = zf.namelist()[0]
        with zf.open(inner) as fh:
            df = pd.read_csv(
                fh,
                header=None,
                names=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "n_trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )
    return df


def _read_month_from_zip(zip_path: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        inner = zf.namelist()[0]
        with zf.open(inner) as fh:
            df = pd.read_csv(
                fh,
                header=None,
                names=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "n_trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build free order-flow proxy series from Binance public spot klines")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD, exclusive")
    parser.add_argument("--write-buy-sell-ratio", action="store_true")
    parser.add_argument("--zip-dir", default=None, help="Optional local directory of Binance monthly zip files")
    args = parser.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    os.makedirs(args.cache_dir, exist_ok=True)

    parts: list[pd.DataFrame] = []
    months = _month_starts(start, end)
    for month in months:
        month_str = month.strftime("%Y-%m")
        if args.zip_dir:
            zip_path = Path(args.zip_dir) / f"{args.symbol}-{args.interval}-{month_str}.zip"
            if not zip_path.exists():
                raise FileNotFoundError(f"Missing local zip: {zip_path}")
            print(f"[PUB] Reading local {zip_path.name}...")
            parts.append(_read_month_from_zip(str(zip_path)))
        else:
            print(f"[PUB] Fetching {args.symbol} {args.interval} {month_str}...")
            parts.append(_download_month(args.symbol, args.interval, month))

    if not parts:
        raise RuntimeError("No monthly public kline files fetched")

    df = pd.concat(parts, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].copy()
    for col in ["volume", "taker_buy_base"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    sell_vol = (df["volume"] - df["taker_buy_base"]).clip(lower=0.0)
    signed = df["taker_buy_base"] - sell_vol
    imbalance = np.where(df["volume"] > 0.0, signed / df["volume"], np.nan)
    ratio = np.where(sell_vol > 0.0, df["taker_buy_base"] / sell_vol, np.nan)

    out_map = {
        "signed_order_flow": signed,
        "taker_imbalance": imbalance,
    }
    if args.write_buy_sell_ratio:
        out_map["buy_sell_ratio"] = ratio

    index = pd.DatetimeIndex(df["timestamp"], name="time")
    for name, values in out_map.items():
        out_df = pd.DataFrame({name: values}, index=index).sort_index()
        out_path = os.path.join(args.cache_dir, f"{args.cache_tag}_series_{name}.parquet")
        out_df.to_parquet(out_path)
        print(f"[PUB] Wrote {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
