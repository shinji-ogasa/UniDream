from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def _load_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df


def _normalize_time_index(df: pd.DataFrame) -> pd.DataFrame:
    time_candidates = ["time", "open_time", "timestamp", "fundingTime"]
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
    else:
        for col in time_candidates:
            if col in df.columns:
                out = df.copy()
                out[col] = pd.to_datetime(out[col], utc=False)
                out = out.set_index(col)
                break
        else:
            raise ValueError("No datetime index/column found")
    return out.sort_index()


def _prepare_mark(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_time_index(df)
    if "mark_close" in df.columns:
        return df[["mark_close"]].astype(float)
    for col in ["close", "markPrice", "mark_price"]:
        if col in df.columns:
            return df[[col]].rename(columns={col: "mark_close"}).astype(float)
    raise ValueError("Mark source must contain one of: mark_close, close, markPrice, mark_price")


def _prepare_funding(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_time_index(df)
    for col in ["funding_rate", "fundingRate"]:
        if col in df.columns:
            return df[[col]].rename(columns={col: "funding_rate"}).astype(float)
    raise ValueError("Funding source must contain one of: funding_rate, fundingRate")


def _prepare_oi(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_time_index(df)
    for col in ["open_interest", "sumOpenInterest", "openInterest"]:
        if col in df.columns:
            return df[[col]].rename(columns={col: "open_interest"}).astype(float)
    raise ValueError("OI source must contain one of: open_interest, sumOpenInterest, openInterest")


def _prepare_spot_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_time_index(df)
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Spot OHLCV is missing columns: {missing}")
    return df[required].astype(float)


def _prepare_generic_series(df: pd.DataFrame, name: str, value_col: str | None = None) -> pd.DataFrame:
    df = _normalize_time_index(df)
    if value_col and value_col in df.columns:
        col = value_col
    else:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) != 1:
            raise ValueError(
                f"Series '{name}' must have exactly one numeric column when value_col is omitted, got {numeric_cols}"
            )
        col = numeric_cols[0]
    return df[[col]].rename(columns={col: name}).astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build auxiliary source cache parquet files")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--mark-file", default=None)
    parser.add_argument("--funding-file", default=None)
    parser.add_argument("--oi-file", default=None)
    parser.add_argument("--spot-file", default=None)
    parser.add_argument(
        "--extra-series",
        action="append",
        default=[],
        help="name=path or name=path:column for arbitrary aligned external series",
    )
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    if args.spot_file:
        spot = _prepare_spot_ohlcv(_load_any(args.spot_file))
        spot_path = os.path.join(args.cache_dir, f"{args.cache_tag}_ohlcv.parquet")
        spot.to_parquet(spot_path)
        print(f"[AUX] Wrote spot ohlcv cache -> {spot_path} ({len(spot)} rows)")

    if args.mark_file:
        mark = _prepare_mark(_load_any(args.mark_file))
        mark_path = os.path.join(args.cache_dir, f"{args.cache_tag}_mark.parquet")
        mark.to_parquet(mark_path)
        print(f"[AUX] Wrote mark cache -> {mark_path} ({len(mark)} rows)")

    if args.funding_file:
        funding = _prepare_funding(_load_any(args.funding_file))
        funding_path = os.path.join(args.cache_dir, f"{args.cache_tag}_funding.parquet")
        funding.to_parquet(funding_path)
        print(f"[AUX] Wrote funding cache -> {funding_path} ({len(funding)} rows)")

    if args.oi_file:
        oi = _prepare_oi(_load_any(args.oi_file))
        oi_path = os.path.join(args.cache_dir, f"{args.cache_tag}_oi.parquet")
        oi.to_parquet(oi_path)
        print(f"[AUX] Wrote oi cache -> {oi_path} ({len(oi)} rows)")

    for spec in args.extra_series:
        if "=" not in spec:
            raise ValueError(f"Invalid --extra-series '{spec}', expected name=path or name=path:column")
        name, rhs = spec.split("=", 1)
        if ":" in rhs:
            path, value_col = rhs.rsplit(":", 1)
        else:
            path, value_col = rhs, None
        series_df = _prepare_generic_series(_load_any(path), name=name, value_col=value_col)
        out_path = os.path.join(args.cache_dir, f"{args.cache_tag}_series_{name}.parquet")
        series_df.to_parquet(out_path)
        print(f"[AUX] Wrote series cache -> {out_path} ({len(series_df)} rows) from {Path(path).name}")


if __name__ == "__main__":
    main()
