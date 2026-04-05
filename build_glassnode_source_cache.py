from __future__ import annotations

import argparse
import os
from urllib.parse import urlencode

import pandas as pd
import requests


def _fetch_glassnode_metric(
    asset: str,
    metric_path: str,
    start: str | None,
    end: str | None,
    interval: str,
    api_key: str,
    base_url: str = "https://api.glassnode.com/v1/metrics",
) -> pd.DataFrame:
    params: dict[str, str | int] = {"a": asset, "i": interval, "api_key": api_key}
    if start:
        params["s"] = int(pd.Timestamp(start, tz="UTC").timestamp())
    if end:
        params["u"] = int(pd.Timestamp(end, tz="UTC").timestamp())
    url = f"{base_url}/{metric_path}?{urlencode(params)}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if not payload:
        return pd.DataFrame(columns=["value"])
    df = pd.DataFrame(payload)
    if "t" not in df.columns or "v" not in df.columns:
        raise ValueError(f"Unexpected Glassnode payload columns: {list(df.columns)}")
    df["time"] = pd.to_datetime(df["t"], unit="s", utc=False)
    df["value"] = pd.to_numeric(df["v"], errors="coerce")
    return df.set_index("time")[["value"]].sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Glassnode metrics into source cache series")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--asset", required=True, help="Glassnode asset id, e.g. BTC")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--interval", default="10m")
    parser.add_argument("--api-key", required=True)
    parser.add_argument(
        "--metric",
        action="append",
        required=True,
        help="alias=metric/path, repeated for each metric",
    )
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    for spec in args.metric:
        if "=" not in spec:
            raise ValueError(f"Invalid --metric '{spec}', expected alias=metric/path")
        alias, metric_path = spec.split("=", 1)
        df = _fetch_glassnode_metric(
            asset=args.asset,
            metric_path=metric_path,
            start=args.start,
            end=args.end,
            interval=args.interval,
            api_key=args.api_key,
        ).rename(columns={"value": alias})
        out_path = os.path.join(args.cache_dir, f"{args.cache_tag}_series_{alias}.parquet")
        df.to_parquet(out_path)
        print(f"[GN] Wrote {out_path} ({len(df)} rows) metric={metric_path}")


if __name__ == "__main__":
    main()
