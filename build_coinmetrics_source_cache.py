from __future__ import annotations

import argparse
import os
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests


def _fetch_coinmetrics_metric(
    asset: str,
    metric: str,
    start: str,
    end: str,
    frequency: str,
    api_key: str | None = None,
    base_url: str = "https://community-api.coinmetrics.io/v4",
) -> pd.DataFrame:
    params = {
        "assets": asset,
        "metrics": metric,
        "start_time": start,
        "end_time": end,
        "frequency": frequency,
    }
    if api_key:
        params["api_key"] = api_key
    url = f"{base_url}/timeseries/asset-metrics?{urlencode(params)}"
    rows: list[dict] = []

    while url:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        rows.extend(payload.get("data", []))
        url = payload.get("next_page_url")

    if not rows:
        return pd.DataFrame(columns=[metric])

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=False)
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.set_index("time")[[metric]].sort_index()
    return df


def _apply_transform(df: pd.DataFrame, col: str, transform: str | None) -> pd.DataFrame:
    if not transform or transform == "none":
        return df
    series = df[col].astype(float)
    if transform == "diff":
        out = series.diff()
    elif transform == "pct_change":
        out = series.pct_change()
    elif transform == "logdiff":
        out = np.log(series.clip(lower=1e-8)).diff()
    else:
        raise ValueError(f"Unsupported transform: {transform}")
    return out.to_frame(col)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Coin Metrics asset metrics into source cache series")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--asset", required=True, help="Coin Metrics asset id, e.g. btc")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--frequency", default="1h")
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--metric",
        action="append",
        required=True,
        help="alias=MetricName or alias=MetricName:transform, repeated for each metric",
    )
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    for spec in args.metric:
        if "=" not in spec:
            raise ValueError(f"Invalid --metric '{spec}', expected alias=MetricName[:transform]")
        alias, rhs = spec.split("=", 1)
        if ":" in rhs:
            metric, transform = rhs.rsplit(":", 1)
        else:
            metric, transform = rhs, None
        df = _fetch_coinmetrics_metric(
            asset=args.asset,
            metric=metric,
            start=args.start,
            end=args.end,
            frequency=args.frequency,
            api_key=args.api_key,
        ).rename(columns={metric: alias})
        df = _apply_transform(df, alias, transform)
        out_path = os.path.join(args.cache_dir, f"{args.cache_tag}_series_{alias}.parquet")
        df.to_parquet(out_path)
        suffix = f":{transform}" if transform else ""
        print(f"[CM] Wrote {out_path} ({len(df)} rows) metric={metric}{suffix}")


if __name__ == "__main__":
    main()
