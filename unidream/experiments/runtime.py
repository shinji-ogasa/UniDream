from __future__ import annotations

import glob
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml

from unidream.data.download import (
    fetch_binance_ohlcv,
    fetch_funding_rate,
    fetch_mark_price_klines,
    fetch_open_interest_hist,
)
from unidream.data.features import align_extra_series, compute_features, get_raw_returns


_CACHE_STALE_DAYS = 7


def cache_is_fresh(path: str, stale_days: int = _CACHE_STALE_DAYS) -> bool:
    if not os.path.exists(path):
        return False
    return (time.time() - os.path.getmtime(path)) / 86400 < stale_days


def resolve_cache_pair(cache_dir: str, cache_tag: str) -> tuple[str, str]:
    features_cache = os.path.join(cache_dir, f"{cache_tag}_features.parquet")
    returns_cache = os.path.join(cache_dir, f"{cache_tag}_returns.parquet")
    if os.path.exists(features_cache) and os.path.exists(returns_cache):
        return features_cache, returns_cache

    feature_candidates = sorted(glob.glob(os.path.join(cache_dir, f"{cache_tag}*_features.parquet")))
    return_candidates = sorted(glob.glob(os.path.join(cache_dir, f"{cache_tag}*_returns.parquet")))
    if feature_candidates and return_candidates:
        return feature_candidates[0], return_candidates[0]
    return features_cache, returns_cache


def resolve_optional_cache(cache_dir: str, cache_tag: str, suffix: str) -> str:
    path = os.path.join(cache_dir, f"{cache_tag}_{suffix}.parquet")
    if os.path.exists(path):
        return path
    candidates = sorted(glob.glob(os.path.join(cache_dir, f"{cache_tag}*_{suffix}.parquet")))
    if candidates:
        return candidates[0]
    return path


def read_optional_parquet(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=False)
            df = df.set_index("time")
        elif "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
            df = df.drop(columns=["timestamp"]).set_index(ts.rename("time"))
    return df.sort_index()


def read_extra_series_caches(cache_dir: str, cache_tag: str) -> dict[str, pd.Series]:
    series_map: dict[str, pd.Series] = {}
    pattern = os.path.join(cache_dir, f"{cache_tag}*_series_*.parquet")
    for path in sorted(glob.glob(pattern)):
        df = read_optional_parquet(path)
        if df is None or df.empty or df.shape[1] == 0:
            continue
        name = os.path.basename(path).split("_series_", 1)[-1].replace(".parquet", "")
        series_map[name] = df.iloc[:, 0].rename(name)
    return series_map


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_costs(cfg: dict, cost_profile: str | None = None) -> tuple[dict, str]:
    resolved_cfg = dict(cfg)
    profile_name = cost_profile or cfg.get("cost_profile") or "default"
    profiles = cfg.get("cost_profiles")

    if profiles:
        if profile_name == "default":
            profile_name = "base" if "base" in profiles else next(iter(profiles))
        if profile_name not in profiles:
            available = ", ".join(profiles.keys())
            raise KeyError(f"Unknown cost profile '{profile_name}'. Available: {available}")
        resolved_cfg["costs"] = dict(profiles[profile_name])
        resolved_cfg["cost_profile"] = profile_name
    else:
        resolved_cfg["costs"] = dict(cfg.get("costs", {}))
        resolved_cfg["cost_profile"] = profile_name

    return resolved_cfg, resolved_cfg["cost_profile"]


def load_training_features(
    *,
    symbol: str,
    interval: str,
    start: str,
    end: str,
    zscore_window: int,
    cache_dir: str,
    cache_tag: str,
    extra_series_mode: str = "derived",
    extra_series_include: list[str] | None = None,
    include_funding: bool = True,
    include_oi: bool = True,
    include_mark: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    features_cache, returns_cache = resolve_cache_pair(cache_dir, cache_tag)
    ohlcv_cache = os.path.join(cache_dir, f"{cache_tag}_ohlcv.parquet")
    funding_cache = os.path.join(cache_dir, f"{cache_tag}_funding.parquet")
    oi_cache = os.path.join(cache_dir, f"{cache_tag}_oi.parquet")
    mark_cache = os.path.join(cache_dir, f"{cache_tag}_mark.parquet")

    if os.path.exists(features_cache) and os.path.exists(returns_cache):
        print("\n[Data] Loading cached features...")
        features_df = pd.read_parquet(features_cache)
        raw_returns = pd.read_parquet(returns_cache).squeeze()
        print(f"  Cached: {features_df.shape} | obs_dim={features_df.shape[1]}")
        return features_df, raw_returns

    df = read_optional_parquet(ohlcv_cache)
    if df is not None:
        print(f"\n[Data] Spot OHLCV cache loaded: {len(df)} bars")
    else:
        print("\n[Data] Fetching OHLCV...")
        df = fetch_binance_ohlcv(symbol, interval, start, end)
        print(f"  Raw data: {len(df)} bars ({df.index[0]} -> {df.index[-1]})")

    funding_df = read_optional_parquet(funding_cache)
    oi_df = read_optional_parquet(oi_cache)
    mark_price_df = read_optional_parquet(mark_cache)
    extra_series = read_extra_series_caches(cache_dir, cache_tag)
    if not include_funding:
        funding_df = None
    elif funding_df is not None:
        print(f"[Data] Funding cache loaded: {len(funding_df)} records")
    else:
        try:
            print("[Data] Fetching funding rate...")
            funding_df = fetch_funding_rate(symbol, start, end)
            print(f"  Funding rate: {len(funding_df)} records")
        except Exception as exc:
            print(f"  Funding rate skipped: {exc}")
    if not include_oi:
        oi_df = None
    elif oi_df is not None:
        print(f"[Data] OI cache loaded: {len(oi_df)} records")
    else:
        try:
            print("[Data] Fetching open interest...")
            oi_df = fetch_open_interest_hist(symbol, interval, start, end)
            print(f"  Open interest: {len(oi_df)} records")
        except Exception as exc:
            print(f"  Open interest skipped: {exc}")
    if not include_mark:
        mark_price_df = None
    elif mark_price_df is not None:
        print(f"[Data] Mark cache loaded: {len(mark_price_df)} records")
    else:
        try:
            print("[Data] Fetching futures mark price...")
            mark_price_df = fetch_mark_price_klines(symbol, interval, start, end)
            print(f"  Mark price: {len(mark_price_df)} records")
        except Exception as exc:
            print(f"  Mark price skipped: {exc}")
    if extra_series_include:
        include_set = set(extra_series_include)
        extra_series = {k: v for k, v in extra_series.items() if k in include_set}

    print("[Data] Computing features...")
    if extra_series_mode == "raw_only":
        features_df = compute_features(
            df,
            zscore_window_days=zscore_window,
            interval=interval,
            funding_df=funding_df,
            oi_df=oi_df,
            mark_price_df=mark_price_df,
            extra_series=None,
        )
        extra_parts = align_extra_series(extra_series, df.index)
        if extra_parts:
            features_df = pd.concat([features_df, *extra_parts], axis=1).dropna()
    else:
        features_df = compute_features(
            df,
            zscore_window_days=zscore_window,
            interval=interval,
            funding_df=funding_df,
            oi_df=oi_df,
            mark_price_df=mark_price_df,
            extra_series=extra_series,
        )
    raw_returns = get_raw_returns(df)
    common_idx = features_df.index.intersection(raw_returns.index)
    features_df = features_df.loc[common_idx]
    raw_returns = raw_returns.loc[common_idx]
    os.makedirs(cache_dir, exist_ok=True)
    features_df.to_parquet(features_cache)
    raw_returns.to_frame(name="returns").to_parquet(returns_cache)
    print(f"  Features: {features_df.shape} | obs_dim={features_df.shape[1]}")
    print(f"  Saved cache: {features_cache}")
    return features_df, raw_returns
