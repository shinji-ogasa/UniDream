from __future__ import annotations

import glob
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from unidream.data.download import fetch_binance_ohlcv
from unidream.data.features import (
    augment_with_context_features,
    augment_with_funding_context_features,
    augment_with_rebound_features,
    compute_features,
    get_raw_returns,
)


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
    if not isinstance(df.index, pd.DatetimeIndex) and "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=False)
        df = df.set_index("time")
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


def load_probe_features(
    *,
    symbol: str,
    interval: str,
    start: str,
    end: str,
    zscore_window: int,
    data_cache_dir: str,
) -> tuple[pd.DataFrame, pd.Series, str]:
    cache_tag = f"{symbol}_{interval}_{start}_{end}_z{zscore_window}"
    features_cache, returns_cache = resolve_cache_pair(data_cache_dir, cache_tag)
    ohlcv_cache = resolve_optional_cache(data_cache_dir, cache_tag, "ohlcv")
    funding_cache = resolve_optional_cache(data_cache_dir, cache_tag, "funding")
    oi_cache = resolve_optional_cache(data_cache_dir, cache_tag, "oi")
    mark_cache = resolve_optional_cache(data_cache_dir, cache_tag, "mark")

    if cache_is_fresh(features_cache) and cache_is_fresh(returns_cache):
        print("[Data] Loading cached features...")
        features_df = pd.read_parquet(features_cache)
        raw_returns = pd.read_parquet(returns_cache).squeeze()
        return features_df, raw_returns, cache_tag

    ohlcv = read_optional_parquet(ohlcv_cache)
    if ohlcv is None:
        print("[Data] Computing features from OHLCV...")
        ohlcv = fetch_binance_ohlcv(symbol=symbol, interval=interval, start=start, end=end)
    else:
        print("[Data] Rebuilding features from cached OHLCV...")

    funding_df = read_optional_parquet(funding_cache)
    oi_df = read_optional_parquet(oi_cache)
    mark_price_df = read_optional_parquet(mark_cache)
    extra_series = read_extra_series_caches(data_cache_dir, cache_tag)
    features_df = compute_features(
        ohlcv,
        zscore_window_days=zscore_window,
        interval=interval,
        funding_df=funding_df,
        oi_df=oi_df,
        mark_price_df=mark_price_df,
        extra_series=extra_series,
    )
    raw_returns = get_raw_returns(ohlcv)
    return features_df, raw_returns, cache_tag


def apply_feature_extras(
    features_df: pd.DataFrame,
    raw_returns: pd.Series,
    *,
    feature_extras_cfg: dict,
    zscore_window: int,
    interval: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if feature_extras_cfg.get("rebound_v1", False):
        features_df = augment_with_rebound_features(
            features_df,
            raw_returns,
            zscore_window_days=zscore_window,
            interval=interval,
            windows_hours=feature_extras_cfg.get("rebound_windows_hours", [24, 72]),
        )
        print(f"[Data] Rebound features enabled -> obs_dim={features_df.shape[1]}")
    if feature_extras_cfg.get("context_v1", False):
        features_df = augment_with_context_features(
            features_df,
            raw_returns,
            zscore_window_days=zscore_window,
            interval=interval,
            windows_hours=feature_extras_cfg.get("context_windows_hours", [4, 24, 72]),
        )
        print(f"[Data] Context features enabled -> obs_dim={features_df.shape[1]}")
    if feature_extras_cfg.get("funding_context_v1", False):
        features_df = augment_with_funding_context_features(
            features_df,
            zscore_window_days=zscore_window,
            interval=interval,
            windows_hours=feature_extras_cfg.get("funding_context_windows_hours", [8, 24, 72]),
        )
        print(f"[Data] Funding context features enabled -> obs_dim={features_df.shape[1]}")
    raw_returns = raw_returns.reindex(features_df.index)
    return features_df, raw_returns
