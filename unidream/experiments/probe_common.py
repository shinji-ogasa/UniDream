from __future__ import annotations

import os

import numpy as np
import pandas as pd

from unidream.data.download import fetch_binance_ohlcv
from unidream.data.features import (
    augment_with_context_features,
    augment_with_funding_context_features,
    augment_with_rebound_features,
    compute_features,
    get_raw_returns,
)
from unidream.experiments.runtime import (
    cache_is_fresh,
    read_extra_series_caches,
    read_optional_parquet,
    resolve_cache_pair,
    resolve_costs,
    resolve_optional_cache,
    set_seed,
)


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
