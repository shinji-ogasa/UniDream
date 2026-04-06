from __future__ import annotations

import glob
import os
import random
import time

import numpy as np
import pandas as pd
import torch


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
