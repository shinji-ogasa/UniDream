from __future__ import annotations

import json
import os
import random
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import yaml

from unidream.data.cache_v4 import CacheV4Error, cache_v4_paths, load_cache_v4
from unidream.data.download import (
    fetch_binance_ohlcv,
    fetch_funding_rate,
    fetch_mark_price_klines,
    fetch_open_interest_hist,
)
from unidream.data.features import align_extra_series, compute_features, get_raw_returns
from unidream.experiments.checkpointing import atomic_text_write


CACHE_CONTRACT_VERSION = 1
_BASE_FEATURE_COLUMNS = {
    "open_ret",
    "high_ret",
    "low_ret",
    "close_ret",
    "vol_ret",
    "RSI_14",
    "macd",
    "macd_signal",
    "atr_norm_ret",
    "atr",
    "rv_4",
    "rv_16",
    "rv_96",
}


def resolve_cache_pair(cache_dir: str, cache_tag: str) -> tuple[str, str]:
    features_cache = os.path.join(cache_dir, f"{cache_tag}_features.parquet")
    returns_cache = os.path.join(cache_dir, f"{cache_tag}_returns.parquet")
    return features_cache, returns_cache


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
    prefix = f"{cache_tag}_series_"
    if not os.path.isdir(cache_dir):
        return series_map
    for filename in sorted(os.listdir(cache_dir)):
        if not filename.startswith(prefix) or not filename.endswith(".parquet"):
            continue
        path = os.path.join(cache_dir, filename)
        df = read_optional_parquet(path)
        if df is None or df.empty or df.shape[1] == 0:
            continue
        name = filename[len(prefix) : -len(".parquet")]
        series_map[name] = df.iloc[:, 0].rename(name)
    return series_map


def _cache_metadata_path(cache_dir: str, cache_tag: str) -> str:
    return os.path.join(cache_dir, f"{cache_tag}_metadata.json")


def cache_quality_status(cache_dir: str, cache_tag: str) -> str:
    """Return an explicit status for a legacy or schema v4 cache hit.

    Historical v3 files intentionally remain readable for compatibility, but
    they are never reported as quality-passed because they have no
    availability sidecar.  A partial or invalid v4 set is also surfaced as a
    failure instead of triggering a raw-data rebuild that could hide the
    broken artifact.
    """
    paths = cache_v4_paths(cache_dir, cache_tag)
    metadata_path = paths["metadata"]
    availability_path = paths["availability"]
    metadata: dict | None = None
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                metadata = payload
        except (OSError, ValueError, json.JSONDecodeError):
            return "v4_invalid" if availability_path.exists() else "legacy_v3_unverified"
    if not availability_path.exists() and (metadata is None or metadata.get("schema_version") != 4):
        return "legacy_v3_unverified"
    if not all(path.exists() for path in paths.values()):
        return "v4_incomplete"
    try:
        load_cache_v4(cache_dir, cache_tag)
    except CacheV4Error:
        return "v4_invalid"
    return "v4_verified"


def _cache_parameters(
    *,
    symbol: str,
    interval: str,
    start: str,
    end: str,
    zscore_window: int,
    extra_series_mode: str,
    extra_series_include: list[str] | None,
    include_funding: bool,
    include_oi: bool,
    include_mark: bool,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "interval": interval,
        "start": start,
        "end": end,
        "zscore_window_days": int(zscore_window),
        "extra_series_mode": str(extra_series_mode),
        "extra_series_include": sorted(str(name) for name in (extra_series_include or [])),
        "include_funding": bool(include_funding),
        "include_oi": bool(include_oi),
        "include_mark": bool(include_mark),
    }


def _validate_training_cache(
    features_df: pd.DataFrame,
    raw_returns: pd.Series,
    *,
    include_funding: bool,
    include_oi: bool,
    include_mark: bool,
    cache_tag: str,
) -> None:
    if features_df.empty or raw_returns.empty:
        raise ValueError(f"cache {cache_tag} is empty")
    if not isinstance(features_df.index, pd.DatetimeIndex):
        raise ValueError(f"cache {cache_tag} features index is not DatetimeIndex")
    if not features_df.index.is_monotonic_increasing or not features_df.index.is_unique:
        raise ValueError(f"cache {cache_tag} features index is not sorted and unique")
    if not isinstance(raw_returns.index, pd.DatetimeIndex):
        raise ValueError(f"cache {cache_tag} returns index is not DatetimeIndex")
    if not features_df.index.equals(raw_returns.index):
        raise ValueError(f"cache {cache_tag} features/returns indices differ")
    columns = set(str(column) for column in features_df.columns)
    required = set(_BASE_FEATURE_COLUMNS)
    if include_funding:
        required.add("funding_rate")
    if include_oi:
        required.add("oi_change")
    if include_mark:
        required.update({"basis", "basis_mom", "basis_abs"})
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"cache {cache_tag} is missing required feature columns: {missing}")
    try:
        feature_values = features_df.to_numpy(dtype=np.float64)
        return_values = raw_returns.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cache {cache_tag} contains non-numeric values") from exc
    if not np.isfinite(feature_values).all() or not np.isfinite(return_values).all():
        raise ValueError(f"cache {cache_tag} contains NaN or infinite values")


def _write_cache_metadata(
    *,
    cache_dir: str,
    cache_tag: str,
    parameters: dict[str, object],
    features_df: pd.DataFrame,
    provenance: str,
) -> None:
    metadata = {
        "schema_version": CACHE_CONTRACT_VERSION,
        "cache_tag": cache_tag,
        "parameters": parameters,
        "feature_columns": [str(column) for column in features_df.columns],
        "rows": int(len(features_df)),
        "first_timestamp": str(features_df.index[0]),
        "last_timestamp": str(features_df.index[-1]),
        "provenance": provenance,
    }
    atomic_text_write(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        _cache_metadata_path(cache_dir, cache_tag),
    )


def _atomic_parquet_write(frame: pd.DataFrame, path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid4().hex}")
    try:
        frame.to_parquet(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


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
    require_v4_cache: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    features_cache, returns_cache = resolve_cache_pair(cache_dir, cache_tag)
    v4_paths = cache_v4_paths(cache_dir, cache_tag)
    metadata_path = _cache_metadata_path(cache_dir, cache_tag)
    v4_sidecar_exists = v4_paths["availability"].exists()
    metadata_version: int | None = None
    if os.path.exists(metadata_path):
        try:
            metadata_payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            if isinstance(metadata_payload, dict):
                metadata_version = metadata_payload.get("schema_version")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if require_v4_cache or v4_sidecar_exists:
                raise ValueError(f"cache metadata cannot be read for v4 validation: {exc}") from exc
    v4_declared = metadata_version == 4
    if (require_v4_cache or v4_declared or v4_sidecar_exists) and not all(
        path.exists() for path in v4_paths.values()
    ):
        missing = [str(path) for path in v4_paths.values() if not path.exists()]
        reason = "required" if require_v4_cache else "declared or partially present"
        raise ValueError(f"v4 cache is {reason} but incomplete; missing files: {missing}")
    parameters = _cache_parameters(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        zscore_window=zscore_window,
        extra_series_mode=extra_series_mode,
        extra_series_include=extra_series_include,
        include_funding=include_funding,
        include_oi=include_oi,
        include_mark=include_mark,
    )
    ohlcv_cache = os.path.join(cache_dir, f"{cache_tag}_ohlcv.parquet")
    funding_cache = os.path.join(cache_dir, f"{cache_tag}_funding.parquet")
    oi_cache = os.path.join(cache_dir, f"{cache_tag}_oi.parquet")
    mark_cache = os.path.join(cache_dir, f"{cache_tag}_mark.parquet")

    if os.path.exists(features_cache) and os.path.exists(returns_cache):
        print("\n[Data] Loading cached features...")
        if v4_sidecar_exists or v4_declared or require_v4_cache:
            try:
                features_df, raw_returns, _availability, v4_metadata = load_cache_v4(
                    cache_dir,
                    cache_tag,
                )
                _validate_training_cache(
                    features_df,
                    raw_returns,
                    include_funding=include_funding,
                    include_oi=include_oi,
                    include_mark=include_mark,
                    cache_tag=cache_tag,
                )
                cached_parameters = v4_metadata.get("parameters")
                if cached_parameters is not None and cached_parameters != parameters:
                    raise ValueError("cache v4 parameters do not match the requested config")
                if include_funding or include_mark:
                    raise ValueError(
                        "full17 v4 training promotion blocked: "
                        "load_training_features cannot propagate availability sidecar into "
                        "SequenceDataset/WFODataset; use an availability-aware training path"
                    )
            except (CacheV4Error, ValueError) as exc:
                raise ValueError(f"cache {cache_tag} failed v4 validation: {exc}") from exc
            print(
                f"  Cached: {features_df.shape} | obs_dim={features_df.shape[1]} "
                "| quality_status=v4_verified"
            )
            return features_df, raw_returns
        try:
            features_df = read_optional_parquet(features_cache)
            returns_frame = read_optional_parquet(returns_cache)
            if features_df is None or returns_frame is None:
                raise ValueError("cache pair disappeared while loading")
            raw_returns = returns_frame.squeeze("columns")
            if isinstance(raw_returns, pd.DataFrame):
                raise ValueError("returns must contain exactly one column")
            _validate_training_cache(
                features_df,
                raw_returns,
                include_funding=include_funding,
                include_oi=include_oi,
                include_mark=include_mark,
                cache_tag=cache_tag,
            )
            if os.path.exists(metadata_path):
                metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
                if not isinstance(metadata, dict):
                    raise ValueError("cache metadata must be a mapping")
                if metadata.get("schema_version") != CACHE_CONTRACT_VERSION:
                    raise ValueError(f"unsupported metadata schema: {metadata.get('schema_version')!r}")
                if metadata.get("cache_tag") != cache_tag:
                    raise ValueError("cache tag does not match metadata")
                if metadata.get("parameters") != parameters:
                    raise ValueError("cache parameters do not match the requested config")
                if metadata.get("feature_columns") != [str(column) for column in features_df.columns]:
                    raise ValueError("cache feature columns do not match metadata")
                if metadata.get("rows") != len(features_df):
                    raise ValueError("cache row count does not match metadata")
                if metadata.get("first_timestamp") != str(features_df.index[0]):
                    raise ValueError("cache first timestamp does not match metadata")
                if metadata.get("last_timestamp") != str(features_df.index[-1]):
                    raise ValueError("cache last timestamp does not match metadata")
            else:
                _write_cache_metadata(
                    cache_dir=cache_dir,
                    cache_tag=cache_tag,
                    parameters=parameters,
                    features_df=features_df,
                    provenance="legacy_unverified",
                )
            print(
                f"  Cached: {features_df.shape} | obs_dim={features_df.shape[1]} "
                "| quality_status=legacy_v3_unverified"
            )
            return features_df, raw_returns
        except Exception as exc:
            print(f"  Cache invalid; rebuilding from raw data: {exc}")

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
            raise RuntimeError(
                "funding rate is required by this training config but could not be fetched"
            ) from exc
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
            raise RuntimeError(
                "open interest is required by this training config but could not be fetched"
            ) from exc
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
            raise RuntimeError(
                "mark price is required by this training config but could not be fetched"
            ) from exc
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
    _validate_training_cache(
        features_df,
        raw_returns,
        include_funding=include_funding,
        include_oi=include_oi,
        include_mark=include_mark,
        cache_tag=cache_tag,
    )
    os.makedirs(cache_dir, exist_ok=True)
    _atomic_parquet_write(features_df, features_cache)
    _atomic_parquet_write(raw_returns.to_frame(name="returns"), returns_cache)
    _write_cache_metadata(
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        parameters=parameters,
        features_df=features_df,
        provenance="generated",
    )
    print(f"  Features: {features_df.shape} | obs_dim={features_df.shape[1]}")
    print(f"  Saved cache: {features_cache}")
    return features_df, raw_returns
