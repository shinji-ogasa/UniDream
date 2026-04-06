from __future__ import annotations

import glob
import os

import yaml


def feature_to_requirement(feature_name: str) -> tuple[str, str]:
    if feature_name.startswith("basis"):
        return ("file", "mark")
    if feature_name.startswith("fund_") or feature_name == "funding_rate":
        return ("file", "funding")
    if feature_name.startswith("oi_") or feature_name == "oi_change":
        return ("file", "oi")

    suffixes = (
        "_delta1",
        "_abs",
        "_mean_16", "_mean_96", "_mean_288",
        "_std_16", "_std_96", "_std_288",
        "_z_16", "_z_96", "_z_288",
        "_impulse_16", "_impulse_96", "_impulse_288",
    )
    base = feature_name
    for suffix in suffixes:
        if feature_name.endswith(suffix):
            base = feature_name[: -len(suffix)]
            break
    if base in {
        "signed_order_flow",
        "taker_imbalance",
        "buy_sell_ratio",
        "exchange_netflow",
        "stablecoin_inflow",
        "active_address_growth",
    }:
        return ("series", base)
    return ("derived", feature_name)


def collect_missing_requirements(cache_dir: str, cache_tag: str, config_path: str) -> list[str]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    feature_subset = ((cfg.get("risk_controller") or {}).get("feature_subset")) or []

    available_files = {
        "ohlcv": os.path.exists(os.path.join(cache_dir, f"{cache_tag}_ohlcv.parquet")),
        "mark": os.path.exists(os.path.join(cache_dir, f"{cache_tag}_mark.parquet")),
        "funding": os.path.exists(os.path.join(cache_dir, f"{cache_tag}_funding.parquet")),
        "oi": os.path.exists(os.path.join(cache_dir, f"{cache_tag}_oi.parquet")),
    }
    available_series = {
        os.path.basename(path).replace(f"{cache_tag}_series_", "").replace(".parquet", "")
        for path in glob.glob(os.path.join(cache_dir, f"{cache_tag}_series_*.parquet"))
    }

    missing: list[str] = []
    for feature_name in feature_subset:
        kind, key = feature_to_requirement(feature_name)
        if kind == "file":
            if not available_files.get(key, False):
                missing.append(f"{feature_name} -> missing {cache_tag}_{key}.parquet")
        elif kind == "series":
            if key not in available_series:
                missing.append(f"{feature_name} -> missing {cache_tag}_series_{key}.parquet")
    return missing
