from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
import yaml

from unidream.data.features import compute_features, get_raw_returns


def _read_optional_parquet(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df.index, pd.DatetimeIndex) and "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=False)
        df = df.set_index("time")
    return df.sort_index()


def _read_extra_series(cache_dir: str, cache_tag: str) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    pattern = os.path.join(cache_dir, f"{cache_tag}_series_*.parquet")
    for path in sorted(glob.glob(pattern)):
        df = _read_optional_parquet(path)
        if df is None or df.empty or df.shape[1] == 0:
            continue
        name = os.path.basename(path).replace(f"{cache_tag}_series_", "").replace(".parquet", "")
        out[name] = df.iloc[:, 0].rename(name)
    return out


def _coverage_summary(name: str, obj: pd.DataFrame | pd.Series | None, target_index: pd.DatetimeIndex) -> str:
    if obj is None:
        return f"{name}: missing"
    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return f"{name}: empty"
        aligned = obj.reindex(target_index, method="ffill")
        any_valid = aligned.notna().any(axis=1)
        first = obj.index.min()
        last = obj.index.max()
        rows = len(obj)
    else:
        if obj.empty:
            return f"{name}: empty"
        aligned = obj.reindex(target_index, method="ffill")
        any_valid = aligned.notna()
        first = obj.index.min()
        last = obj.index.max()
        rows = len(obj)
    coverage = float(any_valid.mean()) if len(any_valid) else 0.0
    return (
        f"{name}: rows={rows} first={first} last={last} "
        f"aligned_coverage={coverage:.1%}"
    )


def _feature_to_source_hint(feature_name: str) -> str:
    if feature_name.startswith("basis"):
        return "mark cache: <cache_tag>_mark.parquet"
    if feature_name.startswith("fund_") or feature_name == "funding_rate":
        return "funding cache: <cache_tag>_funding.parquet"
    if feature_name.startswith("oi_") or feature_name == "oi_change":
        return "oi cache: <cache_tag>_oi.parquet"

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
    if base == feature_name and feature_name in {
        "signed_order_flow",
        "taker_imbalance",
        "buy_sell_ratio",
        "exchange_netflow",
        "stablecoin_inflow",
        "active_address_growth",
    }:
        base = feature_name
    if base != feature_name or feature_name in {
        "signed_order_flow",
        "taker_imbalance",
        "buy_sell_ratio",
        "exchange_netflow",
        "stablecoin_inflow",
        "active_address_growth",
    }:
        return f"extra series cache: <cache_tag>_series_{base}.parquet"
    return "unknown source"


def _collect_config_feature_subset(config_path: str | None) -> list[str]:
    if not config_path:
        return []
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return ((cfg.get("risk_controller") or {}).get("feature_subset")) or []


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect offline source cache completeness")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--zscore-window-days", type=int, default=60)
    parser.add_argument("--config", help="Optional YAML config with risk_controller.feature_subset")
    args = parser.parse_args()

    ohlcv_path = os.path.join(args.cache_dir, f"{args.cache_tag}_ohlcv.parquet")
    funding_path = os.path.join(args.cache_dir, f"{args.cache_tag}_funding.parquet")
    oi_path = os.path.join(args.cache_dir, f"{args.cache_tag}_oi.parquet")
    mark_path = os.path.join(args.cache_dir, f"{args.cache_tag}_mark.parquet")

    print("[INSPECT] Cache files")
    for path in [ohlcv_path, funding_path, oi_path, mark_path]:
        print(f"  {os.path.basename(path)} -> {'yes' if os.path.exists(path) else 'no'}")

    extra_series = _read_extra_series(args.cache_dir, args.cache_tag)
    print(f"[INSPECT] Extra series -> {list(extra_series.keys())}")

    ohlcv = _read_optional_parquet(ohlcv_path)
    if ohlcv is None:
        print("[INSPECT] No OHLCV cache. Cannot rebuild features.")
        return

    print("[INSPECT] Coverage audit")
    print("  " + _coverage_summary("ohlcv", ohlcv, ohlcv.index))
    print("  " + _coverage_summary("funding", _read_optional_parquet(funding_path), ohlcv.index))
    print("  " + _coverage_summary("oi", _read_optional_parquet(oi_path), ohlcv.index))
    print("  " + _coverage_summary("mark", _read_optional_parquet(mark_path), ohlcv.index))
    for name, series in extra_series.items():
        print("  " + _coverage_summary(name, series, ohlcv.index))

    feature_subset = _collect_config_feature_subset(args.config)
    if feature_subset:
        print("[INSPECT] Config source dependencies")
        for name in feature_subset:
            print(f"  {name} -> {_feature_to_source_hint(name)}")

    try:
        features = compute_features(
            ohlcv,
            zscore_window_days=args.zscore_window_days,
            interval=args.interval,
            funding_df=_read_optional_parquet(funding_path),
            oi_df=_read_optional_parquet(oi_path),
            mark_price_df=_read_optional_parquet(mark_path),
            extra_series=extra_series,
        )
    except Exception as e:
        print(f"[INSPECT] Feature rebuild failed: {e}")
        print("[INSPECT] This usually means the cache is too short or missing required columns.")
        return
    raw_returns = get_raw_returns(ohlcv).reindex(features.index)

    print(f"[INSPECT] Rebuilt features -> shape={features.shape}")
    print(f"[INSPECT] Returns aligned -> {raw_returns.shape[0]} rows")

    if args.config:
        if feature_subset:
            present = [name for name in feature_subset if name in features.columns]
            missing = [name for name in feature_subset if name not in features.columns]
            print(
                "[INSPECT] Config feature_subset -> "
                f"present={len(present)} missing={len(missing)}"
            )
            if present:
                print("  present:")
                for name in present:
                    print(f"    {name}")
            if missing:
                print("  missing:")
                for name in missing:
                    print(f"    {name} -> {_feature_to_source_hint(name)}")
    print("[INSPECT] Feature columns:")
    for col in features.columns:
        print(f"  {col}")


if __name__ == "__main__":
    main()
