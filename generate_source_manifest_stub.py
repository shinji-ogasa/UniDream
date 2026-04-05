from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _required_sources(feature_name: str) -> tuple[str | None, str | None]:
    if feature_name.startswith("basis"):
        return "mark_file", None
    if feature_name.startswith("fund_") or feature_name == "funding_rate":
        return "funding_file", None
    if feature_name.startswith("oi_") or feature_name == "oi_change":
        return "oi_file", None

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
        return None, base
    return None, None


def _load_feature_subset(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return ((cfg.get("risk_controller") or {}).get("feature_subset")) or []


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a source manifest stub from config feature requirements")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--config", action="append", required=True, help="Repeat for each config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    market_files: set[str] = {"spot_file"}
    extra_series: set[str] = set()
    config_names: list[str] = []

    for config_path in args.config:
        config_names.append(Path(config_path).name)
        for feature_name in _load_feature_subset(config_path):
            market_key, series_name = _required_sources(feature_name)
            if market_key:
                market_files.add(market_key)
            if series_name:
                extra_series.add(series_name)

    manifest: dict = {
        "cache_dir": args.cache_dir,
        "cache_tag": args.cache_tag,
        "generated_from_configs": config_names,
    }
    if "spot_file" in market_files:
        manifest["spot_file"] = "path/to/spot_ohlcv.csv"
    if "mark_file" in market_files:
        manifest["mark_file"] = "path/to/mark.csv"
    if "funding_file" in market_files:
        manifest["funding_file"] = "path/to/funding.csv"
    if "oi_file" in market_files:
        manifest["oi_file"] = "path/to/oi.csv"
    if extra_series:
        manifest["extra_series"] = {name: {"path": f"path/to/{name}.csv"} for name in sorted(extra_series)}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"[MANIFEST] Wrote stub -> {out_path}")


if __name__ == "__main__":
    main()
