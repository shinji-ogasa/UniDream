from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from check_config_source_requirements import collect_missing_requirements
from source_rollout_plan import dedupe_missing_targets, parse_cache_tag


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a manifest stub for the next blocked source stage")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    symbol, interval, start, end = parse_cache_tag(args.cache_tag)

    next_name = None
    next_targets: list[str] = []
    for config_path in args.config:
        missing = collect_missing_requirements(args.cache_dir, args.cache_tag, config_path)
        if missing:
            next_name = Path(config_path).name
            next_targets = dedupe_missing_targets(missing)
            break

    if next_name is None:
        raise SystemExit("All provided configs are unlocked; no next-stage manifest needed.")

    manifest: dict = {
        "cache_dir": args.cache_dir,
        "cache_tag": args.cache_tag,
        "generated_for_next_config": next_name,
    }

    for target in next_targets:
        if target.endswith("_mark.parquet") or target.endswith("_funding.parquet") or target.endswith("_oi.parquet"):
            binance = manifest.setdefault(
                "binance",
                {"symbol": symbol, "interval": interval, "start": start, "end": end},
            )
            if target.endswith("_mark.parquet"):
                binance["mark"] = True
            elif target.endswith("_funding.parquet"):
                binance["funding"] = True
            elif target.endswith("_oi.parquet"):
                binance["oi"] = True
        elif target.endswith("_series_signed_order_flow.parquet") or target.endswith("_series_taker_imbalance.parquet"):
            binance = manifest.setdefault(
                "binance",
                {"symbol": symbol, "interval": interval, "start": start, "end": end},
            )
            binance["taker_flow"] = True
        elif target.endswith("_series_exchange_netflow.parquet"):
            glassnode = manifest.setdefault(
                "glassnode",
                {
                    "asset": "BTC",
                    "start": start,
                    "end": end,
                    "interval": "1h",
                    "pit": True,
                    "api_key": "<glassnode_api_key>",
                    "metrics": {},
                },
            )
            glassnode["metrics"]["exchange_netflow"] = "transactions/transfers_volume_exchanges_net"
        elif target.endswith("_series_active_address_growth.parquet"):
            coinmetrics = manifest.setdefault(
                "coinmetrics",
                {"asset": "btc", "start": start, "end": end, "frequency": "1h", "metrics": {}},
            )
            coinmetrics["metrics"]["active_address_growth"] = {
                "metric": "AdrActCnt",
                "transform": "logdiff",
            }
        elif target.endswith("_series_stablecoin_inflow.parquet"):
            extra_series = manifest.setdefault("extra_series", {})
            extra_series["stablecoin_inflow"] = {"path": "path/to/stablecoin_inflow.csv", "column": "stablecoin_inflow"}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"[NEXT-MANIFEST] Wrote stub -> {out_path}")


if __name__ == "__main__":
    main()
