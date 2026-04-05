from __future__ import annotations

import argparse
from pathlib import Path

from check_config_source_requirements import collect_missing_requirements


def _dedupe_targets(missing: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in missing:
        target = item.split("missing ", 1)[-1]
        if target not in seen:
            out.append(target)
            seen.add(target)
    return out


def _command_hint(cache_dir: str, cache_tag: str, target: str) -> str:
    if target.endswith("_mark.parquet") or target.endswith("_funding.parquet") or target.endswith("_oi.parquet"):
        return (
            ".\\.venv\\Scripts\\python.exe build_binance_source_cache.py "
            f"--cache-dir {cache_dir} --cache-tag {cache_tag} "
            "--symbol BTCUSDT --interval 15m --start 2021-01-01 --end 2023-06-01"
        )
    if target.endswith("_series_signed_order_flow.parquet") or target.endswith("_series_taker_imbalance.parquet"):
        return (
            ".\\.venv\\Scripts\\python.exe build_binance_source_cache.py "
            f"--cache-dir {cache_dir} --cache-tag {cache_tag} "
            "--symbol BTCUSDT --interval 15m --start 2021-01-01 --end 2023-06-01"
        )
    if target.endswith("_series_exchange_netflow.parquet"):
        return (
            ".\\.venv\\Scripts\\python.exe build_glassnode_source_cache.py "
            f"--cache-dir {cache_dir} --cache-tag {cache_tag} "
            "--asset BTC --start 2021-01-01 --end 2023-06-01 --pit "
            "--interval 1h --api-key <glassnode_key> "
            "--metric exchange_netflow=transactions/transfers_volume_exchanges_net"
        )
    if target.endswith("_series_active_address_growth.parquet"):
        return (
            ".\\.venv\\Scripts\\python.exe build_coinmetrics_source_cache.py "
            f"--cache-dir {cache_dir} --cache-tag {cache_tag} "
            "--asset btc --start 2021-01-01 --end 2023-06-01 "
            "--frequency 1h --metric active_address_growth=AdrActCnt:logdiff"
        )
    if target.endswith("_series_stablecoin_inflow.parquet"):
        return (
            ".\\.venv\\Scripts\\python.exe build_aux_source_cache.py "
            f"--cache-dir {cache_dir} --cache-tag {cache_tag} "
            "--extra-series stablecoin_inflow=path/to/stablecoin_inflow.csv:stablecoin_inflow"
        )
    return f"# No built-in fetch hint for {target}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend concrete fetch commands for the next blocked source stage")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--config", action="append", required=True)
    args = parser.parse_args()

    for config_path in args.config:
        missing = collect_missing_requirements(args.cache_dir, args.cache_tag, config_path)
        if not missing:
            continue
        config_name = Path(config_path).name
        print(f"[FETCH] Next blocked config -> {config_name}")
        for target in _dedupe_targets(missing):
            print(f"[FETCH] {target}")
            print(f"  {_command_hint(args.cache_dir, args.cache_tag, target)}")
        break
    else:
        print("[FETCH] All provided configs are unlocked.")


if __name__ == "__main__":
    main()
