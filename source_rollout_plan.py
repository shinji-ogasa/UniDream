from __future__ import annotations


def dedupe_missing_targets(missing: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in missing:
        target = item.split("missing ", 1)[-1]
        if target not in seen:
            out.append(target)
            seen.add(target)
    return out


def parse_cache_tag(cache_tag: str) -> tuple[str, str, str, str]:
    parts = cache_tag.split("_")
    if len(parts) < 5:
        return ("BTCUSDT", "15m", "2021-01-01", "2023-06-01")
    return (parts[0], parts[1], parts[2], parts[3])


def fetch_command_hint(cache_dir: str, cache_tag: str, target: str) -> str:
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
