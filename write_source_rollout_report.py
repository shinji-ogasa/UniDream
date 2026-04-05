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


def _fetch_hint(cache_dir: str, cache_tag: str, target: str) -> str:
    if target.endswith("_series_signed_order_flow.parquet") or target.endswith("_series_taker_imbalance.parquet"):
        return (
            f"`./.venv/Scripts/python.exe build_binance_source_cache.py --cache-dir {cache_dir} "
            f"--cache-tag {cache_tag} --symbol BTCUSDT --interval 15m --start 2021-01-01 --end 2023-06-01`"
        )
    if target.endswith("_series_exchange_netflow.parquet"):
        return (
            f"`./.venv/Scripts/python.exe build_glassnode_source_cache.py --cache-dir {cache_dir} "
            f"--cache-tag {cache_tag} --asset BTC --start 2021-01-01 --end 2023-06-01 "
            f"--pit --interval 1h --api-key <glassnode_key> "
            f"--metric exchange_netflow=transactions/transfers_volume_exchanges_net`"
        )
    if target.endswith("_series_active_address_growth.parquet"):
        return (
            f"`./.venv/Scripts/python.exe build_coinmetrics_source_cache.py --cache-dir {cache_dir} "
            f"--cache-tag {cache_tag} --asset btc --start 2021-01-01 --end 2023-06-01 "
            f"--frequency 1h --metric active_address_growth=AdrActCnt:logdiff`"
        )
    if target.endswith("_series_stablecoin_inflow.parquet"):
        return (
            f"`./.venv/Scripts/python.exe build_aux_source_cache.py --cache-dir {cache_dir} "
            f"--cache-tag {cache_tag} --extra-series stablecoin_inflow=path/to/stablecoin_inflow.csv:stablecoin_inflow`"
        )
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a markdown report for current source rollout status")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", action="append", required=True)
    args = parser.parse_args()

    unlocked: list[str] = []
    blocked: list[tuple[str, list[str]]] = []
    for config_path in args.config:
        missing = collect_missing_requirements(args.cache_dir, args.cache_tag, config_path)
        name = Path(config_path).name
        if missing:
            blocked.append((name, missing))
        else:
            unlocked.append(name)

    lines: list[str] = []
    lines.append("# Source Rollout Report")
    lines.append("")
    lines.append(f"- Cache dir: `{args.cache_dir}`")
    lines.append(f"- Cache tag: `{args.cache_tag}`")
    lines.append("")
    lines.append("## Unlocked Configs")
    if unlocked:
        for name in unlocked:
            lines.append(f"- `{name}`")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Blocked Configs")
    if blocked:
        for name, missing in blocked:
            lines.append(f"- `{name}`")
            for item in missing:
                lines.append(f"  - {item}")
    else:
        lines.append("- none")
    lines.append("")

    if blocked:
        next_name, next_missing = blocked[0]
        lines.append("## Next Stage")
        lines.append(f"- Next blocked config: `{next_name}`")
        for target in _dedupe_targets(next_missing):
            lines.append(f"- Needed raw source: `{target}`")
            hint = _fetch_hint(args.cache_dir, args.cache_tag, target)
            if hint:
                lines.append(f"  - Fetch hint: {hint}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[REPORT] Wrote markdown report -> {out_path}")


if __name__ == "__main__":
    main()
