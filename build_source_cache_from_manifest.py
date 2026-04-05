from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from build_aux_source_cache import (
    _load_any,
    _prepare_funding,
    _prepare_generic_series,
    _prepare_mark,
    _prepare_oi,
    _prepare_spot_ohlcv,
)
from build_coinmetrics_source_cache import _fetch_coinmetrics_metric
from build_glassnode_source_cache import _fetch_glassnode_metric
from unidream.data.download import (
    fetch_binance_ohlcv,
    fetch_funding_rate,
    fetch_mark_price_klines,
    fetch_open_interest_hist,
    fetch_taker_buy_sell_volume,
)


def _write_optional(cache_dir: str, cache_tag: str, suffix: str, frame) -> None:
    if frame is None:
        return
    out_path = os.path.join(cache_dir, f"{cache_tag}_{suffix}.parquet")
    frame.to_parquet(out_path)
    print(f"[MANIFEST] Wrote {suffix} cache -> {out_path} ({len(frame)} rows)")


def _parse_series_specs(raw_specs) -> list[tuple[str, str, str | None]]:
    if not raw_specs:
        return []
    specs: list[tuple[str, str, str | None]] = []
    if isinstance(raw_specs, dict):
        iterator = raw_specs.items()
    else:
        iterator = []
        for item in raw_specs:
            if not isinstance(item, dict) or "name" not in item or "path" not in item:
                raise ValueError("extra_series list entries must contain 'name' and 'path'")
            iterator.append((item["name"], item))
    for name, value in iterator:
        if isinstance(value, str):
            specs.append((str(name), value, None))
        elif isinstance(value, dict):
            path = value.get("path")
            if not path:
                raise ValueError(f"extra_series '{name}' is missing path")
            specs.append((str(name), path, value.get("column")))
        else:
            raise ValueError(f"Unsupported extra_series spec for '{name}'")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source cache from a YAML manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}

    cache_dir = manifest.get("cache_dir")
    cache_tag = manifest.get("cache_tag")
    if not cache_dir or not cache_tag:
        raise ValueError("Manifest must define cache_dir and cache_tag")

    os.makedirs(cache_dir, exist_ok=True)
    plan: list[str] = []

    if manifest.get("spot_file"):
        plan.append(f"import local spot_file -> {manifest['spot_file']}")
    if manifest.get("mark_file"):
        plan.append(f"import local mark_file -> {manifest['mark_file']}")
    if manifest.get("funding_file"):
        plan.append(f"import local funding_file -> {manifest['funding_file']}")
    if manifest.get("oi_file"):
        plan.append(f"import local oi_file -> {manifest['oi_file']}")
    for name, path, column in _parse_series_specs(manifest.get("extra_series")):
        suffix = f":{column}" if column else ""
        plan.append(f"import local extra_series {name} -> {path}{suffix}")

    binance_cfg = manifest.get("binance") or {}
    if binance_cfg:
        for key in ["symbol", "interval", "start", "end"]:
            if key not in binance_cfg:
                raise ValueError(f"binance section is missing '{key}'")
        plan.append(
            f"fetch binance spot/basis inputs -> symbol={binance_cfg['symbol']} interval={binance_cfg['interval']} "
            f"start={binance_cfg['start']} end={binance_cfg['end']}"
        )
        if binance_cfg.get("mark", False):
            plan.append("  include mark price")
        if binance_cfg.get("funding", False):
            plan.append("  include funding")
        if binance_cfg.get("oi", False):
            plan.append("  include open interest")
        if binance_cfg.get("taker_flow", False):
            plan.append("  include taker buy/sell flow")

    coinmetrics_cfg = manifest.get("coinmetrics") or {}
    if coinmetrics_cfg:
        for key in ["asset", "start", "end"]:
            if key not in coinmetrics_cfg:
                raise ValueError(f"coinmetrics section is missing '{key}'")
        metrics = coinmetrics_cfg.get("metrics") or {}
        if not metrics:
            raise ValueError("coinmetrics section must define metrics")
        plan.append(
            f"fetch coinmetrics metrics -> asset={coinmetrics_cfg['asset']} "
            f"freq={coinmetrics_cfg.get('frequency', '1h')} metrics={list(metrics.keys())}"
        )

    glassnode_cfg = manifest.get("glassnode") or {}
    if glassnode_cfg:
        for key in ["asset", "api_key"]:
            if key not in glassnode_cfg:
                raise ValueError(f"glassnode section is missing '{key}'")
        metrics = glassnode_cfg.get("metrics") or {}
        if not metrics:
            raise ValueError("glassnode section must define metrics")
        plan.append(
            f"fetch glassnode metrics -> asset={glassnode_cfg['asset']} "
            f"interval={glassnode_cfg.get('interval', '1h')} metrics={list(metrics.keys())}"
        )

    if args.dry_run:
        print("[MANIFEST] Dry run plan")
        for item in plan:
            print(f"  - {item}")
        return

    if manifest.get("spot_file"):
        _write_optional(cache_dir, cache_tag, "ohlcv", _prepare_spot_ohlcv(_load_any(manifest["spot_file"])))
    if manifest.get("mark_file"):
        _write_optional(cache_dir, cache_tag, "mark", _prepare_mark(_load_any(manifest["mark_file"])))
    if manifest.get("funding_file"):
        _write_optional(cache_dir, cache_tag, "funding", _prepare_funding(_load_any(manifest["funding_file"])))
    if manifest.get("oi_file"):
        _write_optional(cache_dir, cache_tag, "oi", _prepare_oi(_load_any(manifest["oi_file"])))

    for name, path, column in _parse_series_specs(manifest.get("extra_series")):
        series_df = _prepare_generic_series(_load_any(path), name=name, value_col=column)
        out_path = os.path.join(cache_dir, f"{cache_tag}_series_{name}.parquet")
        series_df.to_parquet(out_path)
        print(f"[MANIFEST] Wrote series cache -> {out_path} ({len(series_df)} rows) from {Path(path).name}")

    if binance_cfg:
        symbol = binance_cfg["symbol"]
        interval = binance_cfg["interval"]
        start = binance_cfg["start"]
        end = binance_cfg["end"]
        _write_optional(cache_dir, cache_tag, "ohlcv", fetch_binance_ohlcv(symbol, interval, start, end))
        if binance_cfg.get("mark", False):
            _write_optional(cache_dir, cache_tag, "mark", fetch_mark_price_klines(symbol, interval, start, end))
        if binance_cfg.get("funding", False):
            _write_optional(cache_dir, cache_tag, "funding", fetch_funding_rate(symbol, start, end))
        if binance_cfg.get("oi", False):
            _write_optional(cache_dir, cache_tag, "oi", fetch_open_interest_hist(symbol, interval, start, end))
        if binance_cfg.get("taker_flow", False):
            taker = fetch_taker_buy_sell_volume(symbol, interval, start, end)
            for col in ["signed_order_flow", "taker_imbalance", "buy_sell_ratio"]:
                if col not in taker.columns:
                    continue
                out_path = os.path.join(cache_dir, f"{cache_tag}_series_{col}.parquet")
                taker[[col]].to_parquet(out_path)
                print(f"[MANIFEST] Wrote series cache -> {out_path} ({len(taker)} rows)")

    if coinmetrics_cfg:
        asset = coinmetrics_cfg["asset"]
        start = coinmetrics_cfg["start"]
        end = coinmetrics_cfg["end"]
        frequency = coinmetrics_cfg.get("frequency", "1h")
        api_key = coinmetrics_cfg.get("api_key")
        for alias, metric_spec in (coinmetrics_cfg.get("metrics") or {}).items():
            if isinstance(metric_spec, dict):
                metric = metric_spec["metric"]
                transform = metric_spec.get("transform")
            else:
                metric = metric_spec
                transform = None
            df = _fetch_coinmetrics_metric(
                asset=asset,
                metric=metric,
                start=start,
                end=end,
                frequency=frequency,
                api_key=api_key,
            ).rename(columns={metric: alias})
            from build_coinmetrics_source_cache import _apply_transform as _apply_cm_transform
            df = _apply_cm_transform(df, alias, transform)
            out_path = os.path.join(cache_dir, f"{cache_tag}_series_{alias}.parquet")
            df.to_parquet(out_path)
            print(f"[MANIFEST] Wrote series cache -> {out_path} ({len(df)} rows) metric={metric}")

    if glassnode_cfg:
        asset = glassnode_cfg["asset"]
        start = glassnode_cfg.get("start")
        end = glassnode_cfg.get("end")
        interval = glassnode_cfg.get("interval", "1h")
        api_key = glassnode_cfg["api_key"]
        pit = bool(glassnode_cfg.get("pit", False))
        for alias, metric_spec in (glassnode_cfg.get("metrics") or {}).items():
            if isinstance(metric_spec, dict):
                metric_path = metric_spec["metric"]
                transform = metric_spec.get("transform")
            else:
                metric_path = metric_spec
                transform = None
            df = _fetch_glassnode_metric(
                asset=asset,
                metric_path=metric_path,
                start=start,
                end=end,
                interval=interval,
                api_key=api_key,
                pit=pit,
            ).rename(columns={"value": alias})
            from build_glassnode_source_cache import _apply_transform as _apply_gn_transform
            df = _apply_gn_transform(df, alias, transform)
            out_path = os.path.join(cache_dir, f"{cache_tag}_series_{alias}.parquet")
            df.to_parquet(out_path)
            print(f"[MANIFEST] Wrote series cache -> {out_path} ({len(df)} rows) metric={metric_path}")


if __name__ == "__main__":
    main()
