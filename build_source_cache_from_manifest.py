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
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}

    cache_dir = manifest.get("cache_dir")
    cache_tag = manifest.get("cache_tag")
    if not cache_dir or not cache_tag:
        raise ValueError("Manifest must define cache_dir and cache_tag")

    os.makedirs(cache_dir, exist_ok=True)

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


if __name__ == "__main__":
    main()
