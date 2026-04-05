from __future__ import annotations

import argparse

import yaml


SUPPORTED_TRANSFORMS = {"none", "diff", "pct_change", "logdiff"}


def validate_manifest(manifest_path: str) -> list[str]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}

    errors: list[str] = []
    if not manifest.get("cache_dir"):
        errors.append("cache_dir is missing")
    if not manifest.get("cache_tag"):
        errors.append("cache_tag is missing")

    binance_cfg = manifest.get("binance") or {}
    if binance_cfg:
        for key in ("symbol", "interval", "start", "end"):
            if not binance_cfg.get(key):
                errors.append(f"binance.{key} is missing")

    coinmetrics_cfg = manifest.get("coinmetrics") or {}
    if coinmetrics_cfg:
        for key in ("asset", "start", "end"):
            if not coinmetrics_cfg.get(key):
                errors.append(f"coinmetrics.{key} is missing")
        metrics = coinmetrics_cfg.get("metrics") or {}
        if not metrics:
            errors.append("coinmetrics.metrics is empty")
        for alias, spec in metrics.items():
            if isinstance(spec, str):
                continue
            if not isinstance(spec, dict):
                errors.append(f"coinmetrics.metrics.{alias} must be string or mapping")
                continue
            if not spec.get("metric"):
                errors.append(f"coinmetrics.metrics.{alias}.metric is missing")
            transform = spec.get("transform")
            if transform and transform not in SUPPORTED_TRANSFORMS:
                errors.append(f"coinmetrics.metrics.{alias}.transform is unsupported: {transform}")

    glassnode_cfg = manifest.get("glassnode") or {}
    if glassnode_cfg:
        for key in ("asset", "api_key"):
            if not glassnode_cfg.get(key):
                errors.append(f"glassnode.{key} is missing")
        metrics = glassnode_cfg.get("metrics") or {}
        if not metrics:
            errors.append("glassnode.metrics is empty")
        for alias, spec in metrics.items():
            if isinstance(spec, str):
                continue
            if not isinstance(spec, dict):
                errors.append(f"glassnode.metrics.{alias} must be string or mapping")
                continue
            if not spec.get("metric"):
                errors.append(f"glassnode.metrics.{alias}.metric is missing")
            transform = spec.get("transform")
            if transform and transform not in SUPPORTED_TRANSFORMS:
                errors.append(f"glassnode.metrics.{alias}.transform is unsupported: {transform}")

    extra_series = manifest.get("extra_series") or {}
    if isinstance(extra_series, list):
        for idx, item in enumerate(extra_series):
            if not isinstance(item, dict):
                errors.append(f"extra_series[{idx}] must be a mapping")
                continue
            if not item.get("name"):
                errors.append(f"extra_series[{idx}].name is missing")
            if not item.get("path"):
                errors.append(f"extra_series[{idx}].path is missing")
    elif isinstance(extra_series, dict):
        for alias, spec in extra_series.items():
            if isinstance(spec, str):
                continue
            if not isinstance(spec, dict):
                errors.append(f"extra_series.{alias} must be string or mapping")
                continue
            if not spec.get("path"):
                errors.append(f"extra_series.{alias}.path is missing")
    elif extra_series:
        errors.append("extra_series must be a mapping or list")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate source-cache manifest YAML")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    errors = validate_manifest(args.manifest)
    if errors:
        print("[MANIFEST] Invalid manifest:")
        for err in errors:
            print(f"  {err}")
        raise SystemExit(1)

    print("[MANIFEST] Manifest is valid.")


if __name__ == "__main__":
    main()
