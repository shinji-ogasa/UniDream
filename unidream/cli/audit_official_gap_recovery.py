"""Probe official Binance Spot sources around development-cache gaps."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import pandas as pd

from unidream.eval.gap_recovery import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    OFFICIAL_SPOT_REST_BASE,
    probe_official_gap_recovery,
    write_gap_recovery_jsonl,
)
from unidream.experiments.runtime import load_config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cache_paths(config: dict) -> tuple[Path, Path]:
    run_cfg = config["run"]
    data_cfg = config["data"]
    zscore_window = int(config["normalization"]["zscore_window_days"])
    tag = (
        f"{data_cfg['symbol']}_{data_cfg['interval']}_{run_cfg['start']}_{run_cfg['end']}"
        f"_z{zscore_window}_v3"
    )
    cache_dir = Path(config["logging"]["cache_dir"])
    return cache_dir / f"{tag}_features.parquet", cache_dir / f"{tag}_returns.parquet"


def _with_provenance(
    report: dict,
    *,
    feature_path: Path,
    returns_path: Path,
    config_path: Path,
) -> dict:
    report["provenance"] = {
        "git_commit": _git_commit(),
        "config_path": str(config_path),
        "features_path": str(feature_path),
        "returns_path": str(returns_path),
        "features_sha256": _sha256_file(feature_path),
        "returns_sha256": _sha256_file(returns_path),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe only official Binance Spot sources around development-cache gaps",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        default="configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml",
    )
    parser.add_argument("--features", "--features-path", dest="features_path", default=None)
    parser.add_argument("--returns", "--returns-path", dest="returns_path", default=None)
    parser.add_argument("--start", default=str(DEVELOPMENT_START.date()))
    parser.add_argument("--end", default=str(DEVELOPMENT_END.date()))
    parser.add_argument("--interval", default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--rest-base-url", default=OFFICIAL_SPOT_REST_BASE)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--no-archive-fallback",
        action="store_true",
        help="do not query official monthly archives after an incomplete REST response",
    )
    parser.add_argument(
        "--ledger",
        default="docs/data_quality_gap_recovery_2018_2024.jsonl",
    )
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="write evidence and return success even when an official source leaves bars unresolved",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.start != str(DEVELOPMENT_START.date()) or args.end != str(DEVELOPMENT_END.date()):
        parser.error("this command is restricted to [2018-01-01, 2024-01-01)")
    if (args.features_path is None) != (args.returns_path is None):
        parser.error("--features and --returns must be supplied together")

    try:
        config_path = Path(args.config)
        config = load_config(str(config_path))
        default_features, default_returns = _cache_paths(config)
        feature_path = Path(args.features_path) if args.features_path else default_features
        returns_path = Path(args.returns_path) if args.returns_path else default_returns
        features = pd.read_parquet(feature_path)
        returns = pd.read_parquet(returns_path)
        data_cfg = config.get("data", {})
        report = probe_official_gap_recovery(
            features,
            returns=returns,
            symbol=str(args.symbol or data_cfg.get("symbol", "BTCUSDT")),
            interval=str(args.interval or data_cfg.get("interval", "15m")),
            start=args.start,
            end=args.end,
            rest_base_url=args.rest_base_url,
            use_archive_fallback=not args.no_archive_fallback,
            timeout=args.timeout,
        )
        _with_provenance(
            report,
            feature_path=feature_path,
            returns_path=returns_path,
            config_path=config_path,
        )
        record_count = write_gap_recovery_jsonl(report, args.ledger)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"[gap-recovery] ERROR: {exc}")
        return 2

    summary = report["summary"]
    print(f"[gap-recovery] status: {summary['status']}")
    print(f"[gap-recovery] gaps: {summary['gap_count']}")
    print(f"[gap-recovery] expected missing bars: {summary['expected_missing_bars']}")
    print(f"[gap-recovery] official covered bars: {summary['official_covered_bars']}")
    print(f"[gap-recovery] unresolved after official probes: {summary['official_missing_after_probe']}")
    print(f"[gap-recovery] ledger records: {record_count} -> {args.ledger}")
    if summary["status"] != "pass" and not args.allow_unresolved:
        print("[gap-recovery] unresolved official bars; no interpolation or cache write was performed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
