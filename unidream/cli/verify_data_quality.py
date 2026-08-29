"""Run the development-cache data and feature quality gate."""
from __future__ import annotations

import argparse
from pathlib import Path

from unidream.eval.data_quality import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    audit_cache,
    render_markdown_report,
    write_jsonl,
)
from unidream.experiments.runtime import load_config


def _cache_paths(config: dict) -> tuple[Path, Path, Path]:
    run_cfg = config["run"]
    data_cfg = config["data"]
    zscore_window = int(config["normalization"]["zscore_window_days"])
    tag = (
        f"{data_cfg['symbol']}_{data_cfg['interval']}_{run_cfg['start']}_{run_cfg['end']}"
        f"_z{zscore_window}_v3"
    )
    cache_dir = Path(config["logging"]["cache_dir"])
    return (
        cache_dir / f"{tag}_features.parquet",
        cache_dir / f"{tag}_returns.parquet",
        cache_dir / f"{tag}_metadata.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit only the 2018-01-01..2024-01-01 development feature cache",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        default="configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml",
        help="strict research config used to resolve the cache and WFO boundaries",
    )
    parser.add_argument("--features", "--features-path", dest="features_path", default=None)
    parser.add_argument("--returns", "--returns-path", dest="returns_path", default=None)
    parser.add_argument("--metadata", "--metadata-path", dest="metadata_path", default=None)
    parser.add_argument("--start", default=str(DEVELOPMENT_START.date()))
    parser.add_argument("--end", default=str(DEVELOPMENT_END.date()))
    parser.add_argument("--interval", default=None)
    parser.add_argument(
        "--ledger",
        default="docs/data_quality_gate_2018_2024.jsonl",
        help="JSONL ledger destination",
    )
    parser.add_argument(
        "--report",
        default="docs/data_quality_gate_2018_2024.md",
        help="Markdown report destination",
    )
    parser.add_argument(
        "--allow-quality-gate-fail",
        action="store_true",
        help="write evidence and return success while preserving a failed gate in the report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.start != str(DEVELOPMENT_START.date()) or args.end != str(DEVELOPMENT_END.date()):
        parser.error(
            "this command is restricted to the development scope "
            "[2018-01-01, 2024-01-01)"
        )

    try:
        config = load_config(args.config)
        default_features, default_returns, default_metadata = _cache_paths(config)
        paths = [
            Path(args.features_path) if args.features_path else default_features,
            Path(args.returns_path) if args.returns_path else default_returns,
            Path(args.metadata_path) if args.metadata_path else default_metadata,
        ]
        if any(value is None for value in (args.features_path, args.returns_path, args.metadata_path)) and any(
            value is not None for value in (args.features_path, args.returns_path, args.metadata_path)
        ):
            parser.error("--features, --returns, and --metadata must be supplied together")
        interval = args.interval or str(config["data"]["interval"])
        report = audit_cache(
            paths[0],
            paths[1],
            paths[2],
            config=config,
            start=args.start,
            end=args.end,
            interval=interval,
        )
        ledger_count = write_jsonl(report, args.ledger)
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            render_markdown_report(report, ledger_path=args.ledger),
            encoding="utf-8",
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"[data-quality] ERROR: {exc}")
        return 2

    print(f"[data-quality] gate: {report['gates']['overall']}")
    print(f"[data-quality] cache contract: {report['gates']['cache_contract']}")
    print(f"[data-quality] causality: {report['gates']['causality']}")
    print(f"[data-quality] same-row fairness: {report['gates']['same_row_ohlcv13_vs_full17']}")
    print(f"[data-quality] availability mask: {report['gates']['external_availability_mask']}")
    print(f"[data-quality] ledger records: {ledger_count} -> {args.ledger}")
    print(f"[data-quality] report: {args.report}")
    if report["gates"]["overall"] != "pass" and not args.allow_quality_gate_fail:
        print("[data-quality] blocking quality gate; use --allow-quality-gate-fail only for evidence export")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
