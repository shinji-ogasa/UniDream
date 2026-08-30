"""Acquire the small D1 signed-flow kline pilot and write evidence ledgers.

The command performs no model training or prediction evaluation.  It downloads
only monthly Spot/USD-M kline metadata archives, verifies each official
CHECKSUM sidecar, builds completed-bar 15-minute features, and probes
aggregate-trade archive sizes with HEAD requests only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from unidream.data.d1_signed_flow import (
    D1_AVAILABILITY_COLUMNS,
    D1_FEATURE_COLUMNS,
    D1_INTERVAL,
    D1_SOURCE_NAMES,
    OFFICIAL_PUBLIC_DATA_README,
    OFFICIAL_SPOT_MARKET_DATA_DOCS,
    OFFICIAL_UM_MARKET_DATA_DOCS,
    append_jsonl,
    build_d1_features,
    classify_archive_revisions,
    d1_bar_ledger_records,
    download_d1_kline_month,
    estimate_aggtrade_archive_storage,
    summarize_d1_pilot,
)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_write(value: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _ledger_record_counts(path: str | Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            counts[str(record.get("record_type", "unknown"))] += 1
    return dict(counts)


def _render_report(
    *,
    symbol: str,
    interval: str,
    month: str,
    summary: Mapping[str, Any],
    source_records: Mapping[str, Mapping[str, Any]],
    capacity: Mapping[str, Any],
    feature_path: str | Path,
    availability_path: str | Path,
    capacity_path: str | Path,
    ledger_path: str | Path,
    feature_sha256: str,
    availability_sha256: str,
    ledger_record_counts: Mapping[str, int],
    ledger_total_record_counts: Mapping[str, int],
    git_commit: str | None,
) -> str:
    lines = [
        "# D1 signed-flow acquisition pilot",
        "",
        "This is a data-only pilot. No model, prediction result, or P2 tournament was run.",
        "",
        f"- Scope: `{symbol}` `{interval}` monthly kline metadata, `{month}`",
        "- Row semantics: `decision_ts = bar_open_ts + 15m`; each feature row covers Binance `[bar_open_ts, decision_ts)` with inclusive `close_time=decision_ts-1ms`.",
        "- Leakage rule: a bar is eligible only after its close; no next bar is read while constructing a row.",
        "- Feature artifact: `" + str(feature_path) + "`",
        "- Availability artifact: `" + str(availability_path) + "`",
        "- Capacity artifact: `" + str(capacity_path) + "`",
        "- Append-only ledger: `" + str(ledger_path) + "`",
        f"- Acquisition code commit: `{git_commit}`",
        f"- Feature SHA256: `{feature_sha256}`",
        f"- Availability SHA256: `{availability_sha256}`",
        "",
        "## Official sources",
        "",
        f"- [Binance Public Data README]({OFFICIAL_PUBLIC_DATA_README})",
        f"- [Spot market-data REST specification]({OFFICIAL_SPOT_MARKET_DATA_DOCS})",
        f"- [USD-M Futures market-data REST specification]({OFFICIAL_UM_MARKET_DATA_DOCS})",
        "",
        "The README documents monthly/daily archives, Spot/Futures klines and checksum sidecars. The USD-M specification documents the corresponding market-data endpoints. Archive publication/download timestamps are recorded separately from live observation timestamps; this archive pilot does not claim live causal availability.",
        "",
        "## Download and revision evidence",
        "",
        "| source | HTTP | checksum | archive revision | parsed rows | live causal eligible |",
        "| --- | ---: | --- | --- | ---: | --- |",
    ]
    for source in D1_SOURCE_NAMES:
        record = source_records.get(source, {})
        lines.append(
            "| `{source}` | `{status}` | `{checksum}` | `{revision}` | `{rows}` | `{live}` |".format(
                source=source,
                status=record.get("http_status"),
                checksum=record.get("checksum_verified"),
                revision=record.get("archive_revision_id"),
                rows=record.get("parsed_rows"),
                live=record.get("live_causal_eligible"),
            )
        )
    lines.extend(
        [
            "",
            "`archive_published_ts` is unknown for these downloaded files and `collector_observed_ts` is null. A later archive revision is never silently substituted: the ledger records previous and replacement revision IDs.",
            "",
            "## D1 feature contract",
            "",
            "- Spot and USD-M fields: trade count, quote volume, taker-buy base volume and taker-buy quote volume.",
            "- Taker imbalance per venue: `(2 * taker_buy_quote / quote_volume) - 1` when quote volume is positive; otherwise the value is NaN and its mask is false.",
            "- Spot-perp basis: `log(perp_close / spot_close)` at the completed bar close.",
            "- Spot-perp return divergence: `log(perp_close_t/perp_close_{t-1}) - log(spot_close_t/spot_close_{t-1})`, requiring adjacent observed bars on both venues.",
            "- Missing rows remain NaN. A numeric zero is retained as a value and is not used as a missing sentinel.",
            "",
            "| item | value |",
            "| --- | ---: |",
            f"| rows | `{summary.get('rows')}` |",
            f"| fully available D1 rows | `{summary.get('d1_features_available_rows')}` ({float(summary.get('d1_features_available_fraction', 0.0)):.2%}) |",
            f"| NaN feature cells | `{summary.get('nan_feature_cells')}` |",
            f"| literal zero feature cells | `{summary.get('zero_valued_feature_cells')}` |",
            "",
            "Availability columns: `" + ", ".join(D1_AVAILABILITY_COLUMNS) + "`.",
            "Feature columns: `" + ", ".join(D1_FEATURE_COLUMNS) + "`.",
            "",
            "Latest-run appended ledger record counts: "
            + ", ".join(
                f"`{key}`={value}" for key, value in ledger_record_counts.items()
            )
            + ".",
            "Tracked append-only ledger total counts: "
            + ", ".join(
                f"`{key}`={value}" for key, value in ledger_total_record_counts.items()
            )
            + ".",
            "",
            "## Aggregate-trade capacity check",
            "",
            f"Method: `{capacity.get('method')}`",
            f"Known compressed bytes across requested Spot + USD-M monthly archives: `{capacity.get('estimated_compressed_bytes_known')}`",
            "",
            "| source | requested months | HTTP 200 | HTTP 404 | known-size months | unknown-size months |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for source in ("spot_aggTrades", "um_aggTrades"):
        source_summary = (capacity.get("sources") or {}).get(source, {})
        lines.append(
            "| `{source}` | `{requested}` | `{http_200}` | `{http_404}` | `{known}` | `{unknown}` |".format(
                source=source,
                requested=source_summary.get("months_requested"),
                http_200=source_summary.get("http_200_count"),
                http_404=source_summary.get("http_404_count"),
                known=source_summary.get("known_size_months"),
                unknown=source_summary.get("unknown_size_months"),
            )
        )
    lines.extend(
        [
            "",
            "No aggregate-trade payload was downloaded. The estimate is based on official `Content-Length` values for HTTP 200 monthly ZIPs only; unknown/404 months remain explicit in the capacity JSON and append-only ledger.",
            "",
            "This artifact is feasibility evidence only. It does not establish that any D1 feature predicts returns or improves trading utility.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    months = list(args.months or ["2024-01"])
    if len(months) != 1:
        raise ValueError("the first D1 pilot intentionally accepts exactly one month")
    month = months[0]
    month_start = pd.Timestamp(f"{month}-01", tz="UTC")
    month_end = (month_start + pd.offsets.MonthBegin(1)).normalize()

    source_frames: dict[str, pd.DataFrame] = {}
    source_records_raw: list[dict[str, Any]] = []
    for source in D1_SOURCE_NAMES:
        frame, record = download_d1_kline_month(
            source,
            symbol=args.symbol,
            interval=args.interval,
            month=month,
            raw_dir=args.raw_dir,
            timeout=args.timeout,
        )
        source_frames[source] = frame
        source_records_raw.append(record)

    ledger_path = Path(args.ledger)
    source_records_list = classify_archive_revisions(
        source_records_raw,
        ledger_path=ledger_path,
    )
    source_records = {record["source"]: record for record in source_records_list}
    source_errors = [record for record in source_records_list if record.get("error")]
    if source_errors:
        append_jsonl(ledger_path, source_records_list)
        raise RuntimeError(
            "D1 archive download/checksum failed: "
            + "; ".join(
                f"{record.get('source')}: {record.get('error')}"
                for record in source_errors
            )
        )

    features, availability = build_d1_features(
        source_frames["spot_klines"],
        source_frames["um_klines"],
        interval=args.interval,
        bar_open_start=month_start,
        bar_open_end=month_end,
    )
    capacity = estimate_aggtrade_archive_storage(
        symbol=args.symbol,
        start=args.capacity_start,
        end=args.capacity_end,
        timeout=args.timeout,
    )
    summary = summarize_d1_pilot(
        features,
        availability,
        source_records=source_records,
        capacity=capacity,
    )

    feature_path = Path(args.features)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(feature_path, index=True)
    availability_path = Path(args.availability)
    availability_path.parent.mkdir(parents=True, exist_ok=True)
    availability.to_csv(availability_path, index=True)
    capacity_path = Path(args.capacity_json)
    _json_write(capacity, capacity_path)
    feature_sha256 = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    availability_sha256 = hashlib.sha256(availability_path.read_bytes()).hexdigest()
    bar_records = d1_bar_ledger_records(
        availability,
        source_records=source_records,
        interval=args.interval,
    )
    capacity_records = list(capacity.get("records") or [])
    ledger_record_counts = {
        "d1_pilot_run": 1,
        "d1_archive_download": len(source_records_list),
        "d1_aggtrade_head_probe": len(capacity_records),
        "d1_bar_availability": len(bar_records),
    }
    run_record = {
        "record_type": "d1_pilot_run",
        "schema_version": 1,
        "git_commit": _git_commit(),
        "symbol": args.symbol,
        "interval": args.interval,
        "month": month,
        "bar_open_scope": {
            "start_inclusive": month_start.isoformat(),
            "end_exclusive": month_end.isoformat(),
        },
        "decision_scope": {
            "start_inclusive": str(features.index[0]) if len(features) else None,
            "end_inclusive": str(features.index[-1]) if len(features) else None,
        },
        "interval_semantics": "[bar_open_ts, decision_ts); close_time=decision_ts-1ms",
        "feature_columns": list(features.columns),
        "availability_columns": list(availability.columns),
        "feature_path": str(feature_path),
        "availability_path": str(availability_path),
        "feature_sha256": feature_sha256,
        "availability_sha256": availability_sha256,
        "ledger_record_counts": ledger_record_counts,
        "summary": summary,
        "official_sources": {
            "public_data_readme": OFFICIAL_PUBLIC_DATA_README,
            "spot_market_data_docs": OFFICIAL_SPOT_MARKET_DATA_DOCS,
            "um_market_data_docs": OFFICIAL_UM_MARKET_DATA_DOCS,
        },
        "model_results_read": False,
        "p2_run": False,
    }
    append_jsonl(
        ledger_path,
        [run_record, *source_records_list, *capacity_records, *bar_records],
    )
    ledger_total_record_counts = _ledger_record_counts(ledger_path)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(
            symbol=args.symbol,
            interval=args.interval,
            month=month,
            summary=summary,
            source_records=source_records,
            capacity=capacity,
            feature_path=feature_path,
            availability_path=availability_path,
            capacity_path=capacity_path,
            ledger_path=ledger_path,
            feature_sha256=feature_sha256,
            availability_sha256=availability_sha256,
            ledger_record_counts=ledger_record_counts,
            ledger_total_record_counts=ledger_total_record_counts,
            git_commit=run_record["git_commit"],
        ),
        encoding="utf-8",
    )
    return {
        "summary": summary,
        "capacity": capacity,
        "source_records": source_records,
        "report": str(report_path),
        "ledger": str(ledger_path),
        "features": str(feature_path),
        "availability": str(availability_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire a small, checksum-verified D1 signed-flow kline pilot",
        allow_abbrev=False,
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default=D1_INTERVAL)
    parser.add_argument(
        "--month",
        dest="months",
        action="append",
        default=None,
        help="one YYYY-MM pilot month (defaults to 2024-01; repeat is reserved for a future multi-month mode)",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--raw-dir", default="checkpoints/d1_signed_flow_raw")
    parser.add_argument(
        "--features",
        default="docs/d1_signed_flow_pilot/pilot_features.csv",
    )
    parser.add_argument(
        "--availability",
        default="docs/d1_signed_flow_pilot/pilot_availability.csv",
    )
    parser.add_argument(
        "--capacity-json",
        default="docs/d1_signed_flow_pilot/aggtrade_capacity.json",
    )
    parser.add_argument(
        "--ledger",
        default="docs/d1_signed_flow_pilot/availability_revision_ledger.jsonl",
    )
    parser.add_argument(
        "--report",
        default="docs/d1_signed_flow_pilot/report.md",
    )
    parser.add_argument("--capacity-start", default="2018-01")
    parser.add_argument("--capacity-end", default="2026-09")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_pilot(args)
    summary = result["summary"]
    print(
        "D1 pilot complete: "
        f"rows={summary['rows']} "
        f"eligible={summary['d1_features_available_rows']} "
        f"aggtrade_known_compressed_bytes={result['capacity']['estimated_compressed_bytes_known']}"
    )
    print(f"Report: {result['report']}")
    print(f"Ledger: {result['ledger']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
