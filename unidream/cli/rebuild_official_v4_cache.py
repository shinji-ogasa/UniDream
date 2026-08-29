"""Rebuild a schema-v4 cache from official Binance archives only."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from unidream.data.cache_v4 import write_cache_v4
from unidream.data.official_v4_sources import probe_official_sources
from unidream.data.rebuild_v4 import (
    DEFAULT_END,
    DEFAULT_INTERVAL,
    DEFAULT_START,
    DEFAULT_SYMBOL,
    EXTERNAL_ARCHIVE_START,
    OfficialSourceError,
    rebuild_official_v4_frames,
)
from unidream.experiments.runtime import load_config


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


def _write_ledger(
    *,
    path: str | Path,
    source_probe: Mapping[str, Any],
    rebuild: Mapping[str, Any] | None,
    error: str | None,
    metadata_path: str | Path | None,
) -> int:
    records: list[dict[str, Any]] = [
        {
            "record_type": "official_v4_rebuild_run",
            "git_commit": _git_commit(),
            "source_probe": source_probe,
            "summary": rebuild.get("summary") if rebuild else None,
            "metadata_path": str(metadata_path) if metadata_path else None,
            "error": error,
        }
    ]
    if rebuild is not None:
        provenance = rebuild.get("provenance", {})
        for key in (
            "spot_archive_records",
            "spot_rest_gap_records",
            "mark_archive_records",
            "funding_archive_records",
        ):
            for payload in provenance.get(key, []):
                records.append(
                    {
                        "record_type": "official_v4_rebuild_source",
                        "source_group": key,
                        "payload": payload,
                    }
                )
        for gap in provenance.get("spot_gap_summary", {}).get("gap_records", []):
            records.append(
                {
                    "record_type": "official_v4_rebuild_gap",
                    "payload": gap,
                }
            )
        for quarantine in provenance.get("spot_off_grid_quarantine", []):
            records.append(
                {
                    "record_type": "official_v4_rebuild_off_grid_quarantine",
                    "payload": quarantine,
                }
            )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )
    return len(records)


def _render_report(
    *,
    source_probe: Mapping[str, Any],
    rebuild: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    cache_dir: str | Path,
    cache_tag: str,
    ledger_path: str | Path,
    error: str | None,
) -> str:
    lines = [
        "# Official-source schema-v4 cache rebuild",
        "",
        "This report reads no model results. It uses only Binance-owned sources and never interpolates missing bars.",
        "",
        f"- Cache output directory: `{cache_dir}`",
        f"- Cache tag: `{cache_tag}`",
        f"- Scope: `[2018-01-01, 2024-01-01)` / `{source_probe.get('scope', {}).get('interval')}`",
        f"- Source probe status: **{str(source_probe.get('status', 'unknown')).upper()}**",
        f"- Source ledger: `{ledger_path}`",
        "",
        "## Official source probe",
        "",
        "| Source | Probe responses | HTTP 200 | HTTP 404 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for source, payload in (source_probe.get("sources") or {}).items():
        lines.append(
            f"| `{source}` | {payload.get('probe_count')} | {payload.get('http_200_count')} | {payload.get('http_404_count')} |"
        )
    lines.extend(
        [
            "",
            f"UM mark/funding archives before `{EXTERNAL_ARCHIVE_START.date()}` are treated as unavailable and their masks are false; no future value is backfilled into that period.",
        ]
    )
    if error is not None:
        lines.extend(["", "## Rebuild status", "", f"- **BLOCKED**: `{error}`"])
        return "\n".join(lines) + "\n"
    assert rebuild is not None
    summary = rebuild["summary"]
    lines.extend(
        [
            "",
            "## Rebuild status",
            "",
            f"- Status: **{str(summary.get('status')).upper()}**",
            f"- Expected full-grid bars: `{summary.get('scope_expected_bars')}`",
            f"- Observed Spot bars: `{summary.get('spot_observed_bars')}`",
            f"- REST-recovered Spot bars: `{summary.get('rest_recovered_bars')}`",
            f"- Unresolved Spot bars: `{summary.get('spot_unresolved_bars')}`",
            f"- Quarantined off-grid Spot bars: `{summary.get('quarantined_off_grid_spot_bars', 0)}`",
            f"- Computed feature rows: `{summary.get('feature_rows')}`",
            f"- Metadata schema: `{metadata.get('schema_version') if metadata else None}`",
            f"- Schema digest: `{metadata.get('schema_digest') if metadata else None}`",
            f"- Source/provenance digest: `{metadata.get('source_provenance_digest') if metadata else None}`",
            f"- Feature content digest: `{metadata.get('content_digests', {}).get('features') if metadata else None}`",
            f"- Returns content digest: `{metadata.get('content_digests', {}).get('returns') if metadata else None}`",
            f"- Availability content digest: `{metadata.get('content_digests', {}).get('availability') if metadata else None}`",
            "",
            "The v4 body excludes unresolved Spot rows; the separate full-grid sidecar marks them `spot_bar_observed=false`. The 18 bars recoverable by official REST are included only when the raw Spot merge and causal feature computation succeed. External masks remain metadata and are not mixed into model inputs.",
            "",
            "No model result was read and no v3 file was overwritten.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild a future schema-v4 cache from official Binance sources",
        allow_abbrev=False,
    )
    parser.add_argument("--config", default="configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml")
    parser.add_argument("--cache-dir", default=None, help="output directory for generated v4 parquet files")
    parser.add_argument("--cache-tag", default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--ledger", default="docs/data_quality_v4_rebuild_2018_2024.jsonl")
    parser.add_argument("--report", default="docs/data_quality_v4_rebuild_2018_2024.md")
    parser.add_argument("--metadata-out", default="docs/data_quality_v4_rebuild_2018_2024_metadata.json")
    parser.add_argument(
        "--probe-month",
        action="append",
        dest="probe_months",
        default=None,
        help="representative YYYY-MM probe month (repeatable; defaults to 2018-01, 2019-12, 2020-01)",
    )
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="return success after writing an explicit sidecar for unresolved Spot gaps",
    )
    parser.add_argument(
        "--allow-off-grid-quarantine",
        action="store_true",
        help="quarantine official Spot rows outside the configured grid with full provenance; never remap them",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        run_cfg = config.get("run", {})
        data_cfg = config.get("data", {})
        normalization_cfg = config.get("normalization", {})
        symbol = str(data_cfg.get("symbol", DEFAULT_SYMBOL))
        interval = str(data_cfg.get("interval", DEFAULT_INTERVAL))
        start = str(run_cfg.get("start", DEFAULT_START.date()))
        end = str(run_cfg.get("end", DEFAULT_END.date()))
        if start != str(DEFAULT_START.date()) or end != str(DEFAULT_END.date()):
            raise OfficialSourceError("rebuild scope is restricted to [2018-01-01, 2024-01-01)")
        zscore_window_days = int(normalization_cfg.get("zscore_window_days", 60))
        cache_dir = Path(args.cache_dir or config.get("logging", {}).get("cache_dir", "checkpoints/data_cache"))
        cache_tag = args.cache_tag or (
            f"{symbol}_{interval}_{start}_{end}_z{zscore_window_days}_v4_official"
        )
        probe = probe_official_sources(
            symbol=symbol,
            interval=interval,
            months=args.probe_months or ["2018-01", "2019-12", "2020-01"],
            timeout=args.timeout,
        )
        rebuild = rebuild_official_v4_frames(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            zscore_window_days=zscore_window_days,
            source_probe=probe,
            timeout=args.timeout,
            allow_off_grid_quarantine=args.allow_off_grid_quarantine,
        )
        metadata = write_cache_v4(
            rebuild["features"],
            rebuild["returns"],
            rebuild["availability"],
            source_provenance=rebuild["provenance"],
            cache_dir=cache_dir,
            cache_tag=cache_tag,
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            parameters={
                "symbol": symbol,
                "interval": interval,
                "start": start,
                "end": end,
                "zscore_window_days": zscore_window_days,
                "extra_series_mode": str(data_cfg.get("extra_series_mode", "derived")),
                "extra_series_include": sorted(
                    str(name) for name in (data_cfg.get("extra_series_include") or [])
                ),
                "include_funding": True,
                "include_oi": False,
                "include_mark": True,
            },
        )
        _json_write(metadata, args.metadata_out)
        error = None
    except (OSError, KeyError, TypeError, ValueError, OfficialSourceError) as exc:
        probe = locals().get("probe") or {
            "status": "blocked",
            "scope": {"symbol": DEFAULT_SYMBOL, "interval": DEFAULT_INTERVAL},
            "sources": {},
        }
        rebuild = locals().get("rebuild")
        metadata = locals().get("metadata")
        error = f"{type(exc).__name__}: {exc}"
        print(f"[v4-rebuild] BLOCKED: {error}")
        _write_ledger(
            path=args.ledger,
            source_probe=probe,
            rebuild=rebuild,
            error=error,
            metadata_path=args.metadata_out if metadata else None,
        )
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            _render_report(
                source_probe=probe,
                rebuild=rebuild,
                metadata=metadata,
                cache_dir=locals().get("cache_dir", args.cache_dir or ""),
                cache_tag=locals().get("cache_tag", args.cache_tag or ""),
                ledger_path=args.ledger,
                error=error,
            ),
            encoding="utf-8",
        )
        return 2

    record_count = _write_ledger(
        path=args.ledger,
        source_probe=probe,
        rebuild=rebuild,
        error=error,
        metadata_path=args.metadata_out,
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        _render_report(
            source_probe=probe,
            rebuild=rebuild,
            metadata=metadata,
            cache_dir=cache_dir,
            cache_tag=cache_tag,
            ledger_path=args.ledger,
            error=error,
        ),
        encoding="utf-8",
    )
    summary = rebuild["summary"]
    print(f"[v4-rebuild] status: {summary['status']}")
    print(f"[v4-rebuild] full-grid bars: {summary['scope_expected_bars']}")
    print(f"[v4-rebuild] observed Spot bars: {summary['spot_observed_bars']}")
    print(f"[v4-rebuild] REST-recovered Spot bars: {summary['rest_recovered_bars']}")
    print(f"[v4-rebuild] unresolved Spot bars: {summary['spot_unresolved_bars']}")
    print(f"[v4-rebuild] feature rows: {summary['feature_rows']}")
    print(f"[v4-rebuild] metadata: {args.metadata_out}")
    print(f"[v4-rebuild] ledger records: {record_count} -> {args.ledger}")
    print(f"[v4-rebuild] report: {args.report}")
    if summary["spot_unresolved_bars"] and not args.allow_unresolved:
        print("[v4-rebuild] unresolved Spot bars remain; use --allow-unresolved only with explicit sidecar policy")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
