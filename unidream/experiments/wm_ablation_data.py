"""Reproducible Binance inputs for the isolated WM indicator ablation.

Rows use UTC bar OPEN timestamps. Candle values become usable only at
``decision_ts = bar_open + 15 minutes``. Metrics and funding use records no
later than bar OPEN, a conservative 15-minute delay before that decision.
Their actual historical publication timestamps are unknown. Acquisition
metadata and archive Last-Modified dates are not live availability evidence.

Missing candles, stale observations, invalid rows, and conflicting duplicate
timestamps remain missing. No price interpolation or zero imputation occurs.
This module downloads data only; it neither trains nor examines predictions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import threading
import time
from typing import Any
import zipfile

import numpy as np
import pandas as pd
import requests

from unidream.data.official_v4_sources import assert_official_url


ROOT = Path("codex_outputs/wm_technical_ablation_20261109/data")
START = pd.Timestamp("2020-09-01", tz="UTC")
END = pd.Timestamp("2026-11-01", tz="UTC")
WARMUP_START = pd.Timestamp("2020-08-01", tz="UTC")
BAR = pd.Timedelta(minutes=15)
KLINE_FIELDS = (
    "open", "high", "low", "close", "volume", "quote_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote",
)
METRIC_FIELDS = (
    "sum_open_interest", "sum_open_interest_value",
    "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
)
_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class ArchiveJob:
    source: str
    period: str
    url: str


def _now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def archive_jobs() -> list[ArchiveJob]:
    jobs = []
    months = pd.date_range(WARMUP_START, END, freq="MS", inclusive="left")
    for month in months:
        period = month.strftime("%Y-%m")
        for source in ("spot", "perp", "mark", "index", "premium", "funding"):
            if source != "spot" and month < START:
                continue
            if source == "spot":
                stem = f"spot/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-{period}"
            elif source == "funding":
                stem = f"futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{period}"
            else:
                endpoint = {"perp": "klines", "mark": "markPriceKlines", "index": "indexPriceKlines", "premium": "premiumIndexKlines"}[source]
                stem = f"futures/um/monthly/{endpoint}/BTCUSDT/15m/BTCUSDT-15m-{period}"
            jobs.append(ArchiveJob(source, period, f"https://data.binance.vision/data/{stem}.zip"))
    for day in pd.date_range(START, END, freq="D", inclusive="left"):
        period = day.strftime("%Y-%m-%d")
        jobs.append(ArchiveJob("metrics", period, f"https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-{period}.zip"))
    return jobs


def _request(url: str, attempts: list[dict[str, Any]]) -> requests.Response:
    assert_official_url(url, archive=True)
    if not hasattr(_THREAD_LOCAL, "session"):
        _THREAD_LOCAL.session = requests.Session()
    last_error: Exception | None = None
    for attempt in range(4):
        started = _now()
        try:
            response = _THREAD_LOCAL.session.get(url, timeout=(15, 60))
            assert_official_url(response.url, archive=True)
            attempts.append({
                "url": url, "final_url": response.url, "request_started_utc": started,
                "response_received_utc": _now(), "http_status": response.status_code,
                "bytes": len(response.content), "sha256": _sha(response.content),
                "last_modified": response.headers.get("Last-Modified"),
                "etag": response.headers.get("ETag"),
            })
            if response.status_code not in {408, 418, 429} and response.status_code < 500:
                return response
            last_error = RuntimeError(f"HTTP {response.status_code}: {url}")
        except requests.RequestException as exc:
            last_error = exc
            attempts.append({"url": url, "request_started_utc": started, "error": str(exc)})
        if attempt < 3:
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(str(last_error))


def acquire_archive(job: ArchiveJob, root: Path) -> dict[str, Any]:
    """Cache ZIP and supplied checksum, retaining request evidence across runs."""
    folder = root / "archives" / job.source
    folder.mkdir(parents=True, exist_ok=True)
    archive_path = folder / Path(job.url).name
    checksum_path = archive_path.with_suffix(".zip.CHECKSUM")
    metadata_path = folder / f"{job.period}.json"
    previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if previous.get("checksum_verified") and archive_path.exists() and checksum_path.exists():
        actual = _sha(archive_path.read_bytes())
        expected = re.search(r"\b[0-9a-fA-F]{64}\b", checksum_path.read_text())
        if expected and actual == expected.group().lower() == previous.get("sha256"):
            return previous
    # A recorded historical 404 is reproducible and does not require re-probing
    # every daily object on a restart. Delete its metadata to deliberately retry.
    if previous.get("http_status") == 404:
        return previous
    record: dict[str, Any] = {
        "source": job.source, "period": job.period, "url": job.url,
        "archive_path": str(archive_path), "historical_publication_timestamp": None,
        "availability_evidence": "archive retrieval only; not live publication proof",
        "attempts": previous.get("attempts", []), "checksum_verified": False,
    }
    try:
        response = _request(job.url, record["attempts"])
        record.update(http_status=response.status_code, final_url=response.url, sha256=_sha(response.content), bytes=len(response.content))
        if response.status_code == 200:
            temporary = archive_path.with_suffix(".zip.part")
            temporary.write_bytes(response.content)
            temporary.replace(archive_path)
            checksum_response = _request(job.url + ".CHECKSUM", record["attempts"])
            record["checksum_http_status"] = checksum_response.status_code
            if checksum_response.status_code == 200:
                checksum_path.write_bytes(checksum_response.content)
                matches = re.findall(r"\b[0-9a-fA-F]{64}\b", checksum_response.text)
                record["supplied_sha256"] = matches[0].lower() if matches else None
                record["checksum_verified"] = record["supplied_sha256"] == record["sha256"]
                if not record["checksum_verified"]:
                    record["error"] = "supplied checksum does not match archive"
            else:
                record["error"] = "archive lacks a successfully retrieved supplied checksum"
        elif response.status_code != 404:
            record["error"] = f"unexpected HTTP {response.status_code}"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["retrieval_completed_utc"] = _now()
    _write_json(metadata_path, record)
    return record


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Remove identical copies; quarantine every row at conflicting timestamps."""
    original = len(frame)
    ordered = frame.sort_index(kind="stable")
    identical = ordered.reset_index().drop_duplicates().set_index(ordered.index.name or "index")
    identical.index.name = ordered.index.name
    conflict_mask = identical.index.duplicated(keep=False)
    conflicts = identical.loc[conflict_mask]
    clean = identical.loc[~conflict_mask]
    return clean, conflicts, {
        "input_rows": original, "identical_duplicates_removed": original - len(identical),
        "conflicting_rows_quarantined": len(conflicts),
        "conflicting_timestamps_quarantined": conflicts.index.nunique(),
    }


def parse_archive(job: ArchiveJob, payload: bytes) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.namelist()
        if len(members) != 1 or not members[0].endswith(".csv"):
            raise ValueError(f"expected exactly one CSV: {members}")
        raw = archive.read(members[0])
    first = raw.splitlines()[0].split(b",")[0]
    header = not first.lstrip(b"+-").isdigit()
    table = pd.read_csv(io.BytesIO(raw), header=0 if header else None)
    metadata: dict[str, Any] = {"member": members[0], "member_sha256": _sha(raw), "header_present": header}
    if job.source == "metrics":
        expected = ["create_time", "symbol", *METRIC_FIELDS]
        if list(table.columns) != expected or not table["symbol"].eq("BTCUSDT").all():
            raise ValueError(f"unexpected metrics schema: {list(table.columns)}")
        timestamps = pd.to_datetime(table["create_time"], utc=True)
        frame = table[list(METRIC_FIELDS)].apply(pd.to_numeric, errors="raise")
        frame.columns = [f"metrics_{c}" for c in frame.columns]
    elif job.source == "funding":
        if list(table.columns) != ["calc_time", "funding_interval_hours", "last_funding_rate"]:
            raise ValueError(f"unexpected funding schema: {list(table.columns)}")
        timestamps = pd.to_datetime(table["calc_time"], unit="ms", utc=True)
        frame = table[["last_funding_rate"]].rename(columns={"last_funding_rate": "funding_rate"})
        frame = frame.apply(pd.to_numeric, errors="raise")
    else:
        if table.shape[1] != 12:
            raise ValueError(f"unexpected kline width: {table.shape[1]}")
        table.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore"]
        table = table.apply(pd.to_numeric, errors="raise")
        unit = "us" if table["open_time"].median() > 100_000_000_000_000 else "ms"
        timestamps = pd.to_datetime(table["open_time"], unit=unit, utc=True)
        close_timestamps = pd.to_datetime(table["close_time"], unit=unit, utc=True)
        good_time = timestamps.eq(timestamps.dt.floor("15min")) & (close_timestamps < timestamps + BAR) & (close_timestamps >= timestamps + BAR - pd.Timedelta(milliseconds=1))
        if job.source in {"spot", "perp"}:
            frame = table[list(KLINE_FIELDS)].copy()
            valid_candle = (table[["open", "high", "low", "close"]] > 0).all(axis=1)
            valid_candle &= table["low"].le(table[["open", "close"]].min(axis=1)) & table["high"].ge(table[["open", "close"]].max(axis=1))
            valid_candle &= table[["volume", "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote"]].ge(0).all(axis=1)
            frame.columns = [f"{job.source}_{c}" for c in frame.columns]
        else:
            frame = table[["close"]].rename(columns={"close": f"{job.source}_close"})
            valid_candle = pd.Series(True, index=table.index) if job.source == "premium" else table["close"].gt(0)
        invalid = ~(good_time & valid_candle)
        metadata["invalid_candle_rows"] = int(invalid.sum())
        frame.loc[invalid] = np.nan
        metadata["timestamp_unit"] = unit
    frame.index = pd.DatetimeIndex(timestamps, name="bar_open")
    finite_cells = np.isfinite(frame.to_numpy(dtype=float))
    metadata["nonfinite_rows"] = int((~finite_cells.all(axis=1)).sum())
    metadata["nonfinite_cells_by_column"] = dict(zip(frame.columns, (~finite_cells).sum(axis=0).tolist(), strict=True))
    # Some 2022 metric archives have OI and global long/short values while
    # toptrader columns are entirely absent. Preserve each observed field.
    frame = frame.where(finite_cells)
    period_start = pd.Timestamp(job.period, tz="UTC")
    period_end = period_start + (pd.Timedelta(days=1) if job.source == "metrics" else pd.offsets.MonthBegin(1))
    outside_period = (frame.index < period_start) | (frame.index >= period_end)
    metadata["outside_requested_period_rows"] = int(outside_period.sum())
    frame = frame.loc[~outside_period]
    frame, conflicts, dedup = deduplicate(frame)
    metadata.update(dedup, parsed_rows=len(frame), parsed_first_timestamp=str(frame.index.min()), parsed_last_timestamp=str(frame.index.max()))
    return frame, metadata, conflicts


def causal_asof(events: pd.DataFrame, grid: pd.DatetimeIndex, max_age: pd.Timedelta) -> tuple[pd.DataFrame, pd.Series]:
    """Use events <= bar OPEN, expiring them rather than filling long outages."""
    result = pd.DataFrame(index=grid, columns=events.columns, dtype=float)
    times = pd.Series(pd.NaT, index=grid, dtype="datetime64[ns, UTC]")
    if events.empty:
        return result, times
    events = events.sort_index()
    positions = events.index.get_indexer(grid, method="pad")
    found = positions >= 0
    if found.any():
        chosen = events.index.take(positions[found])
        fresh = (grid[found] - chosen) <= max_age
        rows = np.flatnonzero(found)[fresh]
        result.iloc[rows] = events.iloc[positions[rows]].to_numpy()
        times.iloc[rows] = events.index.take(positions[rows])
    return result, times


def missing_intervals(mask: pd.Series, name: str) -> list[dict[str, Any]]:
    """List exact contiguous unavailable 15-minute runs on the full grid."""
    missing = ~mask.astype(bool)
    groups = missing.ne(missing.shift()).cumsum()
    intervals = []
    for _, run in missing.groupby(groups):
        if not bool(run.iloc[0]):
            continue
        intervals.append({
            "availability_mask": name, "start_bar_open_utc": run.index[0].isoformat(),
            "end_exclusive_bar_open_utc": (run.index[-1] + BAR).isoformat(),
            "missing_bars": len(run), "duration_hours": len(run) / 4,
        })
    return intervals


def assemble(root: Path, jobs: list[ArchiveJob], records: list[dict[str, Any]]) -> dict[str, Any]:
    frames: dict[str, list[pd.DataFrame]] = {s: [] for s in ("spot", "perp", "mark", "index", "premium", "funding", "metrics")}
    parsing: list[dict[str, Any]] = []
    conflict_frames = []
    for job, record in zip(jobs, records, strict=True):
        if not record.get("checksum_verified"):
            continue
        try:
            frame, metadata, conflicts = parse_archive(job, Path(record["archive_path"]).read_bytes())
            frames[job.source].append(frame)
            parsing.append({"source": job.source, "period": job.period, **metadata})
            if not conflicts.empty:
                conflict_frames.append(conflicts.assign(source=job.source, archive=job.period))
        except Exception as exc:
            parsing.append({"source": job.source, "period": job.period, "error": f"{type(exc).__name__}: {exc}"})
    grid = pd.date_range(WARMUP_START, END, freq="15min", inclusive="left", name="bar_open")
    raw = pd.DataFrame(index=grid)
    availability = pd.DataFrame({"decision_ts": grid + BAR}, index=grid)
    source_audits = {}
    for source, parts in frames.items():
        if not parts:
            columns = [f"{source}_{c}" for c in KLINE_FIELDS] if source in {"spot", "perp"} else (["funding_rate"] if source == "funding" else ([f"metrics_{c}" for c in METRIC_FIELDS] if source == "metrics" else [f"{source}_close"]))
            combined = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="bar_open"), columns=columns, dtype=float)
        else:
            combined, conflicts, cross_dedup = deduplicate(pd.concat(parts))
            if not conflicts.empty:
                conflict_frames.append(conflicts.assign(source=source, archive="cross_archive"))
            source_audits[source] = cross_dedup
        if source in {"metrics", "funding"}:
            raw_events_path = root / "base" / f"{source}_events.parquet"
            combined.to_parquet(raw_events_path)
            max_age = pd.Timedelta(minutes=30) if source == "metrics" else pd.Timedelta(hours=24)
            aligned, event_times = causal_asof(combined, grid, max_age)
            availability[f"{source}_source_ts"] = event_times
            availability[f"{source}_age_seconds"] = (grid.to_series() - event_times).dt.total_seconds()
        else:
            aligned = combined.reindex(grid)
        raw = raw.join(aligned)
        availability[f"{source}_available"] = aligned.notna().all(axis=1)
    # Analysis masks exclude the warm-up but preserve every bar for shared
    # context construction. Missing market rows never alter the spot calendar.
    availability["in_analysis_period"] = grid >= START
    availability["all_market_available"] = availability[[f"{s}_available" for s in frames]].all(axis=1)
    core_metrics = ["metrics_sum_open_interest", "metrics_sum_open_interest_value", "metrics_count_long_short_ratio"]
    availability["metrics_core_available"] = raw[core_metrics].notna().all(axis=1)
    availability["metrics_core_valid_available"] = raw[core_metrics].gt(0).all(axis=1)
    availability["core_market_valid_available"] = availability[[f"{s}_available" for s in frames if s != "metrics"]].all(axis=1) & availability["metrics_core_valid_available"]
    for column in raw.columns:
        availability[f"{column}_available"] = raw[column].notna()
    analysis = raw.loc[START:END - BAR]
    raw.to_parquet(root / "base" / "raw_market.parquet")
    availability.to_parquet(root / "base" / "availability.parquet")
    if conflict_frames:
        pd.concat(conflict_frames, sort=False).to_parquet(root / "base" / "conflicting_duplicates.parquet")
    analysis_availability = availability.loc[START:END - BAR]
    interval_masks = [f"{s}_available" for s in frames] + ["metrics_core_available", "metrics_core_valid_available", "core_market_valid_available"]
    gaps = [gap for name in interval_masks for gap in missing_intervals(analysis_availability[name], name)]
    pd.DataFrame(gaps).to_csv(root / "data_gap_intervals.csv", index=False)
    summary = {
        "created_utc": _now(), "analysis_start_inclusive": str(START), "analysis_end_exclusive": str(END),
        "warmup_start_inclusive": str(WARMUP_START), "index_semantics": "15-minute candle OPEN, UTC",
        "decision_semantics": "bar OPEN + 15 minutes, after source candle close",
        "asof_semantics": {"metrics": "latest source timestamp <= bar OPEN; max age 30 minutes", "funding": "latest source timestamp <= bar OPEN; max age 24 hours"},
        "historical_publication_timestamps": "unknown; conservative source-time lags are assumed, not verified live availability",
        "rows_with_warmup": len(raw), "analysis_rows": len(analysis), "columns": list(raw.columns),
        "analysis_missing_fraction_by_column": analysis.isna().mean().to_dict(),
        "analysis_availability_fraction": availability.loc[START:END - BAR].filter(like="available").mean().to_dict(),
        "analysis_source_availability_by_year": {
            str(year): group[interval_masks].mean().to_dict()
            for year, group in availability.loc[START:END - BAR].groupby(availability.loc[START:END - BAR].index.year)
        },
        "longest_missing_interval_by_mask": {
            name: max((gap for gap in gaps if gap["availability_mask"] == name), key=lambda gap: gap["missing_bars"], default=None)
            for name in interval_masks
        },
        "analysis_nonpositive_metric_counts": analysis.filter(like="metrics_").le(0).sum().to_dict(),
        "archive_jobs": len(jobs), "verified_archives": sum(bool(r.get("checksum_verified")) for r in records),
        "http_status_counts": pd.Series([str(r.get("http_status", "error")) for r in records]).value_counts().to_dict(),
        "acquisition_errors": [{"source": r["source"], "period": r["period"], "error": r.get("error")} for r in records if r.get("error")],
        "missing_archives": [{"source": r["source"], "period": r["period"]} for r in records if r.get("http_status") == 404],
        "parse_errors": [p for p in parsing if p.get("error")], "cross_archive_deduplication": source_audits,
        "invalid_candle_rows": sum(p.get("invalid_candle_rows", 0) for p in parsing),
        "identical_duplicate_rows_removed": sum(p.get("identical_duplicates_removed", 0) for p in parsing),
        "conflicting_duplicate_timestamps": sum(p.get("conflicting_timestamps_quarantined", 0) for p in parsing),
        "raw_market_sha256": _sha((root / "base" / "raw_market.parquet").read_bytes()),
    }
    _write_json(root / "parse_ledger.json", parsing)
    _write_json(root / "acquisition_ledger.json", records)
    _write_json(root / "data_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "base").mkdir(exist_ok=True)
    jobs = archive_jobs()
    records: list[dict[str, Any] | None] = [None] * len(jobs)
    if args.assemble_only:
        for i, job in enumerate(jobs):
            records[i] = json.loads((args.output / "archives" / job.source / f"{job.period}.json").read_text())
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(acquire_archive, job, args.output): i for i, job in enumerate(jobs)}
            for done, future in enumerate(as_completed(futures), start=1):
                i = futures[future]
                records[i] = future.result()
                if done % 100 == 0 or done == len(jobs):
                    known = [r for r in records if r is not None]
                    print(json.dumps({"completed": done, "total": len(jobs), "verified": sum(bool(r.get("checksum_verified")) for r in known), "missing": sum(r.get("http_status") == 404 for r in known), "time_utc": _now()}), flush=True)
    summary = assemble(args.output, jobs, records)  # type: ignore[arg-type]
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
