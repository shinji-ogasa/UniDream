"""Audited monthly Spot BTCUSDT 15-minute archive acquisition.

This module is deliberately data-only.  It downloads Binance's official Spot
monthly kline archives through :func:`download_d1_kline_month`, which performs
the strict archive parser and verifies the official ``.CHECKSUM`` sidecar.
No model, prediction, trading result, or legacy cache is read here.

The acquisition is resumable at month granularity.  A successfully parsed
month is written as a small Parquet checkpoint together with a JSON digest
sidecar.  A checkpoint is reused only when both the Parquet file digest and
the parsed-frame digest match its sidecar.  ZIP payloads are never retained.

The final artifact is a complete 15-minute *bar-open* grid.  Missing source
rows are retained as NaN and are represented by a separate boolean
``spot_bar_observed`` availability artifact.  Archive/download timestamps are
kept separate from live exchange observation timestamps; this acquisition
does not make a live-causality claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from unidream.data.d1_signed_flow import (
    D1_INTERVAL,
    _frame_digest,
    classify_archive_revisions,
    download_d1_kline_month,
)


DEFAULT_OUTPUT_PATH = Path(
    "/Users/sophie/Documents/UniDream/.worktrees/alpha-dd-goal/checkpoints/"
    "alpha_dd_data/spot_15m.parquet"
)
DEFAULT_START_MONTH = "2018-01"
DEFAULT_END_MONTH = "2026-08"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = D1_INTERVAL
MAX_DOWNLOAD_WORKERS = 6

SOURCE = "spot_klines"
OUTPUT_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "n_trades",
    "taker_buy_base",
    "taker_buy_quote",
)
AVAILABILITY_COLUMN = "spot_bar_observed"
SCHEMA_VERSION = 1


class AcquisitionError(RuntimeError):
    """Raised after source errors have been written to the append-only ledger."""

    def __init__(self, message: str, *, result: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.result = dict(result or {})


@dataclass
class MonthResult:
    """In-memory result for one requested archive month."""

    month: str
    frame: pd.DataFrame
    record: dict[str, Any]
    attempted: bool = True
    from_cache: bool = False
    fatal: bool = False


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _month_start(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    # Avoid pandas' timezone-dropping Period conversion.
    return pd.Timestamp(year=timestamp.year, month=timestamp.month, day=1, tz="UTC")


def month_values(
    start_month: str | pd.Timestamp = DEFAULT_START_MONTH,
    end_month: str | pd.Timestamp = DEFAULT_END_MONTH,
) -> list[pd.Timestamp]:
    """Return month starts from ``start_month`` through ``end_month`` inclusive."""
    start = _month_start(start_month)
    end = _month_start(end_month)
    if end < start:
        raise ValueError("end_month must be on or after start_month")
    values: list[pd.Timestamp] = []
    cursor = start
    while cursor <= end:
        values.append(cursor)
        cursor = (cursor + pd.offsets.MonthBegin(1)).normalize()
    return values


def expected_bar_grid(
    start_month: str | pd.Timestamp = DEFAULT_START_MONTH,
    end_month: str | pd.Timestamp = DEFAULT_END_MONTH,
) -> pd.DatetimeIndex:
    """Return the UTC 15-minute bar-open grid for the inclusive month range."""
    start = _month_start(start_month)
    end_exclusive = (_month_start(end_month) + pd.offsets.MonthBegin(1)).normalize()
    return pd.date_range(
        start=start,
        end=end_exclusive - pd.Timedelta(minutes=15),
        freq="15min",
        tz="UTC",
        name="bar_open_ts",
    )


def artifact_paths(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    availability_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    sha_sidecar: str | Path | None = None,
    monthly_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Resolve the final, availability, ledger, digest, and monthly paths."""
    output = Path(output_path)
    stem = output.stem
    return {
        "output": output,
        "availability": Path(availability_path)
        if availability_path is not None
        else output.with_name(f"{stem}_availability.parquet"),
        "ledger": Path(ledger_path)
        if ledger_path is not None
        else output.with_name(f"{stem}.ledger.jsonl"),
        "sha_sidecar": Path(sha_sidecar)
        if sha_sidecar is not None
        else output.with_name(f"{stem}.sha256.json"),
        "sha_text": output.with_name(f"{output.name}.sha256"),
        "monthly": Path(monthly_dir)
        if monthly_dir is not None
        else output.with_name(f"{stem}_monthly"),
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, _json_bytes(value) + b"\n")


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=True)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _append_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    records = list(records)
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    dict(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_json_default,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _month_label(month: pd.Timestamp) -> str:
    return f"{month.year:04d}-{month.month:02d}"


def _empty_record(month: str, *, error: str) -> dict[str, Any]:
    return {
        "record_type": "d1_archive_download",
        "source": SOURCE,
        "symbol": DEFAULT_SYMBOL,
        "interval": DEFAULT_INTERVAL,
        "month": month,
        "archive_url": None,
        "archive_published_ts": None,
        "collector_observed_ts": None,
        "exchange_available_ts": None,
        "live_causal_eligible": False,
        "checksum_verified": False,
        "archive_revision_id": None,
        "http_status": None,
        "error": error,
    }


def _coerce_source_frame(
    frame: pd.DataFrame,
    *,
    month: pd.Timestamp,
    symbol: str,
    interval: str,
) -> pd.DataFrame:
    """Validate the downloader output and retain only the requested kline fields."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("downloader did not return a pandas DataFrame")
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"))
    candidate = frame.copy()
    # ``download_d1_kline_month`` intentionally retains close-time metadata;
    # the final Spot artifact is bar-open data only.
    if "bar_close_ts" in candidate.columns:
        candidate = candidate.drop(columns=["bar_close_ts"])
    if set(candidate.columns) != set(OUTPUT_COLUMNS):
        raise ValueError(
            f"{symbol} {interval} {month:%Y-%m} parser schema mismatch: "
            f"expected {list(OUTPUT_COLUMNS)}, got {list(candidate.columns)}"
        )
    candidate = candidate.loc[:, list(OUTPUT_COLUMNS)]
    if not isinstance(candidate.index, pd.DatetimeIndex):
        raise ValueError("downloader index is not a DatetimeIndex")
    if candidate.index.tz is None:
        raise ValueError("downloader index is timezone-naive; UTC is mandatory")
    candidate.index = candidate.index.tz_convert("UTC")
    candidate.index.name = "bar_open_ts"
    if not candidate.index.is_unique or not candidate.index.is_monotonic_increasing:
        raise ValueError("downloader index must be sorted and unique")
    next_month = (month + pd.offsets.MonthBegin(1)).normalize()
    if ((candidate.index < month) | (candidate.index >= next_month)).any():
        raise ValueError(f"downloader returned rows outside requested month {month:%Y-%m}")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    if ((candidate.index - epoch) % pd.Timedelta(minutes=15) != pd.Timedelta(0)).any():
        raise ValueError("downloader bar-open timestamps are not aligned to 15m")
    try:
        numeric = candidate.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"downloader kline fields are not numeric: {exc}") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("downloader kline fields contain non-finite values")
    if (numeric < 0).any():
        raise ValueError("downloader kline fields contain negative values")
    if (candidate[["open", "high", "low", "close"]].to_numpy(dtype=float) <= 0).any():
        raise ValueError("downloader kline fields contain non-positive OHLC prices")
    trades = candidate["n_trades"].to_numpy(dtype=float)
    if not np.equal(trades, np.floor(trades)).all():
        raise ValueError("downloader trade count is not integral")
    # Stable final schema: NaN is reserved for missing grid rows, and all nine
    # fields therefore use float64 in both monthly and assembled artifacts.
    return candidate.astype("float64")


def _download_one(
    month: pd.Timestamp,
    *,
    symbol: str,
    interval: str,
    timeout: float,
    downloader: Callable[..., tuple[pd.DataFrame, Mapping[str, Any]]],
) -> MonthResult:
    label = _month_label(month)
    try:
        download_kwargs: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "month": label,
            "raw_dir": None,
            "timeout": timeout,
        }
        if downloader is download_d1_kline_month:
            # Binance Spot archives use microseconds from 2025-01 onward;
            # the parser detects that source/month unit while retaining its
            # historical millisecond strict default for other callers.
            download_kwargs.update(
                {
                    "quarantine_invalid_rows": True,
                    "timestamp_unit": "auto",
                }
            )
        frame, raw_record = downloader(SOURCE, **download_kwargs)
        record = dict(raw_record or {})
    except Exception as exc:  # worker errors must become ledger records
        record = _empty_record(
            label,
            error=f"{type(exc).__name__}: {exc}",
        )
        record.update({"symbol": symbol, "interval": interval})
        return MonthResult(label, pd.DataFrame(), record, fatal=True)

    record.setdefault("record_type", "d1_archive_download")
    record.setdefault("source", SOURCE)
    record.setdefault("symbol", symbol)
    record.setdefault("interval", interval)
    record.setdefault("month", label)
    record.setdefault("archive_published_ts", None)
    record.setdefault("collector_observed_ts", None)
    record.setdefault("exchange_available_ts", None)
    record.setdefault("live_causal_eligible", False)
    record.setdefault("checksum_verified", False)
    record["alpha_dd_requested_month"] = label
    record["raw_payload_retained"] = False
    record["timestamp_semantics"] = (
        "archive_published_ts/collector_observed_ts/exchange_available_ts are "
        "not live observation timestamps; live_causal_eligible remains false"
    )

    if record.get("source") != SOURCE:
        record["error"] = f"unexpected downloader source {record.get('source')!r}"
        return MonthResult(label, pd.DataFrame(), record, fatal=True)
    if str(record.get("month")) != label:
        record["error"] = f"downloader month mismatch: expected {label}, got {record.get('month')!r}"
        return MonthResult(label, pd.DataFrame(), record, fatal=True)

    # A frame is admissible only with an explicit boolean True checksum proof.
    # A 404 archive is a legitimate unavailable source gap; a 200 response
    # without a verified sidecar is a fail-closed checksum error.
    checksum_verified = record.get("checksum_verified") is True
    status = record.get("http_status")
    try:
        status_int = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_int = None
    if not checksum_verified:
        record["checksum_required"] = True
        if status_int not in (404, 410) and not record.get("error"):
            record["error"] = "checksum verification is mandatory for available archives"
        if status_int not in (404, 410):
            record["checksum_failure"] = True
            return MonthResult(label, pd.DataFrame(), record, fatal=True)
        return MonthResult(label, pd.DataFrame(), record, fatal=False)

    # A checksum-verified archive can still be rejected by the shared parser.
    # Only an explicitly classified timing rejection is an admissible source
    # gap.  Numeric/integrity/parser-structure failures remain fail-closed;
    # accepting those as gaps could hide a corrupt official archive.
    if frame.empty and record.get("error"):
        record["parse_failure"] = True
        if record.get("parser_error_kind") == "timing":
            record["strict_parser_rejected"] = True
            return MonthResult(label, pd.DataFrame(), record, fatal=False)
        record["parser_error_kind"] = record.get("parser_error_kind") or "integrity_or_structure"
        return MonthResult(label, pd.DataFrame(), record, fatal=True)

    try:
        checked = _coerce_source_frame(
            frame,
            month=month,
            symbol=symbol,
            interval=interval,
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["parse_failure"] = True
        return MonthResult(label, pd.DataFrame(), record, fatal=True)
    if checked.empty:
        record["error"] = record.get("error") or "checksum-verified archive produced no rows"
        record["parse_failure"] = True
        return MonthResult(label, pd.DataFrame(), record, fatal=True)
    record["parsed_output_rows"] = int(len(checked))
    record["parsed_output_frame_sha256"] = _frame_digest(checked)
    return MonthResult(label, checked, record, fatal=False)


def _cached_month(
    month: pd.Timestamp,
    *,
    monthly_dir: Path,
    symbol: str,
    interval: str,
) -> MonthResult | None:
    """Load a checkpoint only when its metadata and both digests verify."""
    label = _month_label(month)
    data_path = monthly_dir / f"{label}.parquet"
    metadata_path = monthly_dir / f"{label}.json"
    if not data_path.exists() and not metadata_path.exists():
        return None
    invalid_reason: str | None = None
    metadata: dict[str, Any] = {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            invalid_reason = "monthly checkpoint schema version mismatch"
        elif metadata.get("symbol") != symbol or metadata.get("interval") != interval:
            invalid_reason = "monthly checkpoint symbol/interval mismatch"
        elif metadata.get("month") != label:
            invalid_reason = "monthly checkpoint month mismatch"
        elif (metadata.get("source_record") or {}).get("compatibility_path"):
            # Older runner revisions normalized microsecond CSVs outside the
            # shared parser.  Rebuild those checkpoints once so the source
            # ledger contains the direct ms/us detection and original parser
            # quarantine evidence.
            invalid_reason = "legacy normalized-timestamp checkpoint requires refresh"
        elif not data_path.exists():
            invalid_reason = "monthly checkpoint Parquet file is missing"
        elif metadata.get("parquet_sha256") != sha256_file(data_path):
            invalid_reason = "monthly checkpoint Parquet SHA-256 mismatch"
        else:
            frame = pd.read_parquet(data_path)
            checked = _coerce_source_frame(
                frame,
                month=month,
                symbol=symbol,
                interval=interval,
            )
            if metadata.get("parsed_output_frame_sha256") != _frame_digest(checked):
                invalid_reason = "monthly checkpoint parsed-frame digest mismatch"
            elif int(metadata.get("parsed_output_rows", -1)) != len(checked):
                invalid_reason = "monthly checkpoint row count mismatch"
            else:
                source_record = dict(metadata.get("source_record") or {})
                source_record.setdefault("record_type", "d1_archive_download")
                source_record.setdefault("source", SOURCE)
                source_record["month"] = label
                source_record["symbol"] = symbol
                source_record["interval"] = interval
                source_record["cache_resumed"] = True
                source_record["cache_path"] = str(data_path)
                source_record["cache_parquet_sha256"] = metadata.get("parquet_sha256")
                source_record["raw_payload_retained"] = False
                return MonthResult(
                    label,
                    checked,
                    source_record,
                    attempted=False,
                    from_cache=True,
                )
    except Exception as exc:
        invalid_reason = f"{type(exc).__name__}: {exc}"

    record = {
        "record_type": "alpha_dd_cache_invalid",
        "source": SOURCE,
        "symbol": symbol,
        "interval": interval,
        "month": label,
        "cache_path": str(data_path),
        "cache_metadata_path": str(metadata_path),
        "error": invalid_reason or "monthly checkpoint is invalid",
        "checksum_verified": False,
        "raw_payload_retained": False,
    }
    # A caller can distinguish this warning from a source result and decide to
    # redownload the month.  Returning no MonthResult keeps the source result
    # append-only and avoids treating a stale cache as observed data.
    return MonthResult(label, pd.DataFrame(), record, attempted=False, fatal=False)


def _persist_month(
    result: MonthResult,
    *,
    monthly_dir: Path,
    symbol: str,
    interval: str,
) -> None:
    if result.frame.empty:
        return
    label = result.month
    data_path = monthly_dir / f"{label}.parquet"
    metadata_path = monthly_dir / f"{label}.json"
    frame = result.frame.loc[:, list(OUTPUT_COLUMNS)].copy()
    frame.index.name = "bar_open_ts"
    frame = frame.astype("float64")
    _atomic_write_parquet(frame, data_path)
    record = dict(result.record)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": "alpha_dd_spot_month_checkpoint",
        "symbol": symbol,
        "interval": interval,
        "month": label,
        "bar_time_semantics": "UTC bar_open_ts; 15-minute Binance completed-bar archive row",
        "columns": list(OUTPUT_COLUMNS),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "parsed_output_rows": int(len(frame)),
        "parsed_output_frame_sha256": _frame_digest(frame),
        "parquet_sha256": sha256_file(data_path),
        "timestamp_unit": record.get("timestamp_unit"),
        "quarantined_rows": int(record.get("quarantined_rows", 0) or 0),
        "checksum_verified": record.get("checksum_verified") is True,
        "archive_revision_id": record.get("archive_revision_id"),
        "source_record": record,
        "raw_payload_retained": False,
        "live_causal_eligible": False,
    }
    _atomic_write_json(metadata_path, metadata)


def _is_archive_unavailable(record: Mapping[str, Any]) -> bool:
    status = record.get("http_status")
    try:
        status_int = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_int = None
    if status_int in (404, 410):
        return True
    error = str(record.get("error") or "")
    return "HTTP 404" in error or "HTTP 410" in error


def _current_month(now: pd.Timestamp | None = None) -> pd.Timestamp:
    current = now or pd.Timestamp.now(tz="UTC")
    return _month_start(current)


def _mark_availability_statuses(
    results: list[MonthResult],
    *,
    now: pd.Timestamp | None,
) -> tuple[list[str], list[str], list[str]]:
    """Mark tail vs historical gaps without hiding any missing month."""
    available_indices = [idx for idx, result in enumerate(results) if not result.frame.empty]
    last_available = max(available_indices) if available_indices else -1
    historical: list[str] = []
    tail: list[str] = []
    fatal: list[str] = []
    current_month = _current_month(now)
    for idx, result in enumerate(results):
        record = result.record
        if not result.frame.empty:
            record["availability_status"] = "available"
            record["source_bar_observed"] = True
            continue
        if (
            result.fatal
            or record.get("checksum_failure")
            or (record.get("parse_failure") and not record.get("strict_parser_rejected"))
        ):
            record["availability_status"] = "error"
            fatal.append(result.month)
            continue
        # Explicitly not attempted records and 404/410 responses are gaps.  A
        # contiguous suffix is the current unavailable tail; a missing month
        # before a later available archive remains a historical gap.
        is_suffix = idx > last_available
        is_future_or_current = _month_start(result.month) >= current_month
        if is_suffix and (_is_archive_unavailable(record) or not result.attempted or is_future_or_current):
            record["availability_status"] = "unavailable_tail"
            tail.append(result.month)
        else:
            record["availability_status"] = "historical_gap"
            historical.append(result.month)
        record["source_bar_observed"] = False
    return historical, tail, fatal


def _build_grid_artifacts(
    results: list[MonthResult],
    *,
    start_month: str | pd.Timestamp,
    end_month: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = expected_bar_grid(start_month, end_month)
    output = pd.DataFrame(index=grid, columns=list(OUTPUT_COLUMNS), dtype="float64")
    availability = pd.DataFrame(
        {AVAILABILITY_COLUMN: pd.Series(False, index=grid, dtype="bool")},
        index=grid,
    )
    for result in results:
        if result.frame.empty:
            continue
        index = result.frame.index
        output.loc[index, list(OUTPUT_COLUMNS)] = result.frame.loc[index, list(OUTPUT_COLUMNS)].to_numpy()
        availability.loc[index, AVAILABILITY_COLUMN] = True
    output.index.name = "bar_open_ts"
    availability.index.name = "bar_open_ts"
    return output, availability


def _latest_archive_identity(results: Iterable[MonthResult]) -> dict[str, Any] | None:
    available = [result for result in results if not result.frame.empty]
    if not available:
        return None
    latest = max(available, key=lambda result: result.month)
    record = latest.record
    fields = (
        "month",
        "archive_url",
        "final_url",
        "archive_revision_id",
        "response_sha256",
        "checksum_expected_sha256",
        "checksum_verified",
        "archive_published_ts",
        "collector_observed_ts",
        "exchange_available_ts",
        "download_ts",
        "live_causal_eligible",
        "timestamp_unit",
        "quarantined_rows",
    )
    identity = {field: record.get(field) for field in fields}
    identity["parsed_output_rows"] = int(len(latest.frame))
    identity["cache_resumed"] = latest.from_cache
    return identity


def _record_for_ledger(result: MonthResult) -> dict[str, Any]:
    record = dict(result.record)
    # Internal status is intentionally represented as JSON evidence, not as a
    # private Python-only field.
    record["schema_version"] = SCHEMA_VERSION
    record["source"] = SOURCE
    record["month"] = result.month
    record["from_cache"] = result.from_cache
    record["attempted"] = result.attempted
    record["raw_payload_retained"] = False
    return record


def _run_record(
    *,
    output_path: Path,
    availability_path: Path,
    ledger_path: Path,
    sha_sidecar: Path,
    symbol: str,
    interval: str,
    start_month: str | pd.Timestamp,
    end_month: str | pd.Timestamp,
    output: pd.DataFrame,
    availability: pd.DataFrame,
    results: list[MonthResult],
    historical_gaps: list[str],
    unavailable_tail: list[str],
    fatal_months: list[str],
    status: str,
    cache_invalid_records: list[dict[str, Any]],
) -> dict[str, Any]:
    available_rows = int(availability[AVAILABILITY_COLUMN].sum())
    missing_rows = int(len(availability) - available_rows)
    month_records = [_record_for_ledger(result) for result in results]
    available_months = [result.month for result in results if not result.frame.empty]
    observed_positions = availability.index[availability[AVAILABILITY_COLUMN].to_numpy()]
    observed_range = {
        "start_inclusive": str(observed_positions[0]) if len(observed_positions) else None,
        "end_exclusive": (
            str(observed_positions[-1] + pd.Timedelta(minutes=15))
            if len(observed_positions)
            else None
        ),
    }
    return {
        "record_type": "alpha_dd_run",
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "symbol": symbol,
        "interval": interval,
        "source": SOURCE,
        "requested_months": [result.month for result in results],
        "start_month": _month_label(_month_start(start_month)),
        "end_month": _month_label(_month_start(end_month)),
        "bar_open_scope": {
            "start_inclusive": str(output.index[0]) if len(output) else None,
            "end_exclusive": str(output.index[-1] + pd.Timedelta(minutes=15)) if len(output) else None,
        },
        "requested_range": {
            "start_inclusive": str(output.index[0]) if len(output) else None,
            "end_exclusive": str(output.index[-1] + pd.Timedelta(minutes=15)) if len(output) else None,
        },
        "actual_acquired_range": observed_range,
        "index_semantics": "UTC DatetimeIndex named bar_open_ts; Binance bar OPEN time",
        "columns": list(OUTPUT_COLUMNS),
        "availability_column": AVAILABILITY_COLUMN,
        "output_path": str(output_path),
        "availability_path": str(availability_path),
        "source_ledger_path": str(ledger_path),
        "sha_sidecar_path": str(sha_sidecar),
        "rows": int(len(output)),
        "available_rows": available_rows,
        "missing_rows": missing_rows,
        "requested_month_count": len(results),
        "available_months": available_months,
        "historical_gap_months": historical_gaps,
        "unavailable_tail_months": unavailable_tail,
        "fatal_months": fatal_months,
        "completeness": {
            "status": status,
            "requested_month_count": len(results),
            "available_month_count": len(available_months),
            "missing_month_count": len(results) - len(available_months),
            "available_row_count": available_rows,
            "missing_row_count": missing_rows,
            "historical_gap_months": historical_gaps,
            "unavailable_tail_months": unavailable_tail,
        },
        "cache_invalid_months": [record.get("month") for record in cache_invalid_records],
        "latest_archive": _latest_archive_identity(results),
        "timestamp_semantics": {
            "archive_published_ts": "unknown unless the official source supplies it; null is retained",
            "collector_observed_ts": "not recorded as a live market observation",
            "exchange_available_ts": "not recorded as a live market observation",
            "live_causal_eligible": False,
        },
        "raw_zip_payloads_retained": False,
        "model_results_read": False,
        "git_commit": _git_commit(),
        "month_record_count": len(month_records),
    }


def run_acquisition(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    start_month: str | pd.Timestamp = DEFAULT_START_MONTH,
    end_month: str | pd.Timestamp = DEFAULT_END_MONTH,
    *,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    timeout: float = 30.0,
    max_workers: int = MAX_DOWNLOAD_WORKERS,
    availability_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    sha_sidecar: str | Path | None = None,
    monthly_dir: str | Path | None = None,
    now: str | pd.Timestamp | None = None,
    downloader: Callable[..., tuple[pd.DataFrame, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Acquire and assemble the inclusive monthly Spot archive range.

    ``max_workers`` is hard-capped at six.  HTTP 404/410 months are retained as
    explicit gaps; a contiguous missing suffix is labelled
    ``unavailable_tail``.  A checksum, parser, or worker error is fail-closed:
    its ledger record is appended and a final artifact is still written before
    :class:`AcquisitionError` is raised.
    """
    if interval != DEFAULT_INTERVAL:
        raise ValueError(f"Spot alpha/DD acquisition supports only {DEFAULT_INTERVAL}, got {interval!r}")
    if not symbol or "/" in symbol:
        raise ValueError(f"invalid symbol {symbol!r}")
    if not isinstance(max_workers, int) or not 1 <= max_workers <= MAX_DOWNLOAD_WORKERS:
        raise ValueError(f"max_workers must be an integer in [1, {MAX_DOWNLOAD_WORKERS}]")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    months = month_values(start_month, end_month)
    paths = artifact_paths(
        output_path,
        availability_path=availability_path,
        ledger_path=ledger_path,
        sha_sidecar=sha_sidecar,
        monthly_dir=monthly_dir,
    )
    paths["monthly"].mkdir(parents=True, exist_ok=True)
    active_downloader = downloader or download_d1_kline_month
    observed_now = _current_month(pd.Timestamp(now) if now is not None else None)

    results: dict[str, MonthResult] = {}
    cache_invalid_records: list[dict[str, Any]] = []
    for month in months:
        cached = _cached_month(
            month,
            monthly_dir=paths["monthly"],
            symbol=symbol,
            interval=interval,
        )
        if cached is None:
            continue
        if cached.record.get("record_type") == "alpha_dd_cache_invalid":
            cache_invalid_records.append(dict(cached.record))
            continue
        results[cached.month] = cached

    pending = [month for month in months if _month_label(month) not in results]
    stopped_tail = False
    # Process chronological batches so a future/current 404 can stop later
    # requests.  All downloads still have a maximum concurrency of six.
    for offset in range(0, len(pending), max_workers):
        if stopped_tail:
            break
        batch = pending[offset : offset + max_workers]
        with ThreadPoolExecutor(max_workers=min(max_workers, MAX_DOWNLOAD_WORKERS)) as executor:
            futures: dict[Future[MonthResult], pd.Timestamp] = {
                executor.submit(
                    _download_one,
                    month,
                    symbol=symbol,
                    interval=interval,
                    timeout=timeout,
                    downloader=active_downloader,
                ): month
                for month in batch
            }
            batch_results: list[MonthResult] = []
            for future in as_completed(futures):
                result = future.result()
                batch_results.append(result)
                results[result.month] = result
                if (
                    _is_archive_unavailable(result.record)
                    and _month_start(result.month) >= observed_now
                ):
                    stopped_tail = True
            # Persist completed valid months immediately after each batch so a
            # process interruption never loses already verified archive work.
            for result in sorted(batch_results, key=lambda item: item.month):
                _persist_month(
                    result,
                    monthly_dir=paths["monthly"],
                    symbol=symbol,
                    interval=interval,
                )
            if stopped_tail:
                for month in pending[offset + len(batch) :]:
                    label = _month_label(month)
                    if label in results:
                        continue
                    results[label] = MonthResult(
                        label,
                        pd.DataFrame(),
                        {
                            "record_type": "d1_archive_download",
                            "source": SOURCE,
                            "symbol": symbol,
                            "interval": interval,
                            "month": label,
                            "archive_url": None,
                            "http_status": None,
                            "checksum_verified": False,
                            "checksum_required": True,
                            "error": "not attempted after current unavailable archive tail",
                            "raw_payload_retained": False,
                            "live_causal_eligible": False,
                            "timestamp_semantics": "not a live observation timestamp",
                        },
                        attempted=False,
                    )

    ordered_results = [results[_month_label(month)] for month in months]
    historical_gaps, unavailable_tail, fatal_months = _mark_availability_statuses(
        ordered_results,
        now=observed_now,
    )
    output, availability = _build_grid_artifacts(
        ordered_results,
        start_month=start_month,
        end_month=end_month,
    )

    # Source/error records are appended before any fail-closed exception can
    # be raised.  classify_archive_revisions adds immutable replacement state
    # for newly downloaded archive identities.
    downloaded_records = [
        _record_for_ledger(result)
        for result in ordered_results
        if result.attempted
    ]
    revision_records = classify_archive_revisions(
        downloaded_records,
        ledger_path=paths["ledger"],
    )
    revision_by_month = {
        str(record.get("month")): record
        for record in revision_records
        if record.get("source") == SOURCE
    }
    ledger_records: list[dict[str, Any]] = []
    ledger_records.extend(cache_invalid_records)
    for result in ordered_results:
        if result.attempted:
            ledger_records.append(revision_by_month.get(result.month, _record_for_ledger(result)))
        else:
            ledger_records.append(_record_for_ledger(result))
    _append_jsonl(paths["ledger"], ledger_records)

    _atomic_write_parquet(output, paths["output"])
    _atomic_write_parquet(availability, paths["availability"])
    output_sha = sha256_file(paths["output"])
    availability_sha = sha256_file(paths["availability"])
    status = "failed" if fatal_months else ("complete_with_gaps" if historical_gaps or unavailable_tail else "complete")
    run_record = _run_record(
        output_path=paths["output"],
        availability_path=paths["availability"],
        ledger_path=paths["ledger"],
        sha_sidecar=paths["sha_sidecar"],
        symbol=symbol,
        interval=interval,
        start_month=start_month,
        end_month=end_month,
        output=output,
        availability=availability,
        results=ordered_results,
        historical_gaps=historical_gaps,
        unavailable_tail=unavailable_tail,
        fatal_months=fatal_months,
        status=status,
        cache_invalid_records=cache_invalid_records,
    )
    run_record.update(
        {
            "output_sha256": output_sha,
            "availability_sha256": availability_sha,
            "monthly_checkpoint_dir": str(paths["monthly"]),
            "download_workers": min(max_workers, MAX_DOWNLOAD_WORKERS),
            "source_records_appended": len(ledger_records),
            "error_months_appended_before_failure": [
                result.month
                for result in ordered_results
                if result.record.get("error")
            ],
        }
    )
    _append_jsonl(paths["ledger"], [run_record])
    ledger_sha = sha256_file(paths["ledger"])
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "kind": "alpha_dd_spot_15m_artifact_sha256",
        "artifact_path": str(paths["output"]),
        "artifact_sha256": output_sha,
        "output": {
            "path": str(paths["output"]),
            "sha256": output_sha,
        },
        "sha256": output_sha,
        "availability_path": str(paths["availability"]),
        "availability_sha256": availability_sha,
        "source_ledger_path": str(paths["ledger"]),
        "source_ledger_sha256": ledger_sha,
        "monthly_checkpoint_dir": str(paths["monthly"]),
        "symbol": symbol,
        "interval": interval,
        "columns": list(OUTPUT_COLUMNS),
        "availability_column": AVAILABILITY_COLUMN,
        "index": {
            "name": "bar_open_ts",
            "timezone": "UTC",
            "semantics": "bar OPEN time",
        },
        "rows": int(len(output)),
        "available_rows": int(availability[AVAILABILITY_COLUMN].sum()),
        "missing_rows": int((~availability[AVAILABILITY_COLUMN]).sum()),
        "start_inclusive": str(output.index[0]) if len(output) else None,
        "end_exclusive": str(output.index[-1] + pd.Timedelta(minutes=15)) if len(output) else None,
        "requested_range": run_record["requested_range"],
        "actual_acquired_range": run_record["actual_acquired_range"],
        "completeness": run_record["completeness"],
        "status": status,
        "latest_archive": run_record.get("latest_archive"),
        "raw_zip_payloads_retained": False,
        "timestamp_semantics": run_record["timestamp_semantics"],
        "model_results_read": False,
    }
    _atomic_write_json(paths["sha_sidecar"], sidecar)
    _atomic_write_bytes(
        paths["sha_text"],
        f"{output_sha}  {paths['output'].name}\n".encode("ascii"),
    )

    result_payload: dict[str, Any] = {
        **run_record,
        "output_path": str(paths["output"]),
        "features": str(paths["output"]),
        "availability_path": str(paths["availability"]),
        "ledger_path": str(paths["ledger"]),
        "sha_sidecar": str(paths["sha_sidecar"]),
        "sha_text": str(paths["sha_text"]),
        "monthly_dir": str(paths["monthly"]),
        "output_sha256": output_sha,
        "availability_sha256": availability_sha,
        "ledger_sha256": ledger_sha,
        "schema": {
            "columns": list(OUTPUT_COLUMNS),
            "availability_column": AVAILABILITY_COLUMN,
            "index_name": "bar_open_ts",
            "index_timezone": "UTC",
            "bar_time": "open",
        },
    }
    if fatal_months:
        raise AcquisitionError(
            "Spot archive acquisition failed closed for month(s): " + ", ".join(fatal_months),
            result=result_payload,
        )
    return result_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire audited Spot BTCUSDT 15-minute monthly archives",
        allow_abbrev=False,
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--start-month", default=DEFAULT_START_MONTH)
    parser.add_argument("--end-month", default=DEFAULT_END_MONTH)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-workers", type=int, default=MAX_DOWNLOAD_WORKERS)
    parser.add_argument("--availability")
    parser.add_argument("--ledger")
    parser.add_argument("--sha-sidecar")
    parser.add_argument("--monthly-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_acquisition(
            args.output,
            args.start_month,
            args.end_month,
            symbol=args.symbol,
            interval=args.interval,
            timeout=args.timeout,
            max_workers=args.max_workers,
            availability_path=args.availability,
            ledger_path=args.ledger,
            sha_sidecar=args.sha_sidecar,
            monthly_dir=args.monthly_dir,
        )
    except AcquisitionError as exc:
        if exc.result:
            print(json.dumps(exc.result, ensure_ascii=False, sort_keys=True, default=_json_default))
        else:
            print(str(exc))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=_json_default))
    return 0


acquire_spot_archives = run_acquisition


__all__ = [
    "AcquisitionError",
    "AVAILABILITY_COLUMN",
    "DEFAULT_END_MONTH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_START_MONTH",
    "OUTPUT_COLUMNS",
    "acquire_spot_archives",
    "artifact_paths",
    "build_parser",
    "expected_bar_grid",
    "main",
    "month_values",
    "run_acquisition",
    "sha256_file",
]


if __name__ == "__main__":
    raise SystemExit(main())
