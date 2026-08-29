"""Official Binance source readers used by the schema-v4 rebuild.

Only Binance-owned HTTPS hosts are accepted.  Every response records the
requested/final URL, response hash, archive-member hash, and parsed schema.
The readers are deliberately side-effect free: they return frames in memory
and never write a cache.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

import pandas as pd
import requests


OFFICIAL_SPOT_ARCHIVE_BASE = "https://data.binance.vision/data/spot/monthly/klines"
OFFICIAL_UM_ARCHIVE_BASE = "https://data.binance.vision/data/futures/um/monthly"
OFFICIAL_SPOT_REST_BASE = "https://data-api.binance.vision"
OFFICIAL_UM_REST_BASE = "https://fapi.binance.com"
OFFICIAL_HOSTS = frozenset(
    {
        "data.binance.vision",
        "data-api.binance.vision",
        "fapi.binance.com",
    }
)
OFFICIAL_ARCHIVE_HOSTS = frozenset({"data.binance.vision"})
KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "n_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
)
FUNDING_COLUMNS = ("calc_time", "funding_interval_hours", "last_funding_rate")


class OfficialSourceError(ValueError):
    """Raised when an official source response cannot be parsed safely."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return _sha256_bytes(payload.encode("utf-8"))


def assert_official_url(url: str, *, archive: bool = False) -> None:
    parsed = urlparse(url)
    allowed = OFFICIAL_ARCHIVE_HOSTS if archive else OFFICIAL_HOSTS
    if parsed.scheme != "https" or parsed.hostname not in allowed:
        raise OfficialSourceError(f"refusing non-official Binance URL: {url}")


def _month(value: str | pd.Timestamp) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    return parsed.to_period("M").to_timestamp()


def _timestamp_ms(value: pd.Timestamp) -> int:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return int(parsed.timestamp() * 1000)


def official_archive_url(source: str, symbol: str, interval: str, month: str | pd.Timestamp) -> str:
    """Build one official monthly archive URL for a supported source."""
    month_value = _month(month)
    if source == "spot_klines":
        url = (
            f"{OFFICIAL_SPOT_ARCHIVE_BASE}/{symbol}/{interval}/"
            f"{symbol}-{interval}-{month_value.year:04d}-{month_value.month:02d}.zip"
        )
    elif source == "um_mark_price_klines":
        url = (
            f"{OFFICIAL_UM_ARCHIVE_BASE}/markPriceKlines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{month_value.year:04d}-{month_value.month:02d}.zip"
        )
    elif source == "um_funding_rate":
        url = (
            f"{OFFICIAL_UM_ARCHIVE_BASE}/fundingRate/{symbol}/"
            f"{symbol}-fundingRate-{month_value.year:04d}-{month_value.month:02d}.zip"
        )
    else:
        raise OfficialSourceError(f"unsupported official source: {source!r}")
    assert_official_url(url, archive=True)
    return url


def _parsed_frame_digest(frame: pd.DataFrame) -> str:
    row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    descriptor = {
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_dtype": str(frame.index.dtype),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _parse_kline_csv(raw_csv: bytes, *, source: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        rows = list(csv.reader(io.StringIO(raw_csv.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise OfficialSourceError(f"{source} archive CSV is not valid UTF-8/CSV: {exc}") from exc
    if not rows:
        raise OfficialSourceError(f"{source} archive CSV is empty")
    header_present = not rows[0][0].lstrip("+-").isdigit()
    data_rows = rows[1:] if header_present else rows
    malformed = 0
    parsed: list[list[Any]] = []
    for row in data_rows:
        if len(row) != len(KLINE_COLUMNS):
            malformed += 1
            continue
        try:
            parsed.append(
                [
                    int(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    int(row[6]),
                    float(row[7]),
                    int(float(row[8])),
                    float(row[9]),
                    float(row[10]),
                    row[11],
                ]
            )
        except (TypeError, ValueError):
            malformed += 1
    if malformed:
        raise OfficialSourceError(f"{source} archive has {malformed} malformed rows")
    if not parsed:
        raise OfficialSourceError(f"{source} archive has no data rows")
    frame = pd.DataFrame(parsed, columns=KLINE_COLUMNS)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms")
    frame = frame.set_index("open_time")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise OfficialSourceError(f"{source} archive timestamps are not sorted and unique")
    selected = frame[["open", "high", "low", "close", "volume"]].astype(float)
    return selected, {
        "schema": {
            "kind": "kline",
            "header_present": header_present,
            "column_count": len(KLINE_COLUMNS),
            "columns": list(KLINE_COLUMNS),
        },
        "parsed_rows": int(len(selected)),
        "parsed_first_timestamp": str(selected.index[0]),
        "parsed_last_timestamp": str(selected.index[-1]),
        "parsed_frame_sha256": _parsed_frame_digest(selected),
    }


def _parse_funding_csv(raw_csv: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        rows = list(csv.reader(io.StringIO(raw_csv.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise OfficialSourceError(f"funding archive CSV is not valid UTF-8/CSV: {exc}") from exc
    if not rows or rows[0] != list(FUNDING_COLUMNS):
        raise OfficialSourceError(
            f"funding archive schema mismatch: expected header {list(FUNDING_COLUMNS)}, "
            f"got {rows[0] if rows else None}"
        )
    parsed: list[tuple[int, float]] = []
    malformed = 0
    for row in rows[1:]:
        if len(row) != len(FUNDING_COLUMNS):
            malformed += 1
            continue
        try:
            parsed.append((int(row[0]), float(row[2])))
        except (TypeError, ValueError):
            malformed += 1
    if malformed:
        raise OfficialSourceError(f"funding archive has {malformed} malformed rows")
    if not parsed:
        raise OfficialSourceError("funding archive has no data rows")
    frame = pd.DataFrame(parsed, columns=["calc_time", "funding_rate"])
    frame["calc_time"] = pd.to_datetime(frame["calc_time"], unit="ms")
    frame = frame.set_index("calc_time")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise OfficialSourceError("funding archive timestamps are not sorted and unique")
    frame = frame[["funding_rate"]].astype(float)
    return frame, {
        "schema": {
            "kind": "funding_rate",
            "header_present": True,
            "column_count": len(FUNDING_COLUMNS),
            "columns": list(FUNDING_COLUMNS),
        },
        "parsed_rows": int(len(frame)),
        "parsed_first_timestamp": str(frame.index[0]),
        "parsed_last_timestamp": str(frame.index[-1]),
        "parsed_frame_sha256": _parsed_frame_digest(frame),
    }


def _response_record(
    *,
    source: str,
    requested_url: str,
    response: requests.Response,
    request_params: Mapping[str, Any] | None = None,
    archive: bool = False,
) -> dict[str, Any]:
    assert_official_url(response.url, archive=archive)
    return {
        "source": source,
        "requested_url": requested_url,
        "final_url": response.url,
        "http_status": int(response.status_code),
        "response_bytes": int(len(response.content)),
        "response_sha256": _sha256_bytes(response.content),
        "request_params": dict(request_params or {}),
    }


def fetch_archive_month(
    source: str,
    *,
    symbol: str,
    interval: str,
    month: str | pd.Timestamp,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch and parse one official archive month without writing files."""
    url = official_archive_url(source, symbol, interval, month)
    active = session or requests.Session()
    owns_session = session is None
    try:
        response = active.get(url, timeout=timeout)
        record = _response_record(
            source=source,
            requested_url=url,
            response=response,
            archive=True,
        )
        record["month"] = f"{_month(month).year:04d}-{_month(month).month:02d}"
        record["archive"] = True
        if response.status_code != 200:
            record["error"] = f"official source returned HTTP {response.status_code}"
            return pd.DataFrame(), record
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                names = archive.namelist()
                if len(names) != 1 or not names[0].lower().endswith(".csv"):
                    raise OfficialSourceError(f"expected one CSV member, found {names}")
                raw_csv = archive.read(names[0])
            record["archive_member"] = names[0]
            record["archive_payload_sha256"] = _sha256_bytes(raw_csv)
            if source == "um_funding_rate":
                frame, parsed = _parse_funding_csv(raw_csv)
            else:
                frame, parsed = _parse_kline_csv(raw_csv, source=source)
            record.update(parsed)
            return frame, record
        except (OSError, zipfile.BadZipFile, OfficialSourceError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            return pd.DataFrame(), record
    finally:
        if owns_session:
            active.close()


def fetch_spot_rest_window(
    *,
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch one official Spot REST window for gap recovery."""
    url = OFFICIAL_SPOT_REST_BASE.rstrip("/") + "/api/v3/klines"
    assert_official_url(url)
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": _timestamp_ms(start),
        "endTime": _timestamp_ms(end),
        "limit": 1000,
    }
    active = session or requests.Session()
    owns_session = session is None
    try:
        response = active.get(url, params=params, timeout=timeout)
        record = _response_record(
            source="spot_klines_rest",
            requested_url=url,
            response=response,
            request_params=params,
            archive=False,
        )
        record["archive"] = False
        if response.status_code != 200:
            record["error"] = f"official source returned HTTP {response.status_code}"
            return pd.DataFrame(), record
        try:
            payload = response.json()
        except ValueError as exc:
            record["error"] = f"invalid JSON: {exc}"
            return pd.DataFrame(), record
        if not isinstance(payload, list):
            record["error"] = "Spot REST payload is not a list"
            return pd.DataFrame(), record
        rows = []
        malformed = 0
        for row in payload:
            if not isinstance(row, list) or len(row) < 7:
                malformed += 1
                continue
            try:
                rows.append(
                    [
                        int(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        float(row[4]),
                        float(row[5]),
                    ]
                )
            except (TypeError, ValueError):
                malformed += 1
        if malformed:
            record["error"] = f"Spot REST has {malformed} malformed rows"
            return pd.DataFrame(), record
        frame = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
        if frame.empty:
            record.update({"returned_rows": 0, "parsed_rows": 0})
            return frame.set_index(pd.DatetimeIndex([], name="open_time")), record
        frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms")
        frame = frame.set_index("open_time")
        frame = frame[["open", "high", "low", "close", "volume"]].astype(float)
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise OfficialSourceError("Spot REST timestamps are not sorted and unique")
        record.update(
            {
                "returned_rows": len(payload),
                "parsed_rows": len(frame),
                "parsed_first_timestamp": str(frame.index[0]),
                "parsed_last_timestamp": str(frame.index[-1]),
                "parsed_frame_sha256": _parsed_frame_digest(frame),
            }
        )
        return frame, record
    finally:
        if owns_session:
            active.close()


def probe_official_sources(
    *,
    symbol: str = "BTCUSDT",
    interval: str = "15m",
    months: Iterable[str] = ("2018-01", "2019-12", "2020-01"),
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Probe source URL/schema availability for a few representative months."""
    active = session or requests.Session()
    owns_session = session is None
    probe_months = list(months)
    records: list[dict[str, Any]] = []
    try:
        for month in probe_months:
            for source in ("spot_klines", "um_mark_price_klines", "um_funding_rate"):
                _frame, record = fetch_archive_month(
                    source,
                    symbol=symbol,
                    interval=interval,
                    month=month,
                    timeout=timeout,
                    session=active,
                )
                records.append(record)
    finally:
        if owns_session:
            active.close()
    by_source: dict[str, dict[str, Any]] = {}
    for source in ("spot_klines", "um_mark_price_klines", "um_funding_rate"):
        source_records = [record for record in records if record["source"] == source]
        by_source[source] = {
            "probe_count": len(source_records),
            "http_200_count": sum(record.get("http_status") == 200 for record in source_records),
            "http_404_count": sum(record.get("http_status") == 404 for record in source_records),
            "records": source_records,
        }
    return {
        "schema_version": 1,
        "source_policy": {
            "allowed_hosts": sorted(OFFICIAL_HOSTS),
            "archive_hosts": sorted(OFFICIAL_ARCHIVE_HOSTS),
            "non_official_provider_used": False,
            "redirects_must_remain_official": True,
        },
        "scope": {"symbol": symbol, "interval": interval, "probe_months": probe_months},
        "sources": by_source,
        "status": "pass" if all(value["http_200_count"] > 0 for value in by_source.values()) else "blocked",
    }


def write_source_probe_jsonl(report: Mapping[str, Any], path: str | Path) -> int:
    """Write one run record plus one record per source response."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = [
        {
            "record_type": "official_v4_source_probe_run",
            "source_policy": report.get("source_policy"),
            "scope": report.get("scope"),
            "status": report.get("status"),
        }
    ]
    for source, payload in (report.get("sources") or {}).items():
        for response in payload.get("records", []):
            records.append(
                {
                    "record_type": "official_v4_source_probe_response",
                    "source": source,
                    "payload": response,
                }
            )
    destination.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )
    return len(records)


__all__ = [
    "FUNDING_COLUMNS",
    "KLINE_COLUMNS",
    "OFFICIAL_ARCHIVE_HOSTS",
    "OFFICIAL_HOSTS",
    "OFFICIAL_SPOT_ARCHIVE_BASE",
    "OFFICIAL_SPOT_REST_BASE",
    "OFFICIAL_UM_ARCHIVE_BASE",
    "OFFICIAL_UM_REST_BASE",
    "OfficialSourceError",
    "assert_official_url",
    "fetch_archive_month",
    "fetch_spot_rest_window",
    "official_archive_url",
    "probe_official_sources",
    "write_source_probe_jsonl",
]
