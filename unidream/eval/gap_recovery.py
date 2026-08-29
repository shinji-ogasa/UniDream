"""Read-only recovery probes for missing development-cache spot bars.

Only Binance-owned Spot market-data hosts are accepted.  This module never
fills, interpolates, sorts, or writes the research cache; it records whether
the official source can return each expected timestamp around a known gap.
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


OFFICIAL_SPOT_REST_BASE = "https://data-api.binance.vision"
OFFICIAL_ARCHIVE_BASE = "https://data.binance.vision/data/spot/monthly/klines"
OFFICIAL_HOSTS = frozenset({"data-api.binance.vision", "data.binance.vision"})
KLINES_PATH = "/api/v3/klines"
DEFAULT_INTERVAL = "15m"
DEVELOPMENT_START = pd.Timestamp("2018-01-01")
DEVELOPMENT_END = pd.Timestamp("2024-01-01")


def _interval_delta(interval: str) -> pd.Timedelta:
    values = {
        "1m": pd.Timedelta(minutes=1),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "4h": pd.Timedelta(hours=4),
        "1d": pd.Timedelta(days=1),
    }
    if interval not in values:
        raise ValueError(f"unsupported gap-recovery interval: {interval!r}")
    return values[interval]


def _timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return parsed


def _normalise_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("cache index is not a DatetimeIndex")
    return pd.DatetimeIndex(pd.to_datetime(index, utc=True)).tz_localize(None)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _timestamp_digest(values: Iterable[pd.Timestamp]) -> str:
    timestamps = sorted(_timestamp(value) for value in values)
    payload = b"".join(int(value.value).to_bytes(8, "big", signed=True) for value in timestamps)
    return _sha256_bytes(payload)


def index_digest(index: pd.DatetimeIndex) -> str:
    """Hash the exact normalized index used for a cache comparison."""
    return _timestamp_digest(index)


def detect_gaps(index: pd.DatetimeIndex, *, interval: str = DEFAULT_INTERVAL) -> list[dict[str, Any]]:
    """Return every non-interval transition without repairing the index."""
    normalized = _normalise_index(index)
    if not normalized.is_monotonic_increasing or not normalized.is_unique:
        raise ValueError("cannot detect gaps from an unsorted or duplicate index")
    delta = _interval_delta(interval)
    gaps: list[dict[str, Any]] = []
    for position, difference in enumerate(normalized[1:] - normalized[:-1]):
        if difference == delta:
            continue
        left = normalized[position]
        right = normalized[position + 1]
        expected: list[str] = []
        cursor = left + delta
        while cursor < right:
            expected.append(str(cursor))
            cursor += delta
        gaps.append(
            {
                "gap_id": len(gaps),
                "left": str(left),
                "right": str(right),
                "delta": str(difference),
                "expected_missing_count": len(expected),
                "expected_missing_timestamps": expected,
            }
        )
    return gaps


def _assert_official_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_HOSTS:
        raise ValueError(f"refusing non-official Binance URL: {url}")


def _request_json(
    url: str,
    params: Mapping[str, Any],
    *,
    timeout: float,
    session: requests.Session,
) -> tuple[dict[str, Any], list[pd.Timestamp]]:
    _assert_official_url(url)
    response = session.get(url, params=dict(params), timeout=timeout)
    response_hash = _sha256_bytes(response.content)
    record: dict[str, Any] = {
        "url": response.url,
        "http_status": response.status_code,
        "response_sha256": response_hash,
        "response_bytes": len(response.content),
        "request_params": dict(params),
        "source": "official_spot_rest",
    }
    try:
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record, []
    if not isinstance(payload, list):
        record["error"] = "official REST payload is not a list"
        return record, []
    record["payload_sha256"] = _sha256_bytes(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    timestamps: list[pd.Timestamp] = []
    malformed = 0
    for row in payload:
        try:
            if not isinstance(row, list) or len(row) < 7:
                raise ValueError("short kline row")
            timestamps.append(pd.Timestamp(int(row[0]), unit="ms"))
        except (TypeError, ValueError):
            malformed += 1
    record["returned_rows"] = len(payload)
    record["malformed_rows"] = malformed
    record["first_timestamp"] = str(min(timestamps)) if timestamps else None
    record["last_timestamp"] = str(max(timestamps)) if timestamps else None
    return record, timestamps


def _archive_url(symbol: str, interval: str, month: pd.Timestamp) -> str:
    return (
        f"{OFFICIAL_ARCHIVE_BASE}/{symbol}/{interval}/"
        f"{symbol}-{interval}-{month.year:04d}-{month.month:02d}.zip"
    )


def _request_archive_month(
    symbol: str,
    interval: str,
    month: pd.Timestamp,
    *,
    timeout: float,
    session: requests.Session,
) -> tuple[dict[str, Any], list[pd.Timestamp]]:
    url = _archive_url(symbol, interval, month)
    _assert_official_url(url)
    response = session.get(url, timeout=timeout)
    response_hash = _sha256_bytes(response.content)
    record: dict[str, Any] = {
        "url": response.url,
        "http_status": response.status_code,
        "response_sha256": response_hash,
        "response_bytes": len(response.content),
        "source": "official_spot_monthly_archive",
        "month": f"{month.year:04d}-{month.month:02d}",
    }
    try:
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = archive.namelist()
            if len(names) != 1:
                raise ValueError(f"expected one CSV in archive, found {names}")
            raw_csv = archive.read(names[0])
        timestamps: list[pd.Timestamp] = []
        malformed = 0
        for row in csv.reader(io.StringIO(raw_csv.decode("utf-8"))):
            try:
                if len(row) < 7:
                    raise ValueError("short archive kline row")
                timestamps.append(pd.Timestamp(int(row[0]), unit="ms"))
            except (TypeError, ValueError):
                malformed += 1
        record["archive_member"] = names[0]
        record["archive_payload_sha256"] = _sha256_bytes(raw_csv)
        record["returned_rows"] = len(timestamps)
        record["malformed_rows"] = malformed
        record["first_timestamp"] = str(min(timestamps)) if timestamps else None
        record["last_timestamp"] = str(max(timestamps)) if timestamps else None
        return record, timestamps
    except (requests.RequestException, OSError, ValueError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record, []


def _gap_coverage(
    gap: Mapping[str, Any],
    *,
    cache_index: pd.DatetimeIndex,
    official_timestamps: Iterable[pd.Timestamp],
    source_records: list[Mapping[str, Any]],
    interval: str,
) -> dict[str, Any]:
    expected = {_timestamp(value) for value in gap["expected_missing_timestamps"]}
    official = {_timestamp(value) for value in official_timestamps}
    covered = sorted(expected & official)
    missing = sorted(expected - official)
    left = _timestamp(gap["left"])
    right = _timestamp(gap["right"])
    delta = _interval_delta(interval)
    cache_window = cache_index[(cache_index >= left - delta) & (cache_index <= right + delta)]
    return {
        **dict(gap),
        "expected_missing_count": len(expected),
        "official_covered_count": len(covered),
        "official_missing_after_probe_count": len(missing),
        "official_covered_timestamps": [str(value) for value in covered],
        "official_missing_after_probe_timestamps": [str(value) for value in missing],
        "coverage_rate": float(len(covered) / len(expected)) if expected else 1.0,
        "cache_window_timestamps": [str(value) for value in cache_window],
        "cache_window_timestamp_sha256": _timestamp_digest(cache_window),
        "official_timestamp_sha256": _timestamp_digest(official),
        "source_records": [dict(record) for record in source_records],
        "recovered_without_interpolation": len(missing) == 0,
        "policy": "record_only; never interpolate or write cache",
    }


def probe_official_gap_recovery(
    features: pd.DataFrame,
    *,
    returns: pd.Series | pd.DataFrame | None = None,
    symbol: str = "BTCUSDT",
    interval: str = DEFAULT_INTERVAL,
    start: Any = DEVELOPMENT_START,
    end: Any = DEVELOPMENT_END,
    rest_base_url: str = OFFICIAL_SPOT_REST_BASE,
    use_archive_fallback: bool = True,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Probe official Spot REST and optional monthly archives around each gap."""
    _assert_official_url(rest_base_url)
    feature_index = _normalise_index(features.index)
    returns_index = None
    if returns is not None:
        returns_frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
        returns_index = _normalise_index(returns_frame.index)
    feature_gaps = detect_gaps(feature_index, interval=interval)
    returns_gaps = detect_gaps(returns_index, interval=interval) if returns_index is not None else None

    owns_session = session is None
    active_session = session or requests.Session()
    archive_cache: dict[str, tuple[dict[str, Any], list[pd.Timestamp]]] = {}
    gap_records: list[dict[str, Any]] = []
    delta = _interval_delta(interval)
    try:
        for gap in feature_gaps:
            left = _timestamp(gap["left"])
            right = _timestamp(gap["right"])
            rest_url = rest_base_url.rstrip("/") + KLINES_PATH
            rest_params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": int((left - delta).timestamp() * 1000),
                "endTime": int((right + delta).timestamp() * 1000),
                "limit": 1000,
            }
            rest_record, rest_timestamps = _request_json(
                rest_url,
                rest_params,
                timeout=timeout,
                session=active_session,
            )
            expected = {_timestamp(value) for value in gap["expected_missing_timestamps"]}
            official = set(rest_timestamps)
            source_records: list[Mapping[str, Any]] = [rest_record]
            if use_archive_fallback and not expected.issubset(official):
                missing = expected - official
                # Normalize to a true month key.  ``Timestamp.replace(day=1)``
                # preserves the hour/minute, which used to make one archive
                # request appear as many distinct records for a single month.
                months = sorted({value.to_period("M").to_timestamp() for value in missing})
                for month in months:
                    month_key = f"{month.year:04d}-{month.month:02d}"
                    if month_key not in archive_cache:
                        archive_cache[month_key] = _request_archive_month(
                            symbol,
                            interval,
                            month,
                            timeout=timeout,
                            session=active_session,
                        )
                    archive_record, archive_timestamps = archive_cache[month_key]
                    source_records.append(archive_record)
                    official.update(archive_timestamps)
            gap_records.append(
                _gap_coverage(
                    gap,
                    cache_index=feature_index,
                    official_timestamps=official,
                    source_records=source_records,
                    interval=interval,
                )
            )
    finally:
        if owns_session:
            active_session.close()

    coverage_complete = all(item["recovered_without_interpolation"] for item in gap_records)
    result: dict[str, Any] = {
        "schema_version": 1,
        "source_policy": {
            "allowed_hosts": sorted(OFFICIAL_HOSTS),
            "rest_base_url": rest_base_url,
            "archive_base_url": OFFICIAL_ARCHIVE_BASE,
            "non_official_provider_used": False,
            "interpolation_used": False,
        },
        "scope": {
            "start": str(_timestamp(start)),
            "end_exclusive": str(_timestamp(end)),
            "symbol": symbol,
            "interval": interval,
        },
        "cache": {
            "feature_rows": len(feature_index),
            "feature_index_sha256": index_digest(feature_index),
            "feature_gap_count": len(feature_gaps),
            "returns_rows": len(returns_index) if returns_index is not None else None,
            "returns_index_sha256": index_digest(returns_index) if returns_index is not None else None,
            "returns_gap_count": len(returns_gaps) if returns_gaps is not None else None,
            "feature_returns_gap_sets_equal": returns_gaps == feature_gaps
            if returns_gaps is not None
            else None,
        },
        "gaps": gap_records,
        "summary": {
            "gap_count": len(gap_records),
            "expected_missing_bars": int(sum(item["expected_missing_count"] for item in gap_records)),
            "official_covered_bars": int(sum(item["official_covered_count"] for item in gap_records)),
            "official_missing_after_probe": int(
                sum(item["official_missing_after_probe_count"] for item in gap_records)
            ),
            "all_recovered_without_interpolation": coverage_complete,
            "status": "pass" if coverage_complete else "unresolved_official_gap",
        },
    }
    return result


def write_gap_recovery_jsonl(report: Mapping[str, Any], path: str | Path) -> int:
    """Write one run record plus one deterministic record per gap."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = [
        {
            "record_type": "official_gap_recovery_run",
            "source_policy": report.get("source_policy"),
            "scope": report.get("scope"),
            "cache": report.get("cache"),
            "summary": report.get("summary"),
            "provenance": report.get("provenance"),
        }
    ]
    for gap in report.get("gaps", []):
        records.append(
            {
                "record_type": "official_gap_recovery_gap",
                "gap_id": gap.get("gap_id"),
                "payload": gap,
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


def render_gap_recovery_markdown(
    report: Mapping[str, Any],
    *,
    ledger_path: str | Path | None = None,
) -> str:
    """Render official-source coverage evidence without model results."""
    scope = report.get("scope", {})
    policy = report.get("source_policy", {})
    cache = report.get("cache", {})
    summary = report.get("summary", {})
    provenance = report.get("provenance", {})
    lines = [
        "# Official Binance gap-recovery audit",
        "",
        "This report probes only Binance-owned Spot market-data sources around the development cache. It does not read model results and never interpolates or writes cache rows.",
        "",
        f"- Scope: `[{scope.get('start')}, {scope.get('end_exclusive')})`",
        f"- Symbol / interval: `{scope.get('symbol')}` / `{scope.get('interval')}`",
        f"- Allowed hosts: `{', '.join(policy.get('allowed_hosts', []))}`",
        f"- REST base: `{policy.get('rest_base_url')}`",
        f"- Archive base: `{policy.get('archive_base_url')}`",
        f"- Non-official provider used: `{policy.get('non_official_provider_used')}`",
        f"- Interpolation used: `{policy.get('interpolation_used')}`",
        f"- Cache feature rows / index digest: `{cache.get('feature_rows')}` / `{cache.get('feature_index_sha256')}`",
        f"- Returns rows / index digest: `{cache.get('returns_rows')}` / `{cache.get('returns_index_sha256')}`",
        f"- Ledger: `{ledger_path}`" if ledger_path is not None else "",
        f"- Probe git commit: `{provenance.get('git_commit')}`" if provenance else "",
        "",
        "## Summary",
        "",
        f"- Status: **{str(summary.get('status', 'unknown')).upper()}**",
        f"- Gaps: `{summary.get('gap_count')}`",
        f"- Expected missing bars: `{summary.get('expected_missing_bars')}`",
        f"- Officially covered bars: `{summary.get('official_covered_bars')}`",
        f"- Unresolved after official probes: `{summary.get('official_missing_after_probe')}`",
        "",
        "An unresolved bar is retained as a data-quality gap. The next cache generation may include an observed-bar availability sidecar, but it must not synthesize the missing OHLCV row.",
        "",
        "The 18 officially covered timestamps are eligible for a future v4 regeneration only after their official OHLCV rows and as-of external inputs are recomputed into the new body. This audit intentionally did not mutate the v3 body.",
        "The remaining 524 timestamps are retained as unresolved exchange/source outages: v4 should mark them `spot_bar_observed=False`, keep external availability masks separate, and exclude every sequence window crossing them.",
        "",
        "## Per-gap coverage",
        "",
        "| Gap | Left | Right | Expected | Covered | Unresolved | Coverage |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for gap in report.get("gaps", []):
        lines.append(
            "| {gap_id} | {left} | {right} | {expected_missing_count} | {official_covered_count} | {official_missing_after_probe_count} | {coverage_rate:.3f} |".format(
                gap_id=gap.get("gap_id"),
                left=gap.get("left"),
                right=gap.get("right"),
                expected_missing_count=gap.get("expected_missing_count"),
                official_covered_count=gap.get("official_covered_count"),
                official_missing_after_probe_count=gap.get("official_missing_after_probe_count"),
                coverage_rate=float(gap.get("coverage_rate", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Future v4 remediation policy",
            "",
            "- Keep the feature body at the exact 17 model columns; store `spot_bar_observed` and external-source availability in a separate sidecar.",
            "- Preserve official source/provenance hashes and the explicit gap list in v4 metadata.",
            "- Exclude sequence windows crossing unresolved gaps; do not sort, drop, fill, or interpolate rows during cache validation.",
            "- If official recovery remains incomplete, execution/evaluation must either segment metrics at the gap or explicitly attribute a return spanning the gap to the position held immediately before the gap. That attribution is a separate contract and must not silently become a post-gap position.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "OFFICIAL_SPOT_REST_BASE",
    "OFFICIAL_ARCHIVE_BASE",
    "OFFICIAL_HOSTS",
    "detect_gaps",
    "index_digest",
    "probe_official_gap_recovery",
    "render_gap_recovery_markdown",
    "write_gap_recovery_jsonl",
]
