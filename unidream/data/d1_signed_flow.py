"""Leak-safe D1 signed-flow acquisition and 15-minute feature assembly.

This module intentionally stops at data acquisition and deterministic feature
construction.  It does not train a model, select a policy, or inspect any
prediction result.

The D1 contract is deliberately explicit:

* a row is identified by ``decision_ts`` (one millisecond after the inclusive
  Binance ``close_time`` of a completed 15-minute bar),
* source kline values are read only from that completed bar,
* missing source rows remain NaN and receive a false availability mask, and
* archive publication/download metadata are never presented as live exchange
  observation metadata.

The monthly kline pilot is small.  Aggregate-trade archives are only probed
with HTTP HEAD requests for capacity estimation; their payloads are never
downloaded by the pilot.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from unidream.data.official_v4_sources import (
    OfficialSourceError,
    assert_official_url,
    official_archive_url,
)


OFFICIAL_PUBLIC_DATA_README = "https://github.com/binance/binance-public-data/blob/master/README.md"
OFFICIAL_SPOT_MARKET_DATA_DOCS = (
    "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints"
)
OFFICIAL_UM_MARKET_DATA_DOCS = (
    "https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data"
)

D1_SOURCE_NAMES = ("spot_klines", "um_klines")
D1_INTERVAL = "15m"
D1_FEATURE_COLUMNS = (
    "spot_trade_count",
    "spot_quote_volume",
    "spot_taker_buy_base",
    "spot_taker_buy_quote",
    "spot_taker_imbalance",
    "perp_trade_count",
    "perp_quote_volume",
    "perp_taker_buy_base",
    "perp_taker_buy_quote",
    "perp_taker_imbalance",
    "spot_perp_basis",
    "spot_perp_return_divergence",
)
D1_AVAILABILITY_COLUMNS = (
    "spot_bar_observed",
    "perp_bar_observed",
    "spot_taker_imbalance_available",
    "perp_taker_imbalance_available",
    "spot_perp_basis_available",
    "spot_perp_return_divergence_available",
    "d1_features_available",
)
_KLINE_FIELD_COLUMNS = (
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
_CHECKSUM_RE = re.compile(r"(?i)\b([0-9a-f]{64})\b")


def _utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _interval_delta(interval: str) -> pd.Timedelta:
    if interval != D1_INTERVAL:
        raise ValueError(f"D1 signed-flow pilot supports only {D1_INTERVAL}, got {interval!r}")
    return pd.Timedelta(minutes=15)


def _now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _frame_digest(frame: pd.DataFrame) -> str:
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


def _archive_month(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = _utc_timestamp(value)
    return timestamp.to_period("M").to_timestamp().tz_localize("UTC")


def _month_values(start: str | pd.Timestamp, end: str | pd.Timestamp) -> list[pd.Timestamp]:
    start_month = _archive_month(start)
    end_month = _archive_month(end)
    if end_month <= start_month:
        raise ValueError("archive estimate end must be after start")
    values: list[pd.Timestamp] = []
    cursor = start_month
    while cursor < end_month:
        values.append(cursor)
        cursor = (cursor + pd.offsets.MonthBegin(1)).normalize()
    return values


def aggtrade_archive_url(source: str, symbol: str, month: str | pd.Timestamp) -> str:
    """Return an official monthly aggregate-trade URL without downloading it."""
    month_value = _archive_month(month)
    if source == "spot_aggTrades":
        base = "https://data.binance.vision/data/spot/monthly/aggTrades"
    elif source == "um_aggTrades":
        base = "https://data.binance.vision/data/futures/um/monthly/aggTrades"
    else:
        raise ValueError(f"unsupported aggregate-trade source: {source!r}")
    url = (
        f"{base}/{symbol}/{symbol}-aggTrades-"
        f"{month_value.year:04d}-{month_value.month:02d}.zip"
    )
    assert_official_url(url, archive=True)
    return url


def _parse_checksum(payload: bytes, *, archive_name: str) -> tuple[str, str | None]:
    try:
        text = payload.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise OfficialSourceError(f"checksum sidecar is not UTF-8: {exc}") from exc
    match = _CHECKSUM_RE.search(text)
    if match is None:
        raise OfficialSourceError("checksum sidecar does not contain a SHA-256 digest")
    expected = match.group(1).lower()
    advertised_name = None
    remainder = text[match.end() :].strip()
    if remainder:
        advertised_name = Path(remainder.split()[-1]).name
        if advertised_name and advertised_name != archive_name:
            raise OfficialSourceError(
                f"checksum sidecar names {advertised_name!r}, expected {archive_name!r}"
            )
    return expected, advertised_name


def _response_metadata(
    response: requests.Response,
    *,
    requested_url: str | None = None,
) -> dict[str, Any]:
    final_url = str(response.url)
    assert_official_url(final_url, archive=True)
    return {
        "requested_url": requested_url or final_url,
        "final_url": final_url,
        "http_status": int(response.status_code),
        "response_bytes": int(len(response.content)),
        "response_sha256": _sha256_bytes(response.content),
    }


def _parse_kline_archive_bytes(
    payload: bytes,
    *,
    source: str,
    interval: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    delta = _interval_delta(interval)
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            csv_names = [name for name in names if name.lower().endswith(".csv")]
            if len(names) != 1 or len(csv_names) != 1:
                raise OfficialSourceError(f"expected one CSV member, found {names}")
            member_name = csv_names[0]
            raw_csv = archive.read(member_name)
    except (OSError, zipfile.BadZipFile) as exc:
        raise OfficialSourceError(f"{source} archive is not a valid zip: {exc}") from exc

    try:
        rows = list(csv.reader(io.StringIO(raw_csv.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise OfficialSourceError(f"{source} archive CSV is invalid: {exc}") from exc
    if not rows:
        raise OfficialSourceError(f"{source} archive CSV is empty")
    header_present = not rows[0][0].lstrip("+-").isdigit()
    data_rows = rows[1:] if header_present else rows
    parsed: list[list[Any]] = []
    malformed = 0
    for row in data_rows:
        if len(row) != 12:
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
                ]
            )
        except (TypeError, ValueError):
            malformed += 1
    if malformed:
        raise OfficialSourceError(f"{source} archive has {malformed} malformed rows")
    if not parsed:
        raise OfficialSourceError(f"{source} archive has no data rows")

    columns = [
        "bar_open_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bar_close_ms",
        "quote_volume",
        "n_trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    frame = pd.DataFrame(parsed, columns=columns)
    frame["bar_open_ts"] = pd.to_datetime(frame.pop("bar_open_ms"), unit="ms", utc=True)
    frame["bar_close_ts"] = pd.to_datetime(frame.pop("bar_close_ms"), unit="ms", utc=True)
    frame = frame.set_index("bar_open_ts")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise OfficialSourceError(f"{source} archive timestamps are not sorted and unique")
    expected_close = frame.index + delta - pd.Timedelta(milliseconds=1)
    if not frame["bar_close_ts"].equals(pd.Series(expected_close, index=frame.index, name="bar_close_ts")):
        mismatched = int((frame["bar_close_ts"] != expected_close).sum())
        raise OfficialSourceError(
            f"{source} archive has {mismatched} rows whose close timestamp does not match {interval}"
        )
    frame = frame[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "n_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "bar_close_ts",
        ]
    ]
    for column in _KLINE_FIELD_COLUMNS:
        frame[column] = frame[column].astype(float)
    frame["n_trades"] = frame["n_trades"].astype("int64")
    return frame, {
        "archive_member": member_name,
        "archive_payload_sha256": _sha256_bytes(raw_csv),
        "schema": {
            "kind": "kline_metadata",
            "header_present": header_present,
            "column_count": 12,
            "fields": [
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
            ],
        },
        "parsed_rows": int(len(frame)),
        "parsed_first_bar_open_ts": str(frame.index[0]),
        "parsed_last_bar_open_ts": str(frame.index[-1]),
        "parsed_first_bar_close_ts": str(frame["bar_close_ts"].iloc[0]),
        "parsed_last_bar_close_ts": str(frame["bar_close_ts"].iloc[-1]),
        "parsed_frame_sha256": _frame_digest(frame),
    }


def download_d1_kline_month(
    source: str,
    *,
    symbol: str,
    interval: str,
    month: str | pd.Timestamp,
    raw_dir: str | Path | None = None,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download one official monthly kline metadata archive with checksum evidence."""
    if source not in D1_SOURCE_NAMES:
        raise ValueError(f"unsupported D1 kline source: {source!r}")
    month_value = _archive_month(month)
    url = official_archive_url(source, symbol, interval, month_value)
    archive_name = Path(urlparse(url).path).name
    active = session or requests.Session()
    owns_session = session is None
    download_ts = _now_iso()
    record: dict[str, Any] = {
        "record_type": "d1_archive_download",
        "source": source,
        "symbol": symbol,
        "interval": interval,
        "month": f"{month_value.year:04d}-{month_value.month:02d}",
        "archive_url": url,
        "archive_published_ts": None,
        "collector_observed_ts": None,
        "exchange_available_ts": None,
        "download_ts": download_ts,
        "availability_certainty": "assumed_archive_event",
        "live_causal_eligible": False,
        "archive_revision_id": None,
        "checksum_url": f"{url}.CHECKSUM",
        "checksum_verified": False,
    }
    try:
        response = active.get(url, timeout=timeout)
        try:
            metadata = _response_metadata(response, requested_url=url)
        except OfficialSourceError:
            raise
        record.update(metadata)
        record["archive_revision_id"] = record["response_sha256"]
        if response.status_code != 200:
            record["error"] = f"official source returned HTTP {response.status_code}"
            return pd.DataFrame(), record

        checksum_response = active.get(f"{url}.CHECKSUM", timeout=timeout)
        checksum_meta = _response_metadata(
            checksum_response,
            requested_url=f"{url}.CHECKSUM",
        )
        record["checksum_http_status"] = checksum_meta["http_status"]
        record["checksum_response_bytes"] = checksum_meta["response_bytes"]
        record["checksum_response_sha256"] = checksum_meta["response_sha256"]
        if checksum_response.status_code != 200:
            record["error"] = f"checksum sidecar returned HTTP {checksum_response.status_code}"
            return pd.DataFrame(), record
        expected_sha, advertised_name = _parse_checksum(
            checksum_response.content,
            archive_name=archive_name,
        )
        record["checksum_expected_sha256"] = expected_sha
        record["checksum_advertised_name"] = advertised_name
        record["checksum_verified"] = expected_sha == record["response_sha256"]
        if not record["checksum_verified"]:
            record["error"] = "archive SHA-256 does not match official CHECKSUM sidecar"
            return pd.DataFrame(), record

        if raw_dir is not None:
            raw_path = Path(raw_dir) / source / archive_name
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(response.content)
            record["raw_path"] = str(raw_path)

        frame, parsed = _parse_kline_archive_bytes(
            response.content,
            source=source,
            interval=interval,
        )
        record.update(parsed)
        return frame, record
    except (OSError, requests.RequestException, OfficialSourceError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(), record
    finally:
        if owns_session:
            active.close()


def _head_content_length(response: requests.Response) -> int | None:
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Content-Length") or headers.get("content-length")
    if value is None:
        return None
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def estimate_aggtrade_archive_storage(
    *,
    symbol: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Estimate compressed aggregate-trade storage with HEAD requests only."""
    months = _month_values(start, end)
    active = session or requests.Session()
    owns_session = session is None
    records: list[dict[str, Any]] = []
    try:
        for source in ("spot_aggTrades", "um_aggTrades"):
            for month in months:
                url = aggtrade_archive_url(source, symbol, month)
                item: dict[str, Any] = {
                    "record_type": "d1_aggtrade_head_probe",
                    "source": source,
                    "symbol": symbol,
                    "month": f"{month.year:04d}-{month.month:02d}",
                    "archive_url": url,
                    "probe_ts": _now_iso(),
                    "archive_published_ts": None,
                    "collector_observed_ts": None,
                    "download_ts": None,
                    "payload_downloaded": False,
                }
                try:
                    response = active.head(url, allow_redirects=True, timeout=timeout)
                    item["final_url"] = str(response.url)
                    assert_official_url(item["final_url"], archive=True)
                    item["http_status"] = int(response.status_code)
                    item["content_length_bytes"] = _head_content_length(response)
                    item["known_size"] = item["content_length_bytes"] is not None
                except (OSError, requests.RequestException, OfficialSourceError) as exc:
                    item["http_status"] = None
                    item["content_length_bytes"] = None
                    item["known_size"] = False
                    item["error"] = f"{type(exc).__name__}: {exc}"
                records.append(item)
    finally:
        if owns_session:
            active.close()

    by_source: dict[str, dict[str, Any]] = {}
    for source in ("spot_aggTrades", "um_aggTrades"):
        source_records = [record for record in records if record["source"] == source]
        known = [record for record in source_records if record.get("known_size")]
        by_source[source] = {
            "months_requested": len(source_records),
            "http_200_count": sum(record.get("http_status") == 200 for record in source_records),
            "http_404_count": sum(record.get("http_status") == 404 for record in source_records),
            "known_size_months": len(known),
            "unknown_size_months": len(source_records) - len(known),
            "estimated_compressed_bytes": int(
                sum(int(record["content_length_bytes"]) for record in known)
            ),
            "records": source_records,
        }
    return {
        "schema_version": 1,
        "scope": {
            "symbol": symbol,
            "start_month": f"{_archive_month(start).year:04d}-{_archive_month(start).month:02d}",
            "end_month_exclusive": f"{_archive_month(end).year:04d}-{_archive_month(end).month:02d}",
        },
        "method": "HTTP HEAD Content-Length; no aggregate-trade payload downloaded",
        "sources": by_source,
        "records": records,
        "estimated_compressed_bytes_known": int(
            sum(value["estimated_compressed_bytes"] for value in by_source.values())
        ),
    }


def _normalise_kline_for_d1(
    frame: pd.DataFrame,
    *,
    source: str,
    interval: str,
) -> pd.DataFrame:
    delta = _interval_delta(interval)
    if frame.empty:
        index = pd.DatetimeIndex([], tz="UTC", name="decision_ts")
        prefix = "spot_" if source == "spot_klines" else "perp_"
        field_columns = [
            f"{prefix}{column.replace('n_trades', 'trade_count')}"
            for column in _KLINE_FIELD_COLUMNS
        ]
        return pd.DataFrame(
            index=index,
            columns=field_columns,
            dtype=float,
        )
    normalised = frame.copy()
    index = pd.DatetimeIndex(normalised.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    normalised.index = index
    if not normalised.index.is_unique or not normalised.index.is_monotonic_increasing:
        raise ValueError(f"{source} kline index must be sorted and unique")
    if "bar_close_ts" in normalised.columns:
        close_ts = pd.to_datetime(normalised["bar_close_ts"], utc=True)
        expected_close = pd.Series(
            normalised.index + delta - pd.Timedelta(milliseconds=1),
            index=normalised.index,
            name="bar_close_ts",
        )
        if not close_ts.equals(expected_close):
            raise ValueError(
                f"{source} kline close timestamps must equal decision_ts-1ms"
            )
    for column in _KLINE_FIELD_COLUMNS:
        if column not in normalised.columns:
            raise ValueError(f"{source} kline metadata is missing {column!r}")
    normalised = normalised[list(_KLINE_FIELD_COLUMNS)].copy()
    normalised = normalised.rename(columns={"n_trades": "trade_count"})
    for column in normalised.columns:
        normalised[column] = pd.to_numeric(normalised[column], errors="coerce").astype(float)
    decision_index = normalised.index + delta
    normalised.index = decision_index
    normalised.index.name = "decision_ts"
    return normalised.add_prefix("spot_" if source == "spot_klines" else "perp_")


def build_d1_features(
    spot: pd.DataFrame,
    perp: pd.DataFrame,
    *,
    interval: str = D1_INTERVAL,
    bar_open_start: str | pd.Timestamp | None = None,
    bar_open_end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build completed-bar D1 features and explicit availability masks.

    ``bar_open_start``/``bar_open_end`` describe an open-time scope
    ``[start, end)``.  The returned index is ``decision_ts = bar_open + 15m``;
    each row therefore represents the Binance half-open interval
    ``[bar_open_ts, decision_ts)`` (whose inclusive ``close_time`` is
    ``decision_ts - 1ms``) and never reads a still-forming bar or a future row.
    """
    delta = _interval_delta(interval)
    spot_norm = _normalise_kline_for_d1(spot, source="spot_klines", interval=interval)
    perp_norm = _normalise_kline_for_d1(perp, source="um_klines", interval=interval)

    if bar_open_start is None:
        open_starts = [idx.min() - delta for idx in (spot_norm.index, perp_norm.index) if len(idx)]
        if not open_starts:
            raise ValueError("cannot infer D1 scope from two empty frames")
        open_start = min(open_starts)
    else:
        open_start = _utc_timestamp(bar_open_start)
    if bar_open_end is None:
        open_ends = [idx.max() for idx in (spot_norm.index, perp_norm.index) if len(idx)]
        if not open_ends:
            raise ValueError("cannot infer D1 scope from two empty frames")
        decision_end = max(open_ends)
    else:
        open_end = _utc_timestamp(bar_open_end)
        if open_end <= open_start:
            raise ValueError("bar_open_end must be after bar_open_start")
        decision_end = open_end
    if decision_end < open_start + delta:
        raise ValueError("D1 scope has no complete bars")
    decision_index = pd.date_range(
        open_start + delta,
        decision_end,
        freq=delta,
        inclusive="both",
        name="decision_ts",
    )
    spot_grid = spot_norm.reindex(decision_index)
    perp_grid = perp_norm.reindex(decision_index)
    values = pd.concat([spot_grid, perp_grid], axis=1)

    spot_observed = values["spot_close"].notna() & np.isfinite(values["spot_close"])
    perp_observed = values["perp_close"].notna() & np.isfinite(values["perp_close"])
    spot_quote = values["spot_quote_volume"]
    perp_quote = values["perp_quote_volume"]
    spot_buy_quote = values["spot_taker_buy_quote"]
    perp_buy_quote = values["perp_taker_buy_quote"]
    spot_imbalance_available = (
        spot_observed & spot_quote.notna() & spot_buy_quote.notna() & (spot_quote > 0)
    )
    perp_imbalance_available = (
        perp_observed & perp_quote.notna() & perp_buy_quote.notna() & (perp_quote > 0)
    )
    values["spot_taker_imbalance"] = pd.Series(
        np.where(
            spot_imbalance_available,
            2.0 * spot_buy_quote / spot_quote - 1.0,
            np.nan,
        ),
        index=decision_index,
    )
    values["perp_taker_imbalance"] = pd.Series(
        np.where(
            perp_imbalance_available,
            2.0 * perp_buy_quote / perp_quote - 1.0,
            np.nan,
        ),
        index=decision_index,
    )
    valid_basis = (
        spot_observed
        & perp_observed
        & values["spot_close"].notna()
        & values["perp_close"].notna()
        & (values["spot_close"] > 0)
        & (values["perp_close"] > 0)
    )
    values["spot_perp_basis"] = pd.Series(
        np.where(
            valid_basis,
            np.log(values["perp_close"] / values["spot_close"]),
            np.nan,
        ),
        index=decision_index,
    )

    adjacent = pd.Series(decision_index, index=decision_index).diff().eq(delta)
    spot_prev = spot_observed & spot_observed.shift(1, fill_value=False) & adjacent
    perp_prev = perp_observed & perp_observed.shift(1, fill_value=False) & adjacent
    valid_return_divergence = valid_basis & spot_prev & perp_prev
    spot_log_return = np.log(values["spot_close"]).diff()
    perp_log_return = np.log(values["perp_close"]).diff()
    values["spot_perp_return_divergence"] = pd.Series(
        np.where(
            valid_return_divergence,
            perp_log_return - spot_log_return,
            np.nan,
        ),
        index=decision_index,
    )

    availability = pd.DataFrame(index=decision_index)
    availability["spot_bar_observed"] = spot_observed.astype(bool)
    availability["perp_bar_observed"] = perp_observed.astype(bool)
    availability["spot_taker_imbalance_available"] = spot_imbalance_available.astype(bool)
    availability["perp_taker_imbalance_available"] = perp_imbalance_available.astype(bool)
    availability["spot_perp_basis_available"] = valid_basis.astype(bool)
    availability["spot_perp_return_divergence_available"] = valid_return_divergence.astype(bool)
    required_columns = [
        "spot_trade_count",
        "spot_quote_volume",
        "spot_taker_buy_base",
        "spot_taker_buy_quote",
        "perp_trade_count",
        "perp_quote_volume",
        "perp_taker_buy_base",
        "perp_taker_buy_quote",
    ]
    finite_required = np.isfinite(values[required_columns]).all(axis=1)
    availability["d1_features_available"] = (
        finite_required
        & availability["spot_taker_imbalance_available"]
        & availability["perp_taker_imbalance_available"]
        & availability["spot_perp_basis_available"]
        & availability["spot_perp_return_divergence_available"]
    ).astype(bool)

    features = values.reindex(columns=D1_FEATURE_COLUMNS).copy()
    for column in D1_FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype(float)
    features.index.name = "decision_ts"
    availability.index.name = "decision_ts"
    return features, availability


def d1_bar_ledger_records(
    availability: pd.DataFrame,
    *,
    source_records: Mapping[str, Mapping[str, Any]],
    interval: str = D1_INTERVAL,
) -> list[dict[str, Any]]:
    """Convert row masks into append-only availability ledger records."""
    delta = _interval_delta(interval)
    records: list[dict[str, Any]] = []
    for decision_ts, row in availability.iterrows():
        decision = _utc_timestamp(decision_ts)
        bar_open = decision - delta
        payload = {
            "record_type": "d1_bar_availability",
            "decision_ts": decision.isoformat(),
            "bar_open_ts": bar_open.isoformat(),
            "bar_close_ts_inclusive": (decision - pd.Timedelta(milliseconds=1)).isoformat(),
            "interval": interval,
            "interval_semantics": "[bar_open_ts, decision_ts); close_time=decision_ts-1ms",
            "source_kind": "archive",
            "archive_published_ts": {
                source: source_records.get(source, {}).get("archive_published_ts")
                for source in D1_SOURCE_NAMES
            },
            "collector_observed_ts": {
                source: source_records.get(source, {}).get("collector_observed_ts")
                for source in D1_SOURCE_NAMES
            },
            "exchange_available_ts": {
                source: source_records.get(source, {}).get("exchange_available_ts")
                for source in D1_SOURCE_NAMES
            },
            "archive_revision_id": {
                source: source_records.get(source, {}).get("archive_revision_id")
                for source in D1_SOURCE_NAMES
            },
            "availability_certainty": {
                source: source_records.get(source, {}).get("availability_certainty")
                for source in D1_SOURCE_NAMES
            },
        }
        for column in D1_AVAILABILITY_COLUMNS:
            payload[column] = bool(row[column])
        records.append(payload)
    return records


def _latest_revisions(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    destination = Path(path)
    if not destination.exists():
        return latest
    for line_no, line in enumerate(destination.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {destination}:{line_no}: {exc}") from exc
        source = record.get("source")
        month = record.get("month")
        if source and month and record.get("record_type") == "d1_archive_download":
            latest[(str(source), str(month))] = record
    return latest


def classify_archive_revisions(
    records: Iterable[Mapping[str, Any]],
    *,
    ledger_path: str | Path,
) -> list[dict[str, Any]]:
    """Annotate source records with initial/unchanged/replaced revision state."""
    latest = _latest_revisions(ledger_path)
    annotated: list[dict[str, Any]] = []
    for source_record in records:
        record = dict(source_record)
        key = (str(record.get("source")), str(record.get("month")))
        previous = latest.get(key)
        previous_sha = previous.get("archive_revision_id") if previous else None
        current_sha = record.get("archive_revision_id")
        if previous is None:
            status = "initial"
        elif previous_sha == current_sha:
            status = "unchanged"
        else:
            status = "replaced"
        record["revision_status"] = status
        record["previous_archive_revision_id"] = previous_sha
        record["revision_first_seen_ts"] = (
            previous.get("revision_first_seen_ts") if previous and status == "unchanged" else _now_iso()
        )
        annotated.append(record)
        latest[key] = record
    return annotated


def append_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")) for record in records]
    if not rows:
        return 0
    with destination.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")
    return len(rows)


def summarize_d1_pilot(
    features: pd.DataFrame,
    availability: pd.DataFrame,
    *,
    source_records: Mapping[str, Mapping[str, Any]],
    capacity: Mapping[str, Any],
) -> dict[str, Any]:
    """Return deterministic, model-free pilot summary fields."""
    finite = np.isfinite(features).all(axis=1)
    return {
        "schema_version": 1,
        "feature_columns": list(features.columns),
        "availability_columns": list(availability.columns),
        "rows": int(len(features)),
        "decision_first_ts": str(features.index[0]) if len(features) else None,
        "decision_last_ts": str(features.index[-1]) if len(features) else None,
        "all_feature_values_finite_rows": int(finite.sum()),
        "d1_features_available_rows": int(availability["d1_features_available"].sum()),
        "d1_features_available_fraction": float(availability["d1_features_available"].mean())
        if len(availability)
        else 0.0,
        "zero_valued_feature_cells": int((features == 0.0).sum().sum()),
        "nan_feature_cells": int(features.isna().sum().sum()),
        "source_records": {
            source: {
                "month": record.get("month"),
                "archive_revision_id": record.get("archive_revision_id"),
                "checksum_verified": record.get("checksum_verified"),
                "parsed_rows": record.get("parsed_rows"),
                "archive_published_ts": record.get("archive_published_ts"),
                "collector_observed_ts": record.get("collector_observed_ts"),
                "exchange_available_ts": record.get("exchange_available_ts"),
                "availability_certainty": record.get("availability_certainty"),
                "live_causal_eligible": record.get("live_causal_eligible"),
            }
            for source, record in source_records.items()
        },
        "aggtrade_capacity": {
            "method": capacity.get("method"),
            "estimated_compressed_bytes_known": capacity.get("estimated_compressed_bytes_known"),
            "source_summaries": {
                source: {
                    key: value
                    for key, value in payload.items()
                    if key != "records"
                }
                for source, payload in (capacity.get("sources") or {}).items()
            },
        },
    }


__all__ = [
    "D1_AVAILABILITY_COLUMNS",
    "D1_FEATURE_COLUMNS",
    "D1_INTERVAL",
    "D1_SOURCE_NAMES",
    "OFFICIAL_PUBLIC_DATA_README",
    "OFFICIAL_SPOT_MARKET_DATA_DOCS",
    "OFFICIAL_UM_MARKET_DATA_DOCS",
    "aggtrade_archive_url",
    "append_jsonl",
    "build_d1_features",
    "classify_archive_revisions",
    "d1_bar_ledger_records",
    "download_d1_kline_month",
    "estimate_aggtrade_archive_storage",
    "summarize_d1_pilot",
]
