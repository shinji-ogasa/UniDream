"""Regenerate a schema-v4 cache from official Binance raw data.

The rebuild is intentionally separate from the legacy runtime cache writer.
It fetches official monthly archives, probes missing Spot timestamps through
the official Spot REST endpoint, computes the existing causal feature
functions on each contiguous observed segment, and writes a full-grid
availability sidecar.  Unresolved bars remain absent from the body and false
in the sidecar; no interpolation is performed.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import requests

from unidream.data.cache_v4 import MODEL_FEATURE_COLUMNS
from unidream.data.features import compute_features, get_raw_returns
from unidream.data.official_v4_sources import (
    OfficialSourceError,
    fetch_archive_month,
    fetch_spot_rest_window,
)


DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "15m"
DEFAULT_START = pd.Timestamp("2018-01-01")
DEFAULT_END = pd.Timestamp("2024-01-01")
EXTERNAL_ARCHIVE_START = pd.Timestamp("2020-01-01")


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
        raise OfficialSourceError(f"unsupported rebuild interval: {interval!r}")
    return values[interval]


def _normalise_timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        return parsed
    return parsed.tz_convert("UTC").tz_localize(None)


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    cursor = start.to_period("M").to_timestamp()
    final = end.to_period("M").to_timestamp()
    months: list[pd.Timestamp] = []
    while cursor < final:
        months.append(cursor)
        cursor = cursor + pd.offsets.MonthBegin(1)
    return months


def _expected_grid(start: pd.Timestamp, end: pd.Timestamp, interval: str) -> pd.DatetimeIndex:
    delta = _interval_delta(interval)
    if end <= start:
        raise OfficialSourceError("rebuild end must be after start")
    span = end - start
    if span % delta != pd.Timedelta(0):
        raise OfficialSourceError("rebuild scope is not aligned to the configured interval")
    return pd.date_range(start, end - delta, freq=delta)


def _empty_source_record(source: str, url: str, error: Exception) -> dict[str, Any]:
    return {
        "source": source,
        "requested_url": url,
        "final_url": None,
        "http_status": None,
        "response_bytes": 0,
        "response_sha256": None,
        "request_params": {},
        "error": f"{type(error).__name__}: {error}",
    }


def _concat_source_frames(
    frames: Iterable[pd.DataFrame],
    *,
    source: str,
) -> pd.DataFrame:
    values = [frame for frame in frames if not frame.empty]
    if not values:
        return pd.DataFrame()
    combined = pd.concat(values, axis=0)
    if not isinstance(combined.index, pd.DatetimeIndex):
        raise OfficialSourceError(f"{source} combined index is not DatetimeIndex")
    if not combined.index.is_monotonic_increasing:
        raise OfficialSourceError(f"{source} monthly sources are not in chronological order")
    if not combined.index.is_unique:
        duplicates = combined.index[combined.index.duplicated(keep=False)].unique()
        raise OfficialSourceError(
            f"{source} monthly sources contain duplicate timestamps: "
            + ", ".join(str(value) for value in duplicates[:5])
        )
    if not np.isfinite(combined.to_numpy(dtype=np.float64)).all():
        raise OfficialSourceError(f"{source} monthly sources contain NaN or infinite values")
    return combined


def _fetch_archive_series(
    source: str,
    *,
    symbol: str,
    interval: str,
    months: Iterable[pd.Timestamp],
    timeout: float,
    session: requests.Session,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for month in months:
        try:
            frame, record = fetch_archive_month(
                source,
                symbol=symbol,
                interval=interval,
                month=month,
                timeout=timeout,
                session=session,
            )
        except (OSError, requests.RequestException, OfficialSourceError) as exc:
            from unidream.data.official_v4_sources import official_archive_url

            record = _empty_source_record(
                source,
                official_archive_url(source, symbol, interval, month),
                exc,
            )
            frame = pd.DataFrame()
        records.append(record)
        if record.get("http_status") != 200 or frame.empty:
            raise OfficialSourceError(
                f"official {source} archive unavailable for {month:%Y-%m}: {record.get('error', 'empty response')}"
            )
        frames.append(frame)
    return _concat_source_frames(frames, source=source), records


def _missing_runs(
    expected: pd.DatetimeIndex,
    observed: pd.DatetimeIndex,
    *,
    interval: str,
) -> list[dict[str, Any]]:
    delta = _interval_delta(interval)
    missing = expected[~expected.isin(observed)]
    if len(missing) == 0:
        return []
    positions = expected.get_indexer(missing)
    runs: list[np.ndarray] = []
    for run in np.split(positions, np.where(np.diff(positions) != 1)[0] + 1):
        if len(run):
            runs.append(run)
    output: list[dict[str, Any]] = []
    for run in runs:
        timestamps = expected[run]
        output.append(
            {
                "gap_id": len(output),
                "left": str(expected[run[0] - 1]) if run[0] > 0 else None,
                "right": str(expected[run[-1] + 1]) if run[-1] + 1 < len(expected) else None,
                "expected_missing_count": int(len(timestamps)),
                "expected_missing_timestamps": [str(value) for value in timestamps],
                "interval": str(delta),
                "expected_positions": [int(value) for value in run],
            }
        )
    return output


def _merge_rest_rows(
    base: pd.DataFrame,
    extra: pd.DataFrame,
    *,
    source: str,
) -> pd.DataFrame:
    if extra.empty:
        return base
    overlap = base.index.intersection(extra.index)
    for timestamp in overlap:
        left = base.loc[timestamp].to_numpy(dtype=np.float64)
        right = extra.loc[timestamp].to_numpy(dtype=np.float64)
        if not np.array_equal(left, right):
            raise OfficialSourceError(f"{source} conflicting values at {timestamp}")
    addition = extra.loc[~extra.index.isin(base.index)]
    if addition.empty:
        return base
    combined = pd.concat([base, addition]).sort_index()
    if not combined.index.is_unique:
        raise OfficialSourceError(f"{source} REST merge produced duplicate timestamps")
    return combined


def recover_spot_gaps(
    spot: pd.DataFrame,
    *,
    expected: pd.DatetimeIndex,
    symbol: str,
    interval: str,
    timeout: float,
    session: requests.Session,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover only missing expected bars through official Spot REST."""
    delta = _interval_delta(interval)
    gaps = _missing_runs(expected, spot.index, interval=interval)
    rest_records: list[dict[str, Any]] = []
    for gap in gaps:
        first = _normalise_timestamp(gap["expected_missing_timestamps"][0])
        last = _normalise_timestamp(gap["expected_missing_timestamps"][-1])
        left = max(expected[0], first - delta)
        right = min(expected[-1] + delta, last + delta)
        try:
            fetched, record = fetch_spot_rest_window(
                symbol=symbol,
                interval=interval,
                start=left,
                end=right,
                timeout=timeout,
                session=session,
            )
        except (OSError, requests.RequestException, OfficialSourceError) as exc:
            record = {
                "source": "spot_klines_rest",
                "requested_url": "https://data-api.binance.vision/api/v3/klines",
                "final_url": None,
                "http_status": None,
                "response_bytes": 0,
                "response_sha256": None,
                "request_params": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
            fetched = pd.DataFrame()
        expected_timestamps = pd.DatetimeIndex(
            [_normalise_timestamp(value) for value in gap["expected_missing_timestamps"]]
        )
        recovered = fetched.index.intersection(expected_timestamps)
        gap["official_rest_covered_count"] = int(len(recovered))
        gap["official_rest_covered_timestamps"] = [str(value) for value in recovered]
        gap["official_rest_missing_count"] = int(len(expected_timestamps) - len(recovered))
        gap["official_rest_missing_timestamps"] = [
            str(value) for value in expected_timestamps[~expected_timestamps.isin(recovered)]
        ]
        record["gap_id"] = gap["gap_id"]
        record["expected_missing_count"] = int(len(expected_timestamps))
        record["covered_count"] = int(len(recovered))
        rest_records.append(record)
        if len(recovered):
            spot = _merge_rest_rows(spot, fetched.loc[recovered], source="spot")
    return spot, gaps, rest_records


def _contiguous_segments(frame: pd.DataFrame, *, interval: str) -> list[pd.DataFrame]:
    if frame.empty:
        return []
    delta = _interval_delta(interval)
    differences = frame.index[1:] - frame.index[:-1]
    boundaries = np.flatnonzero(np.asarray(differences != delta)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(frame)]))
    return [frame.iloc[int(start) : int(end)] for start, end in zip(starts, ends)]


def _asof_available(
    target_index: pd.DatetimeIndex,
    source_index: pd.DatetimeIndex,
    *,
    decision_delta: pd.Timedelta,
    max_age: pd.Timedelta,
) -> np.ndarray:
    if len(source_index) == 0:
        return np.zeros(len(target_index), dtype=bool)
    decision_times = target_index - decision_delta
    positions = source_index.searchsorted(decision_times, side="right") - 1
    available = positions >= 0
    safe_positions = np.maximum(positions, 0)
    ages = decision_times - source_index.take(safe_positions)
    return np.asarray(available & (ages >= pd.Timedelta(0)) & (ages <= max_age), dtype=bool)


def build_full_grid_availability(
    expected: pd.DatetimeIndex,
    *,
    spot_index: pd.DatetimeIndex,
    mark_index: pd.DatetimeIndex,
    funding_index: pd.DatetimeIndex,
    interval: str,
) -> pd.DataFrame:
    """Build masks using as-of timestamps without adding model columns."""
    delta = _interval_delta(interval)
    availability = pd.DataFrame(index=expected)
    availability["spot_bar_observed"] = expected.isin(spot_index)
    availability["funding_rate_available"] = _asof_available(
        expected,
        funding_index,
        decision_delta=delta,
        max_age=pd.Timedelta(hours=8),
    )
    availability["mark_close_available"] = _asof_available(
        expected,
        mark_index,
        decision_delta=delta,
        max_age=delta,
    )
    return availability.astype(bool)


def compute_v4_frames(
    spot: pd.DataFrame,
    *,
    funding: pd.DataFrame,
    mark: pd.DataFrame,
    zscore_window_days: int,
    interval: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute causal 17 features and returns on contiguous Spot segments."""
    feature_parts: list[pd.DataFrame] = []
    return_parts: list[pd.Series] = []
    for segment in _contiguous_segments(spot, interval=interval):
        features = compute_features(
            segment,
            zscore_window_days=zscore_window_days,
            interval=interval,
            funding_df=funding,
            oi_df=None,
            mark_price_df=mark,
        )
        returns = get_raw_returns(segment)
        common = features.index.intersection(returns.index)
        if len(common):
            feature_parts.append(features.loc[common])
            return_parts.append(returns.loc[common])
    if not feature_parts:
        raise OfficialSourceError("official raw data produced no finite feature rows")
    features = pd.concat(feature_parts, axis=0)
    returns = pd.concat(return_parts, axis=0).rename("returns")
    if not features.index.is_unique or not features.index.is_monotonic_increasing:
        raise OfficialSourceError("computed v4 feature index is not sorted and unique")
    if not returns.index.equals(features.index):
        raise OfficialSourceError("computed v4 features and returns are not aligned")
    actual_columns = [str(column) for column in features.columns]
    if actual_columns != list(MODEL_FEATURE_COLUMNS):
        missing = [column for column in MODEL_FEATURE_COLUMNS if column not in actual_columns]
        unexpected = [column for column in actual_columns if column not in MODEL_FEATURE_COLUMNS]
        raise OfficialSourceError(
            "computed feature schema mismatch"
            + (f"; missing={missing}" if missing else "")
            + (f"; unexpected={unexpected}" if unexpected else "")
        )
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all() or not np.isfinite(returns.to_numpy()).all():
        raise OfficialSourceError("computed v4 frames contain NaN or infinite values")
    return features, returns


def rebuild_official_v4_frames(
    *,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    start: Any = DEFAULT_START,
    end: Any = DEFAULT_END,
    zscore_window_days: int = 60,
    source_probe: Mapping[str, Any] | None = None,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch official raw sources and return frames plus provenance evidence."""
    start_value = _normalise_timestamp(start)
    end_value = _normalise_timestamp(end)
    if start_value != DEFAULT_START or end_value != DEFAULT_END:
        raise OfficialSourceError("official v4 rebuild is restricted to [2018-01-01, 2024-01-01)")
    expected = _expected_grid(start_value, end_value, interval)
    months = _month_range(start_value, end_value)
    owns_session = session is None
    active = session or requests.Session()
    try:
        spot, spot_records = _fetch_archive_series(
            "spot_klines",
            symbol=symbol,
            interval=interval,
            months=months,
            timeout=timeout,
            session=active,
        )
        spot = spot.loc[(spot.index >= start_value) & (spot.index < end_value)]
        spot, gap_records, rest_records = recover_spot_gaps(
            spot,
            expected=expected,
            symbol=symbol,
            interval=interval,
            timeout=timeout,
            session=active,
        )
        mark_months = [month for month in months if month >= EXTERNAL_ARCHIVE_START]
        funding_months = mark_months
        mark, mark_records = _fetch_archive_series(
            "um_mark_price_klines",
            symbol=symbol,
            interval=interval,
            months=mark_months,
            timeout=timeout,
            session=active,
        )
        funding, funding_records = _fetch_archive_series(
            "um_funding_rate",
            symbol=symbol,
            interval=interval,
            months=funding_months,
            timeout=timeout,
            session=active,
        )
    finally:
        if owns_session:
            active.close()

    if not spot.index.isin(expected).all():
        raise OfficialSourceError("official Spot source returned bars outside rebuild scope")
    mark = mark.loc[(mark.index >= start_value) & (mark.index < end_value)]
    funding = funding.loc[(funding.index >= start_value) & (funding.index < end_value)]
    features, returns = compute_v4_frames(
        spot,
        funding=funding,
        mark=mark,
        zscore_window_days=zscore_window_days,
        interval=interval,
    )
    availability = build_full_grid_availability(
        expected,
        spot_index=spot.index,
        mark_index=mark.index,
        funding_index=funding.index,
        interval=interval,
    )
    unresolved = int((~availability["spot_bar_observed"]).sum())
    recovered = int(sum(gap.get("official_rest_covered_count", 0) for gap in gap_records))
    provenance = {
        "kind": "official_binance_v4_rebuild",
        "source_policy": {
            "allowed_hosts": [
                "data-api.binance.vision",
                "data.binance.vision",
                "fapi.binance.com",
            ],
            "archive_only_for_monthly_raw": True,
            "non_official_provider_used": False,
            "interpolation_used": False,
        },
        "scope": {
            "symbol": symbol,
            "interval": interval,
            "start": str(start_value),
            "end_exclusive": str(end_value),
        },
        "source_probe": dict(source_probe) if source_probe is not None else None,
        "spot_archive_records": spot_records,
        "spot_rest_gap_records": rest_records,
        "mark_archive_records": mark_records,
        "funding_archive_records": funding_records,
        "external_source_start": str(EXTERNAL_ARCHIVE_START),
        "pre_external_start_policy": "availability_false; feature_value_zero_only_when_unavailable",
        "spot_gap_policy": "unresolved_bars_false; exclude_sequence_windows; no_interpolation",
        "spot_gap_summary": {
            "gap_count": len(gap_records),
            "expected_missing_bars_before_rest": int(sum(g["expected_missing_count"] for g in gap_records)),
            "official_rest_recovered_bars": recovered,
            "unresolved_spot_bars_after_rest": unresolved,
            "gap_records": gap_records,
        },
        "availability_columns": [
            "spot_bar_observed",
            "funding_rate_available",
            "mark_close_available",
        ],
    }
    return {
        "features": features,
        "returns": returns,
        "availability": availability,
        "provenance": provenance,
        "summary": {
            "scope_expected_bars": len(expected),
            "spot_observed_bars": int(availability["spot_bar_observed"].sum()),
            "spot_unresolved_bars": unresolved,
            "rest_recovered_bars": recovered,
            "feature_rows": len(features),
            "external_source_start": str(EXTERNAL_ARCHIVE_START),
            "external_pre_start_false": True,
            "status": "generated_with_explicit_spot_gaps" if unresolved else "generated",
        },
    }


__all__ = [
    "DEFAULT_END",
    "DEFAULT_INTERVAL",
    "DEFAULT_START",
    "DEFAULT_SYMBOL",
    "EXTERNAL_ARCHIVE_START",
    "OfficialSourceError",
    "build_full_grid_availability",
    "compute_v4_frames",
    "rebuild_official_v4_frames",
    "recover_spot_gaps",
]
