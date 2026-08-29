"""Regenerate a schema-v4 cache from official Binance raw data.

The rebuild is intentionally separate from the legacy runtime cache writer.
It fetches official monthly archives, probes missing Spot timestamps through
the official Spot REST endpoint, computes the existing causal feature
functions on each contiguous observed segment, and writes a full-grid
availability sidecar.  Unresolved bars remain absent from the body and false
in the sidecar; no interpolation is performed.
"""
from __future__ import annotations

import hashlib
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


def _source_month_for_timestamp(
    timestamp: pd.Timestamp,
    records: Iterable[Mapping[str, Any]],
) -> str | None:
    """Return the archive month whose parsed range contains ``timestamp``."""
    for record in records:
        first = record.get("parsed_first_timestamp")
        last = record.get("parsed_last_timestamp")
        if not first or not last:
            continue
        if _normalise_timestamp(first) <= timestamp <= _normalise_timestamp(last):
            # Monthly archive records carry an explicit month. REST records
            # deliberately do not: their calendar month is derived in the
            # human-readable diagnostic, but must not be mistaken for an
            # archive source when choosing the provenance branch.
            month = record.get("month")
            return str(month) if month else None
    return None


def _spot_timestamp_diagnostic(
    index: pd.DatetimeIndex,
    timestamp: pd.Timestamp,
    *,
    source_records: Iterable[Mapping[str, Any]],
    source: str,
    interval: str,
    scope_start: pd.Timestamp,
    scope_end: pd.Timestamp,
) -> str:
    """Describe a rejected Spot timestamp for the rebuild ledger/error."""
    position = index.get_loc(timestamp)
    previous = index[position - 1] if position > 0 else None
    following = index[position + 1] if position + 1 < len(index) else None
    source_month = _source_month_for_timestamp(timestamp, source_records)
    remainder = (timestamp - scope_start) % _interval_delta(interval)
    parts = [
        f"first={timestamp}",
        f"source={source}",
        f"source_month={source_month or timestamp.strftime('%Y-%m')}",
        f"range=[{scope_start}, {scope_end})",
    ]
    if previous is not None:
        parts.append(f"previous={previous}")
        parts.append(f"delta_from_previous={timestamp - previous}")
    if following is not None:
        parts.append(f"next={following}")
        parts.append(f"delta_to_next={following - timestamp}")
    parts.append(f"interval={interval}")
    parts.append(f"grid_remainder={remainder}")
    return "; ".join(parts)


def _row_sha256(frame: pd.DataFrame, timestamp: pd.Timestamp) -> str:
    """Hash one parsed raw row without changing its values or timestamp."""
    row_hash = pd.util.hash_pandas_object(frame.loc[[timestamp]], index=True).to_numpy()
    return hashlib.sha256(row_hash.tobytes()).hexdigest()


def _quarantine_off_grid_spot(
    spot: pd.DataFrame,
    *,
    expected: pd.DatetimeIndex,
    source_records: list[dict[str, Any]],
    interval: str,
    scope_start: pd.Timestamp,
    scope_end: pd.Timestamp,
    allow_off_grid_quarantine: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Validate or quarantine invalid Spot rows with auditable records."""
    off_grid = spot.index[
        (spot.index >= scope_start)
        & (spot.index < scope_end)
        & ~spot.index.isin(expected)
    ]
    if len(off_grid) == 0:
        return spot, []
    if not allow_off_grid_quarantine:
        raise OfficialSourceError(
            "official Spot source returned an off-grid timestamp: "
            + _spot_timestamp_diagnostic(
                spot.index,
                off_grid[0],
                source_records=source_records,
                source="spot_klines_archive",
                interval=interval,
                scope_start=scope_start,
                scope_end=scope_end,
            )
        )

    grouped: dict[str, list[pd.Timestamp]] = {}
    for timestamp in off_grid:
        month = _source_month_for_timestamp(timestamp, source_records)
        if month is None:
            raise OfficialSourceError(
                "cannot quarantine off-grid Spot row without archive source month: "
                + _spot_timestamp_diagnostic(
                    spot.index,
                    timestamp,
                    source_records=source_records,
                    source="spot_klines_archive",
                    interval=interval,
                    scope_start=scope_start,
                    scope_end=scope_end,
                )
            )
        grouped.setdefault(month, []).append(timestamp)

    quarantines: list[dict[str, Any]] = []
    for month, timestamps in grouped.items():
        first = timestamps[0]
        row_records: list[dict[str, Any]] = []
        for timestamp in timestamps:
            position = spot.index.get_loc(timestamp)
            previous = spot.index[position - 1] if position > 0 else None
            following = spot.index[position + 1] if position + 1 < len(spot.index) else None
            row: dict[str, Any] = {
                "timestamp": str(timestamp),
                "row_sha256": _row_sha256(spot, timestamp),
                "grid_remainder": str((timestamp - scope_start) % _interval_delta(interval)),
            }
            if previous is not None:
                row["previous_timestamp"] = str(previous)
                row["delta_from_previous"] = str(timestamp - previous)
            if following is not None:
                row["next_timestamp"] = str(following)
                row["delta_to_next"] = str(following - timestamp)
            row_records.append(row)
        first_row = row_records[0]
        entry: dict[str, Any] = {
            "source": "spot_klines_archive",
            "source_month": month,
            "quarantined_count": len(timestamps),
            "timestamps": [str(value) for value in timestamps],
            "row_sha256": [row["row_sha256"] for row in row_records],
            "rows": row_records,
            "first_timestamp": str(first),
            "last_timestamp": str(timestamps[-1]),
            "grid_remainder": first_row["grid_remainder"],
            "policy": "explicit_off_grid_quarantine; no_timestamp_remap; no_interpolation",
        }
        for key in (
            "previous_timestamp",
            "delta_from_previous",
            "next_timestamp",
            "delta_to_next",
        ):
            if key in first_row:
                entry[key] = first_row[key]
        quarantines.append(entry)

        for record in source_records:
            if record.get("month") == month:
                record["off_grid_quarantine_count"] = len(timestamps)
                record["off_grid_quarantine_first_timestamp"] = str(first)
                record["off_grid_quarantine_row_sha256"] = entry["row_sha256"]
                record["off_grid_quarantine_policy"] = entry["policy"]
                break

    return spot.drop(index=off_grid), quarantines


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
        out_of_scope = fetched.index[~fetched.index.isin(expected)]
        record["outside_expected_count"] = int(len(out_of_scope))
        record["outside_expected_timestamps"] = [str(value) for value in out_of_scope[:5]]
        # A recovery request may include the end-exclusive boundary because
        # Binance REST treats endTime as inclusive.  That one timestamp is
        # explicitly audited and may be discarded; every other unexpected or
        # off-grid response fails closed instead of being hidden by filtering.
        allowed_boundary = pd.DatetimeIndex([expected[-1] + delta])
        unexpected = out_of_scope[~out_of_scope.isin(allowed_boundary)]
        if len(unexpected):
            diagnostic = _spot_timestamp_diagnostic(
                fetched.index,
                unexpected[0],
                source_records=[record],
                source="spot_klines_rest",
                interval=interval,
                scope_start=expected[0],
                scope_end=expected[-1] + delta,
            )
            raise OfficialSourceError(
                "official Spot REST returned an unexpected timestamp: " + diagnostic
            )
        record["outside_expected_allowed_timestamps"] = [str(value) for value in out_of_scope]
        record["outside_expected_policy"] = (
            "only exact end-exclusive boundary is allowed and excluded; all other rows fail closed"
        )
        # The REST window intentionally includes one observed boundary bar on
        # either side.  Record and discard those boundary rows before merging;
        # never let a recovery request expand the declared cache scope.
        fetched_in_scope = fetched.loc[fetched.index.isin(expected)]
        recovered = fetched_in_scope.index.intersection(expected_timestamps)
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
            spot = _merge_rest_rows(spot, fetched_in_scope.loc[recovered], source="spot")
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
    # Mark is a candle-level diagnostic input at the same causal timestamp as
    # the shifted feature row: target t uses the exact mark observation at
    # decision time t-interval. A prior mark candle is not an exact
    # observation for this target, even though compute_basis may safely carry
    # its value forward for model feature calculation.
    availability["mark_close_available"] = (expected - delta).isin(mark_index)
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
    allow_off_grid_quarantine: bool = False,
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
        # Monthly archives are requested by calendar month, but validate the
        # exclusive rebuild boundary before applying it. Binance archives may
        # expose the first bar of the following month in a month-end payload;
        # only that exact boundary row is an accepted, recorded drop.
        out_of_scope = spot.index[(spot.index < start_value) | (spot.index >= end_value)]
        if len(out_of_scope):
            non_boundary = out_of_scope[out_of_scope != end_value]
            if len(non_boundary):
                diagnostic = _spot_timestamp_diagnostic(
                    spot.index,
                    non_boundary[0],
                    source_records=spot_records,
                    source="spot_klines_archive",
                    interval=interval,
                    scope_start=start_value,
                    scope_end=end_value,
                )
                raise OfficialSourceError(
                    "official Spot archive returned an unexpected out-of-scope timestamp: "
                    + diagnostic
                )
            for record in spot_records:
                boundary_count = sum(
                    1
                    for value in out_of_scope
                    if value == end_value
                    and record.get("parsed_first_timestamp")
                    and _normalise_timestamp(record["parsed_first_timestamp"])
                    <= value
                    <= _normalise_timestamp(record.get("parsed_last_timestamp"))
                )
                if boundary_count:
                    record["scope_boundary_rows_dropped"] = boundary_count
                    record["scope_boundary_policy"] = (
                        "validated exact end-exclusive boundary row; excluded from rebuild"
                    )
        spot = spot.loc[(spot.index >= start_value) & (spot.index < end_value)]
        off_grid = spot.index[~spot.index.isin(expected)]
        spot_off_grid_quarantine: list[dict[str, Any]] = []
        if len(off_grid):
            spot, spot_off_grid_quarantine = _quarantine_off_grid_spot(
                spot,
                expected=expected,
                source_records=spot_records,
                interval=interval,
                scope_start=start_value,
                scope_end=end_value,
                allow_off_grid_quarantine=allow_off_grid_quarantine,
            )
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

    out_of_range = spot.index[(spot.index < start_value) | (spot.index >= end_value)]
    off_grid = spot.index[~spot.index.isin(expected)]
    if len(out_of_range):
        diagnostic = _spot_timestamp_diagnostic(
            spot.index,
            out_of_range[0],
            source_records=rest_records,
            source="spot_klines_rest",
            interval=interval,
            scope_start=start_value,
            scope_end=end_value,
        )
        raise OfficialSourceError(
            "official Spot REST returned a bar outside rebuild scope: " + diagnostic
        )
    if len(off_grid):
        first = off_grid[0]
        rest_month = _source_month_for_timestamp(first, rest_records)
        source = "spot_klines_rest" if rest_month is not None else "spot_klines_archive"
        diagnostic = _spot_timestamp_diagnostic(
            spot.index,
            first,
            source_records=rest_records if rest_month is not None else spot_records,
            source=source,
            interval=interval,
            scope_start=start_value,
            scope_end=end_value,
        )
        raise OfficialSourceError(
            "official Spot source returned an off-grid timestamp: " + diagnostic
        )
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
    availability_names = [
        "spot_bar_observed",
        "funding_rate_available",
        "mark_close_available",
    ]
    availability_coverage: dict[str, Any] = {}
    for name in availability_names:
        true_count = int(availability[name].sum())
        availability_coverage[name] = {
            "true_count": true_count,
            "total_count": int(len(availability)),
            "true_fraction": float(true_count / len(availability)) if len(availability) else 0.0,
        }
    both_external = availability["funding_rate_available"] & availability["mark_close_available"]
    all_three = availability[availability_names].all(axis=1)
    for name, mask in (("funding_and_mark_available", both_external), ("all_three_available", all_three)):
        true_count = int(mask.sum())
        availability_coverage[name] = {
            "true_count": true_count,
            "total_count": int(len(availability)),
            "true_fraction": float(true_count / len(availability)) if len(availability) else 0.0,
        }
    feature_all_three = int(availability.loc[features.index, availability_names].all(axis=1).sum())
    post_rest_gap_runs = _missing_runs(expected, spot.index, interval=interval)
    feature_rows_by_quality = {
        "feature_rows": int(len(features)),
        "observed_spot_rows": int(availability["spot_bar_observed"].sum()),
        "observed_spot_minus_feature_rows": int(availability["spot_bar_observed"].sum() - len(features)),
        "all_three_available_rows": feature_all_three,
        "not_all_three_available_rows": int(len(features) - feature_all_three),
        "body_policy": "compute causal features on observed contiguous Spot segments; do not filter body by external masks",
        "reduction_reason": "rolling indicator warmup and unresolved Spot gaps split segments; each segment drops its own invalid warmup rows",
    }
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
        "availability_semantics": {
            "spot_bar_observed": "exact Spot bar at row timestamp t",
            "funding_rate_available": "funding observation as-of decision timestamp t-interval, age <= 8h",
            "mark_close_available": "exact mark candle at causal decision timestamp t-interval",
            "basis_history_requirement": "basis_mom and basis_abs also depend on prior causal mark rows; full17 training must validate that history per sequence",
        },
        "availability_coverage": availability_coverage,
        "feature_rows_by_quality": feature_rows_by_quality,
        "spot_gap_policy": "unresolved_bars_false; exclude_sequence_windows; no_interpolation",
        "spot_off_grid_quarantine": spot_off_grid_quarantine,
        "spot_gap_summary": {
            "gap_count": len(gap_records),
            "expected_missing_bars_before_rest": int(sum(g["expected_missing_count"] for g in gap_records)),
            "official_rest_recovered_bars": recovered,
            "unresolved_spot_bars_after_rest": unresolved,
            "gap_count_after_rest": len(post_rest_gap_runs),
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
            "quarantined_off_grid_spot_bars": int(
                sum(item["quarantined_count"] for item in spot_off_grid_quarantine)
            ),
            "feature_rows": len(features),
            "external_source_start": str(EXTERNAL_ARCHIVE_START),
            "external_pre_start_false": True,
            "availability_coverage": availability_coverage,
            "feature_rows_by_quality": feature_rows_by_quality,
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
