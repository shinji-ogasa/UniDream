"""Fail-closed schema v4 cache with a separate availability sidecar.

The feature parquet remains the model input: exactly the canonical 17 feature
columns and no availability flags.  Availability is kept in a separate
parquet indexed by the complete expected bar grid.  A missing spot bar is
therefore represented by ``spot_bar_observed=False`` rather than by a
synthetic feature row or an ambiguous zero.

This module deliberately does not upgrade or overwrite the historical v3
cache.  Callers must opt into the v4 file names (or pass ``require_v4`` to the
training runtime) and validate every cache hit before consuming it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd


CACHE_V4_SCHEMA_VERSION = 4
DEFAULT_INTERVAL = "15m"
DEFAULT_GAP_POLICY = "exclude_windows_crossing_gaps"
MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "open_ret",
    "high_ret",
    "low_ret",
    "close_ret",
    "vol_ret",
    "RSI_14",
    "macd",
    "macd_signal",
    "atr_norm_ret",
    "atr",
    "rv_4",
    "rv_16",
    "rv_96",
    "funding_rate",
    "basis",
    "basis_mom",
    "basis_abs",
)
REQUIRED_AVAILABILITY_COLUMNS: tuple[str, ...] = (
    "spot_bar_observed",
    "funding_rate_available",
    "mark_close_available",
)


class CacheV4Error(ValueError):
    """Raised when a schema v4 cache or its metadata fails validation."""

    def __init__(self, issues: Iterable[str] | str):
        if isinstance(issues, str):
            values = [issues]
        else:
            values = [str(value) for value in issues]
        self.issues = tuple(values)
        super().__init__("cache v4 validation failed: " + "; ".join(values))


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _frame_content_digest(frame: pd.DataFrame) -> str:
    """Hash index, labels, dtypes, and values of one materialized frame."""
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    if not isinstance(frame, pd.DataFrame):
        raise CacheV4Error("content digest input must be a DataFrame")
    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated(keep=False)].unique()
        raise CacheV4Error(
            "frame contains duplicate columns: "
            + ", ".join(str(value) for value in duplicates[:5])
        )
    descriptor = {
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "index_name": frame.index.name,
        "index_dtype": str(frame.index.dtype),
        "columns": [str(column) for column in frame.columns],
        "column_dtypes": [str(dtype) for dtype in frame.dtypes],
    }
    try:
        row_hashes = pd.util.hash_pandas_object(
            frame,
            index=True,
            categorize=True,
        ).to_numpy(dtype=np.uint64, copy=False)
    except (TypeError, ValueError) as exc:
        raise CacheV4Error(f"could not hash frame content: {exc}") from exc
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            descriptor,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    digest.update(row_hashes.tobytes(order="C"))
    return digest.hexdigest()


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
        raise CacheV4Error(f"unsupported cache interval: {interval!r}")
    return values[interval]


def _path_value(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def cache_v4_paths(cache_dir: str | Path, cache_tag: str) -> dict[str, Path]:
    """Return the explicit v4 body, returns, sidecar, and metadata paths."""
    root = _path_value(cache_dir)
    return {
        "features": root / f"{cache_tag}_features.parquet",
        "returns": root / f"{cache_tag}_returns.parquet",
        "availability": root / f"{cache_tag}_availability.parquet",
        "metadata": root / f"{cache_tag}_metadata.json",
    }


def _normalise_paths(
    *,
    cache_dir: str | Path | None,
    cache_tag: str | None,
    feature_path: str | Path | None,
    returns_path: str | Path | None,
    availability_path: str | Path | None,
    metadata_path: str | Path | None,
) -> dict[str, Path]:
    explicit = {
        "features": feature_path,
        "returns": returns_path,
        "availability": availability_path,
        "metadata": metadata_path,
    }
    if any(value is not None for value in explicit.values()):
        if any(value is None for value in explicit.values()):
            raise CacheV4Error(
                "feature_path, returns_path, availability_path, and metadata_path must be supplied together"
            )
        return {key: _path_value(value) for key, value in explicit.items()}
    if cache_dir is None or cache_tag is None:
        raise CacheV4Error("cache_dir and cache_tag are required when explicit v4 paths are absent")
    return cache_v4_paths(cache_dir, cache_tag)


def _as_datetime_index(index: Any, *, name: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise CacheV4Error(f"{name} index is not a DatetimeIndex")
    if len(index) == 0:
        raise CacheV4Error(f"{name} index is empty")
    if not index.is_unique:
        duplicates = index[index.duplicated(keep=False)].unique()
        names = ", ".join(str(value) for value in duplicates[:5])
        raise CacheV4Error(f"{name} index has duplicate timestamps: {names}")
    if not index.is_monotonic_increasing:
        raise CacheV4Error(f"{name} index is not strictly increasing; refusing to sort")
    return index


def _as_returns_frame(returns: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(returns, pd.Series):
        return returns.to_frame(name="returns")
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] != 1:
            raise CacheV4Error("returns must contain exactly one column")
        frame = returns.copy()
        frame.columns = ["returns"]
        return frame
    raise CacheV4Error("returns must be a pandas Series or one-column DataFrame")


def _normalise_gap_policy(value: str | Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        name = DEFAULT_GAP_POLICY
        policy: dict[str, Any] = {}
    elif isinstance(value, str):
        name = value
        policy = {}
    elif isinstance(value, Mapping):
        policy = dict(value)
        name = str(policy.get("name", ""))
    else:
        raise CacheV4Error("gap_policy must be a string or mapping")
    if name != DEFAULT_GAP_POLICY:
        raise CacheV4Error(
            f"unsupported gap_policy {name!r}; expected {DEFAULT_GAP_POLICY!r}"
        )
    policy.update(
        {
            "name": DEFAULT_GAP_POLICY,
            "interpolation": "forbidden",
            "missing_bar_action": "exclude_sequence_windows",
            "execution_evaluation": "segment_or_explicit_pre_gap_attribution",
        }
    )
    return policy


def _schema_digest(
    *,
    feature_columns: Sequence[str],
    availability_columns: Sequence[str],
    interval: str,
    gap_policy: Mapping[str, Any],
) -> str:
    return _canonical_sha256(
        {
            "schema_version": CACHE_V4_SCHEMA_VERSION,
            "feature_columns": list(feature_columns),
            "availability_columns": list(availability_columns),
            "returns_columns": ["returns"],
            "interval": interval,
            "gap_policy": dict(gap_policy),
        }
    )


def _timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return parsed


def _timestamp_strings(values: Iterable[Any]) -> list[str]:
    return [str(_timestamp(value)) for value in values]


def _gap_list_from_mask(
    index: pd.DatetimeIndex,
    observed: pd.Series,
    *,
    interval: str,
) -> list[dict[str, Any]]:
    delta = _interval_delta(interval)
    false_positions = np.flatnonzero(~observed.to_numpy(dtype=bool))
    gaps: list[dict[str, Any]] = []
    for run in np.split(false_positions, np.where(np.diff(false_positions) != 1)[0] + 1):
        if len(run) == 0:
            continue
        timestamps = index[run]
        left = index[run[0] - 1] if run[0] > 0 else None
        right = index[run[-1] + 1] if run[-1] + 1 < len(index) else None
        gaps.append(
            {
                "gap_id": len(gaps),
                "left": str(left) if left is not None else None,
                "right": str(right) if right is not None else None,
                "delta": str(right - left) if left is not None and right is not None else None,
                "expected_missing_count": int(len(timestamps)),
                "expected_missing_timestamps": [str(value) for value in timestamps],
                "interval": str(delta),
            }
        )
    return gaps


def _gap_timestamp_sets(gaps: Sequence[Mapping[str, Any]]) -> list[set[pd.Timestamp]]:
    sets: list[set[pd.Timestamp]] = []
    for gap in gaps:
        values = gap.get("expected_missing_timestamps")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise CacheV4Error("gap_list entries must contain expected_missing_timestamps")
        sets.append({_timestamp(value) for value in values})
    return sets


def _validate_gap_list(
    metadata_gaps: Any,
    derived_gaps: Sequence[Mapping[str, Any]],
) -> list[str]:
    issues: list[str] = []
    if not isinstance(metadata_gaps, list):
        return ["metadata gap_list must be a list"]
    try:
        actual_sets = _gap_timestamp_sets(metadata_gaps)
        expected_sets = _gap_timestamp_sets(derived_gaps)
    except (CacheV4Error, TypeError, ValueError) as exc:
        return [str(exc)]
    if sorted(actual_sets, key=lambda values: min(values) if values else pd.Timestamp.max) != sorted(
        expected_sets,
        key=lambda values: min(values) if values else pd.Timestamp.max,
    ):
        actual = sorted(str(value) for values in actual_sets for value in values)
        expected = sorted(str(value) for values in expected_sets for value in values)
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        if missing:
            issues.append(f"metadata gap_list is missing timestamps: {missing[:5]}")
        if unexpected:
            issues.append(f"metadata gap_list has unexpected timestamps: {unexpected[:5]}")
        if not missing and not unexpected:
            issues.append("metadata gap_list grouping does not match availability sidecar")
    return issues


def _validate_frames(
    features: pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    availability: pd.DataFrame,
    metadata: Mapping[str, Any],
    *,
    expected_cache_tag: str | None = None,
) -> None:
    issues: list[str] = []
    if not isinstance(features, pd.DataFrame):
        issues.append("features must be a DataFrame")
    if not isinstance(availability, pd.DataFrame):
        issues.append("availability must be a DataFrame")
    if issues:
        raise CacheV4Error(issues)
    returns_frame: pd.DataFrame
    try:
        returns_frame = _as_returns_frame(returns)
    except CacheV4Error as exc:
        issues.extend(exc.issues)
        returns_frame = pd.DataFrame()

    feature_index: pd.DatetimeIndex | None = None
    returns_index: pd.DatetimeIndex | None = None
    availability_index: pd.DatetimeIndex | None = None
    for frame, name in (
        (features, "features"),
        (returns_frame, "returns"),
        (availability, "availability sidecar"),
    ):
        try:
            normalized = _as_datetime_index(frame.index, name=name)
        except CacheV4Error as exc:
            issues.extend(exc.issues)
            continue
        if name == "features":
            feature_index = normalized
        elif name == "returns":
            returns_index = normalized
        else:
            availability_index = normalized

    if feature_index is not None and returns_index is not None and not feature_index.equals(returns_index):
        issues.append("features and returns indexes are not exactly aligned")
    if feature_index is not None and availability_index is not None:
        if not feature_index.isin(availability_index).all():
            missing = feature_index[~feature_index.isin(availability_index)]
            issues.append(
                "feature timestamps missing from availability sidecar: "
                + ", ".join(str(value) for value in missing[:5])
            )
        elif "spot_bar_observed" in availability:
            observed_on_body = availability.loc[feature_index, "spot_bar_observed"]
            if not observed_on_body.all():
                missing = feature_index[~observed_on_body.to_numpy(dtype=bool)]
                issues.append(
                    "feature rows marked spot_bar_observed=False: "
                    + ", ".join(str(value) for value in missing[:5])
                )
        else:
            issues.append("availability sidecar is missing required column: spot_bar_observed")

    metadata_columns = metadata.get("feature_columns")
    if not isinstance(metadata_columns, list):
        issues.append("metadata feature_columns must be a list")
        metadata_columns = []
    if len(set(str(column) for column in metadata_columns)) != len(metadata_columns):
        issues.append("metadata feature_columns contains duplicate columns")
    if metadata_columns != list(MODEL_FEATURE_COLUMNS):
        issues.append(
            "metadata feature_columns must exactly equal canonical 17 columns: "
            f"{list(MODEL_FEATURE_COLUMNS)}"
        )
    actual_columns = [str(column) for column in features.columns]
    if actual_columns != metadata_columns:
        missing = [column for column in metadata_columns if column not in actual_columns]
        unexpected = [column for column in actual_columns if column not in metadata_columns]
        issues.append(
            "feature column order/schema mismatch"
            + (f"; missing columns: {missing}" if missing else "")
            + (f"; unexpected columns: {unexpected}" if unexpected else "")
        )
    if len(actual_columns) != len(set(actual_columns)):
        issues.append("features contains duplicate columns")
    if metadata.get("returns_columns") != ["returns"]:
        issues.append("metadata returns_columns must be exactly ['returns']")
    elif len(returns_frame.columns) == 1 and str(returns_frame.columns[0]) != "returns":
        issues.append(
            "returns column must be named 'returns'; "
            f"got {str(returns_frame.columns[0])!r}"
        )

    availability_columns = metadata.get("availability_columns")
    if not isinstance(availability_columns, list):
        issues.append("metadata availability_columns must be a list")
        availability_columns = []
    if len(set(str(column) for column in availability_columns)) != len(availability_columns):
        issues.append("metadata availability_columns contains duplicate columns")
    missing_availability = [
        column for column in REQUIRED_AVAILABILITY_COLUMNS if column not in availability_columns
    ]
    if missing_availability:
        issues.append(f"metadata availability_columns missing required columns: {missing_availability}")
    actual_availability_columns = [str(column) for column in availability.columns]
    if actual_availability_columns != availability_columns:
        missing = [column for column in availability_columns if column not in actual_availability_columns]
        unexpected = [column for column in actual_availability_columns if column not in availability_columns]
        issues.append(
            "availability column order/schema mismatch"
            + (f"; missing columns: {missing}" if missing else "")
            + (f"; unexpected columns: {unexpected}" if unexpected else "")
        )

    interval = str(metadata.get("interval", DEFAULT_INTERVAL))
    try:
        delta = _interval_delta(interval)
    except CacheV4Error as exc:
        issues.extend(exc.issues)
        delta = None
    if availability_index is not None and delta is not None and len(availability_index) > 1:
        differences = availability_index[1:] - availability_index[:-1]
        bad = np.flatnonzero(np.asarray(differences != delta))
        if len(bad):
            first = int(bad[0])
            issues.append(
                "availability sidecar index has non-contiguous interval at "
                f"{availability_index[first]} -> {availability_index[first + 1]}; refusing repair"
            )

    if availability_index is not None:
        for column in availability_columns:
            if column not in availability:
                continue
            if not pd.api.types.is_bool_dtype(availability[column].dtype):
                issues.append(f"availability column {column!r} must have boolean dtype")
            elif availability[column].isna().any():
                issues.append(f"availability column {column!r} contains NaN")

    try:
        feature_values = features.to_numpy(dtype=np.float64)
        returns_values = returns_frame.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        issues.append(f"features or returns contain non-numeric values: {exc}")
    else:
        if not np.isfinite(feature_values).all():
            issues.append("features contain NaN or infinite values")
        if not np.isfinite(returns_values).all():
            issues.append("returns contain NaN or infinite values")

    content_digests = metadata.get("content_digests")
    if not isinstance(content_digests, Mapping):
        issues.append("metadata content_digests must contain features, returns, and availability")
    else:
        for name, frame in (
            ("features", features),
            ("returns", returns_frame),
            ("availability", availability),
        ):
            try:
                actual_digest = _frame_content_digest(frame)
            except CacheV4Error as exc:
                issues.extend(exc.issues)
                continue
            expected_digest = content_digests.get(name)
            if expected_digest != actual_digest:
                issues.append(
                    f"{name} content digest mismatch: "
                    f"metadata={expected_digest!r}, actual={actual_digest}"
                )

    if metadata.get("rows") != len(features):
        issues.append(f"metadata rows mismatch: metadata={metadata.get('rows')!r}, actual={len(features)}")
    if metadata.get("sidecar_rows") != len(availability):
        issues.append(
            "metadata sidecar_rows mismatch: "
            f"metadata={metadata.get('sidecar_rows')!r}, actual={len(availability)}"
        )
    if feature_index is not None:
        for key, value in (
            ("first_timestamp", str(feature_index[0])),
            ("last_timestamp", str(feature_index[-1])),
        ):
            if metadata.get(key) != value:
                issues.append(f"metadata {key} mismatch: metadata={metadata.get(key)!r}, actual={value!r}")
    if availability_index is not None:
        for key, value in (
            ("sidecar_first_timestamp", str(availability_index[0])),
            ("sidecar_last_timestamp", str(availability_index[-1])),
        ):
            if metadata.get(key) != value:
                issues.append(f"metadata {key} mismatch: metadata={metadata.get(key)!r}, actual={value!r}")

    policy = metadata.get("gap_policy")
    try:
        normalized_policy = _normalise_gap_policy(policy if isinstance(policy, (str, Mapping)) else None)
    except CacheV4Error as exc:
        issues.extend(exc.issues)
        normalized_policy = None
    if normalized_policy is not None:
        if metadata.get("gap_policy") != normalized_policy:
            issues.append("metadata gap_policy is not canonical")

    if metadata.get("schema_version") != CACHE_V4_SCHEMA_VERSION:
        issues.append(
            f"metadata schema_version={metadata.get('schema_version')!r} is not v4; legacy v3 is not accepted"
        )
    if expected_cache_tag is not None and metadata.get("cache_tag") != expected_cache_tag:
        issues.append(
            "metadata cache_tag mismatch: "
            f"metadata={metadata.get('cache_tag')!r}, expected={expected_cache_tag!r}"
        )
    if isinstance(metadata.get("source_provenance"), Mapping):
        expected = _canonical_sha256(metadata["source_provenance"])
        if metadata.get("source_provenance_digest") != expected:
            issues.append(
                "source/provenance digest mismatch: "
                f"metadata={metadata.get('source_provenance_digest')!r}, expected={expected}"
            )
    else:
        issues.append("metadata source_provenance must be a mapping")

    if normalized_policy is not None:
        expected_schema_digest = _schema_digest(
            feature_columns=metadata_columns,
            availability_columns=availability_columns,
            interval=interval,
            gap_policy=normalized_policy,
        )
        if metadata.get("schema_digest") != expected_schema_digest:
            issues.append(
                "schema digest mismatch: "
                f"metadata={metadata.get('schema_digest')!r}, expected={expected_schema_digest}"
            )

    if availability_index is not None and "spot_bar_observed" in availability:
        derived_gaps = _gap_list_from_mask(
            availability_index,
            availability["spot_bar_observed"],
            interval=interval,
        )
        issues.extend(_validate_gap_list(metadata.get("gap_list"), derived_gaps))

    if issues:
        raise CacheV4Error(issues)


def build_v4_metadata(
    features: pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    availability: pd.DataFrame,
    *,
    source_provenance: Mapping[str, Any],
    cache_tag: str | None = None,
    symbol: str = "BTCUSDT",
    interval: str = DEFAULT_INTERVAL,
    start: str | None = None,
    end: str | None = None,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
    availability_columns: Sequence[str] | None = None,
    gap_policy: str | Mapping[str, Any] = DEFAULT_GAP_POLICY,
    gaps: Sequence[Mapping[str, Any]] | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical v4 metadata after validating the frame contract."""
    if not isinstance(source_provenance, Mapping):
        raise CacheV4Error("source_provenance must be a mapping")
    returns_frame = _as_returns_frame(returns)
    columns = list(feature_columns)
    sidecar_columns = list(availability_columns or availability.columns)
    normalized_policy = _normalise_gap_policy(gap_policy)
    _interval_delta(interval)
    if columns != list(MODEL_FEATURE_COLUMNS):
        raise CacheV4Error(
            "v4 feature body must contain the canonical 17 columns in order: "
            f"{list(MODEL_FEATURE_COLUMNS)}"
        )
    if gaps is None:
        if "spot_bar_observed" not in availability:
            raise CacheV4Error("availability sidecar is missing required column: spot_bar_observed")
        sidecar_index = _as_datetime_index(availability.index, name="availability sidecar")
        derived_gaps = _gap_list_from_mask(
            sidecar_index,
            availability["spot_bar_observed"],
            interval=interval,
        )
        gaps_list = derived_gaps
    else:
        gaps_list = [dict(gap) for gap in gaps]

    metadata: dict[str, Any] = {
        "schema_version": CACHE_V4_SCHEMA_VERSION,
        "cache_tag": cache_tag,
        "symbol": symbol,
        "interval": interval,
        "start": start,
        "end_exclusive": end,
        "parameters": dict(parameters) if parameters is not None else None,
        "feature_columns": columns,
        "availability_columns": sidecar_columns,
        "returns_columns": ["returns"],
        "rows": int(len(features)),
        "sidecar_rows": int(len(availability)),
        "first_timestamp": str(features.index[0]) if len(features) else None,
        "last_timestamp": str(features.index[-1]) if len(features) else None,
        "sidecar_first_timestamp": str(availability.index[0]) if len(availability) else None,
        "sidecar_last_timestamp": str(availability.index[-1]) if len(availability) else None,
        "source_provenance": dict(source_provenance),
        "source_provenance_digest": _canonical_sha256(source_provenance),
        "gap_list": gaps_list,
        "gap_policy": normalized_policy,
        "content_digests": {
            "features": _frame_content_digest(features),
            "returns": _frame_content_digest(returns_frame),
            "availability": _frame_content_digest(availability),
        },
    }
    metadata["schema_digest"] = _schema_digest(
        feature_columns=columns,
        availability_columns=sidecar_columns,
        interval=interval,
        gap_policy=normalized_policy,
    )
    _validate_frames(features, returns_frame, availability, metadata)
    return metadata


def validate_cache_v4(
    features: pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    availability: pd.DataFrame,
    metadata: Mapping[str, Any],
    *,
    expected_cache_tag: str | None = None,
) -> dict[str, Any]:
    """Validate a v4 cache and return a small verified status record."""
    if not isinstance(metadata, Mapping):
        raise CacheV4Error("metadata must be a mapping")
    _validate_frames(
        features,
        returns,
        availability,
        metadata,
        expected_cache_tag=expected_cache_tag,
    )
    return {
        "status": "v4_verified",
        "schema_version": CACHE_V4_SCHEMA_VERSION,
        "schema_digest": metadata.get("schema_digest"),
        "source_provenance_digest": metadata.get("source_provenance_digest"),
        "rows": len(features),
        "sidecar_rows": len(availability),
        "gap_count": len(metadata.get("gap_list", [])),
    }


def _atomic_parquet_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}")
    try:
        frame.to_parquet(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_write(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_cache_v4(
    features: pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    availability: pd.DataFrame,
    *,
    source_provenance: Mapping[str, Any],
    cache_dir: str | Path | None = None,
    cache_tag: str | None = None,
    feature_path: str | Path | None = None,
    returns_path: str | Path | None = None,
    availability_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    symbol: str = "BTCUSDT",
    interval: str = DEFAULT_INTERVAL,
    start: str | None = None,
    end: str | None = None,
    gap_policy: str | Mapping[str, Any] = DEFAULT_GAP_POLICY,
    gaps: Sequence[Mapping[str, Any]] | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and atomically write a future schema v4 cache."""
    paths = _normalise_paths(
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        feature_path=feature_path,
        returns_path=returns_path,
        availability_path=availability_path,
        metadata_path=metadata_path,
    )
    returns_frame = _as_returns_frame(returns)
    metadata = build_v4_metadata(
        features,
        returns_frame,
        availability,
        source_provenance=source_provenance,
        cache_tag=cache_tag,
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        gap_policy=gap_policy,
        gaps=gaps,
        parameters=parameters,
    )
    # If caller supplied gaps, validation still derives the sidecar gap list
    # and compares it before any file is touched.
    validate_cache_v4(
        features,
        returns_frame,
        availability,
        metadata,
        expected_cache_tag=cache_tag,
    )
    _atomic_parquet_write(features, paths["features"])
    _atomic_parquet_write(returns_frame.rename(columns={returns_frame.columns[0]: "returns"}), paths["returns"])
    _atomic_parquet_write(availability, paths["availability"])
    _atomic_json_write(metadata, paths["metadata"])
    return metadata


def load_cache_v4(
    cache_dir: str | Path | None = None,
    cache_tag: str | None = None,
    *,
    feature_path: str | Path | None = None,
    returns_path: str | Path | None = None,
    availability_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, Any]]:
    """Load and validate a complete v4 cache; never sort or repair it."""
    paths = _normalise_paths(
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        feature_path=feature_path,
        returns_path=returns_path,
        availability_path=availability_path,
        metadata_path=metadata_path,
    )
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise CacheV4Error(f"v4 cache is incomplete; missing files: {missing}")
    try:
        features = pd.read_parquet(paths["features"])
        returns_frame = pd.read_parquet(paths["returns"])
        availability = pd.read_parquet(paths["availability"])
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise CacheV4Error(f"could not read v4 cache: {type(exc).__name__}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise CacheV4Error("metadata must be a JSON object")
    validate_cache_v4(
        features,
        returns_frame,
        availability,
        metadata,
        expected_cache_tag=cache_tag,
    )
    returns = returns_frame.iloc[:, 0].rename("returns")
    return features, returns, availability, metadata


__all__ = [
    "CACHE_V4_SCHEMA_VERSION",
    "DEFAULT_GAP_POLICY",
    "MODEL_FEATURE_COLUMNS",
    "REQUIRED_AVAILABILITY_COLUMNS",
    "CacheV4Error",
    "build_v4_metadata",
    "cache_v4_paths",
    "load_cache_v4",
    "validate_cache_v4",
    "write_cache_v4",
]
