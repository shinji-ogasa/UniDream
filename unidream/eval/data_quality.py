"""Development-cache data and feature quality gates.

This module is intentionally independent of the forecast tournament.  It audits
the materialized research cache before any model result is read.  The cache
metadata is the schema authority; the checks below never infer a missing value
by sorting, dropping, or filling rows.

The development scope for this audit is the right-exclusive interval
``[2018-01-01, 2024-01-01)``.  A cache with an availability mask is not required
by the historical v3 format, but the absence of that mask is a *failed quality
gate*: a finite zero cannot be distinguished from a zero used for unavailable
external data.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from unidream.data.dataset import get_wfo_splits
from unidream.data.features import (
    align_funding_rate,
    compute_basis_features,
    compute_features,
)


DEVELOPMENT_START = pd.Timestamp("2018-01-01")
DEVELOPMENT_END = pd.Timestamp("2024-01-01")
DEFAULT_INTERVAL = "15m"
QUALITY_SCHEMA_VERSION = 1

OHLCV_FEATURES: tuple[str, ...] = (
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
)
EXTERNAL_FEATURES: tuple[str, ...] = (
    "funding_rate",
    "basis",
    "basis_mom",
    "basis_abs",
)
FULL17_FEATURES: tuple[str, ...] = OHLCV_FEATURES + EXTERNAL_FEATURES
RAW_OHLCV_INPUTS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class DataQualityError(ValueError):
    """Raised when a cache fails a fail-closed contract validation."""

    def __init__(self, issues: Iterable[str] | str):
        if isinstance(issues, str):
            values = [issues]
        else:
            values = [str(value) for value in issues]
        self.issues = tuple(values)
        super().__init__("data quality contract failed: " + "; ".join(self.issues))


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return parsed


def _normalise_index(index: Any) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index is not a DatetimeIndex")
    # Treat timezone-aware cache indexes as UTC and compare them in one stable
    # representation.  This does not repair ordering, duplicates, or gaps.
    parsed = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    return parsed.tz_localize(None)


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
        raise ValueError(f"unsupported interval for data quality audit: {interval!r}")
    return values[interval]


def _index_diagnostics(index: Any, *, name: str, interval: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "is_datetime_index": isinstance(index, pd.DatetimeIndex),
        "is_unique": None,
        "is_monotonic_increasing": None,
        "row_count": int(len(index)) if hasattr(index, "__len__") else None,
        "duplicate_count": None,
        "non_15m_step_count": None,
        "missing_bar_count": None,
        "first_gap": None,
        "status": "fail",
    }
    if not isinstance(index, pd.DatetimeIndex):
        return result

    result["is_unique"] = bool(index.is_unique)
    result["is_monotonic_increasing"] = bool(index.is_monotonic_increasing)
    result["duplicate_count"] = int(index.duplicated(keep=False).sum())
    if len(index) < 2:
        result["non_15m_step_count"] = 0
        result["missing_bar_count"] = 0
        result["status"] = "pass" if index.is_unique else "fail"
        return result

    expected = _interval_delta(interval)
    differences = index[1:] - index[:-1]
    bad = differences != expected
    result["non_15m_step_count"] = int(np.sum(bad))
    missing = 0
    first_gap: dict[str, Any] | None = None
    for position in np.flatnonzero(np.asarray(bad, dtype=bool)):
        delta = differences[position]
        if delta > expected:
            # Exact multiples are the usual missing-candle case.  Irregular
            # positive deltas still count as at least one missing boundary.
            missing += max(1, int(delta // expected) - 1)
        if first_gap is None:
            first_gap = {
                "left": str(index[position]),
                "right": str(index[position + 1]),
                "delta": str(delta),
                "expected": str(expected),
            }
    result["missing_bar_count"] = int(missing)
    result["first_gap"] = first_gap
    result["status"] = (
        "pass"
        if bool(index.is_unique)
        and bool(index.is_monotonic_increasing)
        and not bool(np.any(bad))
        else "fail"
    )
    return result


def _numeric_diagnostics(frame: pd.DataFrame, *, name: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "non_numeric_count": 0,
        "nonfinite_count": 0,
        "first_nonfinite": [],
        "status": "pass",
    }
    try:
        converted = frame.apply(lambda series: pd.to_numeric(series, errors="coerce"))
        non_numeric = frame.notna() & converted.isna()
        result["non_numeric_count"] = int(non_numeric.to_numpy(dtype=bool).sum())
        values = converted.to_numpy(dtype=np.float64)
    except (TypeError, ValueError):
        result["non_numeric_count"] = max(1, int(frame.size))
        result["status"] = "fail"
        return result

    finite = np.isfinite(values)
    result["nonfinite_count"] = int((~finite).sum())
    bad_positions = np.argwhere(~finite)
    result["first_nonfinite"] = [
        {
            "row": int(row),
            "column": str(frame.columns[column]),
            "timestamp": str(frame.index[row]) if row < len(frame.index) else None,
            "value": str(frame.iloc[row, column]) if row < len(frame.index) else None,
        }
        for row, column in bad_positions[:5]
    ]
    result["status"] = (
        "pass"
        if result["non_numeric_count"] == 0 and result["nonfinite_count"] == 0
        else "fail"
    )
    return result


def _as_returns_frame(returns: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(returns, pd.Series):
        return returns.to_frame(name=returns.name or "returns")
    return returns.copy()


def _metadata_columns(metadata: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    value = metadata.get("feature_columns")
    issues: list[str] = []
    if not isinstance(value, list) or not value or any(
        not isinstance(column, str) or not column.strip() for column in value
    ):
        return [], ["metadata.feature_columns must be a non-empty list of column names"]
    columns = [str(column) for column in value]
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        issues.append(f"metadata feature_columns contains duplicate columns: {duplicates}")
    return columns, issues


def _expected_external_columns(metadata: Mapping[str, Any]) -> list[str]:
    parameters = metadata.get("parameters")
    if not isinstance(parameters, Mapping):
        return list(EXTERNAL_FEATURES)
    expected: list[str] = []
    if bool(parameters.get("include_funding", True)):
        expected.append("funding_rate")
    if bool(parameters.get("include_mark", True)):
        expected.extend(["basis", "basis_mom", "basis_abs"])
    return expected


def _required_raw_input_columns(metadata: Mapping[str, Any]) -> list[str]:
    """Resolve raw source columns from the cache metadata feature contract."""
    parameters = metadata.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    required = list(RAW_OHLCV_INPUTS)
    if bool(parameters.get("include_funding", True)):
        required.append("funding_rate")
    if bool(parameters.get("include_oi", False)):
        required.append("open_interest")
    if bool(parameters.get("include_mark", True)):
        required.append("mark_close")
    return required


def inspect_feature_contract(
    features: pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    metadata: Mapping[str, Any],
    *,
    start: Any = DEVELOPMENT_START,
    end: Any = DEVELOPMENT_END,
    interval: str = DEFAULT_INTERVAL,
) -> dict[str, Any]:
    """Inspect a cache without mutating or silently repairing its rows.

    The returned object is suitable for a JSONL ledger.  Use
    :func:`validate_feature_contract` when a caller needs a fail-closed
    exception instead of a diagnostic object.
    """
    issues: list[str] = []
    if not isinstance(metadata, Mapping):
        metadata = {}
        issues.append("metadata must be a JSON object")
    metadata_columns, metadata_issues = _metadata_columns(metadata)
    issues.extend(metadata_issues)

    try:
        feature_index = _normalise_index(features.index)
        feature_frame = features.copy()
        feature_frame.index = feature_index
    except (TypeError, ValueError, AttributeError) as exc:
        feature_index = features.index
        feature_frame = features.copy()
        issues.append(f"features index is not a valid DatetimeIndex: {exc}")

    returns_frame = _as_returns_frame(returns)
    try:
        returns_index = _normalise_index(returns_frame.index)
        returns_frame.index = returns_index
    except (TypeError, ValueError, AttributeError) as exc:
        returns_index = returns_frame.index
        issues.append(f"returns index is not a valid DatetimeIndex: {exc}")

    feature_columns = [str(column) for column in feature_frame.columns]
    duplicate_data_columns = sorted(
        {column for column in feature_columns if feature_columns.count(column) > 1}
    )
    if duplicate_data_columns:
        issues.append(f"features contains duplicate columns: {duplicate_data_columns}")
    if metadata_columns and feature_columns != metadata_columns:
        missing = [column for column in metadata_columns if column not in feature_columns]
        unexpected = [column for column in feature_columns if column not in metadata_columns]
        if not missing and not unexpected:
            issues.append(
                "feature column order mismatch: "
                f"expected {metadata_columns}, got {feature_columns}"
            )
        else:
            issues.append(
                "feature columns do not match metadata: "
                f"missing {missing or []}; unexpected {unexpected or []}"
            )

    parameters = metadata.get("parameters") if isinstance(metadata, Mapping) else {}
    if not isinstance(parameters, Mapping):
        parameters = {}
        issues.append("metadata.parameters must be a JSON object")
    required_external = _expected_external_columns(metadata)
    missing_external = [column for column in required_external if column not in feature_columns]
    if missing_external:
        issues.append(f"missing required external feature columns: {missing_external}")
    missing_ohlcv = [column for column in OHLCV_FEATURES if column not in feature_columns]
    if missing_ohlcv:
        issues.append(f"missing required OHLCV feature columns: {missing_ohlcv}")
    expected_parameter_values = {
        "interval": interval,
        "start": _timestamp(start),
        "end": _timestamp(end),
    }
    for key, expected in expected_parameter_values.items():
        if key not in parameters:
            continue
        actual = parameters[key]
        try:
            comparable = _timestamp(actual) if key in {"start", "end"} else str(actual)
        except (TypeError, ValueError):
            comparable = actual
        if comparable != expected:
            issues.append(
                f"metadata parameters.{key} mismatch: metadata={actual!r}, expected={expected}"
            )

    feature_index_diag = _index_diagnostics(
        feature_index,
        name="features",
        interval=interval,
    )
    returns_index_diag = _index_diagnostics(
        returns_index,
        name="returns",
        interval=interval,
    )
    for name, diagnostic in (
        ("features", feature_index_diag),
        ("returns", returns_index_diag),
    ):
        if diagnostic["status"] != "pass":
            if not diagnostic["is_datetime_index"]:
                issues.append(f"{name} index must be a DatetimeIndex")
            if diagnostic["duplicate_count"]:
                issues.append(f"{name} index contains duplicate timestamps")
            if diagnostic["is_monotonic_increasing"] is False:
                issues.append(f"{name} index is not monotonic increasing")
            if diagnostic["non_15m_step_count"]:
                issues.append(
                    f"{name} index has {diagnostic['non_15m_step_count']} non-{interval} gaps"
                )

    alignment: dict[str, Any] = {
        "same_index": False,
        "feature_rows": int(len(feature_frame)),
        "returns_rows": int(len(returns_frame)),
        "status": "fail",
    }
    if isinstance(feature_index, pd.DatetimeIndex) and isinstance(returns_index, pd.DatetimeIndex):
        alignment["same_index"] = bool(feature_index.equals(returns_index))
        if not alignment["same_index"]:
            issues.append("features/returns indices are not exactly aligned")
        alignment["status"] = "pass" if alignment["same_index"] else "fail"
    else:
        issues.append("features/returns alignment cannot be checked without datetime indexes")

    feature_numeric = _numeric_diagnostics(feature_frame, name="features")
    returns_numeric = _numeric_diagnostics(returns_frame, name="returns")
    if feature_numeric["status"] != "pass":
        issues.append(
            "features contains non-numeric or non-finite values: "
            f"non_numeric={feature_numeric['non_numeric_count']}, "
            f"nonfinite={feature_numeric['nonfinite_count']}"
        )
    if returns_numeric["status"] != "pass":
        issues.append(
            "returns contains non-numeric or non-finite values: "
            f"non_numeric={returns_numeric['non_numeric_count']}, "
            f"nonfinite={returns_numeric['nonfinite_count']}"
        )

    range_check: dict[str, Any] = {
        "requested_start": str(_timestamp(start)),
        "requested_end_exclusive": str(_timestamp(end)),
        "features_within_scope": None,
        "returns_within_scope": None,
        "status": "fail",
    }
    try:
        start_ts = _timestamp(start)
        end_ts = _timestamp(end)
        if start_ts >= end_ts:
            issues.append("quality-gate scope start must be earlier than end")
        else:
            feature_in_scope = (
                isinstance(feature_index, pd.DatetimeIndex)
                and len(feature_index) > 0
                and bool(feature_index.min() >= start_ts)
                and bool(feature_index.max() < end_ts)
            )
            returns_in_scope = (
                isinstance(returns_index, pd.DatetimeIndex)
                and len(returns_index) > 0
                and bool(returns_index.min() >= start_ts)
                and bool(returns_index.max() < end_ts)
            )
            range_check["features_within_scope"] = feature_in_scope
            range_check["returns_within_scope"] = returns_in_scope
            if not feature_in_scope:
                issues.append(
                    f"features timestamps must be within [{start_ts}, {end_ts}); "
                    f"observed {feature_index.min() if len(feature_index) else None} .. "
                    f"{feature_index.max() if len(feature_index) else None}"
                )
            if not returns_in_scope:
                issues.append(
                    f"returns timestamps must be within [{start_ts}, {end_ts}); "
                    f"observed {returns_index.min() if len(returns_index) else None} .. "
                    f"{returns_index.max() if len(returns_index) else None}"
                )
            range_check["status"] = "pass" if feature_in_scope and returns_in_scope else "fail"
    except (TypeError, ValueError) as exc:
        issues.append(f"invalid quality-gate scope: {exc}")

    metadata_rows = metadata.get("rows")
    if metadata_rows is not None and metadata_rows != len(feature_frame):
        issues.append(
            f"metadata rows mismatch: metadata={metadata_rows}, actual={len(feature_frame)}"
        )
    for key, index in (("first_timestamp", feature_index), ("last_timestamp", feature_index)):
        expected = metadata.get(key)
        if expected is None or not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
            continue
        try:
            actual_ts = index[0] if key == "first_timestamp" else index[-1]
            if _timestamp(expected) != actual_ts:
                issues.append(f"metadata {key} mismatch: metadata={expected}, actual={actual_ts}")
        except (TypeError, ValueError):
            issues.append(f"metadata {key} is not a valid timestamp: {expected!r}")

    schema_payload = {
        "schema_version": metadata.get("schema_version"),
        "cache_tag": metadata.get("cache_tag"),
        "parameters": parameters,
        "feature_columns": metadata_columns,
    }
    metadata_digest = _canonical_sha256(dict(metadata))
    schema_digest = _canonical_sha256(schema_payload)
    schema = {
        "metadata_authoritative": True,
        "metadata_feature_columns": metadata_columns,
        "actual_feature_columns": feature_columns,
        "required_ohlcv13": list(OHLCV_FEATURES),
        "required_external4": list(EXTERNAL_FEATURES),
        "required_raw_inputs": _required_raw_input_columns(metadata),
        "metadata_digest": metadata_digest,
        "schema_digest": schema_digest,
        "metadata_schema_version": metadata.get("schema_version"),
        "cache_tag": metadata.get("cache_tag"),
        "status": "pass" if not any("metadata" in issue for issue in issues) else "fail",
    }
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "schema": schema,
        "features": {
            "index": feature_index_diag,
            "numeric": feature_numeric,
        },
        "returns": {
            "index": returns_index_diag,
            "numeric": returns_numeric,
        },
        "alignment": alignment,
        "scope": range_check,
    }


def validate_feature_contract(
    features: pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    metadata: Mapping[str, Any],
    *,
    start: Any = DEVELOPMENT_START,
    end: Any = DEVELOPMENT_END,
    interval: str = DEFAULT_INTERVAL,
) -> dict[str, Any]:
    """Validate a cache and raise with all detected contract violations."""
    result = inspect_feature_contract(
        features,
        returns,
        metadata,
        start=start,
        end=end,
        interval=interval,
    )
    if result["status"] != "pass":
        raise DataQualityError(result["issues"])
    return result


def _availability_mask_columns(columns: Iterable[str], external: Iterable[str]) -> dict[str, str]:
    names = {str(column) for column in columns}
    result: dict[str, str] = {}
    suffixes = ("_available", "_availability", "_observed", "_present", "_mask")
    for name in external:
        for suffix in suffixes:
            candidate = f"{name}{suffix}"
            if candidate in names:
                result[str(name)] = candidate
                break
    return result


def external_coverage(
    frame: pd.DataFrame,
    *,
    external: Iterable[str] = EXTERNAL_FEATURES,
    zero_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Record finite/zero/nonzero/missing coverage without hiding ambiguity."""
    external_names = [str(name) for name in external]
    rows = int(len(frame))
    masks = _availability_mask_columns(frame.columns, external_names)
    columns: dict[str, dict[str, Any]] = {}
    missing_columns: list[str] = []
    for name in external_names:
        if name not in frame.columns:
            missing_columns.append(name)
            columns[name] = {
                "rows": rows,
                "finite_count": 0,
                "missing_count": rows,
                "zero_count": 0,
                "nonzero_count": 0,
                "finite_rate": 0.0 if rows else None,
                "missing_rate": 1.0 if rows else None,
                "zero_rate": 0.0 if rows else None,
                "nonzero_rate": 0.0 if rows else None,
                "availability_mask_column": None,
                "status": "fail_missing_column",
            }
            continue
        numeric = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(numeric)
        zero = finite & (np.abs(numeric) <= zero_tolerance)
        nonzero = finite & ~zero
        denominator = max(rows, 1)
        columns[name] = {
            "rows": rows,
            "finite_count": int(finite.sum()),
            "missing_count": int((~finite).sum()),
            "zero_count": int(zero.sum()),
            "nonzero_count": int(nonzero.sum()),
            "finite_rate": float(finite.sum() / denominator) if rows else None,
            "missing_rate": float((~finite).sum() / denominator) if rows else None,
            "zero_rate": float(zero.sum() / denominator) if rows else None,
            "nonzero_rate": float(nonzero.sum() / denominator) if rows else None,
            "availability_mask_column": masks.get(name),
            "status": "pass" if finite.all() else "fail_nonfinite",
        }

    mask_present = bool(masks)
    gate = {
        "status": "pass" if mask_present else "fail",
        "availability_mask_present": mask_present,
        "availability_mask_columns": masks,
        "reason": (
            "availability mask is present"
            if mask_present
            else "cache has no availability mask; zero and missing/imputed values are indistinguishable"
        ),
    }
    status = "pass"
    if missing_columns or not mask_present or any(
        value["status"] != "pass" for value in columns.values()
    ):
        status = "fail"
    return {
        "rows": rows,
        "external": columns,
        "missing_columns": missing_columns,
        "availability_gate": gate,
        "status": status,
        "contract_note": (
            "No availability mask means finite zero values cannot be proven to be observed "
            "external data rather than an unavailable-value fill."
        ),
    }


def feature_coverage(
    frame: pd.DataFrame,
    *,
    external: Iterable[str] = EXTERNAL_FEATURES,
) -> dict[str, Any]:
    """Public alias used by tests and report builders."""
    return external_coverage(frame, external=external)


def _slice(frame: pd.DataFrame, start: Any, end: Any) -> pd.DataFrame:
    start_ts = _timestamp(start)
    end_ts = _timestamp(end)
    return frame.loc[(frame.index >= start_ts) & (frame.index < end_ts)]


def annual_external_coverage(
    features: pd.DataFrame,
    *,
    start: Any = DEVELOPMENT_START,
    end: Any = DEVELOPMENT_END,
    external: Iterable[str] = EXTERNAL_FEATURES,
) -> dict[str, Any]:
    """Return calendar-year external coverage inside the development scope."""
    start_ts = _timestamp(start)
    end_ts = _timestamp(end)
    years: dict[str, Any] = {}
    for year in range(start_ts.year, end_ts.year):
        year_start = max(start_ts, pd.Timestamp(year=year, month=1, day=1))
        year_end = min(end_ts, pd.Timestamp(year=year + 1, month=1, day=1))
        selected = _slice(features, year_start, year_end)
        years[str(year)] = {
            "start": str(year_start),
            "end_exclusive": str(year_end),
            "coverage": external_coverage(selected, external=external),
        }
    return years


def wfo_external_coverage(
    features: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
    *,
    start: Any = DEVELOPMENT_START,
    end: Any = DEVELOPMENT_END,
    external: Iterable[str] = EXTERNAL_FEATURES,
) -> list[dict[str, Any]]:
    """Record external coverage for each configured WFO train/val/test slice."""
    data_cfg = config.get("data", {}) if isinstance(config, Mapping) else {}
    run_cfg = config.get("run", {}) if isinstance(config, Mapping) else {}
    train_years = int(data_cfg.get("train_years", 2))
    val_months = int(data_cfg.get("val_months", 3))
    test_months = int(data_cfg.get("test_months", 3))
    splits = get_wfo_splits(
        features,
        train_years=train_years,
        val_months=val_months,
        test_months=test_months,
    )
    configured = run_cfg.get("folds") if isinstance(run_cfg, Mapping) else None
    if isinstance(configured, list):
        selected_ids = {int(value) for value in configured}
        splits = [split for split in splits if split.fold_idx in selected_ids]
    elif configured not in (None, "all"):
        selected_ids = {int(value) for value in configured}
        splits = [split for split in splits if split.fold_idx in selected_ids]

    start_ts = _timestamp(start)
    end_ts = _timestamp(end)
    result: list[dict[str, Any]] = []
    for split in splits:
        phases = (
            ("train", split.train_start, split.train_end),
            ("val", split.val_start, split.val_end),
            ("test", split.test_start, split.test_end),
        )
        for phase, phase_start, phase_end in phases:
            phase_start_ts = max(start_ts, _timestamp(phase_start))
            phase_end_ts = min(end_ts, _timestamp(phase_end))
            if phase_start_ts >= phase_end_ts:
                selected = features.iloc[0:0]
            else:
                selected = _slice(features, phase_start_ts, phase_end_ts)
            result.append(
                {
                    "fold": int(split.fold_idx),
                    "phase": phase,
                    "start": str(phase_start_ts),
                    "end_exclusive": str(phase_end_ts),
                    "coverage": external_coverage(selected, external=external),
                }
            )
    return result


def same_row_fairness(
    features: pd.DataFrame,
    *,
    ohlcv_columns: Iterable[str] = OHLCV_FEATURES,
    full_columns: Iterable[str] = FULL17_FEATURES,
    interval: str = DEFAULT_INTERVAL,
) -> dict[str, Any]:
    """Compare OHLCV13/full17 eligibility using one pre-declared row rule."""
    ohlcv = [str(column) for column in ohlcv_columns]
    full = [str(column) for column in full_columns]
    missing_ohlcv = [column for column in ohlcv if column not in features.columns]
    missing_full = [column for column in full if column not in features.columns]
    result: dict[str, Any] = {
        "rule": "finite intersection of all full17 columns within the development scope",
        "ohlcv13_columns": ohlcv,
        "full17_columns": full,
        "missing_ohlcv_columns": missing_ohlcv,
        "missing_full17_columns": missing_full,
        "ohlcv13_eligible_rows": 0,
        "full17_eligible_rows": 0,
        "same_row_eligibility": False,
        "full17_eligible_start": None,
        "full17_eligible_end_exclusive": None,
        "status": "fail",
    }
    if missing_ohlcv or missing_full:
        return result

    try:
        ohlcv_values = features[ohlcv].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        full_values = features[full].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    except (TypeError, ValueError):
        return result
    ohlcv_mask = np.isfinite(ohlcv_values).all(axis=1)
    full_mask = np.isfinite(full_values).all(axis=1)
    result["ohlcv13_eligible_rows"] = int(ohlcv_mask.sum())
    result["full17_eligible_rows"] = int(full_mask.sum())
    result["same_row_eligibility"] = bool(
        np.array_equal(ohlcv_mask, full_mask)
        and np.array_equal(features.index[ohlcv_mask], features.index[full_mask])
    )
    eligible_index = features.index[full_mask]
    if len(eligible_index):
        result["full17_eligible_start"] = str(eligible_index[0])
        result["full17_eligible_end_exclusive"] = str(
            eligible_index[-1] + _interval_delta(interval)
        )
    result["status"] = "pass" if result["same_row_eligibility"] else "fail"
    return result


def _synthetic_ohlcv(rows: int = 720) -> pd.DataFrame:
    rng = np.random.default_rng(20260830)
    index = pd.date_range("2021-01-01", periods=rows, freq="15min")
    close = 20_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0015, rows)))
    spread = np.abs(rng.normal(0.0, 0.001, rows))
    return pd.DataFrame(
        {
            "open": close * (1.0 + rng.normal(0.0, 0.0003, rows)),
            "high": close * (1.0 + spread),
            "low": close * (1.0 - spread),
            "close": close,
            "volume": np.exp(rng.normal(3.0, 0.3, rows)),
        },
        index=index,
    )


def _max_prefix_difference(left: pd.DataFrame, right: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[int, float]:
    common = left.index.intersection(right.index)
    common = common[common <= cutoff]
    if len(common) == 0:
        raise ValueError("causality probe has no common prefix rows")
    left_values = left.loc[common].to_numpy(dtype=np.float64)
    right_values = right.loc[common].to_numpy(dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError("causality probe feature shapes differ on common prefix")
    difference = np.abs(left_values - right_values)
    return int(len(common)), float(np.nanmax(difference))


def run_causality_probes() -> dict[str, Any]:
    """Run deterministic future-perturbation and offset probes.

    Every comparison is made only on a prefix ending at the perturbation bar.
    This exercises the real feature functions, including funding and mark
    alignment, without training or reading a model result.
    """
    try:
        rows = 720
        cutoff_position = 480
        spot = _synthetic_ohlcv(rows)
        cutoff = spot.index[cutoff_position]
        funding_index = spot.index[320::32]
        funding = pd.DataFrame(
            {"funding_rate": np.linspace(-0.0003, 0.0005, len(funding_index))},
            index=funding_index,
        )
        mark_index = spot.index[300::32]
        mark = pd.DataFrame({"mark_close": spot.loc[mark_index, "close"] * 1.001}, index=mark_index)

        perturbed_spot = spot.copy()
        perturbed_spot.loc[perturbed_spot.index >= cutoff, ["open", "high", "low", "close"]] *= 1.07
        perturbed_spot.loc[perturbed_spot.index >= cutoff, "volume"] *= 3.0
        perturbed_funding = funding.copy()
        perturbed_funding.loc[perturbed_funding.index >= cutoff, "funding_rate"] += 0.01
        perturbed_mark = mark.copy()
        perturbed_mark.loc[perturbed_mark.index >= cutoff, "mark_close"] *= 1.2

        base = compute_features(
            spot,
            zscore_window_days=1,
            interval=DEFAULT_INTERVAL,
            funding_df=funding,
            mark_price_df=mark,
        )
        perturbed = compute_features(
            perturbed_spot,
            zscore_window_days=1,
            interval=DEFAULT_INTERVAL,
            funding_df=perturbed_funding,
            mark_price_df=perturbed_mark,
        )
        prefix_rows, prefix_max_diff = _max_prefix_difference(base, perturbed, cutoff)

        prefix_spot = spot.iloc[: cutoff_position + 1]
        prefix_funding = funding.loc[funding.index <= cutoff]
        prefix_mark = mark.loc[mark.index <= cutoff]
        prefix_features = compute_features(
            prefix_spot,
            zscore_window_days=1,
            interval=DEFAULT_INTERVAL,
            funding_df=prefix_funding,
            mark_price_df=prefix_mark,
        )
        invariant_rows, invariant_max_diff = _max_prefix_difference(base, prefix_features, cutoff)

        mark_start = mark.index[0]
        mark_head = compute_basis_features(spot["close"], mark).loc[lambda frame: frame.index < mark_start]
        offset_mark = mark.copy()
        offset_mark.iloc[0:3, 0] *= 10.0
        offset_head = compute_basis_features(spot["close"], offset_mark).loc[mark_head.index]
        offset_max_diff = float(np.max(np.abs(mark_head.to_numpy() - offset_head.to_numpy())))

        funding_start = funding.index[0]
        aligned_funding = align_funding_rate(funding, spot.index)
        shifted_funding = funding.copy()
        shifted_funding.iloc[0, 0] *= 10.0
        aligned_shifted = align_funding_rate(shifted_funding, spot.index)
        funding_head = aligned_funding.loc[aligned_funding.index < funding_start]
        funding_before = funding_head.to_numpy(dtype=np.float64)
        funding_after = aligned_shifted.loc[funding_head.index].to_numpy(dtype=np.float64)
        funding_finite_before = np.isfinite(funding_before)
        funding_finite_after = np.isfinite(funding_after)
        if not np.array_equal(funding_finite_before, funding_finite_after):
            funding_offset_max_diff = float("inf")
        elif np.any(funding_finite_before):
            funding_offset_max_diff = float(
                np.max(np.abs(funding_before[funding_finite_before] - funding_after[funding_finite_before]))
            )
        else:
            funding_offset_max_diff = 0.0

        checks = {
            "future_perturbation_prefix": {
                "status": "pass" if prefix_max_diff <= 1e-12 else "fail",
                "rows_checked": prefix_rows,
                "cutoff": str(cutoff),
                "max_abs_diff": prefix_max_diff,
                "claim": "feature row t is invariant to spot/funding/mark values at t and later",
            },
            "prefix_invariance": {
                "status": "pass" if invariant_max_diff <= 1e-12 else "fail",
                "rows_checked": invariant_rows,
                "cutoff": str(cutoff),
                "max_abs_diff": invariant_max_diff,
                "claim": "adding later rows does not change an already-materialized feature prefix",
            },
            "mark_offset_no_future_bfill": {
                "status": "pass" if offset_max_diff <= 1e-12 else "fail",
                "rows_checked": int(len(mark_head)),
                "mark_start": str(mark_start),
                "max_abs_diff_before_mark_start": offset_max_diff,
                "claim": "mark values after the source offset cannot populate earlier rows",
            },
            "funding_offset_asof": {
                "status": "pass" if funding_offset_max_diff <= 1e-12 else "fail",
                "rows_checked": int(len(funding_head)),
                "funding_start": str(funding_start),
                "max_abs_diff_before_funding_start": funding_offset_max_diff,
                "claim": "funding alignment uses the latest known value and never future-bfills",
            },
        }
        return {
            "status": "pass" if all(item["status"] == "pass" for item in checks.values()) else "fail",
            "checks": checks,
            "synthetic_rows": rows,
            "zscore_window_days": 1,
        }
    except Exception as exc:  # pragma: no cover - diagnostic path is reported by the CLI
        return {
            "status": "fail",
            "checks": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def audit_cache(
    features_path: str | Path,
    returns_path: str | Path,
    metadata_path: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    start: Any = DEVELOPMENT_START,
    end: Any = DEVELOPMENT_END,
    interval: str = DEFAULT_INTERVAL,
) -> dict[str, Any]:
    """Load and audit one development cache, returning a JSON-safe report."""
    features_path = Path(features_path)
    returns_path = Path(returns_path)
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = pd.read_parquet(features_path)
    returns = pd.read_parquet(returns_path)
    contract = inspect_feature_contract(
        features,
        returns,
        metadata,
        start=start,
        end=end,
        interval=interval,
    )

    result: dict[str, Any] = {
        "quality_schema_version": QUALITY_SCHEMA_VERSION,
        "scope": {
            "start": str(_timestamp(start)),
            "end_exclusive": str(_timestamp(end)),
            "interval": interval,
            "model_results_read": False,
        },
        "provenance": {
            "git_commit": _git_commit(),
            "features_path": str(features_path),
            "returns_path": str(returns_path),
            "metadata_path": str(metadata_path),
            "features_sha256": _sha256_file(features_path),
            "returns_sha256": _sha256_file(returns_path),
            "metadata_file_sha256": _sha256_file(metadata_path),
            "metadata_digest": contract["schema"]["metadata_digest"],
            "schema_digest": contract["schema"]["schema_digest"],
            "config_digest": _canonical_sha256(config) if config is not None else None,
        },
        "contract": contract,
        "causality": run_causality_probes(),
        "annual_coverage": {},
        "wfo_coverage": [],
        "fairness": {},
        "gates": {},
    }

    coverage_index_usable = (
        isinstance(features.index, pd.DatetimeIndex)
        and bool(features.index.is_unique)
        and bool(features.index.is_monotonic_increasing)
    )
    if coverage_index_usable:
        normalized_features = features.copy()
        normalized_features.index = _normalise_index(normalized_features.index)
        result["annual_coverage"] = annual_external_coverage(
            normalized_features,
            start=start,
            end=end,
        )
        result["wfo_coverage"] = wfo_external_coverage(
            normalized_features,
            config,
            start=start,
            end=end,
        )
        result["fairness"] = same_row_fairness(normalized_features, interval=interval)
    else:
        result["annual_coverage"] = {
            "status": "not_run_invalid_index",
            "reason": "timestamp index is not sorted/unique; no rows were silently repaired",
        }
        result["wfo_coverage"] = {
            "status": "not_run_invalid_index",
            "reason": "timestamp index is not sorted/unique; no rows were silently repaired",
        }
        result["fairness"] = {
            "status": "not_run_invalid_index",
            "rule": "finite intersection of all full17 columns within the development scope",
        }

    contract_gate = contract["status"] == "pass"
    causality_gate = result["causality"]["status"] == "pass"
    fairness_gate = result["fairness"].get("status") == "pass"
    # Coverage is diagnostic and can still be measured when the structural
    # gate fails (for example, to show exactly which years contain gaps).  It
    # never repairs or drops those rows.
    all_coverage = external_coverage(features)
    availability_gate = all_coverage["availability_gate"]["status"] == "pass"
    result["coverage_gate_all_rows"] = all_coverage
    result["gates"] = {
        "cache_contract": "pass" if contract_gate else "fail",
        "causality": "pass" if causality_gate else "fail",
        "same_row_ohlcv13_vs_full17": "pass" if fairness_gate else "fail",
        "external_availability_mask": "pass" if availability_gate else "fail",
        "overall": "pass"
        if contract_gate and causality_gate and fairness_gate and availability_gate
        else "fail",
        "blocking_reason": (
            None
            if availability_gate
            else "v3 cache has no availability mask; external zero and missing values cannot be distinguished"
        ),
    }
    return result


def ledger_records(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten an audit report into deterministic JSONL records."""
    records: list[dict[str, Any]] = [
        {
            "record_type": "run",
            "quality_schema_version": report.get("quality_schema_version"),
            "scope": report.get("scope"),
            "provenance": report.get("provenance"),
            "gates": report.get("gates"),
        },
        {"record_type": "contract", "payload": report.get("contract")},
        {"record_type": "causality", "payload": report.get("causality")},
        {"record_type": "coverage_all_rows", "payload": report.get("coverage_gate_all_rows")},
        {"record_type": "fairness", "payload": report.get("fairness")},
    ]
    annual = report.get("annual_coverage", {})
    if isinstance(annual, Mapping):
        for year in sorted(annual):
            records.append(
                {
                    "record_type": "annual_external_coverage",
                    "period": year,
                    "payload": annual[year],
                }
            )
    wfo = report.get("wfo_coverage", [])
    if isinstance(wfo, list):
        for item in wfo:
            records.append(
                {
                    "record_type": "wfo_external_coverage",
                    "fold": item.get("fold"),
                    "phase": item.get("phase"),
                    "payload": item,
                }
            )
    return records


def write_jsonl(report: Mapping[str, Any], path: str | Path) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = ledger_records(report)
    text = "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ) + "\n"
    destination.write_text(text, encoding="utf-8")
    return len(records)


def render_markdown_report(report: Mapping[str, Any], *, ledger_path: str | Path | None = None) -> str:
    """Render a concise report with no model-performance results."""
    gates = report.get("gates", {})
    contract = report.get("contract", {})
    schema = contract.get("schema", {}) if isinstance(contract, Mapping) else {}
    causality = report.get("causality", {})
    fairness = report.get("fairness", {})
    lines = [
        "# Development data / feature quality gate",
        "",
        "This report audits only the materialized development cache. No model, forecast, or future-return result was read.",
        "",
        f"- Scope: `[{report.get('scope', {}).get('start')}, {report.get('scope', {}).get('end_exclusive')})`",
        f"- Interval: `{report.get('scope', {}).get('interval')}`",
        f"- Overall gate: **{str(gates.get('overall', 'unknown')).upper()}**",
        f"- JSONL ledger: `{ledger_path}`" if ledger_path is not None else "",
        f"- Metadata digest: `{report.get('provenance', {}).get('metadata_digest')}`",
        f"- Schema digest: `{report.get('provenance', {}).get('schema_digest')}`",
        "",
        "## Gate matrix",
        "",
        "| Gate | Status |",
        "| --- | --- |",
        f"| cache contract | {gates.get('cache_contract')} |",
        f"| causal feature probes | {gates.get('causality')} |",
        f"| OHLCV13/full17 same-row eligibility | {gates.get('same_row_ohlcv13_vs_full17')} |",
        f"| external availability mask | {gates.get('external_availability_mask')} |",
        "",
        "## Schema and alignment",
        "",
        f"- Metadata is authoritative: `{schema.get('metadata_authoritative')}`",
        f"- Feature columns: `{schema.get('actual_feature_columns')}`",
        f"- Features/returns exact index alignment: `{contract.get('alignment', {}).get('same_index')}`",
        f"- Feature timestamp diagnostics: `{contract.get('features', {}).get('index', {}).get('status')}`",
        f"- Returns timestamp diagnostics: `{contract.get('returns', {}).get('index', {}).get('status')}`",
        f"- Non-finite values: features `{contract.get('features', {}).get('numeric', {}).get('nonfinite_count')}`, returns `{contract.get('returns', {}).get('numeric', {}).get('nonfinite_count')}`",
        "",
        "## Causality probes",
        "",
        f"- Overall: `{causality.get('status')}`",
    ]
    for name, check in (causality.get("checks", {}) or {}).items():
        lines.append(
            f"- `{name}`: {check.get('status')}, max difference `{check.get('max_abs_diff', check.get('max_abs_diff_before_mark_start', check.get('max_abs_diff_before_funding_start')))}`"
        )
    lines.extend(
        [
            "",
            "## External coverage",
            "",
            "Coverage records finite/nonfinite, zero, and nonzero values independently. The v3 cache has no availability mask, so the zero-vs-missing gate is intentionally failed even when the observed rows are finite.",
            "",
            "| Year | Rows | funding zero/nonzero/missing | basis zero/nonzero/missing | basis_mom zero/nonzero/missing | basis_abs zero/nonzero/missing |",
            "| --- | ---: | --- | --- | --- | --- |",
        ]
    )
    annual = report.get("annual_coverage", {})
    if isinstance(annual, Mapping) and "status" not in annual:
        for year, payload in annual.items():
            coverage = payload.get("coverage", {}) if isinstance(payload, Mapping) else {}
            external = coverage.get("external", {}) if isinstance(coverage, Mapping) else {}
            def counts(name: str) -> str:
                item = external.get(name, {})
                return f"{item.get('zero_count', '-')}/{item.get('nonzero_count', '-')}/{item.get('missing_count', '-')}"
            lines.append(
                f"| {year} | {coverage.get('rows', '-')} | {counts('funding_rate')} | {counts('basis')} | {counts('basis_mom')} | {counts('basis_abs')} |"
            )
    lines.extend(
        [
            "",
            "## OHLCV13 vs full17 fairness",
            "",
            f"- Rule: `{fairness.get('rule')}`",
            f"- OHLCV13 eligible rows: `{fairness.get('ohlcv13_eligible_rows')}`",
            f"- Full17 eligible rows: `{fairness.get('full17_eligible_rows')}`",
            f"- Same row mask: `{fairness.get('same_row_eligibility')}`",
            f"- Full17 eligible period: `[{fairness.get('full17_eligible_start')}, {fairness.get('full17_eligible_end_exclusive')})`",
            "",
            "## WFO coverage",
            "",
            "WFO rows are reported by configured fold and right-exclusive train/val/test phase; no performance metric is computed.",
            "",
        ]
    )
    wfo = report.get("wfo_coverage", [])
    if isinstance(wfo, list):
        lines.extend(
            [
                "| Fold | Phase | Rows | availability-mask gate |",
                "| ---: | --- | ---: | --- |",
            ]
        )
        for item in wfo:
            coverage = item.get("coverage", {})
            gate = coverage.get("availability_gate", {})
            lines.append(
                f"| {item.get('fold')} | {item.get('phase')} | {coverage.get('rows')} | {gate.get('status')} |"
            )
    lines.extend(
        [
            "",
            "## Blocking limitation",
            "",
            f"{gates.get('blocking_reason') or 'No blocking limitation.'}",
            "",
            "The pre-declared fair comparison rule is to use the intersection of rows finite for all full17 features for both ablations. If external availability is required for a future cache, add explicit per-column availability masks before treating zero as observed data.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "DataQualityError",
    "DEVELOPMENT_START",
    "DEVELOPMENT_END",
    "OHLCV_FEATURES",
    "EXTERNAL_FEATURES",
    "FULL17_FEATURES",
    "RAW_OHLCV_INPUTS",
    "inspect_feature_contract",
    "validate_feature_contract",
    "external_coverage",
    "feature_coverage",
    "annual_external_coverage",
    "wfo_external_coverage",
    "same_row_fairness",
    "run_causality_probes",
    "audit_cache",
    "ledger_records",
    "write_jsonl",
    "render_markdown_report",
]
