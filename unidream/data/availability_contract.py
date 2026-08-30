"""Fail-closed availability contracts for v4 training data.

The model input remains the canonical feature body.  Availability is a
separate, timestamped sidecar and is consumed only to decide whether a row or
sequence window is eligible.  This module intentionally never repairs,
sorts, interpolates, or compacts a sidecar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


AVAILABILITY_COLUMNS: tuple[str, ...] = (
    "spot_bar_observed",
    "funding_rate_available",
    "mark_close_available",
)


class AvailabilityContractError(ValueError):
    """Raised when availability cannot be trusted for eligibility decisions."""


@dataclass(frozen=True)
class AvailabilitySelection:
    """Validated sidecar and the row mask required by one feature body."""

    sidecar: pd.DataFrame | Mapping[str, np.ndarray]
    required_columns: tuple[str, ...]
    row_eligible: np.ndarray


def _validate_datetime_index(index: Any, *, name: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise AvailabilityContractError(f"{name} index must be a DatetimeIndex")
    if not index.is_unique:
        duplicates = index[index.duplicated(keep=False)].unique()
        raise AvailabilityContractError(
            f"{name} index contains duplicate timestamps: "
            + ", ".join(str(value) for value in duplicates[:5])
        )
    if not index.is_monotonic_increasing:
        raise AvailabilityContractError(
            f"{name} index is not strictly increasing; refusing to sort"
        )
    return index


def required_availability_columns(
    *,
    include_funding: bool = True,
    include_mark: bool = True,
) -> tuple[str, ...]:
    """Return sidecar columns required by the configured feature contract."""
    required = ["spot_bar_observed"]
    if bool(include_funding):
        required.append("funding_rate_available")
    if bool(include_mark):
        required.append("mark_close_available")
    return tuple(required)


def _validate_boolean_array(values: Any, *, column: str, expected_len: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != expected_len:
        raise AvailabilityContractError(
            f"availability column {column!r} must be one-dimensional and aligned "
            f"to the feature index (expected {expected_len} rows)"
        )
    if array.dtype != np.bool_:
        raise AvailabilityContractError(
            f"availability column {column!r} must have boolean dtype"
        )
    return array.astype(bool, copy=False)


def validate_availability(
    availability: pd.DataFrame | Mapping[str, Any],
    feature_index: pd.DatetimeIndex,
    *,
    include_funding: bool = True,
    include_mark: bool = True,
) -> AvailabilitySelection:
    """Validate and align a sidecar without changing its row set.

    A v4 sidecar normally covers the complete expected grid while the feature
    body may contain only observed contiguous Spot segments.  Consequently a
    sidecar is allowed to be a strict superset of ``feature_index``; every
    feature timestamp must still be present.  The original sidecar is
    returned unchanged and its rows are never compacted.
    """
    body_index = _validate_datetime_index(feature_index, name="feature body")
    required = required_availability_columns(
        include_funding=include_funding,
        include_mark=include_mark,
    )

    if isinstance(availability, pd.DataFrame):
        sidecar_index = _validate_datetime_index(
            availability.index,
            name="availability sidecar",
        )
        if not body_index.isin(sidecar_index).all():
            missing = body_index[~body_index.isin(sidecar_index)]
            raise AvailabilityContractError(
                "feature timestamps missing from availability sidecar: "
                + ", ".join(str(value) for value in missing[:5])
            )
        if not availability.columns.is_unique:
            duplicates = availability.columns[availability.columns.duplicated(keep=False)].unique()
            raise AvailabilityContractError(
                "availability sidecar contains duplicate columns: "
                + ", ".join(str(value) for value in duplicates[:5])
            )
        missing_columns = [column for column in required if column not in availability.columns]
        if missing_columns:
            raise AvailabilityContractError(
                "availability sidecar missing required columns: "
                + ", ".join(missing_columns)
            )
        # Validate every known mask that is present.  An optional source may be
        # omitted when disabled, but a malformed present column is never
        # silently ignored.
        for column in AVAILABILITY_COLUMNS:
            if column not in availability.columns:
                continue
            values = availability[column]
            if not pd.api.types.is_bool_dtype(values.dtype):
                raise AvailabilityContractError(
                    f"availability column {column!r} must have boolean dtype"
                )
            if values.isna().any():
                raise AvailabilityContractError(
                    f"availability column {column!r} contains missing values"
                )
        aligned = availability.loc[body_index, list(required)]
        # The membership check above makes this a defensive assertion against
        # an exotic pandas index implementation; no fill/reindex repair is
        # allowed here.
        if aligned.isna().any().any():
            raise AvailabilityContractError(
                "availability sidecar alignment produced missing required values"
            )
        row_eligible = aligned.to_numpy(dtype=bool).all(axis=1)
        return AvailabilitySelection(
            sidecar=availability,
            required_columns=required,
            row_eligible=row_eligible,
        )

    if not isinstance(availability, Mapping):
        raise AvailabilityContractError(
            "availability must be a DataFrame or a mapping of boolean masks"
        )
    missing_columns = [column for column in required if column not in availability]
    if missing_columns:
        raise AvailabilityContractError(
            "availability mapping missing required columns: "
            + ", ".join(missing_columns)
        )
    masks: list[np.ndarray] = []
    normalized: dict[str, np.ndarray] = {}
    for column in AVAILABILITY_COLUMNS:
        if column not in availability:
            continue
        normalized[column] = _validate_boolean_array(
            availability[column],
            column=column,
            expected_len=len(body_index),
        )
    for column in required:
        masks.append(normalized[column])
    row_eligible = np.logical_and.reduce(masks) if masks else np.ones(len(body_index), dtype=bool)
    return AvailabilitySelection(
        sidecar=normalized,
        required_columns=required,
        row_eligible=row_eligible,
    )


def row_eligibility(
    availability: pd.DataFrame | Mapping[str, Any],
    feature_index: pd.DatetimeIndex,
    *,
    include_funding: bool = True,
    include_mark: bool = True,
) -> np.ndarray:
    """Return the fail-closed required-source mask for each body row."""
    return validate_availability(
        availability,
        feature_index,
        include_funding=include_funding,
        include_mark=include_mark,
    ).row_eligible


__all__ = [
    "AVAILABILITY_COLUMNS",
    "AvailabilityContractError",
    "AvailabilitySelection",
    "required_availability_columns",
    "row_eligibility",
    "validate_availability",
]
