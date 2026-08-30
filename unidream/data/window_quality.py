"""Gap-aware sequence-window eligibility checks.

The training pipeline historically receives a dense ``ndarray`` and therefore
cannot tell whether adjacent rows are adjacent market bars.  These helpers
accept the original timestamp index and, optionally, the v4 availability
sidecar.  They return only windows whose rows are all required-source
observed and contiguous.  No rows are sorted, dropped, or filled here.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .availability_contract import (
    AvailabilityContractError,
    validate_availability,
)


class WindowQualityError(ValueError):
    """Raised when timestamps or an observation mask cannot be trusted."""


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
        raise WindowQualityError(f"unsupported interval: {interval!r}")
    return values[interval]


def _validate_index(index: Any) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise WindowQualityError("window timestamps must be a DatetimeIndex")
    if len(index) == 0:
        return index
    if not index.is_unique:
        duplicates = index[index.duplicated(keep=False)].unique()
        raise WindowQualityError(
            "window timestamps contain duplicate timestamps: "
            + ", ".join(str(value) for value in duplicates[:5])
        )
    if not index.is_monotonic_increasing:
        raise WindowQualityError("window timestamps are not strictly increasing; refusing to sort")
    return index


def valid_sequence_starts(
    index: pd.DatetimeIndex,
    seq_len: int,
    *,
    interval: str = "15m",
    spot_bar_observed: np.ndarray | pd.Series | None = None,
    availability: pd.DataFrame | Mapping[str, Any] | None = None,
    include_funding: bool = True,
    include_mark: bool = True,
) -> np.ndarray:
    """Return row offsets whose full sequence is gap-free and observed.

    A start is valid only when every adjacent pair in the ``seq_len`` rows is
    exactly one configured interval apart.  If ``spot_bar_observed`` is
    supplied it must align one-for-one with ``index`` and every row in the
    window must be ``True``.  ``availability`` applies the same rule to the
    required Spot/funding/mark sidecar columns.  The returned offsets index the
    caller's original rows, so no hidden reindexing or compaction occurs.
    """
    if not isinstance(seq_len, (int, np.integer)) or isinstance(seq_len, bool) or seq_len <= 0:
        raise WindowQualityError(f"seq_len must be a positive integer, got {seq_len!r}")
    timestamps = _validate_index(index)
    delta = _interval_delta(interval)
    row_count = len(timestamps)
    if availability is not None:
        try:
            selected = validate_availability(
                availability,
                timestamps,
                include_funding=include_funding,
                include_mark=include_mark,
            )
        except AvailabilityContractError as exc:
            raise WindowQualityError(str(exc)) from exc
        observed = selected.row_eligible
        if spot_bar_observed is not None:
            explicit_spot = np.asarray(spot_bar_observed)
            if explicit_spot.ndim != 1 or len(explicit_spot) != row_count:
                raise WindowQualityError(
                    "spot_bar_observed must be a one-dimensional mask aligned to timestamps"
                )
            if explicit_spot.dtype != np.bool_:
                raise WindowQualityError("spot_bar_observed must have boolean dtype")
            # A caller may pass the historical standalone mask together with a
            # sidecar, but contradictory values are a contract violation.
            sidecar_spot = validate_availability(
                availability,
                timestamps,
                include_funding=False,
                include_mark=False,
            ).row_eligible
            if not np.array_equal(explicit_spot, sidecar_spot):
                raise WindowQualityError(
                    "spot_bar_observed conflicts with availability sidecar"
                )
    elif spot_bar_observed is not None:
        observed = np.asarray(spot_bar_observed)
        if observed.ndim != 1 or len(observed) != row_count:
            raise WindowQualityError(
                "spot_bar_observed must be a one-dimensional mask aligned to timestamps"
            )
        if observed.dtype != np.bool_:
            raise WindowQualityError("spot_bar_observed must have boolean dtype")
    else:
        observed = np.ones(row_count, dtype=bool)

    if row_count < seq_len:
        return np.empty(0, dtype=np.int64)
    if seq_len == 1:
        return np.flatnonzero(observed).astype(np.int64)

    differences = timestamps[1:] - timestamps[:-1]
    contiguous_edges = np.asarray(differences == delta, dtype=bool)
    # A prefix sum makes the check O(T), while preserving original row offsets.
    broken_edges = (~contiguous_edges).astype(np.int64)
    edge_prefix = np.concatenate(([0], np.cumsum(broken_edges)))
    starts = np.arange(row_count - seq_len + 1, dtype=np.int64)
    edge_end = starts + seq_len - 1
    gap_free = (edge_prefix[edge_end] - edge_prefix[starts]) == 0
    observed_prefix = np.concatenate(([0], np.cumsum((~observed).astype(np.int64))))
    observed_free = (observed_prefix[starts + seq_len] - observed_prefix[starts]) == 0
    return starts[gap_free & observed_free]


def window_is_gap_free(
    index: pd.DatetimeIndex,
    start: int,
    seq_len: int,
    *,
    interval: str = "15m",
    spot_bar_observed: np.ndarray | pd.Series | None = None,
    availability: pd.DataFrame | Mapping[str, Any] | None = None,
    include_funding: bool = True,
    include_mark: bool = True,
) -> bool:
    """Check one sequence start using the same fail-closed contract."""
    if not isinstance(start, (int, np.integer)) or isinstance(start, bool):
        raise WindowQualityError(f"start must be an integer, got {start!r}")
    return int(start) in set(
        valid_sequence_starts(
            index,
            seq_len,
            interval=interval,
            spot_bar_observed=spot_bar_observed,
            availability=availability,
            include_funding=include_funding,
            include_mark=include_mark,
        ).tolist()
    )


__all__ = ["WindowQualityError", "valid_sequence_starts", "window_is_gap_free"]
