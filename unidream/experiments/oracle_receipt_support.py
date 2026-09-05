"""Decision-time eligibility of supplied bar metadata, including receipt time.

This module does not inspect market values, labels, forecasts or a live feed.
Per-row eligibility does not establish complete rolling feature history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


INTERVAL = pd.Timedelta(minutes=15)


def _times(value, name, *, length=None, broadcast=False, missing=False):
    if pd.api.types.is_scalar(value):
        if not broadcast:
            raise ValueError(f"{name} must be a one-dimensional timestamp array")
        values = [value] * length
    else:
        try:
            array = np.asarray(value, dtype=object)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a one-dimensional timestamp array") from exc
        if array.ndim != 1 or (length is not None and len(array) != length):
            raise ValueError(f"{name} must be one-dimensional and positionally aligned")
        values = array.tolist()
    if not values:
        raise ValueError("nonempty required-bar metadata needed")
    parsed = []
    for value in values:
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid {name} timestamp") from exc
        if pd.isna(stamp):
            if not missing:
                raise ValueError(f"{name} cannot contain missing timestamps")
            parsed.append(pd.NaT)
        else:
            if stamp.tzinfo is None:
                raise ValueError(f"{name} timestamps must be timezone-aware")
            parsed.append(stamp.tz_convert("UTC"))
    try:
        return pd.DatetimeIndex(parsed, tz="UTC").as_unit("ns")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} timestamps must fit the UTC nanosecond calendar") from exc


def receipt_support(event_open, event_close, received_at, decision_at, *,
                    decision_deadline=None, step="15min") -> pd.DataFrame:
    """Audit the supplied required bar rows at their specified decision times.

    ``event_open`` and ``event_close`` are nonempty aligned 1D arrays, with
    unique increasing raw opens on the UTC 15-minute grid. The inherited raw
    close convention is open + 15 minutes - 1 millisecond. ``received_at`` and
    ``decision_at`` may each be an aware scalar broadcast to all rows, or an
    aligned 1D array. Alignment is positional, never pandas label alignment.
    Only received_at may be unknown (None/NaT). Aware timezone representations
    are normalized to UTC; naive timestamps are rejected rather than localized.

    decision_at is the nominal feature origin, on the 15-minute grid. Optional
    decision_deadline is a scalar or aligned aware timestamp array in
    [decision_at, decision_at + 15 minutes); by default it equals decision_at.
    Thus a shortly delayed final-bar receipt can be admitted without admitting
    the current unfinished feature bar or data after the next-open fill.

    A feature bar is usable only at open + 15 minutes or later and when its
    receipt is no later than the deadline. A receipt before that full-bar
    boundary is inconsistent with finalized-bar receipt metadata and rejected.
    This input must describe receipt of the completed bar, not an earlier
    provisional candle update. Receipt authenticity is the caller's contract.

    Reason precedence is BAR_NOT_CLOSED, RECEIPT_UNKNOWN, RECEIPT_LATE, ELIGIBLE.
    Sparse required-bar rows are retained without filling: callers must verify
    the entire required history, coverage, field validity and provenance. This
    helper makes no finding about separately observed current-open availability.
    """
    try:
        interval = pd.Timedelta(step)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("registered step must be 15 minutes") from exc
    if interval != INTERVAL:
        raise ValueError("registered step must be 15 minutes")
    opens = _times(event_open, "event_open")
    closes = _times(event_close, "event_close", length=len(opens))
    receipts = _times(received_at, "received_at", length=len(opens), broadcast=True, missing=True)
    decisions = _times(decision_at, "decision_at", length=len(opens), broadcast=True)
    deadlines = decisions if decision_deadline is None else _times(
        decision_deadline, "decision_deadline", length=len(opens), broadcast=True)
    if not opens.is_unique or not opens.is_monotonic_increasing:
        raise ValueError("event_open timestamps must be unique and increasing")
    if np.any(opens.asi8 % INTERVAL.value) or np.any(decisions.asi8 % INTERVAL.value):
        raise ValueError("event_open and decision_at must align to the UTC 15-minute grid")
    if np.any(deadlines < decisions) or np.any(deadlines >= decisions + INTERVAL):
        raise ValueError("deadline must be at or after nominal decision and strictly before next-open fill")
    try:
        ends = opens + INTERVAL
    except (ValueError, OverflowError) as exc:
        raise ValueError("bar end exceeds the supported timestamp calendar") from exc
    if not closes.equals(ends - pd.Timedelta(milliseconds=1)):
        raise ValueError("event_close must equal event_open + 15min - 1ms")
    receipt_known = ~receipts.isna()
    if np.any(receipt_known & (receipts < ends)):
        raise ValueError("finalized-bar receipt cannot precede the full-bar end")
    # Equivalent to open <= decision - one bar, without subtracting near the
    # lower nanosecond timestamp bound.
    prior = np.asarray(ends <= decisions)
    closed = np.asarray(ends <= decisions)
    event_eligible = prior & closed
    received_in_time = receipt_known & np.asarray(receipts <= deadlines)
    eligible = event_eligible & received_in_time
    late = receipt_known & np.asarray(receipts > deadlines)
    reason = np.full(len(opens), "ELIGIBLE", dtype=object)
    reason[late] = "RECEIPT_LATE"
    reason[~receipt_known] = "RECEIPT_UNKNOWN"
    reason[~event_eligible] = "BAR_NOT_CLOSED"
    result = pd.DataFrame({
        "event_close": closes, "bar_end": ends, "received_at": receipts, "decision_at": decisions,
        "decision_deadline": deadlines,
        "is_prior_bar": prior, "bar_closed_by_decision": closed,
        "event_time_eligible": event_eligible, "receipt_known": receipt_known,
        "received_by_deadline": received_in_time, "receipt_eligible": eligible,
        "archive_event_time_only": event_eligible & ~receipt_known,
        "receipt_late": late, "reason": reason,
    }, index=opens.rename("event_open"))
    result.attrs.update(schema="oracle-receipt-support-v1", step_nanoseconds=INTERVAL.value,
        full_feature_history_verified=False, bar_values_validated=False,
        current_open_availability_evaluated=False, receipt_authenticity_verified=False)
    return result


__all__ = ["receipt_support"]
