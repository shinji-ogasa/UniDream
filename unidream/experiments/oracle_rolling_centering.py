"""Causal rolling centering of two already frozen, same-fold mean forecasts.

Only metadata and explicitly eligible values enter a decision. Callers bind the
same-model forecast provenance and reconstruct canonical h24 labels independently
of old scoring masks. Event-time label maturity does not prove receipt-time access.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import pandas as pd

GROUPS = ("technical", "perp_delay0")
WINDOW_MONTHS = 3
MATURITY_MINUTES = 375
STEP_MINUTES = 15


def _index(value, name):
    if not isinstance(value, pd.DatetimeIndex) or not len(value):
        raise ValueError(f"{name} must be a nonempty DatetimeIndex")
    if value.tz is None or str(value.tz) != "UTC" or value.hasnans:
        raise ValueError(f"{name} requires explicit UTC timestamps without NaT")
    step = pd.Timedelta(minutes=STEP_MINUTES).value
    if np.any(value.asi8 % step) or np.any(np.diff(value.asi8) != step):
        raise ValueError(f"{name} must be a sorted unique complete 15-minute grid")
    return value


def _mask(value, length, name):
    result = np.asarray(value)
    if result.shape != (length,) or result.dtype != np.dtype(bool):
        raise ValueError(f"{name} must be an aligned boolean vector")
    return result


def _vector(value, length, name):
    result = np.asarray(value, dtype=object)
    if result.shape != (length,):
        raise ValueError(f"{name} must be an aligned one-dimensional vector")
    return result


def _actual(value, length):
    result = np.asarray(value, dtype=object)
    if result.shape != (length, 3):
        raise ValueError("actual must have shape (N, 3)")
    return result


def _scalar(value, name):
    raw = np.asarray(value, dtype=object)
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a finite real scalar")
    raw = raw.item()
    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _selected(array, selected, name):
    return [_scalar(v, name) for v in array[selected]]


def _mean(values):
    values = list(values)
    try:
        result = math.fsum(float(v) / len(values) for v in values)
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("nonfinite or overflowing moment") from exc
    if not math.isfinite(result):
        raise ValueError("nonfinite or overflowing moment")
    return result


def _difference(left, right):
    result = [float(a) - float(b) for a, b in zip(left, right)]
    if not all(math.isfinite(v) for v in result):
        raise ValueError("nonfinite or overflowing difference")
    return result


def rolling_centered_forecasts(history_index, raw_predictions, actual, current_index, *,
                               history_forecast_mask, label_available_mask,
                               inference_mask, weights, minimum_pairs=64):
    """Return evaluation-calendar rolling means, availability and auditable traces.

    ``raw_predictions`` and ``weights`` have exactly technical/perp_delay0 keys.
    Raw vectors, actual (N,3), and both history masks align to the complete history
    grid; inference_mask aligns to its current-index subcalendar. Actual columns
    one and two are never inspected. Label availability is canonical observed-bar
    support, not an old fit/calibration/score mask. Origin availability must be
    shared between the two frozen models and restricted to UTC six-hour decisions.

    For each current I row, H contains origins in [t-3 calendar months,t) whose
    h24 outcome has matured (origin+375 minutes <= t), with both history masks
    true. Values are accessed only after forming H. Each selected value must be
    finite and real; unavailable or future values may contain arbitrary objects.
    Both raw averages and the return anchor use this same H and fsum(value/n).

    Forecasts are anchor + fixed_weight * (current_raw - historical_raw_mean).
    Weight zero copies the anchor exactly. Fewer than minimum_pairs fails closed
    on that decision without extending the window or filling values. The override
    exists for synthetic tests; a registered runtime caller must pin 64.
    Same-fold model identity and source hashes are caller responsibilities.
    """
    history = _index(history_index, "history_index")
    current = _index(current_index, "current_index")
    n, m = len(history), len(current)
    locations = history.get_indexer(current)
    if np.any(locations < 0):
        raise ValueError("current_index must be a subcalendar of history_index")
    if not isinstance(raw_predictions, Mapping) or set(raw_predictions) != set(GROUPS):
        raise ValueError("raw_predictions must name exactly technical and perp_delay0")
    raw = {g: _vector(raw_predictions[g], n, f"{g} raw") for g in GROUPS}
    y = _actual(actual, n)
    origin = _mask(history_forecast_mask, n, "history_forecast_mask")
    labels = _mask(label_available_mask, n, "label_available_mask")
    inference = _mask(inference_mask, m, "inference_mask")
    schedule = np.asarray((history.hour % 6 == 0) & (history.minute == 0))
    if np.any(origin & ~schedule):
        raise ValueError("history forecasts must be on UTC six-hour decisions")
    if np.any(inference & ~origin[locations]):
        raise ValueError("current inference requires origin-available shared forecasts")
    if not isinstance(weights, Mapping) or set(weights) != set(GROUPS):
        raise ValueError("weights must name exactly technical and perp_delay0")
    w = {g: _scalar(weights[g], f"{g} weight") for g in GROUPS}
    if any(v < 0 or v > 1 for v in w.values()):
        raise ValueError("fixed weights must lie in [0, 1]")
    if isinstance(minimum_pairs, (bool, np.bool_)) or not isinstance(minimum_pairs, Integral) or minimum_pairs < 1:
        raise ValueError("minimum_pairs must be a positive integer")
    names = ("rolling_anchor", *(g + "_rolling" for g in GROUPS))
    means = {name: np.full(m, np.nan) for name in names}
    available = np.zeros(m, dtype=bool)
    counts = np.zeros(m, dtype=np.int64)
    reasons = np.full(m, "not_inference", dtype="<U24")
    trace = []
    for i in np.flatnonzero(inference):
        t = current[i]
        lower = t - pd.DateOffset(months=WINDOW_MONTHS)
        if lower < history[0]:
            raise ValueError("history grid must cover the full three-calendar-month window")
        maturity_limit = t - pd.Timedelta(minutes=MATURITY_MINUTES)
        # Timestamp eligibility must precede any selected raw/label value access.
        temporal = np.asarray((history >= lower) & (history < t) & (history <= maturity_limit))
        selected = temporal & origin & labels
        selected_index = history[selected]
        count = int(selected.sum())
        counts[i] = count
        current_raw = {g: _scalar(raw[g][locations[i]], f"{g} current raw") for g in GROUPS}
        observed = _selected(y[:, 0], selected, "mature actual")
        history_raw = {g: _selected(raw[g], selected, f"{g} mature raw") for g in GROUPS}
        anchor, raw_means = None, {g: None for g in GROUPS}
        forecasts = {name: None for name in names}
        if count >= minimum_pairs:
            anchor = _mean(observed)
            raw_means = {g: _mean(history_raw[g]) for g in GROUPS}
            forecasts["rolling_anchor"] = anchor
            for g in GROUPS:
                value = anchor if w[g] == 0 else anchor + w[g] * (current_raw[g] - raw_means[g])
                if not math.isfinite(value):
                    raise ValueError("nonfinite or overflowing rolling forecast")
                forecasts[g + "_rolling"] = value
            for name, value in forecasts.items():
                means[name][i] = value
            available[i], reasons[i] = True, "available"
        else:
            reasons[i] = "insufficient_history"
        trace.append({
            "decision_at": t.isoformat(), "reason": str(reasons[i]),
            "history_count": count, "minimum_pairs": int(minimum_pairs),
            "window_start": lower.isoformat(), "window_end_exclusive": t.isoformat(),
            "maturity_limit_origin": maturity_limit.isoformat(),
            "oldest_origin": selected_index[0].isoformat() if count else None,
            "latest_origin": selected_index[-1].isoformat() if count else None,
            "latest_maturity": (selected_index[-1] + pd.Timedelta(minutes=MATURITY_MINUTES)).isoformat() if count else None,
            "history_timestamp_sha256": hashlib.sha256(selected_index.asi8.tobytes()).hexdigest(),
            "forecast_history_count": int((temporal & origin).sum()),
            "mature_label_missing_count": int((temporal & origin & ~labels).sum()),
            "rolling_anchor": anchor, "raw_means": raw_means, "weights": dict(w),
            "current_raw": current_raw, "forecasts": forecasts,
        })
    return {"means": means, "available": available, "paired_count": counts,
            "reason_code": reasons, "trace": trace}


def score_decomposition(actual, mu, anchor_array, score_mask):
    """Descriptive MSE decomposition allowing a different anchor at each row.

    On at least 16 fixed scored rows, d=mu-a and r=y-a. Population moments give
    MSE(mu)-MSE(a) = Var(d)-2Cov(d,r) + mean(d)^2-2mean(d)mean(r).
    The identity holds for a varying causal anchor; it is not a calibration or
    significance test. Only selected return values are inspected.
    """
    prediction = np.asarray(mu, dtype=object)
    if prediction.ndim != 1 or not len(prediction):
        raise ValueError("mu must be a nonempty one-dimensional vector")
    n = len(prediction)
    y, anchors = _actual(actual, n), _vector(anchor_array, n, "anchor")
    mask = _mask(score_mask, n, "score_mask")
    if int(mask.sum()) < 16:
        raise ValueError("at least 16 scored rows required")
    observed = _selected(y[:, 0], mask, "actual")
    predicted, a = _selected(prediction, mask, "mu"), _selected(anchors, mask, "anchor")
    d, r = _difference(predicted, a), _difference(observed, a)
    error = _difference(observed, predicted)
    mean_d, mean_r = _mean(d), _mean(r)
    centered_d = _difference(d, [mean_d] * len(d))
    centered_r = _difference(r, [mean_r] * len(r))
    b, c = _mean(v * v for v in d), _mean(x * z for x, z in zip(d, r))
    variance = _mean(v * v for v in centered_d)
    covariance = _mean(x * z for x, z in zip(centered_d, centered_r))
    candidate_mse, anchor_mse = _mean(v * v for v in error), _mean(v * v for v in r)
    lossdiff = candidate_mse - anchor_mse
    centered = variance - 2 * covariance
    drift = mean_d * mean_d - 2 * mean_d * mean_r
    result = {"n": len(d), "candidate_mse": candidate_mse, "anchor_mse": anchor_mse,
        "lossdiff": lossdiff, "mean_d": mean_d, "mean_r": mean_r,
        "innovation_secondmoment": b, "crossmoment": c,
        "centered_variance_d": variance, "centered_covariance": covariance,
        "centered_component": centered, "drift_component": drift,
        "identityresidual": lossdiff - (centered + drift)}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite or overflowing decomposition")
    return result


__all__ = ["rolling_centered_forecasts", "score_decomposition"]
