"""Scale-only mean reliability and drift-aware descriptive loss decomposition.

No feature construction, prediction model fit, scoring-mask selection, or policy
rollout. Callers bind the chronological scale/inference/score provenance.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np


def _mask(value, *, length, minimum):
    mask = np.asarray(value)
    if mask.dtype != np.dtype(bool) or mask.shape != (length,):
        raise ValueError("aligned one-dimensional boolean mask required")
    if int(mask.sum()) < minimum:
        raise ValueError(f"at least {minimum} selected rows required")
    return mask


def _scalar(value, name):
    scalar = np.asarray(value, dtype=object)
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be a finite real scalar")
    raw = scalar.item()
    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        number = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _vector(value, length=None):
    # Object conversion preserves booleans in mixed Python lists until only
    # the explicitly selected values are type-checked and converted to float.
    array = np.asarray(value, dtype=object)
    if array.ndim != 1 or not len(array) or (length is not None and len(array) != length):
        raise ValueError("aligned nonempty one-dimensional array required")
    return array


def _actual(value, length):
    array = np.asarray(value, dtype=object)
    if array.shape != (length, 3):
        raise ValueError("actual must have shape (N, 3)")
    return array


def _selected(array, mask, name):
    return np.asarray([_scalar(v, name) for v in array[mask]], dtype=float)


def _mean(values):
    values = list(values)
    try:
        result = math.fsum(float(v) / len(values) for v in values)
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("nonfinite or overflowing moment") from exc
    if not math.isfinite(result):
        raise ValueError("nonfinite or overflowing moment")
    return result


def _differences(left, right):
    result = [float(a) - float(b) for a, b in zip(left, right)]
    if not all(math.isfinite(v) for v in result):
        raise ValueError("nonfinite or overflowing difference")
    return result


def _constant(array, mask):
    selected = _selected(array, mask, "anchor")
    if not np.all(selected == selected[0]):
        raise ValueError("anchor must be exactly constant on selected support")
    return selected


def fit_reliability(full_mu, actual, *, scale_mask, anchor):
    """Fit w in [0,1] on at least 64 supplied scale rows, ignoring all others.

    d=full_mu-anchor, r=y-anchor, B=mean(d²), C=mean(d*r). The unconstrained
    coefficient C/B is clipped by fixed algebraic cases, not selected from an
    old weight grid. B=0 is explicitly unidentified and returns zero. Anchor
    must exactly match the saved scale mean's inherited fsum arithmetic.
    Only actual column zero is inspected; future/interval labels are ignored.
    """
    full = _vector(full_mu)
    y = _actual(actual, len(full))
    mask = _mask(scale_mask, length=len(full), minimum=64)
    anchor_value = _scalar(anchor, "anchor")
    observed, predicted = _selected(y[:, 0], mask, "actual"), _selected(full, mask, "full_mu")
    if anchor_value != _mean(observed):
        raise ValueError("anchor must exactly equal the saved fsum scale mean")
    constant = [anchor_value] * len(observed)
    d, r = _differences(predicted, constant), _differences(observed, constant)
    b, c = _mean(v * v for v in d), _mean(a * z for a, z in zip(d, r))
    if b == 0:
        weight, case = 0., "zero_dispersion"
    elif c <= 0:
        weight, case = 0., "nonpositive_crossmoment"
    elif c >= b:
        weight, case = 1., "upper_endpoint"
    else:
        weight, case = c / b, "interior"
    if not math.isfinite(weight):
        raise ValueError("nonfinite reliability weight")
    return {"weight": float(weight), "n": len(d), "anchor": anchor_value,
        "mean_d": _mean(d), "mean_r": _mean(r), "innovation_secondmoment": b,
        "crossmoment": c, "identifiable": b > 0, "weight_case": case}


def apply_reliability(full_mu, anchor_array, *, inference_mask, weight):
    """Apply the frozen weight to unchanged inference support; no labels enter.

    Exact weight-zero/one branches copy the corresponding endpoint, including
    signed zeros. Interior weights use a convex sum; weight .5 matches half_mean.
    All unselected values are ignored and the output remains NaN there.
    """
    full = _vector(full_mu)
    anchor = _vector(anchor_array, len(full))
    mask = _mask(inference_mask, length=len(full), minimum=1)
    w = _scalar(weight, "weight")
    if not 0 <= w <= 1:
        raise ValueError("weight must lie in [0, 1]")
    predicted, constant = _selected(full, mask, "full_mu"), _constant(anchor, mask)
    result = np.full(len(full), np.nan)
    if w == 0:
        result[mask] = constant
    elif w == 1:
        result[mask] = predicted
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            result[mask] = w * predicted + (1 - w) * constant
    if not np.isfinite(result[mask]).all():
        raise ValueError("nonfinite or overflowing convex forecast")
    return result


def score_decomposition(actual, mu, anchor_array, score_mask):
    """Population-moment MSE decomposition on at least 16 fixed scored rows.

    With d=mu-anchor and r=y-anchor, lossdiff=B-2C equals
    (Var(d)-2Cov(d,r)) + (mean(d)²-2mean(d)mean(r)). Do not assume that
    evaluation innovations retain their scale-period mean. The returned
    identityresidual is direct lossdiff minus centered_component+drift_component.
    These are retrospective descriptors, not a new calibration or significance test.
    """
    prediction = _vector(mu)
    y, anchors = _actual(actual, len(prediction)), _vector(anchor_array, len(prediction))
    mask = _mask(score_mask, length=len(prediction), minimum=16)
    observed = _selected(y[:, 0], mask, "actual")
    predicted, constant = _selected(prediction, mask, "mu"), _constant(anchors, mask)
    d, r = _differences(predicted, constant), _differences(observed, constant)
    error = _differences(observed, predicted)
    mean_d, mean_r = _mean(d), _mean(r)
    centered_d = _differences(d, [mean_d] * len(d))
    centered_r = _differences(r, [mean_r] * len(r))
    b, c = _mean(v * v for v in d), _mean(a * z for a, z in zip(d, r))
    variance = _mean(v * v for v in centered_d)
    covariance = _mean(a * z for a, z in zip(centered_d, centered_r))
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


__all__ = ["fit_reliability", "apply_reliability", "score_decomposition"]
