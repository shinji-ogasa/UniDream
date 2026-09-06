"""Hindsight-only sign or magnitude substitution on frozen scoring support.

This diagnostic changes a component of return forecasts, never their risk or
availability. Unscored inference rows retain their learned return forecasts.
It is not a deployable predictor, training teacher or global optimum.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np

COMPONENTS = ("sign", "magnitude")


def _real(value, name, *, allow_nan=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must contain real numeric scalars, not bool or complex")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric scalars") from exc
    if not math.isfinite(number) and not (allow_nan and math.isnan(number)):
        raise ValueError(f"{name} must be finite on selected support")
    return number


def _frozen_forecast(value, inference, name, *, nonnegative=False):
    # Object conversion preserves booleans in mixed Python lists until validation.
    array = np.asarray(value, dtype=object)
    if array.shape != inference.shape:
        raise ValueError(f"{name} must be an aligned one-dimensional vector")
    result = np.empty(len(inference), dtype=float)
    for i, raw in enumerate(array):
        number = _real(raw, name, allow_nan=not inference[i])
        if inference[i]:
            if nonnegative and number < 0:
                raise ValueError("variance must be nonnegative on inference support")
        elif not math.isnan(number):
            raise ValueError("unavailable frozen forecasts must be explicitly NaN")
        result[i] = number
    return result


def substitute_return_component(mu, variance, *, inference_mask, score_support,
                                actual, component):
    """Substitute future return sign or magnitude on the existing score rows.

    sign: sign(y) * abs(mu); magnitude: sign(mu) * abs(y), with y=actual[:,0].
    Only selected return values are inspected: other outcome columns and every
    unscored outcome may contain arbitrary objects. Selected values must be
    finite real scalars; booleans, complex values and numeric strings fail.

    Inference and scoring are aligned strict boolean vectors, with a nonempty
    score subset of inference. Frozen mean/risk values must be finite on
    inference (risk nonnegative) and explicit NaN elsewhere. Return arrays are
    float64 copies. Risk values and mean values outside score are unchanged,
    including signed zeros. Missing inputs never become available. np.sign(0)
    is zero without an epsilon or a directional tie breaker.

    This function intentionally uses future information on scored rows. It
    neither creates scoring support nor establishes causality or optimality.
    """
    if not isinstance(component, str) or component not in COMPONENTS:
        raise ValueError("component must be exactly sign or magnitude")
    inference, score = np.asarray(inference_mask), np.asarray(score_support)
    if (inference.ndim != 1 or not len(inference) or inference.dtype != np.dtype(bool)
            or score.shape != inference.shape or score.dtype != np.dtype(bool)
            or not score.any() or np.any(score & ~inference)):
        raise ValueError("nonempty frozen score subset of aligned boolean inference required")
    mean = _frozen_forecast(mu, inference, "mu")
    risk = _frozen_forecast(variance, inference, "variance", nonnegative=True)
    observed = np.asarray(actual, dtype=object)
    if observed.shape != (len(inference), 3):
        raise ValueError("actual must have shape (N, 3)")
    # Select the allowed rows AND the sole used column before inspecting values.
    y = np.asarray([_real(v, "scored actual return") for v in observed[score, 0]], dtype=float)
    old_mean = mean[score].copy()
    if component == "sign":
        mean[score] = np.sign(y) * np.abs(old_mean)
    else:
        mean[score] = np.sign(old_mean) * np.abs(y)
    if not np.isfinite(mean[inference]).all():
        raise ValueError("nonfinite component substitution")
    metadata = {
        "diagnostic_kind": "hybrid_hindsight_return_component_substitution", "component": component,
        "formula": "sign(y) * abs(mu)" if component == "sign" else "sign(mu) * abs(y)",
        "hindsight_only": True, "future_information_used_for_decisions": True,
        "deployable": False, "teacher_use_allowed": False, "global_optimum_claimed": False,
        "inference_rows": int(inference.sum()), "replacement_rows": int(score.sum()),
        "learned_remainder_rows": int((inference & ~score).sum()),
        "inference_and_missing_action_support_unchanged": True, "variance_unchanged": True,
        "other_outcome_columns_used": False, "zero_sign_rule": "np.sign(0) = 0",
    }
    return {"mu": mean, "variance": risk, "inference_mask": inference.copy(),
            "score_support": score.copy(), "metadata": metadata}


__all__ = ["COMPONENTS", "substitute_return_component"]
