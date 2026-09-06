"""Selected-support probability and direction diagnostics, without model fitting.

Absolute-return-weighted probabilities target a magnitude-tilted distribution;
they are not claims about the ordinary probability of a positive return. Scores
are descriptive and cannot establish economic value or independent skill.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np


def _finite_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must contain real numeric scalars, not bool or complex")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real numeric scalars") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite on score support")
    return result


def _finite_sum(values, name):
    try:
        result = math.fsum(values)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} is not representable as a finite float") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _mean(values, name):
    n = len(values)
    try:
        # Sum first preserves subnormal values that individual division loses.
        total = math.fsum(values)
    except OverflowError:
        # A representable mean may still have an overflowing intermediate sum.
        return _finite_sum((v / n for v in values), name)
    return _finite_real(total / n, name)


def _weighted_mean(values, weights, total, name):
    if total == 0.0:
        return None
    # A constant score is reproduced exactly, including maximal finite losses.
    if all(v == values[0] for v in values):
        return values[0]
    return _finite_sum((a / total * v for a, v in zip(weights, values)), name)


def direction_scores(actual, logits, score_mask):
    """Score selected return labels and logits, returning JSON scalar metrics.

    ``actual`` has shape (N, 3), ``logits`` has shape (N,), and ``score_mask``
    is a nonempty strict boolean vector with at least one selected row. Only
    ``actual[score_mask, 0]`` and ``logits[score_mask]`` are inspected as values;
    all other outcome cells and unselected logits may contain arbitrary objects.
    Selected booleans, complex values, strings and nonfinite scalars fail.

    The binary outcome is y > 0 and its prediction is z > 0. A zero logit is a
    non-positive binary prediction but contributes sign(z)=0 to the uncosted
    signed-return diagnostic. Probabilities use a scalar float64 sigmoid; log
    loss uses logaddexp(0, z) - (y > 0) * z, with no probability/loss clipping.
    Magnitude-weighted losses divide by sum(abs(y)) and are null when it is zero.
    They assess a tilted target, not ordinary event-probability calibration.
    This function neither changes support nor produces trading actions.
    """
    mask = np.asarray(score_mask)
    if (mask.ndim != 1 or not len(mask) or mask.dtype != np.dtype(bool)
            or not mask.any()):
        raise ValueError("score_mask must be a nonempty strict boolean vector with selected rows")
    observed = np.asarray(actual, dtype=object)
    margins = np.asarray(logits, dtype=object)
    if observed.shape != (len(mask), 3):
        raise ValueError("actual must have shape (N, 3)")
    if margins.shape != mask.shape:
        raise ValueError("logits must be an aligned one-dimensional vector")
    y = [_finite_real(v, "selected return") for v in observed[mask, 0]]
    z = [_finite_real(v, "selected logit") for v in margins[mask]]
    absolute = [abs(v) for v in y]
    total = _finite_sum(absolute, "absolute_return_sum")
    losses, briers, correct, signed = [], [], [], []
    for value, margin in zip(y, z):
        label = int(value > 0.0)
        exponential = math.exp(-abs(margin))
        probability = (1.0 / (1.0 + exponential) if margin >= 0.0
                       else exponential / (1.0 + exponential))
        loss = float(np.logaddexp(0.0, margin)) - label * margin
        losses.append(_finite_real(loss, "log loss"))
        briers.append((probability - label) ** 2)
        correct.append(float((margin > 0.0) == bool(label)))
        signed.append((float(margin > 0.0) - float(margin < 0.0)) * value)
    return {
        "rows": len(y),
        "zero_actual_rows": sum(v == 0.0 for v in y),
        "zero_logit_rows": sum(v == 0.0 for v in z),
        "log_loss": _mean(losses, "log_loss"),
        "brier": _mean(briers, "brier"),
        "binary_accuracy": _mean(correct, "binary_accuracy"),
        "signed_return_mean": _mean(signed, "signed_return_mean"),
        "weighted_log_loss": _weighted_mean(losses, absolute, total, "weighted_log_loss"),
        "weighted_brier": _weighted_mean(briers, absolute, total, "weighted_brier"),
        "weighted_binary_accuracy": _weighted_mean(correct, absolute, total,
                                                    "weighted_binary_accuracy"),
        "absolute_return_sum": total,
        "absolute_return_mean": _mean(absolute, "absolute_return_mean"),
    }


__all__ = ["direction_scores"]
