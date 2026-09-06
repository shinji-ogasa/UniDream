"""Fixed no-fit probability-to-mean mapping and its frozen constant controls."""
from __future__ import annotations

import math
from numbers import Real

import numpy as np


MEAN_IDS = ("soft", "mapped_prior", "fit_mean", "zero")
PRIOR_IDENTITY_ATOL = 1e-14
PRIOR_IDENTITY_RTOL = 1e-12


def _finite_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real numeric scalar, not bool or complex")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real numeric scalar") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def map_soft_direction(probabilities, *, inference_mask, fit_abs_return_mean,
                       saved_weighted_prior_probability, fit_return_mean) -> dict:
    """Map saved probabilities as a_T * (2.0*q - 1.0), without re-estimation.

    ``probabilities`` and ``inference_mask`` are aligned one-dimensional vectors.
    The mask is strictly boolean and selects at least one row. Only selected
    probabilities are inspected; they must be finite real numbers in [0, 1].
    Unselected cells may contain arbitrary objects. Boolean/complex/string
    selected values and non-scalar statistics are rejected without coercion.

    The caller supplies a positive finite saved T mean absolute return, a saved
    magnitude-weighted prior probability in [0, 1], and a finite saved T return
    mean. Use the stored probability, not a statistical prior or recomputed
    sigmoid of a prior logit; their floating values can differ. The
    mapped prior uses the identical arithmetic. Its difference from the stored
    fit mean must satisfy abs(diff) <= 1e-14 + 1e-12*abs(fit_return_mean).
    Both constants are retained separately; this check does not replace one
    with the other. The stored fit mean is copied exactly, including signed zero.

    Returns four independent full-length float64 arrays under ``means``:
    soft, mapped_prior, fit_mean, zero. All are NaN outside inference. q=0.5
    maps exactly to zero without an epsilon. The first two means have absolute
    value at most a_T; that bound is not imposed on the separately stored mean.
    Underflow to zero is not repaired. No sigmoid, clipping, fit, outcome input,
    score mask, model selection, calendar validation or receipt proof is used.
    The caller binds saved input provenance and causal feature chronology.
    """
    mask = np.asarray(inference_mask)
    if (mask.ndim != 1 or not len(mask) or mask.dtype != np.dtype(bool)
            or not mask.any()):
        raise ValueError("inference_mask must be a nonempty strict boolean vector with selected rows")
    # Preserve types in mixed Python lists until selected-value validation.
    raw = np.asarray(probabilities, dtype=object)
    if raw.shape != mask.shape:
        raise ValueError("probabilities must be an aligned one-dimensional vector")
    selected = np.asarray([_finite_real(v, "selected probability") for v in raw[mask]],
                          dtype=np.float64)
    if np.any(selected < 0.) or np.any(selected > 1.):
        raise ValueError("selected probabilities must be in [0, 1]")

    amplitude = _finite_real(fit_abs_return_mean, "fit_abs_return_mean")
    prior = _finite_real(saved_weighted_prior_probability, "saved_weighted_prior_probability")
    stored_mean = _finite_real(fit_return_mean, "fit_return_mean")
    if amplitude <= 0.:
        raise ValueError("fit_abs_return_mean must be strictly positive")
    if not 0. <= prior <= 1.:
        raise ValueError("saved_weighted_prior_probability must be in [0, 1]")

    mapped_prior = amplitude * (2.0 * prior - 1.0)
    residual = mapped_prior - stored_mean
    tolerance = PRIOR_IDENTITY_ATOL + PRIOR_IDENTITY_RTOL * abs(stored_mean)
    if (not math.isfinite(mapped_prior) or not math.isfinite(residual)
            or not math.isfinite(tolerance) or abs(residual) > tolerance):
        raise ValueError("mapped prior and stored fit mean violate the fixed identity tolerance")
    # q in [0, 1] and finite a_T guarantee a bounded finite product.
    soft = amplitude * (2.0 * selected - 1.0)
    if not np.isfinite(soft).all() or np.any(np.abs(soft) > amplitude):
        raise ValueError("nonfinite or out-of-bound soft mean")
    means = {name: np.full(len(mask), np.nan, dtype=np.float64) for name in MEAN_IDS}
    means["soft"][mask] = soft
    means["mapped_prior"][mask] = mapped_prior
    means["fit_mean"][mask] = stored_mean
    means["zero"][mask] = 0.
    diagnostic = {
        "schema": "oracle-soft-direction-mapping-v1",
        "formula": "fit_abs_return_mean * (2.0 * saved_probability - 1.0)",
        "prior_formula": "fit_abs_return_mean * (2.0 * saved_weighted_prior_probability - 1.0)",
        "fit_abs_return_mean": amplitude, "saved_weighted_prior_probability": prior,
        "fit_return_mean": stored_mean, "mapped_prior": mapped_prior,
        "prior_identity_signed_difference": residual,
        "prior_identity_absolute_difference": abs(residual),
        "prior_identity_tolerance": tolerance,
        "prior_identity_atol": PRIOR_IDENTITY_ATOL, "prior_identity_rtol": PRIOR_IDENTITY_RTOL,
        "prior_identity_passed": True,
        "total_rows": len(mask), "inference_rows": int(mask.sum()),
        "noninference_rows": int((~mask).sum()),
        "probability_half_rows": int((selected == .5).sum()),
        "model_fits": 0, "calibration_fits": 0,
        "probabilities_recomputed": False, "future_outcomes_or_score_support_used": False,
        "calendar_or_receipt_causality_verified": False,
        "saved_statistics_provenance_verified": False,
    }
    return {"means": means, "inference_mask": mask.copy(), "diagnostic": diagnostic}


__all__ = ["MEAN_IDS", "PRIOR_IDENTITY_ATOL", "PRIOR_IDENTITY_RTOL", "map_soft_direction"]
