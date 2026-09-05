"""Causal constant-mean controls and separate validation return scoring."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata


def _mask(value, name, *, length=None):
    mask = np.asarray(value)
    if mask.ndim != 1 or mask.dtype != np.dtype(bool):
        raise ValueError(f"{name} must be a one-dimensional boolean mask")
    if length is not None and len(mask) != length:
        raise ValueError(f"{name} length mismatch")
    return mask


def _actual(value, name):
    actual = np.asarray(value)
    if actual.ndim != 2 or actual.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    return actual


def _finite_scalar(value, name):
    array = np.asarray(value)
    if array.ndim != 0 or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite numeric scalar")
    try:
        number = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric scalar") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite numeric scalar")
    return number


def _finite_selected(values, name):
    try:
        selected = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite selected values") from exc
    if not len(selected) or not np.isfinite(selected).all():
        raise ValueError(f"{name} must contain nonempty finite selected values")
    return selected


def constant_means(*, inference_mask, fit_mean, calibration_actual, scale_mask):
    """Return zero/fit_mean/scale_mean on causal inference slots only.

    Calibration has shape (N, 3); only column zero on scale_mask contributes
    to scale_mean. Interval rows and all other outcomes are ignored. The
    inference calendar can have a different length from calibration. No
    validation outcome or validation scoring mask enters these forecasts.
    """
    inference = _mask(inference_mask, "inference_mask")
    fitted = _finite_scalar(fit_mean, "fit_mean")
    actual = _actual(calibration_actual, "calibration_actual")
    scale = _mask(scale_mask, "scale_mask", length=len(actual))
    selected = _finite_selected(actual[scale, 0], "scale returns")
    calibrated = math.fsum(float(v) / len(selected) for v in selected)
    if not math.isfinite(calibrated):
        raise ValueError("scale mean must be finite")
    result = {}
    for name, value in (("zero", 0.), ("fit_mean", fitted), ("scale_mean", calibrated)):
        forecast = np.full(len(inference), np.nan)
        forecast[inference] = value
        result[name] = forecast
    return result


def return_scores(actual, mu, score_mask, fit_mean):
    """Score return column zero on explicit finite rows; never change forecasts.

    Other outcomes and unscored rows are ignored. Sign accuracy follows the
    existing experiment convention, comparing return > 0 with forecast > 0.
    Rank IC is undefined for a constant forecast or a constant outcome.
    """
    actual = _actual(actual, "actual")
    forecast = np.asarray(mu)
    if forecast.ndim != 1 or len(forecast) != len(actual):
        raise ValueError("mu must be one-dimensional and aligned to actual")
    scoring = _mask(score_mask, "score_mask", length=len(actual))
    fitted = _finite_scalar(fit_mean, "fit_mean")
    observed = _finite_selected(actual[scoring, 0], "scored returns")
    predicted = _finite_selected(forecast[scoring], "scored forecasts")
    n = len(observed)
    try:
        mse = math.fsum((float(a) - float(b)) ** 2 / n for a, b in zip(observed, predicted))
        mae = math.fsum(abs(float(a) - float(b)) / n for a, b in zip(observed, predicted))
        zero_mse = math.fsum(float(a) ** 2 / n for a in observed)
        fit_mse = math.fsum((float(a) - fitted) ** 2 / n for a in observed)
    except OverflowError as exc:
        raise ValueError("return scoring produced a nonfinite loss") from exc
    if not all(math.isfinite(v) for v in (mse, mae, zero_mse, fit_mse)):
        raise ValueError("return scoring produced a nonfinite loss")
    ic = None
    if np.unique(observed).size > 1 and np.unique(predicted).size > 1:
        ranks_y, ranks_mu = rankdata(observed), rankdata(predicted)
        center = (n + 1) / 2
        numerator = math.fsum((float(a) - center) * (float(b) - center)
                              for a, b in zip(ranks_y, ranks_mu))
        denominator = math.sqrt(math.fsum((float(a) - center) ** 2 for a in ranks_y)
                                * math.fsum((float(b) - center) ** 2 for b in ranks_mu))
        ic = numerator / denominator
    return {"rows": n, "return_mse": mse, "return_mae": mae,
        "return_sign_accuracy": float(np.mean((observed > 0) == (predicted > 0))),
        "zero_return_mse": zero_mse, "fit_mean_return_mse": fit_mse,
        "return_rank_ic": ic}


__all__ = ["constant_means", "return_scores"]
