"""Fixed mean/risk fitting and disjoint calibration for chronological adapters.

Only explicitly selected fit and calibration outcomes are inspected. Evaluation
labels cannot change inference support; timestamp and label-completion proofs
remain the caller's responsibility. There is no scoring or policy construction.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .oracle_mean_controls import constant_means
from .oracle_mean_shrinkage import half_mean
from .oracle_risk_calibration import corrected_quantile, scale_and_bias


VARIANCE_FLOOR = 1e-12
NOMINAL_COVERAGE = .90
GROUPS = ("technical", "perp_delay0")


def _mask(value, name, length):
    mask = np.asarray(value)
    if mask.shape != (length,) or mask.dtype != np.dtype(bool):
        raise ValueError(f"{name} must be an aligned one-dimensional boolean mask")
    return mask.copy()


def _features(groups, length, selected, outcome_index):
    if not isinstance(groups, Mapping) or not set(GROUPS).issubset(groups):
        raise ValueError("technical and perp_delay0 feature groups required")
    arrays, columns, index = {}, {}, outcome_index
    for name in GROUPS:
        value = groups[name]
        if isinstance(value, pd.DataFrame):
            if value.columns.has_duplicates or not value.index.is_unique or not value.index.is_monotonic_increasing:
                raise ValueError("feature frame index and columns must be ordered and unique")
            if index is not None and not value.index.equals(index):
                raise ValueError("feature/outcome DataFrame calendars must align exactly")
            index = value.index
            columns[name] = [str(column) for column in value.columns]
        else:
            columns[name] = None
        try:
            raw_array = value.to_numpy() if isinstance(value, pd.DataFrame) else np.asarray(value)
            if np.iscomplexobj(raw_array) or (raw_array.dtype.kind == "O" and
                    any(isinstance(v, (complex, np.complexfloating)) for v in raw_array.flat)):
                raise ValueError("complex feature values are not supported")
            array = np.asarray(raw_array, float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"numeric {name} feature matrix required") from exc
        if array.ndim != 2 or array.shape[0] != length or array.shape[1] == 0:
            raise ValueError(f"aligned nonempty {name} feature matrix required")
        if not np.isfinite(array[selected]).all():
            raise ValueError(f"claimed-valid {name} fit/predict features must be finite")
        arrays[name] = array
    return arrays, columns


def fit_frozen_forecasts(groups, outcomes, *, fit_mask, scale_mask, interval_mask,
                         predict_mask, inference_mask) -> dict:
    """Fit two fixed Ridge means and one technical HGB variance forecast.

    ``groups`` contains technical/perp_delay0 full matrices or DataFrames.
    Outcomes have shape (N, 3): return, downside, and future RMS; only the first
    and third columns enter fitting/calibration. All selected outcome columns
    must be finite. Five masks are positional, strict boolean length-N arrays.

    Fit, scale, interval and inference supports must be disjoint and ordered by
    row position. Minima are 512/64/64 for fit/scale/interval. Predict is wholly
    after fit and includes scale, interval and inference; it may include other
    feature-valid rows whose outcomes are unknown. Empty inference is allowed.
    No mask is changed according to an outcome or feature value: invalid claimed
    support raises. The caller proves causal features, the six-hour UTC clock,
    label completion/purging and chronological row/calendar correspondence.

    Means and the shared variance are NaN outside inference. Raw predictions
    remain available on predict for parity/auditing. Calibration actuals are NaN
    outside scale|interval. Returned model objects use the frozen hyperparameters
    and the inherited arithmetic order; no model or weight is selected here.
    """
    y = np.asarray(outcomes)
    if y.ndim != 2 or y.shape[1] != 3 or not len(y):
        raise ValueError("nonempty outcome matrix with shape (N, 3) required")
    n = len(y)
    masks = {name: _mask(value, name, n) for name, value in (
        ("fit", fit_mask), ("scale", scale_mask), ("interval", interval_mask),
        ("predict", predict_mask), ("inference", inference_mask))}
    for name, minimum in (("fit", 512), ("scale", 64), ("interval", 64)):
        if masks[name].sum() < minimum:
            raise ValueError(f"{name} requires at least {minimum} rows")
    ordered = [np.flatnonzero(masks[name]) for name in ("fit", "scale", "interval", "inference")]
    for left, right in zip(ordered[:-1], ordered[1:]):
        if right.size and left[-1] >= right[0]:
            raise ValueError("fit, scale, interval and inference must be disjoint and ordered")
    required_predictions = masks["scale"] | masks["interval"] | masks["inference"]
    if np.any(required_predictions & ~masks["predict"]):
        raise ValueError("predict must include scale, interval and inference support")
    if np.flatnonzero(masks["predict"])[0] <= ordered[0][-1]:
        raise ValueError("predict support must be wholly after fit")
    label_support = masks["fit"] | masks["scale"] | masks["interval"]
    try:
        selected_values = y[label_support]
        if np.iscomplexobj(selected_values) or (selected_values.dtype.kind == "O" and
                any(isinstance(v, (complex, np.complexfloating)) for v in selected_values.flat)):
            raise ValueError("complex selected outcome values are not supported")
        selected_y = np.asarray(selected_values, float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("numeric selected fit/calibration outcomes required") from exc
    if not np.isfinite(selected_y).all() or np.any(selected_y[:, 2] < 0):
        raise ValueError("selected fit/calibration outcomes must be finite with nonnegative RMS")
    # Unselected labels are never converted, validated or used to gate inference.
    selected_outcomes = np.full((n, 3), np.nan)
    selected_outcomes[label_support] = selected_y
    y = selected_outcomes
    outcome_index = outcomes.index if isinstance(outcomes, pd.DataFrame) else None
    if outcome_index is not None and (not outcome_index.is_unique or not outcome_index.is_monotonic_increasing):
        raise ValueError("outcome DataFrame index must be ordered and unique")
    x, columns = _features(groups, n, masks["fit"] | masks["predict"], outcome_index)
    actual_variance = np.full(n, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        actual_variance[label_support] = np.maximum(y[label_support, 2] ** 2, VARIANCE_FLOOR)
    if not np.isfinite(actual_variance[label_support]).all():
        raise ValueError("selected outcome RMS produces nonfinite variance")
    fit, scale, interval, predict, inference = (masks[name] for name in
                                              ("fit", "scale", "interval", "predict", "inference"))
    fit_mean = float(y[fit, 0].mean())
    models = {name + "_mean": make_pipeline(StandardScaler(), Ridge(alpha=100.)) for name in GROUPS}
    models["technical_variance"] = HistGradientBoostingRegressor(
        max_iter=100, max_leaf_nodes=7, min_samples_leaf=64, learning_rate=.04,
        l2_regularization=10., early_stopping=False, random_state=20260905)
    raw = {name: {"mu": np.full(n, np.nan)} for name in GROUPS}
    raw["technical"]["log_variance"] = np.full(n, np.nan)
    with threadpool_limits(limits=2):
        models["technical_mean"].fit(x["technical"][fit], y[fit, 0])
        models["technical_variance"].fit(x["technical"][fit], np.log(actual_variance[fit]))
        raw["technical"]["mu"][predict] = models["technical_mean"].predict(x["technical"][predict])
        raw["technical"]["log_variance"][predict] = models["technical_variance"].predict(x["technical"][predict])
        models["perp_delay0_mean"].fit(x["perp_delay0"][fit], y[fit, 0])
        raw["perp_delay0"]["mu"][predict] = models["perp_delay0_mean"].predict(x["perp_delay0"][predict])
    if not all(np.isfinite(array[predict]).all() for group in raw.values() for array in group.values()):
        raise ValueError("nonfinite model prediction on claimed predict support")
    logvar = raw["technical"]["log_variance"]
    raw_variance = np.exp(np.clip(logvar, np.log(VARIANCE_FLOOR), 0))
    raw["technical"]["variance"] = raw_variance
    technical_bias, multiplier = scale_and_bias(
        y[scale, 0], actual_variance[scale], raw["technical"]["mu"][scale], raw_variance[scale])
    perp_bias = float(np.mean(y[scale, 0] - raw["perp_delay0"]["mu"][scale]))
    biases = {"technical": technical_bias, "perp_delay0": perp_bias}
    if not np.isfinite([fit_mean, technical_bias, perp_bias, multiplier]).all() or multiplier <= 0:
        raise ValueError("nonfinite mean or invalid variance calibration")
    scaled = {}
    for name in GROUPS:
        scaled[name] = raw[name]["mu"].copy()
        scaled[name] += biases[name]
    scaled_variance = raw_variance.copy()
    scaled_variance *= multiplier
    scaled_variance = np.maximum(scaled_variance, VARIANCE_FLOOR)
    if not np.isfinite(scaled_variance[predict]).all() or not all(np.isfinite(v[predict]).all() for v in scaled.values()):
        raise ValueError("nonfinite scaled forecast")
    quantiles = {}
    for version, mean, variance in (("raw", raw["technical"]["mu"], np.maximum(raw_variance, VARIANCE_FLOOR)),
                                     ("scaled", scaled["technical"], scaled_variance)):
        quantiles[version] = {
            "return_quantile": corrected_quantile(np.abs(y[interval, 0] - mean[interval]) / np.sqrt(variance[interval]), NOMINAL_COVERAGE),
            "volatility_quantile": corrected_quantile(np.abs(.5 * np.log(actual_variance[interval] / variance[interval])), NOMINAL_COVERAGE)}
    calibration_actual = np.full((n, 3), np.nan)
    calibration_actual[scale | interval] = y[scale | interval]
    constants = constant_means(inference_mask=inference, fit_mean=fit_mean,
                               calibration_actual=calibration_actual, scale_mask=scale)
    anchor = constants["scale_mean"]
    anchor_value = float(anchor[inference][0]) if inference.any() else float(constant_means(
        inference_mask=np.ones(1, bool), fit_mean=fit_mean,
        calibration_actual=calibration_actual, scale_mask=scale)["scale_mean"][0])
    means = {"scale_mean": anchor}
    for name in GROUPS:
        means[name + "_scaled"] = np.where(inference, scaled[name], np.nan)
        half_name = "technical_half" if name == "technical" else "perp_delay0_half"
        means[half_name] = half_mean(means[name + "_scaled"], anchor, inference_mask=inference) if inference.any() else np.full(n, np.nan)
    provenance = {
        "schema": "oracle-frozen-forecasts-v1", "model_selection_performed": False,
        "evaluation_labels_used": False, "chronology_verified": "ordered positional supports only",
        "timestamp_feature_causality_and_label_completion_verified": False,
        "feature_columns": columns, "feature_counts": {name: x[name].shape[1] for name in GROUPS},
        "mask_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "mask_ranges": {name: [int(np.flatnonzero(mask)[0]), int(np.flatnonzero(mask)[-1])] if mask.any() else None for name, mask in masks.items()},
        "mask_position_sha256": {name: hashlib.sha256(np.asarray([n], "<i8").tobytes() + mask.astype("u1").tobytes()).hexdigest() for name, mask in masks.items()},
        "parameters": {"return_ridge_alpha": 100., "variance_floor": VARIANCE_FLOOR,
            "nominal_coverage": NOMINAL_COVERAGE, "half_weight": .5, "threadpool_limit": 2,
            "technical_variance": {"max_iter": 100, "max_leaf_nodes": 7, "min_samples_leaf": 64,
                "learning_rate": .04, "l2_regularization": 10., "early_stopping": False, "random_state": 20260905}},
        "raw_log_variance_clip_count": int(np.sum((logvar[predict] < np.log(VARIANCE_FLOOR)) | (logvar[predict] > 0))),
    }
    return {"means": means, "variance": np.where(inference, scaled_variance, np.nan),
        "models": models, "raw_predictions": raw, "masks": masks, "provenance": provenance,
        "calibration": {"return_bias": biases, "variance_multiplier": multiplier,
            "fit_mean": fit_mean, "scale_mean": anchor_value,
            "technical_quantiles": quantiles,
            "counts": {name: int(masks[name].sum()) for name in ("fit", "scale", "interval")}},
        "calibration_arrays": {"actual": calibration_actual,
            "actual_variance": np.where(scale | interval, actual_variance, np.nan)}}


__all__ = ["fit_frozen_forecasts"]
