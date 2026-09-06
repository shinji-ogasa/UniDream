"""Pure chronological raw-mean fitting for a fixed short-feature family.

Only selected fit returns and selected fit/predict features are inspected. The
caller proves timestamp availability, feature causality, label maturation and
purging. This helper neither changes support nor fits risk or calibration.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


GROUPS = (
    "technical", "technical_short_price", "technical_short_flow",
    "technical_short_both",
)


def _mask(value, name, n):
    result = np.asarray(value)
    if result.shape != (n,) or result.dtype != np.dtype(bool):
        raise ValueError(f"{name} must be an aligned one-dimensional boolean mask")
    return result.copy()


def _index(index, name):
    if (isinstance(index, pd.MultiIndex) or index.hasnans or
            not index.is_unique or not index.is_monotonic_increasing):
        raise ValueError(f"{name} index must be non-null, unique and increasing")


def _finite_real(selected, name):
    """Convert selected cells only, rejecting implicit bool/string/complex casts."""
    array = np.asarray(selected)
    if array.dtype.kind == "O":
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
               for value in array.flat):
            raise ValueError(f"{name} must contain real numeric values, not bool or complex")
    elif array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values, not bool or complex")
    try:
        result = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real numeric values") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real numeric values")
    return np.ascontiguousarray(result)


def _matrix_digest(array):
    """Shape-prefixed C-order little-endian float64 digest of selected values."""
    array = np.asarray(array, dtype="<f8", order="C")
    header = np.asarray([array.ndim, *array.shape], dtype="<i8").tobytes()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _mask_digest(mask):
    return hashlib.sha256(np.asarray([len(mask)], dtype="<i8").tobytes() +
                          mask.astype("u1").tobytes()).hexdigest()


def _index_digest(index):
    header = json.dumps({"type": type(index).__name__, "dtype": str(index.dtype),
                         "length": len(index)}, sort_keys=True).encode("utf-8")
    values = pd.util.hash_pandas_object(index, index=False).to_numpy(dtype="<u8")
    return hashlib.sha256(header + b"\n" + values.tobytes()).hexdigest()


def fit_raw_mean_family(groups, outcomes, *, fit_mask, predict_mask) -> dict:
    """Fit exactly four StandardScaler + Ridge(alpha=100) raw return means.

    ``groups`` must have exactly ``GROUPS`` keys, each holding a full-N DataFrame
    with the same non-null, unique, increasing index and nonempty unique string
    column names. Column order is retained, not sorted. Outcomes are an N-by-3
    DataFrame or ndarray; only ``outcomes[fit_mask, 0]`` is read/converted. An
    outcomes DataFrame must have the identical index. Outcome columns 1/2 and
    all non-fit outcome rows may contain arbitrary unavailable values.

    Masks are strict boolean length-N vectors. Fit has at least 512 rows,
    predict is nonempty, and every prediction position follows every fit row.
    Every selected feature cell must be finite and real; values on neither
    support are ignored. Invalid claimed support raises, never drops a row.
    Positional chronology does not prove a time frequency or label maturity.

    Returns ``models[group]``, ``raw[group]`` (full-N float64 mu, NaN outside
    predict), ``fit_return_mean`` using the inherited numpy mean arithmetic,
    copied ``masks`` and JSON-compatible ``provenance``. All fits and predictions
    use threadpool limit 2. No bias, weights, risk model or selection is fitted.
    """
    if not isinstance(groups, Mapping) or set(groups) != set(GROUPS):
        raise ValueError("exactly the four registered short-mean feature groups are required")
    if not isinstance(outcomes, (pd.DataFrame, np.ndarray)):
        raise ValueError("outcomes must be a DataFrame or ndarray of shape (N, 3)")
    if outcomes.ndim != 2 or outcomes.shape[1] != 3 or not len(outcomes):
        raise ValueError("nonempty outcomes with shape (N, 3) required")
    n = len(outcomes)
    fit, predict = (_mask(value, name, n) for value, name in (
        (fit_mask, "fit"), (predict_mask, "predict")))
    fit_positions, predict_positions = np.flatnonzero(fit), np.flatnonzero(predict)
    if len(fit_positions) < 512:
        raise ValueError("fit requires at least 512 rows")
    if not len(predict_positions):
        raise ValueError("predict requires at least one row")
    if predict_positions[0] <= fit_positions[-1]:
        raise ValueError("fit and predict must be disjoint with predict wholly after fit")

    index = outcomes.index if isinstance(outcomes, pd.DataFrame) else None
    if index is not None:
        _index(index, "outcome")
    columns, x_fit, x_predict = {}, {}, {}
    for name in GROUPS:
        frame = groups[name]
        if not isinstance(frame, pd.DataFrame) or len(frame) != n or not frame.shape[1]:
            raise ValueError(f"{name} must be an aligned nonempty feature DataFrame")
        _index(frame.index, name)
        if index is not None and not frame.index.equals(index):
            raise ValueError("feature and outcome DataFrame indices must align exactly")
        index = frame.index
        if (not frame.columns.is_unique or
                any(not isinstance(column, str) or not column for column in frame.columns)):
            raise ValueError("feature columns must be nonempty unique string names")
        columns[name] = list(frame.columns)
        # Select before conversion: arbitrary unavailable cells cannot alter the
        # model, support or provenance merely through an unselected row.
        x_fit[name] = _finite_real(frame.iloc[fit_positions].to_numpy(), f"{name} fit features")
        x_predict[name] = _finite_real(frame.iloc[predict_positions].to_numpy(),
                                       f"{name} predict features")
    selected_return = (outcomes.iloc[fit_positions, 0].to_numpy()
                       if isinstance(outcomes, pd.DataFrame) else outcomes[fit_positions, 0])
    y_fit = _finite_real(selected_return, "selected fit returns")
    fit_return_mean = float(y_fit.mean())
    if not np.isfinite(fit_return_mean):
        raise ValueError("selected fit-return mean must be finite")

    models = {name: make_pipeline(StandardScaler(), Ridge(alpha=100.)) for name in GROUPS}
    raw = {name: np.full(n, np.nan, dtype=np.float64) for name in GROUPS}
    with threadpool_limits(limits=2):
        for name in GROUPS:
            models[name].fit(x_fit[name], y_fit)
            raw[name][predict] = models[name].predict(x_predict[name])
            scaler, ridge = models[name].steps[0][1], models[name].steps[1][1]
            if not all(np.isfinite(value).all() for value in (
                    scaler.mean_, scaler.var_, scaler.scale_, ridge.coef_,
                    np.asarray(ridge.intercept_), raw[name][predict])):
                raise ValueError(f"nonfinite fitted parameters or prediction for {name}")

    masks = {"fit": fit, "predict": predict}
    provenance = {
        "schema": "oracle-short-mean-fit-v1", "model_selection_performed": False,
        "evaluation_labels_used": False, "risk_or_calibration_fitted": False,
        "chronology_verified": "ordered positional supports only",
        "timestamp_feature_causality_and_label_completion_verified": False,
        "feature_columns": columns,
        "feature_counts": {name: len(value) for name, value in columns.items()},
        "index_sha256": _index_digest(index),
        "mask_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "mask_ranges": {"fit": [int(fit_positions[0]), int(fit_positions[-1])],
                        "predict": [int(predict_positions[0]), int(predict_positions[-1])]},
        "mask_position_sha256": {name: _mask_digest(mask) for name, mask in masks.items()},
        "matrix_digest_format": "ndim+shape int64le prefix; C-order float64le values",
        "fit_return_sha256": _matrix_digest(y_fit),
        "fit_features_sha256": {name: _matrix_digest(x_fit[name]) for name in GROUPS},
        "predict_features_sha256": {name: _matrix_digest(x_predict[name]) for name in GROUPS},
        "fit_features_and_return_sha256": {
            name: _matrix_digest(np.column_stack((x_fit[name], y_fit))) for name in GROUPS},
        "parameters": {"return_ridge_alpha": 100., "threadpool_limit": 2,
                       "standard_scaler": "defaults", "ridge_other_parameters": "defaults"},
    }
    return {"models": models, "raw": raw, "fit_return_mean": fit_return_mean,
            "masks": masks, "provenance": provenance}


__all__ = ["GROUPS", "fit_raw_mean_family"]
