"""Fixed Technical37 direction fits using the frozen Stage17 C=1 contract.

This sibling keeps the original models immutable and changes only the selected
feature family. Canonical columns are enforced before fitting; no incomplete
row is removed from the caller's claimed fit or prediction supports.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .alpha_dd_features import FEATURE_NAMES as BASE_FEATURE_NAMES
from .oracle_frontier_features import TECHNICAL_FEATURE_NAMES
from .oracle_short_features import PRICE_FEATURE_NAMES, FLOW_FEATURE_NAMES
from .oracle_direction_fit import (
    WEIGHTINGS, LOGISTIC_PARAMETERS, STATIONARITY_GRADIENT_BOUND,
    SCALAR_LOGIT_ATOL, SCALAR_PROBABILITY_ATOL, _scalar_verification,
)
from .oracle_short_mean_fit import (
    _finite_real, _index, _index_digest, _mask, _mask_digest, _matrix_digest,
)

GROUP = "technical_short_both"
GROUPS = (GROUP,)
MODEL_IDS = tuple(GROUP + "_" + weighting for weighting in WEIGHTINGS)
FEATURE_NAMES = (BASE_FEATURE_NAMES + TECHNICAL_FEATURE_NAMES
                 + PRICE_FEATURE_NAMES + FLOW_FEATURE_NAMES)


def fit_short_direction_family(features: pd.DataFrame, outcomes, *, fit_mask, predict_mask) -> dict:
    """Fit the two fixed C=1 losses on the exact ordered short-feature37 frame.

    ``features`` is a full-N DataFrame with columns exactly ``FEATURE_NAMES``.
    ``outcomes`` is an aligned Nx3 DataFrame or ndarray. Only return column zero
    on ``fit_mask`` enters fitting, weights, priors or amplitude statistics.
    Other outcome cells and features outside fit/predict may contain arbitrary
    poison; selected cells must be finite real nonboolean numeric values.
    No row is dropped and neither mask is narrowed. Fit needs >=512 rows;
    predict is nonempty and entirely after fit in positional index order.
    Caller must separately prove event time, receipt time and label maturity.

    Returns the Stage17 keys: models/logits/probabilities keyed by MODEL_IDS,
    fit_priors/fit_weights keyed by WEIGHTINGS, fit_labels, fit_return_mean
    (numpy.mean), fit_abs_return_mean (fsum(abs(y)/n)), copied masks and JSON-safe
    provenance. Predictions are full-N with NaN outside predict. Both scalers
    fit unweighted; magnitude weights affect only the classifier. Original
    scalar objective/stationarity/predictor checks are mandatory before return.
    No later label, risk fit, calibration, retry or parameter selection is used.
    """
    if not isinstance(features, pd.DataFrame):
        raise ValueError("features must be the canonical short-feature37 DataFrame")
    if tuple(features.columns) != FEATURE_NAMES:
        raise ValueError("features must have exactly the canonical ordered 37 columns")
    groups = {GROUP: features}
    if (not isinstance(outcomes, (pd.DataFrame, np.ndarray)) or outcomes.ndim != 2
            or outcomes.shape[1] != 3 or not len(outcomes)):
        raise ValueError("nonempty outcomes DataFrame or ndarray of shape (N, 3) required")
    n = len(outcomes)
    fit, predict = (_mask(value, name, n) for value, name in (
        (fit_mask, "fit"), (predict_mask, "predict")))
    fp, pp = np.flatnonzero(fit), np.flatnonzero(predict)
    if len(fp) < 512:
        raise ValueError("fit requires at least 512 rows")
    if not len(pp) or pp[0] <= fp[-1]:
        raise ValueError("nonempty predict support must be wholly after fit")
    index = outcomes.index if isinstance(outcomes, pd.DataFrame) else None
    if index is not None:
        _index(index, "outcome")
    columns, x_fit, x_predict = {}, {}, {}
    for group in GROUPS:
        frame = groups[group]
        if not isinstance(frame, pd.DataFrame) or len(frame) != n or not frame.shape[1]:
            raise ValueError(f"{group} must be an aligned nonempty feature DataFrame")
        _index(frame.index, group)
        if index is not None and not frame.index.equals(index):
            raise ValueError("feature and outcome DataFrame indices must align exactly")
        index = frame.index
        if (not frame.columns.is_unique or
                any(not isinstance(c, str) or not c for c in frame.columns)):
            raise ValueError("feature columns must be nonempty unique string names")
        columns[group] = list(frame.columns)
        x_fit[group] = _finite_real(frame.iloc[fp].to_numpy(), group + " fit features")
        x_predict[group] = _finite_real(frame.iloc[pp].to_numpy(), group + " predict features")
    selected = outcomes.iloc[fp, 0].to_numpy() if isinstance(outcomes, pd.DataFrame) else outcomes[fp, 0]
    returns = _finite_real(selected, "selected fit returns")
    labels = (returns > 0).astype(np.int64)
    abs_mean = math.fsum(abs(float(v)) / len(fp) for v in returns)
    return_mean = float(np.mean(returns))
    if not math.isfinite(abs_mean) or abs_mean <= 0 or not math.isfinite(return_mean):
        raise ValueError("fit return mean must be finite and magnitude mean positive")
    weights = {"ordinary": np.ones(len(fp), dtype=np.float64),
               "magnitude": np.abs(returns) / abs_mean}
    priors, weight_info = {}, {}
    for weighting, w in weights.items():
        if not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("sample weights must be finite and nonnegative")
        total = math.fsum(float(v) for v in w)
        class_weight = [math.fsum(float(v) for v in w[labels == cls]) for cls in (0, 1)]
        if (not math.isfinite(total) or total <= 0 or
                any(not math.isfinite(v) or v <= 0 for v in class_weight)):
            raise ValueError("both positive-weight classes are required for every objective")
        if not math.isclose(total / len(fp), 1., rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("registered sample weights must have mean one")
        priors[weighting] = class_weight[1] / total
        weight_info[weighting] = {
            "weight_sha256": _matrix_digest(w), "sum_fsum": total,
            "sum_numpy_used_by_solver": float(np.sum(w)), "mean_fsum": total / len(fp),
            "positive_weight_by_class": class_weight,
            "zero_weight_rows": int((w == 0).sum()), "positive_prior": priors[weighting],
        }

    models, logits, probabilities, fitted_state = {}, {}, {}, {}
    with threadpool_limits(limits=2):
        for group in GROUPS:
            for weighting in WEIGHTINGS:
                mid = group + "_" + weighting
                model = make_pipeline(StandardScaler(), LogisticRegression(**LOGISTIC_PARAMETERS))
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    try:
                        model.fit(x_fit[group], labels,
                                  logisticregression__sample_weight=weights[weighting])
                    except ConvergenceWarning as exc:
                        raise ValueError(f"logistic convergence failed for {mid}; no retry permitted") from exc
                scaler, logistic = model.steps[0][1], model.steps[1][1]
                if (not np.array_equal(logistic.classes_, np.array([0, 1])) or
                        np.asarray(logistic.n_iter_).shape != (1,) or
                        np.any(logistic.n_iter_ < 0) or np.any(logistic.n_iter_ >= 1000)):
                    raise ValueError(f"invalid classes or iteration limit reached for {mid}")
                z = model.decision_function(x_predict[group])
                probability = model.predict_proba(x_predict[group])
                state = (scaler.mean_, scaler.var_, scaler.scale_, logistic.coef_,
                         logistic.intercept_, logistic.n_iter_, z, probability)
                if (not all(np.isfinite(v).all() for v in state) or
                        np.any(scaler.scale_ <= 0) or np.any(scaler.var_ < 0) or
                        np.asarray(z).shape != (len(pp),) or probability.shape != (len(pp), 2) or
                        np.any(probability < 0) or np.any(probability > 1) or
                        not np.allclose(probability.sum(axis=1), 1., rtol=0, atol=1e-15)):
                    raise ValueError(f"nonfinite or invalid fitted state/prediction for {mid}")
                verification = _scalar_verification(
                    scaler, logistic, x_fit[group], labels, weights[weighting],
                    x_predict[group], z, probability[:, 1])
                models[mid] = model
                logits[mid], probabilities[mid] = np.full(n, np.nan), np.full(n, np.nan)
                logits[mid][predict], probabilities[mid][predict] = z, probability[:, 1]
                fitted_state[mid] = {
                    "group": group, "weighting": weighting, "n_iter": logistic.n_iter_.tolist(),
                    "scalar_verification": verification,
                    "scaler_mean": scaler.mean_.tolist(), "scaler_variance": scaler.var_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(), "scaler_rows": int(scaler.n_samples_seen_),
                    "classes": logistic.classes_.tolist(), "coefficient": logistic.coef_.tolist(),
                    "intercept": logistic.intercept_.tolist(),
                    "fit_features_labels_weights_sha256": _matrix_digest(np.column_stack(
                        (x_fit[group], labels, weights[weighting]))),
                    "coefficient_sha256": _matrix_digest(logistic.coef_),
                    "intercept_sha256": _matrix_digest(logistic.intercept_),
                    "predict_logits_sha256": _matrix_digest(z),
                    "predict_probability_sha256": _matrix_digest(probability[:, 1]),
                }
    masks = {"fit": fit, "predict": predict}
    provenance = {
        "schema": "oracle-short-direction-fit-v1", "model_selection_performed": False,
        "evaluation_labels_used": False, "risk_or_calibration_fitted": False,
        "chronology_verified": "ordered positional supports only",
        "timestamp_feature_causality_and_label_completion_verified": False,
        "feature_columns": columns, "feature_counts": {g: len(c) for g, c in columns.items()},
        "canonical_feature_order_required": True, "support_narrowed": False,
        "index_sha256": _index_digest(index),
        "mask_counts": {k: int(v.sum()) for k, v in masks.items()},
        "mask_ranges": {"fit": [int(fp[0]), int(fp[-1])], "predict": [int(pp[0]), int(pp[-1])]},
        "mask_position_sha256": {k: _mask_digest(v) for k, v in masks.items()},
        "matrix_digest_format": "ndim+shape int64le prefix; C-order float64le values",
        "fit_return_sha256": _matrix_digest(returns), "fit_binary_labels_sha256": _matrix_digest(labels),
        "fit_class_counts": [int((labels == cls).sum()) for cls in (0, 1)],
        "fit_features_sha256": {g: _matrix_digest(x_fit[g]) for g in GROUPS},
        "predict_features_sha256": {g: _matrix_digest(x_predict[g]) for g in GROUPS},
        "fit_features_and_return_sha256": {g: _matrix_digest(np.column_stack((x_fit[g], returns))) for g in GROUPS},
        "sample_weights": weight_info, "fitted_state": fitted_state,
        "parameters": {"logistic": LogisticRegression(**LOGISTIC_PARAMETERS).get_params(),
            "threadpool_limit": 2, "standard_scaler": "defaults; fit unweighted",
            "magnitude_normalization": "abs(y) / math.fsum(abs(float(y_i))/n for i in fit)",
            "ordinary_weights": "ones", "binary_label": "fit_return > 0",
            "sklearn_version": sklearn.__version__},
        "objective": {
            "normalized_loss": "(sum_i w_i*logloss_i + ||coefficient||^2/(2*C))/sum_i w_i",
            "intercept_penalized": False, "solver_weight_sum": "numpy.sum(sample_weight)",
            "solver_l2_strength": "1/(C*numpy.sum(sample_weight))",
            "scipy_lbfgs_gtol": 1e-8, "scipy_lbfgs_ftol": float(64*np.finfo(float).eps),
            "scipy_lbfgs_maxls": 50,
            "independent_normalized_gradient_bound": STATIONARITY_GRADIENT_BOUND,
            "gradient_bound_reason": "100*gtol allows the separately fixed ftol stopping route; no tuning",
            "stationarity_checked_by_this_fit_helper": True,
            "finite_scalar_objective_required": True,
            "all_predict_rows_scalar_verified": True,
            "scalar_predict_logit_atol": SCALAR_LOGIT_ATOL,
            "scalar_predict_probability_atol": SCALAR_PROBABILITY_ATOL,
        },
    }
    return {"models": models, "logits": logits, "probabilities": probabilities,
            "fit_priors": priors, "fit_weights": {k: v.copy() for k, v in weights.items()},
            "fit_labels": labels.copy(), "fit_return_mean": return_mean,
            "fit_abs_return_mean": abs_mean, "masks": masks, "provenance": provenance}


__all__ = ["GROUP", "GROUPS", "FEATURE_NAMES", "WEIGHTINGS", "MODEL_IDS",
           "fit_short_direction_family"]
