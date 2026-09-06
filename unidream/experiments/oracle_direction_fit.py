"""Fixed chronological direction fits; selected training returns only.

These four logistic models do not map probabilities into returns or orders.
The caller binds original feature columns, event/receipt causality, label
maturity and time purging. This helper verifies positional chronology only.
"""
from __future__ import annotations

from collections.abc import Mapping
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

from .oracle_short_mean_fit import (
    _finite_real, _index, _index_digest, _mask, _mask_digest, _matrix_digest,
)


GROUPS = ("technical", "perp_delay0")
WEIGHTINGS = ("ordinary", "magnitude")
MODEL_IDS = tuple(group + "_" + weighting for group in GROUPS for weighting in WEIGHTINGS)
LOGISTIC_PARAMETERS = {
    "C": 1., "l1_ratio": 0., "solver": "lbfgs", "tol": 1e-8,
    "max_iter": 1000, "fit_intercept": True, "random_state": 20260906,
}
STATIONARITY_GRADIENT_BOUND = 1e-6
SCALAR_LOGIT_ATOL = 1e-12
SCALAR_PROBABILITY_ATOL = 1e-14


def _scalar_verification(scaler, logistic, x_fit, labels, weights, x_predict,
                         predicted_logits, predicted_probabilities):
    """Check the selected weighted objective and all predictions without BLAS."""
    beta = [float(v) for v in logistic.coef_[0]]
    intercept = float(logistic.intercept_[0])
    center, scale = [float(v) for v in scaler.mean_], [float(v) for v in scaler.scale_]
    total = float(np.sum(weights))  # Exact denominator used by the pinned solver.
    l2 = 1 / (float(logistic.C) * total)

    def sigmoid(value):
        if value >= 0:
            return 1 / (1 + math.exp(-value))
        e = math.exp(value)
        return e / (1 + e)

    def transformed(row):
        return [(float(v)-m)/s for v, m, s in zip(row, center, scale)]

    def logit(row):
        value = intercept + math.fsum(v*b for v, b in zip(row, beta))
        if not all(math.isfinite(v) for v in row) or not math.isfinite(value):
            raise ValueError("nonfinite scalar transformation or logit")
        return value

    try:
        design = [transformed(row) for row in x_fit]
        fit_logits = [logit(row) for row in design]
        weighted_losses, residuals = [], []
        for z, label, weight in zip(fit_logits, labels, weights):
            # Stable log loss and residual retain small terms for either class.
            signed = -z if label else z
            loss = max(signed, 0.) + math.log1p(math.exp(-abs(signed)))
            residual = -sigmoid(-z) if label else sigmoid(z)
            weighted_losses.append(float(weight)/total*loss)
            residuals.append(float(weight)/total*residual)
        objective = math.fsum(weighted_losses) + .5*l2*math.fsum(v*v for v in beta)
        gradient = [math.fsum(r*row[j] for r, row in zip(residuals, design)) + l2*beta[j]
                    for j in range(len(beta))]
        gradient.append(math.fsum(residuals))  # The intercept is not penalized.
        if not math.isfinite(objective) or not all(math.isfinite(v) for v in gradient):
            raise ValueError("nonfinite scalar objective or gradient")
        gradient_max = max(abs(v) for v in gradient)
        if gradient_max > STATIONARITY_GRADIENT_BOUND:
            raise ValueError("scalar stationarity bound exceeded; no retry permitted")
        scalar_logits = [logit(transformed(row)) for row in x_predict]
        scalar_probability = [sigmoid(z) for z in scalar_logits]
        logit_difference = max(abs(a-float(b)) for a,b in zip(scalar_logits,predicted_logits))
        probability_difference = max(abs(a-float(b)) for a,b in zip(scalar_probability,predicted_probabilities))
        if (not math.isfinite(logit_difference) or not math.isfinite(probability_difference)
                or logit_difference > SCALAR_LOGIT_ATOL
                or probability_difference > SCALAR_PROBABILITY_ATOL):
            raise ValueError("scalar predictor parity failed; no retry permitted")
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("nonfinite scalar objective or predictor arithmetic") from exc
    return {"checked": True, "fit_rows": len(x_fit), "predict_rows": len(x_predict),
            "normalized_objective": objective, "normalized_gradient": gradient,
            "normalized_gradient_infinity": gradient_max,
            "max_abs_logit_difference": logit_difference,
            "max_abs_probability_difference": probability_difference,
            "solver_weight_sum": total, "l2_gradient_strength": l2,
            "gradient_bound": STATIONARITY_GRADIENT_BOUND,
            "logit_atol": SCALAR_LOGIT_ATOL, "probability_atol": SCALAR_PROBABILITY_ATOL,
            "arithmetic": "Python float and math.fsum; no BLAS"}


def fit_direction_family(groups, outcomes, *, fit_mask, predict_mask) -> dict:
    """Fit group x ordinary/magnitude StandardScaler + logistic classifiers.

    ``groups`` contains exactly technical/perp_delay0 full-N DataFrames, with
    aligned unique increasing indices and ordered, unique nonempty string
    columns. Outcomes have shape (N, 3), optionally as an aligned DataFrame.
    Only column zero on the strict boolean fit mask is inspected. Non-fit
    labels, all other outcome columns, and features outside fit/predict may
    contain arbitrary unavailable values. Claiming a selected nonfinite value
    raises; no row is removed. Fit requires >=512 rows and predictions must be
    nonempty and wholly after the final fit position.

    Labels are return > 0. Magnitude weights are abs(return) divided by the
    fsum(abs(return)/n) mean; ordinary weights are ones. Both positive-weight
    classes are required for both objectives. Each scaler is fit UNWEIGHTED;
    only the classifier receives sample weights. The sklearn 1.8 L2 API is
    l1_ratio=0, omitting deprecated penalty. ConvergenceWarning, iteration
    limit, or nonfinite fitted state/prediction fails without retry. Every model
    must also pass a scalar normalized-gradient bound of 1e-6, a finite scalar
    objective, and scalar logit/probability parity on every predict row.

    Returns models, full-N logits/probabilities (NaN outside predict), scalar
    fit_priors keyed by weighting, selected-T fit_weights and binary fit_labels,
    fit_return_mean, fit_abs_return_mean, copied masks and JSON-safe provenance.
    Probabilities index class 1 and are not calibrated on later outcomes.
    """
    if not isinstance(groups, Mapping) or set(groups) != set(GROUPS):
        raise ValueError("exactly technical and perp_delay0 feature groups are required")
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
        "schema": "oracle-direction-fit-v1", "model_selection_performed": False,
        "evaluation_labels_used": False, "risk_or_calibration_fitted": False,
        "chronology_verified": "ordered positional supports only",
        "timestamp_feature_causality_and_label_completion_verified": False,
        "feature_columns": columns, "feature_counts": {g: len(c) for g, c in columns.items()},
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


__all__ = ["GROUPS", "WEIGHTINGS", "MODEL_IDS", "fit_direction_family"]
