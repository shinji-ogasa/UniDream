"""Small causal comparators for retrospective significance assessment.

These are new, fixed comparators, not modifications of the frozen 336-cell
experiments.  Estimators and feature standardization use the supplied Train
decisions only.  The caller supplies an already frozen variance-filter fit;
its EWMA decay may have been chosen in the original first Val month.  No
hyperparameter, label map, or decision rule is selected from Test outcomes.

``causal_simple`` contains the primary point comparator for each task and a
three-feature logistic classifier for every task.  ``train_constant`` is a
weak comparator, while ``lowdim_ridge`` exposes a secondary point comparator.
Point forecasts and classification probabilities are intentionally separate:
the class head of ``causal_simple`` is not a conversion of its point head.
"""
from __future__ import annotations

import copy
import hashlib
import warnings
from typing import Mapping

import numpy as np
from scipy.special import expit, softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .wm_ablation_evaluation import VARIANCE_FLOOR
from .wm_comparison_classical import _window_returns, ewma_variance_forecast
from .wm_full_grid_evaluation import make_task_labels
from .wm_seven_task_evaluation import TASKS


FORMAT = "wm_significance_simple_baselines_v1"
FEATURE_NAMES = ("log_past16_integrated_variance", "log_ewma_integrated_variance", "past16_log_return")
POINT_COMPARATORS = {
    "return": "zero_return",
    "variance": "frozen_ewma",
    "downside": "train_median",
    "crash": "three_feature_logistic_probability",
    "dd_excess": "train_median",
    "advantage": "three_feature_ridge",
    "recovery": "clipped_three_feature_ridge",
}


def _ids(ids, n_rows: int, *, context: int) -> np.ndarray:
    supplied = np.asarray(ids)
    if (supplied.ndim != 1 or not len(supplied)
            or not np.issubdtype(supplied.dtype, np.integer)):
        raise ValueError("decisions must be a nonempty one-dimensional integer vector")
    selected = supplied.astype(np.int64)
    if (np.any(selected < context) or np.any(selected >= n_rows)
            or np.any(np.diff(selected) <= 0)):
        raise ValueError("decisions must be unique increasing in-range endpoints with complete context")
    return selected


def _close(arrays: Mapping) -> np.ndarray:
    if "close" in arrays:
        values = np.asarray(arrays["close"], dtype=np.float64)
    else:
        raw = np.asarray(arrays["raw"])
        if raw.ndim != 2 or raw.shape[1] < 4:
            raise ValueError("raw observations must include the close in column3")
        values = np.asarray(raw[:, 3], dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("close must be a one-dimensional full-grid vector")
    return values


def causal_features(arrays: Mapping, ids, *, initial_variance: float, decay: float,
                    context: int = 256, horizon: int = 16) -> np.ndarray:
    """Observe only prices ``close[t-context:t]`` at decision ``t``.

    Both variance features are integrated over the forecast horizon.  EWMA
    filtering restarts from the explicit frozen Train initialization at the
    beginning of each context, exactly as in the original variance baseline.
    Invalid histories raise instead of changing the evaluation cohort.
    """
    if (int(context) != context or context < 17 or int(horizon) != horizon or horizon != 16):
        raise ValueError("integer context>=17 and the fixed16-bar horizon required")
    prices = _close(arrays)
    decisions = _ids(ids, len(prices), context=int(context))
    observed = _window_returns(prices, decisions, 17)
    past_variance = np.sum(np.square(observed), axis=1)
    ewma = ewma_variance_forecast(prices, decisions, initial_variance=initial_variance,
        decay=decay, context=int(context), horizon=int(horizon))
    features = np.column_stack((np.log(np.maximum(past_variance, VARIANCE_FLOOR)),
        np.log(np.maximum(ewma, VARIANCE_FLOOR)), observed.sum(axis=1)))
    if not np.isfinite(features).all():
        raise ValueError("causal baseline history is incomplete or nonfinite; no cohort rows may be removed")
    return features


def _filter_parameters(variance_filter_fit: Mapping, context: int, horizon: int) -> dict:
    if variance_filter_fit.get("format") != "seven_task_variance_filters_v1":
        raise ValueError("the frozen seven-task variance-filter fit is required")
    if (int(variance_filter_fit["context"]) != context
            or int(variance_filter_fit["horizon"]) != horizon):
        raise ValueError("frozen variance filter context/horizon differs from the requested comparator")
    initial, decay = float(variance_filter_fit["one_bar_variance"]), float(variance_filter_fit["ewma_decay"])
    if not np.isfinite([initial, decay]).all() or initial <= 0 or not 0 < decay < 1:
        raise ValueError("frozen EWMA parameters must be finite and valid")
    return {"initial_variance": initial, "decay": decay, "context": context, "horizon": horizon}


def fit_baselines(arrays: Mapping, train_ids, label_contract: Mapping,
                  variance_filter_fit: Mapping, *, context: int = 256, horizon: int = 16,
                  fold: int | None = None, target_constants: Mapping | None = None) -> dict:
    """Fit fixed C=1 logistic and alpha=1 Ridge comparators on Train only.

    ``arrays`` contains full-grid ``raw[:,3]`` (or ``close``) and each of the
    seven target arrays with shape ``[rows,1]``.  All supplied Train decisions
    are retained.  Label priors are checked against their empirical Train
    support, so a different fold's label contract cannot silently be reused.
    Hyperparameters are fixed; Val and Test targets are never read.
    """
    if int(context) != context or context < 17 or int(horizon) != horizon or horizon != 16:
        raise ValueError("integer context>=17 and the fixed16-bar horizon required")
    context, horizon = int(context), int(horizon)
    prices = _close(arrays)
    ids = _ids(train_ids, len(prices), context=context)
    parameters = _filter_parameters(variance_filter_fit, context, horizon)
    features = causal_features(arrays, ids, **parameters)
    scaler = StandardScaler().fit(features)
    x = scaler.transform(features)
    y = {}
    logistic = {}
    constants = {}
    support = {}
    numerical_warnings = []
    for task in TASKS:
        values = np.asarray(arrays[task])
        if values.shape != (len(prices), 1):
            raise ValueError("all baseline target arrays must have shape[rows,1]")
        selected = np.asarray(values[ids, 0], dtype=np.float64)
        if not np.isfinite(selected).all():
            raise ValueError(f"Train {task} contains missing or nonfinite outcomes")
        if task in ("variance", "downside", "dd_excess") and np.any(selected < 0):
            raise ValueError("variance/downside/DD-excess Train outcomes must be nonnegative")
        if task == "crash" and not np.isin(selected, (0, 1)).all():
            raise ValueError("Train crash outcomes must be binary")
        if task == "recovery" and np.any(np.abs(selected) > 1 + 1e-12):
            raise ValueError("Train recovery outcomes must lie in[-1,1]")
        entry = label_contract[task]
        classes = 2 if task == "crash" else 3
        priors = np.asarray(entry["priors"], dtype=np.float64)
        labels = make_task_labels(selected, task, label_contract)
        counts = np.bincount(labels, minlength=classes)
        if (priors.shape != (classes,) or not np.isfinite(priors).all()
                or np.any(priors <= 0) or not np.isclose(priors.sum(), 1)
                or np.any(counts == 0)
                or not np.allclose(priors, counts / len(ids), rtol=0, atol=1e-10)):
            raise ValueError("frozen label priors must match every supplied Train class support")
        estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
            tol=1e-8, fit_intercept=True, random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(x, labels)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise RuntimeError(f"fixed three-feature logistic comparator did not converge for {task}")
        if not np.isfinite(estimator.coef_).all() or not np.isfinite(estimator.intercept_).all():
            raise RuntimeError(f"three-feature logistic comparator has nonfinite parameters for {task}")
        messages = {}
        for item in caught:
            key = (item.category.__name__, str(item.message))
            messages[key] = messages.get(key, 0) + 1
        numerical_warnings.extend({"task": task, "category": category, "message": message, "count": count}
            for (category, message), count in sorted(messages.items()))
        if not np.array_equal(estimator.classes_, np.arange(classes)):
            raise ValueError("logistic comparator did not retain the registered class universe")
        logistic[task] = estimator
        y[task] = selected
        support[task] = counts.tolist()
        constants[task] = (0.0 if task == "return" else
            float(np.median(selected)) if task in ("downside", "dd_excess") else float(selected.mean()))
    ridge = Ridge(alpha=1.0, fit_intercept=True, solver="cholesky")
    with warnings.catch_warnings(record=True) as caught_ridge:
        ridge.fit(x, np.column_stack([y[task] for task in TASKS]))
    ridge_messages = {}
    for item in caught_ridge:
        key = (item.category.__name__, str(item.message))
        ridge_messages[key] = ridge_messages.get(key, 0) + 1
    numerical_warnings.extend({"task": "ridge_all", "category": category, "message": message, "count": count}
        for (category, message), count in sorted(ridge_messages.items()))
    if not np.isfinite(ridge.coef_).all() or not np.isfinite(ridge.intercept_).all():
        raise RuntimeError("three-feature Ridge comparator has nonfinite fitted parameters")
    return {"format": FORMAT, "fold": fold, "filter_parameters": parameters,
        "label_contract": copy.deepcopy(dict(label_contract)), "scaler": scaler,
        "logistic_models": logistic, "ridge_model": ridge, "train_constants": constants,
        "metadata": {"fit_split": "Train_only", "n_train": len(ids),
            "train_ids_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
            "feature_names": list(FEATURE_NAMES), "feature_mean": scaler.mean_.tolist(),
            "feature_scale": scaler.scale_.tolist(), "feature_fit_split": "Train_only",
            "history_information": "decision t observes closes only through t-1; complete original context",
            "logistic_C": 1.0, "ridge_alpha": 1.0, "hyperparameter_selection": "none; fixed before new Test comparisons",
            "class_decision_rule": "argmax(p/Train_prior)", "train_class_support": support,
            "logistic_iterations": {task: logistic[task].n_iter_.tolist() for task in TASKS},
            "optimizer_numerical_warnings": numerical_warnings,
            "optimizer_warning_policy": "capture warnings for audit; require convergence and finite final parameters; no fallback",
            "point_comparators": dict(POINT_COMPARATORS), "target_constants": copy.deepcopy(target_constants),
            "ewma_origin": "supplied frozen original Train/first-Val-month variance-filter fit",
            "ewma_original_metadata": copy.deepcopy(variance_filter_fit.get("metadata", {})),
            "test_labels_used": False}}


def _scalar_prediction(task: str, scalar: np.ndarray, entry: Mapping) -> dict:
    """Build an optional secondary point head and its fixed-label proxy."""
    values = np.asarray(scalar, dtype=np.float64)
    if task in ("variance", "downside", "dd_excess"):
        values = np.maximum(values, VARIANCE_FLOOR if task == "variance" else 0)
    if task == "crash":
        values = np.clip(values, 0, 1)
        probabilities = np.column_stack((1 - values, values))
        labels = np.argmax(probabilities / np.asarray(entry["priors"]), axis=1)
    else:
        if task == "recovery":
            values = np.clip(values, -1, 1)
        probabilities = None
        labels = make_task_labels(values, task, entry)
    return {"scalar": values, "probabilities": probabilities, "predicted_class": labels}


def predict_baselines(bundle: Mapping, arrays: Mapping, ids, fold: int | None = None) -> dict:
    """Return baseline -> task -> scalar/probabilities/predicted_class arrays.

    Predictions preserve the supplied row order and never read target arrays.
    ``causal_simple`` uses separate primary point/class comparators, as fixed
    in ``POINT_COMPARATORS``.  Constant class probabilities equal the frozen
    Train priors; their prior-adjusted class decision is the exact class0 tie.
    """
    if bundle.get("format") != FORMAT:
        raise ValueError("unrecognized significance comparator bundle")
    if fold is not None and bundle.get("fold") is not None and fold != bundle["fold"]:
        raise ValueError("baseline bundle belongs to a different fold")
    features = causal_features(arrays, ids, **bundle["filter_parameters"])
    x = bundle["scaler"].transform(features)
    # Three columns need no BLAS dispatch. This also avoids spurious Apple
    # BLAS floating-status warnings seen for otherwise finite tiny products.
    ridge_model = bundle["ridge_model"]
    ridge = np.einsum("ij,kj->ik", x, ridge_model.coef_, optimize=False) + ridge_model.intercept_
    if ridge.shape != (len(x), len(TASKS)) or not np.isfinite(ridge).all():
        raise ValueError("three-feature Ridge forecasts must be finite and complete")
    ewma = np.exp(features[:, 1])
    output = {name: {} for name in ("causal_simple", "train_constant", "lowdim_ridge")}
    for column, task in enumerate(TASKS):
        entry = bundle["label_contract"][task]
        priors = np.asarray(entry["priors"], dtype=np.float64)
        classifier = bundle["logistic_models"][task]
        logits = np.einsum("ij,kj->ik", x, classifier.coef_, optimize=False) + classifier.intercept_
        if logits.shape[1] == 1:
            positive = expit(logits[:, 0])
            probabilities = np.column_stack((1 - positive, positive))
        else:
            probabilities = softmax(logits, axis=1)
        if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-12)):
            raise ValueError("logistic comparator probabilities must be finite and normalized")
        point = (np.zeros(len(x)) if task == "return" else ewma if task == "variance" else
            np.full(len(x), bundle["train_constants"][task]) if task in ("downside", "dd_excess") else
            probabilities[:, 1] if task == "crash" else ridge[:, column])
        point = _scalar_prediction(task, point, entry)["scalar"]
        output["causal_simple"][task] = {"scalar": point, "probabilities": probabilities,
            "predicted_class": np.argmax(probabilities / priors, axis=1)}
        constant = _scalar_prediction(task, np.full(len(x), bundle["train_constants"][task]), entry)
        constant.update(probabilities=np.tile(priors, (len(x), 1)), predicted_class=np.zeros(len(x), dtype=np.int64))
        output["train_constant"][task] = constant
        output["lowdim_ridge"][task] = _scalar_prediction(task, ridge[:, column], entry)
    return output
