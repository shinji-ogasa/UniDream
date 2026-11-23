"""Leakage-aware scoring/calibration for scalar versus categorical returns.

Labels use fixed Train-return RMS thresholds, not realized or predicted
integrated variance.  Calibration callers must supply ONLY the middle Val
month, after first-month model selection; these pure functions open no data.
The primary probability decision is p/Train prior, raw argmax is auxiliary.
Thresholding a scalar forecast is a decision, not a probability forecast, and
therefore receives neither log loss nor a Brier score.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .wm_classification_evaluation import (
    _decision_scores,
    _probabilities,
    make_labels,
    score_classification,
)


CLASSES = (0, 1, 2)
PROBABILITY_FLOOR = 1e-12
TEMPERATURE_BOUNDS = (0.1, 10.0)


def _vector(value: Any, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional vector")
    return value


def _class_vector(value: Any, name: str = "labels") -> np.ndarray:
    value = _vector(value, name)
    if not np.isin(value, CLASSES).all():
        raise ValueError(f"{name} must contain only integer classes 0/1/2")
    return value.astype(np.int64)


def make_return_labels(returns: np.ndarray, theta: float | np.ndarray) -> np.ndarray:
    """0=down, 1=flat, 2=up; equality at either positive threshold is flat."""
    return make_labels(_vector(returns, "returns"), theta)


def return_class_priors(labels: np.ndarray) -> np.ndarray:
    """Estimate unsmoothed Train priors; all three Train classes are required."""
    labels = _class_vector(labels)
    counts = np.bincount(labels, minlength=3)
    if not (counts > 0).all():
        raise ValueError("all three classes must occur in Train to define p/Train-prior decisions")
    return counts / counts.sum()


def _normalize_positive(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.maximum(np.asarray(probabilities, dtype=float), PROBABILITY_FLOOR)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], PROBABILITY_FLOOR)).mean())


def fit_return_calibrator(
    actual_returns: np.ndarray,
    scalar_predictions: np.ndarray,
    theta: float,
    train_rms: float,
) -> dict[str, Any]:
    """Fixed multinomial logistic P(class | scalar_prediction / Train_RMS).

    Inputs must be frozen-model forecasts and outcomes from ONLY the middle
    Val month.  C=1 is fixed; there is no hyperparameter or Test selection.
    The JSON-compatible coefficients replay sklearn's multiclass or binary
    probabilities, retaining all three output columns if a Val class is absent.
    A single observed Val class uses a constant empirical forecast.  Missing
    class columns receive only the predeclared numerical floor, not fitted
    probabilities from another split.
    """
    actual_returns = _vector(actual_returns, "actual_returns")
    scalar_predictions = _vector(scalar_predictions, "scalar_predictions")
    if actual_returns.shape != scalar_predictions.shape:
        raise ValueError("calibration actual returns and scalar predictions must match")
    if not np.isfinite(train_rms) or train_rms <= 0:
        raise ValueError("train_rms must be finite and positive")
    if np.ndim(theta) != 0:
        raise ValueError("each fold requires one fixed scalar Train threshold")
    labels = make_return_labels(actual_returns, theta)
    normalized = scalar_predictions / float(train_rms)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized scalar calibration inputs must remain finite")
    classes = np.unique(labels)
    fit: dict[str, Any] = {"kind": "scalar_multinomial_logistic", "classes": classes.tolist(),
        "train_rms": float(train_rms), "theta": float(theta), "n_fit": len(labels),
        "fit_label_counts": np.bincount(labels, minlength=3).tolist(), "regularization_C": 1.0,
        "probability_floor": PROBABILITY_FLOOR, "fit_input": "frozen scalar prediction divided by Train RMS",
        "fit_scope_contract": "caller supplies only middle validation month; no Test access or fitting in this module"}
    if len(classes) == 1:
        fit.update({"mode": "constant", "constant_probabilities": (np.bincount(labels, minlength=3) / len(labels)).tolist(),
                    "coef": [], "intercept": [], "n_iter": 0})
    else:
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=4000, tol=1e-8, random_state=20261110)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(normalized[:, None], labels)
        if any(issubclass(warning.category, ConvergenceWarning) for warning in caught):
            raise RuntimeError("scalar logistic calibration did not converge")
        fit.update({"mode": "binary" if len(classes) == 2 else "multinomial", "classes": model.classes_.tolist(),
                    "coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(), "n_iter": int(max(model.n_iter_))})
    fit["fit_log_loss"] = _log_loss(labels, calibrated_probs(scalar_predictions, fit))
    return fit


def fit_probability_calibrator(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    """Fit one CE-optimal positive temperature on ONLY the middle Val month.

    Uses log-probabilities / T with fixed T in [0.1,10].  Positive temperature
    preserves raw class ordering; prior-adjusted decisions can change.  The
    identity and both endpoints are included, so the recorded Val CE cannot
    worsen relative to T=1 merely because of optimizer tolerance.
    """
    labels = _class_vector(labels)
    probs = _probabilities(probs)
    if len(labels) != len(probs):
        raise ValueError("calibration labels and probability rows must match")
    log_probabilities = np.log(np.maximum(probs, PROBABILITY_FLOOR))

    def objective(log_temperature):
        logits = log_probabilities / np.exp(log_temperature)
        losses = logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]
        return float(losses.mean())

    minimum = minimize_scalar(objective, bounds=np.log(TEMPERATURE_BOUNDS), method="bounded",
                              options={"xatol": 1e-10})
    if not minimum.success or not np.isfinite(minimum.fun):
        raise RuntimeError("temperature calibration optimization failed")
    candidates = [0.0, float(minimum.x), *np.log(TEMPERATURE_BOUNDS)]
    best = min(candidates, key=lambda value: (objective(value), abs(value)))
    fit = {"kind": "temperature", "temperature": float(np.clip(np.exp(best), *TEMPERATURE_BOUNDS)), "bounds": list(TEMPERATURE_BOUNDS),
        "n_fit": len(labels), "fit_label_counts": np.bincount(labels, minlength=3).tolist(),
        "fit_log_loss_before": _log_loss(labels, probs), "fit_log_loss_after": objective(best),
        "probability_floor": PROBABILITY_FLOOR, "optimization_success": True,
        "fit_scope_contract": "caller supplies only middle validation month; no Test access or fitting in this module"}
    return fit


def calibrated_probs(scalar_or_probs: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    """Apply frozen JSON calibration parameters without inspecting outcomes."""
    if fit["kind"] == "temperature":
        probs = _probabilities(scalar_or_probs)
        temperature = float(fit["temperature"])
        if not np.isfinite(temperature) or not TEMPERATURE_BOUNDS[0] <= temperature <= TEMPERATURE_BOUNDS[1]:
            raise ValueError("saved temperature is outside its positive registered bounds")
        logits = np.log(np.maximum(probs, PROBABILITY_FLOOR)) / temperature
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    if fit["kind"] != "scalar_multinomial_logistic":
        raise ValueError("unknown calibration kind")
    scalar = _vector(scalar_or_probs, "scalar_predictions")
    rms = float(fit["train_rms"])
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("saved Train RMS must be positive and finite")
    if fit["mode"] == "constant":
        constants = np.asarray(fit["constant_probabilities"], float)
        if constants.shape != (3,) or not np.isfinite(constants).all() or np.any(constants < 0) or not np.isclose(constants.sum(), 1):
            raise ValueError("saved constant calibration probabilities are invalid")
        return _normalize_positive(np.tile(constants, (len(scalar), 1)))
    coefficients, intercept = np.asarray(fit["coef"], float), np.asarray(fit["intercept"], float)
    classes = np.asarray(fit["classes"])
    if classes.ndim != 1 or len(np.unique(classes)) != len(classes) or not np.isin(classes, CLASSES).all():
        raise ValueError("saved logistic classes must be unique members of 0/1/2")
    normalized = scalar / rms
    if not np.isfinite(normalized).all() or not np.isfinite(coefficients).all() or not np.isfinite(intercept).all():
        raise ValueError("saved calibration inputs and coefficients must be finite")
    probabilities = np.zeros((len(scalar), 3), float)
    if fit["mode"] == "binary" and coefficients.shape == (1, 1) and intercept.shape == (1,) and len(classes) == 2:
        positive = expit(normalized * coefficients[0, 0] + intercept[0])
        probabilities[:, int(classes[0])] = 1 - positive
        probabilities[:, int(classes[1])] = positive
    elif fit["mode"] == "multinomial" and coefficients.shape == (3, 1) and intercept.shape == (3,) and len(classes) == 3:
        logits = normalized[:, None] * coefficients[:, 0][None, :] + intercept[None, :]
        probabilities[:, classes.astype(int)] = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    else:
        raise ValueError("saved logistic coefficient shape does not match its mode")
    return _normalize_positive(probabilities)


def score_scalar_labels(actual_labels: np.ndarray, predicted_labels: np.ndarray) -> dict[str, Any]:
    """Bare classification decisions; deliberately no probability loss scores."""
    actual = _class_vector(actual_labels, "actual_labels")
    predicted = _class_vector(predicted_labels, "predicted_labels")
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted class vectors must match")
    return {"n": len(actual), **_decision_scores(actual, predicted), "probability_metrics_available": False}


def score_return_formulation(actual_returns: np.ndarray, scalar_predictions: np.ndarray, theta: float | np.ndarray,
                             priors: np.ndarray, return_calibrator: dict | None = None) -> dict[str, Any]:
    """Report raw scalar threshold decisions and optional Val-fitted probabilities."""
    actual_returns = _vector(actual_returns, "actual_returns")
    scalar_predictions = _vector(scalar_predictions, "scalar_predictions")
    if actual_returns.shape != scalar_predictions.shape:
        raise ValueError("actual returns and scalar forecasts must match")
    actual = make_return_labels(actual_returns, theta)
    labels = make_return_labels(scalar_predictions, theta)
    result = {"scalar_projection": score_scalar_labels(actual, labels)}
    if return_calibrator is not None:
        result["calibrated"] = score_classification(actual, calibrated_probs(scalar_predictions, return_calibrator), priors)
    return result


def score_probability_formulation(labels: np.ndarray, probs: np.ndarray, priors: np.ndarray,
                                  calibrator: dict | None = None) -> dict[str, Any]:
    """Primary p/Train-prior and auxiliary raw decisions for both probability sets."""
    labels = _class_vector(labels)
    result = {"raw_probabilities": score_classification(labels, probs, priors)}
    if calibrator is not None:
        result["calibrated_probabilities"] = score_classification(labels, calibrated_probs(probs, calibrator), priors)
    return result


def compare_matched_formulations(actual_returns: np.ndarray, regression_predictions: np.ndarray,
                                  classification_probs: np.ndarray, theta: float | np.ndarray,
                                  train_priors: np.ndarray, return_calibrator: dict,
                                  probability_calibrator: dict | None = None) -> dict[str, Any]:
    """Same-row scalar projection/calibration versus categorical probabilities.

    No finite filtering is allowed: all methods must have the identical cohort.
    There is no assumption that integrated realized variance is a Gaussian
    variance for the terminal return, and no probability is synthesized from it.
    """
    actual_returns = _vector(actual_returns, "actual_returns")
    probabilities = _probabilities(classification_probs)
    if len(probabilities) != len(actual_returns):
        raise ValueError("both formulations must share identical row coverage")
    return {
        "regression": score_return_formulation(actual_returns, regression_predictions, theta, train_priors, return_calibrator),
        "classification": score_probability_formulation(make_return_labels(actual_returns, theta), probabilities,
                                                          train_priors, probability_calibrator),
        "n_matched": len(actual_returns), "primary_decision": "argmax(probabilities / Train class priors)",
        "neural_seed_count": 1, "seed_stability_claim": False,
    }
