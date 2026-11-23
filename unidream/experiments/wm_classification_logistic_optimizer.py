"""Full-dimensional coordinate preconditioning for the SAME L2 logistic loss.

This proposed optimizer is isolated from the registered experiment.  Covariance
eigenvalues only define an invertible change of coordinates; every eigenvector
is retained, and no covariance ridge is added to the statistical objective.
Final convergence is certified by the gradient in ORIGINAL feature coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import logsumexp

from .wm_classification_classical import validation_scores, validate_priors


def _array_sha(values):
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _design(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("design must be a nonempty finite matrix")
    return values


def _labels(values, n):
    values = np.asarray(values)
    if values.shape != (n,) or not np.isin(values, (0, 1, 2)).all():
        raise ValueError("labels must contain one class 0/1/2 per Train row")
    if not np.array_equal(np.unique(values), (0, 1, 2)):
        raise ValueError("Train must contain all three classes")
    return values.astype(np.int64, copy=False)


def _regularization(c, n):
    if not np.isfinite(c) or c <= 0:
        raise ValueError("C must be finite and positive")
    value = 1. / (float(c) * n)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("regularization strength must be finite and positive")
    return value


def original_objective_gradient(x, labels, coefficients, intercept, c):
    """Mean NLL + ||W||²/(2*C*N), including the unpenalized intercept gradient."""
    x = _design(x)
    labels = _labels(labels, len(x))
    coefficients = np.asarray(coefficients, dtype=np.float64)
    intercept = np.asarray(intercept, dtype=np.float64)
    if coefficients.shape != (3, x.shape[1]) or intercept.shape != (3,):
        raise ValueError("coefficient/intercept shapes must be [3,p] and [3]")
    if not np.isfinite(coefficients).all() or not np.isfinite(intercept).all():
        raise ValueError("coefficients and intercept must be finite")
    regularization = _regularization(c, len(x))
    raw = x @ coefficients.T + intercept
    log_probability = raw - logsumexp(raw, axis=1, keepdims=True)
    objective = -log_probability[np.arange(len(x)), labels].mean()
    objective += .5 * regularization * np.square(coefficients).sum()
    residual = np.exp(log_probability)
    residual[np.arange(len(x)), labels] -= 1.
    residual /= len(x)
    grad_coefficients = residual.T @ x + regularization * coefficients
    grad_intercept = residual.sum(0)
    if not np.isfinite(objective) or not np.isfinite(grad_coefficients).all() or not np.isfinite(grad_intercept).all():
        raise FloatingPointError("nonfinite original logistic objective/gradient")
    return float(objective), grad_coefficients, grad_intercept


@dataclass
class LogisticPreconditioner:
    x: np.ndarray
    mean: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    rotated: np.ndarray
    metadata: dict


def prepare_logistic_preconditioner(x):
    """Fit a full Train covariance basis, reused for every C candidate."""
    started = time.monotonic()
    x = _design(x)
    mean = x.mean(0)
    centered = x - mean
    begun = time.monotonic()
    covariance = centered.T @ centered / len(x)
    covariance = np.asfortranarray(.5 * (covariance + covariance.T))
    if not np.isfinite(covariance).all():
        raise FloatingPointError("nonfinite Train covariance")
    gram_seconds = time.monotonic() - begun
    begun = time.monotonic()
    values, vectors = eigh(covariance, overwrite_a=True, check_finite=False, driver="evd")
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise FloatingPointError("nonfinite covariance eigendecomposition")
    tolerance = 64. * np.finfo(np.float64).eps * x.shape[1] * max(1., float(np.max(np.abs(values))))
    minimum = float(values.min())
    if minimum < -tolerance:
        raise FloatingPointError("Train covariance has materially negative eigenvalues")
    clipped = np.maximum(values, 0.)
    eig_seconds = time.monotonic() - begun
    begun = time.monotonic()
    rotated = centered @ vectors
    if not np.isfinite(rotated).all():
        raise FloatingPointError("nonfinite full-dimensional rotated design")
    rotation_seconds = time.monotonic() - begun
    metadata = {
        "n_train": len(x), "original_dimension": x.shape[1], "retained_dimension": x.shape[1],
        "eigenvalue_min_before_rounding": minimum, "eigenvalue_max": float(values.max()),
        "negative_eigenvalues_rounded": int(np.sum(values < 0)), "eigenvalue_rounding_tolerance": tolerance,
        "mean_sha256": _array_sha(mean), "rounded_eigenvalues_sha256": _array_sha(clipped),
        "eigenvectors_sha256": _array_sha(vectors), "gram_seconds": gram_seconds,
        "eigendecomposition_seconds": eig_seconds, "rotation_seconds": rotation_seconds,
        "preparation_seconds": time.monotonic() - started,
        "added_statistical_ridge": 0., "preconditioner_scale_rule": "sqrt(max(eigenvalue,0)+5/(C*N))",
    }
    metadata["metadata_sha256"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return LogisticPreconditioner(x, mean, clipped, vectors, rotated, metadata)


@dataclass
class PreconditionedLogisticClassifier:
    coef_: np.ndarray
    intercept_: np.ndarray
    C: float
    n_iter_: np.ndarray
    max_iter: int
    tol: float
    classes_: np.ndarray
    class_weight: None = None
    solver: str = "full_covariance_preconditioned_lbfgs"

    @property
    def n_features_in_(self):
        return self.coef_.shape[1]

    def decision_function(self, x):
        x = _design(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("prediction design has a different original feature dimension")
        return x @ self.coef_.T + self.intercept_

    def predict_proba(self, x):
        raw = self.decision_function(x)
        return np.exp(raw - logsumexp(raw, axis=1, keepdims=True))

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(1)]


def transformed_objective_gradient(parameters, z, labels, scale, regularization):
    """Unconstrained all-three-class objective in invertible coordinates."""
    dimension = z.shape[1]
    packed = np.asarray(parameters).reshape(3, dimension + 1)
    coefficients, intercept = packed[:, :-1], packed[:, -1]
    raw = z @ coefficients.T + intercept
    log_probability = raw - logsumexp(raw, axis=1, keepdims=True)
    objective = -log_probability[np.arange(len(z)), labels].mean()
    objective += .5 * regularization * np.square(coefficients / scale).sum()
    residual = np.exp(log_probability)
    residual[np.arange(len(z)), labels] -= 1.
    residual /= len(z)
    gradient = np.empty_like(packed)
    gradient[:, :-1] = residual.T @ z + regularization * coefficients / np.square(scale)
    gradient[:, -1] = residual.sum(0)
    if not np.isfinite(objective) or not np.isfinite(gradient).all():
        raise FloatingPointError("nonfinite transformed logistic objective/gradient")
    return float(objective), gradient.ravel()


def fit_preconditioned_logistic(prepared, labels, c, *, tol=1e-4, max_iter=1000):
    """Fit one C and certify the original-space weight AND bias gradient."""
    labels = _labels(labels, len(prepared.x))
    if not np.isfinite(tol) or tol <= 0 or max_iter < 1:
        raise ValueError("tol and max_iter must be positive")
    regularization = _regularization(c, len(labels))
    scale = np.sqrt(prepared.eigenvalues + 5. * regularization)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise FloatingPointError("preconditioner scale is not invertible")
    z = prepared.rotated / scale
    # g_W = (g_B * s) V.T + g_bias * mean. Include the intercept's
    # contribution to original weight gradients for nonzero Train means.
    bound = float(np.max(np.abs(prepared.eigenvectors * scale[None, :]).sum(1)))
    bound += float(np.max(np.abs(prepared.mean)))
    transformed_tolerance = float(tol) / (4. * max(1., bound))
    initial = np.zeros(3 * (z.shape[1] + 1), dtype=np.float64)
    begun = time.monotonic()
    result = minimize(
        transformed_objective_gradient, initial,
        args=(z, labels, scale, regularization), method="L-BFGS-B", jac=True,
        options={"maxiter": int(max_iter), "gtol": transformed_tolerance,
                 "ftol": 1e-15, "maxls": 50, "maxcor": 20},
    )
    packed = result.x.reshape(3, z.shape[1] + 1)
    coefficients = (packed[:, :-1] / scale) @ prepared.eigenvectors.T
    intercept = packed[:, -1] - coefficients @ prepared.mean
    objective, gradient, bias_gradient = original_objective_gradient(prepared.x, labels, coefficients, intercept, c)
    maximum_gradient = max(float(np.abs(gradient).max()), float(np.abs(bias_gradient).max()))
    diagnostics = {
        "C": float(c), "lambda": regularization, "n_iter": [int(result.nit)],
        "function_evaluations": int(result.nfev), "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message), "objective": objective,
        "original_weight_gradient_linf": float(np.abs(gradient).max()),
        "original_bias_gradient_linf": float(np.abs(bias_gradient).max()),
        "original_gradient_linf": maximum_gradient, "original_gradient_all_finite": True,
        "converged_original_gradient": bool(maximum_gradient <= tol), "requested_original_tolerance": float(tol),
        "transformed_gradient_tolerance": transformed_tolerance, "coordinate_gradient_bound": bound,
        "transformed_gradient_linf": float(np.abs(result.jac).max()),
        "fit_seconds": time.monotonic() - begun, "full_dimension_retained": z.shape[1],
        "preconditioner_metadata_sha256": prepared.metadata["metadata_sha256"],
    }
    if maximum_gradient > tol:
        raise RuntimeError("Preconditioned logistic did not meet ORIGINAL gradient tolerance: " + json.dumps(diagnostics))
    model = PreconditionedLogisticClassifier(
        coefficients, intercept, float(c), np.array([result.nit]), int(max_iter), float(tol), np.arange(3)
    )
    return model, diagnostics


def fit_logistic_candidates_preconditioned(x, labels, xv, labels_val, priors, config=None):
    """Same C grid and shared validation selector; one Train covariance fit."""
    config = config or {}
    priors = validate_priors(priors)
    prepared = prepare_logistic_preconditioner(x)
    labels = _labels(labels, len(prepared.x))
    xv = _design(xv)
    labels_val = _labels(labels_val, len(xv))
    candidates = []
    for c in config.get("logistic_Cs", [.001, .01, .1, 1.]):
        model, diagnostics = fit_preconditioned_logistic(
            prepared, labels, c, tol=float(config.get("logistic_tol", 1e-4)),
            max_iter=int(config.get("logistic_max_iter", 1000)),
        )
        row = {**diagnostics, **validation_scores(labels_val, model.predict_proba(xv), priors),
               "fit_attempts": [{"max_iter": model.max_iter, "n_iter": model.n_iter_.tolist(),
                                 "convergence_warnings": [], "elapsed_seconds": diagnostics["fit_seconds"]}]}
        candidates.append((model, row))
    if not candidates:
        raise ValueError("at least one C candidate is required")
    selected, selected_row = min(candidates, key=lambda item: (-item[1]["balanced_accuracy"], item[1]["log_loss"]))
    return selected, {"selected": selected_row, "candidates": [row for _, row in candidates],
                      "preconditioner": prepared.metadata,
                      "selection": "highest shared validation BA, ties lowest shared log loss; original-space gradient-certified identical L2 objective"}
