"""Pure chronological EWMA/GARCH baselines for the additional WM comparison.

Return inputs are fifteen-minute log returns on an uncompressed time grid.
Train fitting must receive the returns calculated *after* slicing Train
closes, so its first return is unknown rather than crossing the Train start.
Missing returns split the normal likelihood into independently initialized
segments. The supplied initial variance is a Train-only one-bar mean r^2.

Forecast decision t observes exactly close[t-context:t]. Its first price
initializes the local return path, yielding context-1 returns, never a return
from before that window. Parameters and initialization are explicit; these
functions neither select hyperparameters nor read experiment artifacts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import spearmanr

from .wm_ablation_evaluation import variance_qlike


@dataclass(frozen=True)
class GARCHFit:
    """GARCH parameters in original log-return variance units, plus fit status."""

    omega: float
    alpha: float
    beta: float
    initial_variance: float
    return_scale: float
    success: bool
    optimizer_success: bool
    status: int
    message: str
    nobs: int
    n_segments: int
    iterations: int
    function_evaluations: int
    optimizer_objective_scaled: float
    mean_negative_log_likelihood: float
    gradient_norm: float
    constraints_valid: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def log_returns_15m(close: Sequence[float] | np.ndarray) -> np.ndarray:
    """Same-length log returns with NaN at every missing-price boundary."""
    values = np.asarray(close, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("close must be a one-dimensional full-grid vector")
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values) & (values > 0)
    adjacent = valid[1:] & valid[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        result[1:] = np.where(adjacent, np.log(values[1:]) - np.log(values[:-1]), np.nan)
    return result


def _segments(returns: np.ndarray) -> list[np.ndarray]:
    finite = np.isfinite(returns)
    changes = np.diff(np.r_[False, finite, False].astype(np.int8))
    return [returns[start:end] for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True)]


def _validate_variance(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("initial_variance must be a positive finite Train-only variance")
    return value


def _validate_parameters(omega: float, alpha: float, beta: float) -> None:
    if not np.isfinite([omega, alpha, beta]).all() or omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        raise ValueError("GARCH requires omega>0, alpha,beta>=0 and alpha+beta<1")


def _variance_path(segment: np.ndarray, omega: float, alpha: float, beta: float, initial_variance: float) -> np.ndarray:
    """Variance BEFORE each observed return; independently reset per segment."""
    driver = np.empty(len(segment), dtype=np.float64)
    driver[0] = initial_variance
    driver[1:] = omega + alpha * np.square(segment[:-1])
    return lfilter([1.0], [1.0, -beta], driver)


def garch_conditional_variance(
    returns: Sequence[float] | np.ndarray, *, omega: float, alpha: float,
    beta: float, initial_variance: float,
) -> np.ndarray:
    """Full-grid h[t] before r[t], resetting to initial_variance after gaps."""
    _validate_parameters(omega, alpha, beta)
    initial_variance = _validate_variance(initial_variance)
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    result = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    changes = np.diff(np.r_[False, finite, False].astype(np.int8))
    for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True):
        result[start:end] = _variance_path(values[start:end], omega, alpha, beta, initial_variance)
    return result


def _physical_parameters(theta: np.ndarray) -> tuple[float, float, float]:
    omega = float(np.exp(theta[0]))
    # The positive residual category enforces strict covariance stationarity.
    exponents = np.exp(theta[1:])
    denominator = 1.0 + float(exponents.sum())
    return omega, float(exponents[0] / denominator), float(exponents[1] / denominator)


def _objective_and_gradient(theta: np.ndarray, segments: list[np.ndarray], initial_variance: float) -> tuple[float, np.ndarray]:
    omega, alpha, beta = _physical_parameters(theta)
    total = 0.0
    gradient = np.zeros(3, dtype=np.float64)
    nobs = 0
    for segment in segments:
        h = _variance_path(segment, omega, alpha, beta, initial_variance)
        squares = np.square(segment)
        total += float(np.sum(0.5 * (np.log(2.0 * np.pi) + np.log(h) + squares / h)))
        derivative_driver = np.zeros((len(segment), 3), dtype=np.float64)
        derivative_driver[1:, 0] = 1.0
        derivative_driver[1:, 1] = squares[:-1]
        derivative_driver[1:, 2] = h[:-1]
        derivatives = lfilter([1.0], [1.0, -beta], derivative_driver, axis=0)
        coefficient = 0.5 * (1.0 - squares / h) / h
        gradient += np.einsum("i,ij->j", coefficient, derivatives, optimize=False)
        nobs += len(segment)
    gradient /= nobs
    transformed = np.array([
        gradient[0] * omega,
        gradient[1] * alpha * (1.0 - alpha) - gradient[2] * alpha * beta,
        -gradient[1] * alpha * beta + gradient[2] * beta * (1.0 - beta),
    ])
    return total / nobs, transformed


def fit_gap_reset_garch(
    returns: Sequence[float] | np.ndarray, *, initial_variance: float,
    return_scale: float = 100.0, maxiter: int = 500,
) -> GARCHFit:
    """One fixed-start zero-mean GARCH(1,1) normal MLE using Train only.

    Finite runs remain distinct likelihood segments. The fixed initial
    alpha=.05, beta=.90 and L-BFGS-B settings are numerical choices, not a
    model search. Failed fits are returned with success=False; no substitute
    parameters, EWMA fallback, or second optimization is silently attempted.
    Scaling returns by 100 is internal only; returned omega is divided by
    scale^2 so forecasting accepts ordinary log returns and variances.
    """
    initial_variance = _validate_variance(initial_variance)
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if not np.isfinite(return_scale) or return_scale <= 0 or maxiter < 1:
        raise ValueError("return_scale and maxiter must be positive")
    segments = [segment * return_scale for segment in _segments(values)]
    nobs = sum(len(segment) for segment in segments)
    if nobs < 10:
        raise ValueError("at least ten finite Train returns are required")
    scaled_initial = initial_variance * return_scale ** 2
    initial = np.array([np.log(scaled_initial * .05), 0.0, np.log(18.0)])
    result = minimize(
        _objective_and_gradient, initial, args=(segments, scaled_initial), jac=True,
        method="L-BFGS-B", bounds=((-30.0, 30.0), (-20.0, 20.0), (-20.0, 20.0)),
        options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-7, "maxls": 40},
    )
    scaled_omega, alpha, beta = _physical_parameters(result.x)
    omega = scaled_omega / return_scale ** 2
    constraint_valid = bool(np.isfinite([omega, alpha, beta]).all() and omega > 0 and alpha >= 0 and beta >= 0 and alpha + beta < 1)
    finite = bool(np.isfinite(result.fun) and np.isfinite(result.jac).all())
    return GARCHFit(
        omega=float(omega), alpha=float(alpha), beta=float(beta),
        initial_variance=initial_variance, return_scale=float(return_scale),
        success=bool(result.success and constraint_valid and finite),
        optimizer_success=bool(result.success), status=int(result.status), message=str(result.message),
        nobs=nobs, n_segments=len(segments), iterations=int(result.nit),
        function_evaluations=int(result.nfev), optimizer_objective_scaled=float(result.fun),
        mean_negative_log_likelihood=float(result.fun - np.log(return_scale)),
        gradient_norm=float(np.linalg.norm(result.jac, ord=np.inf)), constraints_valid=constraint_valid,
    )


def _window_returns(close: Sequence[float] | np.ndarray, decisions: Sequence[int] | np.ndarray, context: int) -> np.ndarray:
    prices = np.asarray(close, dtype=np.float64)
    supplied = np.asarray(decisions)
    if prices.ndim != 1 or supplied.ndim != 1 or context < 2:
        raise ValueError("close/decisions must be one-dimensional and context>=2")
    indices = supplied.astype(np.int64)
    if not np.equal(supplied, indices).all() or np.any(indices < context) or np.any(indices > len(prices)):
        raise ValueError("decisions must be integer endpoints with a complete preceding context")
    positions = indices[:, None] - context + np.arange(context)[None, :]
    windows = prices[positions]
    valid = np.isfinite(windows) & (windows > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(np.log(windows), axis=1)
    # All comparison arms use complete context windows. An incomplete window
    # remains unavailable, rather than changing its effective history length.
    returns[~valid.all(axis=1)] = np.nan
    return returns


def ewma_variance_forecast(
    close: Sequence[float] | np.ndarray, decisions: Sequence[int] | np.ndarray,
    *, initial_variance: float, decay: float, context: int = 256, horizon: int = 16,
) -> np.ndarray:
    """Horizon variance = h * locally filtered next-step EWMA variance."""
    initial_variance = _validate_variance(initial_variance)
    if not 0 < decay < 1 or horizon < 1 or int(horizon) != horizon:
        raise ValueError("decay must be in (0,1) and horizon a positive integer")
    observed = _window_returns(close, decisions, context)
    state = np.full(len(observed), initial_variance)
    for column in range(observed.shape[1]):
        state = decay * state + (1.0 - decay) * np.square(observed[:, column])
    return state * horizon


def garch_variance_forecast(
    close: Sequence[float] | np.ndarray, decisions: Sequence[int] | np.ndarray,
    fit: GARCHFit, *, initial_variance: float, context: int = 256, horizon: int = 16,
) -> np.ndarray:
    """Sum future conditional variances after local context-only filtering."""
    if not fit.success:
        raise ValueError("cannot forecast with an unsuccessful GARCH fit; no fallback")
    _validate_parameters(fit.omega, fit.alpha, fit.beta)
    initial_variance = _validate_variance(initial_variance)
    if horizon < 1 or int(horizon) != horizon:
        raise ValueError("horizon must be a positive integer")
    observed = _window_returns(close, decisions, context)
    state = np.full(len(observed), initial_variance)
    for column in range(observed.shape[1]):
        state = fit.omega + fit.alpha * np.square(observed[:, column]) + fit.beta * state
    result = state.copy()
    for _ in range(1, int(horizon)):
        state = fit.omega + (fit.alpha + fit.beta) * state
        result += state
    return result


def score_forecasts(
    *, actual_return: Sequence[float] | np.ndarray | None = None,
    pred_return: Sequence[float] | np.ndarray | None = None,
    actual_variance: Sequence[float] | np.ndarray | None = None,
    pred_variance: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Independent optional heads; finite masks/counts remain explicit."""
    result: dict[str, Any] = {
        "n_return": 0, "return_rmse": None, "return_mae": None,
        "return_oos_r2": None, "return_spearman_ic": None,
        "n_variance": 0, "variance_qlike": None, "variance_rmse": None,
    }
    def pair(actual, predicted):
        if actual is None or predicted is None:
            if actual is not None or predicted is not None:
                raise ValueError("a forecast head requires both actual and predicted")
            return None
        a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
        if a.ndim != 1 or a.shape != p.shape:
            raise ValueError("actual/predicted must be matching one-dimensional vectors")
        finite = np.isfinite(a) & np.isfinite(p)
        return a[finite], p[finite]
    returns = pair(actual_return, pred_return)
    if returns is not None:
        a, p = returns
        result["n_return"] = len(a)
        if len(a):
            error = p - a
            denominator = float(np.sum(a * a))
            result.update(return_rmse=float(np.sqrt(np.mean(error * error))),
                          return_mae=float(np.mean(np.abs(error))),
                          return_oos_r2=1.0 - float(np.sum(error * error)) / denominator if denominator > 0 else None,
                          return_spearman_ic=float(spearmanr(a, p).statistic) if np.ptp(a) > 0 and np.ptp(p) > 0 else None)
    variance = pair(actual_variance, pred_variance)
    if variance is not None:
        a, p = variance
        if np.any(a < 0) or np.any(p < 0):
            raise ValueError("variance observations/forecasts cannot be negative")
        result["n_variance"] = len(a)
        if len(a):
            result.update(variance_qlike=float(np.mean(variance_qlike(a, p))),
                          variance_rmse=float(np.sqrt(np.mean((p - a) ** 2))))
    return result
