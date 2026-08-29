"""Wave3C causal context/forecast tournament for Plan011 development folds.

This module is intentionally separate from the frozen Wave3A tournament.  It
tests whether a causal context around each pre-shifted feature row and a
risk-aware objective are worth carrying into a larger model.  The only
development folds allowed here are 0, 2, and 8; fold test paths are emitted
once as a report-only tournament screen.

The portfolio metrics and timing/null definitions are shared with
``forecast_tournament``.  Only the context construction, downside target, and
candidate-specific policy/diagnostic logic live here.
"""
from __future__ import annotations

import json
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from unidream.eval.backtest import validate_execution_delay
from unidream.eval.forecast_tournament import (
    BENCHMARK_POSITION,
    CONSTANT_EXPOSURE_GRID,
    DEFAULT_EXECUTION_DELAY_BARS,
    DEFAULT_HORIZONS,
    DEV_CUTOFF,
    DEV_FOLDS,
    EXTERNAL_FEATURES,
    EXPECTED_FEATURES,
    PolicyParams,
    DevelopmentData,
    TIMING_MIN_WIN_RATE,
    TIMING_SUPERIORITY_MARGIN_PT,
    _canonical_sha256,
    _fold_bounds,
    _metric_at_horizon,
    _safe_median,
    backtest_metrics,
    circular_shift_positions,
    feature_ablation_columns,
    fit_candidate,
    forecast_metrics,
    future_targets,
    select_fixed_exposure,
    sha256_file,
)
from unidream.experiments.run_config import config_fingerprint, source_fingerprint


SCHEMA_VERSION = 1
WAVE_NAME = "wave3c_forecast_context_tournament"
WAVE_FOLDS = DEV_FOLDS
WAVE_SEED = 7
FIXED_EXECUTION_DELAY_BARS = DEFAULT_EXECUTION_DELAY_BARS
CONTEXT_LAGS = (1, 4, 16, 64)
CONTEXT_WINDOWS = (4, 16, 64, 256)
CONTEXT_STATS = ("mean", "std", "slope")
HORIZON_GRID = DEFAULT_HORIZONS
REGRESSION_CANDIDATES = (
    "context_ridge_risk_adjusted",
    "context_histgb_risk_adjusted",
)
CLASSIFIER_CANDIDATE = "context_downside_classifier"
ALL_CANDIDATES = (*REGRESSION_CANDIDATES, CLASSIFIER_CANDIDATE)
REGRESSION_HYPERPARAMETERS = {
    "context_ridge_risk_adjusted": {"alpha": 1.0},
    "context_histgb_risk_adjusted": {
        "learning_rate": 0.05,
        "max_iter": 30,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
    },
}
CLASSIFIER_HYPERPARAMETERS = {
    "learning_rate": 0.05,
    "max_iter": 30,
    "max_leaf_nodes": 15,
    "l2_regularization": 1.0,
}
FIXED_RISK_PENALTY = 0.50
TRAIN_DOWNSIDE_QUANTILE = 0.75
CLASSIFIER_PROBABILITY_THRESHOLD = 0.50
DEFAULT_OVERLAY_GRID = (0.04, 0.08, 0.12)
DEFAULT_HYSTERESIS_GRID = (0.0, 0.25)
DEFAULT_MIN_HOLD_GRID = (0, 32)
DEFAULT_MAX_POSITION_STEP = 0.08
TIMING_NULL_SHIFTS = (1, 16, 64)
TIMING_LAGS = (1, 16)
CONTEXT_FEATURE_SET_ORDER = ("ohlcv13", "full17")
BAR_INTERVAL = pd.Timedelta(minutes=15)
MAX_CONTEXT_HISTORY = max((*CONTEXT_LAGS, *CONTEXT_WINDOWS))


@dataclass(frozen=True)
class ContextRegressionCandidate:
    """Fitted context regression candidate and validation-scale contract."""

    name: str
    horizons: tuple[int, ...]
    return_models: dict[int, Any]
    risk_models: dict[int, Any]
    fit_rows: int
    dropped_nonfinite_rows: int

    def predict(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        values = features.to_numpy(dtype=np.float64)
        finite_rows = np.isfinite(values).all(axis=1)
        returns: list[np.ndarray] = []
        risks: list[np.ndarray] = []
        for horizon in self.horizons:
            predicted_return = np.full(len(values), np.nan, dtype=np.float64)
            predicted_risk = np.full(len(values), np.nan, dtype=np.float64)
            if np.any(finite_rows):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=RuntimeWarning,
                        module=r"sklearn\.linear_model\._base",
                    )
                    predicted_return[finite_rows] = self.return_models[horizon].predict(
                        values[finite_rows]
                    )
                    predicted_risk[finite_rows] = self.risk_models[horizon].predict(
                        values[finite_rows]
                    )
            returns.append(predicted_return)
            risks.append(np.maximum(predicted_risk, 0.0))
        return {"return": np.column_stack(returns), "risk": np.column_stack(risks)}


@dataclass(frozen=True)
class DownsideClassifierCandidate:
    """Fitted future-downside classifier with train-only event thresholds."""

    name: str
    horizons: tuple[int, ...]
    models: dict[int, Any]
    target_thresholds: dict[int, float]
    train_event_rates: dict[int, float]
    fit_rows: int
    unavailable: dict[int, str]


def _strict_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    """Validate fixed positive integer horizons without coercion/clamping."""
    values = tuple(horizons)
    if not values:
        raise ValueError("at least one positive integer horizon is required")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
        for value in values
    ):
        raise ValueError(f"horizons must be positive integers, got {values!r}")
    return tuple(sorted(set(int(value) for value in values)))


def _strict_wave_folds(splits: Sequence[Any]) -> tuple[int, ...]:
    """Require every development fold exactly once for the formal screen."""
    folds = tuple(int(split.fold_idx) for split in splits)
    if len(folds) != len(WAVE_FOLDS) or len(set(folds)) != len(folds) or set(folds) != set(WAVE_FOLDS):
        raise ValueError(
            f"Wave3C requires exactly one each of development folds {list(WAVE_FOLDS)}, got {list(folds)}"
        )
    return tuple(sorted(folds))


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _finite_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Validate a time-indexed feature frame without imputing missing values."""
    if not isinstance(frame, pd.DataFrame) or len(frame) == 0:
        raise ValueError(f"{label} must be a non-empty DataFrame")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError(f"{label} index must be sorted and unique")
    values = frame.to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError(f"{label} contains infinite base values")
    return frame.astype(np.float64, copy=False)


def _contiguous_transition_mask(index: pd.DatetimeIndex, transitions: int) -> np.ndarray:
    """Mark rows whose preceding/future transition window is 15-minute exact."""
    if transitions <= 0:
        return np.ones(len(index), dtype=bool)
    if len(index) <= transitions:
        return np.zeros(len(index), dtype=bool)
    deltas = np.diff(index.asi8) == int(BAR_INTERVAL.value)
    windows = np.lib.stride_tricks.sliding_window_view(deltas, transitions)
    result = np.zeros(len(index), dtype=bool)
    # A context row t uses transitions t-transitions ... t-1. A target row t
    # uses transitions t ... t+h-1 (handled by future_target_mask below).
    result[transitions:] = windows.all(axis=1)
    return result


def context_eligibility_mask(index: Sequence[Any], *, history: int = MAX_CONTEXT_HISTORY) -> np.ndarray:
    """Return the strict contiguous-history mask for context rows."""
    parsed = pd.DatetimeIndex(pd.to_datetime(index))
    if not parsed.is_monotonic_increasing or not parsed.is_unique:
        raise ValueError("context timestamps must be sorted and unique")
    return _contiguous_transition_mask(parsed, int(history))


def future_target_eligibility_mask(
    index: Sequence[Any],
    horizons: Iterable[int],
) -> np.ndarray:
    """Return per-horizon masks requiring exact 15-minute future transitions."""
    parsed = pd.DatetimeIndex(pd.to_datetime(index))
    if not parsed.is_monotonic_increasing or not parsed.is_unique:
        raise ValueError("target timestamps must be sorted and unique")
    horizon_values = _strict_horizons(horizons)
    masks = np.zeros((len(parsed), len(horizon_values)), dtype=bool)
    if len(parsed) <= 1:
        return masks
    deltas = np.diff(parsed.asi8) == int(BAR_INTERVAL.value)
    for column, horizon in enumerate(horizon_values):
        if len(parsed) <= horizon:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(deltas, horizon)
        masks[: len(parsed) - horizon, column] = windows.all(axis=1)
    return masks


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Return a causal rolling slope using values before the current row."""
    values = series.to_numpy(dtype=np.float64)
    output = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) <= window:
        return pd.Series(output, index=series.index, name=series.name)
    x = np.arange(window, dtype=np.float64)
    x_centered = x - float(x.mean())
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0.0:
        return pd.Series(output, index=series.index, name=series.name)
    # For row t, the causal input passed by build_causal_context is already
    # ``base.shift(1)``. The rolling window is therefore series[t-window+1:t+1]
    # (the same window used by pandas rolling below), and it contains only
    # base[t-window:t]. Use an explicit sliding-window dot product: np.convolve
    # reverses its second operand and would invert the slope sign.
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    dot_values = windows @ x
    sums = windows.sum(axis=1)
    slopes = (dot_values - float(x.mean()) * sums) / denominator
    output[window - 1 :] = slopes
    return pd.Series(output, index=series.index, name=series.name)


def build_causal_context(features: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic lag/rolling context with no current/future leakage.

    The base row is already pre-shifted by the research feature pipeline.  Lag
    columns use strictly older base rows, and each rolling statistic uses
    ``features.shift(1)`` so the context itself is also strictly historical.
    Initial warm-up rows are intentionally NaN and are removed by the
    train-only finite-row mask in candidate fitting; no zero imputation occurs.
    """
    base = _finite_frame(features, "context base features")
    pieces: list[pd.DataFrame] = [base.add_prefix("base__")]
    for lag in CONTEXT_LAGS:
        pieces.append(base.shift(lag).add_prefix(f"lag_{lag}__"))

    historical = base.shift(1)
    for window in CONTEXT_WINDOWS:
        rolling = historical.rolling(window=window, min_periods=window)
        pieces.append(rolling.mean().add_prefix(f"rolling_mean_{window}__"))
        pieces.append(rolling.std(ddof=0).add_prefix(f"rolling_std_{window}__"))
        slopes = pd.DataFrame(
            {
                name: _rolling_slope(historical[name], window)
                for name in base.columns
            },
            index=base.index,
        )
        pieces.append(slopes.add_prefix(f"rolling_slope_{window}__"))
    result = pd.concat(pieces, axis=1)
    if result.columns.duplicated().any():
        raise ValueError("causal context generated duplicate feature names")
    eligible = context_eligibility_mask(result.index)
    result.loc[~eligible, :] = np.nan
    return result


def context_prefix_is_causal(
    features: pd.DataFrame,
    *,
    cutoff: int,
    perturbation: float = 17.0,
) -> bool:
    """Check that changing a future suffix leaves the earlier context intact."""
    if cutoff <= 0 or cutoff >= len(features):
        raise ValueError("causality cutoff must split the feature frame")
    original = build_causal_context(features)
    changed = features.copy()
    changed.iloc[cutoff:, :] = changed.iloc[cutoff:, :] + float(perturbation)
    mutated = build_causal_context(changed)
    return bool(
        np.allclose(
            original.iloc[:cutoff].to_numpy(dtype=np.float64),
            mutated.iloc[:cutoff].to_numpy(dtype=np.float64),
            equal_nan=True,
        )
    )


def future_downside_targets(
    returns: Sequence[float] | np.ndarray,
    horizons: Iterable[int] = HORIZON_GRID,
    *,
    timestamps: Sequence[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return worst prefix loss over each ``t+1..t+h`` future window.

    ``downside[t, h]`` is ``-min(0, min_k sum(returns[t+1:t+k]))``.  It is a
    non-negative drawdown-from-entry proxy and therefore excludes
    ``returns[t]`` exactly like the direct-return target contract.
    """
    values = np.asarray(returns, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("future-downside returns must be non-empty and finite")
    horizon_values = _strict_horizons(horizons)
    targets = np.zeros((len(values), len(horizon_values)), dtype=np.float64)
    masks = np.zeros_like(targets, dtype=bool)
    for column, horizon in enumerate(horizon_values):
        valid_length = len(values) - horizon
        if valid_length <= 0:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(values, horizon + 1)[:, 1:]
        prefix_returns = np.cumsum(windows, axis=1)
        downside = -np.minimum(np.min(prefix_returns, axis=1), 0.0)
        targets[:valid_length, column] = downside[:valid_length]
        masks[:valid_length, column] = True
    if timestamps is not None:
        timestamps_arr = pd.DatetimeIndex(pd.to_datetime(timestamps))
        if len(timestamps_arr) != len(values):
            raise ValueError("future-downside timestamps must align with returns")
        masks &= future_target_eligibility_mask(timestamps_arr, horizon_values)
    return targets, masks


def train_downside_event_contract(
    train_targets: np.ndarray,
    train_mask: np.ndarray,
    *,
    quantile: float = TRAIN_DOWNSIDE_QUANTILE,
) -> tuple[float, np.ndarray, float]:
    """Fit a train-only quantile event threshold and return binary labels."""
    valid = np.asarray(train_mask, dtype=bool) & np.isfinite(train_targets)
    values = np.asarray(train_targets, dtype=np.float64)[valid]
    if len(values) < 2:
        raise ValueError("downside event has fewer than two finite training rows")
    threshold = float(np.quantile(values, float(quantile)))
    labels = np.asarray(train_targets, dtype=np.float64) >= threshold
    labels = labels & np.asarray(train_mask, dtype=bool)
    event_rate = float(np.mean(labels[valid]))
    return threshold, labels, event_rate


def _future_regression_targets(
    returns: np.ndarray,
    timestamps: Sequence[Any],
    horizons: tuple[int, ...],
    *,
    target_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    horizon_values = _strict_horizons(horizons)
    targets, masks = future_targets(returns, horizon_values, target_kind=target_kind)
    masks &= future_target_eligibility_mask(timestamps, horizon_values)
    return targets, masks


def _metric_reason(metrics: Mapping[str, Any], reason: str) -> dict[str, Any]:
    output = dict(metrics)
    output["status"] = "N/A"
    output["reason"] = reason
    return output


def classifier_metrics(
    probabilities: Sequence[float] | np.ndarray,
    downside_targets: Sequence[float] | np.ndarray,
    mask: Sequence[bool] | np.ndarray,
    *,
    target_threshold: float,
    split: str,
    horizon: int,
    probability_threshold: float = CLASSIFIER_PROBABILITY_THRESHOLD,
) -> dict[str, Any]:
    """Compute classifier metrics, preserving metric-specific N/A reasons."""
    probabilities_arr = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    targets_arr = np.asarray(downside_targets, dtype=np.float64).reshape(-1)
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    if len(probabilities_arr) != len(targets_arr) or len(targets_arr) != len(mask_arr):
        raise ValueError("classifier metric arrays must have equal lengths")
    valid = mask_arr & np.isfinite(probabilities_arr) & np.isfinite(targets_arr)
    result: dict[str, Any] = {
        "split": split,
        "target_kind": "future_downside_event",
        "horizon": int(horizon),
        "target_threshold": float(target_threshold),
        "probability_threshold": float(probability_threshold),
        "n_valid": int(valid.sum()),
        "metrics": {
            "auc": None,
            "brier": None,
            "precision": None,
            "recall": None,
        },
        "status": "N/A",
    }
    if not np.any(valid):
        result["reason"] = "no finite classifier probability/target rows"
        return result
    actual = (targets_arr[valid] >= float(target_threshold)).astype(np.int8)
    predicted_probability = np.clip(probabilities_arr[valid], 0.0, 1.0)
    predicted = (predicted_probability >= float(probability_threshold)).astype(np.int8)
    reasons: list[str] = []
    if np.unique(actual).size < 2:
        reasons.append("AUC is N/A because the split target has one class")
    else:
        result["metrics"]["auc"] = float(roc_auc_score(actual, predicted_probability))
    result["metrics"]["brier"] = float(brier_score_loss(actual, predicted_probability))
    result["metrics"]["precision"] = float(
        precision_score(actual, predicted, zero_division=0)
    )
    result["metrics"]["recall"] = float(recall_score(actual, predicted, zero_division=0))
    result["status"] = "ok"
    if reasons:
        result["reason"] = "; ".join(reasons)
    return result


def continuous_overlay_positions(
    score: Sequence[float] | np.ndarray,
    params: PolicyParams,
    *,
    benchmark: float = BENCHMARK_POSITION,
    min_position: float = 0.50,
    max_position: float = 1.12,
    max_position_step: float = DEFAULT_MAX_POSITION_STEP,
    eligible_mask: Sequence[bool] | np.ndarray | None = None,
) -> np.ndarray:
    """Map a causal score to a bounded continuous overlay with hysteresis."""
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if eligible_mask is None:
        eligible = finite
    else:
        eligible = np.asarray(eligible_mask, dtype=bool).reshape(-1)
        if len(eligible) != len(values):
            raise ValueError("continuous overlay eligibility must align with score")
        eligible &= finite
    if params.threshold < 0 or params.overlay_magnitude < 0 or not 0 <= params.hysteresis < 1:
        raise ValueError("invalid continuous overlay policy parameters")
    positions = np.full(len(values), float(benchmark), dtype=np.float64)
    current = float(benchmark)
    hold_remaining = 0
    active = False
    for index, value in enumerate(values):
        if not eligible[index]:
            # Keep the calendar and returns path intact. The fixed execution
            # delay then carries the last eligible decision through a gap bar.
            current = float(benchmark)
            active = False
            hold_remaining = 0
            positions[index] = current
            continue
        if hold_remaining > 0:
            hold_remaining -= 1
        else:
            absolute = abs(float(value))
            enter = absolute > float(params.threshold)
            stay = absolute > float(params.threshold) * (1.0 - float(params.hysteresis))
            active = stay if active else enter
            normalized = float(np.tanh(value))
            raw_overlay = float(params.overlay_magnitude) * normalized if active else 0.0
            target = float(np.clip(benchmark + raw_overlay, min_position, max_position))
            delta = float(np.clip(target - current, -max_position_step, max_position_step))
            next_position = float(np.clip(current + delta, min_position, max_position))
            if abs(next_position - current) > 1e-12 and params.min_hold > 0:
                hold_remaining = int(params.min_hold)
            current = next_position
        positions[index] = current
    return positions


def _score_threshold_grid(score: np.ndarray) -> tuple[float, ...]:
    finite = np.abs(np.asarray(score, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return (0.0,)
    values = (0.0, float(np.quantile(finite, 0.50)), float(np.quantile(finite, 0.75)))
    return tuple(sorted(set(values)))


def _validation_scale(values: np.ndarray) -> tuple[float, list[str]]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    reasons: list[str] = []
    if len(finite) == 0:
        return 1.0, ["validation scale fallback: no finite predictions"]
    scale = float(np.quantile(np.abs(finite), 0.75))
    if not np.isfinite(scale) or scale <= 1e-12:
        reasons.append("validation scale fallback: near-zero prediction magnitude")
        return 1.0, reasons
    return scale, reasons


def risk_adjusted_scores(
    prediction: Mapping[str, np.ndarray],
    horizons: Iterable[int],
    *,
    validation: bool,
    return_scales: Mapping[int, float] | None = None,
    risk_scales: Mapping[int, float] | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, float], dict[int, float], dict[int, list[str]]]:
    """Combine predicted return/risk using scales fitted on validation only."""
    scores: dict[int, np.ndarray] = {}
    selected_return_scales: dict[int, float] = {}
    selected_risk_scales: dict[int, float] = {}
    reasons: dict[int, list[str]] = {}
    return_predictions = np.asarray(prediction["return"], dtype=np.float64)
    risk_predictions = np.asarray(prediction["risk"], dtype=np.float64)
    horizon_values = _strict_horizons(horizons)
    if return_predictions.ndim != 2 or risk_predictions.shape != return_predictions.shape:
        raise ValueError("risk-adjusted predictions must be aligned two-dimensional arrays")
    if return_predictions.shape[1] != len(horizon_values):
        raise ValueError("risk-adjusted prediction horizon count does not match horizons")
    for column, horizon in enumerate(horizon_values):
        prediction_for_horizon = {
            "return": return_predictions[:, column],
            "risk": risk_predictions[:, column],
        }
        if validation:
            return_scale, return_reason = _validation_scale(prediction_for_horizon["return"])
            risk_scale, risk_reason = _validation_scale(prediction_for_horizon["risk"])
            reasons[horizon] = [*return_reason, *risk_reason]
        else:
            if return_scales is None or risk_scales is None:
                raise ValueError("test risk-adjusted scores require validation scales")
            return_scale = float(return_scales[horizon])
            risk_scale = float(risk_scales[horizon])
            reasons[horizon] = []
        selected_return_scales[horizon] = float(return_scale)
        selected_risk_scales[horizon] = float(risk_scale)
        scores[horizon] = (
            prediction_for_horizon["return"] / float(return_scale)
            - FIXED_RISK_PENALTY
            * np.maximum(prediction_for_horizon["risk"], 0.0)
            / float(risk_scale)
        )
    return scores, selected_return_scales, selected_risk_scales, reasons


def _classifier_scores(
    probabilities: Mapping[int, np.ndarray],
    train_event_rates: Mapping[int, float],
) -> dict[int, np.ndarray]:
    scores = {}
    for horizon, values in probabilities.items():
        if horizon not in train_event_rates:
            continue
        baseline = float(train_event_rates[horizon])
        denominator = max(baseline, 0.05)
        scores[horizon] = (baseline - np.asarray(values, dtype=np.float64)) / denominator
    return scores


def _selection_score(metrics: Mapping[str, Any]) -> float:
    return (
        float(metrics["alpha_excess_pt"])
        - max(0.0, float(metrics["maxdd_delta_pt"]))
        - 0.05 * float(metrics["turnover"])
    )


def select_horizon_and_policy(
    scores: Mapping[int, np.ndarray],
    validation_returns: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    constant_exposure: float,
    execution_delay: int = FIXED_EXECUTION_DELAY_BARS,
) -> tuple[int, PolicyParams, dict[str, Any]]:
    """Select horizon and continuous overlay controls from validation only."""
    validated_delay = validate_execution_delay(execution_delay)
    if validated_delay != FIXED_EXECUTION_DELAY_BARS:
        raise ValueError("Wave3C execution delay is fixed to one operational bar")
    constant_metrics = backtest_metrics(
        validation_returns,
        np.full(len(validation_returns), constant_exposure, dtype=np.float64),
        cfg,
        benchmark=BENCHMARK_POSITION,
        execution_delay_bars=FIXED_EXECUTION_DELAY_BARS,
    )
    trials: list[dict[str, Any]] = []
    for horizon in sorted(scores):
        score = np.asarray(scores[horizon], dtype=np.float64)
        for threshold in _score_threshold_grid(score):
            for magnitude in DEFAULT_OVERLAY_GRID:
                for hysteresis in DEFAULT_HYSTERESIS_GRID:
                    for min_hold in DEFAULT_MIN_HOLD_GRID:
                        params = PolicyParams(
                            threshold=float(threshold),
                            overlay_magnitude=float(magnitude),
                            hysteresis=float(hysteresis),
                            min_hold=int(min_hold),
                            execution_delay=FIXED_EXECUTION_DELAY_BARS,
                        )
                        # The dynamic path is centered on the validation-
                        # selected constant exposure. This makes its timing
                        # increment a pure timing comparison rather than a
                        # hidden exposure bet against B&H=1.0.
                        positions = continuous_overlay_positions(
                            score,
                            params,
                            benchmark=constant_exposure,
                            eligible_mask=np.isfinite(score),
                        )
                        metrics = backtest_metrics(
                            validation_returns,
                            positions,
                            cfg,
                            benchmark=BENCHMARK_POSITION,
                            execution_delay_bars=FIXED_EXECUTION_DELAY_BARS,
                        )
                        trials.append(
                            {
                                "horizon": int(horizon),
                                "params": params,
                                "metrics": metrics,
                                "score": _selection_score(metrics),
                            }
                        )
    if not trials:
        raise ValueError("context validation horizon/policy grid is empty")
    selected = max(
        enumerate(trials),
        key=lambda pair: (float(pair[1]["score"]), -int(pair[0])),
    )[1]
    params: PolicyParams = selected["params"]
    return int(selected["horizon"]), params, {
        "split": "validation",
        "status": "selected_on_validation",
        "constant_exposure": float(constant_exposure),
        "policy_baseline": float(constant_exposure),
        "constant_metrics": constant_metrics,
        "horizon_grid": sorted(int(value) for value in scores),
        "selected_horizon": int(selected["horizon"]),
        "selected_params": {
            "threshold": float(params.threshold),
            "overlay_magnitude": float(params.overlay_magnitude),
            "hysteresis": float(params.hysteresis),
            "min_hold": int(params.min_hold),
            "execution_delay": int(params.execution_delay),
        },
        "execution_delay_selection": "fixed_operational_contract",
        "selected_metrics": selected["metrics"],
        "selection_score": float(selected["score"]),
        "candidate_trial_count": int(len(trials)),
        "risk_penalty": float(FIXED_RISK_PENALTY),
    }


def fit_downside_classifier(
    features: pd.DataFrame,
    returns: np.ndarray,
    *,
    timestamps: Sequence[Any] | None = None,
    horizons: Iterable[int] = HORIZON_GRID,
    seed: int = WAVE_SEED,
    max_fit_rows: int = 20_000,
) -> DownsideClassifierCandidate:
    """Fit one HistGB event classifier per horizon using train-only labels."""
    values = features.to_numpy(dtype=np.float64)
    finite_x = np.isfinite(values).all(axis=1)
    selected_indices = (
        np.arange(len(features), dtype=np.int64)
        if len(features) <= max_fit_rows
        else np.linspace(0, len(features) - 1, max_fit_rows, dtype=np.int64)
    )
    horizon_values = _strict_horizons(horizons)
    target_values, target_masks = future_downside_targets(
        returns,
        horizon_values,
        timestamps=timestamps,
    )
    models: dict[int, Any] = {}
    thresholds: dict[int, float] = {}
    event_rates: dict[int, float] = {}
    unavailable: dict[int, str] = {}
    for column, horizon in enumerate(horizon_values):
        valid = selected_indices[target_masks[selected_indices, column] & finite_x[selected_indices]]
        if len(valid) < 2:
            unavailable[horizon] = "fewer than two finite train rows"
            continue
        try:
            threshold, labels, event_rate = train_downside_event_contract(
                target_values[valid, column],
                target_masks[valid, column],
            )
            y = labels.astype(np.int8)
            if np.unique(y).size < 2:
                unavailable[horizon] = "train quantile label has one class"
                continue
            model = HistGradientBoostingClassifier(
                **CLASSIFIER_HYPERPARAMETERS,
                early_stopping=False,
                random_state=int(seed),
            )
            model.fit(values[valid], y)
            models[horizon] = model
            thresholds[horizon] = float(threshold)
            event_rates[horizon] = float(event_rate)
        except (ValueError, RuntimeError) as exc:
            unavailable[horizon] = str(exc)
    if not models:
        raise ValueError(f"all downside classifier horizons failed: {unavailable}")
    return DownsideClassifierCandidate(
        name=CLASSIFIER_CANDIDATE,
        horizons=horizon_values,
        models=models,
        target_thresholds=thresholds,
        train_event_rates=event_rates,
        fit_rows=int(len(selected_indices)),
        unavailable=unavailable,
    )


def predict_downside_probabilities(
    candidate: DownsideClassifierCandidate,
    features: pd.DataFrame,
) -> dict[int, np.ndarray]:
    values = features.to_numpy(dtype=np.float64)
    finite_rows = np.isfinite(values).all(axis=1)
    output: dict[int, np.ndarray] = {}
    for horizon in candidate.horizons:
        probabilities = np.full(len(values), np.nan, dtype=np.float64)
        if horizon in candidate.models and np.any(finite_rows):
            probabilities[finite_rows] = candidate.models[horizon].predict_proba(
                values[finite_rows]
            )[:, 1]
        output[horizon] = probabilities
    return output


def _provenance(
    *,
    data: DevelopmentData,
    cfg: Mapping[str, Any],
    config_path: str,
    seed: int,
    feature_sets: tuple[str, ...],
    candidates: tuple[str, ...],
    horizons: tuple[int, ...],
    execution_delay: int,
) -> dict[str, Any]:
    contract = {
        "wave": WAVE_NAME,
        "allowed_folds": list(WAVE_FOLDS),
        "requested_folds": [int(split.fold_idx) for split in data.splits],
        "cutoff_exclusive": str(DEV_CUTOFF),
        "right_exclusive_split": True,
        "train_years": int(data.train_years),
        "val_months": int(data.val_months),
        "test_months": int(data.test_months),
        "target_windows": "t+1..t+h",
        "horizons": list(horizons),
        "context_lags": list(CONTEXT_LAGS),
        "context_rolling_windows": list(CONTEXT_WINDOWS),
        "context_rolling_stats": list(CONTEXT_STATS),
        "feature_sets": list(feature_sets),
        "feature_schema_expected": list(EXPECTED_FEATURES),
        "ohlcv13_primary": True,
        "full17_secondary_quality_flag": True,
        "candidates": list(candidates),
        "execution_delay": {
            "operational_bars": int(execution_delay),
            "sensitivity_lags": list(TIMING_LAGS),
        },
        "null_shifts": list(TIMING_NULL_SHIFTS),
        "risk_penalty": float(FIXED_RISK_PENALTY),
        "downside_label_quantile": float(TRAIN_DOWNSIDE_QUANTILE),
        "classifier_probability_threshold": float(CLASSIFIER_PROBABILITY_THRESHOLD),
        "selection": "train fit; horizon/policy/threshold selected validation-only; development test report-only",
    }
    feature_hash = sha256_file(data.feature_path)
    returns_hash = sha256_file(data.returns_path)
    return {
        "commit_hash": _git_commit(),
        "config_path": str(config_path),
        "config_sha256": config_fingerprint(dict(cfg)),
        "source_sha256": source_fingerprint(),
        "data_contract": contract,
        "data_contract_sha256": _canonical_sha256(contract),
        "data_artifacts": {
            "features": str(data.feature_path),
            "features_sha256": feature_hash,
            "returns": str(data.returns_path),
            "returns_sha256": returns_hash,
        },
        "data_sha256": _canonical_sha256(
            {
                "features_sha256": feature_hash,
                "returns_sha256": returns_hash,
                "folds": [int(split.fold_idx) for split in data.splits],
                "horizons": list(horizons),
            }
        ),
        "seed": int(seed),
    }


def _quality_record(frame: pd.DataFrame, feature_set: str, fold: int, split_name: str) -> dict[str, Any]:
    external: dict[str, Any] = {}
    flags: list[str] = []
    for name in EXTERNAL_FEATURES:
        if name not in frame.columns:
            external[name] = {
                "rows": int(len(frame)),
                "finite_count": 0,
                "missing_count": int(len(frame)),
                "zero_count": 0,
                "nonzero_count": 0,
                "status": "N/A_excluded_from_ohlcv13",
                "quality_flag": "N/A_external_columns_not_in_ohlcv13",
            }
            flags.append("N/A_external_columns_not_in_ohlcv13")
            continue
        values = frame[name].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        zeros = finite & (np.abs(values) <= 1e-12)
        nonzero = finite & ~zeros
        missing = ~finite
        external[name] = {
            "rows": int(len(values)),
            "finite_count": int(finite.sum()),
            "missing_count": int(missing.sum()),
            "zero_count": int(zeros.sum()),
            "nonzero_count": int(nonzero.sum()),
            "finite_rate": float(finite.mean()) if len(values) else None,
            "missing_rate": float(missing.mean()) if len(values) else None,
            "zero_rate": float(zeros.mean()) if len(values) else None,
            "nonzero_rate": float(nonzero.mean()) if len(values) else None,
            "status": "ok_with_quality_flag",
            "quality_flag": "N/A_zero_vs_missing_indistinguishable",
        }
        flags.append("N/A_zero_vs_missing_indistinguishable")
    return {
        "fold": int(fold),
        "feature_set": feature_set,
        "split": split_name,
        "rows": int(len(frame)),
        "external": external,
        "quality_flags": sorted(set(flags)),
        "status": "ok_with_quality_flag" if flags else "ok",
        "promotion_status": (
            "blocked_by_data_quality" if feature_set == "full17" else "eligible_for_promotion"
        ),
        "availability_mask_present": False,
        "contract_note": "No availability mask exists; zero and missing/imputed values remain ambiguous and are never zero-filled.",
    }


def _classifier_target_metrics(
    probabilities: Mapping[int, np.ndarray],
    returns: np.ndarray,
    candidate: DownsideClassifierCandidate,
    split: str,
    timestamps: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    targets, masks = future_downside_targets(
        returns,
        candidate.horizons,
        timestamps=timestamps,
    )
    records = []
    for column, horizon in enumerate(candidate.horizons):
        if horizon not in candidate.models:
            records.append(
                {
                    "split": split,
                    "target_kind": "future_downside_event",
                    "horizon": int(horizon),
                    "status": "N/A",
                    "n_valid": 0,
                    "metrics": {"auc": None, "brier": None, "precision": None, "recall": None},
                    "reason": candidate.unavailable.get(horizon, "classifier horizon unavailable"),
                }
            )
            continue
        records.append(
            classifier_metrics(
                probabilities[horizon],
                targets[:, column],
                masks[:, column],
                target_threshold=candidate.target_thresholds[horizon],
                split=split,
                horizon=horizon,
            )
        )
    return records


def _append_record(records: list[dict[str, Any]], provenance: Mapping[str, Any], **payload: Any) -> None:
    records.append({**provenance, "schema_version": SCHEMA_VERSION, **payload})


def _candidate_hyperparameters(candidate: str) -> dict[str, Any]:
    if candidate in REGRESSION_HYPERPARAMETERS:
        return dict(REGRESSION_HYPERPARAMETERS[candidate])
    if candidate == CLASSIFIER_CANDIDATE:
        return dict(CLASSIFIER_HYPERPARAMETERS)
    raise ValueError(f"unknown Wave3C candidate: {candidate}")


def _path_alpha(row: Mapping[str, Any], path: str, key: str | None = None) -> float | None:
    try:
        value: Any = row["test_economics"][path]
        if key is not None:
            value = value[key]
        value = float(value["alpha_excess_pt"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _timing_superiority(
    dynamic: Sequence[float | None], comparator: Sequence[float | None]
) -> dict[str, float | int | None]:
    differences = [
        float(dynamic_value) - float(comparator_value)
        for dynamic_value, comparator_value in zip(dynamic, comparator)
        if dynamic_value is not None
        and comparator_value is not None
        and np.isfinite(float(dynamic_value))
        and np.isfinite(float(comparator_value))
    ]
    wins = sum(value > TIMING_SUPERIORITY_MARGIN_PT for value in differences)
    return {
        "median_alpha_difference_pt": _safe_median(differences),
        "win_folds": int(wins),
        "win_rate": float(wins / len(differences)) if differences else None,
        "margin_pt": float(TIMING_SUPERIORITY_MARGIN_PT),
        "required_win_rate": float(TIMING_MIN_WIN_RATE),
    }


def aggregate_context_gate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Apply the fixed Wave3C quality/economic/timing gate fail-closed."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["feature_set"]), str(row["candidate"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (feature_set, candidate), values in sorted(grouped.items()):
        folds = tuple(int(row.get("fold", -1)) for row in values)
        complete_folds = len(folds) == len(WAVE_FOLDS) and set(folds) == set(WAVE_FOLDS)
        quality_values: list[float | None] = []
        quality_metric = "auc" if candidate == CLASSIFIER_CANDIDATE else "spearman_ic"
        for row in values:
            horizon = int(row.get("selected_horizon", -1))
            records = (
                row.get("test_classifier_metrics", [])
                if candidate == CLASSIFIER_CANDIDATE
                else row.get("test_return_metrics", [])
            )
            quality_values.append(_metric_at_horizon(records, horizon, quality_metric))
        quality_positive = sum(
            value is not None
            and np.isfinite(float(value))
            and (float(value) > 0.50 if quality_metric == "auc" else float(value) > 0.0)
            for value in quality_values
        )
        dynamic_alpha = [_path_alpha(row, "dynamic") for row in values]
        constant_alpha = [_path_alpha(row, "constant") for row in values]
        lag16_alpha = [_path_alpha(row, "lags", "16") for row in values]
        null64_alpha = [_path_alpha(row, "nulls", "64") for row in values]
        timing_increment = [
            float(dynamic_value) - float(constant_value)
            for dynamic_value, constant_value in zip(dynamic_alpha, constant_alpha)
            if dynamic_value is not None and constant_value is not None
        ]
        dynamic_dd = [
            float(row["test_economics"]["dynamic"]["maxdd_delta_pt"])
            for row in values
            if row.get("test_economics", {}).get("dynamic", {}).get("maxdd_delta_pt") is not None
        ]
        constant_dd = [
            float(row["test_economics"]["constant"]["maxdd_delta_pt"])
            for row in values
            if row.get("test_economics", {}).get("constant", {}).get("maxdd_delta_pt") is not None
        ]
        turnover = [
            float(row["test_economics"]["dynamic"]["turnover"])
            for row in values
            if row.get("test_economics", {}).get("dynamic", {}).get("turnover") is not None
        ]
        comparisons = {
            "constant": _timing_superiority(dynamic_alpha, constant_alpha),
            "lag16": _timing_superiority(dynamic_alpha, lag16_alpha),
            "null_shift64": _timing_superiority(dynamic_alpha, null64_alpha),
            "lag1": _timing_superiority(
                dynamic_alpha,
                [_path_alpha(row, "lags", "1") for row in values],
            ),
            "null_shift1": _timing_superiority(
                dynamic_alpha,
                [_path_alpha(row, "nulls", "1") for row in values],
            ),
            "null_shift16": _timing_superiority(
                dynamic_alpha,
                [_path_alpha(row, "nulls", "16") for row in values],
            ),
        }

        def beats(name: str) -> bool:
            comparison = comparisons[name]
            median = comparison["median_alpha_difference_pt"]
            win_rate = comparison["win_rate"]
            return bool(
                median is not None
                and float(median) > TIMING_SUPERIORITY_MARGIN_PT
                and win_rate is not None
                and float(win_rate) >= TIMING_MIN_WIN_RATE
            )

        median_timing = _safe_median(timing_increment)
        median_alpha = _safe_median(dynamic_alpha)
        criteria = {
            "complete_development_folds": bool(complete_folds),
            "unique_development_folds": bool(
                len(folds) == len(set(folds)) and set(folds) == set(WAVE_FOLDS)
            ),
            "forecast_quality_stable": bool(
                quality_positive >= max(2, (len(values) + 1) // 2)
            ),
            "median_timing_increment_positive": bool(
                median_timing is not None and float(median_timing) > 0.0
            ),
            "median_alpha_excess_positive": bool(
                median_alpha is not None and float(median_alpha) > 0.0
            ),
            "timing_beats_constant": beats("constant"),
            "timing_beats_lag16": beats("lag16"),
            "timing_beats_null_shift64": beats("null_shift64"),
            "dd_turnover_tradeoff": bool(
                _safe_median(dynamic_dd) is not None
                and _safe_median(constant_dd) is not None
                and _safe_median(turnover) is not None
                and float(_safe_median(dynamic_dd))
                <= float(_safe_median(constant_dd)) + 0.05
                and float(_safe_median(turnover)) <= 6.5
            ),
        }
        criteria_pass = bool(all(criteria.values()))
        promotion_status = (
            "blocked_by_data_quality"
            if feature_set == "full17"
            else "eligible_for_promotion"
        )
        gate_status = (
            "blocked_by_data_quality"
            if feature_set == "full17"
            else ("pass" if criteria_pass else "fail")
        )
        failure_reasons = [name for name, passed in criteria.items() if not passed]
        if feature_set == "full17":
            failure_reasons.append("full17 availability mask is absent")
        output.append(
            {
                "feature_set": feature_set,
                "candidate": candidate,
                "folds": int(len(values)),
                "fold_ids": list(folds),
                "forecast_quality_metric": quality_metric,
                "test_median_forecast_quality": _safe_median(quality_values),
                "test_positive_quality_folds": int(quality_positive),
                "test_median_alpha_excess_pt": median_alpha,
                "test_median_timing_increment_alpha_excess_pt": median_timing,
                "timing_superiority": comparisons,
                "test_median_maxdd_delta_pt": _safe_median(dynamic_dd),
                "test_median_constant_maxdd_delta_pt": _safe_median(constant_dd),
                "test_median_turnover": _safe_median(turnover),
                "timing_superiority_margin_pt": float(TIMING_SUPERIORITY_MARGIN_PT),
                "timing_min_win_rate": float(TIMING_MIN_WIN_RATE),
                "criteria": criteria,
                "status": gate_status,
                "criteria_pass": criteria_pass,
                "promotion_status": promotion_status,
                "selection_contract": "validation-only horizon/policy; development test report-only",
                "failure_reasons": failure_reasons,
            }
        )
    return output


def _missing_gate_record(feature_set: str, candidate: str) -> dict[str, Any]:
    """Emit an explicit fail-closed gate row when every fold fit failed."""
    criteria = {
        "complete_development_folds": False,
        "unique_development_folds": False,
        "forecast_quality_stable": False,
        "median_timing_increment_positive": False,
        "median_alpha_excess_positive": False,
        "timing_beats_constant": False,
        "timing_beats_lag16": False,
        "timing_beats_null_shift64": False,
        "dd_turnover_tradeoff": False,
    }
    blocked = feature_set == "full17"
    return {
        "feature_set": feature_set,
        "candidate": candidate,
        "folds": 0,
        "fold_ids": [],
        "forecast_quality_metric": "auc" if candidate == CLASSIFIER_CANDIDATE else "spearman_ic",
        "test_median_forecast_quality": None,
        "test_positive_quality_folds": 0,
        "test_median_alpha_excess_pt": None,
        "test_median_timing_increment_alpha_excess_pt": None,
        "timing_superiority": {
            name: {
                "median_alpha_difference_pt": None,
                "win_folds": 0,
                "win_rate": None,
                "margin_pt": float(TIMING_SUPERIORITY_MARGIN_PT),
                "required_win_rate": float(TIMING_MIN_WIN_RATE),
            }
            for name in ("constant", "lag16", "null_shift64", "lag1", "null_shift1", "null_shift16")
        },
        "test_median_maxdd_delta_pt": None,
        "test_median_constant_maxdd_delta_pt": None,
        "test_median_turnover": None,
        "timing_superiority_margin_pt": float(TIMING_SUPERIORITY_MARGIN_PT),
        "timing_min_win_rate": float(TIMING_MIN_WIN_RATE),
        "criteria": criteria,
        "criteria_pass": False,
        "promotion_status": "blocked_by_data_quality" if blocked else "eligible_for_promotion",
        "status": "blocked_by_data_quality" if blocked else "fail",
        "selection_contract": "validation-only horizon/policy; development test report-only",
        "failure_reasons": ["no successful candidate rows for any required development fold"],
    }


def _slice_frame(frame: pd.DataFrame | pd.Series, start: pd.Timestamp, end: pd.Timestamp):
    mask = (frame.index >= start) & (frame.index < end)
    selected = frame.loc[np.asarray(mask, dtype=bool)]
    if len(selected) == 0:
        raise ValueError(f"empty right-exclusive slice [{start}, {end})")
    return selected


def _fit_regression_candidate(
    candidate: str,
    train_context: pd.DataFrame,
    train_returns: np.ndarray,
    *,
    horizons: tuple[int, ...],
    seed: int,
    max_fit_rows: int,
):
    if candidate == "context_ridge_risk_adjusted":
        model_kind = "ridge"
    elif candidate == "context_histgb_risk_adjusted":
        model_kind = "histgb"
    else:
        raise ValueError(f"not a regression candidate: {candidate}")
    values = train_context.to_numpy(dtype=np.float64)
    finite_x = np.isfinite(values).all(axis=1)
    selected_indices = (
        np.arange(len(train_context), dtype=np.int64)
        if len(train_context) <= max_fit_rows
        else np.linspace(0, len(train_context) - 1, max_fit_rows, dtype=np.int64)
    )
    return_targets, return_masks = _future_regression_targets(
        train_returns,
        train_context.index,
        horizons,
        target_kind="return",
    )
    risk_targets, risk_masks = _future_regression_targets(
        train_returns,
        train_context.index,
        horizons,
        target_kind="risk",
    )
    return_models: dict[int, Any] = {}
    risk_models: dict[int, Any] = {}
    for column, horizon in enumerate(horizons):
        valid_return = selected_indices[return_masks[selected_indices, column] & finite_x[selected_indices]]
        valid_risk = selected_indices[risk_masks[selected_indices, column] & finite_x[selected_indices]]
        if len(valid_return) < 2 or len(valid_risk) < 2:
            raise ValueError(f"{candidate} has insufficient contiguous train rows at horizon {horizon}")
        if model_kind == "ridge":
            params = _candidate_hyperparameters(candidate)
            return_model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=float(params["alpha"]), solver="lsqr", max_iter=1000, tol=1e-8),
            )
            risk_model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=float(params["alpha"]), solver="lsqr", max_iter=1000, tol=1e-8),
            )
        else:
            params = _candidate_hyperparameters(candidate)
            common = {
                "learning_rate": float(params["learning_rate"]),
                "max_iter": int(params["max_iter"]),
                "max_leaf_nodes": int(params["max_leaf_nodes"]),
                "l2_regularization": float(params["l2_regularization"]),
                "early_stopping": False,
                "random_state": int(seed),
            }
            return_model = HistGradientBoostingRegressor(**common)
            risk_model = HistGradientBoostingRegressor(**common)
        return_model.fit(values[valid_return], return_targets[valid_return, column])
        risk_model.fit(values[valid_risk], risk_targets[valid_risk, column])
        return_models[horizon] = return_model
        risk_models[horizon] = risk_model
    return ContextRegressionCandidate(
        name=candidate,
        horizons=horizons,
        return_models=return_models,
        risk_models=risk_models,
        fit_rows=int(len(selected_indices)),
        dropped_nonfinite_rows=int(np.sum(~finite_x[selected_indices])),
    )


def _metric_records_for_regression(
    prediction: Mapping[str, np.ndarray],
    returns: np.ndarray,
    timestamps: Sequence[Any],
    horizons: tuple[int, ...],
    *,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return_targets, return_mask = _future_regression_targets(
        returns,
        timestamps,
        horizons,
        target_kind="return",
    )
    risk_targets, risk_mask = _future_regression_targets(
        returns,
        timestamps,
        horizons,
        target_kind="risk",
    )
    return (
        forecast_metrics(
            prediction["return"],
            return_targets,
            return_mask,
            horizons,
            split=split,
            target_kind="return",
        ),
        forecast_metrics(
            prediction["risk"],
            risk_targets,
            risk_mask,
            horizons,
            split=split,
            target_kind="risk",
        ),
    )


def _split_eligibility_summary(
    context: pd.DataFrame,
    returns: np.ndarray,
    timestamps: Sequence[Any],
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    """Summarize finite-context and contiguous-target eligibility for one split."""
    timestamp_index = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if len(context) != len(returns) or len(context) != len(timestamp_index):
        raise ValueError("split eligibility inputs must have equal lengths")
    context_finite = np.isfinite(context.to_numpy(dtype=np.float64)).all(axis=1)
    transition_gaps = (
        int(np.sum(np.diff(timestamp_index.asi8) != int(BAR_INTERVAL.value)))
        if len(timestamp_index) > 1
        else 0
    )
    timestamp_target_mask = future_target_eligibility_mask(timestamp_index, horizons)
    return_target_values, return_raw_mask = future_targets(
        returns,
        horizons,
        target_kind="return",
    )
    risk_target_values, risk_raw_mask = future_targets(
        returns,
        horizons,
        target_kind="risk",
    )
    # The values are intentionally computed here even though the summary only
    # stores counts: this validates that every target contract has the same
    # horizon ordering and that no gap is silently treated as adjacency.
    if return_target_values.shape != risk_target_values.shape:
        raise ValueError("return/risk target shape mismatch")

    def target_counts(raw_mask: np.ndarray) -> dict[str, Any]:
        eligible = raw_mask & timestamp_target_mask
        gap_excluded = raw_mask & ~timestamp_target_mask
        return {
            "valid_rows_by_horizon": {
                str(horizon): int(eligible[:, column].sum())
                for column, horizon in enumerate(horizons)
            },
            "excluded_rows_by_horizon": {
                str(horizon): int((~eligible[:, column]).sum())
                for column, horizon in enumerate(horizons)
            },
            "gap_excluded_rows_by_horizon": {
                str(horizon): int(gap_excluded[:, column].sum())
                for column, horizon in enumerate(horizons)
            },
        }

    return {
        "rows": int(len(context)),
        "context_eligible_rows": int(context_finite.sum()),
        "context_ineligible_rows": int((~context_finite).sum()),
        "non_15m_transitions": transition_gaps,
        "target_eligibility": {
            "return": target_counts(return_raw_mask),
            "risk": target_counts(risk_raw_mask),
        },
    }


def _strict_nonnegative_ints(values: Iterable[int], label: str) -> tuple[int, ...]:
    parsed = tuple(values)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        for value in parsed
    ):
        raise ValueError(f"{label} must contain only non-negative integers")
    return tuple(int(value) for value in parsed)


def _apply_execution_delay(positions: np.ndarray, delay: int) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64).reshape(-1)
    delay = validate_execution_delay(delay)
    if delay == 0 or len(values) == 0:
        return values.copy()
    d = min(delay, len(values))
    return np.concatenate([np.full(d, values[0]), values[:-d]])


def aligned_timing_economics(
    returns: np.ndarray,
    dynamic_positions: np.ndarray,
    constant_exposure: float,
    cfg: Mapping[str, Any],
    *,
    benchmark: float = BENCHMARK_POSITION,
    execution_delay: int = FIXED_EXECUTION_DELAY_BARS,
    lags: Iterable[int] = TIMING_LAGS,
    null_shifts: Iterable[int] = TIMING_NULL_SHIFTS,
) -> dict[str, Any]:
    """Compare timing paths on one common right-side return window.

    Each path is delayed first, then all paths are trimmed from the maximum
    effective delay. This prevents AlphaEx differences from including a
    different number of leading/padded return bars.
    """
    returns_arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    dynamic_arr = np.asarray(dynamic_positions, dtype=np.float64).reshape(-1)
    if len(returns_arr) != len(dynamic_arr) or len(returns_arr) == 0:
        raise ValueError("timing returns and dynamic positions must be non-empty and aligned")
    validated_delay = validate_execution_delay(execution_delay)
    lag_values = _strict_nonnegative_ints(lags, "timing lags")
    null_values = _strict_nonnegative_ints(null_shifts, "timing null shifts")
    effective_delays = (validated_delay, *(validated_delay + lag for lag in lag_values))
    common_start = max(effective_delays)
    if common_start >= len(returns_arr):
        raise ValueError(
            "timing comparison requires more return bars than the maximum effective delay"
        )

    def metrics(positions: np.ndarray, delay: int) -> dict[str, Any]:
        effective = _apply_execution_delay(positions, delay)
        window_positions = effective[common_start:]
        previous_position = (
            float(effective[common_start - 1]) if common_start > 0 else None
        )
        result = backtest_metrics(
            returns_arr[common_start:],
            window_positions,
            cfg,
            benchmark=benchmark,
            execution_delay_bars=0,
            initial_position=previous_position,
        )
        result["evaluation_initial_position"] = previous_position
        return result

    dynamic = metrics(dynamic_arr, validated_delay)
    constant = metrics(np.full(len(returns_arr), constant_exposure, dtype=np.float64), validated_delay)
    lags_result = {
        str(lag): metrics(dynamic_arr, validated_delay + lag)
        for lag in lag_values
    }
    nulls_result = {
        str(shift): metrics(circular_shift_positions(dynamic_arr, shift), validated_delay)
        for shift in null_values
    }
    return {
        "common_evaluation_start_bars": int(common_start),
        "common_evaluation_rows": int(len(returns_arr) - common_start),
        "operational_execution_delay_bars": validated_delay,
        "dynamic": dynamic,
        "constant": constant,
        "lags": lags_result,
        "nulls": nulls_result,
        "timing_increment_alpha_excess_pt": float(
            dynamic["alpha_excess_pt"] - constant["alpha_excess_pt"]
        ),
        "constant_exposure_component_alpha_excess_pt": float(constant["alpha_excess_pt"]),
        "dynamic_alpha_excess_pt": float(dynamic["alpha_excess_pt"]),
    }


def _legacy_centered_positions(
    forecast: Sequence[float] | np.ndarray,
    params: PolicyParams,
    *,
    baseline: float,
    min_position: float = 0.50,
    max_position: float = 1.12,
    max_position_step: float = DEFAULT_MAX_POSITION_STEP,
) -> np.ndarray:
    """Replay the Wave3A sign overlay around a supplied constant baseline."""
    values = np.asarray(forecast, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Wave3A replay forecast contains non-finite values")
    if params.threshold < 0 or params.overlay_magnitude < 0 or not 0 <= params.hysteresis < 1:
        raise ValueError("invalid replay policy parameters")
    positions = np.full(len(values), float(baseline), dtype=np.float64)
    current = float(baseline)
    hold_remaining = 0
    active = False
    for index, value in enumerate(values):
        if hold_remaining > 0:
            hold_remaining -= 1
        else:
            absolute = abs(float(value))
            enter = absolute > float(params.threshold)
            stay = absolute > float(params.threshold) * (1.0 - float(params.hysteresis))
            active = stay if active else enter
            raw_overlay = np.sign(value) * float(params.overlay_magnitude) if active else 0.0
            target = float(np.clip(baseline + raw_overlay, min_position, max_position))
            delta = float(np.clip(target - current, -max_position_step, max_position_step))
            next_position = float(np.clip(current + delta, min_position, max_position))
            if abs(next_position - current) > 1e-12 and params.min_hold > 0:
                hold_remaining = int(params.min_hold)
            current = next_position
        positions[index] = current
    return positions


def replay_wave3a_corrected(
    *,
    data: DevelopmentData,
    cfg: Mapping[str, Any],
    frozen_result_path: str | Path = "docs/forecast_tournament_plan011_dev/result.json",
    max_fit_rows: int = 20_000,
) -> dict[str, Any]:
    """Re-score frozen Wave3A Ridge/HistGB rows with Wave3C alignment.

    This is a report-only compatibility replay.  It never changes Wave3C
    validation selection or the promotion gate, and it deliberately reads
    only the frozen development result artifact.
    """
    source_path = Path(frozen_result_path)
    if not source_path.exists():
        return {
            "status": "N/A",
            "reason": "frozen Wave3A result artifact is not present",
            "source_result_path": str(source_path),
            "rows": [],
        }
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "N/A",
            "reason": f"could not read frozen Wave3A result: {exc}",
            "source_result_path": str(source_path),
            "rows": [],
        }
    source_rows = [
        row
        for row in source.get("rows", [])
        if row.get("feature_set") == "ohlcv13"
        and row.get("candidate")
        in {"causal_trend_vol_rule", "ridge_direct_forecast", "histgb_direct_forecast"}
        and int(row.get("fold", -1)) in WAVE_FOLDS
    ]
    splits_by_fold = {int(split.fold_idx): split for split in data.splits}
    replay_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for source_row in sorted(
        source_rows,
        key=lambda row: (int(row["fold"]), str(row["candidate"])),
    ):
        fold = int(source_row["fold"])
        candidate = str(source_row["candidate"])
        try:
            if fold not in splits_by_fold:
                raise ValueError(f"frozen row fold {fold} is not in current development data")
            policy_data = source_row.get("policy", {})
            execution_delay = validate_execution_delay(policy_data.get("execution_delay"))
            if execution_delay != FIXED_EXECUTION_DELAY_BARS:
                raise ValueError(
                    f"frozen Wave3A row has non-operational delay {execution_delay}; expected {FIXED_EXECUTION_DELAY_BARS}"
                )
            split = splits_by_fold[fold]
            columns = feature_ablation_columns(data.features)["ohlcv13"]
            base_features = data.features.loc[:, columns]
            train_features = _slice_frame(base_features, split.train_start, split.train_end)
            test_features = _slice_frame(base_features, split.test_start, split.test_end)
            train_returns = _slice_frame(
                data.returns, split.train_start, split.train_end
            ).to_numpy(dtype=np.float64)
            test_returns = _slice_frame(
                data.returns, split.test_start, split.test_end
            ).to_numpy(dtype=np.float64)
            fitted = fit_candidate(
                candidate,
                train_features,
                train_returns,
                horizons=(16,),
                hyperparameters=source_row.get("hyperparameters", {}),
                seed=WAVE_SEED,
                max_fit_rows=min(int(max_fit_rows), int(source_row.get("fit_rows", max_fit_rows))),
            )
            prediction = fitted.predict(test_features)["return"][:, 0]
            constant_exposure = float(
                source_row["constant_selection"]["selected_candidate"]
            )
            params = PolicyParams(
                threshold=float(policy_data["threshold"]),
                overlay_magnitude=float(policy_data["overlay_magnitude"]),
                hysteresis=float(policy_data["hysteresis"]),
                min_hold=int(policy_data["min_hold"]),
                execution_delay=execution_delay,
            )
            positions = _legacy_centered_positions(
                prediction,
                params,
                baseline=constant_exposure,
            )
            economics = aligned_timing_economics(
                test_returns,
                positions,
                constant_exposure,
                cfg,
                benchmark=BENCHMARK_POSITION,
                execution_delay=FIXED_EXECUTION_DELAY_BARS,
            )
            replay_rows.append(
                {
                    "status": "ok",
                    "fold": fold,
                    "candidate": candidate,
                    "feature_set": "ohlcv13",
                    "fold_bounds": _fold_bounds(split),
                    "policy_baseline": constant_exposure,
                    "source_policy": policy_data,
                    "source_hyperparameters": source_row.get("hyperparameters", {}),
                    "test_is_report_only": True,
                    "test_economics": economics,
                }
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            failures.append(
                {
                    "status": "N/A",
                    "fold": fold,
                    "candidate": candidate,
                    "reason": str(exc),
                }
            )
    dynamic_alpha = [
        float(row["test_economics"]["dynamic"]["alpha_excess_pt"])
        for row in replay_rows
    ]
    constant_alpha = [
        float(row["test_economics"]["constant"]["alpha_excess_pt"])
        for row in replay_rows
    ]
    return {
        "status": "complete" if replay_rows and not failures else ("partial" if replay_rows else "N/A"),
        "source_result_path": str(source_path),
        "source_commit_hash": source.get("commit_hash"),
        "selection_contract": "report-only corrected replay; excluded from Wave3C selection and gate",
        "comparator_contract": "validation-selected constant exposure, common right-side return window, fixed delay 1",
        "rows": replay_rows,
        "failures": failures,
        "summary": {
            "rows": len(replay_rows),
            "median_dynamic_alpha_excess_pt": _safe_median(dynamic_alpha),
            "median_constant_alpha_excess_pt": _safe_median(constant_alpha),
            "median_timing_increment_alpha_excess_pt": _safe_median(
                [dynamic - constant for dynamic, constant in zip(dynamic_alpha, constant_alpha)]
            ),
        },
    }


def run_context_tournament(
    *,
    data: DevelopmentData,
    cfg: Mapping[str, Any],
    config_path: str,
    seed: int = WAVE_SEED,
    horizons: Iterable[int] = HORIZON_GRID,
    feature_sets: Iterable[str] = CONTEXT_FEATURE_SET_ORDER,
    candidates: Iterable[str] = ALL_CANDIDATES,
    max_fit_rows: int = 20_000,
    output_dir: str | Path = "docs/forecast_context_tournament_plan011_dev",
    wave3a_result_path: str | Path = "docs/forecast_tournament_plan011_dev/result.json",
) -> dict[str, Any]:
    """Run the single formal Wave3C development screen and write its outputs."""
    if int(seed) != WAVE_SEED:
        raise ValueError(f"Wave3C formal screen fixes seed={WAVE_SEED}")
    horizon_values = _strict_horizons(horizons)
    if horizon_values != tuple(HORIZON_GRID):
        raise ValueError(f"Wave3C formal screen fixes horizons={list(HORIZON_GRID)}")
    feature_schema = feature_ablation_columns(data.features)
    feature_set_values = tuple(str(value) for value in feature_sets)
    if feature_set_values != CONTEXT_FEATURE_SET_ORDER:
        raise ValueError(
            f"Wave3C formal screen fixes feature-set order={list(CONTEXT_FEATURE_SET_ORDER)}"
        )
    if any(value not in feature_schema for value in feature_set_values):
        raise ValueError(f"unknown feature ablation in {feature_set_values}")
    candidate_values = tuple(str(value) for value in candidates)
    if candidate_values != ALL_CANDIDATES:
        raise ValueError(f"Wave3C formal screen fixes candidates={list(ALL_CANDIDATES)}")
    folds = _strict_wave_folds(data.splits)
    configured_delay = validate_execution_delay(
        cfg.get("eval", {}).get(
            "forecast_execution_delay_bars", FIXED_EXECUTION_DELAY_BARS
        )
    )
    if configured_delay != FIXED_EXECUTION_DELAY_BARS:
        raise ValueError(
            f"Wave3C requires config execution delay {FIXED_EXECUTION_DELAY_BARS}, got {configured_delay}"
        )
    provenance = _provenance(
        data=data,
        cfg=cfg,
        config_path=config_path,
        seed=seed,
        feature_sets=feature_set_values,
        candidates=candidate_values,
        horizons=horizon_values,
        execution_delay=FIXED_EXECUTION_DELAY_BARS,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ledger_path = output_path / "forecast_context_tournament_ledger.jsonl"
    ledger_records: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    feature_quality_rows: list[dict[str, Any]] = []
    failed_candidates: list[dict[str, Any]] = []
    started = time.perf_counter()

    for feature_set in feature_set_values:
        columns = feature_schema[feature_set]
        base_features = data.features.loc[:, columns]
        context_features = build_causal_context(base_features)
        context_feature_count = int(context_features.shape[1])
        for split in data.splits:
            bounds = _fold_bounds(split)
            train_base = _slice_frame(base_features, split.train_start, split.train_end)
            val_base = _slice_frame(base_features, split.val_start, split.val_end)
            test_base = _slice_frame(base_features, split.test_start, split.test_end)
            train_context = _slice_frame(context_features, split.train_start, split.train_end)
            val_context = _slice_frame(context_features, split.val_start, split.val_end)
            test_context = _slice_frame(context_features, split.test_start, split.test_end)
            train_returns = _slice_frame(data.returns, split.train_start, split.train_end).to_numpy(dtype=np.float64)
            val_returns = _slice_frame(data.returns, split.val_start, split.val_end).to_numpy(dtype=np.float64)
            test_returns = _slice_frame(data.returns, split.test_start, split.test_end).to_numpy(dtype=np.float64)
            train_timestamps = train_context.index
            val_timestamps = val_context.index
            test_timestamps = test_context.index
            eligibility = {
                "train": _split_eligibility_summary(
                    train_context, train_returns, train_timestamps, horizon_values
                ),
                "validation": _split_eligibility_summary(
                    val_context, val_returns, val_timestamps, horizon_values
                ),
                "test": _split_eligibility_summary(
                    test_context, test_returns, test_timestamps, horizon_values
                ),
            }
            for split_name, frame in (
                ("train", train_base),
                ("validation", val_base),
                ("test", test_base),
            ):
                quality = _quality_record(frame, feature_set, split.fold_idx, split_name)
                quality["fold_bounds"] = bounds
                quality["context_feature_count"] = context_feature_count
                quality["eligibility"] = eligibility[split_name]
                feature_quality_rows.append(quality)
                _append_record(
                    ledger_records,
                    provenance,
                    record_type="feature_coverage",
                    status=quality["status"],
                    fold=int(split.fold_idx),
                    feature_set=feature_set,
                    fit_split="none",
                    selection_split="none",
                    report_split=split_name,
                    fold_bounds=bounds,
                    metrics=quality,
                )

            constant_exposure, constant_trials = select_fixed_exposure(
                val_returns,
                CONSTANT_EXPOSURE_GRID,
                cfg,
                benchmark=BENCHMARK_POSITION,
                execution_delay_bars=FIXED_EXECUTION_DELAY_BARS,
            )
            constant_selection = {
                "split": "validation",
                "status": "selected_on_validation",
                "candidate_grid": list(CONSTANT_EXPOSURE_GRID),
                "selected_candidate": float(constant_exposure),
                "selection_score": float(
                    max(constant_trials, key=lambda row: float(row["score"]))["score"]
                ),
            }
            for candidate in candidate_values:
                candidate_started = time.perf_counter()
                try:
                    if candidate in REGRESSION_CANDIDATES:
                        fitted = _fit_regression_candidate(
                            candidate,
                            train_context,
                            train_returns,
                            horizons=horizon_values,
                            seed=seed,
                            max_fit_rows=max_fit_rows,
                        )
                        val_prediction = fitted.predict(val_context)
                        test_prediction = fitted.predict(test_context)
                        validation_scores, return_scales, risk_scales, scale_reasons = risk_adjusted_scores(
                            val_prediction,
                            horizon_values,
                            validation=True,
                        )
                        test_scores, _, _, _ = risk_adjusted_scores(
                            test_prediction,
                            horizon_values,
                            validation=False,
                            return_scales=return_scales,
                            risk_scales=risk_scales,
                        )
                        val_return_metrics, val_risk_metrics = _metric_records_for_regression(
                            val_prediction,
                            val_returns,
                            val_timestamps,
                            horizon_values,
                            split="validation",
                        )
                        test_return_metrics, test_risk_metrics = _metric_records_for_regression(
                            test_prediction,
                            test_returns,
                            test_timestamps,
                            horizon_values,
                            split="development_test",
                        )
                        selected_horizon, policy, policy_selection = select_horizon_and_policy(
                            validation_scores,
                            val_returns,
                            cfg,
                            constant_exposure=constant_exposure,
                            execution_delay=FIXED_EXECUTION_DELAY_BARS,
                        )
                        fit_details = {
                            "model_type": fitted.name,
                            "hyperparameters": _candidate_hyperparameters(candidate),
                            "fit_rows": int(fitted.fit_rows),
                            "fit_rows_dropped_nonfinite": int(fitted.dropped_nonfinite_rows),
                            "validation_return_scales": return_scales,
                            "validation_risk_scales": risk_scales,
                            "validation_scale_reasons": scale_reasons,
                        }
                        validation_classifier_metrics: list[dict[str, Any]] = []
                        test_classifier_metrics: list[dict[str, Any]] = []
                    else:
                        fitted = fit_downside_classifier(
                            train_context,
                            train_returns,
                            timestamps=train_timestamps,
                            horizons=horizon_values,
                            seed=seed,
                            max_fit_rows=max_fit_rows,
                        )
                        val_probabilities = predict_downside_probabilities(fitted, val_context)
                        test_probabilities = predict_downside_probabilities(fitted, test_context)
                        validation_scores = _classifier_scores(
                            val_probabilities,
                            fitted.train_event_rates,
                        )
                        test_scores = _classifier_scores(
                            test_probabilities,
                            fitted.train_event_rates,
                        )
                        val_return_metrics = []
                        val_risk_metrics = []
                        test_return_metrics = []
                        test_risk_metrics = []
                        validation_classifier_metrics = _classifier_target_metrics(
                            val_probabilities,
                            val_returns,
                            fitted,
                            "validation",
                            timestamps=val_timestamps,
                        )
                        test_classifier_metrics = _classifier_target_metrics(
                            test_probabilities,
                            test_returns,
                            fitted,
                            "development_test",
                            timestamps=test_timestamps,
                        )
                        selected_horizon, policy, policy_selection = select_horizon_and_policy(
                            validation_scores,
                            val_returns,
                            cfg,
                            constant_exposure=constant_exposure,
                            execution_delay=FIXED_EXECUTION_DELAY_BARS,
                        )
                        fit_details = {
                            "model_type": CLASSIFIER_CANDIDATE,
                            "hyperparameters": _candidate_hyperparameters(candidate),
                            "fit_rows": int(fitted.fit_rows),
                            "target_thresholds_train": fitted.target_thresholds,
                            "train_event_rates": fitted.train_event_rates,
                            "unavailable_horizons": fitted.unavailable,
                            "probability_threshold_for_diagnostics": CLASSIFIER_PROBABILITY_THRESHOLD,
                        }
                        scale_reasons = {}

                    test_positions = continuous_overlay_positions(
                        test_scores[selected_horizon],
                        policy,
                        benchmark=constant_exposure,
                        eligible_mask=np.isfinite(test_scores[selected_horizon]),
                    )
                    economics = aligned_timing_economics(
                        test_returns,
                        test_positions,
                        constant_exposure,
                        cfg,
                        benchmark=BENCHMARK_POSITION,
                        execution_delay=FIXED_EXECUTION_DELAY_BARS,
                        null_shifts=TIMING_NULL_SHIFTS,
                    )
                    row = {
                        **provenance,
                        "schema_version": SCHEMA_VERSION,
                        "record_type": "forecast_context_candidate",
                        "status": "ok",
                        "candidate": candidate,
                        "feature_set": feature_set,
                        "fold": int(split.fold_idx),
                        "fold_bounds": bounds,
                        "fit_split": "train",
                        "selection_split": "validation",
                        "report_split": "development_test",
                        "test_is_report_only": True,
                        "context_feature_count": context_feature_count,
                        "eligibility": eligibility,
                        "selected_horizon": int(selected_horizon),
                        "policy_baseline": float(constant_exposure),
                        "policy": {
                            "threshold": float(policy.threshold),
                            "overlay_magnitude": float(policy.overlay_magnitude),
                            "hysteresis": float(policy.hysteresis),
                            "min_hold": int(policy.min_hold),
                            "execution_delay": int(policy.execution_delay),
                        },
                        "fit": fit_details,
                        "constant_selection": constant_selection,
                        "validation_policy_selection": policy_selection,
                        "validation_return_metrics": val_return_metrics,
                        "validation_risk_metrics": val_risk_metrics,
                        "test_return_metrics": test_return_metrics,
                        "test_risk_metrics": test_risk_metrics,
                        "validation_classifier_metrics": validation_classifier_metrics,
                        "test_classifier_metrics": test_classifier_metrics,
                        "test_economics": economics,
                        "runtime_seconds": float(time.perf_counter() - candidate_started),
                    }
                    selected_rows.append(row)
                    ledger_records.append(row)
                    for metric_group, metric_rows in (
                        ("return", val_return_metrics),
                        ("risk", val_risk_metrics),
                        ("classifier_validation", validation_classifier_metrics),
                        ("classifier_test", test_classifier_metrics),
                        ("return_test", test_return_metrics),
                        ("risk_test", test_risk_metrics),
                    ):
                        for metric_row in metric_rows:
                            _append_record(
                                ledger_records,
                                provenance,
                                record_type="forecast_metric",
                                candidate=candidate,
                                feature_set=feature_set,
                                fold=int(split.fold_idx),
                                fold_bounds=bounds,
                                fit_split="train",
                                selection_split="validation",
                                report_split=(
                                    "validation"
                                    if metric_group in {"return", "risk", "classifier_validation"}
                                    else "development_test"
                                ),
                                test_is_report_only=metric_group not in {
                                    "return",
                                    "risk",
                                    "classifier_validation",
                                },
                                metric_group=metric_group,
                                **metric_row,
                            )
                    for method, metrics in (
                        [("dynamic", economics["dynamic"]),
                         ("constant_validation_selected", economics["constant"])]
                        + [(f"lag_{key}", metrics) for key, metrics in economics["lags"].items()]
                        + [(f"null_shift_{key}", metrics) for key, metrics in economics["nulls"].items()]
                    ):
                        _append_record(
                            ledger_records,
                            provenance,
                            record_type="economic_metric",
                            status="ok",
                            candidate=candidate,
                            feature_set=feature_set,
                            fold=int(split.fold_idx),
                            method=method,
                            fold_bounds=bounds,
                            fit_split="train",
                            selection_split="validation",
                            report_split="development_test",
                            test_is_report_only=True,
                            metrics=metrics,
                        )
                    _append_record(
                        ledger_records,
                        provenance,
                        record_type="timing_attribution",
                        status="ok",
                        candidate=candidate,
                        feature_set=feature_set,
                        fold=int(split.fold_idx),
                        fold_bounds=bounds,
                        fit_split="train",
                        selection_split="validation",
                        report_split="development_test",
                        test_is_report_only=True,
                        metrics={
                            "constant_exposure_component_alpha_excess_pt": economics[
                                "constant_exposure_component_alpha_excess_pt"
                            ],
                            "timing_increment_alpha_excess_pt": economics[
                                "timing_increment_alpha_excess_pt"
                            ],
                            "dynamic_alpha_excess_pt": economics["dynamic_alpha_excess_pt"],
                            "lag1": economics["lags"]["1"],
                            "lag16": economics["lags"]["16"],
                            "null_shift1": economics["nulls"]["1"],
                            "null_shift16": economics["nulls"]["16"],
                            "null_shift64": economics["nulls"]["64"],
                        },
                    )
                except (KeyError, OverflowError, ValueError, RuntimeError) as exc:
                    failure = {
                        "candidate": candidate,
                        "feature_set": feature_set,
                        "fold": int(split.fold_idx),
                        "status": "N/A",
                        "reason": str(exc),
                        "runtime_seconds": float(time.perf_counter() - candidate_started),
                    }
                    failed_candidates.append(failure)
                    _append_record(
                        ledger_records,
                        provenance,
                        record_type="forecast_context_candidate",
                        status="N/A",
                        candidate=candidate,
                        feature_set=feature_set,
                        fold=int(split.fold_idx),
                        fold_bounds=bounds,
                        fit_split="train",
                        selection_split="validation",
                        report_split="development_test",
                        test_is_report_only=True,
                        reason=str(exc),
                        runtime_seconds=failure["runtime_seconds"],
                    )

    gate_rows = aggregate_context_gate(selected_rows)
    gate_keys = {(str(row["feature_set"]), str(row["candidate"])) for row in gate_rows}
    for feature_set in feature_set_values:
        for candidate in candidate_values:
            key = (feature_set, candidate)
            if key not in gate_keys:
                gate_rows.append(_missing_gate_record(feature_set, candidate))
    gate_rows.sort(key=lambda row: (str(row["feature_set"]), str(row["candidate"])))
    for gate in gate_rows:
        _append_record(
            ledger_records,
            provenance,
            record_type="successive_halving_gate",
            fold=None,
            fit_split="train",
            selection_split="validation",
            report_split="development_test",
            test_is_report_only=True,
            **gate,
        )
    next_wave = [
        {"feature_set": row["feature_set"], "candidate": row["candidate"]}
        for row in sorted(
            gate_rows,
            key=lambda item: float(item["test_median_alpha_excess_pt"] or -np.inf),
            reverse=True,
        )
        if row["status"] == "pass" and row["feature_set"] == "ohlcv13"
    ]
    wave3a_replay = replay_wave3a_corrected(
        data=data,
        cfg=cfg,
        frozen_result_path=wave3a_result_path,
        max_fit_rows=max_fit_rows,
    )
    for replay_row in wave3a_replay.get("rows", []):
        _append_record(
            ledger_records,
            provenance,
            record_type="wave3a_corrected_replay",
            status=replay_row.get("status", "N/A"),
            candidate=replay_row.get("candidate"),
            feature_set=replay_row.get("feature_set", "ohlcv13"),
            fold=replay_row.get("fold"),
            fit_split="train",
            selection_split="none",
            report_split="development_test",
            test_is_report_only=True,
            metrics=replay_row.get("test_economics"),
            comparator_contract=wave3a_replay.get("comparator_contract"),
        )
    for replay_failure in wave3a_replay.get("failures", []):
        _append_record(
            ledger_records,
            provenance,
            record_type="wave3a_corrected_replay",
            status="N/A",
            candidate=replay_failure.get("candidate"),
            feature_set="ohlcv13",
            fold=replay_failure.get("fold"),
            fit_split="train",
            selection_split="none",
            report_split="development_test",
            test_is_report_only=True,
            reason=replay_failure.get("reason"),
        )
    runtime = float(time.perf_counter() - started)
    result = {
        **provenance,
        "schema_version": SCHEMA_VERSION,
        "record_type": "forecast_context_tournament_report",
        "status": "complete",
        "folds": list(folds),
        "fold_count": len(folds),
        "horizons": list(horizon_values),
        "feature_sets": list(feature_set_values),
        "candidates": list(candidate_values),
        "context_lags": list(CONTEXT_LAGS),
        "context_rolling_windows": list(CONTEXT_WINDOWS),
        "context_rolling_stats": list(CONTEXT_STATS),
        "execution_delay_bars": FIXED_EXECUTION_DELAY_BARS,
        "timing_lags": list(TIMING_LAGS),
        "null_shifts": list(TIMING_NULL_SHIFTS),
        "policy_baseline_contract": "validation-selected constant exposure; dynamic overlay is centered on that baseline",
        "wave3a_comparison": "Wave3A centered dynamic paths on B&H=1.0; Wave3C uses the validation-selected constant as the overlay baseline so timing increment is exposure-matched.",
        "full17_promotion_status": "blocked_by_data_quality: cache has no availability mask, so external zero and missing/imputed values cannot be distinguished",
        "formal_promotion_feature_set": "ohlcv13",
        "wave3a_corrected_replay": wave3a_replay,
        "wave3a_result_frozen": True,
        "selection_contract": "fit train only; select horizon, risk scale, threshold, overlay, hysteresis, and minimum hold on validation; development test report-only",
        "rows": selected_rows,
        "feature_quality": feature_quality_rows,
        "failed_candidates": failed_candidates,
        "gate": gate_rows,
        "next_wave_candidates": next_wave,
        "all_candidates_failed_gate": not bool(next_wave),
        "runtime_seconds": runtime,
        "ledger_path": str(ledger_path),
    }
    ledger_records.insert(
        0,
        {
            **provenance,
            "schema_version": SCHEMA_VERSION,
            "record_type": "forecast_context_tournament_run",
            "status": "complete",
            "fold": None,
            "fit_split": "train",
            "selection_split": "validation",
            "report_split": "development_test",
            "test_is_report_only": True,
            "runtime_seconds": runtime,
            "config": {
                "folds": list(folds),
                "horizons": list(horizon_values),
                "feature_sets": list(feature_set_values),
                "candidates": list(candidate_values),
                "max_fit_rows": int(max_fit_rows),
                "context_lags": list(CONTEXT_LAGS),
                "context_rolling_windows": list(CONTEXT_WINDOWS),
                "risk_penalty": float(FIXED_RISK_PENALTY),
                "downside_label_quantile": float(TRAIN_DOWNSIDE_QUANTILE),
                "execution_delay_bars": FIXED_EXECUTION_DELAY_BARS,
            },
        },
    )
    with ledger_path.open("w", encoding="utf-8") as handle:
        for record in ledger_records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    result_path = output_path / "result.json"
    report_path = output_path / "report.md"
    errata_path = output_path / "wave3a_errata.md"
    result["result_path"] = str(result_path)
    result["report_path"] = str(report_path)
    result["wave3a_errata_path"] = str(errata_path)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(build_report_markdown(result), encoding="utf-8")
    errata_path.write_text(build_wave3a_errata_markdown(result), encoding="utf-8")
    return result


def _report_number(value: Any, digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(parsed):
        return "N/A"
    return f"{parsed:+.{digits}f}{suffix}"


def _report_metric(
    records: Sequence[Mapping[str, Any]], horizon: int, metric: str
) -> tuple[Any, str | None]:
    for record in records:
        if int(record.get("horizon", -1)) == int(horizon):
            metrics = record.get("metrics", {})
            return metrics.get(metric), record.get("reason")
    return None, "horizon metric record is unavailable"


def build_wave3a_errata_markdown(payload: Mapping[str, Any]) -> str:
    """Describe why the frozen Wave3A result is superseded by Wave3C."""
    replay = payload.get("wave3a_corrected_replay", {})
    summary = replay.get("summary", {})
    lines = [
        "# Wave3A frozen-result errata",
        "",
        "Wave3A's original artifact is intentionally left unchanged. It is retained as a historical development screen only.",
        "",
        "The original Wave3A comparison centered the dynamic path on B&H=1.0 while its constant comparator was selected independently on validation (for example, 0.5 or 1.12). Its reported timing increment therefore mixed temporal timing with an exposure-level difference. Its fold/gate robustness contract also predated the strict Wave3C checks.",
        "",
        "Wave3C is the formal superseding screen: it uses the validation-selected constant exposure as the overlay baseline, fixed execution delay 1, a common right-side return window for dynamic/constant/lag/null paths, exact development folds `{0, 2, 8}`, and test results only for the report-only development tournament.",
        "",
        f"Frozen source artifact: `{replay.get('source_result_path', 'N/A')}`",
        f"Frozen source commit: `{replay.get('source_commit_hash', 'N/A')}`",
        f"Corrected replay status: `{replay.get('status', 'N/A')}`; rows={summary.get('rows', 0)}; failures={len(replay.get('failures', []))}",
        f"Corrected replay median dynamic AlphaEx: `{_report_number(summary.get('median_dynamic_alpha_excess_pt'), suffix='pt')}`",
        f"Corrected replay median constant AlphaEx: `{_report_number(summary.get('median_constant_alpha_excess_pt'), suffix='pt')}`",
        f"Corrected replay median timing increment: `{_report_number(summary.get('median_timing_increment_alpha_excess_pt'), suffix='pt')}`",
        "",
        "The corrected replay covers the frozen causal trend+vol rule and Ridge/HistGB rows. It is report-only and is not used to select a Wave3C candidate, threshold, horizon, or next-wave promotion.",
    ]
    if replay.get("failures"):
        lines.extend(["", "Replay failures:"])
        lines.extend(
            f"- fold {item.get('fold')}, {item.get('candidate')}: {item.get('reason', 'N/A')}"
            for item in replay["failures"]
        )
    return "\n".join(lines) + "\n"


def build_report_markdown(payload: Mapping[str, Any]) -> str:
    """Render a self-contained human-readable Wave3C report."""
    lines = [
        "# Plan011 Wave3C causal-context forecast tournament",
        "",
        "## Scope and fixed contracts",
        "",
        f"- development folds only: `{payload.get('folds', [])}` (exactly once each)",
        f"- seed: `{payload.get('seed', 'N/A')}`; horizons: `{payload.get('horizons', [])}`",
        f"- causal context lags: `{payload.get('context_lags', [])}`; rolling windows: `{payload.get('context_rolling_windows', [])}`; statistics: `{payload.get('context_rolling_stats', [])}`",
        f"- commit: `{payload.get('commit_hash', 'N/A')}`",
        f"- config SHA-256: `{payload.get('config_sha256', 'N/A')}`",
        f"- data SHA-256: `{payload.get('data_sha256', 'N/A')}`; source SHA-256: `{payload.get('source_sha256', 'N/A')}`",
        f"- fixed operational execution delay: `{payload.get('execution_delay_bars', 'N/A')}` bar; sensitivity lags: `{payload.get('timing_lags', [])}`; deterministic null shifts: `{payload.get('null_shifts', [])}`",
        "",
        "Feature rows are pre-shifted and context uses only prior rows. Direct targets exclude the current reward and use `t+1..t+h`; therefore the primary economic contract applies a decision at `t` to `returns[t+1]` with delay 1. Timestamp gaps invalidate context/target windows but do not delete returns; ineligible policy rows emit the selected constant baseline.",
        "",
        "Candidates are fixed before execution: Ridge and HistGradientBoosting direct return/risk regressors over causal lag/rolling context, plus a train-quantile future-downside classifier. Validation alone selects horizon, validation scales, constant exposure, threshold, overlay magnitude, hysteresis, and minimum hold. Development test is report-only and is used only as the explicitly labeled development tournament screen.",
        "",
        "## Successive-halving gate",
        "",
        "A candidate must have exact folds, stable forecast quality on at least two of three folds, positive median AlphaEx, positive median dynamic-minus-constant timing increment, and must beat the same-baseline constant, lag16, and shift64 temporal-destruction null by the fixed 0.02pt margin on at least two folds. Lag1, shift1, and shift16 remain robustness diagnostics. The median MaxDDDelta must be no more than 0.05pt worse than the constant and dynamic turnover must be at most 6.5. Full17 can be measured but is never promotion-eligible because the cache has no availability mask.",
        "",
        "| feature set | candidate | folds | median quality | quality+ folds | median AlphaEx | Δconstant | Δlag16 | Δshift64 | median timing increment | median MaxDDDelta | median turnover | status |",
        "|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---|",
    ]
    for gate in payload.get("gate", []):
        def comparison_text(name: str) -> str:
            comparison = gate.get("timing_superiority", {}).get(name, {})
            median = _report_number(comparison.get("median_alpha_difference_pt"), suffix="pt")
            wins = comparison.get("win_folds", 0)
            rate = _report_number(comparison.get("win_rate"), digits=0, suffix="%")
            if rate != "N/A":
                rate = _report_number(float(comparison.get("win_rate")) * 100.0, digits=0, suffix="%")
            return f"{median}; {wins}/{rate}"

        lines.append(
            "| {feature_set} | {candidate} | {folds} | {quality} | {quality_folds} | {alpha} | {constant} | {lag16} | {shift64} | {timing} | {dd} | {turnover} | {status} |".format(
                feature_set=gate.get("feature_set", "N/A"),
                candidate=gate.get("candidate", "N/A"),
                folds=gate.get("folds", 0),
                quality=_report_number(gate.get("test_median_forecast_quality")),
                quality_folds=gate.get("test_positive_quality_folds", 0),
                alpha=_report_number(gate.get("test_median_alpha_excess_pt"), suffix="pt"),
                constant=comparison_text("constant"),
                lag16=comparison_text("lag16"),
                shift64=comparison_text("null_shift64"),
                timing=_report_number(gate.get("test_median_timing_increment_alpha_excess_pt"), suffix="pt"),
                dd=_report_number(gate.get("test_median_maxdd_delta_pt"), suffix="pt"),
                turnover=_report_number(gate.get("test_median_turnover")),
                status=gate.get("status", "N/A"),
            )
        )
    lines.extend(
        [
            "",
            f"Next-wave candidates (OHLCV13 only; development tournament screen): `{payload.get('next_wave_candidates', [])}`",
            f"Formal gate result: `{'all candidates failed' if payload.get('all_candidates_failed_gate') else 'one or more candidates passed'}`",
            "",
            "## Selected per-fold paths",
            "",
            "| fold | feature set | candidate | baseline | horizon | constant AlphaEx | timing increment | dynamic AlphaEx | MaxDDDelta | turnover |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(payload.get("rows", []), key=lambda item: (int(item.get("fold", -1)), str(item.get("feature_set")), str(item.get("candidate")))):
        economics = row.get("test_economics", {})
        dynamic = economics.get("dynamic", {})
        constant = economics.get("constant", {})
        lines.append(
            "| {fold} | {feature_set} | {candidate} | {baseline} | {horizon} | {constant_alpha} | {increment} | {dynamic_alpha} | {dd} | {turnover} |".format(
                fold=row.get("fold", "N/A"),
                feature_set=row.get("feature_set", "N/A"),
                candidate=row.get("candidate", "N/A"),
                baseline=_report_number(row.get("policy_baseline"), digits=3),
                horizon=row.get("selected_horizon", "N/A"),
                constant_alpha=_report_number(constant.get("alpha_excess_pt"), suffix="pt"),
                increment=_report_number(economics.get("timing_increment_alpha_excess_pt"), suffix="pt"),
                dynamic_alpha=_report_number(dynamic.get("alpha_excess_pt"), suffix="pt"),
                dd=_report_number(dynamic.get("maxdd_delta_pt"), suffix="pt"),
                turnover=_report_number(dynamic.get("turnover")),
            )
        )

    lines.extend(
        [
            "",
            "## Forecast and downside diagnostics",
            "",
            "Regression metrics are MAE, RMSE, Spearman IC, and return sign accuracy. Risk sign accuracy is N/A because realized risk is one-sided. The classifier reports AUC, Brier, precision, and recall where the split has finite labels; AUC is N/A for one-class splits with the reason preserved in the ledger.",
            "",
            "| fold | feature set | candidate | split | horizon | target | quality | MAE | RMSE | sign/AUC | Brier | precision | recall | reason |",
            "|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(payload.get("rows", []), key=lambda item: (int(item.get("fold", -1)), str(item.get("feature_set")), str(item.get("candidate")))):
        candidate = row.get("candidate")
        metric_sets = (
            ("validation", row.get("validation_return_metrics", []), "return"),
            ("development_test", row.get("test_return_metrics", []), "return"),
            ("validation", row.get("validation_risk_metrics", []), "risk"),
            ("development_test", row.get("test_risk_metrics", []), "risk"),
        )
        if candidate == CLASSIFIER_CANDIDATE:
            metric_sets = (
                ("validation", row.get("validation_classifier_metrics", []), "downside_event"),
                ("development_test", row.get("test_classifier_metrics", []), "downside_event"),
            )
        for split, records, target in metric_sets:
            for record in records:
                metrics = record.get("metrics", {})
                quality = metrics.get("auc") if target == "downside_event" else metrics.get("spearman_ic")
                sign_or_auc = metrics.get("auc") if target == "downside_event" else (metrics.get("sign_accuracy") if target == "return" else None)
                lines.append(
                    "| {fold} | {feature_set} | {candidate} | {split} | {horizon} | {target} | {quality} | {mae} | {rmse} | {sign_auc} | {brier} | {precision} | {recall} | {reason} |".format(
                        fold=row.get("fold", "N/A"),
                        feature_set=row.get("feature_set", "N/A"),
                        candidate=candidate,
                        split=split,
                        horizon=record.get("horizon", "N/A"),
                        target=target,
                        quality=_report_number(quality),
                        mae=_report_number(metrics.get("mae")),
                        rmse=_report_number(metrics.get("rmse")),
                        sign_auc=_report_number(sign_or_auc),
                        brier=_report_number(metrics.get("brier")),
                        precision=_report_number(metrics.get("precision")),
                        recall=_report_number(metrics.get("recall")),
                        reason=record.get("reason", ""),
                    )
                )

    lines.extend(
        [
            "",
            "## Feature coverage and timestamp eligibility",
            "",
            "External coverage counts deliberately separate finite zero, finite nonzero, and missing values. Because the current cache has no availability mask, a zero cannot be asserted to be observed rather than imputed; full17 is therefore secondary and `blocked_by_data_quality`.",
            "",
            "| fold | feature set | split | rows | external nonzero/missing | context excluded | non-15m transitions | return valid h4/h16/h64 | return gap-excluded h4/h16/h64 | promotion |",
            "|---:|---|---|---:|---|---:|---:|---|---|---|",
        ]
    )
    for quality in payload.get("feature_quality", []):
        external = quality.get("external", {})
        external_text = ", ".join(
            f"{name}={values.get('nonzero_count', 0)}/{values.get('missing_count', 0)}"
            for name, values in external.items()
        )
        eligibility = quality.get("eligibility", {})
        target = eligibility.get("target_eligibility", {}).get("return", {})
        valid = target.get("valid_rows_by_horizon", {})
        gap = target.get("gap_excluded_rows_by_horizon", {})
        lines.append(
            "| {fold} | {feature_set} | {split} | {rows} | {external} | {context} | {gaps} | {valid} | {gap_valid} | {promotion} |".format(
                fold=quality.get("fold", "N/A"),
                feature_set=quality.get("feature_set", "N/A"),
                split=quality.get("split", "N/A"),
                rows=quality.get("rows", 0),
                external=external_text or "N/A",
                context=eligibility.get("context_ineligible_rows", "N/A"),
                gaps=eligibility.get("non_15m_transitions", "N/A"),
                valid="/".join(str(valid.get(str(horizon), "N/A")) for horizon in HORIZON_GRID),
                gap_valid="/".join(str(gap.get(str(horizon), "N/A")) for horizon in HORIZON_GRID),
                promotion=quality.get("promotion_status", "N/A"),
            )
        )

    replay = payload.get("wave3a_corrected_replay", {})
    lines.extend(
        [
            "",
            "## Frozen Wave3A comparison",
            "",
            "Wave3A output is not overwritten. Its causal trend+vol rule and Ridge/HistGB OHLCV13 rows are replayed here under the Wave3C validation-selected-constant comparator and common right-side delay alignment. This replay is report-only and excluded from every Wave3C choice.",
            "",
            "| source | status | rows | median dynamic AlphaEx | median constant AlphaEx | median timing increment |",
            "|---|---|---:|---:|---:|---:|",
            "| {source} | {status} | {rows} | {dynamic} | {constant} | {increment} |".format(
                source=replay.get("source_result_path", "N/A"),
                status=replay.get("status", "N/A"),
                rows=replay.get("summary", {}).get("rows", 0),
                dynamic=_report_number(replay.get("summary", {}).get("median_dynamic_alpha_excess_pt"), suffix="pt"),
                constant=_report_number(replay.get("summary", {}).get("median_constant_alpha_excess_pt"), suffix="pt"),
                increment=_report_number(replay.get("summary", {}).get("median_timing_increment_alpha_excess_pt"), suffix="pt"),
            ),
            "",
            "## Artifacts",
            "",
            f"- machine-readable ledger: `{payload.get('ledger_path', 'N/A')}`",
            f"- result JSON: `{payload.get('result_path', 'N/A')}`",
            f"- Wave3A errata: `{payload.get('wave3a_errata_path', 'N/A')}`",
            "",
            "No holdout folds 15–23 or future fold 24 were loaded, selected, or inspected by this screen.",
        ]
    )
    return "\n".join(lines) + "\n"
