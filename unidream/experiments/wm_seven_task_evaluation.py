"""Shared targets and metrics for the isolated seven-task WM benchmark.

Rows are UTC 15-minute bar opens.  Decision ``t`` sees closes through ``t-1``;
its future path contains closes ``t`` through ``t+h-1``.  Return is a log
return; downside is a fractional decline from the reference close to the
minimum future close, floored at zero, not peak-to-trough drawdown.  Variance
is the integrated sum of squared future 15-minute log returns, not volatility.

The last three labels are explicitly specified synthetic research targets.
They do not establish executable trading profit or the value of adding risk.
In particular ``advantage`` combines a log return and a fractional downside
penalty by design.  Their constants must be fixed before validation/test.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

from unidream.experiments.wm_ablation_evaluation import (
    BAR_INTERVAL,
    VARIANCE_FLOOR,
    WalkForwardFold,
    _grid,
    build_future_targets,
    variance_qlike,
)


TASKS = ("return", "variance", "downside", "crash", "dd_excess", "advantage", "recovery")

TARGET_DEFINITIONS = {
    "decision": "At bar-open t, only observations through close[t-1] are available.",
    "return": "log(close[t+h-1] / close[t-1])",
    "variance": "sum(log(close[j] / close[j-1])**2 for j=t,...,t+h-1)",
    "downside": "max(0, 1-min(close[t:t+h])/close[t-1]); future closes only; fractional decline",
    "crash": "float(downside >= crash_threshold)",
    "dd_excess": "max(downside-crash_threshold, 0); excess size, not exceedance probability",
    "advantage": "advantage_delta*(return-dd_penalty*downside-transition_cost); synthetic score",
    "recovery": "clip((return-recovery_dd_penalty*downside)/(sqrt(variance)+1e-6), -5, 5)/5; synthetic score",
    "label_end_ns": "UTC close time of final outcome bar: decision time+h*15min; require strictly before split end",
    "baseline_variance": "sum of preceding 16 available log-return squares, multiplied by h/16",
}


def build_seven_targets(
    index: pd.DatetimeIndex,
    close: Sequence[float] | np.ndarray,
    observed: Sequence[bool] | np.ndarray | None = None,
    horizons: Sequence[int] = (16,),
    crash_threshold: float = 0.012,
    advantage_delta: float = 0.06,
    dd_penalty: float = 0.35,
    recovery_dd_penalty: float = 0.50,
    transition_cost: float = 0.00055,
) -> dict[str, Any]:
    """Return each target as ``[full_grid_rows, horizons]`` with shared masks.

    Reference/future missingness invalidates all seven labels together.  The
    baseline has its own causal history requirement and can remain NaN while
    future labels are valid.  ``horizons`` preserves the configured column
    order; no observations or missing bars are removed from the grid.
    """
    constants = np.asarray([advantage_delta, dd_penalty, recovery_dd_penalty, transition_cost], float)
    if not np.isfinite(constants).all() or advantage_delta <= 0 or np.any(constants[1:] < 0):
        raise ValueError("target constants must be finite; advantage_delta positive and penalties/cost nonnegative")
    base = build_future_targets(index, close, observed, horizons, crash_threshold)
    result: dict[str, Any] = {
        "horizons": base.horizons,
        "return": base.returns,
        "variance": base.variance,
        "downside": base.downside,
        "crash": base.crash,
        "dd_excess": np.maximum(base.downside - crash_threshold, 0.0),
        "advantage": advantage_delta * (base.returns - dd_penalty * base.downside - transition_cost),
        "recovery": np.clip(
            (base.returns - recovery_dd_penalty * base.downside) / (np.sqrt(base.variance) + 1e-6),
            -5.0, 5.0,
        ) / 5.0,
        "mask": base.mask,
        "label_end_ns": base.label_end.astype("int64"),
        "baseline_variance": base.baseline_variance,
    }
    return result


def seven_split_mask(
    index: pd.DatetimeIndex,
    targets: Mapping[str, Any],
    fold: WalkForwardFold,
    split: str,
    horizon: int | None = None,
    context_valid: Sequence[bool] | np.ndarray | None = None,
) -> np.ndarray:
    """Select a split with full future paths and a strict label-end purge.

    Without ``horizon``, all configured labels must be valid and end strictly
    before the boundary, ensuring comparisons share the maximum-horizon purge.
    The caller must pass the original complete grid used to create targets.
    """
    index = _grid(index)
    mask = np.asarray(targets["mask"], dtype=bool)
    ends = np.asarray(targets["label_end_ns"], dtype=np.int64)
    horizons = tuple(targets["horizons"])
    if mask.shape != (len(index), len(horizons)) or ends.shape != mask.shape:
        raise ValueError("target masks and label ends must match the full grid and horizons")
    expected_ends = index.asi8[:, None] + np.asarray(horizons)[None, :] * BAR_INTERVAL.value
    if not np.array_equal(ends, expected_ends):
        raise ValueError("target and split grids must match")
    start, end = fold.bounds(split)
    eligible = np.asarray((index >= start) & (index < end))
    complete = mask & (ends < end.value)
    if horizon is None:
        eligible &= complete.all(axis=1)
    else:
        eligible &= complete[:, horizons.index(horizon)]
    if context_valid is not None:
        context_valid = np.asarray(context_valid, dtype=bool)
        if context_valid.shape != eligible.shape:
            raise ValueError("context_valid must contain one boolean per grid row")
        eligible &= context_valid
    return eligible


def fit_target_scales(targets: Mapping[str, Any], train_ids: Sequence[int] | np.ndarray) -> dict[str, list[float]]:
    """Fit per-horizon scales using ONLY the supplied purged Train decisions.

    Continuous scales are RMS with a 1e-4 floor, except variance uses its mean
    with a 1e-8 floor.  Crash uses an unscaled Bernoulli loss; its Train prior
    is returned separately as ``crash_prior``.  Missing Train cells are ignored
    per horizon, and an entirely unavailable horizon is an error.
    """
    ids = np.asarray(train_ids)
    if ids.dtype == bool:
        if ids.shape != (np.asarray(targets["return"]).shape[0],):
            raise ValueError("boolean train_ids must have one entry per grid row")
        ids = np.flatnonzero(ids)
    elif ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("train_ids must be a one-dimensional integer index or boolean mask")
    if not len(ids):
        raise ValueError("cannot fit target scales without Train observations")
    n_rows = np.asarray(targets["return"]).shape[0]
    if np.any(ids < 0) or np.any(ids >= n_rows) or len(np.unique(ids)) != len(ids):
        raise ValueError("train_ids must be unique in-range row indices")
    scales: dict[str, list[float]] = {}
    for task in TASKS:
        values = np.asarray(targets[task], dtype=np.float64)
        if values.ndim != 2 or values.shape != np.asarray(targets["mask"]).shape:
            raise ValueError("targets must share a two-dimensional [rows,horizons] shape")
        selected = values[ids]
        valid = np.isfinite(selected) & np.asarray(targets["mask"], bool)[ids]
        if not valid.any(axis=0).all():
            raise ValueError(f"{task} has a horizon without finite Train labels")
        selected = np.where(valid, selected, 0.0)
        count = valid.sum(axis=0)
        if task == "crash":
            if not np.isin(selected[valid], (0.0, 1.0)).all():
                raise ValueError("crash Train labels must be binary")
            scales[task] = np.ones(selected.shape[1]).tolist()
            scales["crash_prior"] = (selected.sum(axis=0) / count).tolist()
        elif task == "variance":
            if np.any(selected[valid] < 0):
                raise ValueError("realized variance cannot be negative")
            scales[task] = np.maximum(selected.sum(axis=0) / count, 1e-8).tolist()
        else:
            scales[task] = np.maximum(np.sqrt(np.square(selected).sum(axis=0) / count), 1e-4).tolist()
    return scales


def _correlations(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    if len(actual) < 2 or np.ptp(actual) == 0 or np.ptp(predicted) == 0:
        return {"pearson_ic": None, "spearman_ic": None}
    return {
        "pearson_ic": float(pearsonr(actual, predicted).statistic),
        "spearman_ic": float(spearmanr(actual, predicted).statistic),
    }


def _errors(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    error = predicted - actual
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
            **_correlations(actual, predicted)}


def _probability_calibration(actual: np.ndarray, predicted: np.ndarray, bins: int = 10) -> dict[str, Any]:
    assignments = np.minimum((predicted * bins).astype(int), bins - 1)
    reliability = []
    weighted_error = 0.0
    for i in range(bins):
        selected = assignments == i
        n = int(selected.sum())
        mean_p = float(predicted[selected].mean()) if n else None
        frequency = float(actual[selected].mean()) if n else None
        if n:
            weighted_error += n * abs(mean_p - frequency)
        reliability.append({"lower": i / bins, "upper": (i + 1) / bins, "n": n,
                            "mean_probability": mean_p, "observed_frequency": frequency})
    return {"ece": weighted_error / len(actual), "bins": reliability}


def score_seven(
    actual: Mapping[str, np.ndarray],
    predicted: Mapping[str, np.ndarray],
    baseline_variance: np.ndarray | None = None,
    *,
    quantile_predictions: Mapping[float, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Score all seven tasks on one COMMON finite set, with no fitting.

    All task arrays must have the same one- or two-dimensional shape.  A 2-D
    input is explicitly pooled across its horizon cells; callers should pass
    individual horizon columns to produce interpretable horizon-specific
    comparisons.  Nonfinite predictions in any task remove that cell from
    ALL task metrics and are counted.  Baseline/quantile missingness affects
    only its optional comparison, never the common learned-model cohort.

    ``quantile_predictions`` maps predeclared levels to DD-excess conditional
    quantile predictions.  MAE/RMSE assess the point head; pinball loss assesses
    these quantile heads.  No class or threshold selection occurs here.
    """
    arrays_a = {task: np.asarray(actual[task], dtype=np.float64) for task in TASKS}
    arrays_p = {task: np.asarray(predicted[task], dtype=np.float64) for task in TASKS}
    shape = arrays_a["return"].shape
    if len(shape) not in (1, 2) or any(value.shape != shape for value in [*arrays_a.values(), *arrays_p.values()]):
        raise ValueError("all task predictions and outcomes must have the same 1-D or 2-D shape")
    keep = np.ones(shape, bool)
    for value in [*arrays_a.values(), *arrays_p.values()]:
        keep &= np.isfinite(value)
    result: dict[str, Any] = {"n": int(keep.sum()), "n_nonfinite_excluded": int((~keep).sum())}
    if not keep.any():
        return result
    a = {task: value[keep] for task, value in arrays_a.items()}
    p = {task: value[keep] for task, value in arrays_p.items()}
    if np.any(a["variance"] < 0) or np.any(a["downside"] < 0) or np.any(a["dd_excess"] < 0):
        raise ValueError("realized variance, downside and DD excess must be nonnegative")
    if not np.isin(a["crash"], (0.0, 1.0)).all() or np.any((p["crash"] < 0) | (p["crash"] > 1)):
        raise ValueError("crash labels must be binary and predictions probabilities")
    for task in ("return", "downside", "dd_excess", "advantage", "recovery"):
        result[task] = _errors(a[task], p[task])
    denominator = float(np.square(a["return"]).sum())
    result["return"]["zero_oos_r2"] = (
        1.0 - float(np.square(p["return"] - a["return"]).sum()) / denominator if denominator > 0 else None
    )
    predicted_v = np.maximum(p["variance"], VARIANCE_FLOOR)
    actual_v = a["variance"]
    result["variance"] = {
        **_errors(actual_v, predicted_v),
        "qlike": float(variance_qlike(actual_v, predicted_v).mean()),
        "prediction_floor_count": int(np.sum(p["variance"] < VARIANCE_FLOOR)),
        "actual_mean": float(actual_v.mean()),
        "predicted_mean": float(predicted_v.mean()),
        "calibration_mean_ratio": float(predicted_v.sum() / actual_v.sum()) if actual_v.sum() > 0 else None,
    }
    if baseline_variance is not None:
        baseline = np.asarray(baseline_variance, dtype=float)
        if baseline.shape != shape:
            raise ValueError("baseline_variance must match the task-array shape")
        baseline = baseline[keep]
        valid = np.isfinite(baseline) & (baseline >= 0)
        result["variance"]["baseline_n"] = int(valid.sum())
        if valid.any():
            baseline_loss = variance_qlike(actual_v[valid], baseline[valid])
            model_loss = variance_qlike(actual_v[valid], predicted_v[valid])
            result["variance"]["baseline_qlike"] = float(baseline_loss.mean())
            result["variance"]["qlike_improvement_vs_baseline"] = float((baseline_loss - model_loss).mean())
        else:
            result["variance"]["baseline_qlike"] = None
            result["variance"]["qlike_improvement_vs_baseline"] = None
    probability = np.clip(p["crash"], 1e-12, 1 - 1e-12)
    event = a["crash"]
    result["crash"] = {
        "brier": float(np.square(p["crash"] - event).mean()),
        "log_loss": float(-(event * np.log(probability) + (1 - event) * np.log1p(-probability)).mean()),
        "pr_auc": float(average_precision_score(event, p["crash"])) if np.any(event == 1) else None,
        "prevalence": float(event.mean()),
        "mean_probability": float(p["crash"].mean()),
        "calibration": _probability_calibration(event, p["crash"]),
    }
    if quantile_predictions is not None:
        result["dd_excess"]["pinball"] = {}
        for quantile, value in quantile_predictions.items():
            if not 0 < quantile < 1:
                raise ValueError("quantile levels must lie strictly between zero and one")
            value = np.asarray(value, float)
            if value.shape != shape:
                raise ValueError("DD-excess quantile predictions must match the task-array shape")
            value = value[keep]
            valid = np.isfinite(value)
            error = a["dd_excess"][valid] - value[valid]
            result["dd_excess"]["pinball"][str(quantile)] = {
                "n": int(valid.sum()),
                "loss": float(np.maximum(quantile * error, (quantile - 1) * error).mean()) if valid.any() else None,
            }
    return result
