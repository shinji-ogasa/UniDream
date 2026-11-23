"""Train-only labels and native/downstream scores for the 224-cell study.

Every fold shares one label contract across all inputs and model families.
There is no fitting on Val/Test, no calibration, no missing-row removal, and
no assumption relating integrated variance to terminal-return Gaussian risk.
Native continuous scores and downstream classification scores are separate.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

from .wm_ablation_evaluation import VARIANCE_FLOOR, variance_qlike
from .wm_seven_task_evaluation import TASKS


NATIVE_PRIMARY = {"return": "zero_oos_r2", "variance": "qlike", "downside": "mae", "crash": "brier",
                  "dd_excess": "mae", "advantage": "spearman_ic", "recovery": "spearman_ic"}
PROBABILITY_FLOOR = 1e-12


def _finite(values, name):
    values = np.asarray(values, float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain nonempty finite values")
    return values


def _entry(task: str, contract: Mapping) -> Mapping:
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    entry = contract[task] if task in contract else contract
    if int(entry["n_classes"]) != (2 if task == "crash" else 3):
        raise ValueError("label contract has an incorrect class count")
    return entry


def make_task_labels(values: np.ndarray, task: str, contract: Mapping) -> np.ndarray:
    """Apply the frozen Train label map, preserving any supplied array shape.

    Signed bounds are strictly negative/positive events; both equalities are
    neutral.  Quantile equalities stay in the lower class.  DD excess uses
    ``<=0`` for class 0, so negative out-of-domain scalar forecasts cannot be
    mistaken for a positive event.  Crash values are binary event indicators;
    this helper also provides the auxiliary p>0.5 map for probability inputs.
    """
    values = _finite(values, "values")
    entry = _entry(task, contract)
    kind = entry["kind"]
    bounds = np.asarray(entry["bounds"], float)
    if not np.isfinite(bounds).all():
        raise ValueError("label bounds must be finite")
    if kind == "binary_event" and task == "crash":
        if bounds.shape != (1,) or bounds[0] != .5:
            raise ValueError("binary event projection must use the fixed 0.5 boundary")
        return (values > .5).astype(np.int64)
    if bounds.shape != (2,) or bounds[0] >= bounds[1]:
        raise ValueError("three-class bounds must be distinct and increasing")
    low, high = bounds
    if kind == "signed_rms":
        if low >= 0 or high <= 0:
            raise ValueError("signed RMS bounds must straddle zero")
        return np.where(values < low, 0, np.where(values > high, 2, 1)).astype(np.int64)
    if kind == "train_tertiles":
        return np.where(values <= low, 0, np.where(values <= high, 1, 2)).astype(np.int64)
    if kind == "zero_positive_median" and task == "dd_excess":
        if low != 0:
            raise ValueError("DD-excess zero boundary must remain exactly zero")
        return np.where(values <= 0, 0, np.where(values <= high, 1, 2)).astype(np.int64)
    raise ValueError("unknown task label map")


def _valid_outcomes(task: str, values: np.ndarray) -> None:
    if task in ("variance", "downside", "dd_excess") and np.any(values < 0):
        raise ValueError(f"{task} actual outcomes must be nonnegative")
    if task == "crash" and not np.isin(values, (0, 1)).all():
        raise ValueError("crash actual outcomes must be binary 0/1")
    if task == "recovery" and np.any(np.abs(values) > 1 + 1e-12):
        raise ValueError("recovery actual outcomes must lie in [-1,1]")


def fit_label_contract(targets: Mapping[str, np.ndarray], train_ids: np.ndarray) -> dict[str, dict]:
    """Fit each fold's labels using ONLY its purged full Train decisions.

    Inputs are [full_grid_rows,1] arrays.  Missing Train outcomes or a class
    below max(200,ceil(0.5% Train)) cause an explicit failure; no tied values
    are split by row number, shuffled, jittered, or repaired from later data.
    The caller is responsible for providing the original purged Train cohort.
    """
    raw = {task: np.asarray(targets[task], float) for task in TASKS}
    shape = raw["return"].shape
    if len(shape) != 2 or shape[1] != 1 or any(value.shape != shape for value in raw.values()):
        raise ValueError("all target arrays must share shape [rows,1]")
    ids = np.asarray(train_ids)
    if ids.dtype == bool:
        if ids.shape != (shape[0],):
            raise ValueError("boolean train_ids must match the full grid")
        ids = np.flatnonzero(ids)
    if ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("train_ids must be a nonempty one-dimensional integer index")
    if np.any(ids < 0) or np.any(ids >= shape[0]) or len(np.unique(ids)) != len(ids):
        raise ValueError("train_ids must be unique and in range")
    minimum = max(200, int(np.ceil(.005 * len(ids))))
    result = {}
    for task in TASKS:
        values = _finite(raw[task][ids, 0], f"Train {task}")
        _valid_outcomes(task, values)
        entry = {"task": task, "n_classes": 2 if task == "crash" else 3, "n_train": len(ids),
                 "minimum_class_support": minimum, "fit_split": "Train_only", "quantile_method": None, "k": None}
        if task in ("return", "advantage", "recovery"):
            rms = float(np.sqrt(np.mean(values * values)))
            threshold = .5 * rms
            entry.update(kind="signed_rms", bounds=[-threshold, threshold], rms=rms, k=.5,
                         equality_rule="both signed threshold equalities are neutral")
        elif task in ("variance", "downside"):
            low, high = np.quantile(values, [1 / 3, 2 / 3], method="linear")
            entry.update(kind="train_tertiles", bounds=[float(low), float(high)], quantile_method="linear",
                         equality_rule="<=lower is class0; <=upper is class1; above is class2")
        elif task == "dd_excess":
            positives = values[values > 0]
            if not len(positives):
                raise ValueError("DD excess has no positive Train outcomes")
            high = float(np.quantile(positives, .5, method="linear"))
            entry.update(kind="zero_positive_median", bounds=[0., high], quantile_method="linear",
                         n_positive_train=len(positives), equality_rule="zero is class0; positive<=Train positive median is class1")
        else:
            entry.update(kind="binary_event", bounds=[.5], equality_rule="actual 0/1 events unchanged")
        labels = make_task_labels(values, task, entry)
        support = np.bincount(labels, minlength=entry["n_classes"])
        if np.any(support < minimum):
            raise ValueError(f"{task} Train class support {support.tolist()} is below preregistered minimum {minimum}")
        entry.update(priors=(support / support.sum()).tolist(), support=support.tolist())
        result[task] = entry
    return result


def _correlations(actual, predicted):
    if len(actual) < 2 or np.ptp(actual) == 0 or np.ptp(predicted) == 0:
        return {"pearson_ic": None, "spearman_ic": None}
    return {"pearson_ic": float(pearsonr(actual, predicted).statistic),
            "spearman_ic": float(spearmanr(actual, predicted).statistic)}


def _decisions(actual: np.ndarray, predicted: np.ndarray, classes: int) -> dict:
    confusion = np.bincount(actual * classes + predicted, minlength=classes * classes).reshape(classes, classes)
    support, called = confusion.sum(axis=1), confusion.sum(axis=0)
    diagonal = confusion.diagonal()
    recall = np.divide(diagonal, support, out=np.full(classes, np.nan), where=support > 0)
    precision = np.divide(diagonal, called, out=np.zeros(classes), where=called > 0)
    f1 = np.divide(2 * diagonal, support + called, out=np.zeros(classes), where=(support + called) > 0)
    return {"accuracy": float(diagonal.sum() / len(actual)),
        "balanced_accuracy": float(recall.mean()) if np.isfinite(recall).all() else None,
        "macro_f1": float(f1.mean()), "per_class_recall": [float(x) if np.isfinite(x) else None for x in recall],
        "per_class_precision": precision.tolist(), "confusion_matrix": confusion.tolist(),
        "actual_class_frequency": (support / len(actual)).tolist(), "predicted_class_frequency": (called / len(actual)).tolist()}


def _probability_scores(labels: np.ndarray, probs: np.ndarray, priors: np.ndarray) -> dict:
    classes = len(priors)
    if probs.shape != (len(labels), classes) or not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("classification probabilities must be finite [n,K] values in [0,1]")
    if not np.allclose(probs.sum(axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("classification probability rows must sum to one")
    probs = probs / probs.sum(axis=1, keepdims=True)
    primary = np.argmax(probs / priors, axis=1)
    raw = np.argmax(probs, axis=1)
    result = {"n": len(labels), **_decisions(labels, primary, classes),
        "raw": _decisions(labels, raw, classes),
        "log_loss": float(-np.log(np.maximum(probs[np.arange(len(labels)), labels], PROBABILITY_FLOOR)).mean()),
        "multiclass_brier": float(np.square(probs - np.eye(classes)[labels]).sum(axis=1).mean()),
        "probability_metrics_available": True, "primary_rule": "argmax(p/Train_prior)", "raw_rule": "argmax(p)"}
    if classes == 2:
        result["brier"] = float(np.square(probs[:, 1] - labels).mean())
        result["average_precision"] = float(average_precision_score(labels, probs[:, 1])) if np.any(labels == 1) else None
        result["prevalence"] = float(labels.mean())
    return result


def score_grid(task: str, actual: np.ndarray, forecast: np.ndarray, formulation: str,
               contract: Mapping, scales: Mapping | None) -> dict[str, Any]:
    """Score one task/fold/split with native values plus common class labels.

    Regression inputs are finite vectors; classification inputs are finite
    [n,K] probabilities.  Six scalar-regression outputs have no probability
    metrics.  Crash regression is explicitly a binary conditional-mean model:
    its output is already an event probability and receives Brier/log loss.
    The binary Brier is mean event squared error; multiclass Brier sums both
    class errors and is twice that value.  ``scales`` is Train metadata only;
    native errors remain in their original target units.
    """
    entry = _entry(task, contract)
    actual = _finite(actual, "actual")
    if actual.ndim != 1:
        raise ValueError("actual must be a one-dimensional vector")
    _valid_outcomes(task, actual)
    labels = make_task_labels(actual, task, contract)
    classes = int(entry["n_classes"])
    priors = np.asarray(entry["priors"], float)
    if priors.shape != (classes,) or not np.isfinite(priors).all() or np.any(priors <= 0) or not np.isclose(priors.sum(), 1, rtol=0, atol=1e-6):
        raise ValueError("all Train priors must be positive, finite and sum to one")
    if formulation not in ("regression", "classification"):
        raise ValueError("formulation must be regression or classification")
    forecast = _finite(forecast, "forecast")
    native = None
    if formulation == "classification":
        classification = _probability_scores(labels, forecast, priors)
        if task == "crash":
            native = {name: classification[name] for name in ("brier", "log_loss", "average_precision", "prevalence")}
        primary_metric, primary_value = "balanced_accuracy", classification["balanced_accuracy"]
    else:
        if forecast.shape != actual.shape:
            raise ValueError("regression forecast must match the actual vector")
        if task == "crash":
            if np.any(forecast < 0) or np.any(forecast > 1):
                raise ValueError("binary regression forecasts must be event probabilities in [0,1]")
            classification = _probability_scores(labels, np.column_stack((1 - forecast, forecast)), priors)
            native = {name: classification[name] for name in ("brier", "log_loss", "average_precision", "prevalence")}
        else:
            projected = make_task_labels(forecast, task, contract)
            classification = {"n": len(labels), **_decisions(labels, projected, classes),
                              "probability_metrics_available": False, "primary_rule": "fixed Train label map on scalar forecast"}
            error = forecast - actual
            native = {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
                      **_correlations(actual, forecast)}
            if task == "return":
                denominator = float(np.square(actual).sum())
                native["zero_oos_r2"] = 1 - float(np.square(error).sum()) / denominator if denominator > 0 else None
            elif task == "variance":
                native.update(qlike=float(variance_qlike(actual, forecast).mean()),
                    prediction_floor_count=int(np.sum(forecast < VARIANCE_FLOOR)),
                    calibration_mean_ratio=float(np.maximum(forecast, VARIANCE_FLOOR).sum() / actual.sum()) if actual.sum() > 0 else None)
        primary_metric = NATIVE_PRIMARY[task]
        primary_value = native[primary_metric]
    scale = None
    if scales is not None and task in scales:
        values = np.asarray(scales[task], float).reshape(-1)
        if values.shape != (1,) or not np.isfinite(values).all() or values[0] <= 0:
            raise ValueError("target scale must be one positive finite Train-only scalar")
        scale = float(values[0])
    return {"n": len(actual), "task": task, "formulation": formulation, "n_classes": classes,
        "primary_metric": primary_metric, "primary_value": primary_value, "native": native,
        "classification": classification, "target_scale": scale, "calibration_applied": False,
        "n_nonfinite_excluded": 0}
