"""Native-head metrics for the isolated return-only multitask experiment.

This module fits nothing. A scalar forecast has no probability scores; an
inactive head must never be presented as a trained native prediction.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .wm_classification_evaluation import score_classification
from .wm_return_formulation_evaluation import make_return_labels, score_scalar_labels


MODELS = ("direct", "wm")
ARMS = ("reg_only", "class_only", "joint")


def finite_vector(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return result


def regression_metrics(actual, predicted):
    actual, predicted = finite_vector(actual, "actual"), finite_vector(predicted, "predicted")
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted return vectors must match")
    residual = actual - predicted
    zero_sse = float(actual @ actual)
    a, b = actual - actual.mean(), predicted - predicted.mean()
    denominator = float(np.sqrt((a @ a) * (b @ b)))
    return {
        "n": len(actual), "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.abs(residual).mean()),
        "zero_oos_r2": 1. - float(residual @ residual) / zero_sse if zero_sse > 0 else None,
        "pearson_ic": float((a @ b) / denominator) if denominator > 0 else None,
        "spearman_ic": float(spearmanr(actual, predicted).statistic) if np.ptp(actual) > 0 and np.ptp(predicted) > 0 else None,
        "zero_baseline_rmse": float(np.sqrt(zero_sse / len(actual))),
    }


def classification_metrics(actual_class, probabilities, train_priors):
    """Return both p/Train-prior and raw argmax decisions; no calibration."""
    scores = score_classification(actual_class, probabilities, train_priors)
    result = {key: scores[key] for key in ("n", "accuracy", "balanced_accuracy", "macro_f1", "log_loss", "multiclass_brier")}
    result.update({"raw_" + key: scores["raw"][key] for key in ("accuracy", "balanced_accuracy", "macro_f1")})
    for rule, values in (("prior", scores), ("raw", scores["raw"])):
        for category in range(3):
            result[f"{rule}_recall_c{category}"] = values["per_class_recall"][category]
            result[f"{rule}_predicted_fraction_c{category}"] = values["predicted_class_frequency"][category]
            for predicted in range(3):
                result[f"{rule}_confusion_c{category}_to_c{predicted}"] = values["confusion_matrix"][category][predicted]
    return result


def scalar_classification_metrics(actual_class, scalar_prediction, theta):
    labels = make_return_labels(scalar_prediction, theta)
    scores = score_scalar_labels(actual_class, labels)
    return {key: scores[key] for key in ("n", "accuracy", "balanced_accuracy", "macro_f1")}


def coarse_return_projection(probabilities, train_class_means):
    probabilities, means = np.asarray(probabilities, float), np.asarray(train_class_means, float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or means.shape not in ((3,), probabilities.shape):
        raise ValueError("coarse projection needs probabilities [n,3] and Train class means [3] or [n,3]")
    if (not np.isfinite(probabilities).all() or not np.isfinite(means).all()
            or np.any(probabilities < 0) or not np.allclose(probabilities.sum(1), 1., atol=1e-6, rtol=0)):
        raise ValueError("coarse projection probabilities/Train means are invalid")
    return (probabilities * means).sum(axis=1)
