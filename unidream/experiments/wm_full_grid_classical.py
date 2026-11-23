"""Missing classical objectives for the complete seven-target formulation grid.

Fits exactly six fresh three-class targets and one fresh squared-error crash
target. Existing six continuous regression heads and binary crash classifiers
belong to their frozen source experiment and are not refitted here. All new
selection uses the supplied first validation month; no Test path exists.
"""
from __future__ import annotations

from collections.abc import Mapping
import time

import numpy as np
from scipy.linalg import blas, cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from .wm_return_formulation_classical import (
    CLASSES, _features, _fit_logistic, _labels, _probabilities,
    return_classification_log_loss,
)


CLASSIFICATION_TASKS = ("return", "variance", "downside", "dd_excess", "advantage", "recovery")


def _classification_labels(values, n, name, *, train=False):
    if not isinstance(values, Mapping) or not set(CLASSIFICATION_TASKS).issubset(values):
        raise ValueError(f"{name} must contain all six ternary classification targets")
    return {task: _labels(values[task], n, name + "[" + task + "]", all_classes=train)
            for task in CLASSIFICATION_TASKS}


def _crash_target(values, n, name):
    if not isinstance(values, Mapping) or "crash" not in values:
        raise ValueError(f"{name} must provide the binary actual crash outcome")
    vector = np.asarray(values["crash"], dtype=np.float64)
    if vector.shape != (n,) or not np.isfinite(vector).all() or not np.isin(vector, (0., 1.)).all():
        raise ValueError(f"{name}[crash] must contain one finite binary outcome per row")
    return vector


def _linear_rows(labels, maximum):
    n = len(next(iter(labels.values())))
    if maximum is None:
        return np.arange(n, dtype=np.int64), False
    maximum = int(maximum)
    if maximum < 3:
        raise ValueError("linear_max_train_rows must allow three classes")
    rows = np.linspace(0, n - 1, min(n, maximum), dtype=np.int64)
    matrix = np.column_stack([labels[task] for task in CLASSIFICATION_TASKS])
    adjusted = False
    # Only an exceptional missing class changes the evenly spaced sample.
    # A replaceable row must be redundant for EVERY target, so adding one
    # rare class cannot erase another target's sole observation of a class.
    for column in range(len(CLASSIFICATION_TASKS)):
        for category in CLASSES:
            if np.any(matrix[rows, column] == category):
                continue
            counts = np.stack([np.bincount(matrix[rows, j], minlength=3)
                               for j in range(matrix.shape[1])])
            redundant = np.ones(len(rows), dtype=bool)
            for j in range(matrix.shape[1]):
                redundant &= counts[j, matrix[rows, j]] > 1
            replaceable = np.flatnonzero(redundant)
            if not len(replaceable):
                raise ValueError("linear row cap cannot retain all six targets' classes")
            rows[replaceable[-1]] = np.flatnonzero(matrix[:, column] == category)[0]
            adjusted = True
    return np.sort(rows), adjusted


def _brier(actual, prediction):
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != actual.shape or not np.isfinite(prediction).all():
        raise ValueError("crash regression predictions must be matching finite vectors")
    return float(np.mean(np.square(actual - np.clip(prediction, 0., 1.))))


def _fit_crash_ridge(x, actual, xv, actual_val, cfg):
    xmean, ymean = x.mean(0), float(actual.mean())
    centered, target = x - xmean, actual - ymean
    primal = centered.shape[1] <= centered.shape[0]
    # scipy BLAS produces the same arithmetic as NumPy Accelerate matmul,
    # without NumPy's observed spurious floating-point status warnings.
    gram = (blas.dgemm(1., centered, centered, trans_a=True) if primal
            else blas.dgemm(1., centered, centered, trans_b=True))
    cross = blas.dgemv(1., centered, target, trans=1) if primal else target
    diagonal = np.diag_indices_from(gram)
    initial_diagonal = gram[diagonal].copy()
    candidates = []
    for alpha in cfg.get("ridge_alphas", [10., 1000., 100000.]):
        alpha = float(alpha)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge alpha must be positive and finite")
        gram[diagonal] = initial_diagonal + alpha
        solved = cho_solve(cho_factor(gram, lower=True, check_finite=True), cross, check_finite=True)
        coefficients = solved if primal else blas.dgemv(1., centered, solved, trans=1)
        intercept = ymean - float(np.sum(xmean * coefficients))
        predicted = blas.dgemv(1., xv, coefficients) + intercept
        score = _brier(actual_val, predicted)
        head = {"kind": "ridge", "coefficients": coefficients.copy(), "intercept": intercept}
        candidates.append((score, head, {"alpha": alpha, "brier": score}))
    if not candidates:
        raise ValueError("at least one Ridge alpha is required")
    _, head, selected = min(candidates, key=lambda item: item[0])
    return head, {"selected": selected, "candidates": [item[2] for item in candidates],
                  "gram_form": "primal" if primal else "dual"}


def _fit_trees(x, labels, crash, xv, val_labels, crash_val, cfg, threads):
    import lightgbm as lgb

    parameters = {
        "learning_rate": float(cfg.get("lightgbm_learning_rate", .03)),
        "min_data_in_leaf": int(cfg.get("lightgbm_min_data_leaf", 256)),
        "lambda_l2": float(cfg.get("lightgbm_l2", 10.)),
        "max_bin": int(cfg.get("lightgbm_max_bin", 63)),
        "num_threads": threads, "seed": int(cfg.get("seed", 7)),
        "deterministic": True, "force_col_wise": True, "verbosity": -1,
        "feature_fraction": 1., "bagging_fraction": 1., "bagging_freq": 0,
        "metric": "None", "feature_pre_filter": False,
    }
    train_set = lgb.Dataset(x, label=np.zeros(len(x)), params=parameters, free_raw_data=False)
    val_set = lgb.Dataset(xv, label=np.zeros(len(xv)), reference=train_set,
                         params=parameters, free_raw_data=False)
    train_set.construct()
    val_set.construct()
    fits = {"classification": {}, "regression": {}}
    metadata = {"classification": {}, "regression": {}}
    for mode, task in [("classification", target) for target in CLASSIFICATION_TASKS] + [("regression", "crash")]:
        is_classification = mode == "classification"
        train_set.set_label(labels[task] if is_classification else crash)
        val_set.set_label(val_labels[task] if is_classification else crash_val)
        objective_parameters = ({"objective": "multiclass", "num_class": 3}
                                if is_classification else {"objective": "regression", "num_class": 1})
        candidates = []
        for leaves in cfg.get("lightgbm_leaves", [7, 31]):
            history = []

            def metric(predicted, _dataset):
                score = (return_classification_log_loss(val_labels[task], np.asarray(predicted).reshape(-1, 3))
                         if is_classification else _brier(crash_val, np.asarray(predicted)))
                history.append(score)
                return "unweighted_log_loss" if is_classification else "crash_brier", score, False

            started = time.monotonic()
            model = lgb.train(
                {**parameters, **objective_parameters, "num_leaves": int(leaves)}, train_set,
                num_boost_round=int(cfg.get("lightgbm_rounds", 600)),
                valid_sets=[val_set], valid_names=["earlystop_val"], feval=metric,
                callbacks=[lgb.early_stopping(int(cfg.get("lightgbm_patience", 60)),
                                             first_metric_only=True, verbose=False)],
                keep_training_booster=True,
            )
            if not history:
                raise RuntimeError(f"{mode} {task} produced no validation observations")
            iteration = int(np.argmin(history)) + 1
            prediction = model.predict(xv, num_iteration=iteration, num_threads=threads)
            score = (return_classification_log_loss(val_labels[task], prediction)
                     if is_classification else _brier(crash_val, prediction))
            model.free_dataset()
            score_name = "log_loss" if is_classification else "brier"
            row = {"num_leaves": int(leaves), score_name: score,
                   "selected_iteration": iteration, "observed_rounds": len(history),
                   "objective": objective_parameters["objective"], "history": history,
                   "elapsed_seconds": time.monotonic() - started}
            candidates.append((score, {"kind": "lightgbm", "model": model,
                                       "selected_iteration": iteration}, row))
        if not candidates:
            raise ValueError("at least one LightGBM leaf candidate is required")
        _, head, selected = min(candidates, key=lambda item: item[0])
        fits[mode][task] = head
        metadata[mode][task] = {"selected": selected, "candidates": [item[2] for item in candidates]}
    return fits, metadata


def fit_grid_classical(x_train, targets_train, labels_train, x_early,
                       targets_early, labels_early, config=None):
    """Fit exactly six ternary heads and one crash-MSE head per model family."""
    cfg = dict(config or {})
    threads = int(cfg.get("cpu_threads", 2))
    if threads < 1:
        raise ValueError("cpu_threads must be positive")
    x_train = _features(x_train, "x_train")
    x_early = _features(x_early, "x_early", x_train.shape[1])
    train_labels = _classification_labels(labels_train, len(x_train), "labels_train", train=True)
    early_labels = _classification_labels(labels_early, len(x_early), "labels_early")
    crash = _crash_target(targets_train, len(x_train), "targets_train")
    crash_val = _crash_target(targets_early, len(x_early), "targets_early")
    rows, adjusted = _linear_rows(train_labels, cfg.get("linear_max_train_rows", 4096))
    # Same Train-only scaler reductions as both frozen source experiments.
    mean = np.mean(x_train, axis=0, dtype=np.float64)
    std = np.std(x_train, axis=0, dtype=np.float64)
    std = np.where(std > 1e-12, std, 1.)
    linear_x = (x_train[rows].astype(np.float64) - mean) / std
    linear_val = (x_early.astype(np.float64) - mean) / std
    fits = {mode: {model: {} for model in ("linear", "lightgbm")}
            for mode in ("classification", "regression")}
    selection = {mode: {model: {} for model in ("linear", "lightgbm")}
                 for mode in ("classification", "regression")}
    started = time.monotonic()
    with threadpool_limits(limits=threads):
        for task in CLASSIFICATION_TASKS:
            model, info = _fit_logistic(linear_x, train_labels[task][rows], linear_val, early_labels[task], cfg)
            fits["classification"]["linear"][task] = {"kind": "logistic", "model": model}
            selection["classification"]["linear"][task] = info
        head, info = _fit_crash_ridge(linear_x, crash[rows], linear_val, crash_val, cfg)
        fits["regression"]["linear"]["crash"] = head
        selection["regression"]["linear"]["crash"] = info
        trees, tree_info = _fit_trees(x_train, train_labels, crash, x_early, early_labels, crash_val, cfg, threads)
        for mode in fits:
            fits[mode]["lightgbm"] = trees[mode]
            selection[mode]["lightgbm"] = tree_info[mode]
    return {
        "format": "full_grid_classical_v1", "n_features": int(x_train.shape[1]),
        "scaler": {"mean": mean, "scale": std}, "fits": fits,
        "metadata": {
            "train_n": len(x_train), "earlystop_n": len(x_early), "cpu_threads": threads,
            "linear_train_rows": rows.tolist(), "linear_class_coverage_adjustment": adjusted,
            "lightgbm_train_n": len(x_train),
            "train_priors": {task: (np.bincount(train_labels[task], minlength=3) / len(x_train)).tolist()
                             for task in CLASSIFICATION_TASKS},
            "crash_train_mean": float(crash.mean()), "selection": selection,
            "classification_class_order": [0, 1, 2], "class_weight": None,
            "selection_split": "supplied first validation month only", "test_labels_used": False,
            "fresh_objectives": {"classification": list(CLASSIFICATION_TASKS), "regression": ["crash"]},
            "existing_objectives_refitted": False, "linear_scaler_fit": "all Train rows only",
            "tree_feature_coordinates": "unscaled; one shared binned Dataset for all seven fresh objectives",
            "elapsed_seconds": time.monotonic() - started,
        },
    }


def predict_grid_classical(bundle, x):
    """Predict fresh objectives only; labels and outcomes are never accepted."""
    if bundle.get("format") != "full_grid_classical_v1":
        raise ValueError("unrecognized full-grid classical bundle")
    x = _features(x, "prediction features", bundle["n_features"])
    standardized = (x.astype(np.float64) - bundle["scaler"]["mean"]) / bundle["scaler"]["scale"]
    result = {mode: {model: {} for model in ("linear", "lightgbm")}
              for mode in ("classification", "regression")}
    threads = bundle["metadata"]["cpu_threads"]
    with threadpool_limits(limits=threads):
        for mode, families in bundle["fits"].items():
            for family, heads in families.items():
                for task, head in heads.items():
                    if head["kind"] == "logistic":
                        if not np.array_equal(head["model"].classes_, CLASSES):
                            raise ValueError("saved categorical columns are not 0/1/2")
                        prediction = head["model"].predict_proba(standardized)
                    elif head["kind"] == "ridge":
                        prediction = blas.dgemv(1., standardized, head["coefficients"]) + head["intercept"]
                    elif head["kind"] == "lightgbm":
                        prediction = head["model"].predict(x, num_iteration=head["selected_iteration"], num_threads=threads)
                    else:
                        raise ValueError("unknown full-grid fitted objective")
                    if mode == "classification":
                        prediction = _probabilities(prediction, len(x))
                    else:
                        prediction = np.asarray(prediction, dtype=np.float64)
                        if prediction.shape != (len(x),) or not np.isfinite(prediction).all():
                            raise ValueError("nonfinite or incorrectly shaped crash regression output")
                        prediction = np.clip(prediction, 0., 1.)
                    result[mode][family][task] = prediction
    return result
