"""Fresh three-class baselines for return-regression/classification comparison.

The coordinator supplies the same immutable Train cohort and raw tabular
information as the seven-task regression experiment.  This module fits on
Train and chooses hyperparameters and iterations exclusively by unweighted
log loss on the supplied first validation month.  It has no Test-label path,
class balancing, prior-adjusted probabilities, or BA-based selection.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


CLASSES = np.array([0, 1, 2], dtype=np.int64)
PROBABILITY_FLOOR = 1e-12


def _features(values, name, width=None):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if width is not None and values.shape[1] != width:
        raise ValueError(f"{name} must have the Train feature width")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains nonfinite features")
    return values


def _labels(values, n, name, *, all_classes=False):
    values = np.asarray(values)
    if values.shape != (n,) or not np.isin(values, CLASSES).all():
        raise ValueError(f"{name} must contain one class 0=down, 1=flat, 2=up per row")
    values = values.astype(np.int64, copy=False)
    if all_classes and not np.array_equal(np.unique(values), CLASSES):
        raise ValueError("Train must contain all three return classes")
    return values


def _probabilities(values, n):
    values = np.asarray(values, dtype=np.float64)
    if (values.shape != (n, 3) or not np.isfinite(values).all() or np.any(values < 0.)
            or np.any(values > 1.) or not np.allclose(values.sum(1), 1., rtol=1e-6, atol=1e-7)):
        raise ValueError("probabilities must be finite normalized [n,3] down/flat/up columns")
    return values


def return_classification_log_loss(labels, probabilities):
    """Unweighted natural-log multiclass loss with the shared 1e-12 floor."""
    labels = _labels(labels, len(labels), "scoring labels")
    probabilities = _probabilities(probabilities, len(labels))
    return float(-np.mean(np.log(np.maximum(probabilities[np.arange(len(labels)), labels], PROBABILITY_FLOOR))))


def _linear_rows(labels, maximum):
    if maximum is None:
        return np.arange(len(labels), dtype=np.int64)
    maximum = int(maximum)
    if maximum < 3:
        raise ValueError("linear_max_train_rows must allow all three classes")
    rows = np.linspace(0, len(labels) - 1, min(len(labels), maximum), dtype=np.int64)
    # Preserve the deterministic even sampling, replacing only redundant-class
    # rows if a rare class is absent from that subset.  Full Train priors are
    # never estimated on this computational subset.
    for missing in CLASSES[~np.isin(CLASSES, labels[rows])]:
        counts = np.bincount(labels[rows], minlength=3)
        replaceable = np.flatnonzero(counts[labels[rows]] > 1)
        if not len(replaceable):
            raise ValueError("cannot retain all classes in the requested linear row cap")
        rows[replaceable[-1]] = np.flatnonzero(labels == missing)[0]
    return np.sort(rows)


def _fit_logistic(x, labels, xv, labels_val, cfg):
    candidates = []
    for c in cfg.get("logistic_Cs", [.01, .1, 1.]):
        if not np.isfinite(c) or c <= 0:
            raise ValueError("logistic_Cs must contain positive finite numbers")
        attempts = []
        started = time.monotonic()
        for multiplier in (1, 4):
            model = LogisticRegression(
                C=float(c), solver="lbfgs", class_weight=None,
                max_iter=int(cfg.get("logistic_max_iter", 1000)) * multiplier,
                tol=float(cfg.get("logistic_tol", 1e-4)),
                random_state=int(cfg.get("seed", 7)),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x, labels)
            issues = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
            attempts.append({"max_iter": model.max_iter, "n_iter": model.n_iter_.tolist(),
                             "convergence_warnings": issues})
            if not issues:
                break
        else:
            raise RuntimeError(f"return logistic C={c} failed convergence after {model.max_iter} iterations")
        if not np.array_equal(model.classes_, CLASSES):
            raise RuntimeError("logistic probability columns are not down/flat/up")
        score = return_classification_log_loss(labels_val, model.predict_proba(xv))
        row = {"C": float(c), "log_loss": score, "attempts": attempts,
               "elapsed_seconds": time.monotonic() - started}
        candidates.append((score, model, row))
    if not candidates:
        raise ValueError("at least one logistic C is required")
    _, model, selected = min(candidates, key=lambda item: item[0])
    return model, {"selected": selected, "candidates": [item[2] for item in candidates]}


def _fit_lightgbm(x, labels, xv, labels_val, cfg, threads):
    import lightgbm as lgb

    parameters = {
        "objective": "multiclass", "num_class": 3, "metric": "None",
        "learning_rate": float(cfg.get("lightgbm_learning_rate", .03)),
        "min_data_in_leaf": int(cfg.get("lightgbm_min_data_leaf", 256)),
        "lambda_l2": float(cfg.get("lightgbm_l2", 10.)),
        "max_bin": int(cfg.get("lightgbm_max_bin", 63)),
        "num_threads": threads, "seed": int(cfg.get("seed", 7)),
        "deterministic": True, "force_col_wise": True, "verbosity": -1,
        "feature_fraction": 1., "bagging_fraction": 1., "bagging_freq": 0,
        "feature_pre_filter": False,
    }
    train_set = lgb.Dataset(x, label=labels, params=parameters, free_raw_data=False)
    val_set = lgb.Dataset(xv, label=labels_val, reference=train_set, params=parameters,
                         free_raw_data=False)
    candidates = []
    for leaves in cfg.get("lightgbm_leaves", [7, 31]):
        history = []

        def metric(probabilities, _dataset):
            score = return_classification_log_loss(labels_val, np.asarray(probabilities).reshape(-1, 3))
            history.append(score)
            return "unweighted_log_loss", score, False

        started = time.monotonic()
        model = lgb.train(
            {**parameters, "num_leaves": int(leaves)}, train_set,
            num_boost_round=int(cfg.get("lightgbm_rounds", 600)),
            valid_sets=[val_set], valid_names=["earlystop_val"], feval=metric,
            callbacks=[lgb.early_stopping(int(cfg.get("lightgbm_patience", 60)),
                                         first_metric_only=True, verbose=False)],
            keep_training_booster=True,
        )
        if not history:
            raise RuntimeError("LightGBM produced no early-stop validation observations")
        iteration = int(np.argmin(history)) + 1
        probabilities = model.predict(xv, num_iteration=iteration, num_threads=threads)
        score = return_classification_log_loss(labels_val, probabilities)
        model.free_dataset()
        row = {"num_leaves": int(leaves), "log_loss": score,
               "selected_iteration": iteration, "observed_rounds": len(history),
               "history": history, "elapsed_seconds": time.monotonic() - started}
        candidates.append((score, model, row))
    if not candidates:
        raise ValueError("at least one LightGBM leaf candidate is required")
    _, model, selected = min(candidates, key=lambda item: item[0])
    return model, {"selected": selected, "candidates": [item[2] for item in candidates],
                   "parameters": parameters}


def fit_return_classifiers(x_train, labels_train, x_early, labels_early, config=None):
    """Fit fresh classifiers and select solely on first-Val-month log loss."""
    cfg = dict(config or {})
    threads = int(cfg.get("cpu_threads", 1))
    if threads < 1:
        raise ValueError("cpu_threads must be positive")
    x_train = _features(x_train, "x_train")
    x_early = _features(x_early, "x_early", x_train.shape[1])
    labels_train = _labels(labels_train, len(x_train), "labels_train", all_classes=True)
    labels_early = _labels(labels_early, len(x_early), "labels_early")
    rows = _linear_rows(labels_train, cfg.get("linear_max_train_rows", 4096))
    # Exact same all-Train reductions and constant-column rule as the seven-
    # task regression baseline, independent of the classifier row cap.
    mean = np.mean(x_train, axis=0, dtype=np.float64)
    std = np.std(x_train, axis=0, dtype=np.float64)
    std = np.where(std > 1e-12, std, 1.)
    linear_train = (x_train[rows].astype(np.float64) - mean) / std
    linear_early = (x_early.astype(np.float64) - mean) / std
    started = time.monotonic()
    with threadpool_limits(limits=threads):
        logistic, logistic_metadata = _fit_logistic(linear_train, labels_train[rows],
                                                    linear_early, labels_early, cfg)
        lightgbm, lightgbm_metadata = _fit_lightgbm(x_train, labels_train, x_early,
                                                    labels_early, cfg, threads)
    return {
        "format": "return_formulation_classical_v1", "n_features": int(x_train.shape[1]),
        "fits": {"logistic": logistic, "lightgbm": lightgbm},
        "scaler": {"mean": mean, "scale": std},
        "metadata": {
            "class_order": [0, 1, 2], "train_priors": (np.bincount(labels_train, minlength=3) / len(labels_train)).tolist(),
            "train_n": len(x_train), "earlystop_n": len(x_early),
            "linear_train_rows": rows.tolist(), "lightgbm_train_n": len(x_train),
            "linear_scaler_fit": "all Train rows only; same reductions as seven-task regression",
            "class_weight": None, "probabilities_prior_adjusted": False,
            "selection_split": "supplied first validation month only",
            "selection_metric": "unweighted log loss with 1e-12 probability floor",
            "test_labels_used": False, "cpu_threads": threads,
            "logistic": logistic_metadata, "lightgbm": lightgbm_metadata,
            "elapsed_seconds": time.monotonic() - started,
        },
    }


def predict_return_classifiers(bundle, x):
    """Return raw down/flat/up probabilities without accepting any labels."""
    if bundle.get("format") != "return_formulation_classical_v1":
        raise ValueError("unrecognized return-formulation classical bundle")
    x = _features(x, "prediction features", bundle["n_features"])
    standardized = (x.astype(np.float64) - bundle["scaler"]["mean"]) / bundle["scaler"]["scale"]
    threads = bundle["metadata"]["cpu_threads"]
    if not np.array_equal(bundle["fits"]["logistic"].classes_, CLASSES):
        raise ValueError("saved logistic probability columns are not down/flat/up")
    with threadpool_limits(limits=threads):
        logistic = bundle["fits"]["logistic"].predict_proba(standardized)
        lightgbm = bundle["fits"]["lightgbm"].predict(
            x, num_iteration=bundle["metadata"]["lightgbm"]["selected"]["selected_iteration"],
            num_threads=threads,
        )
    return {"logistic": _probabilities(logistic, len(x)),
            "lightgbm": _probabilities(lightgbm, len(x))}
