"""Validation-selected, unweighted classical down/flat/up classifiers.

This module fits on Train and selects on the frozen validation panel only.
It contains no Test prediction path; the experiment's global evaluation
barrier must be released by its coordinating runner.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib
import json
from pathlib import Path
import pickle
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .wm_classification_evaluation import score_classification


CLASS_ORDER = (0, 1, 2)


def _validate_labels(labels, *, require_all_classes=False):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, CLASS_ORDER).all():
        raise ValueError("labels must be a vector containing 0=down, 1=flat, 2=up")
    if require_all_classes and not np.array_equal(np.unique(labels), CLASS_ORDER):
        raise ValueError("classifier Train must contain all three classes")
    return labels.astype(np.int64, copy=False)


def validate_priors(priors):
    priors = np.asarray(priors, dtype=np.float64)
    if priors.shape != (3,) or not np.isfinite(priors).all() or np.any(priors <= 0):
        raise ValueError("Train priors must contain three finite positive probabilities")
    if not np.isclose(priors.sum(), 1., atol=1e-8):
        raise ValueError("Train priors must sum to one")
    return priors


def prior_adjusted_decision(probabilities, priors):
    """A fixed evaluation decision; probabilities themselves are unadjusted."""
    priors = validate_priors(priors)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("probabilities must have shape [n, 3]")
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(1), 1., atol=1e-6)):
        raise ValueError("probabilities must be finite, nonnegative and normalized")
    return np.argmax(probabilities / priors, axis=1)


def validation_scores(labels, probabilities, priors):
    labels = _validate_labels(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(labels) != len(probabilities):
        raise ValueError("validation labels and probabilities must share rows")
    # Use exactly the same 1e-12 true-class probability floor and decision
    # statistics as neural selection and the eventual held-out report.
    prior_adjusted_decision(probabilities, priors)
    shared = score_classification(labels, probabilities, priors)
    return {
        "balanced_accuracy": float(shared["balanced_accuracy"]),
        "log_loss": float(shared["log_loss"]),
        "raw_argmax_balanced_accuracy": float(shared["raw"]["balanced_accuracy"]),
    }


def _rank(scores):
    return (-scores["balanced_accuracy"], scores["log_loss"])


def append_threshold_side(design, theta, ids, return_rms):
    design = np.asarray(design, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.int64)
    selected_theta = np.asarray(theta, dtype=np.float64)[ids]
    if (design.ndim != 2 or len(design) != len(ids) or not np.isfinite(design).all()
            or not np.isfinite(return_rms) or return_rms <= 0
            or not np.isfinite(selected_theta).all() or np.any(selected_theta <= 0)):
        raise ValueError("design, selected thresholds and Train return RMS must be finite and positive where required")
    side = np.log(selected_theta / return_rms).astype(np.float32)
    return np.column_stack((design, side))


def fit_logistic_candidates(x, labels, xv, labels_val, priors, config=None):
    """Fit C candidates without class weights; retry failed convergence."""
    config = config or {}
    labels = _validate_labels(labels, require_all_classes=True)
    labels_val = _validate_labels(labels_val, require_all_classes=True)
    priors = validate_priors(priors)
    base_iterations = int(config.get("logistic_max_iter", 1000))
    candidate_rows, candidates = [], []
    for c in config.get("logistic_Cs", [.001, .01, .1, 1.]):
        attempts = []
        for multiplier in (1, 4, 16):
            iterations = base_iterations * multiplier
            model = LogisticRegression(
                C=float(c), solver="lbfgs", class_weight=None,
                max_iter=iterations, tol=float(config.get("logistic_tol", 1e-4)),
                random_state=int(config.get("seed", 20261110)),
            )
            started = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x, labels)
            convergence_warnings = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
            attempts.append({"max_iter": iterations, "n_iter": model.n_iter_.tolist(),
                             "convergence_warnings": convergence_warnings,
                             "elapsed_seconds": time.monotonic() - started})
            if not convergence_warnings:
                break
        else:
            raise RuntimeError(f"Logistic C={c} did not converge after {iterations} iterations; no Test predictions were produced")
        if not np.array_equal(model.classes_, CLASS_ORDER):
            raise RuntimeError("logistic probability columns are not down/flat/up")
        scores = validation_scores(labels_val, model.predict_proba(xv), priors)
        row = {"C": float(c), **scores, "fit_attempts": attempts}
        candidate_rows.append(row)
        candidates.append((model, row))
    if not candidates:
        raise ValueError("at least one logistic C candidate is required")
    model, selected = min(candidates, key=lambda item: _rank(item[1]))
    return model, {"selected": selected, "candidates": candidate_rows,
                   "selection": "highest prior-adjusted validation balanced accuracy, ties lowest unweighted log loss"}


def fit_lightgbm_candidates(x, labels, xv, labels_val, priors, config=None, *, cpu_threads=4):
    """BA-only early stopping; select observed BA ties by unweighted log loss."""
    import lightgbm as lgb
    config = config or {}
    labels = _validate_labels(labels, require_all_classes=True)
    labels_val = _validate_labels(labels_val, require_all_classes=True)
    priors = validate_priors(priors)
    rounds = int(config.get("lightgbm_rounds", 1500))
    patience = int(config.get("lightgbm_patience", 100))
    candidate_rows, candidates = [], []
    for leaves in config.get("lightgbm_leaves", [7, 31]):
        history = []

        def metric(probabilities, dataset):
            probabilities = np.asarray(probabilities).reshape(-1, 3)
            scores = validation_scores(dataset.get_label(), probabilities, priors)
            history.append({"iteration": len(history) + 1, **scores})
            # Exactly one metric: default multi_logloss must not participate
            # in early stopping or silently stop the BA-selected candidate.
            return "prior_adjusted_balanced_accuracy", scores["balanced_accuracy"], True

        parameters = {
            "objective": "multiclass", "num_class": 3, "metric": "None",
            "num_leaves": int(leaves), "learning_rate": float(config.get("lightgbm_learning_rate", .03)),
            "min_data_in_leaf": int(config.get("lightgbm_min_data_leaf", 256)),
            "lambda_l2": float(config.get("lightgbm_l2", 10.)),
            "max_bin": int(config.get("lightgbm_max_bin", 63)),
            "num_threads": int(cpu_threads), "seed": int(config.get("seed", 20261110)),
            "deterministic": True, "force_col_wise": True, "verbosity": -1,
            "feature_fraction": 1., "bagging_fraction": 1., "bagging_freq": 0,
        }
        train_set = lgb.Dataset(x, label=labels, free_raw_data=True)
        val_set = lgb.Dataset(xv, label=labels_val, reference=train_set, free_raw_data=True)
        started = time.monotonic()
        model = lgb.train(
            parameters, train_set, num_boost_round=rounds,
            valid_sets=[val_set], valid_names=["selection_val"], feval=metric,
            callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)],
            keep_training_booster=True,
        )
        if not history:
            raise RuntimeError("LightGBM produced no validation observations")
        selected_iteration = min(history, key=_rank)["iteration"]
        probabilities = model.predict(xv, num_iteration=selected_iteration, num_threads=cpu_threads)
        scores = validation_scores(labels_val, probabilities, priors)
        row = {"num_leaves": int(leaves), "selected_iteration": int(selected_iteration),
               "ba_early_stopping_best_iteration": int(model.best_iteration),
               "observed_rounds": len(history), "parameters": parameters,
               "elapsed_seconds": time.monotonic() - started, **scores, "history": history}
        model.free_dataset()
        candidate_rows.append(row)
        candidates.append((model, row))
    if not candidates:
        raise ValueError("at least one LightGBM leaf candidate is required")
    model, selected = min(candidates, key=lambda item: _rank(item[1]))
    return model, {"selected": selected, "candidates": candidate_rows,
                   "selection": "BA-only early stopping; highest validation BA, ties lowest unweighted log loss among observed iterations/candidates"}


def fit_classical_task(config_path, task):
    """Lazy root integration; fit both models, and write no Test predictions."""
    root = importlib.import_module("unidream.experiments.wm_volatility_classification")
    cfg = root.load_config(config_path)
    condition = task["condition"]
    if condition not in ("fixed", "wm", "lgbm"):
        raise ValueError("unknown threshold condition")
    out = Path(cfg["output_dir"])
    directory = out / "classical" / task["job"]
    directory.mkdir(parents=True, exist_ok=True)
    registration_sha = root._sha(out / "preregistration.json")
    previous_path = directory / "fit.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if previous["preregistration_sha256"] != registration_sha:
            raise ValueError("existing classical fit has a different registration")
        return {key: previous[key] for key in ("job", "condition", "fold", "elapsed_seconds")}
    with (directory / "fit.log").open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        started = time.monotonic()
        with np.load(out / "prepared" / f"fold_{task['fold']:02d}.npz") as archive:
            data = {name: archive[name] for name in archive.files}
        fold = json.loads((out / "prepared/folds.json").read_text())[task["fold"]]
        train, val = data["train"], data["selection_val"]
        x = append_threshold_side(root.get_design(cfg, train, data), data["theta_" + condition], train, fold["return_rms"])
        xv = append_threshold_side(root.get_design(cfg, val, data), data["theta_" + condition], val, fold["return_rms"])
        labels, labels_val = data["label_" + condition][train], data["label_" + condition][val]
        priors = validate_priors(fold["priors"][condition])
        actual_priors = np.bincount(labels.astype(int), minlength=3) / len(labels)
        if not np.allclose(actual_priors, priors, atol=1e-10):
            raise ValueError("registered Train class priors disagree with actual Train labels")
        threads = int(cfg.get("cpu_threads", 4))
        with threadpool_limits(limits=threads):
            logistic, logistic_history = fit_logistic_candidates(x, labels, xv, labels_val, priors, cfg.get("tabular", {}))
            temporary = directory / "logistic.pkl.tmp"
            temporary.write_bytes(pickle.dumps(logistic, protocol=pickle.HIGHEST_PROTOCOL))
            temporary.replace(directory / "logistic.pkl")
            lightgbm, lightgbm_history = fit_lightgbm_candidates(
                x, labels, xv, labels_val, priors, cfg.get("tabular", {}), cpu_threads=threads
            )
            temporary = directory / "lightgbm.txt.tmp"
            lightgbm.save_model(str(temporary), num_iteration=lightgbm_history["selected"]["selected_iteration"])
            temporary.replace(directory / "lightgbm.txt")
        result = {**task, "preregistration_sha256": registration_sha,
                  "train_n": len(train), "validation_panel_n": len(val), "design_width": x.shape[1],
                  "class_order": list(CLASS_ORDER), "train_class_priors": priors.tolist(),
                  "logistic_path": str(directory / "logistic.pkl"),
                  "lightgbm_path": str(directory / "lightgbm.txt"),
                  "artifact_sha256": {name: root._sha(directory / name) for name in ("logistic.pkl", "lightgbm.txt")},
                  "logistic": logistic_history, "lightgbm": lightgbm_history,
                  "test_predictions_produced": False, "elapsed_seconds": time.monotonic() - started}
        root._json(previous_path, result)
        return {key: result[key] for key in ("job", "condition", "fold", "elapsed_seconds")}
