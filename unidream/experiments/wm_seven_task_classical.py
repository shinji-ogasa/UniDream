"""Train-fit, validation-selected baselines for the seven-task benchmark.

The public fitting function deliberately accepts only Train and validation
arrays.  It never opens experiment artifacts or obtains Test targets.  Trees
consume the original feature coordinates; the linear models share a scaler
estimated exclusively on Train.  Bundles contain ordinary arrays and fitted
estimators and can be pickled by the experiment coordinator.
"""
from __future__ import annotations

from collections.abc import Mapping
import time
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .wm_ablation_evaluation import variance_qlike
from .wm_comparison_classical import (
    GARCHFit, _window_returns, ewma_variance_forecast, fit_gap_reset_garch,
    garch_variance_forecast, log_returns_15m,
)


TASKS = ("return", "variance", "downside", "crash", "dd_excess", "advantage", "recovery")
CONTINUOUS_TASKS = tuple(task for task in TASKS if task != "crash")
NONNEGATIVE_TASKS = ("variance", "downside", "dd_excess")
VARIANCE_FLOOR = 1e-12


def _features(values, name, *, width=None):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError(f"{name} must be a nonempty two-dimensional feature matrix")
    if width is not None and values.shape[1] != width:
        raise ValueError(f"{name} has a different feature width from Train")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains nonfinite features")
    return values


def _targets(values, n, name):
    if not isinstance(values, Mapping) or set(values) != set(TASKS):
        raise ValueError(f"{name} must contain exactly the seven named tasks")
    result = {}
    for task in TASKS:
        vector = np.asarray(values[task], dtype=np.float64)
        if vector.shape != (n,) or not np.isfinite(vector).all():
            raise ValueError(f"{name}[{task}] must be a finite vector matching feature rows")
        if task in NONNEGATIVE_TASKS and np.any(vector < 0):
            raise ValueError(f"{name}[{task}] cannot be negative")
        if task == "crash" and not np.isin(vector, (0., 1.)).all():
            raise ValueError(f"{name}[crash] must contain binary event targets")
        if task == "recovery" and np.any(np.abs(vector) > 1. + 1e-12):
            raise ValueError(f"{name}[recovery] must lie in [-1, 1]")
        result[task] = vector
    return result


def _rows(n, maximum):
    if maximum is None:
        return np.arange(n, dtype=np.int64)
    maximum = int(maximum)
    if maximum < 2:
        raise ValueError("maximum fitting rows must be at least two or None")
    return np.linspace(0, n - 1, min(n, maximum), dtype=np.int64)


def _project(task, predicted):
    predicted = np.asarray(predicted, dtype=np.float64)
    if not np.isfinite(predicted).all():
        raise ValueError(f"nonfinite predictions for {task}")
    if task == "variance":
        return np.maximum(predicted, VARIANCE_FLOOR)
    if task in ("downside", "dd_excess"):
        return np.maximum(predicted, 0.)
    if task == "crash":
        return np.clip(predicted, 0., 1.)
    if task == "recovery":
        return np.clip(predicted, -1., 1.)
    return predicted


def task_selection_score(task, actual, predicted, train_scale):
    """The fixed task-specific validation loss; lower is always better."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = _project(task, predicted)
    if actual.shape != predicted.shape or actual.ndim != 1 or not np.isfinite(actual).all():
        raise ValueError("selection actual/predicted arrays must be matching finite vectors")
    if not np.isfinite(train_scale) or train_scale <= 0:
        raise ValueError("selection normalization must be a positive Train-only scale")
    if task == "variance":
        return float(np.mean(variance_qlike(actual, predicted)))
    if task == "crash":
        return float(np.mean(np.square(actual - predicted)))
    if task in ("downside", "dd_excess"):
        return float(np.mean(np.abs(actual - predicted)) / train_scale)
    return float(np.mean(np.square((actual - predicted) / train_scale)))


def _scales(targets):
    return {
        task: max(float(np.mean(targets[task])) if task in NONNEGATIVE_TASKS
                  else float(np.sqrt(np.mean(np.square(targets[task])))),
                  VARIANCE_FLOOR if task in NONNEGATIVE_TASKS else 1e-8)
        for task in TASKS
    }


def _ridge_candidates(x, targets, xv, val_targets, scales, alphas):
    """Share an exact primal/dual Gram matrix across all six regression heads."""
    x_mean = x.mean(axis=0)
    centered = x - x_mean
    ys = np.column_stack([targets[task] / scales[task] for task in CONTINUOUS_TASKS])
    y_mean = ys.mean(axis=0)
    centered_y = ys - y_mean
    primal = centered.shape[1] <= centered.shape[0]
    gram = centered.T @ centered if primal else centered @ centered.T
    cross = centered.T @ centered_y if primal else centered_y
    diagonal = np.diag_indices_from(gram)
    original_diagonal = gram[diagonal].copy()
    candidates = {task: [] for task in CONTINUOUS_TASKS}
    for alpha in alphas:
        alpha = float(alpha)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge alphas must be finite and positive")
        gram[diagonal] = original_diagonal + alpha
        factor = cho_factor(gram, lower=True, check_finite=False)
        solved = cho_solve(factor, cross, check_finite=False)
        coefficients = solved if primal else centered.T @ solved
        intercept = y_mean - x_mean @ coefficients
        prediction = xv @ coefficients + intercept
        for column, task in enumerate(CONTINUOUS_TASKS):
            score = task_selection_score(task, val_targets[task], prediction[:, column] * scales[task], scales[task])
            candidates[task].append({"alpha": alpha, "score": score,
                                     "coefficients": coefficients[:, column].copy(),
                                     "intercept": float(intercept[column])})
    fits, metadata = {}, {}
    for task in CONTINUOUS_TASKS:
        if not candidates[task]:
            raise ValueError("at least one Ridge alpha is required")
        selected = min(candidates[task], key=lambda row: row["score"])
        fits[task] = {"kind": "ridge", "coefficients": selected["coefficients"],
                      "intercept": selected["intercept"], "target_scale": scales[task]}
        rows = [{"alpha": row["alpha"], "score": row["score"]} for row in candidates[task]]
        metadata[task] = {"selected": {"alpha": selected["alpha"], "score": selected["score"]},
                          "candidates": rows, "gram_form": "primal" if primal else "dual"}
    return fits, metadata


def _logistic_candidates(x, labels, xv, labels_val, scale, cfg):
    if len(np.unique(labels)) == 1:
        value = float(labels[0])
        score = task_selection_score("crash", labels_val, np.full(len(xv), value), scale)
        return {"kind": "constant", "value": value}, {
            "selected": {"score": score, "constant": value}, "candidates": [],
            "constant_train_class": True,
        }
    candidates = []
    for c in cfg.get("logistic_Cs", [.01, .1, 1.]):
        if not np.isfinite(c) or c <= 0:
            raise ValueError("logistic C values must be finite and positive")
        attempts = []
        for multiplier in (1, 4):
            model = LogisticRegression(C=float(c), solver="lbfgs", class_weight=None,
                                       max_iter=int(cfg.get("logistic_max_iter", 1000)) * multiplier,
                                       tol=float(cfg.get("logistic_tol", 1e-4)))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x, labels)
            issues = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
            attempts.append({"max_iter": model.max_iter, "n_iter": model.n_iter_.tolist(),
                             "convergence_warnings": issues})
            if not issues:
                break
        else:
            raise RuntimeError(f"crash logistic C={c} did not converge; no evaluation predictions produced")
        prediction = model.predict_proba(xv)[:, int(np.flatnonzero(model.classes_ == 1)[0])]
        score = task_selection_score("crash", labels_val, prediction, scale)
        candidates.append((score, model, {"C": float(c), "score": score, "attempts": attempts}))
    if not candidates:
        raise ValueError("at least one logistic C is required")
    _, model, selected = min(candidates, key=lambda row: row[0])
    return {"kind": "logistic", "model": model}, {
        "selected": selected, "candidates": [row[2] for row in candidates],
        "constant_train_class": False,
    }


def _tree_candidates(x, targets, xv, val_targets, scales, cfg, threads):
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
    # Construct the bins once, then replace labels for the seven independent
    # tasks.  Every booster is detached from these reusable Dataset objects.
    train_set = lgb.Dataset(x, label=np.zeros(len(x)), params=parameters, free_raw_data=False)
    val_set = lgb.Dataset(xv, label=np.zeros(len(xv)), reference=train_set,
                         params=parameters, free_raw_data=False)
    train_set.construct()
    val_set.construct()
    fits, metadata = {}, {}
    for task in TASKS:
        if task == "crash" and len(np.unique(targets[task])) == 1:
            value = float(targets[task][0])
            score = task_selection_score(task, val_targets[task], np.full(len(xv), value), scales[task])
            fits[task] = {"kind": "constant", "value": value}
            metadata[task] = {"selected": {"score": score, "constant": value},
                              "candidates": [], "constant_train_class": True}
            continue
        scale = 1. if task == "crash" else scales[task]
        objective = ("binary" if task == "crash" else "gamma" if task == "variance"
                     else "regression_l1" if task in ("downside", "dd_excess") else "regression")
        floor = VARIANCE_FLOOR if task == "variance" else -np.inf
        train_set.set_label(np.maximum(targets[task], floor) / scale)
        val_set.set_label(np.maximum(val_targets[task], floor) / scale)
        candidates = []
        for leaves in cfg.get("lightgbm_leaves", [7, 31]):
            history = []

            def metric(prediction, _dataset):
                score = task_selection_score(task, val_targets[task], np.asarray(prediction) * scale, scales[task])
                history.append(score)
                return "task_selection_loss", score, False

            candidate_parameters = {**parameters, "objective": objective, "num_leaves": int(leaves)}
            started = time.monotonic()
            model = lgb.train(candidate_parameters, train_set,
                              num_boost_round=int(cfg.get("lightgbm_rounds", 600)),
                              valid_sets=[val_set], valid_names=["selection_val"], feval=metric,
                              callbacks=[lgb.early_stopping(int(cfg.get("lightgbm_patience", 60)),
                                                           first_metric_only=True, verbose=False)],
                              keep_training_booster=True)
            if not history:
                raise RuntimeError(f"LightGBM {task} produced no validation observations")
            selected_iteration = int(np.argmin(history)) + 1
            predicted = model.predict(xv, num_iteration=selected_iteration, num_threads=threads) * scale
            score = task_selection_score(task, val_targets[task], predicted, scales[task])
            # Detach each fitted booster before the shared Dataset labels are
            # changed.  The selected iteration is explicit on every predict.
            model.free_dataset()
            row = {"num_leaves": int(leaves), "score": score,
                   "selected_iteration": selected_iteration, "observed_rounds": len(history),
                   "objective": objective, "history": history,
                   "elapsed_seconds": time.monotonic() - started}
            candidates.append((score, model, row))
        if not candidates:
            raise ValueError("at least one LightGBM leaf candidate is required")
        _, model, selected = min(candidates, key=lambda item: item[0])
        fits[task] = {"kind": "lightgbm", "model": model, "target_scale": scale,
                      "selected_iteration": selected["selected_iteration"]}
        metadata[task] = {"selected": selected, "candidates": [item[2] for item in candidates]}
    return fits, metadata


def fit_classical_bundle(x_train, targets_train, x_val, targets_val, config=None):
    """Fit on Train, select per task on validation; no Test argument exists.

    Optional row caps are deterministic, evenly spaced subsets of Train.  The
    default linear cap is 8,192; trees use all Train rows unless explicitly
    capped.  The scaler and normalization constants always use all Train.
    """
    cfg = dict(config or {})
    threads = int(cfg.get("cpu_threads", 2))
    if threads < 1:
        raise ValueError("cpu_threads must be positive")
    x_train = _features(x_train, "x_train")
    x_val = _features(x_val, "x_val", width=x_train.shape[1])
    targets_train = _targets(targets_train, len(x_train), "targets_train")
    targets_val = _targets(targets_val, len(x_val), "targets_val")
    scales = _scales(targets_train)
    means = {task: float(np.mean(targets_train[task])) for task in TASKS}
    unconditional = {task: {"kind": "constant", "value": 0. if task in ("return", "advantage") else means[task]}
                     for task in TASKS}
    linear_rows = _rows(len(x_train), cfg.get("linear_max_train_rows", 8192))
    tree_rows = _rows(len(x_train), cfg.get("lightgbm_max_train_rows"))
    # NumPy's float64 reductions avoid a full float64 copy of the potentially
    # multi-gigabyte original design matrix.
    mean = np.mean(x_train, axis=0, dtype=np.float64)
    std = np.std(x_train, axis=0, dtype=np.float64)
    std = np.where(std > 1e-12, std, 1.)
    linear_train = (np.asarray(x_train[linear_rows], dtype=np.float64) - mean) / std
    linear_val = (np.asarray(x_val, dtype=np.float64) - mean) / std
    linear_targets = {task: vector[linear_rows] for task, vector in targets_train.items()}
    started = time.monotonic()
    with threadpool_limits(limits=threads):
        linear, linear_meta = _ridge_candidates(linear_train, linear_targets, linear_val,
                                                targets_val, scales, cfg.get("ridge_alphas", [10., 1000., 100000.]))
        linear["crash"], linear_meta["crash"] = _logistic_candidates(
            linear_train, linear_targets["crash"], linear_val, targets_val["crash"], scales["crash"], cfg)
        trees, tree_meta = _tree_candidates(x_train[tree_rows],
                                            {task: vector[tree_rows] for task, vector in targets_train.items()},
                                            x_val, targets_val, scales, cfg, threads)
    return {
        "format": "seven_task_classical_v1", "tasks": TASKS, "n_features": int(x_train.shape[1]),
        "scaler": {"mean": mean, "scale": std},
        "fits": {"unconditional": unconditional, "linear": linear, "lightgbm": trees},
        "metadata": {
            "selection_split": "validation", "test_predictions_produced": False,
            "train_n": len(x_train), "validation_n": len(x_val), "cpu_threads": threads,
            "train_scales": scales, "train_means": means,
            "linear_train_rows": linear_rows.tolist(), "lightgbm_train_rows": tree_rows.tolist(),
            "linear_scaler_fit": "all Train rows only", "tree_feature_coordinates": "unscaled",
            "models": {"linear": linear_meta, "lightgbm": tree_meta},
            "elapsed_seconds": time.monotonic() - started,
        },
    }


def predict_classical_bundle(bundle, x):
    """Predict all seven tasks without accepting targets or changing fits."""
    if bundle.get("format") != "seven_task_classical_v1":
        raise ValueError("unrecognized seven-task classical bundle")
    x = _features(x, "prediction features", width=bundle["n_features"])
    standardized = (np.asarray(x, dtype=np.float64) - bundle["scaler"]["mean"]) / bundle["scaler"]["scale"]
    result = {}
    threads = bundle["metadata"]["cpu_threads"]
    with threadpool_limits(limits=threads):
        for name, heads in bundle["fits"].items():
            predictions = {}
            for task in TASKS:
                head = heads[task]
                if head["kind"] == "constant":
                    predicted = np.full(len(x), head["value"], dtype=np.float64)
                elif head["kind"] == "ridge":
                    predicted = (standardized @ head["coefficients"] + head["intercept"]) * head["target_scale"]
                elif head["kind"] == "logistic":
                    model = head["model"]
                    predicted = model.predict_proba(standardized)[:, int(np.flatnonzero(model.classes_ == 1)[0])]
                elif head["kind"] == "lightgbm":
                    predicted = head["model"].predict(x, num_iteration=head["selected_iteration"],
                                                      num_threads=threads) * head["target_scale"]
                else:
                    raise ValueError(f"unknown fitted task head {head['kind']}")
                predictions[task] = _project(task, predicted)
            result[name] = predictions
    return result


def fit_variance_filters(close, index_ns, train_start_ns, train_end_ns,
                         earlystop_ids, actual_variance_earlystop, horizon=16,
                         context=256, decays=(.94, .97, .99)):
    """Fit Train GARCH and select EWMA decay on the supplied early-Val panel.

    ``close`` and ``index_ns`` describe the complete 15-minute grid, retaining
    missing-price rows.  Train returns are calculated *after* slicing Train
    closes; the first return is therefore unavailable, and missing prices
    reset the GARCH likelihood.  No Test labels or final-selection Val labels
    are accepted.  A failed optimizer is saved verbatim and never replaced by
    another model.  All metadata is JSON-friendly as well as picklable.
    """
    prices = np.asarray(close, dtype=np.float64)
    supplied_index = np.asarray(index_ns)
    if prices.ndim != 1 or supplied_index.shape != prices.shape or not len(prices):
        raise ValueError("close and index_ns must be matching nonempty vectors")
    if not np.issubdtype(supplied_index.dtype, np.integer):
        raise ValueError("index_ns must contain integer UTC nanoseconds")
    timestamps = supplied_index.astype(np.int64, copy=False)
    if np.any(np.diff(timestamps) != 15 * 60 * 1_000_000_000):
        raise ValueError("variance filters require the complete regular 15-minute grid")
    if int(horizon) != horizon or horizon < 1 or int(context) != context or context < 17:
        raise ValueError("horizon must be positive and context must retain at least 16 returns")
    horizon, context = int(horizon), int(context)
    if int(train_start_ns) != train_start_ns or int(train_end_ns) != train_end_ns or train_end_ns <= train_start_ns:
        raise ValueError("Train timestamps must be increasing integer nanoseconds")
    train_mask = (timestamps >= train_start_ns) & (timestamps < train_end_ns)
    train_returns = log_returns_15m(prices[train_mask])
    finite_train = train_returns[np.isfinite(train_returns)]
    if len(finite_train) < 10:
        raise ValueError("at least ten finite within-Train log returns are required")
    one_bar_variance = float(np.mean(np.square(finite_train)))
    if not np.isfinite(one_bar_variance) or one_bar_variance <= 0:
        raise ValueError("Train one-bar mean squared return must be finite and positive")
    supplied_ids = np.asarray(earlystop_ids)
    if supplied_ids.ndim != 1 or not len(supplied_ids):
        raise ValueError("earlystop_ids must be a nonempty chronological vector")
    ids = supplied_ids.astype(np.int64)
    if (not np.equal(supplied_ids, ids).all() or np.any(ids < context)
            or np.any(ids >= len(prices)) or np.any(np.diff(ids) <= 0)):
        raise ValueError("earlystop_ids must be unique increasing valid decision endpoints")
    if np.any(timestamps[ids] < train_end_ns):
        raise ValueError("early-stop validation decisions cannot be inside Train")
    actual = np.asarray(actual_variance_earlystop, dtype=np.float64)
    if actual.shape != ids.shape or not np.isfinite(actual).all() or np.any(actual < 0):
        raise ValueError("early-stop realized variance must be a matching finite nonnegative vector")
    candidates = []
    for decay in decays:
        forecast = ewma_variance_forecast(prices, ids, initial_variance=one_bar_variance,
                                         decay=float(decay), context=context, horizon=horizon)
        if not np.isfinite(forecast).all():
            raise ValueError("early-stop decisions lack a complete finite price context")
        score = float(np.mean(variance_qlike(actual, forecast)))
        candidates.append({"decay": float(decay), "score": score})
    if not candidates:
        raise ValueError("at least one EWMA decay is required")
    selected = min(candidates, key=lambda row: row["score"])
    started = time.monotonic()
    garch = fit_gap_reset_garch(train_returns, initial_variance=one_bar_variance,
                              return_scale=100., maxiter=500)
    return {
        "format": "seven_task_variance_filters_v1", "horizon": horizon,
        "context": context, "past_variance_lookback": 16,
        "one_bar_variance": one_bar_variance, "ewma_decay": selected["decay"],
        "ewma_selection_score": selected["score"], "ewma_candidates": candidates,
        "garch": garch.as_dict(), "garch_success": bool(garch.success),
        "metadata": {
            "train_start_ns": int(train_start_ns), "train_end_ns": int(train_end_ns),
            "train_price_n": int(np.count_nonzero(train_mask)),
            "train_return_n": int(len(finite_train)), "earlystop_n": len(ids),
            "earlystop_first_ns": int(timestamps[ids[0]]),
            "earlystop_last_ns": int(timestamps[ids[-1]]),
            "ewma_selection_split": "supplied first validation month only",
            "garch_fit_split": "Train only", "garch_maxiter": 500,
            "test_labels_used": False, "failed_garch_fallback": None,
            "garch_elapsed_seconds": time.monotonic() - started,
        },
    }


def predict_variance_filters(close, decision_ids, fit):
    """Return available variance-only forecasts in original return units.

    Missing observations within a complete context propagate NaN equally for
    every filter.  Coordinators must use their common eligible-row mask rather
    than filling those observations or shortening the price history.
    """
    if fit.get("format") != "seven_task_variance_filters_v1":
        raise ValueError("unrecognized variance-filter fit")
    context, horizon = int(fit["context"]), int(fit["horizon"])
    lookback = int(fit["past_variance_lookback"])
    observed = _window_returns(close, decision_ids, context)
    past = np.sum(np.square(observed[:, -lookback:]), axis=1) * horizon / lookback
    result = {
        "past_variance": np.maximum(past, VARIANCE_FLOOR),
        "ewma": ewma_variance_forecast(close, decision_ids,
                                       initial_variance=fit["one_bar_variance"],
                                       decay=fit["ewma_decay"], context=context, horizon=horizon),
    }
    if fit["garch_success"]:
        garch = GARCHFit(**fit["garch"])
        result["garch"] = garch_variance_forecast(close, decision_ids, garch,
                                                 initial_variance=fit["one_bar_variance"],
                                                 context=context, horizon=horizon)
    return result
