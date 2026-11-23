"""Pure targets, scoring and paired inference for the three-class experiment.

Labels are 0=down, 1=flat and 2=up; equality at either threshold is flat.
Probability forecasts are trained with ordinary, unweighted cross entropy.
The primary decision is argmax(p / Train class prior), with deterministic
0/1/2 tie order.  Raw argmax is an auxiliary decision rule.  This changes
decisions, not the probabilities used for proper probability scores.

All primary/auxiliary classification metrics use a fixed three-class label
space. Balanced accuracy is undefined when an actual class is absent; an
absent class never silently turns a three-class task into a two-class task.
The multiclass Brier score is the mean SUM of three squared errors (0..2).

Inference pools seed confusion counts, rather than averaging seed metrics or
ensembling probabilities. At each timestamp confusion counts and log losses
are averaged over seeds before UTC-day moving-block resampling. Blocks cannot
cross a missing 15-minute timestamp or a fold boundary.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


NUMERIC_FLOOR = 1.0e-12
CLASSES = (0, 1, 2)
BAR_INTERVAL = pd.Timedelta(minutes=15)


def _finite_vector(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional vector")
    return value


def make_labels(returns: np.ndarray, theta: float | np.ndarray) -> np.ndarray:
    """Return down/flat/up labels; both threshold equalities are flat."""
    returns = np.asarray(returns, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if not returns.size or not np.isfinite(returns).all():
        raise ValueError("returns must contain finite values")
    if theta.shape not in ((), returns.shape):
        raise ValueError("theta must be scalar or match the returns shape")
    if not np.isfinite(theta).all() or not (theta > 0).all():
        raise ValueError("theta must be positive and finite")
    return np.where(returns < -theta, 0, np.where(returns > theta, 2, 1)).astype(np.int64)


def calibrate_thresholds(
    oof_returns: Sequence[float] | np.ndarray,
    oof_variance: Sequence[float] | np.ndarray,
    *,
    train_returns: Sequence[float] | np.ndarray,
) -> dict[str, float | int]:
    """Fit a and the common fixed threshold from explicitly Train-only inputs.

    a = mean(OOF_return**2 / max(OOF_variance,1e-12)). The fixed threshold
    uses FULL Train returns, not an OOF subset: .5*sqrt(mean(Train_return**2)).
    This function does not construct OOF predictions or check their provenance;
    the caller must supply genuinely chronological Train OOF observations.
    """
    returns = _finite_vector(oof_returns, "oof_returns")
    variance = _finite_vector(oof_variance, "oof_variance")
    train = _finite_vector(train_returns, "train_returns")
    if returns.shape != variance.shape or np.any(variance < 0):
        raise ValueError("OOF returns/variance must match and variance must be nonnegative")
    a = float(np.mean(returns ** 2 / np.maximum(variance, NUMERIC_FLOOR)))
    fixed = float(0.5 * np.sqrt(np.mean(train ** 2)))
    if not np.isfinite(a) or a <= 0 or not np.isfinite(fixed) or fixed <= 0:
        raise ValueError("degenerate Train outcomes cannot identify positive thresholds")
    return {"a": a, "fixed_theta": fixed, "n_oof": len(returns), "n_train": len(train), "variance_floor": NUMERIC_FLOOR}


def variance_thresholds(variance: Sequence[float] | np.ndarray, a: float) -> np.ndarray:
    """Apply the already-fitted Train scalar to another split without refitting."""
    variance = np.asarray(variance, dtype=np.float64)
    if not variance.size or not np.isfinite(variance).all() or np.any(variance < 0):
        raise ValueError("variance must contain finite nonnegative values")
    if not np.isfinite(a) or a <= 0:
        raise ValueError("a must be a positive finite Train-fitted scalar")
    theta = 0.5 * np.sqrt(float(a) * np.maximum(variance, NUMERIC_FLOOR))
    if not np.isfinite(theta).all() or not (theta > 0).all():
        raise ValueError("calibrated theta must be positive and finite")
    return theta


def _probabilities(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 3 or not len(probs):
        raise ValueError("probabilities must have shape [n,3]")
    if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("probabilities must be finite and lie in [0,1]")
    totals = probs.sum(1, keepdims=True)
    if not np.allclose(totals, 1.0, rtol=0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    return probs / totals


def _labels(value: np.ndarray, n: int, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != (n,) or not np.isin(value, CLASSES).all():
        raise ValueError(f"{name} must be one integer class 0/1/2 per probability row")
    return value.astype(np.int64)


def balanced_prior_predictions(probs: np.ndarray, train_priors: np.ndarray) -> np.ndarray:
    """Argmax(p/Train_prior), including deterministic first-class tie breaking."""
    probs = _probabilities(probs)
    priors = np.asarray(train_priors, dtype=np.float64)
    if priors.shape not in ((3,), probs.shape):
        raise ValueError("Train priors must have shape [3] or [n,3]")
    if not np.isfinite(priors).all() or not (priors > 0).all():
        raise ValueError("all Train class priors must be strictly positive and finite")
    if not np.allclose(priors.sum(axis=-1), 1.0, rtol=0, atol=1e-6):
        raise ValueError("Train priors must sum to one")
    return np.argmax(probs / priors, axis=1).astype(np.int64)


def _confusion(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.bincount(actual * 3 + predicted, minlength=9).reshape(3, 3)


def _confusion_statistics(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics of one [3,3] matrix or a batch [...,3,3] of pooled matrices."""
    matrix = np.asarray(matrix, dtype=np.float64)
    support = matrix.sum(axis=-1)
    predicted = matrix.sum(axis=-2)
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    total = support.sum(axis=-1)
    recall = np.divide(diagonal, support, out=np.full_like(diagonal, np.nan), where=support > 0)
    precision = np.divide(diagonal, predicted, out=np.zeros_like(diagonal), where=predicted > 0)
    denominator = support + predicted
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal), where=denominator > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"accuracy": diagonal.sum(axis=-1) / total,
                "balanced_accuracy": recall.mean(axis=-1), "macro_f1": f1.mean(axis=-1),
                "precision": precision, "recall": recall}


def _decision_scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    matrix = _confusion(actual, predicted)
    scores = _confusion_statistics(matrix)
    return {"accuracy": float(scores["accuracy"]),
            "balanced_accuracy": float(scores["balanced_accuracy"]) if np.isfinite(scores["balanced_accuracy"]) else None,
            "macro_f1": float(scores["macro_f1"]),
            "per_class_precision": scores["precision"].tolist(),
            "per_class_recall": [float(v) if np.isfinite(v) else None for v in scores["recall"]],
            "confusion_matrix": matrix.tolist(),
            "actual_class_frequency": (matrix.sum(1) / len(actual)).tolist(),
            "predicted_class_frequency": (matrix.sum(0) / len(actual)).tolist(),
            "neutral_fraction": float(np.mean(predicted == 1))}


def score_classification(
    actual: np.ndarray,
    probs: np.ndarray,
    train_priors: np.ndarray | None,
    *,
    predictions: np.ndarray | None = None,
    raw_predictions: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score probabilities and pooled primary/auxiliary decisions.

    Explicit decision arrays support pooled folds with different priors and
    the Train-majority baseline. That baseline may retain Train-prior
    probabilities for log loss while explicitly predicting the Train majority;
    applying p/prior to it would instead yield a three-way tie.
    """
    probs = _probabilities(probs)
    actual = _labels(actual, len(probs), "actual")
    if train_priors is not None:
        default = balanced_prior_predictions(probs, train_priors)
    elif predictions is None:
        raise ValueError("Train priors are required unless explicit primary decisions are supplied")
    else:
        default = None
    predicted = default if predictions is None else _labels(predictions, len(probs), "predictions")
    raw = np.argmax(probs, axis=1) if raw_predictions is None else _labels(raw_predictions, len(probs), "raw_predictions")
    result = {"n": len(actual), **_decision_scores(actual, predicted)}
    result["log_loss"] = float(-np.log(np.maximum(probs[np.arange(len(actual)), actual], NUMERIC_FLOOR)).mean())
    result["multiclass_brier"] = float(np.square(probs - np.eye(3)[actual]).sum(1).mean())
    result["raw"] = _decision_scores(actual, raw)
    return result


def _row_statistics(actual: np.ndarray, probs: np.ndarray, predicted: np.ndarray, raw: np.ndarray) -> np.ndarray:
    stats = np.zeros((len(actual), 19), dtype=np.float64)
    rows = np.arange(len(actual))
    stats[rows, actual * 3 + predicted] = 1.0
    stats[rows, 9 + actual * 3 + raw] = 1.0
    stats[:, 18] = -np.log(np.maximum(probs[rows, actual], NUMERIC_FLOOR))
    return stats


def _paired_daily_statistics(
    frame: pd.DataFrame, model_a: str, model_b: str, condition: str,
) -> tuple[pd.DataFrame, int, int]:
    if model_a == model_b:
        raise ValueError("paired models must be distinct")
    keys = ["timestamp", "fold", "seed"]
    payload = ["actual", "p0", "p1", "p2", "pred", "pred_raw"]
    parts = []
    for model in (model_a, model_b):
        part = frame.loc[(frame["condition"] == condition) & (frame["model"] == model)].copy()
        part["timestamp"] = pd.to_datetime(part["timestamp"], utc=True)
        if not len(part) or part.duplicated(keys).any() or part[keys].isna().any().any():
            raise ValueError("each model needs unique nonempty timestamp/fold/seed rows in the same condition")
        if np.any(part["timestamp"].astype("int64").to_numpy() % BAR_INTERVAL.value):
            raise ValueError("prediction timestamps must align to UTC 15-minute boundaries")
        probs = _probabilities(part[["p0", "p1", "p2"]].to_numpy())
        actual = _labels(part["actual"].to_numpy(), len(part), "actual")
        predicted = _labels(part["pred"].to_numpy(), len(part), "pred")
        raw = _labels(part["pred_raw"].to_numpy(), len(part), "pred_raw")
        if not np.array_equal(raw, np.argmax(probs, axis=1)):
            raise ValueError("pred_raw must equal the deterministic raw probability argmax")
        computed_ll = -np.log(np.maximum(probs[np.arange(len(part)), actual], NUMERIC_FLOOR))
        for name in ("logloss", "log_loss"):
            if name in part and not np.allclose(part[name], computed_ll, rtol=1e-6, atol=1e-6):
                raise ValueError("saved log loss does not match the probability forecasts")
        normalized = part[keys + payload].copy()
        normalized[["p0", "p1", "p2"]] = probs
        parts.append(normalized)
    paired = parts[0].merge(parts[1], on=keys, suffixes=("_a", "_b"), how="outer", indicator=True, validate="one_to_one")
    if not (paired["_merge"] == "both").all():
        raise ValueError("paired models must share identical timestamp/fold/seed coverage")
    if not np.array_equal(paired["actual_a"], paired["actual_b"]):
        raise ValueError("paired models in one condition must share identical actual classes")
    if (paired.groupby(["fold", "timestamp"])["actual_a"].nunique() != 1).any():
        raise ValueError("all seeds must share identical actual classes")
    n_seeds = int(paired["seed"].nunique())
    if not (paired.groupby(["fold", "timestamp"])["seed"].nunique() == n_seeds).all():
        raise ValueError("every timestamp must contain the same complete seed set")
    stats = []
    for suffix in ("a", "b"):
        stats.append(_row_statistics(paired[f"actual_{suffix}"].to_numpy(dtype=int),
            paired[[f"p0_{suffix}", f"p1_{suffix}", f"p2_{suffix}"]].to_numpy(),
            paired[f"pred_{suffix}"].to_numpy(dtype=int), paired[f"pred_raw_{suffix}"].to_numpy(dtype=int)))
    columns = [f"s{i}" for i in range(38)]
    losses = pd.DataFrame(np.concatenate(stats, axis=1), columns=columns)
    losses["fold"] = paired["fold"].to_numpy()
    losses["timestamp"] = paired["timestamp"].to_numpy()
    averaged = losses.groupby(["fold", "timestamp"], as_index=False)[columns].mean().sort_values(["timestamp", "fold"])
    if averaged["timestamp"].duplicated().any():
        raise ValueError("test folds must not overlap")
    breaks = (averaged["timestamp"].diff() != BAR_INTERVAL) | (averaged["fold"] != averaged["fold"].shift())
    averaged["segment"] = breaks.cumsum().to_numpy()
    averaged["date"] = averaged["timestamp"].dt.floor("D")
    averaged["count"] = 1.0
    daily = averaged.groupby(["segment", "date"], as_index=False)[[*columns, "count"]].sum()
    return daily, len(averaged), n_seeds


def _paired_statistics(sums: np.ndarray) -> dict[str, np.ndarray]:
    sums = np.asarray(sums, dtype=np.float64)
    shape = sums.shape[:-1] + (3, 3)
    a = _confusion_statistics(sums[..., :9].reshape(shape))
    b = _confusion_statistics(sums[..., 19:28].reshape(shape))
    raw_a = _confusion_statistics(sums[..., 9:18].reshape(shape))
    raw_b = _confusion_statistics(sums[..., 28:37].reshape(shape))
    result = {name + "_difference": b[name] - a[name] for name in ("balanced_accuracy", "accuracy", "macro_f1")}
    result.update({"raw_" + name + "_difference": raw_b[name] - raw_a[name] for name in ("balanced_accuracy", "accuracy", "macro_f1")})
    result["log_loss_improvement"] = (sums[..., 18] - sums[..., 37]) / sums[..., 38]
    return result


def paired_classification_block_bootstrap(
    frame: pd.DataFrame,
    model_a: str,
    model_b: str,
    *,
    condition: str,
    block_days: Sequence[int] = (1, 7, 28),
    replicates: int = 2000,
    random_seed: int = 20261110,
) -> list[dict[str, Any]]:
    """Paired block CIs; positive differences mean model B performs better.

    The primary balanced-accuracy P value is a two-sided, centered-bootstrap
    approximation, suitable for the separately preregistered Holm family. The
    95% intervals themselves are pointwise. Absent-class resamples are counted
    and excluded for BA rather than silently changing its class universe.
    Deterministic replay of entire short segments does not produce inference.
    """
    if replicates < 2 or int(replicates) != replicates:
        raise ValueError("replicates must be an integer of at least two")
    if not block_days or any(int(days) != days or days < 1 for days in block_days):
        raise ValueError("block_days must contain positive whole days")
    daily, n_timestamps, n_seeds = _paired_daily_statistics(frame, model_a, model_b, condition)
    cols = [f"s{i}" for i in range(38)] + ["count"]
    segments = [part[cols].to_numpy(dtype=float) for _, part in daily.groupby("segment", sort=True)]
    total = daily[cols].sum().to_numpy(dtype=float)
    estimate = _paired_statistics(total)
    result = []
    for supplied_days in block_days:
        days = int(supplied_days)
        rng = np.random.default_rng(np.random.SeedSequence([int(random_seed), days]))
        resampled = np.zeros((int(replicates), len(cols)), dtype=float)
        n_short = 0
        n_resampleable = 0
        resampleable_count = 0.0
        for segment in segments:
            length = len(segment)
            block = min(days, length)
            n_short += length < days
            n_resampleable += length > block
            if length > block:
                resampleable_count += float(segment[:, -1].sum())
            n_blocks = (length + block - 1) // block
            starts = rng.integers(0, length - block + 1, size=(int(replicates), n_blocks))
            indices = (starts[:, :, None] + np.arange(block)).reshape(int(replicates), -1)[:, :length]
            resampled += segment[indices].sum(axis=1)
        bootstrap = _paired_statistics(resampled)
        metrics = {}
        for name, supplied_value in estimate.items():
            value = float(supplied_value)
            distribution = bootstrap[name]
            finite = distribution[np.isfinite(distribution)]
            row = {"estimate": value if np.isfinite(value) else None, "ci_low": None, "ci_high": None,
                   "n_valid_replicates": int(len(finite)), "n_undefined_replicates": int(replicates - len(finite))}
            if name == "balanced_accuracy_difference":
                row["p_value"] = None
            if np.isfinite(value) and len(finite) and n_resampleable:
                low, high = np.quantile(finite, [0.025, 0.975])
                row.update(ci_low=float(low), ci_high=float(high))
                if name == "balanced_accuracy_difference":
                    row["p_value"] = float((1 + np.sum(np.abs(finite - value) >= abs(value))) / (len(finite) + 1))
            metrics[name] = row
        result.append({"condition": condition, "comparison": f"{model_b}-{model_a}",
                       "block_days": days, "replicates": int(replicates),
                       "n_paired_timestamps": int(n_timestamps), "n_seeds": n_seeds,
                       "n_segments": len(segments), "n_segments_shorter_than_block": int(n_short),
                       "n_resampleable_segments": int(n_resampleable),
                       "resampleable_timestamp_fraction": resampleable_count / total[-1],
                       "inference_available": bool(n_resampleable), "primary_inference": days == 7,
                       "metrics": metrics})
    return result
