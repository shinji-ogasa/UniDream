"""Matched native/auxiliary reports for the fresh return-only three-arm study.

The caller controls the global Test gate. This report fits or selects nothing.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .wm_classification_evaluation import balanced_prior_predictions
from .wm_return_formulation_evaluation import make_return_labels
from .wm_return_multitask_evaluation import (
    ARMS, MODELS, classification_metrics, coarse_return_projection,
    regression_metrics, scalar_classification_metrics,
)


PRIORS = [f"train_prior_{c}" for c in range(3)]
MEANS = [f"train_class_return_mean_{c}" for c in range(3)]
PROBABILITIES = [f"p{c}" for c in range(3)]
IDENTITY = ["fold", "seed", "decision_id", "timestamp"]
OUTCOMES = ["actual_return", "actual_class", "theta", "train_rms", *PRIORS, *MEANS]
DAILY_KEYS = ["split", "group", "model", "arm", "fold", "seed", "utc_day", "segment_id"]
REGRESSION_SUMS = ["pred_sum", "pred2_sum", "actual_pred_sum", "sse_sum"]
CLASSIFICATION_SUMS = [*(f"prior_correct_c{c}" for c in range(3)),
                       *(f"raw_correct_c{c}" for c in range(3)), "log_loss_sum", "brier_sum"]
SUM_COLUMNS = ["n_rows", "actual_sum", "actual2_sum", *REGRESSION_SUMS,
               *(f"actual_c{c}" for c in range(3)), *CLASSIFICATION_SUMS]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_multitask_predictions(predictions, config=None):
    frame = predictions.copy()
    required = {*IDENTITY, *OUTCOMES, *PROBABILITIES, "pred_return", "group", "model", "arm", "split"}
    if missing := required - set(frame):
        raise ValueError(f"missing multitask prediction columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("nonempty matched predictions required")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    if frame.timestamp.isna().any() or np.any(frame.timestamp.astype("int64") % (15 * 60 * 10**9)):
        raise ValueError("timestamps must align to the UTC 15-minute grid")
    if not frame.split.isin(("selection", "test")).all():
        raise ValueError("report accepts only third-Val-month selection and report-only Test")
    if set(frame.group) != {"D"} or set(frame.seed) != {7}:
        raise ValueError("registered study requires group D and seed 7")
    if not frame.model.isin(MODELS).all() or not frame.arm.isin(ARMS).all():
        raise ValueError("unregistered model or multitask arm")
    for name in ("fold", "seed", "decision_id", "actual_class"):
        values = frame[name].to_numpy(float)
        if not np.isfinite(values).all() or np.any(values != np.floor(values)) or np.any(values < 0):
            raise ValueError(f"{name} must contain finite nonnegative integers")
    if not np.isfinite(frame[OUTCOMES].to_numpy(float)).all():
        raise ValueError("outcomes and Train contract values must be finite")
    if (not frame.actual_class.isin((0, 1, 2)).all() or (frame.train_rms <= 0).any()
            or not np.allclose(frame.theta, .5 * frame.train_rms, rtol=1e-10, atol=1e-14)):
        raise ValueError("labels need fixed positive half-Train-RMS thresholds")
    labels = make_return_labels(frame.actual_return.to_numpy(), frame.theta.to_numpy())
    if not np.array_equal(labels, frame.actual_class):
        raise ValueError("actual classes do not match fixed Train thresholds")
    priors = frame[PRIORS].to_numpy(float)
    if np.any(priors <= 0) or not np.allclose(priors.sum(1), 1., rtol=0, atol=1e-6):
        raise ValueError("Train priors must be strictly positive and sum to one")
    for _, part in frame.groupby("fold", sort=False):
        if (part[["theta", "train_rms", *PRIORS, *MEANS]].nunique() != 1).any():
            raise ValueError("each fold requires one immutable Train threshold/prior/class-mean contract")
    if frame.duplicated(["split", "model", "arm", *IDENTITY]).any():
        raise ValueError("duplicate prediction decision keys")
    expected_cells = {(model, arm) for model in MODELS for arm in ARMS}
    for split, part in frame.groupby("split", sort=False):
        actual_cells = set(part[["model", "arm"]].itertuples(index=False, name=None))
        if actual_cells != expected_cells:
            raise ValueError("each split requires all six model/arm conditions")
        reference = None
        for _, panel in part.groupby(["model", "arm"], sort=True):
            cohort = panel.sort_values(IDENTITY)[IDENTITY + OUTCOMES].reset_index(drop=True)
            if config and "folds" in config and set(cohort.fold) != set(config["folds"]):
                raise ValueError("report is missing a registered fold")
            if reference is None:
                reference = cohort
                if cohort.timestamp.duplicated().any():
                    raise ValueError("evaluated folds must not overlap timestamp coverage")
            elif not reference.equals(cohort):
                raise ValueError("all arms/models must share identical decisions and outcomes")
    reg = frame.arm.isin(("reg_only", "joint"))
    cls = frame.arm.isin(("class_only", "joint"))
    if not np.isfinite(frame.loc[reg, "pred_return"]).all() or not frame.loc[~reg, "pred_return"].isna().all():
        raise ValueError("only trained regression heads may have native scalar predictions")
    if not frame.loc[~cls, PROBABILITIES].isna().all().all():
        raise ValueError("inactive classification heads must not claim native probabilities")
    # This validates all probability rows, their normalization, and priors.
    balanced_prior_predictions(frame.loc[cls, PROBABILITIES].to_numpy(), frame.loc[cls, PRIORS].to_numpy())
    return frame


def _metric_rows(panel, base):
    arm = panel.arm.iloc[0]
    actual, labels = panel.actual_return.to_numpy(), panel.actual_class.to_numpy(int)
    rows = []
    if arm in ("reg_only", "joint"):
        rows.append({**base, "task": "regression", "variant": "native", "native": True,
                     **regression_metrics(actual, panel.pred_return.to_numpy())})
    if arm in ("class_only", "joint"):
        probs, priors = panel[PROBABILITIES].to_numpy(), panel[PRIORS].to_numpy()
        rows.append({**base, "task": "classification", "variant": "native", "native": True,
                     **classification_metrics(labels, probs, priors)})
    if arm == "reg_only":
        rows.append({**base, "task": "classification", "variant": "auxiliary_scalar_threshold", "native": False,
                     **scalar_classification_metrics(labels, panel.pred_return.to_numpy(), panel.theta.to_numpy())})
    if arm == "class_only":
        projection = coarse_return_projection(panel[PROBABILITIES].to_numpy(), panel[MEANS].to_numpy())
        rows.append({**base, "task": "regression", "variant": "auxiliary_train_class_mean", "native": False,
                     **regression_metrics(actual, projection)})
    return rows


def daily_statistics(predictions):
    """Exact additive sufficient statistics; no original row ordering is lost."""
    records = []
    for (split, model, arm, fold, seed), panel in predictions.groupby(["split", "model", "arm", "fold", "seed"], sort=True):
        panel = panel.sort_values("timestamp")
        n = len(panel)
        actual, labels = panel.actual_return.to_numpy(float), panel.actual_class.to_numpy(int)
        day = pd.DataFrame({"timestamp": panel.timestamp.to_numpy(), "n_rows": np.ones(n, dtype=np.int64),
                            "actual_sum": actual, "actual2_sum": actual ** 2})
        for column in [*REGRESSION_SUMS, *CLASSIFICATION_SUMS]:
            day[column] = np.nan
        for c in range(3):
            day[f"actual_c{c}"] = (labels == c).astype(np.int64)
        if arm in ("reg_only", "joint"):
            scalar = panel.pred_return.to_numpy(float)
            day["pred_sum"], day["pred2_sum"] = scalar, scalar ** 2
            day["actual_pred_sum"], day["sse_sum"] = actual * scalar, (actual - scalar) ** 2
        if arm in ("class_only", "joint"):
            probs = panel[PROBABILITIES].to_numpy()
            prior = balanced_prior_predictions(probs, panel[PRIORS].to_numpy())
            raw = probs.argmax(axis=1)
            for c in range(3):
                day[f"prior_correct_c{c}"] = ((labels == c) & (prior == c)).astype(np.int64)
                day[f"raw_correct_c{c}"] = ((labels == c) & (raw == c)).astype(np.int64)
            day["log_loss_sum"] = -np.log(np.maximum(probs[np.arange(n), labels], 1e-12))
            day["brier_sum"] = np.square(probs - np.eye(3)[labels]).sum(1)
        stamps = pd.to_datetime(day.timestamp, utc=True)
        day["utc_day"] = stamps.dt.floor("D")
        day["segment_id"] = stamps.diff().ne(pd.Timedelta(minutes=15)).cumsum().to_numpy() - 1
        grouped = day.groupby(["utc_day", "segment_id"], sort=True)[SUM_COLUMNS].sum(min_count=1).reset_index()
        for name, value in {"split": split, "group": "D", "model": model, "arm": arm, "fold": fold, "seed": seed}.items():
            grouped[name] = value
        records.append(grouped[DAILY_KEYS + SUM_COLUMNS])
    return pd.concat(records, ignore_index=True)


def write_multitask_report(predictions, output_dir, config=None):
    """Write pooled and per-fold native metrics plus clearly marked projections."""
    config = config or {}
    frame = validate_multitask_predictions(predictions, config)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    pooled, folds = [], []
    for (split, model, arm), panel in frame.groupby(["split", "model", "arm"], sort=True):
        base = {"split": split, "group": "D", "model": model, "arm": arm, "seed": 7}
        pooled.extend(_metric_rows(panel, base))
        for fold, part in panel.groupby("fold", sort=True):
            folds.extend(_metric_rows(part, {**base, "fold": int(fold)}))
    metrics = pd.DataFrame(pooled)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(folds).to_csv(output / "fold_metrics.csv", index=False)
    daily_statistics(frame).to_csv(output / "day_statistics.csv", index=False)
    native = metrics.loc[metrics.native]
    columns = ["split", "model", "arm", "task", "n", "rmse", "zero_oos_r2", "pearson_ic",
               "balanced_accuracy", "raw_balanced_accuracy", "log_loss", "multiclass_brier"]
    body = native.reindex(columns=columns).to_html(index=False, float_format=lambda v: f"{v:.6g}", na_rep="—")
    note = ("D / 4-hour / seed 7. All arms are fresh return-only fits; inactive heads have no native scores. "
            "Classification BA uses argmax(p / Train prior); raw argmax is reported separately. "
            "Scalar-threshold and Train-class-mean projections are auxiliary, not trained native heads. "
            "Historical Test has been seen in earlier studies: this is exploratory, report-only evidence, not a new holdout. "
            "One seed does not establish seed stability or downstream RL value.")
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Return multitask benchmark</title>"
            "<style>body{font:15px system-ui;margin:32px}table{border-collapse:collapse}th,td{padding:7px;border:1px solid #ccc}"
            "th{background:#eee}p{max-width:1100px}</style><h1>Return-only multitask benchmark</h1><p>"
            + html.escape(note) + "</p>" + body + "<h2>All native and auxiliary scores</h2>"
            + metrics.to_html(index=False, float_format=lambda v: f"{v:.6g}", na_rep="—") + "</html>")
    (output / "report.html").write_text(page, encoding="utf-8")
    summary = {"complete": True, "n_pooled_metrics": len(metrics), "n_folds": int(frame.fold.nunique()),
               "n_unique_decisions_by_split": {split: int(part[["fold", "decision_id"]].drop_duplicates().shape[0])
                                               for split, part in frame.groupby("split")},
               "models": list(MODELS), "arms": list(ARMS), "calibration": "identity",
               "native_and_auxiliary_separate": True, "test_report_only": True,
               "historical_seen_test": True, "seed_stability_claim": False,
               "metrics_sha256": _sha(output / "metrics.csv"),
               "day_statistics_sha256": _sha(output / "day_statistics.csv"),
               "metrics_csv": str((output / "metrics.csv").resolve()), "note": note}
    (output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
