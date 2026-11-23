"""Read-only 336-cell exports combining the frozen224 controls and joint112.

Each joint condition is ONE cell with two heads and two native score sets.
Classical joint rows are explicitly reused independent-head controls; only
the neural joint rows share trainable representations. No selector is fitted.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .wm_ablation_evaluation import variance_qlike
from .wm_full_grid_evaluation import TASKS, NATIVE_PRIMARY, make_task_labels, score_grid
from .wm_full_grid_report import (MODELS, DAILY_COLUMNS, NATIVE_UNITS, _sha, _json,
    _gate, _read_panel, _native, _confusion_scores, _flatten, _daily)


CONDITION_KEYS = ("group", "model", "task", "formulation")
NATIVE_DAILY_COLUMNS = ("split", "group", "model", "task", "formulation", "fold", "seed",
    "utc_day", "segment_id", "n_rows", "sum_squared_error", "sum_absolute_error",
    "sum_qlike", "sum_actual_squared", "sum_brier")
BAR_NS = 15 * 60 * 10**9


def _parent_root(config):
    if "parent_output_dir" in config:
        return Path(config["parent_output_dir"]).resolve()
    path = Path(config["parent_config"])
    cfg = yaml.safe_load(path.read_text())
    output = Path(cfg["output_dir"])
    return output if output.is_absolute() else Path(__file__).resolve().parents[2] / output


def _native_daily(writer, base, frame, task, forecast):
    actual = frame["actual_" + task].to_numpy(float)
    forecast = np.asarray(forecast, float)
    if forecast.shape != actual.shape or not np.isfinite(forecast).all():
        raise ValueError("joint native forecasts must match the finite full cohort")
    times = frame.timestamp.astype("int64").to_numpy()
    data = pd.DataFrame({"utc_day": frame.timestamp.dt.strftime("%Y-%m-%d"),
        "segment_id": np.r_[0, np.cumsum(np.diff(times) != BAR_NS)], "n_rows": 1,
        "sum_squared_error": (forecast - actual) ** 2,
        "sum_absolute_error": np.abs(forecast - actual),
        "sum_qlike": variance_qlike(actual, forecast) if task == "variance" else 0.,
        "sum_actual_squared": actual ** 2,
        "sum_brier": (forecast - actual) ** 2 if task == "crash" else 0.})
    for row in data.groupby(["utc_day", "segment_id"], sort=True).sum().reset_index().to_dict("records"):
        writer.writerow({**base, **row})


def _pooled(task, frames, by_fold):
    n = sum(map(len, frames))
    classification = _confusion_scores(sum(np.asarray(x["classification"]["confusion_matrix"]) for x in by_fold))
    classification.update(probability_metrics_available=True, primary_rule="argmax(p/Train_prior)")
    classification["raw"] = _confusion_scores(sum(np.asarray(x["classification"]["raw"]["confusion_matrix"]) for x in by_fold))
    for metric in ("log_loss", "multiclass_brier", "brier"):
        if metric in by_fold[0]["classification"]:
            classification[metric] = float(sum(x["n"] * x["classification"][metric] for x in by_fold) / n)
    native = _native(task, np.concatenate([f["actual_" + task].to_numpy() for f in frames]),
        np.concatenate([f["scalar_" + task].to_numpy() for f in frames]))
    return {"n": n, "n_classes": 2 if task == "crash" else 3, "native": native,
        "classification": classification, "primary_metric": "balanced_accuracy",
        "primary_value": classification["balanced_accuracy"]}


def _joint_row(base, scores, native, n_folds, fold_ba):
    # A joint cell has both native regression and classification metrics.
    row = _flatten({**base, "formulation": "classification"}, scores)
    row.update(formulation="joint", n_folds=n_folds, n_seeds=1,
        fold_mean_balanced_accuracy=float(np.mean(fold_ba)) if fold_ba else None,
        folds_with_defined_ba=len(fold_ba), regression_primary_metric=NATIVE_PRIMARY[base["task"]],
        regression_primary_value=native[NATIVE_PRIMARY[base["task"]]],
        regression_primary_unit=NATIVE_UNITS[base["task"]],
        classification_primary_metric="balanced_accuracy", classification_primary_value=scores["classification"]["balanced_accuracy"],
        classification_balanced_accuracy=scores["classification"]["balanced_accuracy"],
        shared_parameters=base["model"] in ("direct", "wm"),
        implementation="shared_neural_joint" if base["model"] in ("direct", "wm") else "reused_independent_heads",
        implementation_type="shared_neural_joint" if base["model"] in ("direct", "wm") else "independent_head_control",
        training_origin="fresh_joint_fit" if base["model"] in ("direct", "wm") else "frozen224_regression_and_classification_fits",
        fresh_training=base["model"] in ("direct", "wm"),
        control_only=base["model"] in ("linear", "lightgbm"),
        reused_control_heads=base["model"] in ("linear", "lightgbm"))
    row.update({"native_" + key: value for key, value in native.items()})
    return row


def _validate_grid(frame, formulations, cells):
    expected = {(g, m, t, f) for g in "ABCD" for m in MODELS for t in TASKS for f in formulations}
    actual = set(frame[list(CONDITION_KEYS)].itertuples(index=False, name=None))
    if len(frame) != cells or frame.duplicated(list(CONDITION_KEYS)).any() or actual != expected:
        raise ValueError(f"exactly {cells} unique complete registered conditions are required")
    if not np.isfinite(pd.to_numeric(frame.primary_value, errors="coerce")).all():
        raise ValueError("every condition must have a finite primary metric; no cohort filtering")
    if "joint" in formulations:
        joint = frame.loc[frame.formulation == "joint"]
        if not np.isfinite(pd.to_numeric(joint.regression_primary_value, errors="coerce")).all():
            raise ValueError("every joint native regression metric must be finite")


def _html(table):
    columns = ["group", "model", "task", "formulation", "primary_metric", "primary_value",
        "regression_primary_metric", "regression_primary_value", "balanced_accuracy", "shared_parameters", "implementation"]
    return ('<!doctype html><html lang="ja"><meta charset="utf-8"><title>336条件比較</title>'
        '<style>body{font-family:system-ui;margin:24px;line-height:1.6}td,th{border:1px solid #ccc;padding:5px}table{border-collapse:collapse;font-size:12px}</style>'
        '<h1>4入力 × 4モデル × 7対象 × 3方式 = 336条件</h1>'
        '<p>同時学習は各条件を1行とし、回帰と分類の両指標を併記します。分類のBAと回帰の指標は単位が違うため直接順位付けしません。'
        'Transformer・WMは共有表現を学習し、線形・LightGBMは既存の独立した回帰・分類headを束ねた対照条件です。'
        '線形・LightGBMの同時学習による改善は構造上ゼロで、新しい共有表現の学習ではありません。'
        '336行の内訳は既存224条件・新規共有表現学習56条件・代数的な独立head対照56条件です。異なる予測を実験した条件は280で、336種類の新規学習を意味しません。'
        '全選択はVal内で固定し、既存のTestは探索的OOSの結果報告専用です。</p>'
        '<p><a href="results_336.csv">全336条件CSV</a> / <a href="results_336.json">JSON</a> / '
        '<a href="joint_inference.csv">時系列Block Bootstrap</a> / <a href="report_contract.json">評価契約</a></p>'
        + table.reindex(columns=columns).to_html(index=False, escape=True) + '</html>')


def write_joint_grid_report(output_dir, config, include_test=False):
    """Export joint112 selection cells or full336 final cells after frozen gates."""
    root = Path(output_dir).resolve()
    registration = json.loads((root / "preregistration.json").read_text())
    reg_sha, fit_hashes = _gate(root, registration, include_test)
    if config.get("groups", list("ABCD")) != list("ABCD"):
        raise ValueError("all four input groups are required")
    folds = {int(row["fold"]): row for row in registration["folds"]}
    parent = _parent_root(config)
    parent_reg = json.loads((parent / "preregistration.json").read_text())
    expected_parent_sha = registration.get("parent_registration_sha256")
    if expected_parent_sha and expected_parent_sha != _sha(parent / "preregistration.json"):
        raise ValueError("parent full-grid registration changed")
    if include_test:
        _gate(parent, parent_reg, True)
        own_barrier = json.loads((root / "test_barrier.json").read_text())
        expected_results = own_barrier.get("parent_results_224_sha256")
        if expected_results != _sha(parent / "final/results_224.csv"):
            raise ValueError("parent224 result table changed after the joint Test barrier")
        completion = json.loads((parent / "final/report_completion.json").read_text())
        if not completion.get("report_complete"):
            raise ValueError("parent224 report must be complete before full336 export")
    destination = root / ("final" if include_test else "selection")
    destination.mkdir(parents=True, exist_ok=True)
    neural = {(int(x["fold"]), x["group"], x["model"]): x["job"] for x in registration["neural_tasks"]}
    classical = {(int(x["fold"]), x["group"]): x["job"] for x in registration["classical_tasks"]}
    parent_neural = {(int(x["fold"]), x["group"], x["model"], x["formulation"]): x["job"] for x in parent_reg["neural_tasks"]}
    parent_classical = {(int(x["fold"]), x["group"]): x["job"] for x in parent_reg["classical_tasks"]}
    provenance, references, table, details, folded = {}, {}, [], [], []
    split = "test" if include_test else "selection"
    filename = "test_predictions.parquet" if include_test else "validation_predictions.parquet"
    with (destination / "day_classification_counts.csv").open("w", newline="") as counts, (destination / "day_native_statistics.csv").open("w", newline="") as native_stream:
        writer = csv.DictWriter(counts, fieldnames=DAILY_COLUMNS); writer.writeheader()
        native_writer = csv.DictWriter(native_stream, fieldnames=NATIVE_DAILY_COLUMNS); native_writer.writeheader()
        for group in "ABCD":
            for model in MODELS:
                frames, scores = [], {task: [] for task in TASKS}
                for fold, meta in sorted(folds.items()):
                    kind = "neural" if model in ("direct", "wm") else "classical"
                    job = neural[(fold, group, model)] if kind == "neural" else classical[(fold, group)]
                    expected = {"fold": fold, "group": group, "model": model, "formulation": "joint", "split": split}
                    frame = _read_panel(root / kind / job / filename, expected, provenance, include_test)
                    columns = ["decision_id", "timestamp", *[p + task for task in TASKS for p in ("actual_", "label_")]]
                    reference = frame[columns].reset_index(drop=True)
                    if fold in references and not references[fold].equals(reference):
                        raise ValueError("joint conditions must retain the identical full outcome/time cohort")
                    references[fold] = reference
                    parent_frame = None
                    if include_test:
                        pjob = parent_neural[(fold, group, model, "regression")] if kind == "neural" else parent_classical[(fold, group)]
                        parent_frame = _read_panel(parent / kind / pjob / filename, {**expected, "formulation": "regression"}, provenance, True)
                        if not parent_frame[columns].reset_index(drop=True).equals(reference):
                            raise ValueError("joint and parent regression cohorts differ")
                    frames.append(frame)
                    for task in TASKS:
                        labels = make_task_labels(frame["actual_" + task].to_numpy(), task, meta["label_contract"])
                        if not np.array_equal(labels, frame["label_" + task].to_numpy()):
                            raise ValueError("joint labels changed from the frozen Train map")
                        classes = 2 if task == "crash" else 3
                        pcols = [f"p_{task}_{i}" for i in range(classes)]
                        scalar = frame["scalar_" + task].to_numpy()
                        regression = score_grid(task, frame["actual_" + task].to_numpy(), scalar, "regression", meta["label_contract"], meta.get("scales"))
                        classification = score_grid(task, frame["actual_" + task].to_numpy(), frame[pcols].to_numpy(), "classification", meta["label_contract"], meta.get("scales"))
                        if classes == 2 and not frame["p_crash_2"].isna().all():
                            raise ValueError("crash joint head must remain binary")
                        scores[task].append(classification)
                        base = {"split": split, "group": group, "model": model, "task": task, "formulation": "joint", "fold": fold, "seed": 7}
                        folded.append(_joint_row(base, classification, regression["native"], 1,
                            [classification["primary_value"]] if classification["primary_value"] is not None else []))
                        decisions = np.argmax(frame[pcols].to_numpy() / np.asarray(meta["label_contract"][task]["priors"]), axis=1)
                        _daily(writer, base, frame, labels, decisions, classes)
                        _native_daily(native_writer, base, frame, task, scalar)
                        if parent_frame is not None:
                            _native_daily(native_writer, {**base, "formulation": "regression"}, parent_frame, task, parent_frame["scalar_" + task].to_numpy())
                if pd.concat([f.timestamp for f in frames]).duplicated().any():
                    raise ValueError("registered fold evaluation periods overlap")
                for task in TASKS:
                    pooled = _pooled(task, frames, scores[task])
                    ba = [x["primary_value"] for x in scores[task] if x["primary_value"] is not None]
                    base = {"split": split, "group": group, "model": model, "task": task, "formulation": "joint"}
                    row = _joint_row(base, pooled, pooled["native"], len(folds), ba)
                    table.append(row)
                    details.append({**row, "regression_native": pooled["native"], "classification": pooled["classification"]})
    joint = pd.DataFrame(table)
    _validate_grid(joint, ("joint",), 112)
    joint.to_csv(destination / "joint_results_112.csv", index=False)
    _json(destination / "joint_results_112.json", details)
    pd.DataFrame(folded).to_csv(destination / "joint_fold_metrics.csv", index=False)
    combined = joint
    if include_test:
        parent_csv = parent / "final/results_224.csv"
        controls = pd.read_csv(parent_csv)
        _validate_grid(controls, ("regression", "classification"), 224)
        provenance[str(parent_csv)] = _sha(parent_csv)
        parent_details = json.loads((parent / "final/results_224.json").read_text())
        combined = pd.concat([controls, joint], ignore_index=True)
        _validate_grid(combined, ("regression", "classification", "joint"), 336)
        combined.to_csv(destination / "results_336.csv", index=False)
        _json(destination / "results_336.json", parent_details + details)
        # Classification inference compares the probability head with its
        # existing classification control, on the same daily class supports.
        counts = pd.concat([pd.read_csv(parent / "final/day_classification_counts.csv"),
            pd.read_csv(destination / "day_classification_counts.csv")], ignore_index=True)
        counts.loc[counts.split == "test"].to_csv(destination / "day_classification_counts.csv", index=False)
        (destination / "index.html").write_text(_html(combined), encoding="utf-8")
    contract = {"complete": True, "cells": 336 if include_test else 112, "complete_cells_per_split": 336 if include_test else 112,
        "registration_sha256": reg_sha, "parent_registration_sha256": _sha(parent / "preregistration.json"),
        "frozen_fit_sha256": fit_hashes, "prediction_file_sha256": provenance, "report_source_sha256": _sha(Path(__file__)),
        "n_folds": len(folds), "n_seeds": 1, "joint_cell_contains_two_native_heads": True,
        "shared_neural_joint_cells": 56, "independent_reused_classical_joint_cells": 56,
        "distinct_empirical_prediction_cells": 280, "algebraic_independent_head_control_cells": 56,
        "all_336_are_fresh_training": False,
        "classical_joint_is_separable_control": True, "native_and_classification_metrics_separate": True,
        "test_selection_allowed": False, "test_report_only": True, "historical_seen_test": True,
        "statistics_family": "112 neural contrasts (56 native +56 BA), Holm112; classical112 exact-zero controls",
        "native_inference_metric": {**NATIVE_PRIMARY, "advantage": "rmse", "recovery": "rmse"},
        "rank_task_inference": "Spearman point metrics only; secondary RMSE block inference, no approximate Spearman CI"}
    _json(destination / "report_contract.json", contract)
    summary = {"report_complete": True, "complete": True, "include_test": include_test,
        "cells": len(combined), "cells_per_split": {split: len(combined)}, "n_result_rows": len(combined),
        "report_dir": str(destination), "registration_sha256": reg_sha,
        "results_csv": str(destination / ("results_336.csv" if include_test else "joint_results_112.csv")),
        "n_joint_cells": 112, "n_shared_neural_joint_cells": 56, "n_independent_classical_joint_cells": 56}
    summary.update(n_independent_joint_control_cells=56, n_fresh_joint_neural_cells=56)
    _json(destination / "report_completion.json", summary)
    return summary
