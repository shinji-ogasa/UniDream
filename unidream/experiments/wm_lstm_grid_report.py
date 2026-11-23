"""Matched-metric exports for the extra84 LSTM research conditions.

Selection exports read only the last Val month. After the global Test gate,
the new84 rows can be combined with the immutable336-row comparison table
inside this new output directory. No prior result or selector is modified.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .wm_full_grid_evaluation import NATIVE_PRIMARY, make_task_labels, score_grid
from .wm_full_grid_report import (
    DAILY_COLUMNS, NATIVE_UNITS, _confusion_scores, _daily, _flatten, _gate,
    _json, _native, _read_panel, _sha,
)
from .wm_joint_grid_report import NATIVE_DAILY_COLUMNS, _native_daily
from .wm_seven_task_evaluation import TASKS


GROUPS = tuple("ABCD")
FORMULATIONS = ("regression", "classification", "joint")
KEYS = ("group", "model", "task", "formulation")


def _validate(frame, models, cells):
    expected = {(g, m, t, f) for g in GROUPS for m in models for t in TASKS for f in FORMULATIONS}
    if (len(frame) != cells or frame.duplicated(list(KEYS)).any()
            or set(frame[list(KEYS)].itertuples(index=False, name=None)) != expected):
        raise ValueError(f"exactly{cells} complete unique comparison conditions required")
    if not np.isfinite(pd.to_numeric(frame.primary_value, errors="coerce")).all():
        raise ValueError("every reported primary metric must be finite")


def _pooled(task, frames, fold_scores, formulation):
    n = sum(map(len, frames))
    classification = _confusion_scores(sum(np.asarray(s["classification"]["confusion_matrix"]) for s in fold_scores))
    classification.update(probability_metrics_available=fold_scores[0]["classification"]["probability_metrics_available"],
        primary_rule=fold_scores[0]["classification"]["primary_rule"])
    for metric in ("log_loss", "multiclass_brier", "brier"):
        if metric in fold_scores[0]["classification"]:
            classification[metric] = float(sum(s["n"] * s["classification"][metric] for s in fold_scores) / n)
    if "raw" in fold_scores[0]["classification"]:
        classification["raw"] = _confusion_scores(sum(np.asarray(s["classification"]["raw"]["confusion_matrix"]) for s in fold_scores))
    actual = np.concatenate([frame["actual_" + task].to_numpy(float) for frame in frames])
    native = None
    if formulation != "classification":
        native = _native(task, actual, np.concatenate([frame["scalar_" + task].to_numpy(float) for frame in frames]))
    elif task == "crash":
        native = _native(task, actual, np.concatenate([frame["p_crash_1"].to_numpy(float) for frame in frames]))
    metric = NATIVE_PRIMARY[task] if formulation == "regression" else "balanced_accuracy"
    return {"n": n, "n_classes": 2 if task == "crash" else 3, "classification": classification,
        "native": native, "primary_metric": metric,
        "primary_value": native[metric] if formulation == "regression" else classification["balanced_accuracy"]}


def _row(base, scores, n_folds, fold_ba):
    mode = base["formulation"]
    row = _flatten({**base, "formulation": "classification" if mode == "joint" else mode}, scores)
    row.update(formulation=mode, n_folds=n_folds, n_seeds=1,
        fold_mean_balanced_accuracy=float(np.mean(fold_ba)) if fold_ba else None,
        folds_with_defined_ba=len(fold_ba), shared_parameters=mode == "joint",
        implementation="shared_lstm_joint" if mode == "joint" else "lstm_single_formulation",
        implementation_type="shared_neural_joint" if mode == "joint" else "neural_single_formulation",
        training_origin="fresh_lstm_fit", fresh_training=True, control_only=False,
        architecture="causal_lstm", reused_control_heads=False)
    if mode == "joint":
        native = scores["native"]
        row.update(regression_primary_metric=NATIVE_PRIMARY[base["task"]],
            regression_primary_value=native[NATIVE_PRIMARY[base["task"]]],
            regression_primary_unit=NATIVE_UNITS[base["task"]],
            classification_primary_metric="balanced_accuracy",
            classification_primary_value=scores["classification"]["balanced_accuracy"],
            classification_balanced_accuracy=scores["classification"]["balanced_accuracy"])
    return row


def _capacity(root, tasks, cfg):
    audit = []
    for task in tasks:
        fit = json.loads((root / "neural" / task["job"] / "fit.json").read_text())
        audit.append({"fold": task["fold"], "group": task["group"], "formulation": task["formulation"],
            "parameters": fit.get("parameters", fit.get("n_parameters")),
            "steps": fit.get("steps"), "best_step": fit.get("best_step"),
            "stop_reason": fit.get("stop_reason"), "max_step_censored": fit.get("max_step_censored")})
    return {"model_config": cfg.get("model"), "training_config": cfg.get("training"), "fits": audit,
        "same_parameter_count_as_other_architectures_claimed": False,
        "same_step_budget_is_not_equal_compute": True,
        "training_information": "same frozen input masks,256 observations,15fold cohorts,Train label maps and seed7; actual capacity and training budgets shown separately"}


def _html(table, combined=False):
    title = "LSTM追加比較420条件" if combined else "LSTM84条件"
    columns = ["group", "model", "task", "formulation", "primary_metric", "primary_value",
        "regression_primary_metric", "regression_primary_value", "balanced_accuracy", "n_rows"]
    return ('<!doctype html><html lang="ja"><meta charset="utf-8"><title>' + title + '</title>'
        '<style>body{font-family:system-ui;margin:24px;line-height:1.65}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:5px}p{max-width:1100px}</style>'
        '<h1>' + title + '</h1><p>4入力 × 7対象 × 回帰・分類・同時学習のLSTM84条件。'
        '同時学習の1条件には回帰・分類の両指標を併記します。回帰指標と分類BAを直接順位付けしません。'
        '急落は2クラス、他は3クラス。分類境界と事前確率は各foldのTrainだけから決め、確率をTrain事前確率で割ったargmaxを使います。</p>'
        '<p>同じ入力履歴・15fold・seed・学習ステップ上限で比較し、実際のパラメータ数と学習量は'
        '<a href="capacity_audit.json">容量・学習量監査</a>に保存します。アーキテクチャ間の同一パラメータ数や同一計算量は保証しません。'
        'モデルとcheckpointの選択はValで完結します。過去に確認したTestでの探索的OOS評価であり、未使用Holdoutの再現確認ではありません。'
        'この追加LSTM比較には新しい有意性検定を付けていません。既存336条件の検定をLSTMへ流用しません。</p>'
        '<p><a href="results_84.csv">LSTM84条件CSV</a> / <a href="results_84.json">詳細JSON</a> / '
        '<a href="results_420.csv">既存336＋LSTM84条件CSV</a> / <a href="index420.html">420条件一覧</a> / '
        '<a href="fold_metrics.csv">fold別</a> / <a href="report_contract.json">評価契約</a></p>'
        + table.reindex(columns=columns).to_html(index=False, escape=True) + '</html>')


def write_lstm_grid_report(output_dir, cfg, include_test=False):
    root = Path(output_dir).resolve()
    registration = json.loads((root / "preregistration.json").read_text())
    reg_sha, fit_hashes = _gate(root, registration, include_test)
    tasks = registration["neural_tasks"]
    expected = {(fold, group, mode) for fold in range(15) for group in GROUPS for mode in FORMULATIONS}
    if (len(tasks) != 180 or registration.get("classical_tasks", [])
            or {(int(t["fold"]), t["group"], t["formulation"]) for t in tasks} != expected
            or any(t.get("model") != "lstm" or int(t.get("seed", 7)) != 7 for t in tasks)):
        raise ValueError("exactly180 fresh LSTM fits across15folds/four inputs/three formulations required")
    folds = {int(row["fold"]): row for row in registration["folds"]}
    if set(folds) != set(range(15)):
        raise ValueError("the complete frozen15fold label contracts are required")
    jobs = {(int(t["fold"]), t["group"], t["formulation"]): t["job"] for t in tasks}
    split = "test" if include_test else "selection"
    filename = "test_predictions.parquet" if include_test else "validation_predictions.parquet"
    destination = root / ("final" if include_test else "selection")
    destination.mkdir(parents=True, exist_ok=True)
    capacity = _capacity(root, tasks, cfg)
    _json(destination / "capacity_audit.json", capacity)
    references, provenance, rows, folded, details = {}, {}, [], [], []
    with (destination / "day_classification_counts.csv").open("w", newline="") as count_stream, (destination / "day_native_statistics.csv").open("w", newline="") as native_stream:
        count_writer, native_writer = csv.DictWriter(count_stream, fieldnames=DAILY_COLUMNS), csv.DictWriter(native_stream, fieldnames=NATIVE_DAILY_COLUMNS)
        count_writer.writeheader(); native_writer.writeheader()
        for group in GROUPS:
            for mode in FORMULATIONS:
                frames, scores = [], {task: [] for task in TASKS}
                for fold, meta in sorted(folds.items()):
                    directory = root / "neural" / jobs[(fold, group, mode)]
                    if include_test and json.loads((directory / "evaluation.json").read_text()).get("barrier_sha256") != _sha(root / "test_barrier.json"):
                        raise ValueError("LSTM prediction evaluation belongs to a different Test barrier")
                    frame = _read_panel(directory / filename, {"fold": fold, "group": group, "model": "lstm",
                        "formulation": mode, "split": split}, provenance, include_test)
                    reference = frame[["decision_id", "timestamp", *[prefix + task for task in TASKS for prefix in ("actual_", "label_")]]].reset_index(drop=True)
                    if fold in references and not references[fold].equals(reference):
                        raise ValueError("all LSTM conditions must preserve the identical evaluation cohort")
                    references[fold] = reference
                    frames.append(frame)
                    for task in TASKS:
                        actual = frame["actual_" + task].to_numpy(float)
                        labels = make_task_labels(actual, task, meta["label_contract"])
                        if not np.array_equal(labels, frame["label_" + task].to_numpy()):
                            raise ValueError("LSTM labels differ from the frozen Train contract")
                        k = 2 if task == "crash" else 3
                        forecast = (frame["scalar_" + task].to_numpy(float) if mode == "regression" else
                            frame[[f"p_{task}_{c}" for c in range(k)]].to_numpy(float))
                        score = score_grid(task, actual, forecast, "regression" if mode == "regression" else "classification", meta["label_contract"], meta.get("scales"))
                        if mode == "joint":
                            score["native"] = score_grid(task, actual, frame["scalar_" + task].to_numpy(float), "regression", meta["label_contract"], meta.get("scales"))["native"]
                        scores[task].append(score)
                        base = {"split": split, "group": group, "model": "lstm", "task": task, "formulation": mode}
                        ba = score["classification"]["balanced_accuracy"]
                        folded.append(_row({**base, "fold": fold, "seed": 7}, score, 1, [] if ba is None else [ba]))
                        if mode == "regression" and task != "crash":
                            decision = make_task_labels(forecast, task, meta["label_contract"])
                        else:
                            p = (np.column_stack((1 - forecast, forecast)) if mode == "regression" else forecast)
                            decision = np.argmax((p / p.sum(1, keepdims=True)) / np.asarray(meta["label_contract"][task]["priors"]), axis=1)
                        _daily(count_writer, {**base, "fold": fold, "seed": 7}, frame, labels, decision, k)
                        if mode != "classification":
                            _native_daily(native_writer, {**base, "fold": fold, "seed": 7}, frame, task, frame["scalar_" + task].to_numpy(float))
                if pd.concat([frame.timestamp for frame in frames]).duplicated().any():
                    raise ValueError("LSTM evaluation folds overlap")
                for task in TASKS:
                    pooled = _pooled(task, frames, scores[task], mode)
                    ba = [s["classification"]["balanced_accuracy"] for s in scores[task] if s["classification"]["balanced_accuracy"] is not None]
                    row = _row({"split": split, "group": group, "model": "lstm", "task": task, "formulation": mode}, pooled, len(folds), ba)
                    rows.append(row)
                    details.append({**row, "native": pooled["native"], "classification": pooled["classification"]})
    table = pd.DataFrame(rows)
    _validate(table, ("lstm",), 84)
    table.to_csv(destination / "results_84.csv", index=False)
    _json(destination / "results_84.json", details)
    pd.DataFrame(folded).to_csv(destination / "fold_metrics.csv", index=False)
    (destination / "index.html").write_text(_html(table), encoding="utf-8")
    combined_cells = None
    comparison_sha = None
    if include_test:
        comparison_path = Path(cfg["point_table_path"]).resolve()
        comparison_sha = _sha(comparison_path)
        expected_sha = registration.get("parent_results_336_sha256")
        if expected_sha is not None and expected_sha != comparison_sha:
            raise ValueError("frozen336 comparison table changed after LSTM registration")
        controls = pd.read_csv(comparison_path)
        _validate(controls, ("linear", "lightgbm", "direct", "wm"), 336)
        if not controls.split.eq("test").all():
            raise ValueError("combined comparison requires the frozen336 Test rows")
        combined = pd.concat((controls, table), ignore_index=True, sort=False)
        _validate(combined, ("linear", "lightgbm", "direct", "wm", "lstm"), 420)
        combined.to_csv(destination / "results_420.csv", index=False)
        _json(destination / "results_420.json", json.loads(combined.to_json(orient="records")))
        (destination / "index420.html").write_text(_html(combined, combined=True), encoding="utf-8")
        combined_cells = 420
    contract = {"registration_sha256": reg_sha, "frozen_fit_sha256": fit_hashes,
        "prediction_file_sha256": provenance, "report_source_sha256": _sha(Path(__file__)),
        "cells": 84, "combined_cells": combined_cells, "parent_results_336_sha256": comparison_sha,
        "primary_class_rule": "argmax(row-normalized probability / fold Train prior)",
        "native_and_classification_metrics_separate": True, "shared_joint_lstm": True,
        "historical_seen_test": True, "test_selection_allowed": False, "independently_unused_holdout": False,
        "new_significance_tests_included": False, "capacity_audit_sha256": _sha(destination / "capacity_audit.json")}
    _json(destination / "report_contract.json", contract)
    completion = {"report_complete": True, "complete": True, "include_test": include_test,
        "cells": 84, "n_result_rows": 84, "combined_cells": combined_cells,
        "registration_sha256": reg_sha, "results_csv": str(destination / "results_84.csv"),
        "combined_results_csv": str(destination / "results_420.csv") if include_test else None,
        "report_dir": str(destination), "n_folds": 15, "n_seeds": 1}
    _json(destination / "report_completion.json", completion)
    return completion


write_lstm_report = write_lstm_grid_report
