"""Read-only descriptive exports for every cell of the fresh 224-cell grid.

Fold-dependent label maps and priors are applied before pooling. Native
regression metrics and classification accuracy are kept separate. This module
never fits, selects a checkpoint, calibrates, or opens market target arrays.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

from .wm_ablation_evaluation import variance_qlike
from .wm_full_grid_evaluation import TASKS, NATIVE_PRIMARY, make_task_labels, score_grid


MODELS = ("linear", "lightgbm", "direct", "wm")
FORMULATIONS = ("regression", "classification")
BAR_NS = 15 * 60 * 10**9
DAILY_COLUMNS = ("split", "group", "model", "task", "formulation", "fold", "seed", "utc_day",
                 "segment_id", "n_classes", "n_rows", "actual_c0", "actual_c1", "actual_c2",
                 "correct_c0", "correct_c1", "correct_c2")
NATIVE_UNITS = {"return": "unitless_R2", "variance": "unitless_QLIKE", "downside": "price_fraction",
                "crash": "probability_squared", "dd_excess": "price_fraction",
                "advantage": "unitless_rank_correlation", "recovery": "unitless_rank_correlation"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _confusion_scores(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    support, called = confusion.sum(1), confusion.sum(0)
    diagonal = np.diag(confusion)
    recalls = np.divide(diagonal, support, out=np.full(len(support), np.nan), where=support > 0)
    precision = np.divide(diagonal, called, out=np.zeros(len(support)), where=called > 0)
    f1 = np.divide(2 * diagonal, support + called, out=np.zeros(len(support)), where=(support + called) > 0)
    return {"accuracy": float(diagonal.sum() / confusion.sum()),
            "balanced_accuracy": float(recalls.mean()) if np.isfinite(recalls).all() else None,
            "macro_f1": float(f1.mean()), "confusion_matrix": confusion.tolist(),
            "per_class_recall": [float(x) if np.isfinite(x) else None for x in recalls],
            "per_class_precision": precision.tolist(), "actual_class_frequency": (support / support.sum()).tolist(),
            "predicted_class_frequency": (called / called.sum()).tolist()}


def _native(task, actual, forecast):
    if task == "crash":
        p = forecast
        return {"brier": float(np.mean((p - actual) ** 2)),
                "log_loss": float(-np.log(np.maximum(np.where(actual == 1, p, 1 - p), 1e-12)).mean()),
                "average_precision": float(average_precision_score(actual, p)) if np.any(actual == 1) else None,
                "prevalence": float(actual.mean())}
    error = forecast - actual
    correlations = {"pearson_ic": None, "spearman_ic": None}
    if len(actual) > 1 and np.ptp(actual) > 0 and np.ptp(forecast) > 0:
        correlations = {"pearson_ic": float(pearsonr(actual, forecast).statistic),
                        "spearman_ic": float(spearmanr(actual, forecast).statistic)}
    result = {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))), **correlations}
    if task == "return":
        denominator = float(np.sum(actual ** 2))
        result["zero_oos_r2"] = 1 - float(np.sum(error ** 2)) / denominator if denominator > 0 else None
    elif task == "variance":
        result.update(qlike=float(variance_qlike(actual, forecast).mean()),
                      prediction_floor_count=int(np.sum(forecast < 1e-12)),
                      calibration_mean_ratio=float(np.maximum(forecast, 1e-12).sum() / actual.sum()) if actual.sum() > 0 else None)
    return result


def _flatten(base, scores):
    classification = scores["classification"]
    row = {**base, "n_rows": scores["n"], "n_classes": scores["n_classes"],
           "primary_metric": scores["primary_metric"], "primary_value": scores["primary_value"],
           "primary_unit": "fraction" if base["formulation"] == "classification" else NATIVE_UNITS[base["task"]],
           "primary_direction": "higher" if base["formulation"] == "classification" or base["task"] in ("return", "advantage", "recovery") else "lower",
           "balanced_accuracy": classification["balanced_accuracy"], "accuracy": classification["accuracy"],
           "macro_f1": classification["macro_f1"], "primary_decision_rule": classification["primary_rule"],
           "probability_metrics_available": classification["probability_metrics_available"],
           "log_loss": classification.get("log_loss"), "multiclass_brier": classification.get("multiclass_brier"),
           "raw_balanced_accuracy": classification.get("raw", {}).get("balanced_accuracy")}
    row.update({"native_" + key: value for key, value in (scores["native"] or {}).items()})
    for index in range(scores["n_classes"]):
        row[f"recall_c{index}"] = classification["per_class_recall"][index]
        row[f"precision_c{index}"] = classification["per_class_precision"][index]
    return row


def _daily(writer, base, frame, labels, decisions, classes):
    timestamps = frame["timestamp"].astype("int64").to_numpy()
    segments = np.r_[0, np.cumsum(np.diff(timestamps) != BAR_NS)]
    daily = pd.DataFrame({"utc_day": frame["timestamp"].dt.strftime("%Y-%m-%d"),
                          "segment_id": segments, "n_rows": np.ones(len(frame), dtype=np.int64)})
    for index in range(3):
        daily[f"actual_c{index}"] = (labels == index).astype(np.int64)
        daily[f"correct_c{index}"] = ((labels == index) & (decisions == labels)).astype(np.int64)
    grouped = daily.groupby(["utc_day", "segment_id"], sort=True).sum().reset_index()
    for row in grouped.to_dict("records"):
        writer.writerow({**base, **row, "n_classes": classes})


def _gate(root, registration, include_test):
    reg_sha = _sha(root / "preregistration.json")
    fits = {}
    for kind, key in (("neural", "neural_tasks"), ("classical", "classical_tasks")):
        for task in registration[key]:
            path = root / kind / task["job"] / "fit.json"
            if not path.exists():
                raise ValueError(f"descriptive report requires every registered fit: {path}")
            fit = json.loads(path.read_text())
            if fit["registration_sha256"] != reg_sha:
                raise ValueError("fit registration changed")
            vf = path.parent / "validation_predictions.parquet"
            if _sha(vf) != fit["validation_predictions_sha256"]:
                raise ValueError("frozen validation prediction file changed")
            fits[str(path.relative_to(root))] = _sha(path)
    if include_test:
        path = root / "test_barrier.json"
        if not path.exists():
            raise ValueError("full-grid Test barrier is closed")
        barrier = json.loads(path.read_text())
        if barrier.get("registration_sha256") != reg_sha or barrier.get("frozen_fit_sha256") != fits:
            raise ValueError("Test barrier registration or frozen fit manifest changed")
        if barrier.get("validation_choices_sha256") != _sha(root / "validation_choices.json"):
            raise ValueError("validation choices were not frozen before Test")
    return reg_sha, fits


def _read_panel(path, expected, provenance, include_test):
    if not path.exists():
        raise ValueError(f"missing complete prediction file: {path}")
    if path.name.startswith("test_"):
        evaluation = json.loads((path.parent / "evaluation.json").read_text())
        if evaluation["prediction_sha256"] != _sha(path):
            raise ValueError("frozen Test predictions changed")
    provenance.setdefault(str(path), _sha(path))
    frame = pd.read_parquet(path)
    for key, value in expected.items():
        frame = frame.loc[frame[key] == value]
    frame = frame.copy().sort_values("decision_id").reset_index(drop=True)
    if frame.empty or frame["decision_id"].duplicated().any():
        raise ValueError("missing or duplicate condition/fold prediction rows")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if frame["timestamp"].isna().any() or frame["timestamp"].duplicated().any():
        raise ValueError("prediction timestamps must be finite and unique")
    if not frame["timestamp"].is_monotonic_increasing or np.any(frame["timestamp"].astype("int64").to_numpy() % BAR_NS):
        raise ValueError("prediction timestamps must be ordered UTC 15-minute bars")
    if not (frame["seed"] == 7).all():
        raise ValueError("full-grid first pass must contain seed7 only")
    return frame


def _html(destination, summary):
    return f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>224条件の予測比較</title>
<style>body{{font-family:system-ui;margin:24px;line-height:1.7}}select{{margin:6px;padding:6px}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:6px;border:1px solid #ccc;text-align:right}}th{{position:sticky;top:0;background:white}}.table{{overflow:auto;max-height:70vh}}.note{{max-width:1100px}}</style>
<h1>4入力 × 4モデル × 7対象 × 回帰・分類：224条件</h1>
<p class="note">回帰の主指標は対象ごとの元の単位、分類の主指標はBA（0〜1）。これらの値を直接大小比較しません。共通のクラス判定BAと、foldを同じ重みで平均したBAも併記します。ラベル境界・クラス比率は各foldのTrainだけで固定し、統合前に適用しています。急落は2クラス、他は3クラスです。</p>
<p class="note">ニューラル回帰は7つの点予測損失（急落はBrier）、分類は6つのCEと急落BCEを同時学習します。教師と損失の変更に伴うタスク間の相互作用を含みます。古典モデルはタスク別です。seed7の1回だけで、seedの安定性は未検証です。過去に確認済みのTestは探索的OOSであり、新規未使用期間の確認とは区別します。校正・Testによる選択は行いません。</p>
<p>表示: <select id="split"></select><select id="group"></select><select id="model"></select><select id="task"></select><select id="formulation"></select> <span id="count"></span></p>
<div class="table"><table><thead id="head"></thead><tbody id="body"></tbody></table></div>
<p><a href="results_224.csv">全条件CSV</a> / <a href="results_224.json">詳細JSON</a> / <a href="fold_metrics.csv">fold別指標</a> / <a href="coverage.csv">期間・件数</a> / <a href="day_classification_counts.csv">時系列推論用の日次集計</a> / <a href="report_contract.json">評価契約・入力hash</a></p>
<p>この表は記述統計です。<a href="formulation_inference.csv">分類−回帰のBA差・Block Bootstrap信頼区間・112比較のHolm補正</a>は別処理で出力します。主解析は7日、1日・28日は感度確認です。</p>
<script>const data={json.dumps(summary['table'], ensure_ascii=False, allow_nan=False)};
const keys=['split','group','model','task','formulation','primary_metric','primary_value','primary_unit','balanced_accuracy','fold_mean_balanced_accuracy','n_folds','n_rows'];
const filters=['split','group','model','task','formulation'];
for(const k of filters){{const e=document.getElementById(k);for(const x of ['すべて',...new Set(data.map(r=>r[k]))]){{const o=document.createElement('option');o.value=x;o.textContent=x;e.append(o)}}e.onchange=draw}}
function draw(){{const rows=data.filter(r=>filters.every(k=>document.getElementById(k).value==='すべて'||r[k]===document.getElementById(k).value));document.getElementById('count').textContent=rows.length+' 条件';document.getElementById('head').innerHTML='<tr>'+keys.map(k=>'<th>'+k+'</th>').join('')+'</tr>';const b=document.getElementById('body');b.replaceChildren();for(const r of rows){{const tr=document.createElement('tr');for(const k of keys){{const td=document.createElement('td');td.textContent=r[k]===null||r[k]===undefined?'—':typeof r[k]==='number'&&!Number.isInteger(r[k])?r[k].toPrecision(6):r[k];tr.append(td)}}b.append(tr)}}}}draw();</script></html>'''


def write_full_grid_report(output_dir, config, include_test=False):
    """Export 224 complete cells per split; preserve selection/final folders."""
    root = Path(output_dir).resolve()
    registration = json.loads((root / "preregistration.json").read_text())
    registration_sha, frozen_fits = _gate(root, registration, include_test)
    folds = {int(row["fold"]): row for row in registration["folds"]}
    groups = tuple(config.get("groups", ("A", "B", "C", "D")))
    if set(groups) != set("ABCD") or len(groups) != 4 or not folds:
        raise ValueError("complete four-input nonempty fold grid required")
    destination = root / ("final" if include_test else "selection")
    destination.mkdir(parents=True, exist_ok=True)
    neural = {(int(task["fold"]), task["group"], task["model"], task["formulation"]): task["job"]
              for task in registration["neural_tasks"]}
    classical = {(int(task["fold"]), task["group"]): task["job"] for task in registration["classical_tasks"]}
    references, provenance, table, details, folded, coverage = {}, {}, [], [], [], []
    splits = ("selection", "test") if include_test else ("selection",)
    with (destination / "day_classification_counts.csv").open("w", newline="") as stream:
        daily_writer = csv.DictWriter(stream, fieldnames=DAILY_COLUMNS)
        daily_writer.writeheader()
        for split in splits:
            for group in groups:
                for model in MODELS:
                    for formulation in FORMULATIONS:
                        frames, scores = [], {task: [] for task in TASKS}
                        for fold, fold_meta in sorted(folds.items()):
                            kind = "neural" if model in ("direct", "wm") else "classical"
                            job = neural[(fold, group, model, formulation)] if kind == "neural" else classical[(fold, group)]
                            path = root / kind / job / ("test_predictions.parquet" if split == "test" else "validation_predictions.parquet")
                            expected = {"fold": fold, "group": group, "model": model, "formulation": formulation, "split": split}
                            frame = _read_panel(path, expected, provenance, include_test)
                            reference_columns = ["decision_id", "timestamp", *[prefix + task for task in TASKS for prefix in ("actual_", "label_")]]
                            reference = frame[reference_columns].reset_index(drop=True)
                            reference_key = (split, fold)
                            if reference_key in references:
                                if not references[reference_key].equals(reference):
                                    raise ValueError("all conditions must retain exactly the same outcomes, labels and time cohort")
                            else:
                                references[reference_key] = reference
                            frames.append(frame)
                            for task in TASKS:
                                classes = 2 if task == "crash" else 3
                                labels = make_task_labels(frame["actual_" + task].to_numpy(), task, fold_meta["label_contract"])
                                if not np.array_equal(frame["label_" + task].to_numpy(), labels):
                                    raise ValueError("saved class labels differ from frozen Train map")
                                pcols = [f"p_{task}_{index}" for index in range(3)]
                                if formulation == "regression":
                                    if not frame[pcols].isna().all().all():
                                        raise ValueError("scalar regression rows must not invent class probabilities")
                                    forecast = frame["scalar_" + task].to_numpy()
                                    decisions = (np.argmax(np.column_stack((1 - forecast, forecast)) / np.asarray(fold_meta["label_contract"][task]["priors"]), axis=1)
                                                 if task == "crash" else make_task_labels(forecast, task, fold_meta["label_contract"]))
                                else:
                                    if not frame["scalar_" + task].isna().all() or (classes == 2 and not frame[pcols[2]].isna().all()):
                                        raise ValueError("classification rows must not invent scalar forecasts or a third crash class")
                                    forecast = frame[pcols[:classes]].to_numpy()
                                    decisions = np.argmax(forecast / np.asarray(fold_meta["label_contract"][task]["priors"]), axis=1)
                                result = score_grid(task, frame["actual_" + task].to_numpy(), forecast, formulation,
                                                    fold_meta["label_contract"], fold_meta.get("scales"))
                                scores[task].append(result)
                                base = {"split": split, "group": group, "model": model, "task": task,
                                        "formulation": formulation, "fold": fold, "seed": 7}
                                folded.append(_flatten(base, result))
                                _daily(daily_writer, base, frame, labels, decisions, classes)
                        timestamps = pd.concat([frame["timestamp"] for frame in frames], ignore_index=True)
                        if timestamps.duplicated().any():
                            raise ValueError("evaluated folds must not overlap in time")
                        n_rows = sum(map(len, frames))
                        coverage.append({"split": split, "group": group, "model": model, "formulation": formulation,
                                         "n_rows": n_rows, "n_folds": len(frames), "n_seeds": 1,
                                         "first_utc": str(timestamps.min()), "last_utc": str(timestamps.max())})
                        for task in TASKS:
                            by_fold = scores[task]
                            actual = np.concatenate([frame["actual_" + task].to_numpy() for frame in frames])
                            scalar = np.concatenate([frame["scalar_" + task].to_numpy() for frame in frames]) if formulation == "regression" else None
                            classification = _confusion_scores(sum(np.asarray(row["classification"]["confusion_matrix"]) for row in by_fold))
                            classification.update(n=n_rows, probability_metrics_available=by_fold[0]["classification"]["probability_metrics_available"],
                                                  primary_rule=by_fold[0]["classification"]["primary_rule"])
                            for metric in ("log_loss", "multiclass_brier", "brier"):
                                if metric in by_fold[0]["classification"]:
                                    classification[metric] = float(sum(row["n"] * row["classification"][metric] for row in by_fold) / n_rows)
                            if "raw" in by_fold[0]["classification"]:
                                classification["raw"] = _confusion_scores(sum(np.asarray(row["classification"]["raw"]["confusion_matrix"]) for row in by_fold))
                            native = _native(task, actual, scalar) if formulation == "regression" else None
                            if formulation == "classification" and task == "crash":
                                p = np.concatenate([frame["p_crash_1"].to_numpy() for frame in frames])
                                native = _native(task, actual, p)
                            primary_metric = "balanced_accuracy" if formulation == "classification" else NATIVE_PRIMARY[task]
                            pooled = {"n": n_rows, "n_classes": 2 if task == "crash" else 3, "native": native,
                                      "classification": classification, "primary_metric": primary_metric,
                                      "primary_value": classification["balanced_accuracy"] if formulation == "classification" else native[primary_metric]}
                            base = {"split": split, "group": group, "model": model, "task": task, "formulation": formulation}
                            row = _flatten(base, pooled)
                            supported = [score["classification"]["balanced_accuracy"] for score in by_fold if score["classification"]["balanced_accuracy"] is not None]
                            row.update(n_folds=len(by_fold), n_seeds=1,
                                       fold_mean_balanced_accuracy=float(np.mean(supported)) if supported else None,
                                       folds_with_defined_ba=len(supported))
                            table.append(row)
                            details.append({**base, **pooled, "n_folds": len(by_fold),
                                            "fold_mean_balanced_accuracy": row["fold_mean_balanced_accuracy"],
                                            "folds_with_defined_ba": len(supported)})
    for split in splits:
        if sum(row["split"] == split for row in table) != 224:
            raise ValueError("incomplete 224-cell result grid")
    primary_split = "test" if include_test else "selection"
    primary_table = [row for row in table if row["split"] == primary_split]
    primary_details = [row for row in details if row["split"] == primary_split]
    pd.DataFrame(primary_table).to_csv(destination / "results_224.csv", index=False)
    if include_test:
        pd.DataFrame(table).to_csv(destination / "all_split_metrics.csv", index=False)
        pd.DataFrame([row for row in table if row["split"] == "selection"]).to_csv(destination / "selection_results_224.csv", index=False)
        _json(destination / "all_split_metrics.json", details)
        _json(destination / "selection_results_224.json", [row for row in details if row["split"] == "selection"])
    pd.DataFrame(folded).to_csv(destination / "fold_metrics.csv", index=False)
    pd.DataFrame(coverage).to_csv(destination / "coverage.csv", index=False)
    _json(destination / "results_224.json", primary_details)
    contract = {"registration_sha256": registration_sha, "frozen_fit_sha256": frozen_fits,
                "prediction_file_sha256": provenance, "report_source_sha256": _sha(Path(__file__)),
                "complete_cells_per_split": 224, "n_folds": len(folds), "n_seeds": 1,
                "native_and_classification_metrics_separate": True, "calibration_applied": False,
                "per_fold_maps_before_pooling": True, "no_finite_filtering_or_cohort_dropping": True,
                "classification_primary_rule": "p/each-fold Train priors; raw argmax auxiliary",
                "fold_mean_ba": "unweighted arithmetic mean over folds with all classes present; support count explicit",
                "daily_counts": {"columns": list(DAILY_COLUMNS), "segment_rule": "restart within each split/fold at every missing 15-minute bar; never bridge folds or gaps"},
                "historical_seen_test": True, "test_selection_allowed": False,
                "statistics": "descriptive only; paired block inference is exported separately", "config": config}
    _json(destination / "report_contract.json", contract)
    summary = {"report_complete": True, "include_test": include_test, "cells_per_split": {split: 224 for split in splits},
               "primary_split": primary_split, "n_result_rows": len(primary_table), "n_all_split_result_rows": len(table),
               "n_fold_metric_rows": len(folded), "n_folds": len(folds), "n_seeds": 1,
               "report_dir": str(destination), "results_csv": str(destination / "results_224.csv"),
               "index_html": str(destination / "index.html"), "daily_counts_csv": str(destination / "day_classification_counts.csv"),
               "table": primary_table}
    (destination / "index.html").write_text(_html(destination, summary), encoding="utf-8")
    _json(destination / "report_completion.json", {key: value for key, value in summary.items() if key != "table"})
    return {key: value for key, value in summary.items() if key != "table"}
