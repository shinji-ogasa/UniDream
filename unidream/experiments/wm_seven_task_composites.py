"""Frozen, exploratory direct-head versus plug-in synthetic-score comparison.

This module fits nothing and never changes the primary seven-task family.
Register it before the global Test barrier, then evaluate only saved outputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .wm_seven_task_evaluation import TASKS


MODELS = ("unconditional", "linear", "lightgbm", "direct", "wm")
COMPOSITES = ("advantage", "recovery")
FORMULAS = {
    "advantage": "advantage_delta*(pred_return-dd_penalty*pred_downside-transition_cost)",
    "recovery": "clip((pred_return-recovery_dd_penalty*pred_downside)/(sqrt(max(pred_variance,1e-12))+1e-6),-5,5)/5",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _constants(registration: dict) -> dict[str, float]:
    targets = registration["config"]["targets"]
    names = ("advantage_delta", "dd_penalty", "recovery_dd_penalty", "transition_cost")
    constants = {key: float(targets[key]) for key in names}
    if not np.isfinite(list(constants.values())).all() or constants["advantage_delta"] <= 0:
        raise ValueError("composite constants must be finite and advantage_delta positive")
    if any(value < 0 for key, value in constants.items() if key != "advantage_delta"):
        raise ValueError("composite penalties and cost must be nonnegative")
    return constants


def _verify_registration(out: Path) -> dict:
    frozen = json.loads((out / "composites/preregistration.json").read_text())
    parent = out / "preregistration.json"
    if frozen["parent_registration_sha256"] != _sha(parent):
        raise ValueError("main experiment registration changed after composite registration")
    if frozen["source_sha256"] != _sha(Path(__file__)):
        raise ValueError("composite source changed after registration")
    if frozen["constants"] != _constants(json.loads(parent.read_text())):
        raise ValueError("composite constants changed after registration")
    return frozen


def preregister_composites(output_dir: str | Path) -> dict:
    """Freeze this comparison before any Test barrier or saved Test outputs.

    Repeating an already valid registration is read-only, including after the
    barrier.  A new registration cannot be created after Test is opened.
    """
    out = Path(output_dir).resolve()
    destination = out / "composites/preregistration.json"
    if destination.exists():
        return _verify_registration(out)
    if (out / "test_barrier.json").exists() or next(out.glob("neural/*/test_predictions.parquet"), None) is not None or next(out.glob("classical/*/test_predictions.parquet"), None) is not None:
        raise ValueError("register composites before the Test barrier and Test predictions")
    parent = out / "preregistration.json"
    registration = json.loads(parent.read_text())
    value = {"registered_utc": datetime.now(timezone.utc).isoformat(), "parent_registration_sha256": _sha(parent),
        "source_sha256": _sha(Path(__file__)), "constants": _constants(registration), "formulas": FORMULAS,
        "models": list(MODELS), "targets": list(COMPOSITES), "variance_floor": 1e-12,
        "test_barrier_absent_at_registration": True, "test_prediction_files_absent_at_registration": True,
        "fitting_allowed": False, "test_selection_allowed": False, "status": "exploratory addendum fixed before Test access",
        "primary_seven_task_family_changed": False,
        "comparison": "same actual labels and timestamps; existing direct composite heads versus plug-in computed from existing return/variance/downside predictions",
        "aggregation": "pooled RMSE = sqrt(mean of seed-level squared errors); Pearson/Spearman computed separately per seed then averaged; seed SD is descriptive",
        "scope": "all saved splits; only complete seven-head models, exclude variance-only baselines",
        "inference": "descriptive metrics only; no additional significance claims or tuning",
        "interpretation": "f(E[...]) != E[f(...)]; nonlinear clipping/variance normalization can distinguish plug-in and direct conditional-mean recovery even with perfect component means; advantage is affine in return/downside but trained heads can be inconsistent",
        "historical_test_seen": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    _json(destination, value)
    return value


def derive_composites(predicted_return: np.ndarray, predicted_variance: np.ndarray,
                      predicted_downside: np.ndarray, constants: dict[str, float]) -> dict[str, np.ndarray]:
    """Deterministic registered formulas, applied separately to each seed."""
    ret, variance, downside = map(lambda value: np.asarray(value, dtype=float),
                                 (predicted_return, predicted_variance, predicted_downside))
    if ret.shape != variance.shape or ret.shape != downside.shape or not np.isfinite([ret, variance, downside]).all():
        raise ValueError("component predictions must be aligned finite arrays")
    return {
        "advantage": constants["advantage_delta"] * (ret - constants["dd_penalty"] * downside - constants["transition_cost"]),
        "recovery": np.clip((ret - constants["recovery_dd_penalty"] * downside) /
                            (np.sqrt(np.maximum(variance, 1e-12)) + 1e-6), -5, 5) / 5,
    }


def _score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    error = np.asarray(predicted) - np.asarray(actual)
    varying = len(actual) > 1 and np.ptp(actual) > 0 and np.ptp(predicted) > 0
    return {"n": len(actual), "mse": float(np.square(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
        "pearson_ic": float(pearsonr(actual, predicted).statistic) if varying else None,
        "spearman_ic": float(spearmanr(actual, predicted).statistic) if varying else None}


def write_composite_report(output_dir: str | Path) -> dict:
    """Read completed predictions; save paired metrics without fitting/selecting."""
    out = Path(output_dir).resolve()
    frozen = _verify_registration(out)
    source = out / "all_predictions.parquet"
    source_hash = _sha(source)
    frame = pd.read_parquet(source)
    frame = frame.loc[frame.model.isin(frozen["models"])].copy()
    if frame.empty:
        raise ValueError("no complete-head model predictions")
    columns = [prefix + task for prefix in ("actual_", "pred_") for task in TASKS]
    if not np.isfinite(frame[columns].to_numpy(float)).all():
        raise ValueError("all seven heads must be finite; cannot drop a model's failed predictions")
    keys = ["split", "group", "model", "fold", "seed", "timestamp", "horizon"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate prediction keys")
    cohort_keys = ["split", "group", "model", "fold", "timestamp", "horizon"]
    seed_sets = frame.groupby(cohort_keys)["seed"].agg(frozenset)
    for _, panel in seed_sets.groupby(level=["split", "group", "model", "fold", "horizon"]):
        if len(set(panel)) != 1:
            raise ValueError("each fold/model needs a complete seed cohort at every timestamp")
    if (frame.groupby(cohort_keys)[["actual_" + task for task in COMPOSITES]].nunique() != 1).any().any():
        raise ValueError("all seeds must share the same actual composite labels")
    derived = derive_composites(frame.pred_return.to_numpy(), frame.pred_variance.to_numpy(),
                                frame.pred_downside.to_numpy(), frozen["constants"])
    for task in COMPOSITES:
        frame["plugin_" + task] = derived[task]
    scores, fold_scores = [], []
    for grouping, destination in ((["split", "group", "model", "horizon", "seed"], scores),
                                   (["split", "group", "model", "horizon", "fold", "seed"], fold_scores)):
        for values, panel in frame.groupby(grouping, sort=True):
            base = dict(zip(grouping, values))
            for task in COMPOSITES:
                for method, column in (("direct_head", "pred_" + task), ("plugin", "plugin_" + task)):
                    destination.append({**base, "task": task, "method": method,
                        **_score(panel["actual_" + task].to_numpy(), panel[column].to_numpy())})
    per_seed = pd.DataFrame(scores)
    by_fold = pd.DataFrame(fold_scores)
    pooled = []
    for values, panel in per_seed.groupby(["split", "group", "model", "horizon", "task", "method"], sort=True):
        row = dict(zip(("split", "group", "model", "horizon", "task", "method"), values))
        # Complete neural seed panels contain the same number of timestamps;
        # weighted MSE remains correct if a deterministic model has one seed.
        row.update({"rmse": float(np.sqrt(np.average(panel.mse, weights=panel.n))),
                    "n_seeds": len(panel), "rows_including_seeds": int(panel.n.sum())})
        for metric in ("rmse", "pearson_ic", "spearman_ic"):
            available = pd.to_numeric(panel[metric], errors="coerce").dropna()
            if metric != "rmse":
                row[metric + "_mean_over_seeds"] = float(available.mean()) if len(available) else np.nan
            row[metric + "_sd_over_seeds"] = float(available.std(ddof=1)) if len(available) > 1 else np.nan
            if metric != "rmse":
                row[metric + "_defined_seeds"] = len(available)
        pooled.append(row)
    pooled = pd.DataFrame(pooled)
    directory = out / "composites"
    per_seed.to_csv(directory / "metrics_by_seed.csv", index=False)
    by_fold.to_csv(directory / "metrics_by_fold_seed.csv", index=False)
    pooled.to_csv(directory / "metrics_pooled.csv", index=False)
    compact = frame[[*keys, "decision_id", "actual_advantage", "actual_recovery", "pred_advantage", "pred_recovery", "plugin_advantage", "plugin_recovery"]]
    compact.to_parquet(directory / "derived_predictions.parquet", index=False)
    paired = pooled.pivot(index=["split", "group", "model", "horizon", "task"], columns="method",
                          values=["rmse", "pearson_ic_mean_over_seeds", "spearman_ic_mean_over_seeds"])
    paired.columns = [metric + "_" + method for metric, method in paired.columns]
    paired = paired.reset_index()
    paired["rmse_direct_minus_plugin"] = paired.rmse_direct_head - paired.rmse_plugin
    paired.to_csv(directory / "comparison.csv", index=False)
    html = ["<!doctype html><html lang='ja'><meta charset='utf-8'><title>直接予測と合成予測の比較</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:1300px;margin:32px auto;padding:0 20px;line-height:1.8;color:#172033}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}th,td{padding:6px 9px;border-bottom:1px solid #ddd;white-space:nowrap}th{background:#eef3f8}code{font-size:12px}a{color:#195bc5}</style>",
        "<h1>買い増し優位性・回復度：直接ヘッドと合成予測</h1>",
        "<p>既存の合成スコア予測ヘッドと、同じモデルのリターン・分散・下落予測から決定論的に計算した値を、同一の正解・時点で比較します。追加学習・補正・Testによる選択は行いません。主解析7タスクの検定は変更せず、本比較はTest開封前に固定した探索的な追加分析です。</p>",
        "<p>RMSEは全seedの二乗誤差から計算。Pearson・Spearman ICはseedごとに計算して平均しています。定数予測などでICを定義できないseed数もCSVに記録します。seedは独立市場サンプルとは扱いません。RMSE差（直接−合成）が正なら合成の誤差が小さいことを表します。</p>",
        "<p><strong>f(E[...]) ≠ E[f(...)]</strong> に注意してください。特に回復度は分散の平方根による割り算とclipを含むため、要素の条件付き平均を完全に予測できても、その合成値が回復度の条件付き平均になるとは限りません。優位性の式はリターンと下落に対して線形ですが、別々に学習した各ヘッドがその関係を満たす保証はありません。</p>",
        "<p>買い増し優位性は固定在庫増分を仮定した合成スコア、回復度はclipした将来リスク調整リターンです。実売買損益・回復時間を表しません。過去Testは既に探索に使われた期間で、未使用期間の最終確認とは区別します。</p>"]
    for split in ("test", "selection", "earlystop", "calibration"):
        selected = paired.loc[(paired.split == split) & (paired.group == "D")]
        if not selected.empty:
            html.append(f"<h2>{escape(split)}・全特徴量D</h2>" + selected.to_html(index=False, border=0, na_rep="—", float_format=lambda value: f"{value:.6g}"))
    html.append("<h2>固定した計算式</h2>")
    for task, formula in frozen["formulas"].items():
        html.append(f"<p>{escape(task)}：<code>{escape(formula)}</code></p>")
    html.append("<pre>" + escape(json.dumps(frozen["constants"], indent=2, ensure_ascii=False)) + "</pre>")
    for name in ("comparison.csv", "metrics_pooled.csv", "metrics_by_seed.csv", "metrics_by_fold_seed.csv", "derived_predictions.parquet", "preregistration.json"):
        html.append(f"<p><a href='{name}'>{name}</a></p>")
    html.append("</html>")
    (directory / "report.html").write_text("\n".join(html))
    completion = {"created_utc": datetime.now(timezone.utc).isoformat(), "main_predictions_sha256": source_hash,
        "composite_registration_sha256": _sha(directory / "preregistration.json"), "source_sha256": _sha(Path(__file__)),
        "prediction_rows": len(frame), "pooled_metric_rows": len(pooled), "paired_comparison_rows": len(paired),
        "test_report_only": True, "historical_seen_test": True, "new_models_fitted": False,
        "derived_predictions_sha256": _sha(directory / "derived_predictions.parquet"), "report_sha256": _sha(directory / "report.html")}
    _json(directory / "completion.json", completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "report"))
    parser.add_argument("output_dir")
    args = parser.parse_args()
    result = preregister_composites(args.output_dir) if args.command == "register" else write_composite_report(args.output_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
