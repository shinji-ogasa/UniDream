"""CSV, HTML and static figures for the completed classification experiment.

Reporting is deliberately outside the training source dependency registry.
It never selects a checkpoint, threshold, model, input, or hyperparameter.
Regression-to-class conversions are descriptive controls with no invented
probabilities or classification-bootstrap claims.
"""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .wm_classification_evaluation import _confusion_statistics, make_labels


LEARNERS = ("logistic", "lightgbm", "direct", "wm")
MODEL_ORDER = (*LEARNERS, "train_majority", "always_flat")
CLASS_NAMES = ("down", "flat", "up")
CONDITION_LABELS = {"lgbm": "LightGBM θ（主条件）", "wm": "WM θ（探索的）", "fixed": "固定 θ（探索的）"}


def _flatten_metrics(records: list[dict[str, Any]]) -> pd.DataFrame:
    result = []
    for record in records:
        row = {key: record[key] for key in ("condition", "model", "split", "n", "accuracy", "balanced_accuracy", "macro_f1", "log_loss", "multiclass_brier", "neutral_fraction")}
        row["condition_role"] = "primary" if row["condition"] == "lgbm" else "exploratory"
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "neutral_fraction"):
            row["raw_" + key] = record["raw"][key]
        for index, label in enumerate(CLASS_NAMES):
            for key in ("actual_class_frequency", "predicted_class_frequency", "per_class_precision", "per_class_recall"):
                row[key + "_" + label] = record[key][index]
        result.append(row)
    return pd.DataFrame(result)


def _inference_table(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for inference in summary["inference"]:
        for metric, values in inference["metrics"].items():
            row = {key: inference[key] for key in ("condition", "comparison", "block_days", "replicates", "n_paired_timestamps", "n_seeds")}
            for key in ("n_segments", "n_segments_shorter_than_block", "n_resampleable_segments",
                        "resampleable_timestamp_fraction", "inference_available"):
                row[key] = inference.get(key)
            row.update(metric=metric, **values)
            row["condition_role"] = "primary" if row["condition"] == "lgbm" else "exploratory"
            row["holm_adjusted_p"] = None
            if metric == "balanced_accuracy_difference" and inference["primary_inference"]:
                row["holm_adjusted_p"] = summary["holm_adjusted_p"][row["condition"]][row["comparison"]]
            rows.append(row)
    return pd.DataFrame(rows)


def _calibration_rows(cfg: dict[str, Any], arrays: dict[str, np.ndarray], folds: list[dict[str, Any]]) -> pd.DataFrame:
    out = Path(cfg["output_dir"])
    rows = []
    for fold in folds:
        with np.load(out / "prepared" / f"fold_{fold['fold']:02d}.npz") as z:
            data = {key: z[key] for key in z.files}
        for condition in cfg["conditions"]:
            calibration = fold["calibration"][condition]
            for split in ("train", "val", "test"):
                ids = data[split]
                theta = data["theta_" + condition][ids]
                returns = arrays["returns"][ids, 1]
                labels = data["label_" + condition][ids]
                standardized = returns / (theta / cfg["k"])
                if not np.isfinite(standardized).all():
                    raise ValueError("calibration report encountered invalid frozen inputs")
                row = {"fold": fold["fold"], "condition": condition, "split": split, "n": len(ids),
                       "a_train_fitted": calibration.get("a"), "fixed_theta": calibration.get("fixed_theta"),
                       "mean_theta": float(np.mean(theta)), "theta_q05": float(np.quantile(theta, .05)),
                       "theta_median": float(np.median(theta)), "theta_q95": float(np.quantile(theta, .95)),
                       "standardized_return_mean": float(np.mean(standardized)),
                       "standardized_return_mean_square": float(np.mean(standardized ** 2)),
                       "standardized_return_rms": float(np.sqrt(np.mean(standardized ** 2)))}
                for quantile in (.01, .05, .25, .5, .75, .95, .99):
                    row["standardized_return_q" + str(int(100 * quantile)).zfill(2)] = float(np.quantile(standardized, quantile))
                frequency = np.bincount(labels, minlength=3) / len(labels)
                for index, label in enumerate(CLASS_NAMES):
                    row["actual_" + label + "_frequency"] = float(frequency[index])
                rows.append(row)
    return pd.DataFrame(rows)


def _data_scope_rows(cfg: dict[str, Any], parent: dict[str, Any], arrays: dict[str, np.ndarray], folds: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for fold in folds:
        with np.load(Path(cfg["output_dir"]) / "prepared" / f"fold_{fold['fold']:02d}.npz") as z:
            data = {key: z[key] for key in ("train", "val", "test")}
        with np.load(Path(parent["output_dir"]) / "prepared" / f"fold_{fold['fold']:02d}.npz") as z:
            original = {key: z[key] for key in ("train", "val", "test")}
        preserved = {split: np.array_equal(data[split], original[split]) for split in ("val", "test")}
        if not all(preserved.values()):
            raise ValueError("classification Val/Test must preserve the old immutable cohorts")
        row = {"fold": fold["fold"], "original_train_start": fold["train_start"],
               "original_train_end_exclusive": fold["train_end"], "original_train_n": len(original["train"]),
               "common_oof_train_n": len(data["train"]), "train_retained_fraction": len(data["train"]) / len(original["train"]),
               "val_n": len(data["val"]), "test_n": len(data["test"]),
               "val_preserved": bool(preserved["val"]), "test_preserved": bool(preserved["test"])}
        for split in ("train", "val", "test"):
            row[split + "_first_decision"] = str(pd.to_datetime(arrays["index_ns"][data[split][0]], utc=True))
            row[split + "_last_decision"] = str(pd.to_datetime(arrays["index_ns"][data[split][-1]], utc=True))
        rows.append(row)
    return pd.DataFrame(rows)


def _optimizer_diagnostics(cfg: dict[str, Any], frozen: dict[str, Any]) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    out = Path(cfg["output_dir"])
    if not (out / "optimizer_amendment.json").exists():
        return None, pd.DataFrame()
    from .wm_classification_optimizer_amendment import verify_amendment
    amendment, amendment_sha = verify_amendment(cfg)
    rows = []
    for task in frozen["classical_tasks"]:
        fit = json.loads((out / "classical" / task["job"] / "fit.json").read_text())
        history = fit["logistic"]
        if (history.get("optimizer_amendment_sha256") != amendment_sha
                or history.get("optimizer_source_sha256") != amendment["source_sha256"]
                or [row["C"] for row in history["candidates"]] != cfg["tabular"]["logistic_Cs"]
                or history["preconditioner"]["retained_dimension"] != 4909):
            raise ValueError("report requires the same amended optimizer for every LR task")
        for candidate in history["candidates"]:
            if (not np.isfinite(candidate["original_gradient_linf"])
                    or candidate["original_gradient_linf"] > cfg["tabular"]["logistic_tol"]
                    or not candidate["converged_original_gradient"]):
                raise ValueError("uncertified LR candidate cannot enter the report")
            rows.append({"condition": task["condition"], "fold": task["fold"],
                         "C": candidate["C"], "selected": candidate["C"] == history["selected"]["C"],
                         "n_train": fit["train_n"], "retained_dimension": history["preconditioner"]["retained_dimension"],
                         "iterations": candidate["n_iter"][0], "optimizer_success": candidate["optimizer_success"],
                         "optimizer_message": candidate["optimizer_message"], "objective": candidate["objective"],
                         "original_gradient_linf": candidate["original_gradient_linf"],
                         "original_gradient_tolerance": candidate["requested_original_tolerance"],
                         "optimizer_amendment_sha256": amendment_sha})
    return amendment, pd.DataFrame(rows)


def _directional_scores(actual: np.ndarray, down: np.ndarray, up: np.ndarray, timestamps: np.ndarray) -> dict[str, Any]:
    """Descriptive direction discrimination, conditional on actual non-flat.

    Zero directional probability mass maps to a preregistered score of .5,
    preserving the same actual down/up subset for every model. Binary
    decisions use strict up>down (ties down). No parameter is selected here.
    """
    from sklearn.metrics import roc_auc_score
    actual, down, up, timestamps = map(np.asarray, (actual, down, up, timestamps))
    if (actual.ndim != 1 or any(value.shape != actual.shape for value in (down, up, timestamps))
            or not np.isin(actual, (0, 1, 2)).all()
            or not np.isfinite(down).all() or not np.isfinite(up).all()
            or np.any(down < 0) or np.any(up < 0) or np.any(down + up > 1 + 1e-6)):
        raise ValueError("directional rows need matching classes/timestamps and valid probability components")
    use = actual != 1
    y = actual[use] == 2
    mass = down[use] + up[use]
    if not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("directional diagnostics require finite nonnegative probability mass")
    nonzero = mass > 0
    ratio = np.divide(up[use], mass, out=np.full(len(mass), .5), where=nonzero)
    prediction = up[use] > down[use]
    confusion = np.bincount(y.astype(int) * 2 + prediction.astype(int), minlength=4).reshape(2, 2)
    support = confusion.sum(1)
    balanced = float(np.mean(np.diag(confusion) / support)) if np.all(support > 0) else None
    auc = float(roc_auc_score(y, ratio)) if len(np.unique(y)) == 2 else None
    n_directional = int(use.sum())
    return {"n_prediction_rows": len(actual), "n_unique_timestamps": int(len(np.unique(timestamps))),
            "n_actual_directional": n_directional,
            "n_unique_directional_timestamps": int(len(np.unique(timestamps[use]))),
            "actual_directional_coverage": n_directional / len(actual) if len(actual) else None,
            "n_directional_probability": n_directional,
            "n_nonzero_directional_probability_mass": int(nonzero.sum()),
            "n_zero_directional_probability_mass": int((~nonzero).sum()),
            "directional_probability_coverage": 1.0 if n_directional else None,
            "native_directional_probability_coverage": float(nonzero.mean()) if len(nonzero) else None,
            "directional_auroc": auc, "raw_binary_balanced_accuracy": balanced,
            "raw_binary_accuracy": float(np.mean(y == prediction)) if len(y) else None,
            "actual_down_n": int(support[0]), "actual_up_n": int(support[1]),
            "binary_confusion_matrix": confusion.tolist(),
            "scope": "Descriptive actual-down/up subset; no confidence interval or significance claim"}


def _directional_diagnostics(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["condition", "model", "split", "fold", "seed", "timestamp", "actual", "p0", "p2"]
    pooled: dict[tuple[str, str, str], list[tuple[np.ndarray, ...]]] = {}
    by_fold = []
    for path in sorted((out / "predictions").glob("fold_*.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        for (condition, model, split, fold, seed), part in frame.groupby(["condition", "model", "split", "fold", "seed"], sort=False):
            values = (part.actual.to_numpy(dtype=np.int8), part.p0.to_numpy(), part.p2.to_numpy(),
                      pd.to_datetime(part.timestamp, utc=True).astype("int64").to_numpy())
            by_fold.append({"condition": condition, "model": model, "split": split, "fold": fold, "seed": seed,
                            **_directional_scores(*values)})
            pooled.setdefault((condition, model, split), []).append(values)
    aggregate = []
    for (condition, model, split), parts in sorted(pooled.items()):
        values = tuple(np.concatenate([part[column] for part in parts]) for column in range(4))
        aggregate.append({"condition": condition, "model": model, "split": split, **_directional_scores(*values)})
    return pd.DataFrame(aggregate), pd.DataFrame(by_fold)


def _control_scores(matrix: np.ndarray) -> dict[str, Any]:
    statistics = _confusion_statistics(matrix)
    total = float(matrix.sum())
    return {"n": int(total), "accuracy": float(statistics["accuracy"]),
            "balanced_accuracy": float(statistics["balanced_accuracy"]) if np.isfinite(statistics["balanced_accuracy"]) else None,
            "macro_f1": float(statistics["macro_f1"]), "neutral_fraction": float(matrix[:, 1].sum() / total),
            "confusion_matrix": matrix.astype(int).tolist(), "log_loss": None, "multiclass_brier": None,
            "interpretation": "Descriptive threshold conversion; historical training and selection differ from classification fits"}


def _regression_controls(cfg: dict[str, Any], arrays: dict[str, np.ndarray], folds: list[dict[str, Any]], comparison_cfg: dict[str, Any]) -> pd.DataFrame:
    source = Path(comparison_cfg["output_dir"]) / "predictions.parquet"
    if not source.is_file():
        raise ValueError("completed regression comparison predictions are required for descriptive controls")
    required = ["timestamp", "fold", "group", "seed", "split", "pred_return", "actual_return"]
    pooled: dict[tuple[str, str, str], np.ndarray] = {}
    for fold in folds:
        with np.load(Path(cfg["output_dir"]) / "prepared" / f"fold_{fold['fold']:02d}.npz") as z:
            data = {key: z[key] for key in z.files}
        previous = pd.read_parquet(source, columns=required, filters=[("fold", "==", fold["fold"])])
        for (model, split, _seed), frame in previous.groupby(["group", "split", "seed"], sort=False):
            if not frame["pred_return"].notna().any():
                continue
            frame = frame.sort_values("timestamp")
            ids = data[split]
            times = pd.to_datetime(frame["timestamp"], utc=True).astype("int64").to_numpy()
            if not np.array_equal(times, arrays["index_ns"][ids]) or not np.array_equal(frame["actual_return"].to_numpy(), arrays["returns"][ids, 1]):
                raise ValueError("regression controls must exactly share frozen classification evaluation dates and returns")
            for condition in cfg["conditions"]:
                predicted = make_labels(frame["pred_return"].to_numpy(), data["theta_" + condition][ids])
                actual = data["label_" + condition][ids]
                matrix = np.bincount(actual.astype(int) * 3 + predicted, minlength=9).reshape(3, 3)
                key = (condition, model, split)
                pooled[key] = pooled.get(key, np.zeros((3, 3), dtype=np.int64)) + matrix
    return pd.DataFrame([{"condition": condition, "regression_model": model, "split": split, **_control_scores(matrix)}
                         for (condition, model, split), matrix in sorted(pooled.items())])


def _plot_figures(out: Path, summary: dict[str, Any], flat: pd.DataFrame, calibration: pd.DataFrame, frozen: dict[str, Any]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets = out / "report_assets"
    assets.mkdir(exist_ok=True)
    paths = []
    main = flat.loc[(flat.condition == "lgbm") & (flat.split == "test")].set_index("model").reindex(MODEL_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    x = np.arange(len(main))
    for ax, metric, title in zip(axes, ("balanced_accuracy", "accuracy"), ("Balanced accuracy", "Accuracy")):
        ax.bar(x - .18, main[metric], width=.36, label="Primary: prior-adjusted decision")
        ax.bar(x + .18, main["raw_" + metric], width=.36, alpha=.55, label="Raw probability argmax")
        ax.set_xticks(x, main.index, rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    axes[0].axhline(1 / 3, color="black", linestyle="--", linewidth=.8, label="One-class decision BA = 1/3")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Primary LightGBM-threshold condition: retrospective Test")
    path = assets / "main_test_metrics.png"; fig.savefig(path, dpi=170); plt.close(fig); paths.append(str(path))

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for ax, model in zip(axes.flat, LEARNERS):
        record = next(r for r in summary["metrics"] if r["condition"] == "lgbm" and r["model"] == model and r["split"] == "test")
        matrix = np.asarray(record["confusion_matrix"], dtype=float)
        normalized = np.divide(matrix, matrix.sum(1, keepdims=True), out=np.zeros_like(matrix), where=matrix.sum(1, keepdims=True) > 0)
        ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for row in range(3):
            for column in range(3):
                ax.text(column, row, f"{normalized[row,column]:.1%}", ha="center", va="center", color="white" if normalized[row,column] > .5 else "black")
        ax.set_xticks(range(3), CLASS_NAMES); ax.set_yticks(range(3), CLASS_NAMES)
        ax.set_xlabel("Primary prediction"); ax.set_ylabel("Actual")
        ax.set_title(model + " (row-normalized)")
    fig.suptitle("Primary condition: pooled confusion, not a seed ensemble")
    path = assets / "main_test_confusion.png"; fig.savefig(path, dpi=170); plt.close(fig); paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for condition in ("lgbm", "wm", "fixed"):
        data = calibration.loc[(calibration.condition == condition) & (calibration.split == "test")].sort_values("fold")
        axes[0].plot(data.fold, data.standardized_return_rms, marker=".", label=condition)
        axes[1].plot(data.fold, data.actual_flat_frequency, marker=".", label=condition)
    axes[0].axhline(1, color="black", linestyle="--", linewidth=.8)
    axes[0].set_yscale("log"); axes[0].set_title("Test RMS of R / sqrt(a * variance)")
    axes[1].set_ylim(0, 1); axes[1].set_title("Actual flat-class frequency in Test")
    for ax in axes:
        ax.set_xlabel("Outer fold"); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle("Threshold diagnostics: different conditions define different labels")
    path = assets / "threshold_calibration.png"; fig.savefig(path, dpi=170); plt.close(fig); paths.append(str(path))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True, sharex=True, sharey=True)
    observed_ba = []
    for task in frozen["tasks"]:
        path = out / "neural" / task["job"] / "fit.json"
        result = json.loads(path.read_text())
        ax = axes[("direct", "wm").index(task["model"]), ("lgbm", "wm", "fixed").index(task["condition"])]
        steps = [row["step"] for row in result["history"]]
        ba = [row["val"]["balanced_accuracy"] for row in result["history"]]
        observed_ba.extend(ba)
        ax.plot(steps, ba, color="#2864a0", alpha=.12, linewidth=.7)
        chosen = next(row for row in result["history"] if row["step"] == result["best_step"])
        ax.scatter([result["best_step"]], [chosen["val"]["balanced_accuracy"]], color="black", s=5, alpha=.25)
        ax.scatter([steps[-1]], [ba[-1]], color="#b63838" if result["right_censored"] else "#2864a0", marker="x" if result["right_censored"] else "o", s=12, alpha=.45)
    low = max(0.0, float(np.min(observed_ba)) - .02)
    high = min(1.0, float(np.max(observed_ba)) + .02)
    if high - low < .1:
        center = (low + high) / 2
        low, high = max(0.0, center - .05), min(1.0, center + .05)
    for row, model in enumerate(("direct", "wm")):
        for column, condition in enumerate(("lgbm", "wm", "fixed")):
            ax = axes[row, column]
            ax.set_title(model + ": " + condition + " threshold")
            ax.set_ylim(low, high); ax.grid(alpha=.15)
            ax.axhline(1 / 3, color="black", linestyle="--", linewidth=.6, alpha=.4)
            if row == 1: ax.set_xlabel("Optimizer step")
            if column == 0: ax.set_ylabel("Selection-Val balanced accuracy")
    fig.suptitle("Individual validation curves; black=selected; red x=cap/right-censored")
    path = assets / "validation_learning_curves.png"; fig.savefig(path, dpi=170); plt.close(fig); paths.append(str(path))
    return paths


def _html_table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, na_rep="N/A", float_format=lambda value: f"{value:.6g}", escape=True)


def _results_view(flat: pd.DataFrame, condition: str) -> pd.DataFrame:
    frame = flat.loc[(flat.condition == condition) & (flat.split == "test")].set_index("model").reindex(MODEL_ORDER).reset_index()
    keys = ["model", "n", "balanced_accuracy", "raw_balanced_accuracy", "accuracy", "macro_f1",
            "per_class_recall_down", "per_class_recall_flat", "per_class_recall_up",
            "log_loss", "multiclass_brier", "neutral_fraction"]
    return frame[keys]


def _render_html(summary: dict[str, Any], flat: pd.DataFrame, inference: pd.DataFrame, controls: pd.DataFrame, calibration: pd.DataFrame, audit: dict[str, Any] | None,
                 *, data_scope: pd.DataFrame | None = None, directional: pd.DataFrame | None = None,
                 model_parameters: pd.DataFrame | None = None, optimizer_amendment: dict[str, Any] | None = None) -> str:
    primary = inference.loc[(inference.condition == "lgbm") & (inference.block_days == 7) & (inference.metric == "balanced_accuracy_difference")]
    inference_view = primary[["comparison", "estimate", "ci_low", "ci_high", "p_value", "holm_adjusted_p", "resampleable_timestamp_fraction"]]
    control_view = controls.loc[(controls.condition == "lgbm") & (controls.split == "test"), ["regression_model", "n", "balanced_accuracy", "accuracy", "macro_f1", "neutral_fraction"]]
    calibration_view = []
    for condition in ("lgbm", "wm", "fixed"):
        data = calibration.loc[(calibration.condition == condition) & (calibration.split == "test")]
        calibration_view.append({"condition": condition,
            "Test standardized-return RMS": float(np.sqrt(np.average(data.standardized_return_mean_square, weights=data.n))),
            "Test actual flat frequency": float(np.average(data.actual_flat_frequency, weights=data.n))})
    audit_text = ("独立監査: PASS" if audit and audit.get("checks_passed") else "独立監査の状態は別の監査JSONを参照")
    scope_html = ""
    if data_scope is not None:
        scope_html = ("<h2>データ期間と共通OOF Train</h2><p>元のouter Trainは2年だが、各分類器のCE教師となるdecision行は、両sourceの因果的予測が揃う共通OOF部分に限定する。WMの補助教師は各入力context内のouter Train時刻にも適用されるため、初期foldでは共通OOF開始直前の最大255本も含み得る。Val/Testの補助教師は使わない。最も古い予測quarterは2021年7月。下表は設定上の2年間と実際のCE decision行を分けて示す。Val/Testの日時は旧cohortを完全に保持し、Testは合計 "
                      + str(int(data_scope.test_n.sum())) + " decision日時（seedによる重複を除く）。source WMはquarterごとにseed7だけを使う固定の閾値生成器であり、新規分類器の3 seeds（7/17/27）とは別である。分類器のWMをsource WMの重みからwarm-startしない。</p>"
                      + _html_table(data_scope[["fold", "original_train_start", "original_train_end_exclusive", "original_train_n", "common_oof_train_n", "train_first_decision", "train_last_decision", "val_n", "test_n", "val_preserved", "test_preserved"]]))
    directional_html = ""
    if directional is not None:
        selected = directional.loc[(directional.condition == "lgbm") & (directional.split == "test")].set_index("model").reindex(MODEL_ORDER).reset_index()
        directional_html = ("<h2>方向識別の記述的診断</h2><p>Balanced Accuracyの向上は横ばいクラスの検出改善だけでも生じ、方向を学習した証拠とは限らない。以下は実際のdown/upに限り、p_up/(p_down+p_up)のAUROCと、p_up&gt;p_down（同率はdown）の二値Balanced Accuracyを示す。方向確率の和が0なら事前固定のscore=0.5を使い、全モデルで同じ実actual非横ばい日時を評価する。zero-mass件数とcoverageも明示する。行はseed×日時のpoolでありensembleではない。fold/seed別CSVも併記する。CI・有意性検定は付けず、統計的な方向予測力を主張しない。閾値条件が違うと対象部分集合も変わるため、条件をまたいだ順位づけには使わない。</p>"
                            + _html_table(selected[["model", "n_actual_directional", "n_unique_directional_timestamps", "actual_directional_coverage", "n_zero_directional_probability_mass", "directional_probability_coverage", "directional_auroc", "raw_binary_balanced_accuracy"]]))
    parameters_html = _html_table(model_parameters) if model_parameters is not None else ""
    optimizer_html = ""
    if optimizer_amendment is not None:
        optimizer_html = ("<h2>Logistic最適化手順の事前追記</h2><p>Logisticは通常のsklearn solverをそのまま実行したものではない。Test予測前の "
                          '<a href="optimizer_amendment.json">optimizer amendment</a>（'
                          + escape(optimizer_amendment["registered_utc"]) + "）で、全45 taskにTrain共分散の全rankを保持する座標前処理＋L-BFGSを統一適用した。4909特徴はすべて残し、PCAによる次元削減や追加penaltyは行わない。目的は元座標のmean unweighted NLL＋||W||²/(2 C NTrain)、interceptは無罰則で、元の4候補C・Val selectorを保持する。各候補について元のX/W/interceptで再計算した勾配L∞≤1e−4を確認してから選択する。中断した旧solverの試行は採用しない。"
                          '<a href="logistic_optimizer_diagnostics.csv">全180候補の目的値・収束診断CSV</a>を併記する。</p>')
    body = ["<h1>BTCUSDT 15分足：将来4時間の3クラス分類</h1>",
        "<p>0=下落、1=横ばい、2=上昇。R&lt;−θ、|R|≤θ、R&gt;θで分類する。主条件はLightGBMの因果的分散予測から作るθ。WM θと固定θは、異なる正解定義による探索的な補助条件である。</p>",
        "<p>通常の重みなし交差エントロピーで確率を学習し、主判定は argmax(p / Train class prior)。raw argmaxも併記する。確率そのものは変更せず、log lossとBrierは元の確率で評価する。下表の率は0〜1、Brierは3クラスの二乗誤差和の平均（0〜2）。</p>",
        "<p>Train→Val→Testを時系列に分け、4時間先と補助16時間先のラベルが境界をまたぐ行を除く。a=mean(Train OOF R²/vhat)、θ=0.5√(a vhat)をTrainだけで固定する。現在のθは分類headの追加入力であり、未来の実現値から作った入力ではない。</p>",
        "<p>各neural条件は15 fold×3 seed。数値はseed×日時をpoolした損失と混同行列から算出し、確率ensembleではない。classicalはfoldごとに1 fit。全選択終了後のTestは報告専用で、過去の研究で既知の期間を再利用したretrospective OOSである。</p>",
        "<p>新分類器のtabular入力は、256本×（base 5＋市場14）のtime-major履歴＋最終TA44＝4908列に現在θを加えた4909列。neuralは256本×63chと現在θを受け取る。同じOHLCV・市場系列の履歴を与えるが、手法ごとの表現形式は異なる。固定済みsource分散モデルと回帰controlの旧1394列レシピは変更しない。WM分類器は約236万parameterでauxiliary WM loss＋CE、directは約99万parameterでCEのみであり、同一parameter数の比較でも単一部品だけのablationでもない。実際のparameter数は下表に示す。</p>",
        parameters_html,
        optimizer_html,
        '<nav><a href="classification_results.csv">分類結果CSV</a> · <a href="classification_inference.csv">推論・CI CSV</a> · <a href="data_scope.csv">学習期間・行数CSV</a> · <a href="threshold_calibration.csv">閾値calibration CSV</a> · <a href="directional_diagnostics.csv">方向診断CSV</a> · <a href="directional_diagnostics_by_fold.csv">fold/seed別方向診断</a> · <a href="regression_conversion_controls.csv">回帰変換control CSV</a> · <a href="summary.json">全結果JSON</a> · <a href="preregistration.json">事前登録</a> · <a href="final_independent_audit.json">最終独立監査</a></nav>',
        scope_html,
        "<h2>主条件：LightGBM θ</h2>", _html_table(_results_view(flat, "lgbm")),
        '<img src="report_assets/main_test_metrics.png" alt="Main classification metrics"><img src="report_assets/main_test_confusion.png" alt="Main confusion matrices">',
        directional_html,
        "<h2>事前固定した10比較</h2><p>4分類器間の6比較＋各分類器対Train-majorityの4比較。正の差は比較名B−AのBが優れる方向。7日moving-block bootstrapが主、1/28日は感度確認。CIはpointwise 95%、P値は主Balanced Accuracyの10検定をHolm補正する。raw決定や別の閾値条件を主familyへ混ぜない。blockはfold境界や15分足の欠損をまたがない。指定block長以下の連続segmentは全体を固定して再現するため、その部分の不確実性を推定しない。実際に再標本化できる日時の割合を最終列に示す。segmentがすべて固定ならCI・P値はN/Aとする。</p>", _html_table(inference_view),
        "<h2>回帰予測を閾値で分類したcontrol（記述的）</h2><p>既存の4時間リターン予測に同じθを適用した比較。以前のTrain範囲・教師・選択基準が異なるため、目的関数だけを変更した効果ではない。非確率の予測に確率を捏造せず、log loss/BrierはN/A、bootstrap検定も行わない。EWMA/GARCHは値動きの大きさの予測であり、方向分類器には含めない。</p>", _html_table(control_view),
        "<h2>閾値calibrationと正解クラスの分布</h2><p>R/√(a vhat)のTrain二乗平均は校正の定義により1になる。Testで1に固定されるわけではなく、正規分布も仮定しない。条件間でクラス定義が変わるため、異なる条件のAccuracyを直接比較して勝敗を決めない。</p>", _html_table(pd.DataFrame(calibration_view)),
        '<img src="report_assets/threshold_calibration.png" alt="Threshold calibration diagnostics">',
        "<h2>学習経過</h2><p>個々のVal曲線と選択時点を表示する。終了後の値の補間や、終了したrunの脱落で偏るlate-step平均は作らない。最大step到達（赤い×）はright-censoredであり、収束した証拠とは扱わない。patience停止も事前固定した運用上の停止規則である。</p>",
        '<img src="report_assets/validation_learning_curves.png" alt="Validation learning curves">',
        "<h2>補助条件（各10比較は探索的）</h2>"]
    for condition in ("wm", "fixed"):
        body.extend(["<h3>" + escape(CONDITION_LABELS[condition]) + "</h3>", _html_table(_results_view(flat, condition))])
    body.append("<p>" + escape(audit_text) + "。このreportはモデル、閾値、checkpoint、学習設定を変更しない。</p>")
    css = "body{font:15px/1.65 system-ui,sans-serif;color:#182333;background:#f6f8fb;margin:0 auto;max-width:1250px;padding:28px}h1,h2,h3{line-height:1.3}h2{margin-top:36px}table{background:white;border-collapse:collapse;width:100%;font-size:12px;display:block;overflow:auto}th,td{text-align:right;border-bottom:1px solid #dde3eb;padding:8px;white-space:nowrap}th:first-child,td:first-child{text-align:left}img{display:block;max-width:100%;height:auto;background:white;margin:20px 0}a{color:#125d9b}nav{background:white;padding:12px;border:1px solid #dde3eb}"
    return '<!doctype html><html lang="ja"><meta charset="utf-8"><title>Volatility-threshold classification</title><style>' + css + '</style><body>' + "\n".join(body) + '</body></html>'


def write_classification_report(config_path: str | Path | None = None) -> dict[str, Any]:
    from .wm_volatility_classification import DEFAULT_CONFIG, load_config, parents, verify
    cfg = load_config(config_path or DEFAULT_CONFIG)
    parent, frozen = verify(cfg)
    _parent, comparison_cfg = parents(cfg)
    out = Path(cfg["output_dir"])
    summary = json.loads((out / "summary.json").read_text())
    if summary["test_selection_allowed"] is not False:
        raise ValueError("report requires report-only Test results")
    folds = json.loads((out / "prepared/folds.json").read_text())
    with np.load(Path(parent["output_dir"]) / "prepared/arrays.npz") as z:
        arrays = {key: z[key] for key in ("index_ns", "returns")}
    flat = _flatten_metrics(summary["metrics"])
    inference = _inference_table(summary)
    calibration = _calibration_rows(cfg, arrays, folds)
    data_scope = _data_scope_rows(cfg, parent, arrays, folds)
    directional, directional_by_fold = _directional_diagnostics(out)
    controls = _regression_controls(cfg, arrays, folds, comparison_cfg)
    optimizer_amendment, optimizer_diagnostics = _optimizer_diagnostics(cfg, frozen)
    counts = {name: set() for name in ("direct", "wm")}
    for task in frozen["tasks"]:
        result = json.loads((out / "neural" / task["job"] / "fit.json").read_text())
        counts[task["model"]].add(result["parameters"])
    if any(len(values) != 1 for values in counts.values()):
        raise ValueError("neural parameter counts must be fixed within each registered model")
    model_parameters = pd.DataFrame([{"model": name, "parameters": next(iter(values)), "recipe": "CE" if name == "direct" else "full WM auxiliary objective + CE"} for name, values in counts.items()])
    artifacts = []
    for name, frame in (("classification_results.csv", flat), ("classification_inference.csv", inference),
                        ("threshold_calibration.csv", calibration), ("data_scope.csv", data_scope),
                        ("directional_diagnostics.csv", directional), ("directional_diagnostics_by_fold.csv", directional_by_fold),
                        ("model_parameters.csv", model_parameters),
                        ("regression_conversion_controls.csv", controls)):
        path = out / name; frame.to_csv(path, index=False); artifacts.append(str(path))
    if optimizer_amendment is not None:
        path = out / "logistic_optimizer_diagnostics.csv"
        optimizer_diagnostics.to_csv(path, index=False); artifacts.append(str(path))
    artifacts.extend(_plot_figures(out, summary, flat, calibration, frozen))
    audit_path = out / "final_independent_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    path = out / "report.html"
    path.write_text(_render_html(summary, flat, inference, controls, calibration, audit, data_scope=data_scope, directional=directional, model_parameters=model_parameters, optimizer_amendment=optimizer_amendment), encoding="utf-8")
    artifacts.append(str(path))
    return {"report": str(path), "artifacts": artifacts, "classification_rows": len(flat),
            "control_rows": len(controls), "calibration_rows": len(calibration)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    print(json.dumps(write_classification_report(args.config), indent=2))


if __name__ == "__main__":
    main()
