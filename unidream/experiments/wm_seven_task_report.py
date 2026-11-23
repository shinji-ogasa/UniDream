"""Read-only Japanese reports and paired loss inference for seven-task research.

No estimator, calibration, threshold or checkpoint is fitted here.  Historical
Test is retrospective OOS, not a pristine confirmation period.  Neural seed
losses are averaged at each timestamp BEFORE any bootstrap resampling.
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

from .wm_ablation_evaluation import BAR_INTERVAL, holm_adjust, variance_qlike
from .wm_seven_task_evaluation import TASKS


TASK_LABELS = {"return": "リターン", "variance": "実現分散", "downside": "現在価格からの最大下落",
               "crash": "急落確率", "dd_excess": "下落閾値の超過量", "advantage": "買い増し優位性（合成）",
               "recovery": "回復度（合成）"}
PRIMARY_METRICS = {"return": "zero_oos_r2", "variance": "qlike", "downside": "mae", "crash": "brier",
                   "dd_excess": "mae", "advantage": "rmse", "recovery": "rmse"}
LOSS_NAMES = {"return": "squared_error", "variance": "variance_qlike", "downside": "absolute_error",
              "crash": "brier", "dd_excess": "absolute_error", "advantage": "squared_error", "recovery": "squared_error"}
METRIC_COLUMNS = {
    "return": ("rmse", "zero_oos_r2", "pearson_ic", "spearman_ic"),
    "variance": ("qlike", "rmse", "calibration_mean_ratio"),
    "downside": ("mae", "rmse", "spearman_ic"),
    "crash": ("brier", "log_loss", "pr_auc", "prevalence", "mean_probability"),
    "dd_excess": ("mae", "rmse", "spearman_ic"),
    "advantage": ("rmse", "spearman_ic"), "recovery": ("rmse", "spearman_ic"),
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _loss_panel(frame: pd.DataFrame, group: str, model: str) -> tuple[pd.DataFrame, int]:
    """One loss vector per date after checking complete, non-independent seeds."""
    selected = frame.loc[(frame.group == group) & (frame.model == model)].copy()
    if selected.empty:
        raise ValueError(f"no predictions for {group}/{model}")
    selected["timestamp"] = pd.to_datetime(selected.timestamp, utc=True)
    keys = ["fold", "timestamp"]
    if selected.duplicated([*keys, "seed"]).any():
        raise ValueError("duplicate fold/timestamp/seed predictions")
    seed_sets = selected.groupby(keys)["seed"].agg(frozenset)
    if len(set(seed_sets)) != 1:
        raise ValueError("every timestamp must have the same complete seed set")
    n_seeds = len(seed_sets.iloc[0])
    actual_cols = ["actual_" + task for task in TASKS]
    prediction_cols = ["pred_" + task for task in TASKS]
    if not np.isfinite(selected[actual_cols + prediction_cols].to_numpy(float)).all():
        raise ValueError("paired inference requires all seven finite actuals and predictions")
    if (selected.groupby(keys)[actual_cols].nunique() != 1).any().any():
        raise ValueError("every seed must share identical outcomes")
    if np.any(selected.actual_variance < 0):
        raise ValueError("realized variance cannot be negative")
    if not np.isin(selected.actual_crash, (0, 1)).all() or not selected.pred_crash.between(0, 1).all():
        raise ValueError("crash must have binary labels and probability predictions")
    loss_columns = []
    for task in TASKS:
        actual, predicted = selected["actual_" + task].to_numpy(), selected["pred_" + task].to_numpy()
        loss = variance_qlike(actual, predicted) if task == "variance" else (
            np.abs(predicted - actual) if task in ("downside", "dd_excess") else np.square(predicted - actual)
        )
        column = "loss_" + task
        selected[column] = loss
        loss_columns.append(column)
    # These means are averages of losses, not losses of mean probabilities.
    aggregation = {column: "mean" for column in loss_columns}
    aggregation.update({column: "first" for column in actual_cols})
    return selected.groupby(keys, as_index=False).agg(aggregation), n_seeds


def paired_loss_inference(
    frame: pd.DataFrame,
    reference: tuple[str, str],
    candidate: tuple[str, str],
    *,
    block_days: tuple[int, ...] = (1, 7, 28),
    replicates: int = 2000,
    random_seed: int = 20261110,
) -> pd.DataFrame:
    """Paired, non-circular moving UTC-day blocks, stratified by fold/gaps.

    Improvement is mean reference loss minus mean candidate loss.  Each
    contiguous segment retains its original number of daily observations;
    blocks never bridge missing 15-minute rows or fold boundaries.  Short
    segments are replayed whole and their low resampling diversity is exposed.
    Intervals are percentile 95% intervals, and P values are two-sided centered
    bootstrap approximations.  Seeds are not independent replications.
    """
    if replicates < 2 or not block_days or any(int(d) != d or d < 1 for d in block_days):
        raise ValueError("need at least two replicates and positive whole-day blocks")
    left, seeds_reference = _loss_panel(frame, *reference)
    right, seeds_candidate = _loss_panel(frame, *candidate)
    paired = left.merge(right, on=["fold", "timestamp"], how="outer", suffixes=("_reference", "_candidate"),
                        indicator=True, validate="one_to_one")
    if not (paired["_merge"] == "both").all():
        raise ValueError("paired models must share identical fold/timestamp coverage")
    paired = paired.sort_values(["timestamp", "fold"]).reset_index(drop=True)
    if paired.timestamp.duplicated().any():
        raise ValueError("evaluated folds must not overlap in time")
    loss_cols = []
    for task in TASKS:
        if not np.array_equal(paired["actual_" + task + "_reference"], paired["actual_" + task + "_candidate"]):
            raise ValueError("paired models must share identical outcomes")
        column = "difference_" + task
        paired[column] = paired["loss_" + task + "_reference"] - paired["loss_" + task + "_candidate"]
        loss_cols.append(column)
    discontinuity = (paired.timestamp.diff() != BAR_INTERVAL) | (paired.fold != paired.fold.shift())
    paired["segment"] = discontinuity.cumsum()
    paired["date"] = paired.timestamp.dt.floor("D")
    paired["count"] = 1.0
    daily = paired.groupby(["segment", "date"], as_index=False)[[*loss_cols, "count"]].sum()
    segments = [part[[*loss_cols, "count"]].to_numpy(float) for _, part in daily.groupby("segment", sort=True)]
    total = daily[[*loss_cols, "count"]].sum().to_numpy(float)
    estimates = total[:-1] / total[-1]
    output = []
    for days in block_days:
        days = int(days)
        rng = np.random.default_rng(np.random.SeedSequence([random_seed, days]))
        resampled = np.zeros((replicates, len(TASKS) + 1), float)
        n_short = n_resampleable = n_starts = 0
        resampleable_count = 0.0
        for segment in segments:
            length = len(segment)
            block = min(length, days)
            n_short += length < days
            n_starts += length - block + 1
            if length > block:
                n_resampleable += 1
                resampleable_count += float(segment[:, -1].sum())
            blocks = (length + block - 1) // block
            starts = rng.integers(0, length - block + 1, size=(replicates, blocks))
            indices = (starts[:, :, None] + np.arange(block)).reshape(replicates, -1)[:, :length]
            resampled += segment[indices].sum(axis=1)
        distribution = resampled[:, :-1] / resampled[:, -1, None]
        for task_index, task in enumerate(TASKS):
            estimate = float(estimates[task_index])
            low = high = p_value = None
            if n_resampleable:
                samples = distribution[:, task_index]
                low, high = map(float, np.quantile(samples, [0.025, 0.975]))
                p_value = float((1 + np.sum(np.abs(samples - estimate) >= abs(estimate))) / (replicates + 1))
            output.append({"reference_group": reference[0], "reference_model": reference[1],
                "candidate_group": candidate[0], "candidate_model": candidate[1], "task": task,
                "loss": LOSS_NAMES[task], "block_days": days, "replicates": replicates,
                "estimate": estimate, "ci_low": low, "ci_high": high, "p_value": p_value,
                "n_paired_timestamps": len(paired), "n_seeds_reference": seeds_reference, "n_seeds_candidate": seeds_candidate,
                "n_segments": len(segments), "n_segments_shorter_than_block": n_short,
                "n_resampleable_segments": n_resampleable, "n_possible_block_starts": n_starts,
                "resampleable_timestamp_fraction": resampleable_count / total[-1],
                "inference_available": bool(n_resampleable)})
    return pd.DataFrame(output)


def _readable(pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in pooled.groupby(["split", "group", "model"], sort=True):
        row = dict(zip(("split", "group", "model"), keys))
        for _, score in part.iterrows():
            task = score.task
            for metric in METRIC_COLUMNS.get(task, ()):
                if metric in score:
                    row[f"{task}_{metric}"] = score[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_reports(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fits, curves = [], []
    for path in sorted(out.glob("neural/*/fit.json")):
        fit = json.loads(path.read_text())
        base = {key: fit.get(key) for key in ("fold", "group", "model", "seed", "steps", "best_step", "best_score",
                                               "stop_reason", "max_step_censored", "elapsed_seconds", "parameters")}
        base["target_coverage_positive"] = bool(np.all(np.asarray(fit.get("target_counts", [0])) > 0))
        base["supervised_gradient_coverage_positive"] = bool(np.all(np.asarray(fit.get("supervised_gradient_steps", [0])) > 0))
        fits.append(base)
        for point in fit.get("history", []):
            curves.append({**{key: base[key] for key in ("fold", "group", "model", "seed")},
                "step": point["step"], "earlystop_objective": point["selection"]["score"],
                "batch_total_loss": point["batch_total_loss"], "elapsed_seconds": point.get("elapsed_seconds")})
    return pd.DataFrame(fits), pd.DataFrame(curves)


def _plots(out: Path, curves: pd.DataFrame, fits: pd.DataFrame) -> list[str]:
    if curves.empty:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted(curves.group.unique())
    fig, axes = plt.subplots(len(groups), 2, figsize=(13, 3.1 * len(groups)), squeeze=False)
    colors = {"direct": "#2563eb", "wm": "#dc2626"}
    for i, group in enumerate(groups):
        for j, metric in enumerate(("earlystop_objective", "batch_total_loss")):
            axis = axes[i, j]
            for model, part in curves.loc[curves.group == group].groupby("model"):
                for _, run in part.groupby(["fold", "seed"]):
                    axis.plot(run.step, run[metric], color=colors.get(model, "gray"), alpha=0.08, linewidth=0.7)
                median = part.groupby("step")[metric].median()
                axis.plot(median.index, median, color=colors.get(model, "gray"), label=model, linewidth=2)
            axis.set(title=f"Group {group}: {metric}", xlabel="Training steps", ylabel="Loss")
            axis.grid(alpha=0.2)
            axis.legend()
    fig.suptitle("Thin: each fold/seed; bold: median of runs observed at that step", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model, part in fits.groupby("model"):
        axes[0].hist(part.best_step, bins=np.arange(0, max(400, int(fits.steps.max())) + 400, 200), alpha=0.5, label=model)
    axes[0].set(title="Selected checkpoint step", xlabel="Step", ylabel="Fits")
    axes[0].legend()
    counts = fits.groupby(["model", "stop_reason"]).size().unstack(fill_value=0)
    counts.plot.bar(ax=axes[1], title="Stopping reason", rot=0)
    fig.tight_layout()
    fig.savefig(out / "checkpoint_and_stopping.png", dpi=150)
    plt.close(fig)
    return ["learning_curves.png", "checkpoint_and_stopping.png"]


def _table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, na_rep="—", float_format=lambda value: f"{value:.6g}", escape=True)


def write_report(output_dir: str | Path) -> dict[str, Any]:
    """Write reproducible artifacts from completed/frozen prediction files."""
    out = Path(output_dir).resolve()
    paths = {name: out / name for name in ("all_predictions.parquet", "fold_metrics.csv", "pooled_metrics.csv", "preregistration.json")}
    registration = json.loads(paths["preregistration.json"].read_text())
    cfg = registration["config"]
    predictions = pd.read_parquet(paths["all_predictions.parquet"])
    predictions["timestamp"] = pd.to_datetime(predictions.timestamp, utc=True)
    folds = pd.read_csv(paths["fold_metrics.csv"])
    pooled = pd.read_csv(paths["pooled_metrics.csv"])
    source_hashes = {name: _sha(path) for name, path in paths.items()}
    statistics = cfg.get("statistics", {})
    blocks = tuple(statistics.get("block_days", (1, 7, 28)))
    primary_days = int(statistics.get("primary_block_days", 7))
    replicates = int(statistics.get("bootstrap_replicates", 2000))
    seed = int(statistics.get("seed", 20261110))
    test = predictions.loc[(predictions.split == "test") & (predictions.horizon == cfg.get("primary_horizon", 16))]
    comparisons = [("primary", ("D", "direct"), ("D", "wm")),
                   ("exploratory_LightGBM", ("D", "lightgbm"), ("D", "wm"))]
    comparisons += [("exploratory_TA", (a, model), (b, model)) for model in ("direct", "wm") for a, b in (("A", "B"), ("C", "D"))]
    available = set(zip(test.group, test.model))
    inference_parts, skipped = [], []
    for family, reference, candidate in comparisons:
        if reference not in available or candidate not in available:
            skipped.append({"family": family, "reference": list(reference), "candidate": list(candidate), "reason": "missing completed Test cohort"})
            continue
        scope_seed = seed + int.from_bytes(hashlib.sha256(repr((family, reference, candidate)).encode()).digest()[:4], "little")
        result = paired_loss_inference(test, reference, candidate, block_days=blocks, replicates=replicates, random_seed=scope_seed)
        result["family"] = family
        result["primary_inference"] = (family == "primary") & (result.block_days == primary_days)
        result["p_holm_seven_tasks"] = np.nan
        if family == "primary":
            mask = result.block_days == primary_days
            adjusted = holm_adjust({task: next((None if pd.isna(row.p_value) else float(row.p_value)
                for row in result.loc[mask & (result.task == task)].itertuples()), None) for task in TASKS})
            result.loc[mask, "p_holm_seven_tasks"] = result.loc[mask, "task"].map(adjusted)
        inference_parts.append(result)
    inference = pd.concat(inference_parts, ignore_index=True) if inference_parts else pd.DataFrame()
    inference.to_csv(out / "inference.csv", index=False)
    readable = _readable(pooled)
    readable.to_csv(out / "readable_7task_allgroups.csv", index=False)
    readable.loc[readable.group == "D"].to_csv(out / "readable_7task_D.csv", index=False)
    by_fold = []
    for keys, part in folds.groupby(["split", "group", "model", "fold", "task"]):
        split, group, model, fold, task = keys
        metric = PRIMARY_METRICS.get(task)
        if metric not in part:
            continue
        values = pd.to_numeric(part[metric], errors="coerce").dropna()
        by_fold.append({"split": split, "group": group, "model": model, "fold": int(fold), "task": task,
            "metric": metric, "mean_over_seeds": float(values.mean()) if len(values) else np.nan,
            "sd_over_seeds": float(values.std(ddof=1)) if len(values) > 1 else np.nan, "n_seeds": int(part.seed.nunique())})
    fold_summary = pd.DataFrame(by_fold)
    fold_summary.to_csv(out / "by_fold.csv", index=False)
    fits, curves = _fit_reports(out)
    fits.to_csv(out / "fit_diagnostics.csv", index=False)
    curves.to_csv(out / "learning_curves.csv", index=False)
    figures = _plots(out, curves, fits)
    coverage = []
    for keys, part in predictions.groupby(["split", "group", "model"], sort=True):
        finite = np.isfinite(part[["pred_" + task for task in TASKS]].to_numpy(float))
        coverage.append({"split": keys[0], "group": keys[1], "model": keys[2], "rows_including_seeds": len(part),
            "n_unique_timestamps": part.timestamp.nunique(), "n_folds": part.fold.nunique(), "n_seeds": part.seed.nunique(),
            "start_utc": str(part.timestamp.min()), "end_utc": str(part.timestamp.max()),
            **{task + "_finite_predictions": int(finite[:, i].sum()) for i, task in enumerate(TASKS)}})
    coverage_frame = pd.DataFrame(coverage)
    coverage_frame.to_csv(out / "prediction_coverage.csv", index=False)
    contract = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": source_hashes,
        "report_source_sha256": _sha(Path(__file__)), "test_report_only": True, "historical_seen_test": True,
        "selection": registration.get("selection"), "test_access": registration.get("test_access"),
        "target_definitions": registration.get("target_definitions"), "cohort": registration.get("cohort"),
        "primary_family": "D direct versus WM, seven tasks, paired difference direct_loss-WM_loss; positive favors WM",
        "primary_additive_losses": LOSS_NAMES, "primary_block_days": primary_days, "block_days": list(blocks),
        "replicates": replicates, "seed_handling": "average losses over seeds per timestamp before any resampling; never average predictions first",
        "bootstrap": "non-circular UTC-day moving blocks stratified by contiguous fold segments; do not cross gaps; percentile95% CI and centered two-sided approximate P",
        "multiplicity": "Holm correction for seven primary task P values at primary block length; other blocks sensitivity, other comparisons exploratory",
        "skipped_comparisons": skipped, "limitations": registration.get("limitations", [])}
    _json(out / "prediction_contract.json", contract)

    sections = ["<!doctype html><html lang='ja'><meta charset='utf-8'><title>7タスク比較実験</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 20px;color:#172033;line-height:1.7}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}th,td{padding:6px 10px;border-bottom:1px solid #dce3ec;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#f0f4f9}img{max-width:100%}a{color:#195bc5}.note{background:#fff3d7;padding:14px}h2{margin-top:38px}</style>",
        "<h1>4時間先・7タスクの予測比較</h1>",
        "<p class='note'>過去に確認済みのTestを使う、探索的な時系列OOS比較です。未使用データによる最終検証ではありません。Testの結果からモデル・閾値・checkpointを選び直していません。</p>",
        "<p>Trainは過去2年、Valは次の3か月、Testは次の3か月。Valの第1月で早期停止、第2月を補正用に予約（今回は恒等補正）、第3月をモデル・条件の比較用に分離します。ラベル終端は各分割境界より厳密に前とします。過去実験と共通の16時間先までpurge済みコホートを、今回の4時間先にも保守的に再利用します。</p>",
        "<p>入力A=OHLCV、B=OHLCV+TA、C=OHLCV+市場データ、D=OHLCV+市場データ+TA。TransformerとWMは同じ7教師・教師あり損失を使い、WMは追加の潜在・遷移目的を持ちます。単一タスク学習との比較は今回含みません。</p>",
        "<p>DDは現在価格から将来最安終値までの下落率です。急落は1.2%以上、超過量は1.2%を超えた下落分です。買い増し優位性と回復度は事前固定した合成スコアで、実売買損益や回復時間ではありません。</p>"]
    for split, heading in (("test", "過去Test・全特徴量D"), ("selection", "Val第3月・全特徴量D")):
        sections.append(f"<h2>{heading}</h2>")
        selected = pooled.loc[(pooled.split == split) & (pooled.group == "D")]
        rows = []
        for task in TASKS:
            metric = PRIMARY_METRICS[task]
            row = {"予測タスク": TASK_LABELS[task], "指標": metric + (" ↑" if task == "return" else " ↓")}
            part = selected.loc[selected.task == task]
            for model in ("unconditional", "past_variance", "ewma", "garch", "linear", "lightgbm", "direct", "wm"):
                values = part.loc[part.model == model, metric] if metric in part else pd.Series(dtype=float)
                row[model] = values.iloc[0] if len(values) else np.nan
            rows.append(row)
        sections.append(_table(pd.DataFrame(rows)))
        sections.append("<p>リターンのR²はゼロ予測比。QLIKEは分散の損失です。急落PR-AUCは陽性率と併読してください。各タスクのIC・RMSE・校正などは下の詳細CSVに保存しています。分散専用baselineは他6タスクを予測しません。</p>")
    sections.append("<h2>WMの追加効果・事前固定した主解析</h2><p>正の差はWMの損失が小さいことを表します。seedごとの損失を時点ごとに平均してから、同じ時点を対応させた日単位Block Bootstrapを実施します。7日ブロックが主解析、1日と28日は感度確認です。7タスクのP値にはHolm補正を適用し、3種類のブロック長を独立した証拠とは数えません。</p>")
    if not inference.empty:
        primary = inference.loc[inference.primary_inference, ["task", "loss", "estimate", "ci_low", "ci_high", "p_value", "p_holm_seven_tasks", "n_paired_timestamps", "n_segments", "resampleable_timestamp_fraction"]]
        sections.append(_table(primary))
        sections.append("<p>95%区間は損失差に対する未補正区間です。有意性の主判定にはHolm補正P値を用います。LightGBM比較・TA追加比較は探索的結果としてinference.csvに分けて保存しています。</p>")
    else:
        sections.append("<p>対応するTest予測がまだ揃っていないため、推測統計は未実施です。</p>")
    sections.append("<h2>foldとseedのばらつき</h2><p>表は同じfold内のseed平均です。seed間標準偏差を含むby_fold.csvも保存しています。3seedは3つの独立市場期間を意味しません。</p>")
    for task in TASKS:
        part = fold_summary.loc[(fold_summary.split == "test") & (fold_summary.group == "D") & (fold_summary.task == task)]
        if not part.empty:
            table = part.pivot(index="fold", columns="model", values="mean_over_seeds").reset_index()
            sections.append(f"<h3>{TASK_LABELS[task]}：{PRIMARY_METRICS[task]}</h3>" + _table(table))
    sections.append("<h2>学習曲線・停止理由</h2><p>細線は各fold・seed、太線はそのstepまで到達したrunの中央値です。停止したrunは後のstepに含まれないため、太線には構成の変化があります。早期停止の共通教師あり目的でcheckpointを選びます。batch総lossにはWM固有の補助目的が入り、モデル間の絶対値を直接比較できません。max_steps停止は収束を確認したという意味ではありません。</p>")
    if not fits.empty:
        sections.append(_table(fits.groupby(["group", "model", "stop_reason"], as_index=False).agg(n=("seed", "size"), median_best_step=("best_step", "median"), median_steps=("steps", "median"))))
    for figure in figures:
        sections.append(f"<img src='{figure}' alt='{figure}'>")
    sections.append("<h2>予測の範囲と保存ファイル</h2>" + _table(coverage_frame))
    for name, label in (("readable_7task_D.csv", "全特徴量Dの全指標"), ("readable_7task_allgroups.csv", "全入力条件・全指標"),
                        ("by_fold.csv", "fold・seedばらつき"), ("inference.csv", "全ブロック長・主解析と探索比較"),
                        ("fit_diagnostics.csv", "停止理由・教師と勾配のcoverage"), ("prediction_contract.json", "評価契約と入力ファイルhash")):
        sections.append(f"<p><a href='{name}'>{label}</a></p>")
    sections.append("<h2>事前登録した制約</h2><ul>" + "".join(f"<li>{escape(str(value))}</li>" for value in registration.get("limitations", [])) + "</ul></html>")
    (out / "report.html").write_text("\n".join(sections))
    summary = {"output_dir": str(out), "inference_rows": len(inference), "prediction_rows": len(predictions),
               "test_unique_timestamps": int(test.timestamp.nunique()), "fit_diagnostics_rows": len(fits),
               "source_sha256": source_hashes, "report_sha256": _sha(out / "report.html")}
    _json(out / "report_completion.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    args = parser.parse_args()
    print(json.dumps(write_report(args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
