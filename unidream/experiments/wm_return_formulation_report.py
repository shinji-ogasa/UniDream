"""Matched reports for the isolated fixed-threshold return formulation study.

No fitting or Test-based choice is performed here.  Paired inference uses
non-circular UTC-day moving blocks with boundaries at folds and missing bars.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .wm_ablation_evaluation import holm_adjust
from .wm_classification_evaluation import (
    balanced_prior_predictions,
    paired_classification_block_bootstrap,
    score_classification,
)
from .wm_return_formulation_evaluation import make_return_labels, score_scalar_labels


FAMILIES = ("linear", "lightgbm", "direct", "wm")
VARIANTS = (("regression", "scalar_threshold"), ("regression", "calibrated"),
            ("classification", "raw_probabilities"), ("classification", "calibrated"))
KEYS = ["fold", "seed", "decision_id", "timestamp"]
ACTUAL_COLUMNS = ["actual_return", "theta", "actual", "train_prior_0", "train_prior_1", "train_prior_2"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def validate_matched_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Require all four formulations/four families on exactly the same rows."""
    frame = predictions.copy()
    required = set(KEYS + ACTUAL_COLUMNS + ["family", "split", "formulation", "variant", "p0", "p1", "p2",
                                             "pred", "pred_raw", "scalar_prediction"])
    if missing := required - set(frame):
        raise ValueError(f"missing formulation columns: {sorted(missing)}")
    if frame.empty or frame[list(required - {"p0", "p1", "p2", "scalar_prediction"})].isna().any().any():
        raise ValueError("nonempty complete prediction keys/outcomes are required")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    if not frame.split.isin(("selection", "test")).all() or not frame.family.isin(FAMILIES).all():
        raise ValueError("only selection/Test and the four registered model families are allowed")
    if set(frame.seed) != {7}:
        raise ValueError("this registered comparison uses seed 7 only")
    if np.any(frame.timestamp.astype("int64").to_numpy() % (15 * 60 * 10**9)):
        raise ValueError("timestamps must align to UTC 15-minute boundaries")
    for name in ("fold", "decision_id", "actual", "pred", "pred_raw"):
        values = frame[name].to_numpy(float)
        if not np.isfinite(values).all() or np.any(values != np.floor(values)):
            raise ValueError(f"{name} must contain finite integer values")
    if not frame.actual.isin((0, 1, 2)).all() or not frame.pred.isin((0, 1, 2)).all() or not frame.pred_raw.isin((0, 1, 2)).all():
        raise ValueError("class labels and decisions must be 0/1/2")
    if not np.isfinite(frame[ACTUAL_COLUMNS].to_numpy(float)).all():
        raise ValueError("actuals, thresholds and Train priors must be finite")
    expected = make_return_labels(frame.actual_return.to_numpy(), frame.theta.to_numpy())
    if not np.array_equal(expected, frame.actual.to_numpy()):
        raise ValueError("actual classes do not match the same fixed return thresholds")
    priors = frame[["train_prior_0", "train_prior_1", "train_prior_2"]].to_numpy(float)
    if np.any(priors <= 0) or not np.allclose(priors.sum(axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("Train priors must be strictly positive and sum to one")
    if frame.duplicated(["split", "family", "formulation", "variant", *KEYS]).any():
        raise ValueError("duplicate prediction keys")
    for split, split_frame in frame.groupby("split", sort=True):
        reference = None
        expected_pairs = {(family, formulation, variant) for family in FAMILIES for formulation, variant in VARIANTS}
        actual_pairs = set(map(tuple, split_frame[["family", "formulation", "variant"]].drop_duplicates().to_numpy()))
        if actual_pairs != expected_pairs:
            raise ValueError("every split needs all four variants of all four registered families")
        for _, panel in split_frame.groupby(["family", "formulation", "variant"], sort=True):
            panel = panel.sort_values(KEYS)
            cohort = panel[KEYS + ACTUAL_COLUMNS].reset_index(drop=True)
            if reference is None:
                reference = cohort
                if panel.timestamp.duplicated().any():
                    raise ValueError("folds cannot overlap in evaluated timestamp coverage")
            else:
                if not np.array_equal(reference[KEYS].to_numpy(), cohort[KEYS].to_numpy()):
                    raise ValueError("all variants and families must share identical fold/time decision coverage")
                if not np.array_equal(reference[ACTUAL_COLUMNS].to_numpy(), cohort[ACTUAL_COLUMNS].to_numpy()):
                    raise ValueError("all variants and families must share identical labels, returns, thresholds and Train priors")
        assert reference is not None
        for _, panel in reference.groupby("fold"):
            if not np.all(panel.theta == panel.theta.iloc[0]) or (panel[["train_prior_0", "train_prior_1", "train_prior_2"]].nunique() != 1).any():
                raise ValueError("each fold needs one fixed Train threshold and prior vector")
    scalar_mask = frame.variant == "scalar_threshold"
    scalar = frame.loc[scalar_mask]
    if not scalar[["p0", "p1", "p2"]].isna().all().all() or not np.isfinite(scalar.scalar_prediction).all():
        raise ValueError("scalar threshold rows need finite scalar predictions and no probability claims")
    scalar_labels = make_return_labels(scalar.scalar_prediction.to_numpy(), scalar.theta.to_numpy())
    if not np.array_equal(scalar_labels, scalar.pred.to_numpy()) or not np.array_equal(scalar_labels, scalar.pred_raw.to_numpy()):
        raise ValueError("saved scalar decisions do not match fixed threshold projection")
    probability = frame.loc[~scalar_mask]
    probabilities = probability[["p0", "p1", "p2"]].to_numpy(float)
    probability_priors = probability[["train_prior_0", "train_prior_1", "train_prior_2"]].to_numpy(float)
    predicted = balanced_prior_predictions(probabilities, probability_priors)
    if not np.array_equal(predicted, probability.pred.to_numpy()) or not np.array_equal(probabilities.argmax(axis=1), probability.pred_raw.to_numpy()):
        raise ValueError("saved probability decisions must use p/Train prior and raw argmax")
    if not frame.loc[frame.formulation == "classification", "scalar_prediction"].isna().all():
        raise ValueError("classification rows must not claim a scalar return forecast")
    if not np.isfinite(frame.loc[frame.formulation == "regression", "scalar_prediction"]).all():
        raise ValueError("all regression variants need a finite scalar return forecast")
    return frame


def _score(panel: pd.DataFrame) -> dict:
    if panel.variant.iloc[0] == "scalar_threshold":
        return score_scalar_labels(panel.actual.to_numpy(), panel.pred.to_numpy())
    priors = panel[["train_prior_0", "train_prior_1", "train_prior_2"]].to_numpy()
    return score_classification(panel.actual.to_numpy(), panel[["p0", "p1", "p2"]].to_numpy(), priors,
                                predictions=panel.pred.to_numpy(), raw_predictions=panel.pred_raw.to_numpy())


def _flatten(base: dict, scores: dict) -> dict:
    row = {**base, **{name: scores[name] for name in ("n", "accuracy", "balanced_accuracy", "macro_f1")}}
    row.update({"log_loss": scores.get("log_loss"), "multiclass_brier": scores.get("multiclass_brier")})
    for decision, values in (("primary", scores), ("raw", scores.get("raw"))):
        if values is None:
            continue
        if decision == "raw":
            row.update({"raw_" + name: values[name] for name in ("accuracy", "balanced_accuracy", "macro_f1")})
        for i, name in enumerate(("down", "flat", "up")):
            row[f"{decision}_recall_{name}"] = values["per_class_recall"][i]
            row[f"{decision}_precision_{name}"] = values["per_class_precision"][i]
            row[f"{decision}_actual_fraction_{name}"] = values["actual_class_frequency"][i]
            row[f"{decision}_predicted_fraction_{name}"] = values["predicted_class_frequency"][i]
            for j, predicted in enumerate(("down", "flat", "up")):
                row[f"{decision}_confusion_{name}_to_{predicted}"] = values["confusion_matrix"][i][j]
    return row


def _markdown(frame: pd.DataFrame) -> str:
    def format_value(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                       "| " + " | ".join("---" for _ in frame.columns) + " |",
                       *["| " + " | ".join(map(format_value, row)) + " |" for row in frame.itertuples(index=False, name=None)]])


def write_formulation_report(predictions: pd.DataFrame, output_dir: str | Path, config: dict) -> dict:
    """Write paired descriptive metrics and predeclared four-family Test CIs."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    frame = validate_matched_predictions(predictions)
    metrics, folded, full_metrics = [], [], []
    for grouping, destination in ((["split", "family", "formulation", "variant"], metrics),
                                  (["split", "family", "formulation", "variant", "fold"], folded)):
        for values, panel in frame.groupby(grouping, sort=True):
            base = {key: int(value) if key == "fold" else str(value) for key, value in zip(grouping, values)}
            scores = _score(panel)
            destination.append(_flatten(base, scores))
            if "fold" not in grouping:
                full_metrics.append({**base, "scores": scores})
    metrics = pd.DataFrame(metrics)
    folded = pd.DataFrame(folded)
    metrics.to_csv(out / "formulation_metrics.csv", index=False)
    folded.to_csv(out / "formulation_fold_metrics.csv", index=False)
    _json(out / "formulation_metrics.json", full_metrics)
    statistics = config.get("statistics", {})
    block_days = tuple(statistics.get("block_days", (1, 7, 28)))
    primary_days = int(statistics.get("primary_block_days", 7))
    if primary_days != 7 or primary_days not in block_days:
        raise ValueError("registered primary inference requires seven-day blocks")
    replicates = int(statistics.get("bootstrap_replicates", 2000))
    seed = int(statistics.get("seed", 20261110))
    inference = []
    test = frame.loc[(frame.split == "test") & (frame.variant == "calibrated")]
    if not test.empty:
        for i, family in enumerate(FAMILIES):
            panel = test.loc[test.family == family].copy()
            panel["condition"] = family
            panel["model"] = panel.formulation
            results = paired_classification_block_bootstrap(panel, "regression", "classification", condition=family,
                block_days=block_days, replicates=replicates, random_seed=seed + i)
            for block in results:
                common = {key: value for key, value in block.items() if key != "metrics"}
                for metric, values in block["metrics"].items():
                    inference.append({"family": family, "split": "test", "variant": "calibrated", **common,
                                      "metric": metric, **values})
    inference_frame = pd.DataFrame(inference)
    if not inference_frame.empty:
        primary = (inference_frame.block_days == primary_days) & (inference_frame.metric == "balanced_accuracy_difference")
        inference_frame["primary_family_test"] = primary
        inference_frame["p_holm_four_families"] = np.nan
        adjusted = holm_adjust({family: next((None if pd.isna(row.p_value) else float(row.p_value)
            for row in inference_frame.loc[primary & (inference_frame.family == family)].itertuples()), None) for family in FAMILIES})
        inference_frame.loc[primary, "p_holm_four_families"] = inference_frame.loc[primary, "family"].map(adjusted).astype(float)
    inference_frame.to_csv(out / "formulation_inference.csv", index=False)
    by_fold = folded.loc[folded.variant == "calibrated"].pivot(index=["split", "fold"], columns=["family", "formulation"], values="balanced_accuracy")
    by_fold.columns = [family + "_" + formulation for family, formulation in by_fold.columns]
    by_fold = by_fold.reset_index()
    by_fold.to_csv(out / "formulation_per_fold_ba.csv", index=False)
    coverage = frame.groupby(["split", "family", "formulation", "variant"], as_index=False).agg(
        n_rows=("decision_id", "size"), n_folds=("fold", "nunique"), n_seeds=("seed", "nunique"),
        first_utc=("timestamp", "min"), last_utc=("timestamp", "max"))
    coverage.to_csv(out / "formulation_coverage.csv", index=False)
    lines = ["# リターン回帰と3クラス分類の比較", "",
        "同じ時点・正解・Train固定閾値で、既存回帰モデルと新しい分類モデルを比較する探索的実験。過去に検討済みのTestを使うため、未使用期間での最終確認とは区別する。Testを見てモデルや判定ルールを選び直さない。", "",
        "閾値はfoldごとのTrainリターンRMSの0.5倍。0=下落、1=横ばい、2=上昇で、閾値と等しい場合は横ばい。実現分散を終端リターンのGaussian分散と仮定して確率化していない。", "",
        "モデルはVal第1月で選択し、第2月だけで校正する。回帰値からは固定Cの多クラスロジスティック回帰で確率へ変換し、分類確率には正の温度校正を適用する。Val第3月は別途報告し、判定ルールの主判定は事前固定のp/Trainクラス比率、通常のargmaxは補助。Log Loss・Brierは各variant自身の確率で計算し、p/Train比率による判定スコアには適用しない。", "",
        "校正器の表現力は対称ではない。回帰側は傾き・切片を持つ多クラスロジスティック変換、分類側は単一温度だけなので、校正後の差は予測表現と校正方法の両方を含む。このため未校正のscalar閾値・分類確率も併記し、純粋な損失関数だけの因果比較とは解釈しない。", "",
        "ニューラルモデルはseed 7の1回だけで、seedによる安定性は検証していない。ニューラルモデルは他の6タスクと共有して学習する一方、古典モデルはタスクごとの推定器であり、学習方式の違いを含む。", "",
        "分類ニューラルモデルには小さな隠れ層付きクラス予測ヘッドを追加している。使わない元のscalar出力も構造上は残るため、厳密に同一パラメータ数の比較ではない。他6タスクの教師・損失式は共通だが、共有表現は別々に学習されるので、その予測値まで同一になるとは限らない。", "",
        "## 校正後の分類−回帰：対応するTest比較", "",
        "BA差が正なら分類モデルが良い。時系列の依存を保つ日単位Moving Block Bootstrapを使い、fold境界・15分足の欠損を跨がない。7日を主解析、1日・28日を感度確認とし、4モデルfamilyのBA検定をHolm補正する。信頼区間は点ごとの95%区間で、複数比較補正した区間ではない。3種類のブロック長を独立した証拠とは数えない。", ""]
    if not inference_frame.empty:
        primary = inference_frame.loc[inference_frame.primary_family_test].copy()
        display = primary[["family", "estimate", "ci_low", "ci_high", "p_value", "p_holm_four_families", "n_paired_timestamps"]].copy()
        display[["estimate", "ci_low", "ci_high"]] *= 100
        display = display.rename(columns={"estimate": "BA差pp", "ci_low": "95%下限pp", "ci_high": "95%上限pp"})
        lines.extend([_markdown(display), ""])
    else:
        lines.extend(["Test予測がないため、推測統計はまだ実施していない。", ""])
    for split, title in (("test", "Testの全比較"), ("selection", "Val第3月の全比較")):
        table = metrics.loc[metrics.split == split, ["family", "formulation", "variant", "balanced_accuracy", "macro_f1", "raw_balanced_accuracy", "log_loss", "multiclass_brier"]].copy()
        if table.empty:
            continue
        table[["balanced_accuracy", "macro_f1", "raw_balanced_accuracy"]] *= 100
        lines.extend(["## " + title, "", "BA・Macro F1・raw BAは%。scalar_thresholdは確率を持たず、Log Loss・Brierを評価しない。", "", _markdown(table), ""])
    lines.extend(["## foldごとの校正後BA", "", "以下のBAは割合（0〜1）。", "", _markdown(by_fold), "",
        "## 保存ファイル", "", "- [全指標とクラス別Recall・混同行列](formulation_metrics.csv)",
        "- [指標のJSON](formulation_metrics.json)", "- [fold別全指標](formulation_fold_metrics.csv)",
        "- [全ブロック長の推測統計](formulation_inference.csv)", "- [予測期間とcoverage](formulation_coverage.csv)",
        "- [評価契約と入力hash](formulation_report_contract.json)", ""])
    (out / "report.md").write_text("\n".join(lines))
    input_hash = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()
    contract = {"prediction_table_hash": input_hash, "source_sha256": _sha(Path(__file__)), "n_prediction_rows": len(frame),
        "complete_paired_cohort_validated": True, "primary_comparison": "calibrated classification minus calibrated regression BA, four families",
        "primary_decision": "argmax(p/Train_prior)", "auxiliary_decision": "raw argmax; scalar threshold projection separately",
        "calibration": "caller fit using middle validation month only; report performs no fitting",
        "calibration_capacity_asymmetry": "regression uses multinomial logistic slopes/intercepts; classification uses one temperature; comparison includes calibration choice",
        "classification_capacity_change": "small hidden classification head added; unused scalar head retained; shared six auxiliary tasks, separately fitted representations",
        "historical_seen_test": True, "test_selection_allowed": False, "neural_seed_count": 1, "seed_stability_claim": False,
        "bootstrap": "non-circular moving UTC-day blocks, stratified by folds and contiguous segments; no IID bootstrap",
        "block_days": list(block_days), "primary_block_days": primary_days, "replicates": replicates,
        "multiplicity": "Holm four-family BA P values at seven-day primary block only; other metrics and block lengths exploratory",
        "config": config}
    _json(out / "formulation_report_contract.json", contract)
    summary = {"prediction_rows": len(frame), "metrics_rows": len(metrics), "fold_metric_rows": len(folded),
        "inference_rows": len(inference_frame), "test_unique_timestamps": int(frame.loc[frame.split == "test", "timestamp"].nunique()),
        "selection_unique_timestamps": int(frame.loc[frame.split == "selection", "timestamp"].nunique()),
        "n_neural_seeds": 1, "historical_seen_test": True, "source_sha256": _sha(Path(__file__)),
        "prediction_table_hash": input_hash, "report_sha256": _sha(out / "report.md")}
    _json(out / "formulation_report_completion.json", summary)
    return summary
