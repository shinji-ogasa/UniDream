"""Descriptive overlap and selection-oracle analysis of frozen predictions.

This module never trains or selects anything. A selection oracle may choose
one of the constituent hard predictions using the realized label. It is not
an achievable forecast or an upper bound on all probability ensembles.

Neural seeds are paired by seed, never unioned: each row contains one fitted
model per architecture. Deterministic classical predictions are copied into
each seed panel. Pooled rates match the original classification report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .wm_classification_evaluation import balanced_prior_predictions, make_labels


MODELS = ("logistic", "lightgbm", "direct", "wm")
CLASSICAL = ("logistic", "lightgbm")
META = ("fold", "split", "decision_id", "timestamp", "actual", "theta", "return")
KEYS = ("fold", "split", "timestamp")
RULES = {"balanced_prior": "pred", "raw": "pred_raw"}
NAMES = {"logistic": "Logistic", "lightgbm": "LightGBM", "direct": "Classification Transformer", "wm": "WM"}


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if (cfg["condition"] != "fixed" or tuple(cfg["models"]) != MODELS
            or cfg["splits"] != ["val", "test"]
            or cfg["decision_rules"] != list(RULES)
            or cfg["analysis_role"] != "posthoc_descriptive"
            or cfg["test_selection_allowed"] is not False
            or cfg["ensemble_fitting"] is not False
            or not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"])
            or 0 in cfg["seeds"]):
        raise ValueError("requires frozen fixed-threshold, paired-seed, report-only analysis")
    source, output = Path(cfg["source_dir"]).resolve(), Path(cfg["output_dir"]).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("analysis output must be separate from frozen source outputs")
    return cfg


def align_predictions(frame: pd.DataFrame, seeds: tuple[int, ...], priors: dict[int, np.ndarray]) -> pd.DataFrame:
    """Require exact shared cohorts/labels and retain probabilities + decisions."""
    required = {*META, "condition", "model", "seed", "p0", "p1", "p2", "pred", "pred_raw"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("missing prediction columns or empty predictions")
    if (set(frame.condition) != {"fixed"} or set(frame.model) != set(MODELS)
            or set(frame.split) != {"val", "test"} or not seeds
            or len(set(seeds)) != len(seeds) or 0 in seeds):
        raise ValueError("requires fixed condition, four models, Val/Test and distinct neural seeds")
    if frame[list(required)].isna().any().any():
        raise ValueError("missing prediction values")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    if (frame.timestamp.astype("int64") % pd.Timedelta(minutes=15).value).any():
        raise ValueError("timestamps must align to 15-minute boundaries")
    if frame.duplicated([*KEYS, "model", "seed"]).any():
        raise ValueError("duplicate prediction keys")
    base = frame.loc[frame.model == "lightgbm", list(META)].sort_values(list(KEYS)).reset_index(drop=True)
    if base.duplicated(list(KEYS)).any():
        raise ValueError("classical model must have one fit per cohort")
    if not np.array_equal(make_labels(base["return"].to_numpy(), base.theta.to_numpy()), base.actual):
        raise ValueError("actual labels disagree with frozen returns and thresholds")
    prior_rows = np.array([priors[int(fold)] for fold in base.fold])
    panels = []
    for seed in seeds:
        panel = base.copy()
        panel["seed"] = seed
        for c in range(3):
            panel[f"train_prior_{c}"] = prior_rows[:, c]
        for model in MODELS:
            expected = {0} if model in CLASSICAL else set(seeds)
            all_model = frame.loc[frame.model == model]
            if set(all_model.seed) != expected:
                raise ValueError(f"unexpected source seeds for {model}")
            source_seed = 0 if model in CLASSICAL else seed
            part = all_model.loc[all_model.seed == source_seed].sort_values(list(KEYS)).reset_index(drop=True)
            if not base[list(KEYS)].equals(part[list(KEYS)]):
                raise ValueError(f"identical timestamp/fold/split coverage required for {model}")
            if not base[list(META)].equals(part[list(META)]):
                raise ValueError(f"identical truth/threshold/return/decision IDs required for {model}")
            probabilities = part[["p0", "p1", "p2"]].to_numpy()
            expected_pred = balanced_prior_predictions(probabilities, prior_rows)
            if not np.array_equal(part.pred, expected_pred):
                raise ValueError(f"saved prior-adjusted predictions disagree for {model}")
            if not np.array_equal(part.pred_raw, probabilities.argmax(axis=1)):
                raise ValueError(f"saved raw predictions disagree for {model}")
            panel[f"source_seed_{model}"] = source_seed
            for column in ("p0", "p1", "p2", "pred", "pred_raw"):
                panel[f"{column}_{model}"] = part[column].to_numpy()
            panel[f"correct_{model}"] = part.pred.to_numpy() == base.actual.to_numpy()
            panel[f"correct_raw_{model}"] = part.pred_raw.to_numpy() == base.actual.to_numpy()
        panels.append(panel)
    result = pd.concat(panels, ignore_index=True).sort_values([*KEYS, "seed"]).reset_index(drop=True)
    if result.duplicated([*KEYS, "seed"]).any():
        raise ValueError("duplicate aligned rows")
    return result


def _ratio(a: float, b: float) -> float | None:
    return float(a / b) if b else None


def complementarity_tables(actual: np.ndarray, predictions: np.ndarray, models=MODELS) -> dict[str, list[dict]]:
    """Exact contingency counts, correct-set Jaccard, rescue rates and oracles."""
    actual, predictions = np.asarray(actual), np.asarray(predictions)
    if (actual.ndim != 1 or not len(actual) or predictions.shape != (len(actual), len(models))
            or not np.isin(actual, [0, 1, 2]).all()
            or not np.isin(predictions, [0, 1, 2]).all() or len(set(models)) != len(models)):
        raise ValueError("requires valid three-class predictions on a shared nonempty cohort")
    correct = predictions == actual[:, None]
    support = np.bincount(actual.astype(int), minlength=3)
    if (support == 0).any():
        raise ValueError("all three actual classes are required")
    weights = 1 / (3 * support[actual.astype(int)])
    accuracy = correct.mean(0)
    # Plain reductions avoid platform BLAS floating-point status warnings and
    # are also appropriate for these tiny (four-column) boolean matrices.
    ba = (weights[:, None] * correct).sum(axis=0)
    rows = {"subsets": [], "subset_classes": [], "pairs": [], "pair_classes": [], "patterns": []}
    for size in range(1, len(models) + 1):
        for indices in combinations(range(len(models)), size):
            subset = "+".join(models[i] for i in indices)
            union = correct[:, indices].any(1)
            intersection = correct[:, indices].all(1)
            oracle_ba = float(weights[union].sum())
            best = indices[int(np.argmax(ba[list(indices)]))]
            rows["subsets"].append({
                "subset": subset, "size": size, "n": len(actual),
                "n_any_correct": int(union.sum()), "n_all_correct": int(intersection.sum()),
                "oracle_accuracy": float(union.mean()), "oracle_balanced_accuracy": oracle_ba,
                "best_single_model_by_ba": models[best], "best_single_balanced_accuracy": float(ba[best]),
                "oracle_gain_over_best_single_ba": oracle_ba - float(ba[best]),
                "best_single_accuracy": float(accuracy[list(indices)].max()),
                "oracle_gain_over_best_single_accuracy": float(union.mean() - accuracy[list(indices)].max()),
            })
            for c, name in enumerate(("down", "flat", "up")):
                mask = actual == c
                rows["subset_classes"].append({"subset": subset, "size": size, "class": name,
                    "n": int(support[c]), "n_any_correct": int(union[mask].sum()),
                    "n_all_correct": int(intersection[mask].sum()), "oracle_recall": float(union[mask].mean())})
            if size != 2:
                continue
            a, b = indices
            both, only_a, only_b, neither = (correct[:, a] & correct[:, b],
                correct[:, a] & ~correct[:, b], ~correct[:, a] & correct[:, b],
                ~correct[:, a] & ~correct[:, b])
            counts = [int(x.sum()) for x in (both, only_a, only_b, neither)]
            balanced = [float(weights[x].sum()) for x in (both, only_a, only_b, neither)]
            rows["pairs"].append({**rows["subsets"][-1], "model_a": models[a], "model_b": models[b],
                "model_a_balanced_accuracy": float(ba[a]), "model_b_balanced_accuracy": float(ba[b]),
                **dict(zip(("both_correct_n", "a_only_correct_n", "b_only_correct_n", "both_wrong_n"), counts)),
                **dict(zip(("both_correct_fraction", "a_only_correct_fraction", "b_only_correct_fraction", "both_wrong_fraction"), [x / len(actual) for x in counts])),
                "correct_set_jaccard": _ratio(counts[0], sum(counts[:3])),
                "error_set_jaccard": _ratio(counts[3], sum(counts[1:])),
                "rescue_b_given_a_wrong": _ratio(counts[2], counts[2] + counts[3]),
                "rescue_a_given_b_wrong": _ratio(counts[1], counts[1] + counts[3]),
                "balanced_correct_set_jaccard": _ratio(balanced[0], sum(balanced[:3])),
                "balanced_rescue_b_given_a_wrong": _ratio(balanced[2], balanced[2] + balanced[3]),
                "balanced_rescue_a_given_b_wrong": _ratio(balanced[1], balanced[1] + balanced[3]),
                "prediction_agreement": float(np.mean(predictions[:, a] == predictions[:, b])),
            })
            for c, name in enumerate(("down", "flat", "up")):
                mask = actual == c
                n = [int(x[mask].sum()) for x in (both, only_a, only_b, neither)]
                rows["pair_classes"].append({"subset": subset, "model_a": models[a], "model_b": models[b],
                    "class": name, "n": int(support[c]),
                    **dict(zip(("both_correct_n", "a_only_correct_n", "b_only_correct_n", "both_wrong_n"), n)),
                    "correct_set_jaccard": _ratio(n[0], sum(n[:3])),
                    "rescue_b_given_a_wrong": _ratio(n[2], n[2] + n[3]),
                    "rescue_a_given_b_wrong": _ratio(n[1], n[1] + n[3]),
                    "oracle_recall": sum(n[:3]) / support[c]})
    bits = correct @ (2 ** np.arange(len(models)))
    for pattern in range(2 ** len(models)):
        row = {"correctness_pattern": pattern,
               "correct_models": "+".join(models[i] for i in range(len(models)) if pattern & 2 ** i),
               "n": int((bits == pattern).sum()), "fraction": float(np.mean(bits == pattern))}
        for c, name in enumerate(("down", "flat", "up")):
            row[f"n_{name}"] = int(((bits == pattern) & (actual == c)).sum())
        rows["patterns"].append(row)
    return rows


def analyze_aligned(aligned: pd.DataFrame, group_columns: list[str]) -> dict[str, pd.DataFrame]:
    collected = {key: [] for key in ("subsets", "subset_classes", "pairs", "pair_classes", "patterns")}
    for group, part in aligned.groupby(group_columns, sort=True):
        if not isinstance(group, tuple):
            group = (group,)
        metadata = dict(zip(group_columns, group))
        metadata.update(n_unique_timestamps=len(part[["fold", "timestamp"]].drop_duplicates()),
                        n_seed_panels=int(part.seed.nunique()))
        for rule, prefix in RULES.items():
            tables = complementarity_tables(part.actual.to_numpy(), part[[f"{prefix}_{m}" for m in MODELS]].to_numpy())
            for key, rows in tables.items():
                collected[key].extend({**metadata, "decision_rule": rule, **row} for row in rows)
    return {key: pd.DataFrame(rows) for key, rows in collected.items()}


def audit_tables(aligned: pd.DataFrame, tables: dict[str, pd.DataFrame], source: Path) -> dict:
    """Independently recompute every pooled set count with pandas grouping."""
    checked = 0
    for row in tables["subsets"].to_dict("records"):
        part = aligned.loc[aligned.split == row["split"]]
        prefix = RULES[row["decision_rule"]]
        models = row["subset"].split("+")
        # A separate per-class count route avoids reusing the weighted-vector scorer.
        correct = pd.DataFrame({m: part[f"{prefix}_{m}"].to_numpy() == part.actual.to_numpy() for m in models})
        count = pd.DataFrame({"actual": part.actual.to_numpy(), "any": correct.any(axis=1), "all": correct.all(axis=1)})
        recalls = count.groupby("actual")["any"].mean()
        if (int(count["any"].sum()) != row["n_any_correct"]
                or int(count["all"].sum()) != row["n_all_correct"]
                or not np.isclose(recalls.mean(), row["oracle_balanced_accuracy"], atol=1e-12, rtol=0)):
            raise ValueError("independent set-count oracle audit failed")
        checked += 1
    for row in tables["pairs"].to_dict("records"):
        if sum(row[k] for k in ("both_correct_n", "a_only_correct_n", "b_only_correct_n", "both_wrong_n")) != row["n"]:
            raise ValueError("pair partition must cover all rows")
        if row["oracle_gain_over_best_single_ba"] < -1e-12:
            raise ValueError("oracle cannot underperform a constituent")
        patterns = tables["patterns"].loc[(tables["patterns"].split == row["split"])
            & (tables["patterns"].decision_rule == row["decision_rule"])]
        a_bit, b_bit = 2 ** MODELS.index(row["model_a"]), 2 ** MODELS.index(row["model_b"])
        counts = []
        for a_correct, b_correct in ((True, True), (True, False), (False, True), (False, False)):
            selected = patterns.loc[((patterns.correctness_pattern & a_bit) != 0) == a_correct]
            selected = selected.loc[((selected.correctness_pattern & b_bit) != 0) == b_correct]
            counts.append(int(selected.n.sum()))
        expected = [row[k] for k in ("both_correct_n", "a_only_correct_n", "b_only_correct_n", "both_wrong_n")]
        if counts != expected:
            raise ValueError("independent correctness-pattern pair audit failed")
    reported = pd.read_csv(source / "classification_results.csv")
    for row in tables["subsets"].query("size == 1").to_dict("records"):
        match = reported.loc[(reported.condition == "fixed") & (reported.model == row["subset"]) & (reported.split == row["split"])]
        if len(match) != 1:
            raise ValueError("missing original classification metric")
        key = "balanced_accuracy" if row["decision_rule"] == "balanced_prior" else "raw_balanced_accuracy"
        acc = "accuracy" if row["decision_rule"] == "balanced_prior" else "raw_accuracy"
        if not (np.isclose(float(match.iloc[0][key]), row["oracle_balanced_accuracy"], atol=1e-12, rtol=0)
                and np.isclose(float(match.iloc[0][acc]), row["oracle_accuracy"], atol=1e-12, rtol=0)):
            raise ValueError("single-model scores do not reproduce frozen report")
    return {"status": "passed", "independent_oracle_checks": checked,
            "single_model_metric_checks": 16, "pair_partition_checks": len(tables["pairs"]),
            "all_original_single_model_scores_reproduced": True}


def make_report(output: Path, tables: dict[str, pd.DataFrame], manifest: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = tables["subsets"].query("split == 'test' and decision_rule == 'balanced_prior'")
    pairs = tables["pairs"].query("split == 'test' and decision_rule == 'balanced_prior'")
    matrices = [np.zeros((4, 4)), np.ones((4, 4))]
    for row in primary.query("size == 1").to_dict("records"):
        i = MODELS.index(row["subset"])
        matrices[0][i, i] = 100 * row["oracle_balanced_accuracy"]
    for row in pairs.to_dict("records"):
        i, j = MODELS.index(row["model_a"]), MODELS.index(row["model_b"])
        matrices[0][i, j] = matrices[0][j, i] = 100 * row["oracle_balanced_accuracy"]
        matrices[1][i, j] = matrices[1][j, i] = row["correct_set_jaccard"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    labels = [NAMES[m] for m in MODELS]
    for axis, matrix, title, fmt in zip(axes, matrices,
            ("Selection-oracle balanced accuracy (%)", "Correct-set overlap (Jaccard %)"), (".2f", ".1f")):
        displayed = matrix if axis is axes[0] else matrix * 100
        axis.imshow(displayed, cmap="Blues", vmin=0, vmax=100)
        axis.set_xticks(range(4), labels, rotation=30, ha="right")
        axis.set_yticks(range(4), labels)
        axis.set_title(title)
        for i in range(4):
            for j in range(4):
                axis.text(j, i, format(displayed[i, j], fmt), ha="center", va="center",
                          color="white" if displayed[i, j] > 65 else "black")
    figure.suptitle("Fixed threshold / Test / prior-adjusted decisions\nPaired neural seeds; diagonal = individual model")
    figure.savefig(output / "complementarity.png", dpi=180)
    plt.close(figure)

    html = ['<!doctype html><meta charset="utf-8"><title>4モデルの正解集合とOracle</title>',
        '<style>body{font:16px sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;line-height:1.6}table{border-collapse:collapse;font-size:14px}th,td{padding:6px 9px;border:1px solid #ccc}img{max-width:100%}</style>',
        '<h1>固定閾値：4モデルの相補性</h1>',
        '<p>保存済みの同じ固定閾値・同じ日時を比較した事後的な記述分析。モデル、重み、閾値、checkpointは変更していない。Testは報告専用。</p>',
        '<p>主判定は元実験と同じ argmax(p / Train比率)。raw判定は別表。ニューラル3seedは同じseed同士で対応し、正解集合をseed間で合併しない。古典モデルの同じ予測を各seedパネルに複製し、seed×日時の件数をpoolする。確率アンサンブルではない。</p>',
        '<p>Oracleは、その場の正解を知っていて、構成モデルのどれかの出したクラスを選べる場合の上限。通常のAccuracyと3クラス再現率平均のBAを併記する。確率平均やstacking全般の数学的上限ではない。</p>',
        '<p>Oracleが高いだけでは予測能力や実現可能な改善を示さない。例えば3モデルが常に下落・横ばい・上昇をそれぞれ答えるだけでも、選択Oracleは100%になる。</p>',
        '<p>Jaccardは共通正解数÷少なくとも一方が正解した数。救済率は、片方が誤ったケースのうち相方が正解した割合。件数ベースとクラスを均等にした値をCSVに保存。統計的有意性は検定していない。</p>',
        '<p>Test日時数: '+str(manifest["test_unique_timestamps"])+', seedパネル数: '+str(manifest["n_seeds"])+'. 元の単体スコア16件を再現し、Oracle全60件を別計算で検証。</p>',
        '<img src="complementarity.png" alt="Oracleと正解集合の重なり">']
    for split in ("val", "test"):
        for rule in RULES:
            view = tables["subsets"].loc[(tables["subsets"].split == split) & (tables["subsets"].decision_rule == rule)].copy()
            columns = ["subset", "size", "oracle_balanced_accuracy", "oracle_accuracy", "oracle_gain_over_best_single_ba"]
            for c in columns[2:]:
                view[c] = (100 * view[c]).round(3)
            html += [f'<h2>{escape(split)} / {escape(rule)}：全15組（率は%、差はポイント）</h2>', view[columns].to_html(index=False)]
    html += ['<p><a href="aligned_predictions.parquet">同一日時に整列した全予測</a> / <a href="pairs.csv">ペア比較CSV</a> / <a href="subsets.csv">全組合せOracle CSV</a> / <a href="pair_classes.csv">クラス別ペアCSV</a> / <a href="subsets_by_fold.csv">期間別Oracle CSV</a> / <a href="subsets_by_seed.csv">seed別Oracle CSV</a> / <a href="audit.json">検証結果</a></p>']
    (output / "report.html").write_text("\n".join(html))


def run(config_path: Path) -> dict:
    cfg = load_config(config_path)
    source, output = Path(cfg["source_dir"]), Path(cfg["output_dir"])
    completion = json.loads((source / "completion_manifest.json").read_text())
    if completion["status"] != "complete" or completion["independent_audit"] != "passed":
        raise ValueError("requires completed, audited frozen classification outputs")
    inputs = sorted((source / "predictions").glob("fold_*.parquet"))
    if len(inputs) != completion["walk_forward_folds"]:
        raise ValueError("missing prediction fold files")
    artifacts = [*inputs, source / "classification_results.csv", source / "data_scope.csv", source / "completion_manifest.json",
                 source / "preregistration.json", *sorted((source / "prepared").glob("fold_*.npz"))]
    source_hashes = {str(path): file_hash(path) for path in artifacts}
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "analysis_role": "posthoc_descriptive", "config": cfg,
        "started_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": source_hashes,
        "config_sha256": file_hash(config_path), "analysis_code_sha256": file_hash(Path(__file__)),
        "test_selection_allowed": False, "ensemble_fitting": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    scope = pd.read_csv(source / "data_scope.csv")
    panels = []
    for path in inputs:
        frame = pd.read_parquet(path, filters=[("condition", "==", "fixed"), ("model", "in", list(MODELS))])
        fold_ids = frame.fold.unique()
        if len(fold_ids) != 1:
            raise ValueError("each input must contain one fold")
        fold = int(fold_ids[0])
        with np.load(source / "prepared" / f"fold_{fold:02d}.npz") as z:
            train_labels = z["label_fixed"][z["train"]]
            priors = np.bincount(train_labels, minlength=3) / len(train_labels)
            for split in ("val", "test"):
                ref = frame.loc[(frame.model == "lightgbm") & (frame.split == split)].sort_values("decision_id")
                ids = z[split]
                if not (np.array_equal(ref.decision_id, ids)
                        and np.array_equal(ref.actual, z["label_fixed"][ids])
                        and np.array_equal(ref.theta, z["theta_fixed"][ids])
                        and len(ref) == int(scope.loc[scope.fold == fold, f"{split}_n"].iloc[0])):
                    raise ValueError("prediction cohort differs from frozen prepared arrays")
        panels.append(align_predictions(frame, tuple(cfg["seeds"]), {fold: priors}))
        print(f"Aligned fold {fold:02d}: {len(panels[-1])} seed/date rows", flush=True)
    aligned = pd.concat(panels, ignore_index=True).sort_values([*KEYS, "seed"]).reset_index(drop=True)
    if aligned.duplicated([*KEYS, "seed"]).any():
        raise ValueError("duplicate aggregate keys")
    aligned.to_parquet(output / "aligned_predictions.parquet", index=False)
    if not pd.read_parquet(output / "aligned_predictions.parquet").equals(aligned):
        raise ValueError("saved aligned predictions did not round-trip exactly")
    tables = analyze_aligned(aligned, ["split"])
    for suffix, columns in (("", ["split"]), ("_by_fold", ["split", "fold"]), ("_by_seed", ["split", "seed"])):
        selected = tables if not suffix else analyze_aligned(aligned, columns)
        for name, table in selected.items():
            table.to_csv(output / f"{name}{suffix}.csv", index=False)
    audit = audit_tables(aligned, tables, source)
    audit["saved_predictions_exact_round_trip"] = True
    manifest.update(n_rows=len(aligned), n_seeds=len(cfg["seeds"]),
                    test_unique_timestamps=len(aligned.loc[aligned.split == "test", ["fold", "timestamp"]].drop_duplicates()),
                    val_unique_timestamps=len(aligned.loc[aligned.split == "val", ["fold", "timestamp"]].drop_duplicates()))
    if manifest["test_unique_timestamps"] != completion["test_unique_timestamps"]:
        raise ValueError("Test timestamps differ from frozen completed experiment")
    make_report(output, tables, manifest)
    if source_hashes != {str(path): file_hash(path) for path in artifacts}:
        raise ValueError("frozen source artifacts changed during analysis")
    audit["source_artifacts_unchanged"] = True
    (output / "audit.json").write_text(json.dumps(audit, indent=2))
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(), audit=audit,
                    output_sha256={p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file() and p.name not in {"manifest.json", "run.log"}})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ("status", "n_rows", "test_unique_timestamps", "audit")}, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/wm_classification_complementarity_20261110.yaml"))
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
