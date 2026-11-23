"""Validation-only ensembles of frozen classifiers, with new November Test.

No base model is refitted. Historical Test is explicitly exploratory. The
fresh Test gate opens only after all 45 fold/seed ensemble fits are frozen.
WM addition is compared within each of three preregistered ensemble methods.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
import yaml

from .wm_ablation_evaluation import holm_adjust
from .wm_classification_complementarity import MODELS, file_hash
from .wm_classification_evaluation import balanced_prior_predictions, make_labels, score_classification, paired_classification_block_bootstrap
from .wm_classification_models import ForecastClassifier
from .wm_technical_ablation import _features, _input, _json, _load_config
from .wm_volatility_classification import tabular_design, tabular_scaler, prediction_proba, load_config as classification_config


DEFAULT_CONFIG = Path("configs/experiments/wm_classification_ensemble_20261110.yaml")
METHODS = ("mean", "weighted", "stacking")
GROUPS = {"3": MODELS[:3], "4": MODELS}


def load_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg["test_selection_allowed"] is not False or cfg["base_model_training"] is not False
            or cfg["condition"] != "fixed" or tuple(cfg["models"]) != MODELS
            or tuple(cfg["methods"]) != METHODS or cfg["weight_grid_step"] != .1
            or cfg["base_folds"] != 15 or cfg["fresh_base_fold"] != 14 or cfg["purge_hours"] != 16
            or cfg["stacking"]["inner_train_months"] != 2
            or cfg["statistics"]["holm_family"] != "three_WM_addition_comparisons"
            or pd.Timestamp(cfg["fresh_start"]) < pd.Timestamp(cfg["previously_evaluated_end_exclusive"])):
        raise ValueError("frozen Val-only, no-base-training ensemble contract required")
    cfg["config_path"] = str(Path(path).resolve())
    return cfg


def quick_score(y, p, priors):
    p = np.asarray(p, dtype=float)
    p = p / p.sum(1, keepdims=True)
    pred = np.argmax(p / priors, axis=1)
    matrix = np.bincount(np.asarray(y, int) * 3 + pred, minlength=9).reshape(3, 3)
    support = matrix.sum(1)
    if (support == 0).any():
        raise ValueError("three actual classes required for model selection")
    return {"balanced_accuracy": float(np.mean(np.diag(matrix) / support)),
            "log_loss": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())}


def rank(score):
    return score["balanced_accuracy"], -score["log_loss"]


def probability_tensor(frame, models):
    values = frame[[f"p{c}_{m}" for m in models for c in range(3)]].to_numpy(dtype=float)
    values = values.reshape(len(frame), len(models), 3)
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(2), 1, atol=1e-6):
        raise ValueError("invalid frozen base probabilities")
    return values / values.sum(2, keepdims=True)


@lru_cache(maxsize=2)
def simplex_grid(size):
    return np.array([v for v in product(range(11), repeat=size) if sum(v) == 10], dtype=float) / 10


def select_weights(y, probabilities, priors):
    """Select one nonnegative simplex weight vector from Val only."""
    size = probabilities.shape[1]
    best, chosen = None, None
    history = []
    for weights in simplex_grid(size):
        p = np.sum(probabilities * weights[None, :, None], axis=1)
        score = quick_score(y, p, priors)
        key = (*rank(score), -float(np.sum((weights - 1 / size) ** 2)))
        history.append({"weights": weights.tolist(), **score})
        if best is None or key > best:
            best, chosen = key, weights.copy()
    return chosen, history


def stacking_split(timestamps, val_start, purge_hours=16):
    """First two calendar months fit; third month selects, with a 16h purge."""
    timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    start = pd.Timestamp(val_start)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    cutoff, end = start + pd.DateOffset(months=2), start + pd.DateOffset(months=3)
    if (timestamps < start).any() or (timestamps >= end).any() or not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("Val timestamps must be unique, ordered, and inside the registered quarter")
    fit = np.flatnonzero(timestamps + pd.Timedelta(hours=purge_hours) < cutoff)
    select = np.flatnonzero(timestamps >= cutoff)
    if not len(fit) or not len(select):
        raise ValueError("insufficient chronological meta-training/selection support")
    return fit, select


def fit_stacking(frame, models, fold, cfg):
    from lightgbm import Dataset, early_stopping, log_evaluation, train
    p = probability_tensor(frame, models)
    x = p.reshape(len(frame), -1).astype(np.float32)
    y = frame.actual.to_numpy(dtype=int)
    fit, select = stacking_split(frame.timestamp, fold["val_start"], cfg["purge_hours"])
    priors = np.bincount(y[fit], minlength=3) / len(fit)
    if (priors == 0).any() or min(len(fit), len(select)) < 256:
        raise ValueError("insufficient meta-training class/cohort support")
    options = cfg["stacking"]
    history = []
    for leaves, min_leaf in product(options["num_leaves"], options["min_data_in_leaf"]):
        params = {"objective": "multiclass", "num_class": 3, "metric": "None", "verbosity": -1,
            "num_leaves": leaves, "min_data_in_leaf": min_leaf, "learning_rate": options["learning_rate"],
            "lambda_l2": options["lambda_l2"], "num_threads": cfg["cpu_threads"], "seed": 20261110,
            "deterministic": True, "force_col_wise": True}
        def evaluate(predicted, dataset):
            labels = dataset.get_label().astype(int)
            matrix = np.bincount(labels * 3 + np.argmax(predicted / priors, axis=1), minlength=9).reshape(3, 3)
            return "balanced_accuracy", float(np.mean(np.diag(matrix) / matrix.sum(1))), True
        model = train(params, Dataset(x[fit], label=y[fit]), num_boost_round=options["max_rounds"],
            valid_sets=[Dataset(x[select], label=y[select])], feval=evaluate,
            callbacks=[early_stopping(options["patience"], verbose=False), log_evaluation(0)])
        probabilities = model.predict(x[select], num_threads=cfg["cpu_threads"])
        score = quick_score(y[select], probabilities, priors)
        history.append({"params": params, "rounds": int(model.best_iteration), "selection": score})
    chosen = max(history, key=lambda r: rank(r["selection"]))
    model = train(chosen["params"], Dataset(x, label=y), num_boost_round=chosen["rounds"])
    final_priors = np.bincount(y, minlength=3) / len(y)
    metadata = {"candidates": history, "selected": chosen, "decision_priors": final_priors.tolist(),
        "inner_fit_n": len(fit), "inner_select_n": len(select),
        "inner_fit_last": str(frame.timestamp.iloc[fit[-1]]), "inner_select_first": str(frame.timestamp.iloc[select[0]]),
        "refit_all_val_n": len(frame), "features": [f"p{c}_{m}" for m in models for c in range(3)]}
    return model, metadata


def fit_task(config_path, fold_id, seed):
    cfg = load_config(config_path)
    out = Path(cfg["output_dir"])
    folder = out / "fits" / f"fold_{fold_id:02d}_s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(cfg["aligned_predictions"], filters=[("split", "==", "val"), ("fold", "==", fold_id), ("seed", "==", seed)]).sort_values("timestamp")
    if frame.empty or set(frame.split) != {"val"}:
        raise ValueError("ensemble fitting requires Val rows only")
    fold = json.loads((Path(cfg["classification_dir"]) / "prepared/folds.json").read_text())[fold_id]
    priors = frame[[f"train_prior_{c}" for c in range(3)]].to_numpy()
    if not np.allclose(priors, priors[0], atol=1e-12):
        raise ValueError("base Train prior must be fixed within fold")
    y = frame.actual.to_numpy(dtype=int)
    result = {"fold": fold_id, "seed": seed, "selection_split": "val", "val_n": len(frame),
        "val_first": str(frame.timestamp.iloc[0]), "val_last": str(frame.timestamp.iloc[-1]), "groups": {}}
    for group, models in GROUPS.items():
        p = probability_tensor(frame, models)
        weights, history = select_weights(y, p, priors[0])
        stack, info = fit_stacking(frame, models, fold, cfg)
        stack.save_model(str(folder / f"stacking_{group}.txt"))
        singles = {m: quick_score(y, p[:, i], priors[0]) for i, m in enumerate(models)}
        result["groups"][group] = {"models": list(models), "base_priors": priors[0].tolist(),
            "weighted_weights": weights.tolist(), "weighted_candidates": history,
            "stacking": info, "val_best_model": max(singles, key=lambda m: rank(singles[m])), "single_val_scores": singles}
    result["artifacts"] = {path.name: file_hash(path) for path in folder.glob("*.txt")}
    _json(folder / "fit.json", result)
    return {"fold": fold_id, "seed": seed}


def ensemble_predictions(frame, folder, cfg, scope):
    from lightgbm import Booster
    fit = json.loads((folder / "fit.json").read_text())
    if fit["selection_split"] != "val":
        raise ValueError("non-Val ensemble fit")
    for name, digest in fit["artifacts"].items():
        if file_hash(folder / name) != digest:
            raise ValueError("meta-model changed after fit")
    result = []
    y = frame.actual.to_numpy(dtype=int)
    def append(name, probabilities, priors):
        probabilities = probabilities / probabilities.sum(1, keepdims=True)
        pred = balanced_prior_predictions(probabilities, priors)
        result.append(pd.DataFrame({"scope": scope, "condition": "fixed", "model": name,
            "fold": fit["fold"], "seed": fit["seed"], "timestamp": frame.timestamp.to_numpy(),
            "actual": y, "p0": probabilities[:, 0], "p1": probabilities[:, 1], "p2": probabilities[:, 2],
            "pred": pred, "pred_raw": probabilities.argmax(1),
            "decision_prior_0": priors[0], "decision_prior_1": priors[1], "decision_prior_2": priors[2]}))
    base_priors = np.array(fit["groups"]["4"]["base_priors"])
    for i, model in enumerate(MODELS):
        append(model, probability_tensor(frame, MODELS)[:, i], base_priors)
    for group, models in GROUPS.items():
        info = fit["groups"][group]
        p = probability_tensor(frame, models)
        append(f"mean_{group}", p.mean(1), base_priors)
        append(f"weighted_{group}", np.sum(p * np.array(info["weighted_weights"])[None, :, None], axis=1), base_priors)
        stack = Booster(model_file=str(folder / f"stacking_{group}.txt"))
        probs = stack.predict(p.reshape(len(p), -1).astype(np.float32), num_threads=cfg["cpu_threads"])
        append(f"stacking_{group}", probs, np.array(info["stacking"]["decision_priors"]))
        append(f"val_best_{group}", p[:, models.index(info["val_best_model"])], base_priors)
    return pd.concat(result, ignore_index=True)


def fresh_base_predictions(cfg, out):
    """Inference only: all six neural checkpoints and two classical artifacts."""
    from lightgbm import Booster
    from scipy.special import softmax
    source = Path(cfg["classification_dir"])
    index = cfg["fresh_base_fold"]
    fold = json.loads((source / "prepared/folds.json").read_text())[index]
    parent = _load_config("configs/experiments/wm_technical_ablation_20261109.yaml")
    base_cfg = classification_config("configs/experiments/wm_volatility_classification_20261110.yaml")
    arrays = dict(np.load(out / "fresh_data/arrays.npz"))
    ids = arrays["ids"]
    data = dict(np.load(source / "prepared" / f"fold_{index:02d}.npz"))
    theta = fold["calibration"]["fixed"]["fixed_theta"]
    data["theta_fixed"] = np.full(len(arrays["index_ns"]), theta)
    y = make_labels(arrays["returns"][ids, 1], theta)
    mean, std = tabular_scaler(data)
    design = tabular_design(_features(arrays["raw"], arrays["market"], ids, 256))
    x = np.column_stack(((design - mean) / std, np.full(len(ids), np.log(theta / fold["return_rms"]), dtype=np.float32)))
    folder = source / "classical" / f"fold_{index:02d}_fixed"
    metadata = json.loads((folder / "fit.json").read_text())
    for filename, digest in metadata["artifact_sha256"].items():
        if file_hash(folder / filename) != digest:
            raise ValueError("classical base artifact differs from frozen fit")
    logistic = pickle.loads((folder / "logistic.pkl").read_bytes())
    # Equivalent original-coordinate logistic prediction, avoiding platform BLAS warnings.
    logits = np.stack([(x * row).sum(1) + bias for row, bias in zip(logistic.coef_, logistic.intercept_)], axis=1)
    classical = {"logistic": softmax(logits, axis=1), "lightgbm": Booster(model_file=str(folder / "lightgbm.txt")).predict(x, num_threads=cfg["cpu_threads"])}
    panels = []
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    for seed in cfg["seeds"]:
        panel = pd.DataFrame({"fold": index, "seed": seed, "split": "fresh_test", "decision_id": ids,
            "timestamp": pd.to_datetime(arrays["index_ns"][ids], utc=True), "actual": y, "theta": theta,
            "return": arrays["returns"][ids, 1]})
        for c in range(3):
            panel[f"train_prior_{c}"] = fold["priors"]["fixed"][c]
        for name in MODELS:
            if name in classical:
                p = classical[name]
            else:
                folder = source / "neural" / f"fold_{index:02d}_fixed_{name}_s{seed}"
                info = json.loads((folder / "fit.json").read_text())
                if info["checkpoint_sha256"] != file_hash(folder / "research_model.pt"):
                    raise ValueError("base checkpoint changed")
                checkpoint = torch.load(folder / "research_model.pt", map_location="cpu", weights_only=False)
                if (checkpoint["task"]["fold"] != index or checkpoint["task"]["model"] != name
                        or checkpoint["task"]["seed"] != seed or checkpoint["task"]["condition"] != "fixed"):
                    raise ValueError("wrong neural checkpoint identity")
                model = ForecastClassifier(name, 63, parent["model"], fold["scales"]).to(cfg["device"])
                model.load_state_dict(checkpoint["model"])
                p = prediction_proba(base_cfg, model, arrays, ids, data, fold, "fixed")
                del model
                torch.mps.empty_cache()
            for c in range(3):
                panel[f"p{c}_{name}"] = p[:, c]
        panels.append(panel)
        print(f"Fresh base inference seed {seed} complete", flush=True)
    result = pd.concat(panels, ignore_index=True)
    result.to_parquet(out / "fresh_base_predictions.parquet", index=False)
    return result


def evaluate_frames(cfg, frames, scope):
    all_rows = pd.concat(frames, ignore_index=True)
    metrics = []
    for name, part in all_rows.groupby("model", sort=True):
        score = score_classification(part.actual.to_numpy(), part[["p0", "p1", "p2"]].to_numpy(), None,
            predictions=part.pred.to_numpy(), raw_predictions=part.pred_raw.to_numpy())
        metrics.append({"scope": scope, "model": name, "n_unique_timestamps": int(part.timestamp.nunique()),
            "n_seed_panels": int(part.seed.nunique()), **score})
    statistics = cfg["statistics"]
    inference = []
    contrasts = [(f"{method}_3", f"{method}_4", "WM_addition") for method in METHODS]
    contrasts += [(f"val_best_{group}", f"{method}_{group}", "ensemble_vs_Val_selected_single") for method in METHODS for group in GROUPS]
    for model_a, model_b, role in contrasts:
        result = paired_classification_block_bootstrap(all_rows, model_a, model_b, condition="fixed",
            block_days=statistics["block_days"], replicates=statistics["bootstrap_replicates"], random_seed=statistics["random_seed"])
        for r in result:
            r["scope"] = scope
            r["comparison_role"] = role
        inference.extend(result)
    primary = [r for r in inference if r["block_days"] == statistics["primary_block_days"] and r["comparison_role"] == "WM_addition"]
    adjusted = holm_adjust({r["comparison"]: r["metrics"]["balanced_accuracy_difference"]["p_value"] for r in primary})
    for row in primary:
        row["holm_adjusted_p"] = adjusted[row["comparison"]]
    for row in inference:
        row["confirmatory_interpretation_allowed"] = False
        row["interpretation"] = "new but too short for a confirmatory conclusion" if scope == "fresh" else "historical Test reused; exploratory only"
    return all_rows, metrics, inference


def report(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    html = ['<!doctype html><meta charset="utf-8"><title>WM ensemble comparison</title>',
        '<style>body{font:16px sans-serif;max-width:1100px;margin:32px auto;line-height:1.6}td,th{padding:7px;border:1px solid #ccc}table{border-collapse:collapse}</style>',
        '<h1>固定閾値：WMあり4モデル vs WMなし3モデル</h1>',
        '<p>Base classifierは追加学習していない。平均・重み・メタモデルはValのみから固定。StackingはVal前2か月で学習、16h purge後の最終月で選択し、選択済み設定で全Valをrefit。</p>',
        '<p>平均・重み平均は元のTrain比率で判定補正。Stackingはメタ学習データのクラス比率で判定補正する。確率そのものを補正せずLog Lossを計算。全モデルで同一固定閾値、同一Test日時。</p>',
        '<p>旧Testは探索に使用済みのretrospective OOS。Fresh Testは2026-11-01〜2026-11-10未満（UTC）の新規取得データ。過去のraw入力にこの期間が含まれないことを確認。新規Testは約8日で、7日ブロックの信頼区間は安定性が低く、長期の有意性を確定できない。</p>']
    for scope in ("fresh", "retrospective"):
        rows = []
        for m in summary["metrics"]:
            if m["scope"] == scope:
                rows.append({"model": m["model"], "BA (%)": 100 * m["balanced_accuracy"], "Accuracy (%)": 100 * m["accuracy"], "Macro F1": m["macro_f1"], "Log Loss": m["log_loss"]})
        html += [f'<h2>{scope}</h2>', pd.DataFrame(rows).round(4).to_html(index=False)]
    html += ['<p>3→4の差に対しseedを日時ごとに平均したpaired moving-block bootstrapを2000回。1/7/28日を併記し、fold・欠損区間をまたがない。短い区間を固定している割合はCI CSVに明示。Holm補正は3手法のWM追加比較だけのfamily。Freshは短期間のため確認的な有意差結論は出さない。</p>',
        '<p><a href="ensemble_metrics.csv">全評価指標CSV</a> / <a href="wm_addition_inference.csv">WM追加のCI CSV</a> / <a href="selection_summary.csv">Val選択内容</a> / <a href="summary.json">全結果</a></p>']
    (out / "report.html").write_text("\n".join(html))
    fresh = {m["model"]: m for m in summary["metrics"] if m["scope"] == "fresh"}
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    positions = np.arange(3)
    for offset, group, label in ((-.18, "3", "Without WM (3 models)"), (.18, "4", "With WM (4 models)")):
        axis.bar(positions + offset, [100 * fresh[f"{m}_{group}"]["balanced_accuracy"] for m in METHODS], width=.36, label=label)
    axis.axhline(100 / 3, color="grey", linestyle="--")
    axis.set_xticks(positions, ["Probability mean", "Val weighted", "LightGBM stacking"])
    axis.set_ylabel("Balanced accuracy (%)")
    axis.set_title("New November Test / fixed threshold / frozen base classifiers")
    axis.legend()
    fig.savefig(out / "fresh_comparison.png", dpi=180)
    plt.close(fig)


def run(path):
    cfg = load_config(path)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    registry = out / "preregistration.json"
    paths = [Path(cfg["aligned_predictions"]), Path(__file__), Path(__file__).with_name("wm_ensemble_fresh_data.py"), Path(cfg["config_path"])]
    source = Path(cfg["classification_dir"])
    # Record every frozen artifact used for inference; hashing reads no labels.
    paths += [source / "preregistration.json", source / "prepared/folds.json", source / "prepared/fold_14.npz"]
    paths += list((source / "classical/fold_14_fixed").glob("*"))
    for seed in cfg["seeds"]:
        for model in ("direct", "wm"):
            folder = source / "neural" / f"fold_14_fixed_{model}_s{seed}"
            paths += [folder / "fit.json", folder / "research_model.pt"]
    frozen = json.loads((source / "preregistration.json").read_text())
    if file_hash(source / "prepared/fold_14.npz") != frozen["prepared_sha256"]["fold_14.npz"]:
        raise ValueError("base normalization/threshold preprocessing changed")
    hashes = {str(p): file_hash(p) for p in paths if p.is_file()}
    contract = {"config": cfg, "source_sha256": hashes, "registered_utc": datetime.now(timezone.utc).isoformat(),
        "test_selection_allowed": False, "base_model_training": False,
        "primary_family": [f"{method}_4-{method}_3" for method in METHODS],
        "new_test_status": "not present in historical raw input; acquired independently; base/ensemble choices frozen before inference",
        "historical_test_status": "exploratory; already used for complementarity research"}
    if registry.exists():
        previous = json.loads(registry.read_text())
        if previous["source_sha256"] != hashes or previous["config"] != cfg:
            raise ValueError("registered analysis source/config changed")
    else:
        _json(registry, contract)
    futures = []
    with ProcessPoolExecutor(max_workers=cfg["parallel_workers"], mp_context=mp.get_context("spawn")) as pool:
        for fold in range(cfg["base_folds"]):
            for seed in cfg["seeds"]:
                if not (out / "fits" / f"fold_{fold:02d}_s{seed}" / "fit.json").exists():
                    futures.append(pool.submit(fit_task, cfg["config_path"], fold, seed))
        for future in as_completed(futures):
            print("Val fit complete " + json.dumps(future.result()), flush=True)
    selected = sorted((out / "fits").glob("*/fit.json"))
    if len(selected) != cfg["base_folds"] * len(cfg["seeds"]):
        raise ValueError("global selection barrier failed")
    fit_hashes = {str(p): file_hash(p) for p in (out / "fits").glob("*/*") if p.is_file()}
    _json(out / "test_access_barrier.json", {"passed": True, "released_utc": datetime.now(timezone.utc).isoformat(), "fit_sha256": fit_hashes})
    selection_rows = []
    for p in selected:
        fit = json.loads(p.read_text())
        for group, info in fit["groups"].items():
            selection_rows.append({"fold": fit["fold"], "seed": fit["seed"], "group": group,
                "weights": json.dumps(dict(zip(info["models"], info["weighted_weights"]))),
                "stacking_leaves": info["stacking"]["selected"]["params"]["num_leaves"],
                "stacking_rounds": info["stacking"]["selected"]["rounds"], "val_best_model": info["val_best_model"]})
    pd.DataFrame(selection_rows).to_csv(out / "selection_summary.csv", index=False)
    from .wm_ensemble_fresh_data import acquire
    acquire(cfg)
    fresh_panel = fresh_base_predictions(cfg, out)
    results = {"metrics": [], "inference": [], "fresh_data": json.loads((out / "fresh_data/manifest.json").read_text())}
    for scope in ("fresh", "retrospective"):
        frames = []
        folds = [cfg["fresh_base_fold"]] if scope == "fresh" else range(cfg["base_folds"])
        for fold in folds:
            for seed in cfg["seeds"]:
                if scope == "fresh":
                    panel = fresh_panel.loc[fresh_panel.seed == seed].sort_values("timestamp")
                else:
                    panel = pd.read_parquet(cfg["aligned_predictions"], filters=[("split", "==", "test"), ("fold", "==", fold), ("seed", "==", seed)]).sort_values("timestamp")
                frames.append(ensemble_predictions(panel, out / "fits" / f"fold_{fold:02d}_s{seed}", cfg, scope))
        predictions, metrics, inference = evaluate_frames(cfg, frames, scope)
        predictions.to_parquet(out / f"{scope}_predictions.parquet", index=False)
        results["metrics"].extend(metrics)
        results["inference"].extend(inference)
    if hashes != {str(p): file_hash(p) for p in paths if p.is_file()} or fit_hashes != {str(p): file_hash(p) for p in (out / "fits").glob("*/*") if p.is_file()}:
        raise ValueError("frozen inputs or selected ensembles changed during evaluation")
    _json(out / "summary.json", results)
    reference = pd.read_csv(source / "classification_results.csv")
    for result in results["metrics"]:
        if result["scope"] != "retrospective" or result["model"] not in MODELS:
            continue
        old = reference.loc[(reference.condition == "fixed") & (reference.split == "test") & (reference.model == result["model"])].iloc[0]
        for metric in ("balanced_accuracy", "accuracy", "macro_f1", "log_loss"):
            if not np.isclose(result[metric], old[metric], atol=1e-10, rtol=0):
                raise ValueError("retrospective base scores do not reproduce original frozen report")
    flat_metrics = [{k: v for k, v in m.items() if isinstance(v, (int, float, str))} for m in results["metrics"]]
    pd.DataFrame(flat_metrics).to_csv(out / "ensemble_metrics.csv", index=False)
    inference_rows = []
    for row in results["inference"]:
        for metric, value in row["metrics"].items():
            inference_rows.append({**{k: v for k, v in row.items() if k != "metrics"}, "metric": metric, **value})
    pd.DataFrame(inference_rows).to_csv(out / "wm_addition_inference.csv", index=False)
    report(out, results)
    _json(out / "completion.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "base_models_refitted": 0, "ensemble_fit_jobs": len(selected), "test_selection_allowed": False,
        "source_artifacts_unchanged": True, "selected_ensembles_unchanged": True,
        "retrospective_base_metrics_reproduced": True})
    print("Ensemble evaluation complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run(parser.parse_args().config)
