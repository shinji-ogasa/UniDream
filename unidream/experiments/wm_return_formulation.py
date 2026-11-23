"""Isolated, preregistered return formulation comparison alongside seven-task fits.

Fixed YAML only; existing regression source/config/artifacts are never changed.
The parent's all-fit Test barrier also gates this experiment's Test access.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import time
import traceback

import numpy as np
import pandas as pd
import torch
import yaml
from threadpoolctl import threadpool_limits

from . import wm_seven_task_benchmark as parent
from .wm_technical_ablation import ROOT, _input, _json, _sha
from .wm_return_formulation_models import DirectReturnClassifier, WMReturnClassifier, OTHER_TASKS
from .wm_return_formulation_classical import fit_return_classifiers, predict_return_classifiers
from .wm_return_formulation_evaluation import (
    make_return_labels, return_class_priors, fit_return_calibrator,
    fit_probability_calibrator, calibrated_probs,
)

DEFAULT_CONFIG = ROOT / "configs/experiments/wm_return_formulation_20261110.yaml"


def load_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg["groups"] != ["D"] or cfg["seeds"] != [7] or cfg["horizons"] != [16]
            or sorted(cfg["folds"]) != list(range(15)) or cfg["test_selection_allowed"] is not False):
        raise ValueError("fixed D / seed7 / 15fold / 4h report-only research")
    if cfg["threshold_train_rms_multiplier"] != .5:
        raise ValueError("threshold is fixed before Test")
    if cfg["validation_partition"] != "calendar_months_earlystop_calibration_selection":
        raise ValueError("disjoint validation months required")
    cfg["config_path"] = str(Path(path).resolve())
    for key in ("output_dir", "parent_config"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    names = ("wm_return_formulation.py", "wm_return_formulation_models.py",
             "wm_return_formulation_classical.py", "wm_return_formulation_evaluation.py",
             "wm_return_formulation_report.py", "wm_classification_evaluation.py")
    return {str((Path(__file__).parent / name).relative_to(ROOT)): _sha(Path(__file__).parent / name)
            for name in names}


def train_label_contract(returns, train_ids, k=.5):
    values = np.asarray(returns).reshape(-1)[np.asarray(train_ids, dtype=np.int64)]
    if not len(values) or not np.isfinite(values).all() or k <= 0:
        raise ValueError("finite Train-only returns and positive threshold multiplier required")
    rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    theta = float(k * rms)
    priors = return_class_priors(make_return_labels(values, theta))
    return {"rms": rms, "theta": theta, "priors": priors.tolist(), "n_train": len(values)}


def training_token_mask(arrays, positions, fold):
    return (arrays["mask"][positions]
            & (arrays["label_end_ns"][positions] < pd.Timestamp(fold["train_end"]).value)
            & (arrays["index_ns"][positions, None] >= pd.Timestamp(fold["train_start"]).value))


def calibration_middle_month(frame):
    result = frame.loc[frame["split"] == "calibration"].copy()
    if result.empty or result["decision_id"].duplicated().any():
        raise ValueError("one model's disjoint middle validation month required")
    return result.sort_values("decision_id").reset_index(drop=True)


def prepare(cfg):
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    if (out / "preregistration.json").exists():
        raise ValueError("registration is immutable")
    pcfg = parent.load_config(cfg["parent_config"])
    preg = parent.verify(pcfg, full=True)
    arrays = parent.arrays_for(pcfg)
    folds = []
    for f in cfg["folds"]:
        data = parent.fold_for(pcfg, f)
        old = next(row for row in preg["folds"] if row["fold"] == f)
        contract = train_label_contract(arrays["return"], data["train"], cfg["threshold_train_rms_multiplier"])
        if not np.isclose(contract["rms"], old["scales"]["return"][0], rtol=1e-6):
            raise ValueError("Train RMS differs from regression cohort")
        for split in ("train", "earlystop", "calibration", "selection", "test"):
            if not arrays["mask"][data[split]].all():
                raise ValueError("cohort contains invalid labels")
        folds.append({**old, **contract})
    jobs = [{"fold": f, "group": "D", "seed": 7, "model": m,
             "job": f"fold_{f:02d}_D_{m}_s7"}
            for f in cfg["folds"] for m in ("direct", "wm")]
    classics = [{"fold": f, "group": "D", "seed": 7, "job": f"fold_{f:02d}_D"}
                for f in cfg["folds"]]
    _json(out / "preregistration.json", {
        "registered_utc": utc_now(), "config": cfg, "config_sha256": _sha(cfg["config_path"]),
        "source_sha256": source_hashes(), "parent_registration_sha256": _sha(Path(pcfg["output_dir"]) / "preregistration.json"),
        "folds": folds, "neural_tasks": jobs, "classical_tasks": classics,
        "model": pcfg["model"], "training": {**pcfg["training"], **cfg["training"]},
        "classical": {**pcfg["classical"], **cfg["classical"]},
        "target": "future16-bar log return classes: down<-theta; flat including boundaries; up>theta; theta=.5*fullTrainRMS",
        "selection": "same six auxiliary supervised terms; replace normalized return MSE with all-token CE; common weighted objective firstValmonth768panel; same steps/patience as parent",
        "calibration": "frozen scalar->multinomial logistic C1, frozen categorical->one temperature[.1,10], middleValmonthONLY; thirdValmonthreport-only",
        "decisions": "probabilities: argmax(p/Trainpriors) primary, rawargmax auxiliary; rawscalar: threshold projection without probability scores",
        "test_access": "all classification fits and calibrators frozen; parent all360/60 global barrier and reference predictions also required; historicalTest exploratoryOOS",
        "limitations": ["one neural seed cannot establish seed stability", "shared seven-task neural objective vs per-task classical", "classifier adds a small hidden class head; scalar output unused", "calibrators differ in capacity; report raw and calibrated arms", "classification does not imply profitable policy", "inherited conservative15min market publication assumptions"],
    })
    status(cfg, "registered")


def verify(cfg):
    out = Path(cfg["output_dir"])
    reg = json.loads((out / "preregistration.json").read_text())
    if reg["config_sha256"] != _sha(cfg["config_path"]) or reg["source_sha256"] != source_hashes():
        raise ValueError("registered source/config changed")
    pcfg = parent.load_config(cfg["parent_config"])
    parent.verify(pcfg)
    if _sha(Path(pcfg["output_dir"]) / "preregistration.json") != reg["parent_registration_sha256"]:
        raise ValueError("parent registration changed")
    return reg, pcfg


def task_dir(cfg, kind, task):
    return Path(cfg["output_dir"]) / kind / task["job"]


def model_for(task, cfg, scales):
    cls = WMReturnClassifier if task["model"] == "wm" else DirectReturnClassifier
    return cls(63, cfg["model"], scales)


def low_priority_threads(threads):
    os.nice(10)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def predict(model, arrays, ids, data, cfg):
    pieces = {k: [] for k in (*OTHER_TASKS, "class_probabilities")}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), cfg["training"]["evaluation_batch_size"]):
            x, _ = _input(arrays, ids[start:start + cfg["training"]["evaluation_batch_size"]], data, "D", cfg["model"]["context"])
            result = model(torch.from_numpy(x).to(cfg["device"]))
            for key in pieces:
                pieces[key].append(result[key][:, -1].cpu().numpy())
    answer = {k: np.concatenate(value) for k, value in pieces.items()}
    if any(not np.isfinite(v).all() for v in answer.values()):
        raise RuntimeError("nonfinite classifier prediction")
    return answer


def objective(arrays, ids, result, scales, theta):
    actual = {k: arrays[k][ids] for k in parent.TASKS}
    predicted = {k: result[k] for k in OTHER_TASKS}
    # Preserve the parent's exact six-task NumPy objective.
    actual["return"] = np.zeros_like(actual["return"])
    predicted["return"] = actual["return"]
    answer = parent.objective_numpy(actual, predicted, scales)
    labels = make_return_labels(arrays["return"][ids, 0], theta)
    p = result["class_probabilities"][:, 0]
    ce = float(-np.log(np.clip(p[np.arange(len(ids)), labels], 1e-12, 1)).mean())
    answer["parts"].pop("return")
    answer["parts"]["classification"] = ce
    answer["score"] += ce
    return answer


def probability_frame(arrays, ids, probabilities, fold, family, split, theta):
    p = np.asarray(probabilities).reshape(len(ids), 3)
    if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=1e-5) or (p < 0).any():
        raise ValueError("invalid class probabilities")
    result = pd.DataFrame({"decision_id": ids, "timestamp": pd.to_datetime(arrays["index_ns"][ids], utc=True),
        "fold": fold, "seed": 7, "family": family, "split": split,
        "actual_return": arrays["return"][ids, 0], "theta": theta})
    result["actual"] = make_return_labels(result["actual_return"].to_numpy(), theta)
    result[["p0", "p1", "p2"]] = p
    return result


def write_parquet(frame, path):
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def train_neural(config_path, task):
    cfg = load_config(config_path); reg, pcfg = verify(cfg)
    pcfg["training"] = reg["training"]
    directory = task_dir(cfg, "neural", task); directory.mkdir(parents=True, exist_ok=True)
    with (directory / "training.log").open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        low_priority_threads(cfg["training"]["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        arrays, data = parent.arrays_for(pcfg), parent.fold_for(pcfg, task["fold"])
        fold = next(row for row in reg["folds"] if row["fold"] == task["fold"])
        model = model_for(task, pcfg, fold["scales"]).to(cfg["device"])
        train = reg["training"]
        opt = torch.optim.AdamW(model.parameters(), lr=train["learning_rate"], weight_decay=train["weight_decay"])
        rng = np.random.default_rng(7)
        best, anchor, last_significant, best_step = np.inf, np.inf, 0, 0
        history, best_state, counts = [], None, 0
        gradients = {key: 0 for key in (*OTHER_TASKS, "classification")}
        started, reason = time.monotonic(), "max_steps"
        for step in range(1, train["max_steps"] + 1):
            model.train()
            ids = rng.choice(data["train"], train["batch_size"], replace=True)
            x, base = _input(arrays, ids, data, "D", pcfg["model"]["context"])
            positions = ids[:, None] - pcfg["model"]["context"] + np.arange(1, pcfg["model"]["context"] + 1)[None, :]
            valid = training_token_mask(arrays, positions, fold)
            counts += int(valid.sum())
            targets = {key: torch.from_numpy(arrays[key][positions].astype(np.float32)).to(cfg["device"]) for key in parent.TASKS}
            clean_returns = np.where(valid, arrays["return"][positions], 0.)
            labels = make_return_labels(clean_returns.reshape(-1), fold["theta"]).reshape(valid.shape)
            next_return = np.log(arrays["raw"][positions, 3] / arrays["raw"][positions - 1, 3]).astype(np.float32)
            opt.zero_grad(set_to_none=True)
            loss, parts = model.loss(torch.from_numpy(x).to(cfg["device"]), torch.from_numpy(base).to(cfg["device"]), targets,
                torch.from_numpy(valid).to(cfg["device"]), torch.from_numpy(next_return).to(cfg["device"]),
                torch.from_numpy(labels).to(cfg["device"]))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            coverage = model.supervised_gradient_coverage()
            for key, value in coverage.items(): gradients[key] += int(value)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train["gradient_clip"], error_if_nonfinite=True)
            opt.step()
            if step % train["validation_every"] == 0:
                panel = predict(model, arrays, data["selection_panel"], data, pcfg)
                val = objective(arrays, data["selection_panel"], panel, fold["scales"], fold["theta"])
                history.append({"step": step, "selection": val, "batch_total_loss": float(loss.detach().cpu()),
                    "batch_loss_parts": {k: float(v.cpu()) for k, v in parts.items()}, "elapsed_seconds": time.monotonic() - started})
                print(json.dumps(history[-1]), flush=True)
                if val["score"] < best:
                    best, best_step = val["score"], step
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if not np.isfinite(anchor) or val["score"] < anchor - train["relative_improvement"] * abs(anchor):
                    anchor, last_significant = val["score"], step
                _json(directory / "progress.json", {**task, "step": step, "best_step": best_step, "best_score": best, "elapsed_seconds": time.monotonic() - started})
                if step >= train["min_steps"] and step - last_significant >= train["patience_steps"]:
                    reason = "validation_patience"; break
        if best_state is None or counts == 0 or not all(v > 0 for v in gradients.values()):
            raise RuntimeError("no checkpoint or missing head gradient coverage")
        model.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "task": task, "scales": fold["scales"],
            "registration_sha256": _sha(Path(cfg["output_dir"]) / "preregistration.json")}, directory / "research_model.pt")
        frames = []
        for split in ("earlystop", "calibration", "selection"):
            result = predict(model, arrays, data[split], data, pcfg)
            frames.append(probability_frame(arrays, data[split], result["class_probabilities"], task["fold"], task["model"], split, fold["theta"]))
        write_parquet(pd.concat(frames, ignore_index=True), directory / "validation_predictions.parquet")
        _json(directory / "fit.json", {**task, "registration_sha256": _sha(Path(cfg["output_dir"]) / "preregistration.json"),
            "elapsed_seconds": time.monotonic() - started, "steps": step, "best_step": best_step, "best_score": best,
            "stop_reason": reason, "max_step_censored": reason == "max_steps", "history": history,
            "target_counts": counts, "supervised_gradient_steps": gradients, "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_sha256": _sha(directory / "research_model.pt"), "validation_predictions_sha256": _sha(directory / "validation_predictions.parquet")})
    return task


def train_classical(config_path, task):
    cfg = load_config(config_path); reg, pcfg = verify(cfg)
    directory = task_dir(cfg, "classical", task); directory.mkdir(parents=True, exist_ok=True)
    with (directory / "training.log").open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log), threadpool_limits(1):
        os.nice(10)
        arrays, data = parent.arrays_for(pcfg), parent.fold_for(pcfg, task["fold"])
        fold = next(row for row in reg["folds"] if row["fold"] == task["fold"])
        started = time.monotonic()
        bundle = fit_return_classifiers(parent.tabular_x(pcfg, data["train"], "D"),
            make_return_labels(arrays["return"][data["train"], 0], fold["theta"]),
            parent.tabular_x(pcfg, data["earlystop"], "D"),
            make_return_labels(arrays["return"][data["earlystop"], 0], fold["theta"]), reg["classical"])
        with (directory / "models.pkl").open("wb") as stream: pickle.dump(bundle, stream, protocol=5)
        frames = []
        for split in ("earlystop", "calibration", "selection"):
            probabilities = predict_return_classifiers(bundle, parent.tabular_x(pcfg, data[split], "D"))
            for name, values in probabilities.items():
                family = "linear" if name == "logistic" else name
                frames.append(probability_frame(arrays, data[split], values, task["fold"], family, split, fold["theta"]))
        write_parquet(pd.concat(frames, ignore_index=True), directory / "validation_predictions.parquet")
        _json(directory / "fit.json", {**task, "registration_sha256": _sha(Path(cfg["output_dir"]) / "preregistration.json"),
            "elapsed_seconds": time.monotonic() - started, "metadata": bundle["metadata"],
            "model_sha256": _sha(directory / "models.pkl"), "validation_predictions_sha256": _sha(directory / "validation_predictions.parquet")})
    return task


def status(cfg, stage=None):
    out = Path(cfg["output_dir"])
    reg = json.loads((out / "preregistration.json").read_text())
    old = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
    result = {"utc": utc_now(), "stage": stage or old.get("stage", "training"),
        "neural": {"complete": sum((task_dir(cfg, "neural", t) / "fit.json").exists() for t in reg["neural_tasks"]), "total": len(reg["neural_tasks"])},
        "classical": {"complete": sum((task_dir(cfg, "classical", t) / "fit.json").exists() for t in reg["classical_tasks"]), "total": len(reg["classical_tasks"])},
        "test_opened": (out / "test_barrier.json").exists()}
    _json(out / "status.json", result)
    page = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>回帰と分類の比較</title><style>body{{font-family:system-ui;max-width:850px;margin:40px auto;line-height:1.8}}</style><h1>4時間先リターン：回帰 vs 3クラス分類</h1><p>全特徴量D・15fold・seed7・4モデル。段階: {result['stage']}</p><p>ニューラル {result['neural']['complete']}/30、線形＋LightGBMセット {result['classical']['complete']}/15。Test開放: {result['test_opened']}</p><p>分類の学習が終わっても、元の7タスク実験の全学習とVal選択が完了するまでTestを待つ。</p><p><a href="selection/report.md">Val暫定レポート</a> / <a href="final/report.md">Testを含む結果</a> / <a href="campaign.log">実行ログ</a></p><p>更新UTC: {result['utc']}</p></html>'''
    (out / "progress.html").write_text(page, encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result


def check_fit(cfg, kind, task):
    directory = task_dir(cfg, kind, task)
    fit = json.loads((directory / "fit.json").read_text())
    if fit["registration_sha256"] != _sha(Path(cfg["output_dir"]) / "preregistration.json"):
        raise ValueError("fit registration mismatch")
    model = "research_model.pt" if kind == "neural" else "models.pkl"
    key = "checkpoint_sha256" if kind == "neural" else "model_sha256"
    if _sha(directory / model) != fit[key] or _sha(directory / "validation_predictions.parquet") != fit["validation_predictions_sha256"]:
        raise ValueError("fitted model or validation artifact changed")
    return fit


def regression_path(pcfg, fold, family, split):
    kind = "neural" if family in ("direct", "wm") else "classical"
    job = f"fold_{fold:02d}_D" + (f"_{family}_s7" if kind == "neural" else "")
    filename = "test_predictions.parquet" if split == "test" else "validation_predictions.parquet"
    return Path(pcfg["output_dir"]) / kind / job / filename


def read_regression(pcfg, fold, family, split):
    path = regression_path(pcfg, fold, family, split)
    fit = json.loads((path.parent / "fit.json").read_text())
    expected = fit["validation_predictions_sha256"]
    if split == "test":
        evaluation = json.loads((path.parent / "evaluation.json").read_text())
        if evaluation["barrier_sha256"] != _sha(Path(pcfg["output_dir"]) / "test_barrier.json"):
            raise ValueError("reference Test provenance mismatch")
        expected = evaluation["prediction_sha256"]
    if _sha(path) != expected:
        raise ValueError("reference prediction artifact changed")
    frame = pd.read_parquet(path)
    result = frame.loc[(frame["split"] == split) & (frame["model"] == family)].sort_values("decision_id").reset_index(drop=True)
    if result.empty or result["decision_id"].duplicated().any():
        raise ValueError("missing or duplicate reference predictions")
    return result


def reference_ready(pcfg, reg, test=False):
    split = "test" if test else "calibration"
    for fold in reg["folds"]:
        for family in ("linear", "lightgbm", "direct", "wm"):
            path = regression_path(pcfg, fold["fold"], family, split)
            if not path.exists() or not (path.parent / ("evaluation.json" if test else "fit.json")).exists():
                return False
    return True


def freeze_calibrators(cfg):
    reg, pcfg = verify(cfg)
    out = Path(cfg["output_dir"])
    if (out / "calibrators.json").exists():
        return json.loads((out / "calibrators.json").read_text())
    fits, calibrators, reference_hashes = {}, {}, {}
    for kind, key in (("neural", "neural_tasks"), ("classical", "classical_tasks")):
        for task in reg[key]:
            check_fit(cfg, kind, task)
            fits[f"{kind}/{task['job']}/fit.json"] = _sha(task_dir(cfg, kind, task) / "fit.json")
            vf = pd.read_parquet(task_dir(cfg, kind, task) / "validation_predictions.parquet")
            fold = next(row for row in reg["folds"] if row["fold"] == task["fold"])
            for family, frame in vf.groupby("family"):
                cls = calibration_middle_month(frame)
                expected_ids = parent.fold_for(pcfg, task["fold"])["calibration"]
                if not np.array_equal(cls["decision_id"], expected_ids):
                    raise ValueError("calibration must use exactly the purged middle validation month")
                scalar = read_regression(pcfg, task["fold"], family, "calibration")
                if not np.array_equal(cls["decision_id"], scalar["decision_id"]) or not np.array_equal(cls["actual_return"], scalar["actual_return"]):
                    raise ValueError("calibration cohorts differ between formulations")
                calibrators[f"{task['fold']}/{family}"] = {
                    "classification": fit_probability_calibrator(cls["actual"].to_numpy(), cls[["p0", "p1", "p2"]].to_numpy()),
                    "regression": fit_return_calibrator(scalar["actual_return"].to_numpy(), scalar["pred_return"].to_numpy(), fold["theta"], fold["rms"]),
                    "validation_predictions_sha256": _sha(task_dir(cfg, kind, task) / "validation_predictions.parquet"),
                    "n_middle_month": len(cls), "fit_split": "calibration",
                }
                rp = regression_path(pcfg, task["fold"], family, "calibration")
                reference_hashes[str(rp)] = _sha(rp)
    result = {"registration_sha256": _sha(out / "preregistration.json"), "frozen_fit_sha256": fits,
        "reference_validation_sha256": reference_hashes, "calibrators": calibrators, "frozen_utc": utc_now()}
    _json(out / "calibrators.json", result)
    return result


def enforce_parent_test_barrier(cfg):
    pcfg = parent.load_config(cfg["parent_config"])
    path = Path(pcfg["output_dir"]) / "test_barrier.json"
    if not path.exists():
        raise ValueError("parent global Test barrier is closed")
    barrier = json.loads(path.read_text())
    if barrier["registration_sha256"] != _sha(Path(pcfg["output_dir"]) / "preregistration.json"):
        raise ValueError("parent Test barrier registration mismatch")
    if not barrier.get("validation_choices_sha256") or _sha(Path(pcfg["output_dir"]) / "validation_choices.json") != barrier["validation_choices_sha256"]:
        raise ValueError("parent validation choices not frozen")
    registration = json.loads((Path(pcfg["output_dir"]) / "preregistration.json").read_text())
    required = {f"{kind}/{t['job']}/fit.json"
                for kind, key in (("neural", "neural_tasks"), ("classical", "classical_tasks"))
                for t in registration[key]}
    manifest = barrier.get("frozen_fit_sha256", {})
    if not required or set(manifest) != required:
        raise ValueError("parent global Test barrier lacks the complete fit manifest")
    for relative, expected in manifest.items():
        fit_path = Path(pcfg["output_dir"]) / relative
        if not fit_path.exists() or _sha(fit_path) != expected:
            raise ValueError("parent global Test fit missing or changed")
    return pcfg, barrier


def open_test_barrier(cfg):
    reg, pcfg = verify(cfg)
    enforce_parent_test_barrier(cfg)
    out = Path(cfg["output_dir"])
    frozen = json.loads((out / "calibrators.json").read_text())
    if frozen["registration_sha256"] != _sha(out / "preregistration.json"):
        raise ValueError("calibration provenance mismatch")
    for kind, key in (("neural", "neural_tasks"), ("classical", "classical_tasks")):
        for task in reg[key]:
            check_fit(cfg, kind, task)
            fp = f"{kind}/{task['job']}/fit.json"
            if frozen["frozen_fit_sha256"].get(fp) != _sha(out / fp):
                raise ValueError("fit changed after calibration freeze")
    for path, expected in frozen["reference_validation_sha256"].items():
        if _sha(Path(path)) != expected: raise ValueError("reference validation changed")
    value = {"opened_utc": utc_now(), "registration_sha256": _sha(out / "preregistration.json"),
        "calibrators_sha256": _sha(out / "calibrators.json"),
        "parent_barrier_sha256": _sha(Path(pcfg["output_dir"]) / "test_barrier.json"), "test_selection_allowed": False}
    path = out / "test_barrier.json"
    if path.exists():
        old = json.loads(path.read_text())
        if any(old[k] != value[k] for k in value if k != "opened_utc"):
            raise ValueError("Test barrier artifacts changed")
    else: _json(path, value)


def evaluate_task(config_path, kind, task):
    cfg = load_config(config_path); reg, pcfg = verify(cfg)
    enforce_parent_test_barrier(cfg)
    barrier = json.loads((Path(cfg["output_dir"]) / "test_barrier.json").read_text())
    if (barrier["registration_sha256"] != _sha(Path(cfg["output_dir"]) / "preregistration.json")
            or barrier["parent_barrier_sha256"] != _sha(Path(pcfg["output_dir"]) / "test_barrier.json")
            or barrier["calibrators_sha256"] != _sha(Path(cfg["output_dir"]) / "calibrators.json")):
        raise ValueError("calibration changed after Test barrier")
    frozen = json.loads((Path(cfg["output_dir"]) / "calibrators.json").read_text())
    fp = f"{kind}/{task['job']}/fit.json"
    if frozen["frozen_fit_sha256"].get(fp) != _sha(Path(cfg["output_dir"]) / fp):
        raise ValueError("fit changed after classification freeze")
    check_fit(cfg, kind, task)
    low_priority_threads(1)
    pcfg["training"] = reg["training"]
    arrays, data = parent.arrays_for(pcfg), parent.fold_for(pcfg, task["fold"])
    fold = next(row for row in reg["folds"] if row["fold"] == task["fold"])
    directory = task_dir(cfg, kind, task)
    if kind == "neural":
        state = torch.load(directory / "research_model.pt", map_location="cpu", weights_only=False)
        model = model_for(task, pcfg, state["scales"]).to(cfg["device"])
        model.load_state_dict(state["state_dict"])
        result = predict(model, arrays, data["test"], data, pcfg)
        frames = [probability_frame(arrays, data["test"], result["class_probabilities"], task["fold"], task["model"], "test", fold["theta"])]
    else:
        with (directory / "models.pkl").open("rb") as stream: bundle = pickle.load(stream)
        result = predict_return_classifiers(bundle, parent.tabular_x(pcfg, data["test"], "D"))
        frames = [probability_frame(arrays, data["test"], p, task["fold"], "linear" if name == "logistic" else name, "test", fold["theta"])
                  for name, p in result.items()]
    write_parquet(pd.concat(frames, ignore_index=True), directory / "test_predictions.parquet")
    _json(directory / "evaluation.json", {"barrier_sha256": _sha(Path(cfg["output_dir"]) / "test_barrier.json"),
        "prediction_sha256": _sha(directory / "test_predictions.parquet"), "utc": utc_now()})
    return task


def combined_predictions(cfg, include_test=False):
    reg, pcfg = verify(cfg)
    if include_test:
        open_test_barrier(cfg)
    frozen = json.loads((Path(cfg["output_dir"]) / "calibrators.json").read_text())
    pieces = []
    for kind, key in (("neural", "neural_tasks"), ("classical", "classical_tasks")):
        for task in reg[key]:
            fold = next(row for row in reg["folds"] if row["fold"] == task["fold"])
            for split in (["selection", "test"] if include_test else ["selection"]):
                path = task_dir(cfg, kind, task) / ("test_predictions.parquet" if split == "test" else "validation_predictions.parquet")
                if split == "test":
                    evaluation = json.loads((path.parent / "evaluation.json").read_text())
                    if (evaluation["barrier_sha256"] != _sha(Path(cfg["output_dir"]) / "test_barrier.json")
                            or evaluation["prediction_sha256"] != _sha(path)):
                        raise ValueError("classification Test prediction provenance mismatch")
                cls = pd.read_parquet(path)
                cls = cls.loc[cls["split"] == split]
                for family, frame in cls.groupby("family"):
                    frame = frame.sort_values("decision_id").reset_index(drop=True)
                    scalar = read_regression(pcfg, task["fold"], family, split)
                    if not np.array_equal(frame["decision_id"], scalar["decision_id"]) or not np.array_equal(frame["actual_return"], scalar["actual_return"]):
                        raise ValueError("same-row comparison cannot drop or substitute samples")
                    fit = frozen["calibrators"][f"{task['fold']}/{family}"]
                    for formulation, variant in (("classification", "raw_probabilities"), ("classification", "calibrated"),
                                                   ("regression", "scalar_threshold"), ("regression", "calibrated")):
                        result = frame.copy()
                        result["formulation"], result["variant"] = formulation, variant
                        result["scalar_prediction"] = scalar["pred_return"].to_numpy() if formulation == "regression" else np.nan
                        if variant == "scalar_threshold":
                            result[["p0", "p1", "p2"]] = np.nan
                            result["pred"] = make_return_labels(scalar["pred_return"].to_numpy(), fold["theta"])
                            result["pred_raw"] = result["pred"]
                        else:
                            p = (scalar["pred_return"].to_numpy() if formulation == "regression" else frame[["p0", "p1", "p2"]].to_numpy())
                            if variant == "calibrated": p = calibrated_probs(p, fit[formulation])
                            result[["p0", "p1", "p2"]] = p
                            result["pred"] = np.argmax(p / np.asarray(fold["priors"]), axis=1)
                            result["pred_raw"] = p.argmax(axis=1)
                        for i, prior in enumerate(fold["priors"]): result[f"train_prior_{i}"] = prior
                        pieces.append(result)
    return pd.concat(pieces, ignore_index=True)


def make_report(cfg, include_test=False):
    from .wm_return_formulation_report import write_formulation_report
    frame = combined_predictions(cfg, include_test)
    output = Path(cfg["output_dir"]) / ("final" if include_test else "selection")
    output.mkdir(parents=True, exist_ok=True)
    write_parquet(frame, output / "predictions.parquet")
    return write_formulation_report(frame, output, cfg)


def run_fits(cfg):
    reg, _ = verify(cfg)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(cfg["training"]["workers"], mp_context=context, max_tasks_per_child=1) as gpu, ProcessPoolExecutor(cfg["classical"]["workers"], mp_context=context, max_tasks_per_child=1) as cpu:
        futures = {}
        for kind, key, pool, fn in (("neural", "neural_tasks", gpu, train_neural), ("classical", "classical_tasks", cpu, train_classical)):
            for task in reg[key]:
                if (task_dir(cfg, kind, task) / "fit.json").exists(): check_fit(cfg, kind, task)
                else: futures[pool.submit(fn, cfg["config_path"], task)] = (kind, task)
        for future in as_completed(futures):
            kind, task = futures[future]
            try:
                future.result()
            except Exception as exc:
                _json(Path(cfg["output_dir"]) / "failure.json", {
                    "utc": utc_now(), "kind": kind, "task": task, "error": repr(exc)})
                status(cfg, "failed")
                for pending in futures: pending.cancel()
                raise
            print(json.dumps({"completed": kind, **task}), flush=True)
            status(cfg, "training")


def campaign(cfg):
    out = Path(cfg["output_dir"])
    try:
        reg, pcfg = verify(cfg)
        run_fits(cfg)
        status(cfg, "waiting_for_regression_validation")
        while not reference_ready(pcfg, reg): time.sleep(20)
        freeze_calibrators(cfg)
        status(cfg, "validation_report")
        make_report(cfg)
        status(cfg, "waiting_for_parent_test_barrier")
        while not (Path(pcfg["output_dir"]) / "test_barrier.json").exists(): time.sleep(20)
        open_test_barrier(cfg)
        status(cfg, "test_inference")
        context = mp.get_context("spawn")
        for kind, key, workers in (("neural", "neural_tasks", cfg["training"]["workers"]), ("classical", "classical_tasks", cfg["classical"]["workers"])):
            with ProcessPoolExecutor(workers, mp_context=context, max_tasks_per_child=1) as pool:
                futures = [pool.submit(evaluate_task, cfg["config_path"], kind, task) for task in reg[key]]
                for future in as_completed(futures): future.result()
        status(cfg, "waiting_for_regression_test_predictions")
        while not reference_ready(pcfg, reg, test=True): time.sleep(20)
        status(cfg, "final_statistics")
        make_report(cfg, include_test=True)
        _json(out / "completion.json", {"complete": True, "utc": utc_now(), "registration_sha256": _sha(out / "preregistration.json")})
        status(cfg, "complete")
    except Exception as exc:
        _json(out / "failure.json", {"utc": utc_now(), "error": repr(exc), "traceback": traceback.format_exc()})
        status(cfg, "failed")
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "prepare": prepare(cfg)
    elif args.command == "run": campaign(cfg)
    else: status(cfg)


if __name__ == "__main__":
    main()
