"""Preregistered return-only regression, three-class and joint supervision.

All arms share the same four-output initialization and causal input history.
The six other seven-task objectives are absent. Test is report-only and opens
only after all 90 selected checkpoints and validation choices are frozen.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import pandas as pd
import torch
import yaml

from . import wm_seven_task_benchmark as parent
from .wm_technical_ablation import ROOT, _input, _json, _sha


DEFAULT_CONFIG = ROOT / "configs/experiments/wm_return_multitask_20261111.yaml"
ARMS = ("reg_only", "class_only", "joint")
ARM_WEIGHTS = {"reg_only": (1., 0.), "class_only": (0., 1.), "joint": (.5, .5)}
TRAINING_CONTRACT = ("max_steps", "min_steps", "patience_steps", "relative_improvement",
                    "validation_every", "selection_rows", "batch_size", "evaluation_batch_size",
                    "learning_rate", "weight_decay", "gradient_clip")


def now():
    return datetime.now(timezone.utc).isoformat()


def load_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg["groups"] != ["D"] or cfg["folds"] != list(range(15)) or cfg["seeds"] != [7]
            or cfg["models"] != ["direct", "wm"] or cfg["arms"] != list(ARMS)
            or cfg["horizons"] != [16] or cfg["test_selection_allowed"] is not False
            or cfg["other_supervised_tasks"] is not False or cfg["calibration"] != "identity"
            or cfg["threshold_train_rms_multiplier"] != .5
            or cfg["joint_loss_weights"] != {"normalized_return_mse": .5, "return_cross_entropy": .5}
            or cfg["validation_partition"] != "calendar_months_earlystop_calibration_selection"):
        raise ValueError("fixed D / return-only / seed7 / 15fold / three-arm / report-only scope required")
    cfg["config_path"] = str(Path(path).resolve())
    for name in ("output_dir", "parent_config"):
        cfg[name] = str((ROOT / cfg[name]).resolve())
    if cfg["training"]["workers"] < 1 or cfg["training"]["cpu_threads"] < 1:
        raise ValueError("positive runtime worker and thread limits required")
    return cfg


def source_hashes():
    names = ("wm_return_multitask.py", "wm_return_multitask_models.py",
             "wm_return_multitask_evaluation.py", "wm_return_multitask_report.py",
             "wm_return_multitask_statistics.py", "wm_classification_evaluation.py",
             "wm_return_formulation_evaluation.py", "wm_full_grid_statistics.py")
    return {str((Path(__file__).parent / name).relative_to(ROOT)): _sha(Path(__file__).parent / name)
            for name in names}


def return_labels(values, theta):
    """0=down, 1=flat including +/-theta, 2=up; preserve input shape."""
    values = np.asarray(values)
    if not np.isfinite(values).all() or not np.isfinite(theta) or theta <= 0:
        raise ValueError("finite returns and a positive fixed Train threshold required")
    return np.where(values < -theta, 0, np.where(values > theta, 2, 1)).astype(np.int64)


def train_label_contract(returns, train_ids, multiplier=.5):
    values = np.asarray(returns).reshape(-1)[np.asarray(train_ids, dtype=np.int64)].astype(np.float64)
    if not len(values) or not np.isfinite(values).all() or multiplier != .5:
        raise ValueError("finite Train-only returns and fixed threshold multiplier required")
    rms = float(np.sqrt(np.mean(values ** 2)))
    theta = multiplier * rms
    labels = return_labels(values, theta)
    counts = np.bincount(labels, minlength=3)
    if (counts < 200).any() or (counts / len(values) < .005).any():
        raise ValueError("insufficient Train support for the three fixed return classes")
    return {"rms": rms, "theta": theta, "priors": (counts / len(values)).tolist(),
            "class_counts": counts.tolist(), "n_train": len(values),
            "train_class_return_means": [float(values[labels == c].mean()) for c in range(3)],
            "label_rule": "down<-theta; flat includes both boundaries; up>theta; theta=.5*TrainRMS"}


def training_mask(arrays, positions, fold):
    return (arrays["mask"][positions]
            & (arrays["label_end_ns"][positions] < pd.Timestamp(fold["train_end"]).value)
            & (arrays["index_ns"][positions, None] >= pd.Timestamp(fold["train_start"]).value))


def fold_info(reg, fold):
    return next(row for row in reg["folds"] if row["fold"] == fold)


def directory(cfg, task):
    return Path(cfg["output_dir"]) / "neural" / task["job"]


def prepare(cfg):
    out = Path(cfg["output_dir"]); prepared = out / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    if (out / "preregistration.json").exists():
        raise ValueError("registration already frozen")
    pcfg = parent.load_config(cfg["parent_config"]); preg = parent.verify(pcfg, full=True)
    if cfg["model"] != pcfg["model"] or any(cfg["training"][k] != pcfg["training"][k] for k in TRAINING_CONTRACT):
        raise ValueError("same inherited context, model and training budget required")
    arrays = parent.arrays_for(pcfg); folds = []
    for f in cfg["folds"]:
        data = parent.fold_for(pcfg, f); old = fold_info(preg, f)
        contract = train_label_contract(arrays["return"], data["train"], cfg["threshold_train_rms_multiplier"])
        if not np.isclose(contract["rms"], old["scales"]["return"][0], rtol=1e-6):
            raise ValueError("Train return RMS differs from the immutable common cohort")
        # IDs and validity only: no Test outcome summaries are computed here.
        for split in ("train", "earlystop", "calibration", "selection", "test"):
            ids = data[split]
            if not len(ids) or not arrays["mask"][ids].all():
                raise ValueError("common fold contains invalid or empty labels")
        if not np.isin(data["selection_panel"], data["earlystop"]).all():
            raise ValueError("checkpoint panel must be wholly inside the first Val month")
        folds.append({**old, "label_contract": contract})
    _json(prepared / "fold_contracts.json", folds)
    # Arms are adjacent so runtime drift does not systematically order one arm first.
    jobs = [{"fold": f, "group": "D", "seed": 7, "model": m, "arm": arm,
             "job": f"fold_{f:02d}_D_{m}_{arm}_s7"}
            for f in cfg["folds"] for m in cfg["models"] for arm in cfg["arms"]]
    _json(out / "preregistration.json", {"experiment_id": cfg["experiment_id"], "registered_utc": now(),
        "config": cfg, "config_sha256": _sha(cfg["config_path"]), "source_sha256": source_hashes(),
        "parent_registration_sha256": _sha(Path(pcfg["output_dir"]) / "preregistration.json"),
        "prepared_sha256": {"fold_contracts.json": _sha(prepared / "fold_contracts.json")},
        "folds": folds, "neural_tasks": jobs, "model": cfg["model"], "training": cfg["training"],
        "arm_weights": ARM_WEIGHTS,
        "supervision": "return-only: normalized four-hour return MSE / three-class CE / .5 MSE+.5 CE; the other six supervised objectives are absent in every arm",
        "initialization": "same scalar+three-logit head and model state for all arms of a family/fold/seed; inactive outputs are not reported as fitted forecasts",
        "wm_auxiliaries": "same reconstruction .3, dynamicsKL .2, representationKL .05 and next-bar twohot reward .3 in all WM arms",
        "selection": "first Val month fixed768-row panel, own registered arm objective; common training budget/patience. Third Val month report-only fixed comparisons. Middle Val month reserved, identity calibration.",
        "decisions": "scalar uses fixed +/- .5 TrainRMS; probability primary argmax(p/Trainpriors), raw argmax auxiliary; class-to-scalar projection uses Train class-conditional return means only",
        "test_access": "independent all90-fit and Val-choice global barrier before any new Test inference; historical Test has already been inspected, retrospective OOS only",
        "primary_comparisons": "joint minus reg_only return OOS R2 and joint minus class_only prior-rule BA, each direct/WM: four two-sided paired block-bootstrap tests with Holm4; 7day primary,1/28day sensitivity",
        "limitations": ["one neural seed cannot establish initialization stability", "joint changes objective weights and therefore backbone gradients; output capacity is held equal",
            "WM auxiliaries remain additional objectives relative to direct models; arm comparisons are within family",
            "same inherited 16-hour-purged common cohort is conservative for four-hour returns", "predictive scores do not establish trading profitability"],
        "environment": {"torch": torch.__version__, "numpy": np.__version__, "device": cfg["device"]}})
    return status(cfg, "registered")


def verify(cfg, full=False):
    out = Path(cfg["output_dir"]); reg = json.loads((out / "preregistration.json").read_text())
    if reg["config_sha256"] != _sha(cfg["config_path"]) or reg["source_sha256"] != source_hashes():
        raise ValueError("registered source/config changed")
    pcfg = parent.load_config(cfg["parent_config"]); parent.verify(pcfg, full=full)
    if _sha(Path(pcfg["output_dir"]) / "preregistration.json") != reg["parent_registration_sha256"]:
        raise ValueError("immutable parent registration changed")
    if full:
        for name, expected in reg["prepared_sha256"].items():
            if _sha(out / "prepared" / name) != expected:
                raise ValueError("Train-only fold contract changed")
    return reg, pcfg


def model_for(task, cfg, scales):
    from .wm_return_multitask_models import DirectReturnMultitask, WMReturnMultitask
    cls = WMReturnMultitask if task["model"] == "wm" else DirectReturnMultitask
    return cls(63, cfg["model"], scales, task["arm"])


def worker_setup(threads):
    os.nice(5); torch.set_num_threads(threads); torch.set_num_interop_threads(1)


def initialization_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def predict(model, arrays, ids, data, cfg):
    model.eval(); pieces = {}
    with torch.no_grad():
        for start in range(0, len(ids), cfg["training"]["evaluation_batch_size"]):
            batch = ids[start:start + cfg["training"]["evaluation_batch_size"]]
            x, _ = _input(arrays, batch, data, "D", cfg["model"]["context"])
            values = model.predict_last(torch.from_numpy(x).to(cfg["device"]))
            for target, value in values.items():
                pieces.setdefault(target, []).append(value.cpu().numpy())
    answer = {target: np.concatenate(values) for target, values in pieces.items()}
    if not answer or any(not np.isfinite(value).all() for value in answer.values()):
        raise ValueError("empty/nonfinite return forecast")
    return answer


def supervised_objective(actual, predictions, arm, contract):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    actual = np.asarray(actual).reshape(-1)
    if not len(actual) or not np.isfinite(actual).all():
        raise ValueError("nonempty finite validation returns required")
    weights = ARM_WEIGHTS[arm]; parts = {}
    if weights[0]:
        forecast = np.asarray(predictions["return"]).reshape(-1)
        if forecast.shape != actual.shape or not np.isfinite(forecast).all():
            raise ValueError("aligned finite scalar forecasts required")
        parts["return_mse"] = float(np.mean(((forecast - actual) / max(contract["rms"], 1e-4)) ** 2))
    if weights[1]:
        labels = return_labels(actual, contract["theta"])
        p = np.asarray(predictions["class_probabilities"]).reshape(-1, 3)
        if (len(p) != len(actual) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
                or not np.allclose(p.sum(1), 1, atol=1e-6)):
            raise ValueError("aligned valid class probabilities required")
        parts["return_ce"] = float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean())
    return {"score": weights[0] * parts.get("return_mse", 0.) + weights[1] * parts.get("return_ce", 0.), "parts": parts}


def prediction_frame(arrays, ids, predictions, task, split, contract):
    ids = np.asarray(ids, dtype=np.int64); n = len(ids)
    result = pd.DataFrame({"decision_id": ids, "timestamp": pd.to_datetime(arrays["index_ns"][ids], utc=True),
        "fold": task["fold"], "group": "D", "seed": task["seed"], "model": task["model"],
        "arm": task["arm"], "split": split, "actual_return": arrays["return"][ids, 0],
        "theta": contract["theta"], "train_rms": contract["rms"]})
    result["actual_class"] = return_labels(result["actual_return"].to_numpy(), contract["theta"])
    result["pred_return"] = np.asarray(predictions["return"]).reshape(n) if task["arm"] != "class_only" else np.nan
    if task["arm"] != "reg_only":
        p = np.asarray(predictions["class_probabilities"]).reshape(n, 3)
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-6):
            raise ValueError("invalid categorical forecast")
    for c in range(3):
        result[f"p{c}"] = p[:, c] if task["arm"] != "reg_only" else np.nan
        result[f"train_prior_{c}"] = contract["priors"][c]
        result[f"train_class_return_mean_{c}"] = contract["train_class_return_means"][c]
    if task["arm"] != "class_only" and not np.isfinite(result["pred_return"]).all():
        raise ValueError("invalid scalar forecast")
    return result


def parquet(frame, path):
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False); temporary.replace(path)


@contextmanager
def task_claim(dest):
    dest.mkdir(parents=True, exist_ok=True); path = dest / "training.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError("task already claimed; explicit operational audit required") from None
    try:
        os.write(descriptor, json.dumps({"pid": os.getpid(), "utc": now()}).encode()); os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def train_neural(config_path, task):
    cfg = load_config(config_path); reg, pcfg = verify(cfg); dest = directory(cfg, task)
    if (dest / "fit.json").exists():
        check_fit(cfg, task); return task
    if (Path(cfg["output_dir"]) / "test_barrier.json").exists():
        raise ValueError("a missing fit cannot be retrained after the global Test barrier")
    with task_claim(dest), (dest / "training.log").open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        if (dest / "fit.json").exists():
            check_fit(cfg, task); return task
        worker_setup(cfg["training"]["cpu_threads"]); torch.use_deterministic_algorithms(True)
        random.seed(task["seed"]); np.random.seed(task["seed"]); torch.manual_seed(task["seed"])
        arrays, data = parent.arrays_for(pcfg), parent.fold_for(pcfg, task["fold"])
        fold = fold_info(reg, task["fold"]); contract = fold["label_contract"]
        model = model_for(task, cfg, fold["scales"]).to(cfg["device"])
        initial_sha = initialization_sha256(model); train = cfg["training"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=train["learning_rate"], weight_decay=train["weight_decay"])
        rng = np.random.default_rng(task["seed"]); best = anchor = np.inf
        best_step = last_significant = counts = 0; state = None; history = []
        coverage = {key: 0 for key in (("regression",) if task["arm"] == "reg_only" else
                                      ("classification",) if task["arm"] == "class_only" else ("regression", "classification"))}
        reason = "max_steps"; started = time.monotonic()
        for step in range(1, train["max_steps"] + 1):
            model.train(); ids = rng.choice(data["train"], train["batch_size"], replace=True)
            x, base = _input(arrays, ids, data, "D", cfg["model"]["context"])
            positions = ids[:, None] - cfg["model"]["context"] + np.arange(1, cfg["model"]["context"] + 1)[None, :]
            valid = training_mask(arrays, positions, fold); counts += int(valid.sum())
            actual = arrays["return"][positions].astype(np.float32)
            labels = return_labels(np.where(valid, arrays["return"][positions], 0.), contract["theta"])
            next_return = np.log(arrays["raw"][positions, 3] / arrays["raw"][positions - 1, 3]).astype(np.float32)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model.loss(torch.from_numpy(x).to(cfg["device"]), torch.from_numpy(base).to(cfg["device"]),
                {"return": torch.from_numpy(actual).to(cfg["device"])}, torch.from_numpy(valid).to(cfg["device"]),
                torch.from_numpy(next_return).to(cfg["device"]), torch.from_numpy(labels).to(cfg["device"]))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite training loss")
            loss.backward(); covered = model.supervised_gradient_coverage()
            if set(covered) != set(coverage):
                raise RuntimeError("active supervised gradient contract changed")
            for target, nonzero in covered.items(): coverage[target] += int(nonzero)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            if step % train["validation_every"] == 0:
                panel = predict(model, arrays, data["selection_panel"], data, cfg)
                score = supervised_objective(arrays["return"][data["selection_panel"]], panel, task["arm"], contract)
                history.append({"step": step, "selection": score, "batch_total_loss": float(loss.detach().cpu()),
                    "batch_loss_parts": {key: float(value.detach().cpu()) for key, value in parts.items()},
                    "elapsed_seconds": time.monotonic() - started})
                print(json.dumps(history[-1]), flush=True)
                if score["score"] < best:
                    best, best_step = score["score"], step
                    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                if not np.isfinite(anchor) or score["score"] < anchor - train["relative_improvement"] * abs(anchor):
                    anchor, last_significant = score["score"], step
                _json(dest / "progress.json", {**task, "step": step, "best_step": best_step, "best_score": best,
                    "elapsed_seconds": time.monotonic() - started})
                if step >= train["min_steps"] and step - last_significant >= train["patience_steps"]:
                    reason = "validation_patience"; break
        if state is None or counts == 0 or not all(coverage.values()):
            raise RuntimeError("missing checkpoint, labels or active-head gradients")
        model.load_state_dict(state)
        torch.save({"state_dict": state, "scales": fold["scales"], "task": task,
            "registration_sha256": _sha(Path(cfg["output_dir"]) / "preregistration.json")}, dest / "research_model.pt")
        frames = [prediction_frame(arrays, data[split], predict(model, arrays, data[split], data, cfg),
            task, split, contract) for split in ("earlystop", "calibration", "selection")]
        parquet(pd.concat(frames, ignore_index=True), dest / "validation_predictions.parquet")
        _json(dest / "fit.json", {**task, "registration_sha256": _sha(Path(cfg["output_dir"]) / "preregistration.json"),
            "elapsed_seconds": time.monotonic() - started, "steps": step, "best_step": best_step, "best_score": best,
            "stop_reason": reason, "max_step_censored": reason == "max_steps", "history": history,
            "target_counts": counts, "supervised_gradient_steps": coverage, "initialization_sha256": initial_sha,
            "prediction_availability": model.prediction_availability, "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_sha256": _sha(dest / "research_model.pt"), "validation_predictions_sha256": _sha(dest / "validation_predictions.parquet")})
    return task


def status(cfg, stage=None):
    out = Path(cfg["output_dir"]); path = out / "status.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    reg = json.loads((out / "preregistration.json").read_text())
    value = {"utc": now(), "stage": stage or old.get("stage", "training"), "neural": {
        "complete": sum((directory(cfg, task) / "fit.json").exists() for task in reg["neural_tasks"]),
        "total": len(reg["neural_tasks"])}, "test_opened": (out / "test_barrier.json").exists()}
    _json(path, value)
    page = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Return 回帰・分類・同時学習</title><style>body{{font-family:system-ui;max-width:900px;margin:40px auto;line-height:1.8}}</style><h1>Return の回帰／3クラス分類／同時学習</h1><p>段階: {value['stage']}。学習 {value['neural']['complete']}/{value['neural']['total']}。</p><p>D入力・15fold・seed7、Direct Transformer と WM。全方式で他6タスクの教師を外し、同一のスカラー＋3ロジット出力を用意。</p><p>同時学習は正規化 MSE ×0.5＋CE ×0.5。WM補助損失は全方式共通。Trainだけで分類閾値を固定し、全学習・Val選択完了後にTestを開放。</p><p><a href="selection/report.html">Val結果</a> / <a href="final/report.html">Test結果</a> / <a href="campaign.log">ログ</a></p><p>UTC {value['utc']}</p></html>'''
    temporary = out / "progress.html.tmp"; temporary.write_text(page); temporary.replace(out / "progress.html")
    print(json.dumps(value), flush=True); return value


def check_fit(cfg, task):
    dest = directory(cfg, task); fit = json.loads((dest / "fit.json").read_text())
    if fit["registration_sha256"] != _sha(Path(cfg["output_dir"]) / "preregistration.json"):
        raise ValueError("fit registration mismatch")
    if any(fit.get(key) != value for key, value in task.items()):
        raise ValueError("fit task identity mismatch")
    if (_sha(dest / "research_model.pt") != fit["checkpoint_sha256"]
            or _sha(dest / "validation_predictions.parquet") != fit["validation_predictions_sha256"]):
        raise ValueError("selected model/Val predictions changed")
    return fit


def frozen_fit_manifest(cfg, reg):
    manifest = {}; initializations = {}
    for task in reg["neural_tasks"]:
        path = directory(cfg, task) / "fit.json"
        if not path.exists(): raise ValueError("global Test closed: missing fit")
        fit = check_fit(cfg, task); manifest[str(path.relative_to(Path(cfg["output_dir"])))] = _sha(path)
        key = (task["fold"], task["model"], task["seed"])
        initializations.setdefault(key, set()).add(fit["initialization_sha256"])
    if any(len(values) != 1 for values in initializations.values()):
        raise ValueError("arm initialization states differ within family/fold/seed")
    return manifest


def prediction_files(cfg, reg, split):
    frames = []
    for task in reg["neural_tasks"]:
        dest = directory(cfg, task); check_fit(cfg, task)
        if split == "test":
            test_access(cfg, task)
            evaluation = json.loads((dest / "evaluation.json").read_text())
            filename = "test_predictions.parquet"
            if (evaluation["barrier_sha256"] != _sha(Path(cfg["output_dir"]) / "test_barrier.json")
                    or evaluation["prediction_sha256"] != _sha(dest / filename)):
                raise ValueError("Test prediction provenance changed")
        else: filename = "validation_predictions.parquet"
        frame = pd.read_parquet(dest / filename)
        selected = frame.loc[frame["split"] == split]
        if selected.empty or selected["decision_id"].duplicated().any():
            raise ValueError("one complete unique prediction cohort per task required")
        if not all((selected[key] == task[key]).all() for key in ("fold", "model", "arm", "seed", "group")):
            raise ValueError("prediction task identity mismatch")
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def freeze_validation_choices(cfg):
    from .wm_return_multitask_report import write_multitask_report
    reg, _ = verify(cfg); manifest = frozen_fit_manifest(cfg, reg)
    out = Path(cfg["output_dir"])
    choices_path = out / "validation_choices.json"
    if choices_path.exists():
        previous = json.loads(choices_path.read_text())
        if (previous.get("test_used") is not False
                or previous.get("registration_sha256") != _sha(out / "preregistration.json")
                or previous.get("frozen_fit_sha256") != manifest
                or previous.get("calibration") != "identity"
                or previous.get("selection_metrics_sha256") != _sha(out / "selection" / "metrics.csv")
                or previous.get("selection_day_statistics_sha256") != _sha(out / "selection" / "day_statistics.csv")):
            raise ValueError("existing frozen validation choices changed")
        return previous
    if (out / "test_barrier.json").exists():
        raise ValueError("frozen Val choices cannot be reconstructed after Test opened")
    summary = write_multitask_report(prediction_files(cfg, reg, "selection"), out / "selection", cfg)
    _json(out / "validation_choices.json", {"frozen_utc": now(), "test_used": False,
        "registration_sha256": _sha(out / "preregistration.json"), "frozen_fit_sha256": manifest,
        "calibration": "identity", "rule": "all six fixed family/arm conditions retained; four preregistered comparisons, no Test-dependent choices",
        "selection_metrics_sha256": _sha(out / "selection" / "metrics.csv"),
        "selection_day_statistics_sha256": _sha(out / "selection" / "day_statistics.csv"), "report": summary})


def open_test_barrier(cfg):
    reg, _ = verify(cfg, full=True); out = Path(cfg["output_dir"])
    manifest = frozen_fit_manifest(cfg, reg); choices_path = out / "validation_choices.json"
    if not choices_path.exists(): raise ValueError("Val choices not frozen")
    choices = json.loads(choices_path.read_text())
    if (choices.get("test_used") is not False or choices.get("registration_sha256") != _sha(out / "preregistration.json")
            or choices.get("frozen_fit_sha256") != manifest
            or choices.get("selection_metrics_sha256") != _sha(out / "selection" / "metrics.csv")
            or choices.get("selection_day_statistics_sha256") != _sha(out / "selection" / "day_statistics.csv")):
        raise ValueError("invalid validation choices provenance")
    value = {"opened_utc": now(), "registration_sha256": _sha(out / "preregistration.json"),
        "frozen_fit_sha256": manifest, "validation_choices_sha256": _sha(choices_path), "test_selection_allowed": False}
    path = out / "test_barrier.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if any(previous.get(key) != value[key] for key in value if key != "opened_utc"):
            raise ValueError("frozen Test barrier artifacts changed")
        return previous
    _json(path, value); return value


def test_access(cfg, task):
    reg, _ = verify(cfg); out = Path(cfg["output_dir"]); path = out / "test_barrier.json"
    if not path.exists(): raise ValueError("global Test barrier closed")
    barrier = json.loads(path.read_text())
    if (barrier.get("registration_sha256") != _sha(out / "preregistration.json")
            or barrier.get("test_selection_allowed") is not False
            or barrier.get("validation_choices_sha256") != _sha(out / "validation_choices.json")):
        raise ValueError("Test barrier/choices changed")
    required = {f"neural/{row['job']}/fit.json" for row in reg["neural_tasks"]}
    if set(barrier["frozen_fit_sha256"]) != required:
        raise ValueError("incomplete global fit manifest")
    for relative, expected in barrier["frozen_fit_sha256"].items():
        frozen = out / relative
        if not frozen.exists() or _sha(frozen) != expected:
            raise ValueError("global fit missing or changed after Test barrier")
    relative = str((directory(cfg, task) / "fit.json").relative_to(out))
    if relative not in required:
        raise ValueError("task is outside registered Test manifest")
    check_fit(cfg, task); return barrier


def evaluate_task(config_path, task):
    cfg = load_config(config_path); barrier = test_access(cfg, task); reg, pcfg = verify(cfg)
    dest = directory(cfg, task); existing = dest / "evaluation.json"
    if existing.exists():
        value = json.loads(existing.read_text())
        if (value["barrier_sha256"] != _sha(Path(cfg["output_dir"]) / "test_barrier.json")
                or value["prediction_sha256"] != _sha(dest / "test_predictions.parquet")):
            raise ValueError("existing Test prediction provenance changed")
        return task
    worker_setup(cfg["training"]["cpu_threads"])
    arrays, data = parent.arrays_for(pcfg), parent.fold_for(pcfg, task["fold"])
    fold = fold_info(reg, task["fold"])
    checkpoint = torch.load(dest / "research_model.pt", map_location="cpu", weights_only=False)
    if checkpoint["registration_sha256"] != barrier["registration_sha256"] or checkpoint["task"] != task:
        raise ValueError("checkpoint registration/task mismatch")
    model = model_for(task, cfg, checkpoint["scales"]).to(cfg["device"]); model.load_state_dict(checkpoint["state_dict"])
    result = predict(model, arrays, data["test"], data, cfg)
    parquet(prediction_frame(arrays, data["test"], result, task, "test", fold["label_contract"]), dest / "test_predictions.parquet")
    _json(existing, {"barrier_sha256": _sha(Path(cfg["output_dir"]) / "test_barrier.json"),
        "prediction_sha256": _sha(dest / "test_predictions.parquet"), "utc": now()})
    return task


def run_tasks(cfg, tasks, worker, stage):
    """Only active slots are submitted; no prefetched queue beyond the limit."""
    context = mp.get_context("spawn"); pending = iter(tasks)
    with ProcessPoolExecutor(cfg["training"]["workers"], mp_context=context, max_tasks_per_child=1) as pool:
        running = {}
        def submit():
            task = next(pending, None)
            if task is not None: running[pool.submit(worker, cfg["config_path"], task)] = task
        for _ in range(cfg["training"]["workers"]): submit()
        while running:
            done, _ = wait(running, timeout=20, return_when=FIRST_COMPLETED)
            if not done: status(cfg, stage)
            for future in done:
                task = running.pop(future)
                try: future.result()
                except Exception:
                    for other in running: other.cancel()
                    raise
                print(json.dumps({"completed": stage, **task}), flush=True); status(cfg, stage); submit()


def run_fits(cfg):
    reg, _ = verify(cfg); remaining = []
    for task in reg["neural_tasks"]:
        if (directory(cfg, task) / "fit.json").exists(): check_fit(cfg, task)
        elif (Path(cfg["output_dir"]) / "test_barrier.json").exists():
            raise ValueError("a missing fit cannot be retrained after the global Test barrier")
        else: remaining.append(task)
    run_tasks(cfg, remaining, train_neural, "training")


def campaign(cfg):
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    with (out / "campaign.lock").open("a") as lock:
        try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("campaign coordinator already running") from None
        try:
            verify(cfg, full=True); status(cfg, "training"); run_fits(cfg)
            status(cfg, "freezing_validation"); freeze_validation_choices(cfg); open_test_barrier(cfg)
            status(cfg, "test_inference"); reg, _ = verify(cfg)
            run_tasks(cfg, reg["neural_tasks"], evaluate_task, "test_inference")
            from .wm_return_multitask_report import write_multitask_report
            from .wm_return_multitask_statistics import write_multitask_statistics
            status(cfg, "final_report")
            report = write_multitask_report(prediction_files(cfg, reg, "test"), out / "final", cfg)
            status(cfg, "block_bootstrap_statistics"); inference = write_multitask_statistics(out, cfg)
            _json(out / "completion.json", {"complete": True, "status": "complete", "utc": now(),
                "registration_sha256": _sha(out / "preregistration.json"), "report": report, "statistics": inference})
            status(cfg, "complete")
        except Exception as exc:
            _json(out / "failure.json", {"utc": now(), "error": repr(exc), "traceback": traceback.format_exc()})
            status(cfg, "failed"); raise


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument("command", choices=("prepare", "run", "status"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(); cfg = load_config(args.config)
    if args.command == "prepare": prepare(cfg)
    elif args.command == "run": campaign(cfg)
    else: status(cfg)


if __name__ == "__main__": main()
