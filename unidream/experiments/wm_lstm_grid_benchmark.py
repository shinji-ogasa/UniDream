"""Causal LSTM extension:4 inputs x7 targets x3 formulations =84 cells.

All180 fold fits and validation choices freeze before any LSTM Test inference.
The frozen original336-cell study and its ongoing significance analysis are
read-only dependencies, never expanded or rewritten by this campaign.
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
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import torch
import yaml

from . import wm_full_grid_benchmark as full_grid
from . import wm_joint_grid_benchmark as joint_grid
from . import wm_seven_task_benchmark as parent
from .wm_seven_task_evaluation import TASKS
from .wm_technical_ablation import ROOT, _input


DEFAULT_CONFIG = ROOT / "configs/experiments/wm_lstm_grid_20261112.yaml"
FORMULATIONS = ("regression", "classification", "joint")
TRAINING_KEYS = ("max_steps", "min_steps", "patience_steps", "relative_improvement", "validation_every",
    "selection_rows", "batch_size", "evaluation_batch_size", "learning_rate", "weight_decay", "gradient_clip")


def now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg.get("groups") != list("ABCD") or cfg.get("models") != ["lstm"]
            or cfg.get("folds") != list(range(15)) or cfg.get("seeds") != [7]
            or cfg.get("formulations") != list(FORMULATIONS) or cfg.get("horizons") != [16]
            or cfg.get("test_selection_allowed") is not False or cfg.get("historical_seen_test") is not True
            or cfg.get("calibration") != "identity"
            or cfg.get("validation_partition") != "calendar_months_earlystop_calibration_selection"
            or cfg.get("joint_loss_weights") != {"regression": .5, "classification": .5}
            or cfg.get("model", {}).get("context") != 256
            or cfg.get("model", {}).get("lstm_hidden_size") != 256
            or cfg.get("model", {}).get("lstm_layers") != 2
            or cfg.get("model", {}).get("bidirectional") is not False
            or cfg.get("model", {}).get("carry_state_across_windows") is not False):
        raise ValueError("fixed84-cell causal LSTM, Train labels and report-only historical Test contract required")
    if cfg["device"] not in ("mps", "cpu") or cfg["training"]["workers"] < 1 or cfg["training"]["cpu_threads"] < 1:
        raise ValueError("explicit supported device and positive workers/threads required")
    cfg["config_path"] = str(Path(path).resolve())
    for key in ("output_dir", "parent_config", "joint_config", "point_table_path"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def source_hashes():
    return {str(path.relative_to(ROOT)): _sha(path) for path in sorted(Path(__file__).parent.glob("wm_lstm_grid*.py"))}


def jobs(cfg):
    return [{"fold": fold, "group": group, "seed": 7, "model": "lstm", "formulation": form,
        "job": f"fold_{fold:02d}_{group}_lstm_{form}_s7"}
        for fold in cfg["folds"] for group in ("D", "A", "B", "C") for form in FORMULATIONS]


def directory(cfg, task):
    return Path(cfg["output_dir"]) / "neural" / task["job"]


def model_for(task, cfg, scales):
    from .wm_lstm_grid_models import model_for as build
    return build(task, cfg, scales)


def prepare(cfg):
    root = Path(cfg["output_dir"])
    if (root / "preregistration.json").exists():
        raise ValueError("LSTM registration already frozen")
    fcfg = full_grid.load_config(cfg["parent_config"])
    freg, _ = full_grid.verify(fcfg, full=True)
    jcfg = joint_grid.load_config(cfg["joint_config"])
    _, from_joint, _ = joint_grid.verify(jcfg, full=True)
    if from_joint["config_path"] != fcfg["config_path"]:
        raise ValueError("LSTM controls and original336 report must share the same full-grid parent")
    if any(cfg["training"][key] != fcfg["training"][key] for key in TRAINING_KEYS):
        raise ValueError("LSTM and frozen controls must have the same optimization/early-stop budget")
    for key in ("context", "d_model", "encoder_hidden", "dropout"):
        if cfg["model"][key] != fcfg["model"][key]:
            raise ValueError("LSTM input encoder, context and hidden head must match the controls")
    point = Path(cfg["point_table_path"])
    if point != Path(jcfg["output_dir"]) / "final/results_336.csv":
        raise ValueError("the frozen336 point table is required")
    if not json.loads((Path(jcfg["output_dir"]) / "completion.json").read_text()).get("complete"):
        raise ValueError("the original336 campaign must be complete")
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    for fold in cfg["folds"]:
        name = f"fold_{fold:02d}_labels.npz"
        shutil.copyfile(Path(fcfg["output_dir"]) / "prepared" / name, prepared / name)
        if _sha(prepared / name) != freg["prepared_sha256"][name]:
            raise ValueError("frozen Train label array changed while copying")
    metadata = {}
    for form in FORMULATIONS:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = model_for({"formulation": form, "model": "lstm"}, cfg, freg["folds"][0]["scales"])
            metadata[form] = {"parameters": sum(p.numel() for p in model.parameters()), "metadata": model.checkpoint_metadata()}
    registration = {"experiment_id": cfg["experiment_id"], "registered_utc": now(), "config": cfg,
        "config_sha256": _sha(cfg["config_path"]), "source_sha256": source_hashes(),
        "parent_registration_sha256": _sha(Path(fcfg["output_dir"]) / "preregistration.json"),
        "joint_registration_sha256": _sha(Path(jcfg["output_dir"]) / "preregistration.json"),
        "parent_results_336_sha256": _sha(point), "prepared_sha256": {p.name: _sha(p) for p in sorted(prepared.glob("*.npz"))},
        "folds": freg["folds"], "neural_tasks": jobs(cfg), "classical_tasks": [],
        "grid": {"cells": 84, "inputs": 4, "models": ["lstm"], "targets": 7, "formulations": list(FORMULATIONS),
            "fresh_fits": 180, "combined_cells": 420, "training_seeds": [7]},
        "architecture": "unidirectional two-layer LSTM; no bidirectional attention or state carry between windows; same causal63 features and256-token contexts",
        "model_capacity": metadata, "capacity_policy": "dimensions fixed before Test; report actual parameter counts, not exact capacity-matching or hyperparameter search",
        "supervision": "same seven-target frozen regression/CE-BCE objectives and weights; joint .5 regression+.5 classification; no WM auxiliary loss",
        "selection": "same fixed768 early-stop Val panel and3000-update maximum; all180 fits then third-Val-month report/selection choices freeze before any LSTM Test inference",
        "cohort": "same15folds, Train2years/Val3months/Test3months and conservative16h label-boundary purge inherited from original study",
        "historical_seen_test": True, "independently_unused_holdout": False, "test_selection_allowed": False,
        "significance": "original1056-hypothesis protocol remains unchanged; this84-cell extension is descriptive and needs a separately registered inference extension",
        "parent_336_modified": False, "warm_start": False, "environment": {"torch": torch.__version__, "numpy": np.__version__, "device": cfg["device"]}}
    atomic_json(root / "preregistration.json", registration)
    status(cfg, "registered")
    return registration


def verify(cfg, full=False):
    root = Path(cfg["output_dir"])
    reg = json.loads((root / "preregistration.json").read_text())
    if reg["config_sha256"] != _sha(cfg["config_path"]) or reg["source_sha256"] != source_hashes():
        raise ValueError("registered LSTM source/config changed")
    if reg.get("neural_tasks") != jobs(cfg) or reg.get("classical_tasks") != []:
        raise ValueError("the complete180-fit LSTM grid must remain frozen")
    fcfg = full_grid.load_config(cfg["parent_config"])
    _, pcfg = full_grid.verify(fcfg, full=full)
    jcfg = joint_grid.load_config(cfg["joint_config"])
    joint_grid.verify(jcfg)
    if (reg["parent_registration_sha256"] != _sha(Path(fcfg["output_dir"]) / "preregistration.json")
            or reg["joint_registration_sha256"] != _sha(Path(jcfg["output_dir"]) / "preregistration.json")
            or reg["parent_results_336_sha256"] != _sha(cfg["point_table_path"])):
        raise ValueError("frozen original control registration/result table changed")
    if full:
        for name, expected in reg["prepared_sha256"].items():
            if _sha(root / "prepared" / name) != expected:
                raise ValueError("copied Train label data changed")
    return reg, fcfg, pcfg


@contextmanager
def claim(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("LSTM task/coordinator is already claimed") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def predict(model, arrays, ids, data, task, cfg):
    fn = joint_grid.predict if task["formulation"] == "joint" else full_grid.predict
    return fn(model, arrays, ids, data, task, cfg)


def prediction_frame(arrays, ids, prediction, task, split, contract):
    fn = joint_grid.prediction_frame if task["formulation"] == "joint" else full_grid.prediction_frame
    return fn(arrays, ids, prediction, task, split, contract)


def supervised_objective(actual, prediction, task, fold):
    if task["formulation"] == "joint":
        return joint_grid.supervised_objective(actual, prediction, fold["label_contract"], fold["scales"])
    return full_grid.supervised_objective(actual, prediction, task["formulation"], fold["label_contract"], fold["scales"])


def active_outputs(task):
    return ({mode + "_" + target for mode in ("regression", "classification") for target in TASKS}
        if task["formulation"] == "joint" else set(TASKS))


def check_fit(cfg, task):
    dest = directory(cfg, task)
    fit = json.loads((dest / "fit.json").read_text())
    if any(fit.get(key) != value for key, value in task.items()) or fit.get("registration_sha256") != _sha(Path(cfg["output_dir"]) / "preregistration.json"):
        raise ValueError("LSTM fit task/registration changed")
    for name, key in (("research_model.pt", "checkpoint_sha256"), ("validation_predictions.parquet", "validation_predictions_sha256")):
        if _sha(dest / name) != fit[key]:
            raise ValueError("LSTM selected checkpoint or validation predictions changed")
    if (fit.get("best_step", 0) <= 0 or fit.get("target_counts", 0) <= 0
            or set(fit.get("supervised_gradient_steps", {})) != active_outputs(task)
            or not all(fit["supervised_gradient_steps"].values())):
        raise ValueError("LSTM fit lacks checkpoint/label/active-head gradient coverage")
    return fit


def train_neural(config_path, task):
    cfg = load_config(config_path)
    reg, _, pcfg = verify(cfg)
    dest = directory(cfg, task)
    if (dest / "fit.json").exists():
        check_fit(cfg, task)
        return task
    if (Path(cfg["output_dir"]) / "test_barrier.json").exists():
        raise ValueError("a missing LSTM fit cannot be trained after the global Test barrier")
    with claim(dest / "task.lock"), (dest / "training.log").open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        if (dest / "fit.json").exists():
            check_fit(cfg, task)
            return task
        full_grid.worker_setup(cfg["training"]["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        random.seed(task["seed"])
        np.random.seed(task["seed"])
        torch.manual_seed(task["seed"])
        arrays, data = parent.arrays_for(pcfg), parent.fold_for(pcfg, task["fold"])
        fold = full_grid.fold_info(reg, task["fold"])
        with np.load(Path(cfg["output_dir"]) / "prepared" / f"fold_{task['fold']:02d}_labels.npz") as values:
            labels = {target: values[target] for target in TASKS}
        model = model_for(task, cfg, fold["scales"]).to(cfg["device"])
        train = cfg["training"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=train["learning_rate"], weight_decay=train["weight_decay"])
        rng = np.random.default_rng(task["seed"])
        best = anchor = np.inf
        best_step = last_significant = counts = 0
        state, history = None, []
        coverage = dict.fromkeys(sorted(active_outputs(task)), 0)
        started, reason = time.monotonic(), "max_steps"
        for step in range(1, train["max_steps"] + 1):
            model.train()
            ids = rng.choice(data["train"], train["batch_size"], replace=True)
            x, base = _input(arrays, ids, data, task["group"], cfg["model"]["context"])
            positions = ids[:, None] - cfg["model"]["context"] + np.arange(1, cfg["model"]["context"] + 1)[None, :]
            valid = full_grid.training_mask(arrays, positions, fold)
            counts += int(valid.sum())
            continuous = {t: torch.from_numpy(arrays[t][positions].astype(np.float32)).to(cfg["device"]) for t in TASKS} if task["formulation"] != "classification" else None
            discrete = {t: torch.from_numpy(labels[t][positions].astype(np.int64)).to(cfg["device"]) for t in TASKS} if task["formulation"] != "regression" else None
            targets = {"regression": continuous, "classification": discrete} if task["formulation"] == "joint" else continuous if task["formulation"] == "regression" else discrete
            next_return = np.log(arrays["raw"][positions, 3] / arrays["raw"][positions - 1, 3]).astype(np.float32)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model.loss(torch.from_numpy(x).to(cfg["device"]), torch.from_numpy(base).to(cfg["device"]),
                targets, torch.from_numpy(valid).to(cfg["device"]), torch.from_numpy(next_return).to(cfg["device"]))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite LSTM training loss")
            loss.backward()
            covered = model.supervised_gradient_coverage()
            if set(covered) != set(coverage):
                raise RuntimeError("LSTM active-head gradient contract changed")
            for key, value in covered.items():
                coverage[key] += int(value)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            if step % train["validation_every"] == 0:
                panel = predict(model, arrays, data["selection_panel"], data, task, cfg)
                score = supervised_objective({t: arrays[t][data["selection_panel"]] for t in TASKS}, panel, task, fold)
                history.append({"step": step, "selection": score, "batch_total_loss": float(loss.detach().cpu()),
                    "batch_loss_parts": {key: float(value.detach().cpu()) for key, value in parts.items()},
                    "elapsed_seconds": time.monotonic() - started})
                print(json.dumps(history[-1]), flush=True)
                if score["score"] < best:
                    best, best_step = score["score"], step
                    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                if not np.isfinite(anchor) or score["score"] < anchor - train["relative_improvement"] * abs(anchor):
                    anchor, last_significant = score["score"], step
                atomic_json(dest / "progress.json", {**task, "step": step, "best_step": best_step,
                    "best_score": best, "elapsed_seconds": time.monotonic() - started})
                if step >= train["min_steps"] and step - last_significant >= train["patience_steps"]:
                    reason = "validation_patience"
                    break
        if state is None or not counts or not all(coverage.values()):
            raise RuntimeError("LSTM fit lacks checkpoint, valid labels or gradients")
        model.load_state_dict(state)
        checkpoint = {"state_dict": state, "scales": fold["scales"], "task": task,
            "registration_sha256": _sha(Path(cfg["output_dir"]) / "preregistration.json"), "metadata": model.checkpoint_metadata()}
        temporary = dest / "research_model.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(dest / "research_model.pt")
        frames = [prediction_frame(arrays, data[split], predict(model, arrays, data[split], data, task, cfg),
            task, split, fold["label_contract"]) for split in ("earlystop", "calibration", "selection")]
        full_grid.parquet(pd.concat(frames, ignore_index=True), dest / "validation_predictions.parquet")
        atomic_json(dest / "fit.json", {**task, "registration_sha256": _sha(Path(cfg["output_dir"]) / "preregistration.json"),
            "elapsed_seconds": time.monotonic() - started, "steps": step, "best_step": best_step, "best_score": best,
            "stop_reason": reason, "max_step_censored": reason == "max_steps", "history": history,
            "target_counts": counts, "supervised_gradient_steps": coverage, "metadata": model.checkpoint_metadata(),
            "parameters": sum(p.numel() for p in model.parameters()), "checkpoint_sha256": _sha(dest / "research_model.pt"),
            "validation_predictions_sha256": _sha(dest / "validation_predictions.parquet")})
    return task


def frozen_fit_manifest(cfg, reg):
    return {str((directory(cfg, task) / "fit.json").relative_to(Path(cfg["output_dir"]))): _sha(directory(cfg, task) / "fit.json")
        for task in reg["neural_tasks"] if check_fit(cfg, task)}


def freeze_validation_choices(cfg):
    from .wm_lstm_grid_report import write_lstm_grid_report
    reg, _, _ = verify(cfg)
    manifest = frozen_fit_manifest(cfg, reg)
    root = Path(cfg["output_dir"])
    path = root / "validation_choices.json"
    selection = root / "selection/results_84.csv"
    if path.exists():
        choices = json.loads(path.read_text())
        if (choices.get("registration_sha256") != _sha(root / "preregistration.json")
                or choices.get("frozen_fit_sha256") != manifest or choices.get("test_used") is not False
                or choices.get("selection_report_sha256") != _sha(selection)):
            raise ValueError("frozen LSTM validation choices changed")
        return choices
    report = write_lstm_grid_report(root, cfg, include_test=False)
    table = pd.read_csv(selection)
    choices = {"frozen_utc": now(), "registration_sha256": _sha(root / "preregistration.json"),
        "frozen_fit_sha256": manifest, "test_used": False, "calibration": "identity",
        "selection_report_sha256": _sha(selection), "all_selection_rows": json.loads(table.to_json(orient="records")),
        "rule": "fixed target/formulation comparisons; first-Val-month own-loss checkpoint; third-Val-month descriptive report; no Test adaptation",
        "report": report}
    atomic_json(path, choices)
    return choices


def open_test_barrier(cfg):
    reg, _, _ = verify(cfg)
    manifest = frozen_fit_manifest(cfg, reg)
    root = Path(cfg["output_dir"])
    choices = json.loads((root / "validation_choices.json").read_text())
    expected = {"registration_sha256": _sha(root / "preregistration.json"), "frozen_fit_sha256": manifest,
        "validation_choices_sha256": _sha(root / "validation_choices.json"),
        "parent_results_336_sha256": reg["parent_results_336_sha256"], "test_selection_allowed": False}
    if (choices.get("registration_sha256") != expected["registration_sha256"]
            or choices.get("frozen_fit_sha256") != manifest or choices.get("test_used") is not False
            or choices.get("selection_report_sha256") != _sha(root / "selection/results_84.csv")):
        raise ValueError("all180 LSTM fits and unchanged Val choices must freeze before Test")
    path = root / "test_barrier.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if any(existing.get(key) != value for key, value in expected.items()):
            raise ValueError("LSTM Test barrier cannot be reopened with changed choices/fits")
        return existing
    barrier = {"opened_utc": now(), **expected}
    atomic_json(path, barrier)
    return barrier


def test_access(cfg, task):
    reg, _, _ = verify(cfg)
    root = Path(cfg["output_dir"])
    path = root / "test_barrier.json"
    if not path.exists():
        raise ValueError("LSTM global Test barrier is closed")
    barrier = json.loads(path.read_text())
    required = {str((directory(cfg, job) / "fit.json").relative_to(root)) for job in reg["neural_tasks"]}
    if (barrier.get("registration_sha256") != _sha(root / "preregistration.json")
            or barrier.get("validation_choices_sha256") != _sha(root / "validation_choices.json")
            or set(barrier.get("frozen_fit_sha256", {})) != required
            or barrier.get("parent_results_336_sha256") != _sha(cfg["point_table_path"])):
        raise ValueError("LSTM global frozen-fit/Val/parent barrier changed")
    for relative, expected in barrier["frozen_fit_sha256"].items():
        if _sha(root / relative) != expected:
            raise ValueError("LSTM fit changed after Test barrier")
    check_fit(cfg, task)
    return barrier


def evaluate_task(config_path, task):
    cfg = load_config(config_path)
    test_access(cfg, task)
    reg, _, pcfg = verify(cfg)
    dest = directory(cfg, task)
    prediction_path = dest / "test_predictions.parquet"
    evaluation_path = dest / "evaluation.json"
    if evaluation_path.exists():
        existing = json.loads(evaluation_path.read_text())
        if (existing.get("barrier_sha256") != _sha(Path(cfg["output_dir"]) / "test_barrier.json")
                or existing.get("prediction_sha256") != _sha(prediction_path)):
            raise ValueError("frozen LSTM Test predictions changed")
        return task
    with claim(dest / "evaluation.lock"):
        full_grid.worker_setup(cfg["training"]["cpu_threads"])
        state = torch.load(dest / "research_model.pt", map_location="cpu", weights_only=False)
        if state["task"] != task or state["registration_sha256"] != _sha(Path(cfg["output_dir"]) / "preregistration.json"):
            raise ValueError("LSTM research checkpoint belongs to a different condition")
        model = model_for(task, cfg, state["scales"]).to(cfg["device"])
        model.load_state_dict(state["state_dict"], strict=True)
        arrays, data = parent.arrays_for(pcfg), parent.fold_for(pcfg, task["fold"])
        fold = full_grid.fold_info(reg, task["fold"])
        frame = prediction_frame(arrays, data["test"], predict(model, arrays, data["test"], data, task, cfg), task, "test", fold["label_contract"])
        full_grid.parquet(frame, prediction_path)
        atomic_json(evaluation_path, {"barrier_sha256": _sha(Path(cfg["output_dir"]) / "test_barrier.json"),
            "prediction_sha256": _sha(prediction_path), "utc": now()})
    return task


def status(cfg, stage=None):
    root = Path(cfg["output_dir"])
    reg = json.loads((root / "preregistration.json").read_text())
    previous = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
    tasks = reg["neural_tasks"]
    done = [task for task in tasks if (directory(cfg, task) / "fit.json").exists()]
    result = {"utc": now(), "stage": stage or previous.get("stage", "registered"), "cells": 84,
        "neural": {"complete": len(done), "total": len(tasks), "by_formulation": {
            form: sum(task["formulation"] == form for task in done) for form in FORMULATIONS}},
        "evaluated": sum((directory(cfg, task) / "evaluation.json").exists() for task in tasks),
        "test_opened": (root / "test_barrier.json").exists(), "historical_test_exploratory": True}
    atomic_json(root / "status.json", result)
    (root / "progress.html").write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta http-equiv="refresh" content="20">'
        '<title>LSTM84条件</title><style>body{font-family:system-ui;margin:30px;line-height:1.7}</style><h1>LSTM84条件</h1>'
        f'<p>{result["stage"]}: 学習{len(done)}/{len(tasks)} fit、Test評価{result["evaluated"]}/{len(tasks)}</p>'
        '<p>4入力×7対象×回帰・分類・同時学習。全180fold-fitとVal選択を固定してからTestを開きます。</p>'
        '<p>既存336条件と統計検証は変更しません。この追加実験のTestも探索的OOSです。</p>'
        '<p><a href="final/index.html">LSTM84結果</a> / <a href="final/index420.html">全420条件</a></p></html>', encoding="utf-8")
    return result


def run_tasks(cfg, tasks, worker, stage):
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(cfg["training"]["workers"], mp_context=context, max_tasks_per_child=1) as pool:
        futures = {pool.submit(worker, cfg["config_path"], task): task for task in tasks}
        while futures:
            completed, _ = wait(futures, timeout=20, return_when=FIRST_COMPLETED)
            for future in completed:
                task = futures.pop(future)
                try:
                    future.result()
                except Exception:
                    for other in futures:
                        other.cancel()
                    raise
                print(json.dumps({"completed": stage, **task}), flush=True)
            status(cfg, stage)


def campaign(cfg):
    root = Path(cfg["output_dir"])
    with claim(root / "campaign.lock"):
        atomic_json(root / "coordinator.json", {"pid": os.getpid(), "started_utc": now(), "lock": "flock"})
        try:
            reg, _, _ = verify(cfg, full=True)
            if (root / "completion.json").exists():
                complete = json.loads((root / "completion.json").read_text())
                if complete.get("complete") and complete.get("registration_sha256") == _sha(root / "preregistration.json"):
                    for relative, expected in complete["artifact_sha256"].items():
                        if _sha(root / relative) != expected:
                            raise ValueError("completed LSTM artifact changed")
                    status(cfg, "complete")
                    return complete
                raise ValueError("stale LSTM completion marker")
            pending = []
            for task in reg["neural_tasks"]:
                if (directory(cfg, task) / "fit.json").exists():
                    check_fit(cfg, task)
                else:
                    pending.append(task)
            status(cfg, "training")
            run_tasks(cfg, pending, train_neural, "training")
            status(cfg, "freezing_validation")
            freeze_validation_choices(cfg)
            open_test_barrier(cfg)
            status(cfg, "test_inference")
            run_tasks(cfg, reg["neural_tasks"], evaluate_task, "test_inference")
            from .wm_lstm_grid_report import write_lstm_grid_report
            status(cfg, "reporting")
            report = write_lstm_grid_report(root, cfg, include_test=True)
            if report.get("cells", report.get("n_result_rows")) != 84 or report.get("combined_cells") != 420:
                raise ValueError("complete84 LSTM cells and420 combined cells required")
            paths = ("final/results_84.csv", "final/results_420.csv", "final/index.html", "final/index420.html", "final/report_completion.json")
            complete = {"complete": True, "status": "complete", "utc": now(), "cells": 84, "combined_cells": 420,
                "registration_sha256": _sha(root / "preregistration.json"), "report": report,
                "artifact_sha256": {path: _sha(root / path) for path in paths},
                "historical_seen_test": True, "independently_unused_holdout": False}
            atomic_json(root / "completion.json", complete)
            status(cfg, "complete")
            return complete
        except Exception as error:
            atomic_json(root / "failure.json", {"utc": now(), "error": repr(error), "traceback": traceback.format_exc()})
            if (root / "preregistration.json").exists():
                status(cfg, "failed")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "campaign", "status"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = {"prepare": prepare, "verify": verify, "campaign": campaign, "status": status}[args.command](cfg)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
