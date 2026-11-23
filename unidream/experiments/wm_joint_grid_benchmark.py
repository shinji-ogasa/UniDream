"""Complete the frozen 224-cell grid with 112 explicitly defined joint cells.

The two neural families share one causal representation for both output heads.
Classical joint controls are the exact pair of existing independent estimators:
they do not claim representation sharing or a new LightGBM training algorithm.
Every new fit and validation choice freezes before new Test inference. The final
336-cell export also waits for the parent 224-cell campaign to finish.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import fcntl
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
from . import wm_seven_task_benchmark as parent
from .wm_full_grid_evaluation import make_task_labels
from .wm_seven_task_evaluation import TASKS
from .wm_technical_ablation import ROOT, _input, _json, _sha

DEFAULT_CONFIG = ROOT / "configs/experiments/wm_joint_grid_20261111.yaml"
TRAINING_CONTRACT = ("max_steps", "min_steps", "patience_steps", "relative_improvement",
                     "validation_every", "selection_rows", "batch_size", "evaluation_batch_size",
                     "learning_rate", "weight_decay", "gradient_clip")


def now():
    return datetime.now(timezone.utc).isoformat()


def load_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg["groups"] != ["A", "B", "C", "D"] or cfg["folds"] != list(range(15))
            or cfg["seeds"] != [7] or cfg["models"] != ["direct", "wm"]
            or cfg["formulations"] != ["joint"]
            or cfg["complete_formulations"] != ["regression", "classification", "joint"]
            or cfg["horizons"] != [16] or cfg["test_selection_allowed"] is not False
            or cfg["calibration"] != "identity" or cfg["classical_joint"] != "reused_independent_heads"
            or cfg["joint_loss_weights"] != {"regression": .5, "classification": .5}
            or cfg["validation_partition"] != "calendar_months_earlystop_calibration_selection"):
        raise ValueError("fixed 336-cell scope, Train labels, joint loss and report-only Test required")
    if (cfg["training"]["workers"] < 1 or cfg["training"]["cpu_threads"] < 1
            or not 1 <= cfg["parent_poll_seconds"] <= 60):
        raise ValueError("positive workers/threads and 1-60 second parent polling required")
    cfg["config_path"] = str(Path(path).resolve())
    for key in ("output_dir", "parent_config"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def source_hashes():
    names = ("wm_joint_grid_benchmark.py", "wm_joint_grid_models.py",
             "wm_joint_grid_report.py", "wm_joint_grid_statistics.py")
    return {str((Path(__file__).parent / name).relative_to(ROOT)): _sha(Path(__file__).parent / name)
            for name in names}


def verify(cfg, full=False):
    out = Path(cfg["output_dir"])
    reg = json.loads((out / "preregistration.json").read_text())
    if reg["config_sha256"] != _sha(cfg["config_path"]) or reg["source_sha256"] != source_hashes():
        raise ValueError("registered joint source/config changed")
    fcfg = full_grid.load_config(cfg["parent_config"])
    _, pcfg = full_grid.verify(fcfg, full=full)
    if _sha(Path(fcfg["output_dir"]) / "preregistration.json") != reg["parent_registration_sha256"]:
        raise ValueError("immutable 224-cell parent registration changed")
    if full:
        for name, expected in reg["prepared_sha256"].items():
            if _sha(out / "prepared" / name) != expected:
                raise ValueError("copied immutable Train label arrays changed")
    return reg, fcfg, pcfg


def fold_info(reg, fold):
    return next(row for row in reg["folds"] if row["fold"] == fold)


def directory(cfg, kind, task):
    return Path(cfg["output_dir"]) / kind / task["job"]


def prepare(cfg):
    out = Path(cfg["output_dir"]); prepared = out / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    if (out / "preregistration.json").exists():
        raise ValueError("registration already frozen")
    fcfg = full_grid.load_config(cfg["parent_config"])
    freg, _ = full_grid.verify(fcfg, full=True)
    if (cfg["model"] != fcfg["model"] or cfg["device"] != fcfg["device"]
            or any(cfg["training"][key] != fcfg["training"][key] for key in TRAINING_CONTRACT)):
        raise ValueError("joint and frozen controls must use the same model, data and training budget")
    for fold in cfg["folds"]:
        name = f"fold_{fold:02d}_labels.npz"
        shutil.copyfile(Path(fcfg["output_dir"]) / "prepared" / name, prepared / name)
        if _sha(prepared / name) != freg["prepared_sha256"][name]:
            raise ValueError("parent Train label array changed during copying")
    jobs = [{"fold": f, "group": g, "seed": 7, "model": model, "formulation": "joint",
             "job": f"fold_{f:02d}_{g}_{model}_joint_s7"}
            for f in cfg["folds"] for g in ("D", "A", "B", "C") for model in cfg["models"]]
    classics = [{"fold": f, "group": g, "seed": 7, "job": f"fold_{f:02d}_{g}"}
                for f in cfg["folds"] for g in ("D", "A", "B", "C")]
    _json(out / "preregistration.json", {"experiment_id": cfg["experiment_id"], "registered_utc": now(),
        "config": cfg, "config_sha256": _sha(cfg["config_path"]), "source_sha256": source_hashes(),
        "parent_registration_sha256": _sha(Path(fcfg["output_dir"]) / "preregistration.json"),
        "prepared_sha256": {p.name: _sha(p) for p in sorted(prepared.glob("*.npz"))},
        "folds": freg["folds"], "neural_tasks": jobs, "classical_tasks": classics,
        "grid": {"inputs": 4, "families": 4, "targets": 7, "formulations": 3, "cells": 336,
                 "reused_regression_classification_cells": 224, "new_joint_cells": 112,
                 "new_shared_neural_joint_cells": 56, "independent_classical_joint_control_cells": 56,
                 "new_neural_fits": 120, "classical_reuse_bundles": 60, "neural_seeds": [7]},
        "condition_accounting": {"table_cells": 336, "distinct_empirical_conditions": 280,
            "n_fresh_joint_neural_cells": 56, "n_independent_joint_control_cells": 56,
            "n_reused_parent_empirical_cells": 224, "classical_controls_are_new_training": False},
        "loss_weights": parent.LOSS_WEIGHTS, "joint_loss_weights": cfg["joint_loss_weights"],
        "neural_joint": "one shared causal body/hidden head; scalar7+classification19 outputs; .5 frozen seven-target regression objective plus .5 frozen seven-target CE/BCE objective; WM auxiliaries unchanged",
        "classical_joint": "exact independent-head additive-objective control; reuse each frozen parent regression and classification estimator; shared_parameters=false; no newly trained shared LightGBM or linear representation",
        "selection": "first Val month fixed768-row panel selects own .5/.5 joint checkpoint with the same budget/patience; third Val month descriptive fixed comparisons; middle month reserved; identity calibration",
        "test_access": "all120 new neural fits,60 reused classical Val bundles and own Val choices freeze before new Test inference; parent224 completion required before Test artifacts are reused or merged; historical Test already inspected, retrospective OOS only",
        "primary_comparisons": "112 neural contrasts: joint minus regression native diagnostic and joint minus classification BA, across4inputs2backbones7targets; 10000-replicate paired contiguous-day block bootstrap,7dayprimary,1/28sensitivity,Holm112; classical112 equalities are deterministic controls",
        "regression_inference_metrics": {"return": "zero_oos_r2", "variance": "qlike", "downside": "mae",
            "crash": "brier", "dd_excess": "mae", "advantage": "rmse", "recovery": "rmse"},
        "limitations": ["All seven target pairs train jointly; this is not isolated per-target transfer.",
            "Joint output layer has26 outputs versus7/19 in frozen single-formulation controls; hidden/body initialization matches, extra output capacity is explicit.",
            "Linear/LightGBM joint cells are reused independent-head controls and cannot establish a multitask-learning benefit.",
            "Single seed does not establish initialization stability; retrospective Test is not pristine.",
            "Advantage/recovery report primary Spearman correlations; bootstrap RMSE is a distinct secondary diagnostic and does not imply rank-correlation confidence intervals.",
            "Different target/formulation metrics have different units; no overall336-cell ranking."]})
    return status(cfg, "registered")


def model_for(task, cfg, scales):
    from .wm_joint_grid_models import DirectJointGrid, WMJointGrid
    if task["model"] not in ("direct", "wm") or task["formulation"] != "joint":
        raise ValueError("registered shared neural joint task required")
    return (WMJointGrid if task["model"] == "wm" else DirectJointGrid)(63, cfg["model"], scales)


training_mask = full_grid.training_mask


def supervised_objective(actual, predictions, contract, scales):
    scores = {mode: full_grid.supervised_objective(actual, predictions[mode], mode, contract, scales)
              for mode in ("regression", "classification")}
    return {"score": .5 * sum(value["score"] for value in scores.values()), "parts": scores}


def predict(model, arrays, ids, data, task, cfg):
    model.eval(); parts = {mode: {t: [] for t in TASKS} for mode in ("regression", "classification")}
    with torch.no_grad():
        for start in range(0, len(ids), cfg["training"]["evaluation_batch_size"]):
            batch = ids[start:start + cfg["training"]["evaluation_batch_size"]]
            x, _ = _input(arrays, batch, data, task["group"], cfg["model"]["context"])
            values = model.predict_last(torch.from_numpy(x).to(cfg["device"]))
            for mode, targets in parts.items():
                for target in targets:
                    targets[target].append(values[mode][target].cpu().numpy())
    result = {mode: {target: np.concatenate(value) for target, value in targets.items()}
              for mode, targets in parts.items()}
    if any(not np.isfinite(value).all() for targets in result.values() for value in targets.values()):
        raise ValueError("nonfinite joint forecast")
    return result


def prediction_frame(arrays, ids, predictions, task, split, contract):
    ids = np.asarray(ids, dtype=np.int64)
    result = full_grid.prediction_frame(arrays, ids, predictions["regression"],
        {**task, "formulation": "regression"}, split, contract)
    classes = full_grid.prediction_frame(arrays, ids, predictions["classification"],
        {**task, "formulation": "classification"}, split, contract)
    for target in TASKS:
        scalar = result["scalar_" + target].to_numpy()
        if not np.isfinite(scalar).all(): raise ValueError("nonfinite joint scalar output")
        p = np.asarray(predictions["classification"][target]).reshape(len(ids), contract[target]["n_classes"])
        if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
                or not np.allclose(p.sum(1), 1, atol=1e-6)):
            raise ValueError("invalid joint class probabilities")
        for c in range(3): result[f"p_{target}_{c}"] = classes[f"p_{target}_{c}"]
    result["formulation"] = "joint"; result["shared_parameters"] = True
    result["implementation"] = "shared_neural_joint"
    return result


parquet = full_grid.parquet


def worker_setup(threads):
    os.nice(5); torch.set_num_threads(threads); torch.set_num_interop_threads(1)


@contextmanager
def task_claim(dest):
    """Kernel lock releases on a dead worker; a live task cannot be duplicated."""
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "training.lock").open("a+") as lock:
        try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("task already claimed by a live worker") from None
        lock.seek(0); lock.truncate(); lock.write(json.dumps({"pid": os.getpid(), "utc": now()})); lock.flush()
        yield


def train_neural(config_path, task):
    cfg = load_config(config_path); reg, _, pcfg = verify(cfg)
    dest = directory(cfg, "neural", task)
    if (dest / "fit.json").exists(): check_fit(cfg, "neural", task); return task
    if (Path(cfg["output_dir"]) / "test_barrier.json").exists():
        raise ValueError("a missing fit cannot be retrained after the global Test barrier")
    with task_claim(dest), (dest / "training.log").open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        if (dest / "fit.json").exists(): check_fit(cfg, "neural", task); return task
        worker_setup(cfg["training"]["cpu_threads"]); torch.use_deterministic_algorithms(True)
        random.seed(task["seed"]); np.random.seed(task["seed"]); torch.manual_seed(task["seed"])
        arrays, data = parent.arrays_for(pcfg), parent.fold_for(pcfg, task["fold"])
        fold = fold_info(reg, task["fold"])
        with np.load(Path(cfg["output_dir"]) / "prepared" / f"fold_{task['fold']:02d}_labels.npz") as z:
            labels = {t: z[t] for t in TASKS}
        model = model_for(task, cfg, fold["scales"]).to(cfg["device"]); train = cfg["training"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=train["learning_rate"], weight_decay=train["weight_decay"])
        rng = np.random.default_rng(task["seed"]); best = anchor = np.inf
        best_step = last_significant = counts = 0; state = None; history = []
        coverage = {f"{mode}_{t}": 0 for mode in ("regression", "classification") for t in TASKS}
        reason = "max_steps"; started = time.monotonic()
        for step in range(1, train["max_steps"] + 1):
            model.train(); ids = rng.choice(data["train"], train["batch_size"], replace=True)
            x, base = _input(arrays, ids, data, task["group"], cfg["model"]["context"])
            positions = ids[:, None] - cfg["model"]["context"] + np.arange(1, cfg["model"]["context"] + 1)[None, :]
            valid = training_mask(arrays, positions, fold); counts += int(valid.sum())
            targets = {"regression": {t: torch.from_numpy(arrays[t][positions].astype(np.float32)).to(cfg["device"]) for t in TASKS},
                       "classification": {t: torch.from_numpy(labels[t][positions].astype(np.int64)).to(cfg["device"]) for t in TASKS}}
            next_return = np.log(arrays["raw"][positions, 3] / arrays["raw"][positions - 1, 3]).astype(np.float32)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model.loss(torch.from_numpy(x).to(cfg["device"]), torch.from_numpy(base).to(cfg["device"]),
                targets, torch.from_numpy(valid).to(cfg["device"]), torch.from_numpy(next_return).to(cfg["device"]))
            if not bool(torch.isfinite(loss)): raise RuntimeError("nonfinite joint training loss")
            loss.backward(); covered = model.supervised_gradient_coverage()
            if set(covered) != set(coverage): raise RuntimeError("joint active-head gradient contract changed")
            for key, value in covered.items(): coverage[key] += int(value)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            if step % train["validation_every"] == 0:
                panel = predict(model, arrays, data["selection_panel"], data, task, cfg)
                score = supervised_objective({t: arrays[t][data["selection_panel"]] for t in TASKS}, panel,
                    fold["label_contract"], fold["scales"])
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
            raise RuntimeError("missing joint checkpoint/label/gradient coverage")
        model.load_state_dict(state)
        torch.save({"state_dict": state, "scales": fold["scales"], "task": task,
            "registration_sha256": _sha(Path(cfg["output_dir"]) / "preregistration.json")}, dest / "research_model.pt")
        frames = [prediction_frame(arrays, data[split], predict(model, arrays, data[split], data, task, cfg),
            task, split, fold["label_contract"]) for split in ("earlystop", "calibration", "selection")]
        parquet(pd.concat(frames, ignore_index=True), dest / "validation_predictions.parquet")
        _json(dest / "fit.json", {**task, "registration_sha256": _sha(Path(cfg["output_dir"]) / "preregistration.json"),
            "elapsed_seconds": time.monotonic() - started, "steps": step, "best_step": best_step, "best_score": best,
            "stop_reason": reason, "max_step_censored": reason == "max_steps", "history": history,
            "target_counts": counts, "supervised_gradient_steps": coverage, "metadata": model.checkpoint_metadata(),
            "shared_parameters": True, "implementation": "shared_neural_joint",
            "parameters": sum(p.numel() for p in model.parameters()), "checkpoint_sha256": _sha(dest / "research_model.pt"),
            "validation_predictions_sha256": _sha(dest / "validation_predictions.parquet")})
    return task


def combine_classical_heads(frame):
    """An exact control, never a claim of shared-parameter classical training."""
    frames = []
    for model in ("linear", "lightgbm"):
        reg = frame.loc[(frame["model"] == model) & (frame["formulation"] == "regression")].sort_values(["split", "decision_id"]).reset_index(drop=True)
        cls = frame.loc[(frame["model"] == model) & (frame["formulation"] == "classification")].sort_values(["split", "decision_id"]).reset_index(drop=True)
        if reg.empty or reg[["split", "decision_id"]].duplicated().any() or not reg[["split", "decision_id"]].equals(cls[["split", "decision_id"]]):
            raise ValueError("classical regression/classification cohorts differ")
        for key in ("fold", "group", "seed", "timestamp", *("actual_" + t for t in TASKS), *("label_" + t for t in TASKS)):
            if not np.array_equal(reg[key], cls[key]): raise ValueError("classical paired outcomes or identity differ")
        for task in TASKS:
            if not np.isfinite(reg["scalar_" + task]).all(): raise ValueError("nonfinite reused classical scalar")
            n_classes = 2 if task == "crash" else 3
            p = cls[[f"p_{task}_{c}" for c in range(n_classes)]].to_numpy()
            if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-6)):
                raise ValueError("invalid reused classical probabilities")
            for c in range(3): reg[f"p_{task}_{c}"] = cls[f"p_{task}_{c}"]
        reg["formulation"] = "joint"; reg["shared_parameters"] = False
        reg["implementation"] = "reused_independent_heads"; frames.append(reg)
    return pd.concat(frames, ignore_index=True)


def build_classical_control(cfg, task):
    reg, fcfg, _ = verify(cfg); dest = directory(cfg, "classical", task)
    if (dest / "fit.json").exists(): check_fit(cfg, "classical", task); return task
    if (Path(cfg["output_dir"]) / "test_barrier.json").exists():
        raise ValueError("missing classical control cannot be reconstructed after Test opens")
    source = full_grid.directory(fcfg, "classical", task); full_grid.check_fit(fcfg, "classical", task)
    with task_claim(dest):
        if (dest / "fit.json").exists(): check_fit(cfg, "classical", task); return task
        frame = pd.read_parquet(source / "validation_predictions.parquet")
        if set(frame["split"]) != {"earlystop", "calibration", "selection"}:
            raise ValueError("only three frozen Val partitions are reusable before Test")
        parquet(combine_classical_heads(frame), dest / "validation_predictions.parquet")
        _json(dest / "fit.json", {**task, "registration_sha256": _sha(Path(cfg["output_dir"]) / "preregistration.json"),
            "reused_parent_fit_sha256": _sha(source / "fit.json"), "shared_parameters": False,
            "implementation": "reused_independent_heads", "new_training": False,
            "validation_predictions_sha256": _sha(dest / "validation_predictions.parquet")})
    return task


def check_fit(cfg, kind, task):
    dest = directory(cfg, kind, task); fit = json.loads((dest / "fit.json").read_text())
    if fit["registration_sha256"] != _sha(Path(cfg["output_dir"]) / "preregistration.json"):
        raise ValueError("joint fit registration mismatch")
    if any(fit.get(key) != value for key, value in task.items()): raise ValueError("joint fit task identity changed")
    if _sha(dest / "validation_predictions.parquet") != fit["validation_predictions_sha256"]:
        raise ValueError("selected joint Val predictions changed")
    if kind == "neural":
        if _sha(dest / "research_model.pt") != fit["checkpoint_sha256"]: raise ValueError("selected joint checkpoint changed")
    elif kind == "classical":
        fcfg = full_grid.load_config(cfg["parent_config"])
        source = full_grid.directory(fcfg, "classical", task)
        if fit.get("shared_parameters") is not False or fit.get("implementation") != "reused_independent_heads":
            raise ValueError("classical joint control semantics changed")
        if _sha(source / "fit.json") != fit["reused_parent_fit_sha256"]: raise ValueError("reused classical parent fit changed")
        full_grid.check_fit(fcfg, "classical", task)
    else: raise ValueError("unknown fit kind")
    return fit


def frozen_fit_manifest(cfg, reg):
    manifest = {}
    for kind, key in (("neural", "neural_tasks"), ("classical", "classical_tasks")):
        for task in reg[key]:
            path = directory(cfg, kind, task) / "fit.json"
            if not path.exists(): raise ValueError("global joint Test closed: missing fit")
            check_fit(cfg, kind, task); manifest[str(path.relative_to(Path(cfg["output_dir"])))] = _sha(path)
    return manifest


def status(cfg, stage=None):
    out = Path(cfg["output_dir"]); path = out / "status.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    reg = json.loads((out / "preregistration.json").read_text())
    value = {"utc": now(), "stage": stage or old.get("stage", "training"), "cells": 336,
        "new_joint_cells": 112, "reused_parent_cells": 224,
        "neural": {"complete": sum((directory(cfg, "neural", t) / "fit.json").exists() for t in reg["neural_tasks"]), "total": len(reg["neural_tasks"])},
        "classical_controls": {"complete": sum((directory(cfg, "classical", t) / "fit.json").exists() for t in reg["classical_tasks"]), "total": len(reg["classical_tasks"])},
        "test_opened": (out / "test_barrier.json").exists()}
    _json(path, value)
    html = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>336条件</title><style>body{{font-family:system-ui;max-width:1000px;margin:40px auto;line-height:1.8}}</style><h1>4入力 × 4モデル × 7対象 × 回帰・分類・同時学習 = 336条件</h1><p>段階: {value['stage']}。新しい共有層の同時学習 {value['neural']['complete']}/{value['neural']['total']} fit、独立head対照 {value['classical_controls']['complete']}/{value['classical_controls']['total']} bundle。</p><p>回帰・分類224条件は既存の固定実験を再利用。線形/LightGBMの同時学習枠は同じ独立推定器の組み合わせで、共有表現の追加学習はTransformer/WMのみ。全7対象に回帰値と分類確率を出す。</p><p>Train/Val/TestとTrain由来ラベルは全条件で共通。全新規fitとVal選択を固定し、親224条件の完了後にTestを集計。過去に探索したTestの記述的OOS比較。</p><p><a href="final/index.html">336条件のTest結果</a> / <a href="selection/index.html">追加112条件のVal結果</a> / <a href="campaign.log">ログ</a></p><p>更新 UTC: {value['utc']}</p></html>'''
    temp = out / "progress.html.tmp"; temp.write_text(html, encoding="utf-8"); temp.replace(out / "progress.html")
    print(json.dumps(value), flush=True); return value


def run_tasks(cfg, tasks, worker, stage):
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
    reg, _, _ = verify(cfg); remaining = []
    for task in reg["neural_tasks"]:
        if (directory(cfg, "neural", task) / "fit.json").exists(): check_fit(cfg, "neural", task)
        elif (Path(cfg["output_dir"]) / "test_barrier.json").exists():
            raise ValueError("a missing fit cannot be retrained after the global Test barrier")
        else: remaining.append(task)
    run_tasks(cfg, remaining, train_neural, "training")


def parent_ready(cfg, require_complete=True):
    """Read manifests only while waiting; no Test forecast/outcome read here."""
    _, fcfg, _ = verify(cfg); root = Path(fcfg["output_dir"])
    freg = json.loads((root / "preregistration.json").read_text())
    if not require_complete:
        if not all((full_grid.directory(fcfg, "classical", t) / "fit.json").exists() for t in freg["classical_tasks"]):
            return False
        for task in freg["classical_tasks"]: full_grid.check_fit(fcfg, "classical", task)
        return True
    path = root / "completion.json"
    if not path.exists(): return False
    completion = json.loads(path.read_text()); state = json.loads((root / "status.json").read_text())
    if (completion.get("complete") is not True or completion.get("status") != "complete"
            or completion.get("registration_sha256") != _sha(root / "preregistration.json")):
        raise ValueError("parent224 completion provenance invalid")
    # The parent writes completion.json just before its final status update.
    # That normal atomic-write interval is pending, not a failed experiment.
    if state.get("stage") != "complete": return False
    # Both parent reports and inference must finish; all fit/barrier hashes are rechecked.
    full_grid.frozen_fit_manifest(fcfg, freg)
    first = freg["classical_tasks"][0]; full_grid.test_access(fcfg, "classical", first)
    for filename in ("final/results_224.csv", "statistics_completion.json"):
        if not (root / filename).exists(): raise ValueError("parent224 completion missing final artifacts")
    statistics = json.loads((root / "statistics_completion.json").read_text())
    if (statistics.get("complete") is not True
            or statistics.get("daily_counts_sha256") != _sha(root / "final/day_classification_counts.csv")
            or statistics.get("inference_sha256") != _sha(root / "final/formulation_inference.csv")):
        raise ValueError("parent224 statistics completion provenance invalid")
    return True


def wait_for_parent(cfg, require_complete):
    stage = "waiting_for_parent_224_completion" if require_complete else "waiting_for_parent_classical_fits"
    while not parent_ready(cfg, require_complete):
        status(cfg, stage); time.sleep(cfg["parent_poll_seconds"])


def freeze_validation_choices(cfg):
    from .wm_joint_grid_report import write_joint_grid_report
    reg, _, _ = verify(cfg); manifest = frozen_fit_manifest(cfg, reg); out = Path(cfg["output_dir"])
    path = out / "validation_choices.json"
    if path.exists():
        choices = json.loads(path.read_text())
        if (choices.get("test_used") is not False or choices.get("registration_sha256") != _sha(out / "preregistration.json")
                or choices.get("frozen_fit_sha256") != manifest
                or choices.get("selection_report_sha256") != _sha(out / "selection/joint_results_112.csv")):
            raise ValueError("frozen joint validation choices changed")
        return choices
    if (out / "test_barrier.json").exists(): raise ValueError("Val choices cannot be rebuilt after Test opened")
    report = write_joint_grid_report(out, cfg, include_test=False)
    choices = {"frozen_utc": now(), "registration_sha256": _sha(out / "preregistration.json"),
        "frozen_fit_sha256": manifest, "test_used": False, "calibration": "identity",
        "selection_report_sha256": _sha(out / "selection/joint_results_112.csv"), "report": report,
        "rule": "fixed .5/.5 seven-target checkpoint objective; all112joint conditions retained, no Test-dependent choices"}
    _json(path, choices); return choices


def open_test_barrier(cfg):
    reg, fcfg, _ = verify(cfg, full=True); out = Path(cfg["output_dir"])
    manifest = frozen_fit_manifest(cfg, reg); path = out / "validation_choices.json"
    if not path.exists(): raise ValueError("joint Val choices not frozen")
    choices = json.loads(path.read_text())
    if (choices.get("test_used") is not False or choices.get("registration_sha256") != _sha(out / "preregistration.json")
            or choices.get("frozen_fit_sha256") != manifest
            or choices.get("selection_report_sha256") != _sha(out / "selection/joint_results_112.csv")):
        raise ValueError("invalid joint validation choices provenance")
    if not parent_ready(cfg, require_complete=True): raise ValueError("global joint Test closed: parent224 incomplete")
    parent_root = Path(fcfg["output_dir"])
    value = {"opened_utc": now(), "registration_sha256": _sha(out / "preregistration.json"),
        "frozen_fit_sha256": manifest, "validation_choices_sha256": _sha(path), "test_selection_allowed": False,
        "parent_completion_sha256": _sha(parent_root / "completion.json"),
        "parent_test_barrier_sha256": _sha(parent_root / "test_barrier.json"),
        "parent_results_224_sha256": _sha(parent_root / "final/results_224.csv")}
    barrier_path = out / "test_barrier.json"
    if barrier_path.exists():
        old = json.loads(barrier_path.read_text())
        if any(old.get(key) != val for key, val in value.items() if key != "opened_utc"):
            raise ValueError("frozen joint Test barrier artifacts changed")
        return old
    _json(barrier_path, value); return value


def test_access(cfg, kind, task):
    reg, fcfg, _ = verify(cfg); out = Path(cfg["output_dir"]); path = out / "test_barrier.json"
    if not path.exists(): raise ValueError("global joint Test barrier closed")
    barrier = json.loads(path.read_text())
    if (barrier.get("registration_sha256") != _sha(out / "preregistration.json")
            or barrier.get("test_selection_allowed") is not False
            or barrier.get("validation_choices_sha256") != _sha(out / "validation_choices.json")):
        raise ValueError("joint Test barrier/choices changed")
    required = {f"{kind}/{row['job']}/fit.json" for kind, key in (("neural", "neural_tasks"), ("classical", "classical_tasks")) for row in reg[key]}
    if set(barrier["frozen_fit_sha256"]) != required: raise ValueError("incomplete joint global fit manifest")
    for relative, expected in barrier["frozen_fit_sha256"].items():
        frozen = out / relative
        if not frozen.exists() or _sha(frozen) != expected: raise ValueError("joint global fit changed after Test barrier")
    relative = str((directory(cfg, kind, task) / "fit.json").relative_to(out))
    if relative not in required: raise ValueError("task outside registered joint Test manifest")
    parent_root = Path(fcfg["output_dir"])
    for filename, key in (("completion.json", "parent_completion_sha256"), ("test_barrier.json", "parent_test_barrier_sha256"),
                          ("final/results_224.csv", "parent_results_224_sha256")):
        if _sha(parent_root / filename) != barrier[key]: raise ValueError("frozen parent224 Test provenance changed")
    check_fit(cfg, kind, task); return barrier


def evaluate_task(config_path, kind, task):
    cfg = load_config(config_path); barrier = test_access(cfg, kind, task); reg, fcfg, pcfg = verify(cfg)
    dest = directory(cfg, kind, task); existing = dest / "evaluation.json"
    if existing.exists():
        value = json.loads(existing.read_text())
        if (value["barrier_sha256"] != _sha(Path(cfg["output_dir"]) / "test_barrier.json")
                or value["prediction_sha256"] != _sha(dest / "test_predictions.parquet")):
            raise ValueError("existing joint Test prediction provenance changed")
        return task
    if kind == "neural":
        worker_setup(cfg["training"]["cpu_threads"])
        arrays, data = parent.arrays_for(pcfg), parent.fold_for(pcfg, task["fold"])
        fold = fold_info(reg, task["fold"])
        checkpoint = torch.load(dest / "research_model.pt", map_location="cpu", weights_only=False)
        if checkpoint["registration_sha256"] != barrier["registration_sha256"] or checkpoint["task"] != task:
            raise ValueError("joint checkpoint registration/task mismatch")
        model = model_for(task, cfg, checkpoint["scales"]).to(cfg["device"]); model.load_state_dict(checkpoint["state_dict"])
        frame = prediction_frame(arrays, data["test"], predict(model, arrays, data["test"], data, task, cfg), task, "test", fold["label_contract"])
    else:
        full_grid.test_access(fcfg, "classical", task)
        source = full_grid.directory(fcfg, "classical", task)
        evaluation = json.loads((source / "evaluation.json").read_text())
        if (evaluation["barrier_sha256"] != _sha(Path(fcfg["output_dir"]) / "test_barrier.json")
                or evaluation["prediction_sha256"] != _sha(source / "test_predictions.parquet")):
            raise ValueError("reused parent classical Test predictions changed")
        original = pd.read_parquet(source / "test_predictions.parquet")
        if set(original["split"]) != {"test"}: raise ValueError("parent classical Test artifact has wrong split")
        frame = combine_classical_heads(original)
    parquet(frame, dest / "test_predictions.parquet")
    _json(existing, {"barrier_sha256": _sha(Path(cfg["output_dir"]) / "test_barrier.json"),
        "prediction_sha256": _sha(dest / "test_predictions.parquet"), "utc": now()})
    return task


def evaluate_neural(config_path, task):
    return evaluate_task(config_path, "neural", task)


def campaign(cfg):
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    with (out / "campaign.lock").open("a") as lock:
        try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("joint coordinator already running") from None
        try:
            verify(cfg, full=True); status(cfg, "training"); run_fits(cfg)
            wait_for_parent(cfg, require_complete=False); reg, _, _ = verify(cfg)
            status(cfg, "preparing_independent_classical_controls")
            for task in reg["classical_tasks"]: build_classical_control(cfg, task); status(cfg)
            status(cfg, "freezing_validation"); freeze_validation_choices(cfg)
            wait_for_parent(cfg, require_complete=True); open_test_barrier(cfg)
            status(cfg, "test_inference"); run_tasks(cfg, reg["neural_tasks"], evaluate_neural, "test_inference")
            for task in reg["classical_tasks"]: evaluate_task(cfg["config_path"], "classical", task); status(cfg)
            from .wm_joint_grid_report import write_joint_grid_report
            from .wm_joint_grid_statistics import write_joint_grid_statistics
            status(cfg, "final_report"); report = write_joint_grid_report(out, cfg, include_test=True)
            status(cfg, "block_bootstrap_statistics"); statistics = write_joint_grid_statistics(out, cfg)
            table = pd.read_csv(out / "final/results_336.csv")
            keys = ["group", "model", "task", "formulation"]
            if len(table) != 336 or table.duplicated(keys).any(): raise ValueError("final complete336 scope missing")
            completion = {"complete": True, "status": "complete", "utc": now(), "cells": 336,
                "n_fresh_joint_neural_cells": 56, "n_independent_joint_control_cells": 56,
                "distinct_empirical_conditions": 280,
                "registration_sha256": _sha(out / "preregistration.json"), "report": report, "statistics": statistics,
                "results_336_sha256": _sha(out / "final/results_336.csv")}
            _json(out / "completion.json", completion); status(cfg, "complete")
        except Exception as exc:
            _json(out / "failure.json", {"utc": now(), "error": repr(exc), "traceback": traceback.format_exc()})
            status(cfg, "failed"); raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status", "worker"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--job")
    args = parser.parse_args(); cfg = load_config(args.config)
    if args.command == "prepare": prepare(cfg)
    elif args.command == "run": campaign(cfg)
    elif args.command == "status": status(cfg)
    else:
        reg, _, _ = verify(cfg)
        matches = [task for task in reg["neural_tasks"] if task["job"] == args.job]
        if len(matches) != 1: parser.error("--job must identify one registered neural task")
        train_neural(cfg["config_path"], matches[0])


if __name__ == "__main__": main()
