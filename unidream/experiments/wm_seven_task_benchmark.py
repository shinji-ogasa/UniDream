"""Frozen research-only seven-outcome benchmark, independent of trading CLI.

Commands prepare/pilot/run/status/evaluate/summarize accept only a fixed YAML.
All model selection finishes before a global barrier permits Test inference.
Previously inspected historical Test is explicitly retrospective research OOS.
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

from .wm_technical_ablation import ROOT, _arrays, _features, _input, _json, _load_config, _sha, _verify_freeze
from .wm_ablation_evaluation import variance_qlike
from .wm_comparison_classical import ewma_variance_forecast, fit_gap_reset_garch, garch_variance_forecast, log_returns_15m
from .wm_seven_task_evaluation import TASKS, TARGET_DEFINITIONS, build_seven_targets, fit_target_scales, score_seven
from .wm_seven_task_models import DirectSevenTaskTransformer, SevenTaskWM
from .wm_seven_task_classical import (
    fit_classical_bundle, predict_classical_bundle, fit_variance_filters, predict_variance_filters,
)


DEFAULT_CONFIG = ROOT / "configs/experiments/wm_seven_task_benchmark_20261110.yaml"
MODEL_NAMES = ("direct", "wm")
LOSS_WEIGHTS = {"return": 1., "variance": .2, "downside": .2, "crash": .1,
                "dd_excess": .2, "advantage": .2, "recovery": .2}


def load_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg["test_selection_allowed"] is not False or cfg["horizons"] != [16]:
        raise ValueError("first benchmark is four-hour, Test report-only")
    if cfg["groups"] != ["A", "B", "C", "D"] or cfg["seeds"] != [7, 17, 27]:
        raise ValueError("fixed 2x2 and three-seed family required")
    if cfg["validation_partition"] != "calendar_months_earlystop_calibration_selection":
        raise ValueError("three disjoint validation calendar months required")
    cfg["config_path"] = str(Path(path).resolve())
    for key in ("output_dir", "parent_config", "tabular_cache"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def source_hashes():
    names = ("wm_seven_task_benchmark.py", "wm_seven_task_evaluation.py",
             "wm_seven_task_models.py", "wm_seven_task_classical.py", "wm_seven_task_report.py",
             "wm_technical_ablation.py", "wm_ablation_features.py", "wm_ablation_model.py",
             "wm_ablation_evaluation.py", "wm_comparison_classical.py")
    paths = [Path(__file__).parent / name for name in names]
    paths += [ROOT / "unidream/world_model/transformer.py", ROOT / "unidream/world_model/encoder.py"]
    return {str(path.relative_to(ROOT)): _sha(path) for path in paths}


def validation_parts(ids, index_ns, val_start, horizon):
    """Calendar partition with STRICT label-end purges at every boundary."""
    start = pd.Timestamp(val_start)
    boundaries = [start + pd.DateOffset(months=i) for i in range(4)]
    parts = {}
    for name, low, high in zip(("earlystop", "calibration", "selection"), boundaries[:-1], boundaries[1:]):
        keep = (index_ns[ids] >= low.value) & (index_ns[ids] + horizon * 15 * 60 * 10**9 < high.value)
        parts[name] = np.asarray(ids[keep], dtype=np.int64)
        if len(parts[name]) < 300:
            raise ValueError(f"insufficient {name} validation support: {len(parts[name])}")
    return parts


def tabular_columns(group, context=256):
    """256x19 common raw/market tokens followed by 44 final causal TA values."""
    if group not in ("A", "B", "C", "D"):
        raise ValueError("unknown group")
    channels = np.arange(19 if group in ("C", "D") else 5)
    positions = (np.arange(context)[:, None] * 19 + channels[None, :]).ravel()
    if group in ("B", "D"):
        positions = np.r_[positions, np.arange(context * 19, context * 19 + 44)]
    return positions


def cache_directory(cfg):
    return Path(cfg["output_dir"]) / "cache"


def prepare_tabular_cache(cfg, parent, frozen, arrays):
    """Copy proven old feature rows, computing only the earlier Train rows missing there."""
    pieces = []
    for fold in cfg["folds"]:
        with np.load(Path(parent["output_dir"]) / "prepared" / f"fold_{fold:02d}.npz") as z:
            pieces.extend(z[name] for name in ("train", "val", "test"))
    decisions = np.unique(np.concatenate(pieces))
    old_ids = np.load(Path(cfg["tabular_cache"]) / "decisions.npy", mmap_mode="r")
    old = np.load(Path(cfg["tabular_cache"]) / "tabular_raw.npy", mmap_mode="r")
    if old.shape != (len(old_ids), 4908):
        raise ValueError("unexpected source feature-cache shape")
    cache = cache_directory(cfg); cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "decisions.npy", decisions)
    design = np.lib.format.open_memmap(cache / "tabular_raw.npy", mode="w+", dtype="float32", shape=(len(decisions), 4908))
    missing = []
    for start in range(0, len(decisions), 128):
        ids = decisions[start:start + 128]
        loc = np.searchsorted(old_ids, ids)
        exists = (loc < len(old_ids)) & (old_ids[np.minimum(loc, len(old_ids) - 1)] == ids)
        value = np.empty((len(ids), 4908), dtype=np.float32)
        if exists.any(): value[exists] = old[loc[exists]]
        if (~exists).any():
            new_ids = ids[~exists]
            feature = _features(arrays["raw"], arrays["market"], new_ids, cfg["model"]["context"])
            value[~exists] = np.concatenate((feature[..., :19].reshape(len(new_ids), -1), feature[:, -1, 19:]), axis=1)
            missing.extend(new_ids.tolist())
        if not np.isfinite(value).all(): raise ValueError("nonfinite common input cache")
        design[start:start + len(ids)] = value
        if start % (128 * 100) == 0: print(f"tabular cache {start}/{len(decisions)}", flush=True)
    design.flush(); del design
    _json(cache / "construction.json", {"n_rows": len(decisions), "n_columns": 4908,
        "reused_rows": len(decisions) - len(missing), "computed_missing_train_rows": len(missing),
        "source_path": cfg["tabular_cache"], "source_modified": False})
    return decisions


def prepare(cfg):
    out = Path(cfg["output_dir"])
    if (out / "preregistration.json").exists():
        raise ValueError("registration is immutable; verify/status an existing experiment")
    parent = _load_config(cfg["parent_config"])
    frozen = _verify_freeze(parent)
    original = _arrays(parent["output_dir"])
    targets = build_seven_targets(pd.to_datetime(original["index_ns"], utc=True), original["raw"][:, 3],
        horizons=cfg["horizons"], **cfg["targets"])
    prepared = out / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    for key in (*TASKS, "mask", "label_end_ns", "baseline_variance"):
        np.save(prepared / f"{key}.npy", targets[key])
    decisions = prepare_tabular_cache(cfg, parent, frozen, original)
    cache = np.load(cache_directory(cfg) / "tabular_raw.npy", mmap_mode="r")
    if cache.shape != (len(decisions), 4908):
        raise ValueError("unexpected common-information tabular cache")
    reports = []
    for fold in cfg["folds"]:
        old = frozen["folds"][fold]
        with np.load(Path(parent["output_dir"]) / "prepared" / f"fold_{fold:02d}.npz") as z:
            data = {k: z[k] for k in ("train", "val", "test", "mean", "std")}
        for split in ("train", "val", "test"):
            ids = data[split]
            if not targets["mask"][ids].all():
                raise ValueError("new labels lack common parent support")
            if not np.all(targets["label_end_ns"][ids] < pd.Timestamp(old[f"{split}_end"]).value):
                raise ValueError("label crosses a split boundary")
            loc = np.searchsorted(decisions, ids)
            if not np.array_equal(decisions[loc], ids):
                raise ValueError("cache misses a common parent decision")
        partitions = validation_parts(data["val"], original["index_ns"], old["val_start"], max(cfg["horizons"]))
        data.update(partitions)
        panel = partitions["earlystop"]
        data["selection_panel"] = panel[np.linspace(0, len(panel) - 1, min(len(panel), cfg["training"]["selection_rows"])).astype(int)]
        rng = np.random.default_rng(20261110 + fold)
        data["train_panel"] = np.sort(rng.choice(data["train"], min(len(data["train"]), cfg["training"]["diagnostic_train_rows"]), replace=False))
        np.savez(prepared / f"fold_{fold:02d}.npz", **data)
        scales = fit_target_scales(targets, data["train"])
        reports.append({"fold": fold, **{k: old[k] for k in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end")},
            "scales": scales, "support": {k: len(data[k]) for k in ("train", "val", "test", "earlystop", "calibration", "selection")}})
        print("prepared", fold, reports[-1]["support"], flush=True)
    _json(prepared / "folds.json", reports)
    # D first; its exact jobs are reused by the complete A/B/C/D family.
    jobs = [{"fold": f, "group": g, "seed": s, "model": m, "job": f"fold_{f:02d}_{g}_{m}_s{s}"}
            for g in ("D", "A", "B", "C") for f in cfg["folds"] for s in cfg["seeds"] for m in MODEL_NAMES]
    classical = [{"fold": f, "group": g, "job": f"fold_{f:02d}_{g}"} for g in ("D", "A", "B", "C") for f in cfg["folds"]]
    registration = {"experiment_id": cfg["experiment_id"], "registered_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg, "config_sha256": _sha(cfg["config_path"]), "source_sha256": source_hashes(),
        "parent_registration_sha256": _sha(Path(parent["output_dir"]) / "preregistration.json"),
        "cache_sha256": {name: _sha(cache_directory(cfg) / name) for name in ("decisions.npy", "tabular_raw.npy")},
        "source_cache_sha256": {name: _sha(Path(cfg["tabular_cache"]) / name) for name in ("decisions.npy", "tabular_raw.npy")},
        "prepared_sha256": {p.name: _sha(p) for p in sorted(prepared.iterdir())},
        "target_definitions": TARGET_DEFINITIONS, "loss_weights": LOSS_WEIGHTS, "folds": reports,
        "neural_tasks": jobs, "classical_tasks": classical,
        "selection": "single common seven-supervised-objective checkpoint selected only on first Val month; third Val month report-only model/condition selection; middle month reserved, identity calibration",
        "information": "all methods have the same 256 raw OHLCV/market observations; explicit TA is a deterministic within-window transform; tabular keeps entire 19-channel history plus final44TA, neural all causal TA tokens",
        "test_access": "global all-fit barrier before any new Test inference; historical test is previously inspected retrospective OOS, no pristine performance claim",
        "cohort": "immutable parent common-cohort (all market inputs observed and 16h boundary-purged), conservatively reused for the4h labels; no future row used as input",
        "auxiliary_baselines": "past16 integrated variance, Val-chosen EWMA, Train-fitted gap-reset GARCH; no fallback for failed GARCH",
        "limitations": ["source publication times were not historically verified; parent assumes conservative15min source lag", "advantage is a specified constant-inventory surrogate, not executable policy PnL", "recovery is clipped risk-adjusted forward return, not time-to-recovery", "shared neural objectives may exhibit negative transfer; standalone tasks are a later separately registered comparison"],
        "environment": {"torch": torch.__version__, "numpy": np.__version__, "device": cfg["device"]}}
    _json(out / "preregistration.json", registration)
    print(f"Registered {len(jobs)} neural fits and {len(classical)} classical bundles", flush=True)


def verify(cfg, full=False):
    out = Path(cfg["output_dir"])
    reg = json.loads((out / "preregistration.json").read_text())
    if reg["config_sha256"] != _sha(cfg["config_path"]) or reg["source_sha256"] != source_hashes():
        raise ValueError("registered config/source changed")
    if full:
        for name, expected in reg["prepared_sha256"].items():
            if _sha(out / "prepared" / name) != expected:
                raise ValueError(f"prepared data changed: {name}")
        for name, expected in reg["cache_sha256"].items():
            if _sha(cache_directory(cfg) / name) != expected:
                raise ValueError(f"common tabular cache changed: {name}")
        for name, expected in reg["source_cache_sha256"].items():
            if _sha(Path(cfg["tabular_cache"]) / name) != expected:
                raise ValueError(f"source tabular cache changed: {name}")
    return reg


def arrays_for(cfg):
    parent = _load_config(cfg["parent_config"])
    arrays = dict(_arrays(parent["output_dir"]))
    for task in TASKS:
        arrays[task] = np.load(Path(cfg["output_dir"]) / "prepared" / f"{task}.npy", mmap_mode="r")
    arrays["mask"] = np.load(Path(cfg["output_dir"]) / "prepared/mask.npy", mmap_mode="r")
    arrays["label_end_ns"] = np.load(Path(cfg["output_dir"]) / "prepared/label_end_ns.npy", mmap_mode="r")
    return arrays


def fold_for(cfg, fold):
    with np.load(Path(cfg["output_dir"]) / "prepared" / f"fold_{fold:02d}.npz") as z:
        return {k: z[k] for k in z.files}


def objective_numpy(actual, prediction, scales):
    """Exactly the common supervised component, no WM auxiliary objectives."""
    parts = {}
    for task in TASKS:
        a, p = np.asarray(actual[task]), np.asarray(prediction[task])
        if a.shape != p.shape or not np.isfinite(a).all() or not np.isfinite(p).all():
            raise ValueError("selection requires fully finite aligned targets/predictions")
        if task == "variance":
            parts[task] = float(np.mean(variance_qlike(a, p)))
        elif task == "crash":
            p = np.clip(p, 1e-7, 1 - 1e-7)
            parts[task] = float(np.mean(-(a * np.log(p) + (1 - a) * np.log(1 - p))))
        else:
            scale = np.asarray(scales[task])
            parts[task] = float(np.mean(((p - a) / scale) ** 2))
    return {"score": sum(LOSS_WEIGHTS[k] * v for k, v in parts.items()), "parts": parts}


def make_model(task, cfg, scales):
    cls = SevenTaskWM if task["model"] == "wm" else DirectSevenTaskTransformer
    return cls(63, cfg["model"], scales)


def predict(model, arrays, ids, data, group, cfg):
    model.eval()
    pieces = {k: [] for k in TASKS}
    with torch.no_grad():
        for start in range(0, len(ids), cfg["training"]["evaluation_batch_size"]):
            x, _ = _input(arrays, ids[start:start + cfg["training"]["evaluation_batch_size"]], data, group, cfg["model"]["context"])
            result = model.predict_last(torch.from_numpy(x).to(cfg["device"]))
            for key in TASKS:
                pieces[key].append(result[key].cpu().numpy())
    return {key: np.concatenate(value) for key, value in pieces.items()}


def prediction_frame(arrays, ids, predictions, task, split, horizon=16):
    frame = pd.DataFrame({"decision_id": ids, "timestamp": pd.to_datetime(arrays["index_ns"][ids], utc=True),
        "fold": task["fold"], "group": task["group"], "seed": task.get("seed", 0),
        "model": task["model"], "split": split, "horizon": horizon})
    for key in TASKS:
        frame["actual_" + key] = arrays[key][ids, 0]
        frame["pred_" + key] = np.asarray(predictions[key]).reshape(len(ids))
    return frame


def train_neural(config_path, task):
    cfg = load_config(config_path)
    reg = verify(cfg)
    out = Path(cfg["output_dir"])
    directory = out / "neural" / task["job"]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "fit.json").exists():
        fit = json.loads((directory / "fit.json").read_text())
        if fit["registration_sha256"] != _sha(out / "preregistration.json"):
            raise ValueError("stale neural fit")
        return {"job": task["job"], "elapsed_seconds": fit["elapsed_seconds"], "cached": True}
    with (directory / "training.log").open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        started = time.monotonic()
        torch.set_num_threads(cfg["training"]["cpu_threads"])
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        random.seed(task["seed"]); np.random.seed(task["seed"]); torch.manual_seed(task["seed"])
        if cfg["device"] == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        arrays, data = arrays_for(cfg), fold_for(cfg, task["fold"])
        fold = next(x for x in reg["folds"] if x["fold"] == task["fold"])
        model = make_model(task, cfg, fold["scales"]).to(cfg["device"])
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
        rng = np.random.default_rng(task["seed"])
        best, anchor, last_significant, best_step = np.inf, np.inf, 0, 0
        history, best_state = [], None
        target_counts = np.zeros(len(cfg["horizons"]), dtype=np.int64)
        gradient_steps = np.zeros(len(TASKS) * len(cfg["horizons"]), dtype=np.int64)
        reason = "max_steps"
        for step in range(1, cfg["training"]["max_steps"] + 1):
            model.train()
            ids = rng.choice(data["train"], cfg["training"]["batch_size"], replace=True)
            x, base = _input(arrays, ids, data, task["group"], cfg["model"]["context"])
            # Input token closes at row k-1; its future-only target starts row k.
            positions = ids[:, None] - cfg["model"]["context"] + np.arange(1, cfg["model"]["context"] + 1)[None, :]
            valid = arrays["mask"][positions] & (arrays["label_end_ns"][positions] < pd.Timestamp(fold["train_end"]).value) & (arrays["index_ns"][positions, None] >= pd.Timestamp(fold["train_start"]).value)
            target_counts += valid.sum((0, 1))
            targets = {key: torch.from_numpy(arrays[key][positions].astype(np.float32)).to(cfg["device"]) for key in TASKS}
            next_return = np.log(arrays["raw"][positions, 3] / arrays["raw"][positions - 1, 3]).astype(np.float32)
            opt.zero_grad(set_to_none=True)
            loss, parts = model.loss(torch.from_numpy(x).to(cfg["device"]), torch.from_numpy(base).to(cfg["device"]), targets,
                torch.from_numpy(valid).to(cfg["device"]), torch.from_numpy(next_return).to(cfg["device"]))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            gradient = model.supervised_output_layer.weight.grad
            if gradient is None:
                raise RuntimeError("missing supervised gradient")
            gradient_steps += (gradient.detach().abs().sum(1).cpu().numpy() > 0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"], error_if_nonfinite=True)
            opt.step()
            if step % cfg["training"]["validation_every"] == 0:
                panel = predict(model, arrays, data["selection_panel"], data, task["group"], cfg)
                val = objective_numpy({key: arrays[key][data["selection_panel"]] for key in TASKS}, panel, fold["scales"])
                row = {"step": step, "batch_total_loss": float(loss.detach().cpu()), "selection": val,
                       "batch_loss_parts": {key: float(value.cpu()) for key, value in parts.items()}, "elapsed_seconds": time.monotonic() - started}
                history.append(row); print(json.dumps(row), flush=True)
                if val["score"] < best:
                    best, best_step = val["score"], step
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                if not np.isfinite(anchor) or val["score"] < anchor - cfg["training"]["relative_improvement"] * abs(anchor):
                    anchor, last_significant = val["score"], step
                _json(directory / "progress.json", {**task, "step": step, "best_step": best_step, "best_score": best,
                    "elapsed_seconds": time.monotonic() - started})
                if step >= cfg["training"]["min_steps"] and step - last_significant >= cfg["training"]["patience_steps"]:
                    reason = "validation_patience"; break
        if best_state is None or not (target_counts > 0).all() or not (gradient_steps > 0).all():
            raise RuntimeError("zero target/gradient coverage or no selected checkpoint")
        model.load_state_dict(best_state); model.eval()
        torch.save({"state_dict": best_state, "task": task, "scales": fold["scales"],
                    "registration_sha256": _sha(out / "preregistration.json"), "best_step": best_step}, directory / "research_model.pt")
        frames = []
        for split in ("earlystop", "calibration", "selection"):
            prediction = predict(model, arrays, data[split], data, task["group"], cfg)
            frames.append(prediction_frame(arrays, data[split], prediction, task, split))
        pd.concat(frames, ignore_index=True).to_parquet(directory / "validation_predictions.parquet", index=False)
        trained = predict(model, arrays, data["train_panel"], data, task["group"], cfg)
        training_diagnostic = objective_numpy({key: arrays[key][data["train_panel"]] for key in TASKS}, trained, fold["scales"])
        fit = {**task, "registration_sha256": _sha(out / "preregistration.json"), "elapsed_seconds": time.monotonic() - started,
            "steps": step, "best_step": best_step, "best_score": best, "stop_reason": reason,
            "max_step_censored": reason == "max_steps", "target_counts": target_counts.tolist(),
            "supervised_gradient_steps": gradient_steps.tolist(), "parameters": sum(p.numel() for p in model.parameters()),
            "history": history, "selected_train_diagnostic": training_diagnostic,
            "checkpoint_sha256": _sha(directory / "research_model.pt"), "validation_predictions_sha256": _sha(directory / "validation_predictions.parquet")}
        _json(directory / "fit.json", fit)
        return {"job": task["job"], "elapsed_seconds": fit["elapsed_seconds"], "best_step": best_step, "steps": step}


def tabular_x(cfg, ids, group):
    decisions = np.load(cache_directory(cfg) / "decisions.npy", mmap_mode="r")
    raw = np.load(cache_directory(cfg) / "tabular_raw.npy", mmap_mode="r")
    loc = np.searchsorted(decisions, ids)
    if not np.array_equal(decisions[loc], ids):
        raise ValueError("unknown tabular decision")
    return np.asarray(raw[np.ix_(loc, tabular_columns(group))], dtype=np.float32)


def train_classical(config_path, task):
    cfg = load_config(config_path); reg = verify(cfg)
    directory = Path(cfg["output_dir"]) / "classical" / task["job"]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "fit.json").exists():
        fit = json.loads((directory / "fit.json").read_text())
        if fit["registration_sha256"] != _sha(Path(cfg["output_dir"]) / "preregistration.json"):
            raise ValueError("stale classical fit")
        return {"job": task["job"], "cached": True}
    with (directory / "training.log").open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log), threadpool_limits(cfg["classical"]["cpu_threads"]):
        started = time.monotonic()
        arrays, data = arrays_for(cfg), fold_for(cfg, task["fold"])
        xtrain = tabular_x(cfg, data["train"], task["group"])
        xval = tabular_x(cfg, data["earlystop"], task["group"])
        bundle = fit_classical_bundle(xtrain, {k: arrays[k][data["train"], 0] for k in TASKS},
            xval, {k: arrays[k][data["earlystop"], 0] for k in TASKS}, cfg["classical"])
        if task["group"] == "D":
            fold = next(x for x in reg["folds"] if x["fold"] == task["fold"])
            bundle["variance_filters"] = fit_variance_filters(arrays["raw"][:, 3], arrays["index_ns"],
                pd.Timestamp(fold["train_start"]).value, pd.Timestamp(fold["train_end"]).value,
                data["earlystop"], arrays["variance"][data["earlystop"], 0], horizon=16, context=256)
            bundle["metadata"]["variance_filters"] = bundle["variance_filters"]
        with (directory / "models.pkl").open("wb") as stream:
            pickle.dump(bundle, stream, protocol=5)
        del xtrain, xval
        frames = []
        for split in ("earlystop", "calibration", "selection"):
            result = predict_classical_bundle(bundle, tabular_x(cfg, data[split], task["group"]))
            for name, prediction in result.items():
                frames.append(prediction_frame(arrays, data[split], prediction, {**task, "model": name}, split))
            if "variance_filters" in bundle:
                for name, value in predict_variance_filters(arrays["raw"][:, 3], data[split], bundle["variance_filters"]).items():
                    pred = {key: np.full(len(data[split]), np.nan) for key in TASKS}
                    pred["variance"] = value
                    frames.append(prediction_frame(arrays, data[split], pred, {**task, "model": name}, split))
        pd.concat(frames, ignore_index=True).to_parquet(directory / "validation_predictions.parquet", index=False)
        fit = {**task, "registration_sha256": _sha(Path(cfg["output_dir"]) / "preregistration.json"),
            "elapsed_seconds": time.monotonic() - started, "metadata": bundle["metadata"],
            "model_sha256": _sha(directory / "models.pkl"), "validation_predictions_sha256": _sha(directory / "validation_predictions.parquet")}
        _json(directory / "fit.json", fit)
        return {"job": task["job"], "elapsed_seconds": fit["elapsed_seconds"]}


def status(cfg):
    reg = verify(cfg)
    out = Path(cfg["output_dir"])
    result = {"utc": datetime.now(timezone.utc).isoformat()}
    for kind, key in (("neural", "neural_tasks"), ("classical", "classical_tasks")):
        done = [task for task in reg[key] if (out / kind / task["job"] / "fit.json").exists()]
        result[kind] = {"complete": len(done), "total": len(reg[key]), "by_group": {g: sum(t["group"] == g for t in done) for g in cfg["groups"]}}
    result["test_opened"] = (out / "test_barrier.json").exists()
    _json(out / "status.json", result)
    state_path = out / "campaign_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    page = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>7タスク実験の進捗</title>
<style>body{{font-family:system-ui,sans-serif;max-width:850px;margin:50px auto;padding:20px;line-height:1.8}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:12px}}code{{background:#eee}}</style>
<h1>4時間先・7タスク比較の進捗</h1><p>段階: <b>{state.get('stage', '準備済み')}</b> / 状態: {state.get('status', '進行中')}</p>
<table><tr><th>処理</th><th>完了 / 予定</th></tr><tr><td>Transformer / WM</td><td>{result['neural']['complete']} / {result['neural']['total']}</td></tr><tr><td>線形 / LightGBM等の比較セット</td><td>{result['classical']['complete']} / {result['classical']['total']}</td></tr></table>
<p>条件別ニューラル完了数: {result['neural']['by_group']}</p><p>全条件の学習・Val選択後にTestを評価。過去に確認済みの期間なので、研究用の時系列OOS評価として扱う。</p>
<p>最終更新 UTC: {result['utc']}</p><p><a href="report.html">完成後の結果レポート</a> / <a href="status.json">詳細進捗</a> / <a href="campaign.log">実行ログ</a> / <a href="validation_choices.json">Valで固定した選択</a></p></html>'''
    temporary = out / "progress.html.tmp"
    temporary.write_text(page, encoding="utf-8"); temporary.replace(out / "progress.html")
    print(json.dumps(result), flush=True)
    return result


def run(cfg, pilot=False):
    reg = verify(cfg, full=True)
    jobs = reg["neural_tasks"]
    classics = reg["classical_tasks"]
    if pilot:
        jobs = [t for t in jobs if t["group"] == "D" and t["seed"] == cfg["seeds"][0] and t["fold"] in cfg["pilot_folds"]]
        classics = [t for t in classics if t["group"] == "D" and t["fold"] in cfg["pilot_folds"]]
    out = Path(cfg["output_dir"])
    jobs = [t for t in jobs if not (out / "neural" / t["job"] / "fit.json").exists()]
    classics = [t for t in classics if not (out / "classical" / t["job"] / "fit.json").exists()]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(cfg["training"]["workers"], mp_context=ctx, max_tasks_per_child=1) as neural_pool, ProcessPoolExecutor(cfg["classical"]["workers"], mp_context=ctx, max_tasks_per_child=1) as classical_pool:
        futures = {neural_pool.submit(train_neural, cfg["config_path"], task): ("neural", task) for task in jobs}
        futures.update({classical_pool.submit(train_classical, cfg["config_path"], task): ("classical", task) for task in classics})
        for future in as_completed(futures):
            kind, task = futures[future]
            try:
                result = future.result()
                print(json.dumps({"completed": kind, **result}), flush=True)
            except Exception as exc:
                _json(out / "failure.json", {"kind": kind, "task": task, "error": repr(exc), "utc": datetime.now(timezone.utc).isoformat()})
                for f in futures: f.cancel()
                raise
            status(cfg)
    status(cfg)


def open_test_barrier(cfg):
    reg = verify(cfg, full=True)
    out = Path(cfg["output_dir"])
    fits = {}
    for kind, key in (("neural", "neural_tasks"), ("classical", "classical_tasks")):
        for task in reg[key]:
            path = out / kind / task["job"] / "fit.json"
            if not path.exists():
                raise ValueError(f"Test closed: missing {path}")
            fit = json.loads(path.read_text())
            if fit["registration_sha256"] != _sha(out / "preregistration.json"):
                raise ValueError("fit provenance mismatch")
            selected = "research_model.pt" if kind == "neural" else "models.pkl"
            expected = fit["checkpoint_sha256" if kind == "neural" else "model_sha256"]
            if _sha(path.parent / selected) != expected:
                raise ValueError("selected model artifact changed")
            if _sha(path.parent / "validation_predictions.parquet") != fit["validation_predictions_sha256"]:
                raise ValueError("frozen validation predictions changed")
            fits[str(path.relative_to(out))] = _sha(path)
    choices_path = out / "validation_choices.json"
    barrier = {"opened_utc": datetime.now(timezone.utc).isoformat(), "registration_sha256": _sha(out / "preregistration.json"),
               "frozen_fit_sha256": fits, "test_selection_allowed": False,
               "validation_choices_sha256": _sha(choices_path) if choices_path.exists() else None}
    if (out / "test_barrier.json").exists():
        old = json.loads((out / "test_barrier.json").read_text())
        if old["frozen_fit_sha256"] != fits:
            raise ValueError("fits changed after Test barrier")
        if old.get("validation_choices_sha256") != barrier["validation_choices_sha256"]:
            raise ValueError("validation choices changed after Test barrier")
    else:
        _json(out / "test_barrier.json", barrier)
    return reg


def evaluate_task(config_path, kind, task):
    cfg = load_config(config_path)
    verify(cfg)
    out = Path(cfg["output_dir"])
    if not (out / "test_barrier.json").exists():
        raise ValueError("global Test barrier not opened")
    directory = out / kind / task["job"]
    target = directory / "test_predictions.parquet"
    barrier = json.loads((out / "test_barrier.json").read_text())
    if barrier["registration_sha256"] != _sha(out / "preregistration.json"):
        raise ValueError("Test barrier registration mismatch")
    if barrier.get("validation_choices_sha256") is not None and _sha(out / "validation_choices.json") != barrier["validation_choices_sha256"]:
        raise ValueError("validation choices changed after Test barrier")
    fit_path = directory / "fit.json"
    if barrier["frozen_fit_sha256"].get(str(fit_path.relative_to(out))) != _sha(fit_path):
        raise ValueError("fit changed after global Test barrier")
    fit = json.loads(fit_path.read_text())
    model_path = directory / ("research_model.pt" if kind == "neural" else "models.pkl")
    if _sha(model_path) != fit["checkpoint_sha256" if kind == "neural" else "model_sha256"]:
        raise ValueError("selected checkpoint changed after fit")
    if target.exists() and (directory / "evaluation.json").exists():
        previous = json.loads((directory / "evaluation.json").read_text())
        if previous["barrier_sha256"] != _sha(out / "test_barrier.json") or previous["prediction_sha256"] != _sha(target):
            raise ValueError("cached Test artifact changed")
        return {"job": task["job"], "cached": True}
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    arrays, data = arrays_for(cfg), fold_for(cfg, task["fold"])
    if kind == "neural":
        state = torch.load(directory / "research_model.pt", map_location="cpu", weights_only=False)
        model = make_model(task, cfg, state["scales"]).to(cfg["device"])
        model.load_state_dict(state["state_dict"])
        prediction = predict(model, arrays, data["test"], data, task["group"], cfg)
        frame = prediction_frame(arrays, data["test"], prediction, task, "test")
    else:
        with (directory / "models.pkl").open("rb") as stream: bundle = pickle.load(stream)
        with threadpool_limits(cfg["classical"]["cpu_threads"]):
            result = predict_classical_bundle(bundle, tabular_x(cfg, data["test"], task["group"]))
        frame = pd.concat([prediction_frame(arrays, data["test"], pred, {**task, "model": name}, "test") for name, pred in result.items()], ignore_index=True)
        if "variance_filters" in bundle:
            extra = []
            for name, value in predict_variance_filters(arrays["raw"][:, 3], data["test"], bundle["variance_filters"]).items():
                pred = {key: np.full(len(data["test"]), np.nan) for key in TASKS}
                pred["variance"] = value
                extra.append(prediction_frame(arrays, data["test"], pred, {**task, "model": name}, "test"))
            frame = pd.concat([frame, *extra], ignore_index=True)
    temporary = target.with_suffix(".parquet.part")
    frame.to_parquet(temporary, index=False)
    temporary.replace(target)
    _json(directory / "evaluation.json", {"barrier_sha256": _sha(out / "test_barrier.json"),
        "prediction_sha256": _sha(target), "selected_model_sha256": _sha(model_path), "rows": len(frame)})
    return {"job": task["job"], "rows": len(frame)}


def evaluate(cfg):
    freeze_validation_choices(cfg)
    reg = open_test_barrier(cfg)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(cfg["training"]["workers"], mp_context=ctx, max_tasks_per_child=1) as pool:
        futures = [pool.submit(evaluate_task, cfg["config_path"], kind, task) for kind, key in (("neural", "neural_tasks"), ("classical", "classical_tasks")) for task in reg[key]]
        for future in as_completed(futures): print(json.dumps({"evaluated": future.result()}), flush=True)


def summarize(cfg):
    reg = verify(cfg)
    out = Path(cfg["output_dir"])
    files = list(out.glob("neural/*/validation_predictions.parquet")) + list(out.glob("classical/*/validation_predictions.parquet"))
    if (out / "test_barrier.json").exists():
        files += list(out.glob("neural/*/test_predictions.parquet")) + list(out.glob("classical/*/test_predictions.parquet"))
    if not files:
        raise ValueError("no completed predictions")
    metrics = []
    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        frames.append(frame)
        for keys, group in frame.groupby(["split", "group", "model", "fold", "seed"], sort=False):
            split, condition, model, fold, seed = keys
            scores = score_frame(group)
            for task in TASKS:
                if task not in scores: continue
                values = scores[task]
                metrics.append({"split": split, "group": condition, "model": model, "fold": fold, "seed": seed, "task": task,
                                **{k: v for k, v in values.items() if not isinstance(v, (dict, list))}})
    all_predictions = pd.concat(frames, ignore_index=True)
    all_predictions.to_parquet(out / "all_predictions.parquet", index=False)
    pd.DataFrame(metrics).to_csv(out / "fold_metrics.csv", index=False)
    pooled = []
    for keys, frame in all_predictions.groupby(["split", "group", "model"], sort=False):
        scores = score_frame(frame)
        for task in TASKS:
            if task not in scores: continue
            values = scores[task]
            pooled.append({"split": keys[0], "group": keys[1], "model": keys[2], "task": task,
                "n_unique_timestamps": frame.timestamp.nunique(), "n_seeds": frame.seed.nunique(), **{k: v for k, v in values.items() if not isinstance(v, (dict, list))}})
    pd.DataFrame(pooled).to_csv(out / "pooled_metrics.csv", index=False)
    _json(out / "summary.json", {"metrics": pooled, "n_prediction_rows": len(all_predictions), "test_report_only": True,
        "historical_oos_only": True, "completed_neural_fits": len(list(out.glob("neural/*/fit.json"))),
        "registered_neural_fits": len(reg["neural_tasks"])})
    print(f"Saved {len(all_predictions)} predictions and {len(pooled)} pooled task scores", flush=True)


def score_frame(frame):
    if frame.model.iloc[0] in ("past_variance", "ewma", "garch"):
        actual = frame.actual_variance.to_numpy()
        pred = frame.pred_variance.to_numpy()
        if not np.isfinite(actual).all() or not np.isfinite(pred).all():
            raise ValueError("nonfinite variance baseline forecast")
        return {"variance": {"qlike": float(np.mean(variance_qlike(actual, pred))),
            "rmse": float(np.sqrt(np.mean((actual - pred) ** 2))), "calibration_mean_ratio": float(pred.sum() / actual.sum())}}
    scores = score_seven({k: frame["actual_" + k].to_numpy() for k in TASKS}, {k: frame["pred_" + k].to_numpy() for k in TASKS})
    if scores["n_nonfinite_excluded"]:
        raise ValueError("a learned model has nonfinite forecasts on common evaluation cohort")
    if frame.seed.nunique() > 1:
        # Ranking pooled across different seeds invents cross-model comparisons.
        # Proper row losses remain pooled; rank metrics are averaged per seed.
        by_seed = [score_seven({k: g["actual_" + k].to_numpy() for k in TASKS},
                              {k: g["pred_" + k].to_numpy() for k in TASKS})
                   for _, g in frame.groupby("seed", sort=True)]
        for task in TASKS:
            for metric in ("pearson_ic", "spearman_ic", "pr_auc"):
                if metric not in scores[task]: continue
                values = [s[task][metric] for s in by_seed if s[task].get(metric) is not None]
                scores[task][metric] = float(np.mean(values)) if values else None
    return scores


def freeze_validation_choices(cfg):
    """Save next-stage candidates before any Test inference, using final Val only."""
    reg = verify(cfg)
    out = Path(cfg["output_dir"])
    source_hashes = {}
    for kind, key in (("neural", "neural_tasks"), ("classical", "classical_tasks")):
        for task in reg[key]:
            directory = out / kind / task["job"]
            if not (directory / "fit.json").exists():
                raise ValueError("cannot select before every fit has completed")
            path = directory / "validation_predictions.parquet"
            source_hashes[str(path.relative_to(out))] = _sha(path)
    destination = out / "validation_choices.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["validation_prediction_sha256"] != source_hashes:
            raise ValueError("validation changed after choices were fixed")
        return saved
    if (out / "test_barrier.json").exists():
        raise ValueError("cannot create validation choices after Test was opened")
    summarize(cfg)
    metrics = pd.read_csv(out / "pooled_metrics.csv")
    final = metrics.loc[metrics.split == "selection"]
    primary = {"return": ("zero_oos_r2", False), "variance": ("qlike", True), "downside": ("mae", True),
               "crash": ("brier", True), "dd_excess": ("mae", True), "advantage": ("spearman_ic", False), "recovery": ("spearman_ic", False)}
    choices = {}
    for task, (metric, ascending) in primary.items():
        candidates = final.loc[final.task == task].dropna(subset=[metric]).sort_values([metric, "group", "model"], ascending=[ascending, True, True])
        if not len(candidates):
            choices[task] = {"selected": None, "reason": "no finite primary metric"}
        else:
            best = candidates.iloc[0]
            choices[task] = {"group": str(best.group), "model": str(best.model), "metric": metric, "value": float(best[metric])}
    result = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "selection_split": "third calendar month of Val",
        "identity_calibration": True, "test_used": False, "registration_sha256": _sha(out / "preregistration.json"),
        "validation_prediction_sha256": source_hashes, "next_stage_candidates": choices,
        "warning": "candidate selection is not evidence of out-of-sample improvement; Test never changes these choices"}
    _json(destination, result)
    return result


def campaign(cfg):
    """Complete the registered family, then infer/report without manual tuning."""
    out = Path(cfg["output_dir"])
    state = {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    try:
        for stage, operation in (("小規模学習・実データ確認", lambda: run(cfg, pilot=True)),
                                 ("全4入力条件の学習", lambda: run(cfg)),
                                 ("Val選択の固定とTest評価", lambda: evaluate(cfg)),
                                 ("結果集計", lambda: summarize(cfg))):
            state["stage"] = stage; _json(out / "campaign_state.json", state); status(cfg)
            operation()
        from .wm_seven_task_report import write_report
        state["stage"] = "統計評価とレポート"; _json(out / "campaign_state.json", state); status(cfg)
        write_report(out)
        verify(cfg, full=True)
        state.update(status="complete", stage="完了", completed_utc=datetime.now(timezone.utc).isoformat())
        _json(out / "campaign_state.json", state); _json(out / "completion.json", state); status(cfg)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), failed_utc=datetime.now(timezone.utc).isoformat())
        _json(out / "campaign_state.json", state); status(cfg)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "pilot", "run", "status", "evaluate", "summarize", "campaign"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(); cfg = load_config(args.config)
    if args.command == "prepare": prepare(cfg)
    elif args.command == "pilot": run(cfg, pilot=True)
    elif args.command == "run": run(cfg)
    elif args.command == "evaluate": evaluate(cfg)
    elif args.command == "summarize": summarize(cfg)
    elif args.command == "campaign": campaign(cfg)
    else: status(cfg)


if __name__ == "__main__":
    main()
