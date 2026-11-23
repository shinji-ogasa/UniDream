"""Chronological model comparison and validation-only learning-curve diagnosis.

Reuses immutable data/cohorts from the technical-feature experiment. No mainline
training, checkpoint-loader or trading bundle behavior is changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import scipy.linalg
import torch
import yaml
from threadpoolctl import threadpool_limits

from .wm_ablation_evaluation import holm_adjust, paired_block_bootstrap, variance_qlike
from .wm_ablation_model import FeatureAblationWM
from .wm_technical_ablation import (
    ROOT, _arrays, _features, _input, _json, _load_config, _sha, _verify_freeze,
)
from .wm_comparison_models import DirectForecastTransformer
from .wm_comparison_classical import (
    ewma_variance_forecast, fit_gap_reset_garch, garch_variance_forecast,
    log_returns_15m, score_forecasts,
)

DEFAULT_CONFIG = ROOT / "configs/experiments/wm_model_comparison_20261109.yaml"


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg["test_selection_allowed"] is not False or cfg["input_group"] != "D" or cfg["primary_horizon"] != 16:
        raise ValueError("fixed D-group, four-hour, report-only comparison required")
    cfg["config_path"] = str(Path(path).resolve())
    for name in ("output_dir", "parent_config"):
        cfg[name] = str((ROOT / cfg[name]).resolve())
    return cfg


def source_hashes():
    names = ["wm_model_comparison.py", "wm_comparison_models.py", "wm_comparison_classical.py"]
    return {name: _sha(Path(__file__).parent / name) for name in names}


def tabular_design(windows, market_positions):
    """Same OHLCV history; explicit TA and sparse market lags for tabular models."""
    return np.concatenate((windows[..., :5].reshape(len(windows), -1),
        windows[:, -1, 5:], windows[:, market_positions, 5:19].reshape(len(windows), -1)), axis=1)


def tabular_scaler(fold_data, context, positions):
    def expand(value):
        return np.concatenate((np.tile(value[:5], context), value[5:], np.tile(value[5:19], len(positions))))
    return expand(fold_data["mean"]), expand(fold_data["std"])


def prepare(cfg):
    parent = _load_config(cfg["parent_config"])
    frozen = _verify_freeze(parent)
    out = Path(cfg["output_dir"])
    if (out / "preregistration.json").exists():
        raise ValueError("comparison already registered; do not overwrite")
    out.mkdir(parents=True, exist_ok=True)
    arrays = _arrays(parent["output_dir"])
    indices = []
    for fold in range(len(frozen["folds"])):
        with np.load(Path(parent["output_dir"]) / "prepared" / f"fold_{fold:02d}.npz") as data:
            indices.extend([data[name] for name in ("train", "val", "test")])
    decisions = np.unique(np.concatenate(indices))
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    np.save(cache / "decisions.npy", decisions)
    context = parent["model"]["context"]
    positions = cfg["tabular"]["market_lag_positions"]
    width = context * 5 + 58 + len(positions) * 14
    design = np.lib.format.open_memmap(cache / "tabular_raw.npy", mode="w+", dtype="float32", shape=(len(decisions), width))
    batch = cfg["tabular"]["feature_batch"]
    for start in range(0, len(decisions), batch):
        ids = decisions[start:start + batch]
        design[start:start + len(ids)] = tabular_design(_features(arrays["raw"], arrays["market"], ids, context), positions)
        if start % (batch * 200) == 0:
            print(f"features {start}/{len(decisions)}", flush=True)
    design.flush()
    del design
    tasks = [{"model": name, "fold": f["fold"], "seed": seed,
              "job": f"fold_{f['fold']:02d}_{name}_s{seed}"}
             for f in frozen["folds"] for seed in cfg["seeds"] for name in cfg["neural"]["models"]]
    registration = {
        "experiment_id": cfg["experiment_id"], "registered_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg, "config_sha256": _sha(cfg["config_path"]), "source_sha256": source_hashes(),
        "parent_preregistration_sha256": _sha(Path(parent["output_dir"]) / "preregistration.json"),
        "cache_sha256": {p.name: _sha(p) for p in sorted(cache.iterdir())}, "tasks": tasks,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "pandas", "lightgbm", "arch")},
        "runtime": {"lightgbm_library_sha256": _sha(ROOT / ".venv/lib/python3.12/site-packages/lightgbm/lib/lib_lightgbm.dylib"),
                    "openmp_library_sha256": _sha(ROOT / ".venv/lib/python3.12/site-packages/torch/lib/libomp.dylib"),
                    "macos_setup": "LightGBM library has @loader_path/../../torch/lib added to rpath and an ad-hoc codesign, using the already bundled PyTorch OpenMP runtime."},
        "selection": "All methods share the immutable 384 selection_val decisions. Full Val and Test are report-only.",
        "neural_selection": "minimum return normalized MSE +0.2 variance QLIKE, h16; true argmin independent of patience anchor",
        "information": "D: same 256 completed raw bars; tabular flattens OHLCV and adds final TA/market plus four market lags; neural sees all 63 channels at all tokens",
        "history_filter": "EWMA/GARCH reset each prediction window to train-only unconditional one-bar variance; only 255 within-window close returns used",
        "training": "WM extended starts from scratch; same original architecture, three horizons and full original loss. Direct transformer uses continuous encoder and return/variance loss only; this changes multiple WM components together.",
        "early_stop": "patience measures significant relative improvements, selected checkpoint is always true argmin; max-step termination is right-censored and never labelled convergence",
        "test_access": "No new Test predictions before all neural fits and classical selections are frozen; parent Test results were previously examined, so retrospective research OOS, not a pristine holdout.",
        "statistics": "Classical fits deterministic, one per fold; duplicate their predictions across paired neural seeds solely to average neural loss, never as independent observations; 12 primary tests Holm corrected.",
    }
    _json(out / "preregistration.json", registration)
    print(f"Registered {len(tasks)} neural fits and 15 classical folds", flush=True)


def verify(cfg):
    parent = _load_config(cfg["parent_config"])
    _verify_freeze(parent)
    out = Path(cfg["output_dir"])
    frozen = json.loads((out / "preregistration.json").read_text())
    if frozen["config_sha256"] != _sha(cfg["config_path"]) or frozen["source_sha256"] != source_hashes():
        raise ValueError("comparison config/source changed after registration")
    if frozen["parent_preregistration_sha256"] != _sha(Path(parent["output_dir"]) / "preregistration.json"):
        raise ValueError("parent experiment changed")
    for name, expected in frozen["cache_sha256"].items():
        if _sha(out / "cache" / name) != expected:
            raise ValueError(f"cache changed: {name}")
    for name, expected in frozen["packages"].items():
        if importlib.metadata.version(name) != expected: raise ValueError(f"package changed: {name}")
    return parent, frozen


def update_patience(score, step, anchor, last_significant, relative_improvement):
    """Patience anchor differs from the true checkpoint argmin."""
    if not np.isfinite(score):
        raise ValueError("nonfinite selection score")
    if not np.isfinite(anchor) or score < anchor - relative_improvement * abs(anchor):
        return score, step
    return anchor, last_significant


def diagnostic_scores(arrays, ids, prediction, scale):
    mse = float(np.mean(((prediction["return"][:, 1] - arrays["returns"][ids, 1]) / scale) ** 2))
    qlike = float(np.mean(variance_qlike(arrays["variance"][ids, 1], prediction["variance"][:, 1])))
    return {"return_normalized_mse": mse, "variance_qlike": qlike, "score": mse + .2 * qlike}


def model_for(task, parent, fold):
    if task["model"] == "wm_extended":
        return FeatureAblationWM(63, parent["model"], fold["scales"])
    if task["model"] == "direct_transformer":
        return DirectForecastTransformer(63, parent["model"], fold["scales"])
    raise ValueError(task)


def predict_two(model, arrays, ids, fold_data, parent):
    model.eval()
    pieces = {"return": [], "variance": []}
    with torch.no_grad():
        for start in range(0, len(ids), parent["training"]["evaluation_batch_size"]):
            selected = ids[start:start + parent["training"]["evaluation_batch_size"]]
            x, _ = _input(arrays, selected, fold_data, "D", parent["model"]["context"])
            result = model.predict_last(torch.from_numpy(x).to(parent["device"]))
            for key in pieces:
                pieces[key].append(result[key].cpu().numpy())
    return {key: np.concatenate(value) for key, value in pieces.items()}


def train_task(config_path, task):
    cfg = load_config(config_path)
    parent = _load_config(cfg["parent_config"])
    out = Path(cfg["output_dir"])
    directory = out / "neural" / task["job"]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "fit.json").exists():
        previous = json.loads((directory / "fit.json").read_text())
        if previous["preregistration_sha256"] != _sha(out / "preregistration.json"): raise ValueError("stale neural fit")
        return previous
    with (directory / "training.log").open("w", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
        started = time.monotonic()
        torch.set_num_threads(cfg["cpu_threads_per_worker"])
        torch.use_deterministic_algorithms(True)
        random.seed(task["seed"]); np.random.seed(task["seed"]); torch.manual_seed(task["seed"])
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        arrays = _arrays(parent["output_dir"])
        with np.load(Path(parent["output_dir"]) / "prepared" / f"fold_{task['fold']:02d}.npz") as data:
            fold_data = {key: data[key] for key in data.files}
        fold = json.loads((Path(parent["output_dir"]) / "prepared/folds.json").read_text())[task["fold"]]
        model = model_for(task, parent, fold).to(cfg["device"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=parent["training"]["learning_rate"], weight_decay=parent["training"]["weight_decay"])
        rng = np.random.default_rng(task["seed"])
        diagnostic_rng = np.random.default_rng(cfg["neural"]["diagnostic_seed"] + task["fold"])
        train_panel = np.sort(diagnostic_rng.choice(fold_data["train"], cfg["neural"]["train_diagnostic_rows"], replace=False))
        context = parent["model"]["context"]
        train_start = pd.Timestamp(fold["train_start"]).value
        train_end = pd.Timestamp(fold["train_end"]).value
        best_score = float("inf"); best_step = 0; best_state = None
        anchor = float("inf"); last_significant = 0
        history = []; reason = "max_steps"
        gradient_steps = np.zeros(model.head[-1].out_features, dtype=np.int64)
        target_counts = np.zeros(3, dtype=np.int64)
        for step in range(1, cfg["neural"]["max_steps"] + 1):
            model.train()
            decisions = rng.choice(fold_data["train"], parent["training"]["batch_size"], replace=True)
            x, base = _input(arrays, decisions, fold_data, "D", context)
            positions = decisions[:, None] - context + np.arange(1, context + 1)[None, :]
            ends = arrays["index_ns"][positions, None] + np.array((4, 16, 64))[None, None, :] * 15 * 60 * 10**9
            valid = arrays["target_mask"][positions] & (ends < train_end) & (arrays["index_ns"][positions, None] >= train_start)
            target_counts += valid.sum((0, 1))
            targets = {key: torch.from_numpy(arrays[name][positions].astype(np.float32)).to(cfg["device"])
                for key, name in (("return", "returns"), ("variance", "variance"), ("downside", "downside"), ("crash", "crash"))}
            next_return = np.log(arrays["raw"][positions, 3] / arrays["raw"][positions - 1, 3]).astype(np.float32)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model.loss(torch.from_numpy(x).to(cfg["device"]), torch.from_numpy(base).to(cfg["device"]),
                targets, torch.from_numpy(valid).to(cfg["device"]), torch.from_numpy(next_return).to(cfg["device"]))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            gradient_steps += (model.head[-1].weight.grad.detach().abs().sum(1).cpu().numpy() > 0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), parent["training"]["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            if step % cfg["neural"]["validation_every"] == 0:
                selected = predict_two(model, arrays, fold_data["selection_val"], fold_data, parent)
                val = diagnostic_scores(arrays, fold_data["selection_val"], selected, fold["scales"]["return"][1])
                trained = predict_two(model, arrays, train_panel, fold_data, parent)
                train = diagnostic_scores(arrays, train_panel, trained, fold["scales"]["return"][1])
                row = {"step": step, "batch_total_train_loss": float(loss.detach().cpu()), "train": train, "val": val,
                       "batch_loss_parts": {key: float(value.cpu()) for key, value in parts.items()}}
                history.append(row); print(json.dumps(row), flush=True)
                if val["score"] < best_score:
                    best_score = val["score"]; best_step = step
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                anchor, last_significant = update_patience(val["score"], step, anchor, last_significant, cfg["neural"]["relative_improvement"])
                if step >= cfg["neural"]["min_steps"] and step - last_significant >= cfg["neural"]["patience_steps"]:
                    reason = "validation_patience"; break
        if best_state is None or not np.all(target_counts > 0) or not np.all(gradient_steps > 0):
            raise RuntimeError("no selected checkpoint or empty target/gradient coverage")
        torch.save({"format": "model_comparison_research_v1", "task": task, "model": best_state,
            "best_step": best_step, "best_val_score": best_score, "scales": fold["scales"],
            "preregistration_sha256": _sha(out / "preregistration.json")}, directory / "research_model.pt")
        result = {**task, "steps": step, "stop_reason": reason, "best_step": best_step,
            "best_val_score": best_score, "last_significant_improvement_step": last_significant,
            "right_censored": reason == "max_steps", "parameters": sum(p.numel() for p in model.parameters()),
            "preregistration_sha256": _sha(out / "preregistration.json"),
            "target_counts": target_counts.tolist(), "nonzero_gradient_steps": gradient_steps.tolist(),
            "history": history, "train_diagnostic_indices_sha256": __import__("hashlib").sha256(train_panel.astype('<i8').tobytes()).hexdigest(),
            "checkpoint_sha256": _sha(directory / "research_model.pt"), "elapsed_seconds": time.monotonic() - started}
        _json(directory / "fit.json", result)
        if cfg["device"] == "mps": torch.mps.empty_cache()
        return {key: result[key] for key in ("job", "steps", "stop_reason", "best_step", "elapsed_seconds")}


def run_neural(cfg):
    _, frozen = verify(cfg)
    out = Path(cfg["output_dir"])
    pending = [task for task in frozen["tasks"] if not (out / "neural" / task["job"] / "fit.json").exists()]
    errors = []
    with ProcessPoolExecutor(max_workers=cfg["parallel_workers"], mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(train_task, cfg["config_path"], task): task for task in pending}
        _json(out / "neural_status.json", {"status": "running", "completed": len(frozen["tasks"]) - len(pending), "total": len(frozen["tasks"]), "errors": errors})
        for future in as_completed(futures):
            try: print(json.dumps(future.result()), flush=True)
            except Exception as exc: errors.append({"task": futures[future], "error": repr(exc)})
            completed = sum((out / "neural" / task["job"] / "fit.json").exists() for task in frozen["tasks"])
            _json(out / "neural_status.json", {"status": "failed" if errors else "running", "completed": completed, "total": len(frozen["tasks"]), "errors": errors})
    _json(out / "neural_status.json", {"status": "failed" if errors else "fits_completed", "completed": len(frozen["tasks"]) - len(errors), "total": len(frozen["tasks"]), "errors": errors})
    if errors: raise RuntimeError(errors)


def get_design(cfg, parent, arrays, ids, fold_data):
    out = Path(cfg["output_dir"])
    positions = np.load(out / "cache/decisions.npy", mmap_mode="r")
    selected = np.searchsorted(positions, ids)
    if not np.array_equal(positions[selected], ids): raise ValueError("missing tabular cache rows")
    raw = np.load(out / "cache/tabular_raw.npy", mmap_mode="r")
    mean, std = tabular_scaler(fold_data, parent["model"]["context"], cfg["tabular"]["market_lag_positions"])
    return (raw[selected] - mean) / std


def _classical_predictions(cfg, parent, arrays, fold_data, fit, ids):
    from lightgbm import Booster
    x = get_design(cfg, parent, arrays, ids, fold_data)
    ret_scale = fit["return_scale"]; var_scale = fit["variance_scale"]
    ret_lgb = Booster(model_file=fit["lightgbm_return_path"]).predict(x, num_threads=cfg["classical_threads"]) * ret_scale
    var_lgb = Booster(model_file=fit["lightgbm_variance_path"]).predict(x, num_threads=cfg["classical_threads"]) * var_scale
    ridge = np.load(fit["ridge_path"])
    with threadpool_limits(limits=cfg["classical_threads"]):
        ridge_pred = np.einsum('ij,j->i', x.astype(np.float64), ridge['coefficients'], optimize=False) + float(ridge['intercept'])
    close = arrays["raw"][:, 3]
    # GARCH object is pickled by numpy only inside the local, trusted research artifact.
    garch_object = __import__('pickle').loads(Path(fit["garch_path"]).read_bytes())
    ewma = ewma_variance_forecast(close, ids, initial_variance=fit["one_bar_variance"], decay=fit["ewma_decay"], context=256, horizon=16)
    garch = garch_variance_forecast(close, ids, garch_object, initial_variance=fit["one_bar_variance"], context=256, horizon=16)
    return {
        "zero": (np.zeros(len(ids)), None), "train_mean": (np.full(len(ids), fit["return_mean"]), np.full(len(ids), var_scale)),
        "ridge": (ridge_pred, None), "lightgbm": (ret_lgb, var_lgb),
        "past_variance": (None, arrays["baseline_variance"][ids, 1]),
        "ewma": (None, ewma), "garch": (None, garch),
    }


def run_classical(cfg):
    import lightgbm as lgb
    import pickle
    from dataclasses import asdict
    parent, frozen = verify(cfg)
    out = Path(cfg["output_dir"])
    arrays = _arrays(parent["output_dir"])
    folds = json.loads((Path(parent["output_dir"]) / "prepared/folds.json").read_text())
    with threadpool_limits(limits=cfg["classical_threads"]):
        for fold in folds:
            directory = out / "classical" / f"fold_{fold['fold']:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "fit.json").exists():
                if json.loads((directory / "fit.json").read_text())["preregistration_sha256"] != _sha(out / "preregistration.json"):
                    raise ValueError("stale classical fit")
                continue
            started = time.monotonic()
            data = dict(np.load(Path(parent["output_dir"]) / "prepared" / f"fold_{fold['fold']:02d}.npz"))
            train, val = data["train"], data["selection_val"]
            x = get_design(cfg, parent, arrays, train, data)
            xv = get_design(cfg, parent, arrays, val, data)
            y = arrays["returns"][train, 1].astype(np.float64)
            yv = arrays["returns"][val, 1].astype(np.float64)
            xd = x.astype(np.float64); mean_x = xd.mean(0); mean_y = float(y.mean())
            xd -= mean_x
            gram = xd.T @ xd
            cross = xd.T @ (y - mean_y)
            eig, vectors = scipy.linalg.eigh(gram, check_finite=True)
            if eig.min() < -1e-5: raise RuntimeError("non-PSD Ridge gram")
            transformed = vectors.T @ cross
            candidates = []
            for alpha in cfg["tabular"]["ridge_alphas"]:
                coef = vectors @ (transformed / (np.maximum(eig, 0) + alpha))
                intercept = mean_y - mean_x @ coef
                predicted = xv.astype(np.float64) @ coef + intercept
                candidates.append((float(np.mean((predicted - yv) ** 2)), alpha, coef, intercept))
            best = min(candidates, key=lambda c: c[0])
            ridge_path = directory / "ridge.npz"
            np.savez(ridge_path, coefficients=best[2], intercept=best[3], alpha=best[1])
            del xd, gram, eig, vectors
            selected = {}; lightgbm_history = {}
            for target in ("return", "variance"):
                target_array = arrays["returns" if target == "return" else "variance"][:, 1]
                scale = fold["scales"][target][1]
                target_train = target_array[train] / scale
                target_val = target_array[val] / scale
                if target == "variance":
                    target_train = np.maximum(target_array[train], 1e-12) / scale
                    target_val = np.maximum(target_array[val], 1e-12) / scale
                fitted = []
                for leaves in cfg["tabular"]["lightgbm_leaves"]:
                    model = lgb.LGBMRegressor(objective="regression" if target == "return" else "gamma",
                        n_estimators=cfg["tabular"]["lightgbm_rounds"], learning_rate=cfg["tabular"]["lightgbm_learning_rate"],
                        num_leaves=leaves, min_child_samples=cfg["tabular"]["lightgbm_min_data_leaf"],
                        reg_lambda=cfg["tabular"]["lightgbm_l2"], random_state=cfg["tabular"]["lightgbm_seed"],
                        n_jobs=cfg["classical_threads"], verbosity=-1, deterministic=True, force_col_wise=True,
                        subsample=1., colsample_bytree=1., max_bin=63)
                    model.fit(x, target_train, eval_set=[(xv, target_val)],
                        eval_metric="l2" if target == "return" else "gamma",
                        callbacks=[lgb.early_stopping(cfg["tabular"]["lightgbm_patience"], verbose=False)])
                    p = model.predict(xv)
                    score = float(np.mean((p-target_val)**2)) if target == "return" else float(np.mean(variance_qlike(target_val, p)))
                    fitted.append((score, leaves, model))
                selected_model = min(fitted, key=lambda item: item[0])
                path = directory / f"lightgbm_{target}.txt"
                selected_model[2].booster_.save_model(str(path))
                selected[target] = str(path)
                lightgbm_history[target] = {"selected_leaves": selected_model[1], "selection_score": selected_model[0],
                    "candidates": [{"leaves": leaves, "score": score, "best_iteration": model.best_iteration_, "history": model.evals_result_} for score, leaves, model in fitted]}
            close = arrays["raw"][:, 3]
            raw_train_mask = (arrays["index_ns"] >= pd.Timestamp(fold["train_start"]).value) & (arrays["index_ns"] < pd.Timestamp(fold["train_end"]).value)
            train_returns = log_returns_15m(close[raw_train_mask])
            one_bar = float(np.nanmean(train_returns**2))
            garch = fit_gap_reset_garch(train_returns, initial_variance=one_bar, return_scale=cfg["variance"]["garch_scale"])
            if not garch.success: raise RuntimeError(f"GARCH fit failed: {garch}")
            garch_path = directory / "garch.pkl"
            garch_path.write_bytes(pickle.dumps(garch))
            ewma_candidates = []
            for decay in cfg["variance"]["ewma_decays"]:
                prediction = ewma_variance_forecast(close, val, initial_variance=one_bar, decay=decay, context=256, horizon=16)
                ewma_candidates.append((float(np.mean(variance_qlike(arrays["variance"][val,1], prediction))), decay))
            fit = {"fold": fold["fold"], "train_n": len(train), "selection_val_n": len(val),
                "preregistration_sha256": _sha(out / "preregistration.json"),
                "return_scale": fold["scales"]["return"][1], "variance_scale": fold["scales"]["variance"][1],
                "return_mean": mean_y, "one_bar_variance": one_bar, "ridge_alpha": best[1],
                "ridge_candidates": [{"score": c[0], "alpha": c[1]} for c in candidates], "ridge_path": str(ridge_path),
                "lightgbm_return_path": selected["return"], "lightgbm_variance_path": selected["variance"], "lightgbm": lightgbm_history,
                "garch_path": str(garch_path), "garch": asdict(garch), "ewma_decay": min(ewma_candidates)[1],
                "ewma_candidates": [{"score": score, "decay": decay} for score,decay in ewma_candidates],
                "elapsed_seconds": time.monotonic() - started}
            fit["artifact_sha256"] = {path.name: _sha(path) for path in (ridge_path, garch_path, Path(selected["return"]), Path(selected["variance"]))}
            _json(directory / "fit.json", fit)
            _json(out / "classical_status.json", {"status": "running", "completed": fold["fold"] + 1, "total": len(folds)})
            print(json.dumps({"fold": fold["fold"], "seconds": fit["elapsed_seconds"], "ridge_alpha": best[1], "ewma_decay": fit["ewma_decay"]}), flush=True)
        _json(out / "classical_status.json", {"status": "fits_completed", "completed": len(folds), "total": len(folds)})


def prediction_frame(arrays, ids, fold, model, seed, split, ret, var):
    frame = pd.DataFrame({"timestamp": pd.to_datetime(arrays["index_ns"][ids], utc=True), "fold": fold,
        "group": model, "seed": seed, "horizon": 16, "split": split,
        "actual_return": arrays["returns"][ids, 1], "actual_variance": arrays["variance"][ids, 1],
        "pred_return": np.nan if ret is None else ret, "pred_variance": np.nan if var is None else var})
    for key, value in (("pred_return", ret), ("pred_variance", var)):
        if value is not None and not np.isfinite(frame[key]).all(): raise ValueError(f"nonfinite {model} {key}")
    if var is not None and not (frame.pred_variance > 0).all(): raise ValueError(f"nonpositive {model} variance")
    return frame


def score_frame(frame):
    args = {}
    for target in ("return", "variance"):
        if frame["pred_" + target].notna().all():
            args["actual_" + target] = frame["actual_" + target].to_numpy()
            args["pred_" + target] = frame["pred_" + target].to_numpy()
    return score_forecasts(**args)


def evaluate(cfg):
    parent, frozen = verify(cfg)
    out = Path(cfg["output_dir"]); old = Path(parent["output_dir"])
    if any(not (out / "neural" / t["job"] / "fit.json").exists() for t in frozen["tasks"]):
        raise ValueError("all neural selections must be complete before Test")
    if any(not (out / "classical" / f"fold_{fold:02d}" / "fit.json").exists() for fold in range(15)):
        raise ValueError("all classical selections must be complete before Test")
    arrays = _arrays(parent["output_dir"])
    pieces = []
    for fold_id in range(15):
        data = dict(np.load(old / "prepared" / f"fold_{fold_id:02d}.npz"))
        fold = json.loads((old / "prepared/folds.json").read_text())[fold_id]
        classical = json.loads((out / "classical" / f"fold_{fold_id:02d}" / "fit.json").read_text())
        if classical["fold"] != fold_id or classical["preregistration_sha256"] != _sha(out / "preregistration.json"):
            raise ValueError("classical fit identity mismatch")
        for name, expected in classical["artifact_sha256"].items():
            if _sha(out / "classical" / f"fold_{fold_id:02d}" / name) != expected: raise ValueError("classical artifact changed")
        for split in ("val", "test"):
            ids = data[split]
            for name, (ret, var) in _classical_predictions(cfg, parent, arrays, data, classical, ids).items():
                pieces.append(prediction_frame(arrays, ids, fold_id, name, 0, split, ret, var))
        for seed in cfg["seeds"]:
            for split in ("val", "test"):
                previous = pd.read_parquet(old / "runs" / f"fold_{fold_id:02d}_D_s{seed}" / f"{split}_predictions.parquet").query("horizon==16").copy()
                previous["group"] = "wm_700"
                pieces.append(previous[["timestamp", "fold", "group", "seed", "horizon", "split", "actual_return", "actual_variance", "pred_return", "pred_variance"]])
            for name in cfg["neural"]["models"]:
                task = {"model": name, "fold": fold_id, "seed": seed, "job": f"fold_{fold_id:02d}_{name}_s{seed}"}
                directory = out / "neural" / task["job"]
                fit = json.loads((directory / "fit.json").read_text())
                if _sha(directory / "research_model.pt") != fit["checkpoint_sha256"]: raise ValueError("neural checkpoint changed")
                checkpoint = torch.load(directory / "research_model.pt", map_location="cpu", weights_only=False)
                if checkpoint["preregistration_sha256"] != _sha(out / "preregistration.json"): raise ValueError("checkpoint registration changed")
                if checkpoint["task"] != task or any(fit[key] != task[key] for key in task): raise ValueError("neural task mismatch")
                model = model_for(task, parent, fold).to(cfg["device"])
                model.load_state_dict(checkpoint["model"])
                for split in ("val", "test"):
                    ids = data[split]
                    prediction = predict_two(model, arrays, ids, data, parent)
                    pieces.append(prediction_frame(arrays, ids, fold_id, name, seed, split, prediction["return"][:,1], prediction["variance"][:,1]))
                del model
                torch.mps.empty_cache()
        print(f"Evaluated fold {fold_id}", flush=True)
    predictions = pd.concat(pieces, ignore_index=True)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    metrics = []
    for (name,split), frame in predictions.groupby(["group","split"]):
        metrics.append({"model": name, "split": split, "n": len(frame), **score_frame(frame)})
    pd.DataFrame(metrics).to_csv(out / "aggregate_metrics.csv", index=False)
    fold_metrics = []
    for (name,split,fold,seed), frame in predictions.groupby(["group","split","fold","seed"]):
        fold_metrics.append({"model": name, "split": split, "fold": fold, "seed": seed, "n": len(frame), **score_frame(frame)})
    pd.DataFrame(fold_metrics).to_csv(out / "fold_metrics.csv", index=False)
    test = predictions.loc[predictions.split == "test"]
    inference = []; pvalues = {}
    for first,second,target in cfg["statistics"]["comparisons"]:
        selected = []
        for name in (first,second):
            frame = test.loc[test.group==name].copy()
            if (frame.seed==0).all():
                frame = pd.concat([frame.assign(seed=seed) for seed in cfg["seeds"]],ignore_index=True)
            # The unused head is a constant independent of outcomes and is
            # explicitly excluded from this target-specific inference record.
            if target == "return": frame["pred_variance"] = 1.
            else: frame["pred_return"] = 0.
            selected.append(frame)
        bootstrap = paired_block_bootstrap(pd.concat(selected,ignore_index=True), first,second,horizon=16,
            block_days=cfg["statistics"]["block_days"], replicates=cfg["statistics"]["bootstrap_replicates"], random_seed=cfg["statistics"]["bootstrap_seed"])
        for row in bootstrap:
            row["target"] = target
            wanted = ("return_rmse_improvement", "return_rmse_improvement_pct", "return_oos_r2_difference", "return_mse_improvement") if target == "return" else ("variance_qlike_improvement",)
            row["metrics"] = {key:value for key,value in row["metrics"].items() if key in wanted}
            if row["block_days"] == cfg["statistics"]["primary_block_days"]:
                pvalues[f"{second}-{first}:{target}"] = row["metrics"]["return_mse_improvement" if target=="return" else "variance_qlike_improvement"]["p_value"]
        inference.extend(bootstrap)
    histories = [json.loads((out / "neural" / task["job"] / "fit.json").read_text()) for task in frozen["tasks"]]
    summary = {"completed_neural_runs": len(histories), "classical_folds":15,"aggregate_metrics":metrics,
        "inference":inference,"holm_adjusted_p":holm_adjust(pvalues),
        "convergence":{name:{"runs":sum(r['model']==name for r in histories),"max_steps_count":sum(r['model']==name and r['right_censored'] for r in histories),
            "patience_count":sum(r['model']==name and not r['right_censored'] for r in histories),
            "median_best_step":float(np.median([r['best_step'] for r in histories if r['model']==name]))} for name in cfg['neural']['models']},
        "preregistration_sha256":_sha(out / "preregistration.json"),"test_selection_allowed":False}
    _json(out / "summary.json",summary)
    print(pd.DataFrame(metrics).query("split=='test'").to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("command",choices=("prepare","run-neural","run-classical","evaluate"))
    parser.add_argument("--config",default=str(DEFAULT_CONFIG))
    args=parser.parse_args();cfg=load_config(args.config)
    {"prepare":prepare,"run-neural":run_neural,"run-classical":run_classical,"evaluate":evaluate}[args.command](cfg)


if __name__ == "__main__": main()
