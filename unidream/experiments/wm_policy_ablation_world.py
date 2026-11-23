"""Fresh fold-local WM fits and frozen, split-gated policy latent caches."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from . import wm_policy_ablation_data as data
from .wm_full_grid_benchmark import supervised_objective
from .wm_policy_ablation_models import PolicyAblationWM
from .wm_seven_task_models import TASKS


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _registered(cfg, registration_sha):
    root = Path(cfg["output_dir"])
    path = root / "preregistration.json"
    if not path.exists() or _sha(path) != registration_sha:
        raise ValueError("immutable policy registration required before WM data access")
    manifest = root / "prepared/cache_manifest.json"
    if not manifest.exists() or json.loads(manifest.read_text()).get("registration_sha256") != registration_sha:
        raise ValueError("policy input cache is not bound to the supplied registration")
    return root


def _folder(cfg, fold, seed):
    return Path(cfg["output_dir"]) / "world_models" / f"fold_{int(fold):02d}_s{int(seed)}"


def _scales(fold_arrays):
    return {task: np.asarray(fold_arrays["scale_" + task], float).tolist() for task in TASKS}


def training_supervision(arrays, positions, fold_meta):
    """Mask EVERY context token, including context before policy Train start."""
    positions = np.asarray(positions, np.int64)
    lower, upper = pd.Timestamp(fold_meta["train_start"]).value, pd.Timestamp(fold_meta["train_end"]).value
    times = np.asarray(arrays["raw_index_ns"])[positions]
    valid = (np.asarray(arrays["wm_mask_grid"])[positions]
        & (times >= lower) & (times < upper)
        & (np.asarray(arrays["wm_label_end_ns"])[positions] < upper))
    # Token d observes closes through d-1; its actual next market return is
    # close[d]/close[d-1]. The reward target is never read outside Train.
    reward = np.full(positions.shape, np.nan, np.float32)
    reward_valid = (times >= lower) & (times < upper) & (positions > 0)
    selected = positions[reward_valid]
    close = np.asarray(arrays["raw_close"])
    with np.errstate(divide="ignore", invalid="ignore"):
        reward[reward_valid] = np.log(close[selected] / close[selected - 1]).astype(np.float32)
    return valid[..., None], reward


def _predict(model, arrays, ids, fold_arrays, cfg):
    parts = {task: [] for task in TASKS}
    model.eval()
    size = int(cfg["wm_training"].get("evaluation_batch_size", 128))
    with torch.no_grad():
        for start in range(0, len(ids), size):
            rows = np.asarray(ids[start:start + size], np.int64)
            x = data.x_windows(arrays, rows, fold_arrays, context=int(cfg["wm_model"]["context"]),
                clip=float(cfg.get("policy", {}).get("normalizer_clip", 10.)))
            predicted = model.predict_last(torch.from_numpy(x).to(cfg["device"]))
            for task in TASKS:
                parts[task].append(predicted[task].cpu().numpy())
    return {task: np.concatenate(values) for task, values in parts.items()}


def _write_latents(model, arrays, ids, fold_arrays, cfg, destination):
    count, width = len(arrays["decision_id"]), model.policy_latent_dim
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(count, width))
    output[:] = np.nan
    size = int(cfg["wm_training"].get("evaluation_batch_size", 128))
    for start in range(0, len(ids), size):
        rows = np.asarray(ids[start:start + size], np.int64)
        x = data.x_windows(arrays, rows, fold_arrays, context=int(cfg["wm_model"]["context"]),
            clip=float(cfg.get("policy", {}).get("normalizer_clip", 10.)))
        latent = model.encode_policy_state(torch.from_numpy(x).to(cfg["device"])).cpu().numpy()
        if latent.shape != (len(rows), width) or not np.isfinite(latent).all():
            raise ValueError("nonfinite or incompatible frozen policy latent")
        output[rows] = latent
    output.flush()
    del output
    temporary.replace(destination)


def _latent_normalizer(path, train_ids):
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    total = np.zeros(values.shape[1], np.float64)
    squared = np.zeros_like(total)
    count = 0
    for start in range(0, len(train_ids), 1024):
        part = np.asarray(values[train_ids[start:start + 1024]], np.float64)
        if not np.isfinite(part).all():
            raise ValueError("all policy Train rows require finite frozen latents")
        total += part.sum(0)
        squared += np.square(part).sum(0)
        count += len(part)
    if not count:
        raise ValueError("Train-only latent normalization requires observations")
    mean = total / count
    std = np.sqrt(np.maximum(0., squared / count - mean ** 2))
    std = np.where(std >= 1e-8, std, 1.)
    return mean.astype(np.float32), std.astype(np.float32)


def _check_fit(directory, registration_sha):
    fit = json.loads((directory / "fit.json").read_text())
    if fit.get("complete") is not True or fit.get("registration_sha256") != registration_sha:
        raise ValueError("WM fit belongs to a different policy registration")
    for filename, expected in fit["artifact_sha256"].items():
        if _sha(directory / filename) != expected:
            raise ValueError("selected WM checkpoint or frozen latent cache changed")
    normalizer = directory.parent.parent / "prepared" / f"fold_{int(fit['fold']):02d}.npz"
    if _sha(normalizer) != fit["normalizer_train_fold_sha256"]:
        raise ValueError("policy Train normalization changed after the WM fit")
    return fit


def train_wm(cfg, fold, seed, registration_sha):
    """Train a fresh X-only WM and freeze Train/Val latents; never read Test."""
    root = _registered(cfg, registration_sha)
    directory = _folder(cfg, fold, seed)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "fit.json").exists():
        return _check_fit(directory, registration_sha)
    torch.set_num_threads(int(cfg.get("runtime", {}).get("cpu_threads", 1)))
    torch.use_deterministic_algorithms(True)
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    arrays, fold_data = data.load_arrays(cfg), data.fold_arrays(cfg, fold)
    meta = next(row for row in data.policy_folds(cfg) if row["fold"] == int(fold))
    scales = _scales(fold_data)
    model = PolicyAblationWM(arrays["x_grid"].shape[1], cfg["wm_model"], scales).to(cfg["device"])
    settings = cfg["wm_training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings.get("weight_decay", .0001)))
    train_ids = np.asarray(fold_data["wm_train"], np.int64)
    early = np.asarray(fold_data["wm_earlystop"], np.int64)
    rng = np.random.default_rng(int(seed))
    panel = np.sort(rng.choice(early, min(len(early), int(settings.get("selection_rows", 512))), replace=False))
    if not len(train_ids) or not len(panel):
        raise ValueError("nonempty purged WM Train/first-Val-month decisions required")
    started, history = time.monotonic(), []
    best = anchor = float("inf")
    best_step = last_significant = 0
    best_state = None
    coverage = {task: 0 for task in TASKS}
    target_count = 0
    stop_reason = "max_steps"
    context = int(cfg["wm_model"]["context"])
    for step in range(1, int(settings["max_steps"]) + 1):
        model.train()
        rows = rng.choice(train_ids, int(settings["batch_size"]), replace=True)
        x = data.x_windows(arrays, rows, fold_data, context=context,
            clip=float(cfg.get("policy", {}).get("normalizer_clip", 10.)))
        decisions = np.asarray(arrays["decision_id"])[rows]
        positions = decisions[:, None] - context + 1 + np.arange(context)[None, :]
        valid, reward = training_supervision(arrays, positions, meta)
        target_count += int(valid.sum())
        target_grid = np.asarray(arrays["wm_targets_grid"])[positions].astype(np.float32)
        targets = {task: torch.from_numpy(target_grid[..., col:col + 1]).to(cfg["device"])
                   for col, task in enumerate(TASKS)}
        optimizer.zero_grad(set_to_none=True)
        loss, parts = model.loss(torch.from_numpy(x).to(cfg["device"]), torch.from_numpy(x[..., :5].copy()).to(cfg["device"]),
            targets, torch.from_numpy(valid).to(cfg["device"]), torch.from_numpy(reward).to(cfg["device"]))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite policy WM loss")
        loss.backward()
        gradient = model.supervised_output_layer.weight.grad
        if gradient is None:
            raise RuntimeError("missing supervised WM gradients")
        covered = (torch.isfinite(gradient).all(1) & (gradient.abs().sum(1) > 0)).detach().cpu().numpy()
        for task, output_slice in model.task_output_slices.items():
            coverage[task] += int(covered[output_slice].all())
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("gradient_clip", 10.)), error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % int(settings["validation_every"]) == 0:
            progress = {"fold": int(fold), "seed": int(seed), "step": step,
                "elapsed_seconds": time.monotonic() - started, "best_step": best_step,
                "best_score": best if np.isfinite(best) else None}
            _json(directory / "progress.json", progress)
        if step % int(settings["validation_every"]) == 0 or step == int(settings["max_steps"]):
            predicted = _predict(model, arrays, panel, fold_data, cfg)
            raw_ids = np.asarray(arrays["decision_id"])[panel]
            actual = {task: np.asarray(arrays["wm_targets_grid"])[raw_ids, col:col + 1] for col, task in enumerate(TASKS)}
            score = supervised_objective(actual, predicted, "regression", None, scales)
            if not np.isfinite(score["score"]):
                raise RuntimeError("nonfinite policy WM validation objective")
            history.append({"step": step, "selection_score": float(score["score"]), "selection_parts": score["parts"],
                "training_loss": float(loss.detach().cpu()), "training_parts": {key: float(value.cpu()) for key, value in parts.items()},
                "elapsed_seconds": time.monotonic() - started})
            if score["score"] < best:
                best, best_step = float(score["score"]), step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if not np.isfinite(anchor) or score["score"] < anchor - float(settings.get("relative_improvement", .0005)) * abs(anchor):
                anchor, last_significant = float(score["score"]), step
            _json(directory / "progress.json", {**progress, "best_step": best_step, "best_score": best})
            if step >= int(settings["min_steps"]) and step - last_significant >= int(settings["patience_steps"]):
                stop_reason = "validation_patience"
                break
    if best_state is None or not target_count or not all(coverage.values()):
        raise RuntimeError("WM needs a selected checkpoint and all seven supervised gradient paths")
    model.load_state_dict(best_state)
    temporary = directory / "model.pt.tmp"
    torch.save({"state_dict": best_state, "scales": scales, "fold": int(fold), "seed": int(seed),
        "registration_sha256": registration_sha, "model_cfg": cfg["wm_model"]}, temporary)
    temporary.replace(directory / "model.pt")
    model.freeze_for_policy()
    latent_path = directory / "latent_train_val.npy"
    ids = np.unique(np.concatenate((fold_data["train"], fold_data["val"]))).astype(np.int64)
    _write_latents(model, arrays, ids, fold_data, cfg, latent_path)
    mean, std = _latent_normalizer(latent_path, np.asarray(fold_data["train"], np.int64))
    temporary = directory / "latent_norm.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez(stream, mean=mean, std=std)
    temporary.replace(directory / "latent_norm.npz")
    artifact_sha = {name: _sha(directory / name) for name in ("model.pt", "latent_train_val.npy", "latent_norm.npz")}
    fit = {"complete": True, "fold": int(fold), "seed": int(seed), "registration_sha256": registration_sha,
        "model_sha256": artifact_sha["model.pt"], "checkpoint_sha256": artifact_sha["model.pt"], "artifact_sha256": artifact_sha,
        "normalizer_train_fold_sha256": _sha(root / "prepared" / f"fold_{int(fold):02d}.npz"),
        "latent_normalizer_fit_split": "policy Train only", "latent_dim": model.policy_latent_dim,
        "scales": scales, "steps": step, "best_step": best_step, "best_score": best, "stop_reason": stop_reason,
        "max_step_censored": stop_reason == "max_steps", "supervised_gradient_steps": coverage,
        "supervised_target_count": target_count, "history": history, "elapsed_seconds": time.monotonic() - started,
        "parameters": sum(p.numel() for p in model.parameters()), "wm_observation": "X only; shared frozen C/D fit",
        "reconstruction_target": "same normalized observable X token first five channels, symlog decoder target",
        "latent_inference": "deterministic categorical posterior probabilities concatenated with causal Transformer h",
        "auxiliary_reward_target": "log(raw_close[d]/raw_close[d-1]); token d observes through d-1; Train masked",
        "test_read": False, "test_used": False, "latent_file": str(latent_path), "latent_norm_file": str(directory / "latent_norm.npz")}
    _json(directory / "fit.json", fit)
    return fit


def encode_test_latents(cfg, fold, seed, registration_sha):
    """Encode Test observations only after the complete global fit barrier."""
    root = _registered(cfg, registration_sha)
    from .wm_policy_ablation_benchmark import verify_test_barrier
    verify_test_barrier(cfg)
    barrier_path = root / "test_barrier.json"
    if not barrier_path.exists():
        raise ValueError("global policy Test barrier is closed")
    barrier = json.loads(barrier_path.read_text())
    if barrier.get("registration_sha256") != registration_sha or barrier.get("test_selection_allowed") is not False:
        raise ValueError("invalid policy Test barrier registration")
    directory = _folder(cfg, fold, seed)
    fit_path = directory / "fit.json"
    relative = str(fit_path.relative_to(root))
    manifest = barrier.get("frozen_fit_sha256", {})
    if manifest.get(relative) != _sha(fit_path):
        raise ValueError("selected WM fit is absent from the global Test barrier")
    for name, expected in manifest.items():
        if _sha(root / name) != expected:
            raise ValueError("a global frozen fit changed after the Test barrier")
    _check_fit(directory, registration_sha)
    with (directory / "test_encoding.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        output, marker = directory / "latent_test.npy", directory / "test_latent.json"
        if marker.exists():
            previous = json.loads(marker.read_text())
            if (previous.get("registration_sha256") != registration_sha or previous.get("barrier_sha256") != _sha(barrier_path)
                    or previous.get("checkpoint_sha256") != _sha(directory / "model.pt")
                    or previous.get("latent_sha256") != _sha(output)):
                raise ValueError("frozen policy Test latent cache changed")
            return np.load(output, mmap_mode="r", allow_pickle=False)
        arrays, fold_data = data.load_arrays(cfg), data.fold_arrays(cfg, fold)
        checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
        model = PolicyAblationWM(arrays["x_grid"].shape[1], cfg["wm_model"], checkpoint["scales"]).to(cfg["device"])
        model.load_state_dict(checkpoint["state_dict"])
        model.freeze_for_policy()
        _write_latents(model, arrays, np.asarray(fold_data["test"], np.int64), fold_data, cfg, output)
        _json(marker, {"complete": True, "registration_sha256": registration_sha,
            "barrier_sha256": _sha(barrier_path), "checkpoint_sha256": _sha(directory / "model.pt"),
            "latent_sha256": _sha(output), "test_selection_allowed": False})
        return np.load(output, mmap_mode="r", allow_pickle=False)
