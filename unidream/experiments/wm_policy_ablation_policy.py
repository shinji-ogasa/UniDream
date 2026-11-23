"""Fresh BC -> actual-history PPO fits for the registered A/B/C/D study.

This isolated research entry does not modify or warm-start the v31 mainline.
Individual Test policy inference is gated on the *whole* study fit barrier.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import wm_policy_ablation_data as data
from . import wm_policy_ablation_evaluation as evaluation
from .wm_policy_ablation_models import (PolicyActorCritic, initial_portfolio,
    inventory_features, advance_portfolio, freeze_bc_prior,
    collect_history_rollout, ppo_update)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def train_source_id(fold_data):
    return hashlib.sha256(np.ascontiguousarray(fold_data["train"], dtype=np.int64).tobytes()).hexdigest()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def contiguous_starts(ids, index_ns, horizon):
    """All starts with horizon+1 consecutive eligible Train observations."""
    ids = np.asarray(ids, np.int64)
    if horizon < 1:
        raise ValueError("positive rollout horizon required")
    if not len(ids):
        return np.empty(0, np.int64)
    boundaries = np.flatnonzero(np.r_[True, np.diff(np.asarray(index_ns)[ids]) != data.BAR_NS, True])
    starts = [np.arange(a, b - horizon, dtype=np.int64)
              for a, b in zip(boundaries[:-1], boundaries[1:]) if b - a > horizon]
    return np.concatenate(starts) if starts else np.empty(0, np.int64)


def selection_panel(ids, maximum):
    """Deterministic prefix of a Val-only month, with gaps kept explicit."""
    result = np.asarray(ids, np.int64)[:int(maximum)]
    if not len(result):
        raise ValueError("nonempty validation selection panel required")
    return result


def _latent_data(cfg, fold, seed, arm, test=False):
    latent_dim = int(cfg["wm_model"]["n_categoricals"]) * int(cfg["wm_model"]["n_classes"]) + int(cfg["wm_model"]["d_model"])
    if arm not in ("C", "D"):
        return None, None, latent_dim
    root = Path(cfg["output_dir"]) / "world_models" / f"fold_{fold:02d}_s{seed}"
    fit = json.loads((root / "fit.json").read_text())
    registration_sha = sha(Path(cfg["output_dir"]) / "preregistration.json")
    if not fit.get("complete") or fit.get("registration_sha256") != registration_sha:
        raise ValueError("selected, frozen upstream WM required")
    path = root / ("latent_test.npy" if test else "latent_train_val.npy")
    expected = fit["artifact_sha256"]
    if sha(root / "latent_norm.npz") != expected["latent_norm.npz"]:
        raise ValueError("frozen Train-only latent normalizer changed")
    if not test and sha(path) != expected["latent_train_val.npy"]:
        raise ValueError("frozen causal WM state cache changed")
    if test:
        marker = json.loads((root / "test_latent.json").read_text())
        if (marker.get("registration_sha256") != registration_sha or marker.get("latent_sha256") != sha(path)
                or marker.get("barrier_sha256") != sha(Path(cfg["output_dir"]) / "test_barrier.json")):
            raise ValueError("frozen report-only WM Test states changed")
    with np.load(root / "latent_norm.npz", allow_pickle=False) as obj:
        norm = {key: obj[key] for key in obj.files}
    return np.load(path, mmap_mode="r", allow_pickle=False), norm, latent_dim


def _inputs(arrays, ids, fold_data, cfg, arm, latents, latent_norm, latent_dim):
    ids = np.asarray(ids, np.int64)
    clip = float(cfg["policy"]["normalizer_clip"])
    x = data.x_windows(arrays, ids, fold_data, cfg["policy"]["context"], clip)
    f = (data.f_values(arrays, ids, fold_data, clip) if arm in ("B", "D")
         else np.zeros((len(ids), arrays["f"].shape[1]), np.float32))
    if arm in ("C", "D"):
        z = np.clip((np.asarray(latents[ids]) - latent_norm["mean"]) / latent_norm["std"], -clip, clip).astype(np.float32)
        if not np.isfinite(z).all():
            raise ValueError("selected frozen WM states must be finite for every requested row")
    else:
        z = np.zeros((len(ids), latent_dim), np.float32)
    return x, f, z


def _tensors(values, device):
    return tuple(torch.from_numpy(np.asarray(x, np.float32)).to(device) for x in values)


@torch.no_grad()
def predict_requests(model, arrays, ids, fold_data, cfg, latents=None, latent_norm=None):
    """Encode causal windows in batches; sequential own-inventory heads on CPU.

    CPU heads avoid one GPU synchronization per historical observation. This
    is numerically the same selected actor, never a fitted surrogate policy.
    """
    ids = np.asarray(ids, np.int64)
    if not len(ids):
        raise ValueError("nonempty chronological policy panel required")
    model.eval()
    device = next(model.parameters()).device
    contexts = []
    for start in range(0, len(ids), 256):
        panel = ids[start:start + 256]
        inputs = _inputs(arrays, panel, fold_data, cfg, model.arm, latents, latent_norm, model.latent_dim)
        contexts.append(model.encode_context(*_tensors(inputs, device)).cpu())
    encoded = torch.cat(contexts)
    local = copy.deepcopy(model).cpu().eval()
    times = np.asarray(arrays["index_ns"])[ids]
    reset = np.r_[True, np.diff(times) != data.BAR_NS]
    state = initial_portfolio(1, "cpu", cfg, dtype=torch.float64)
    requests = np.empty(len(ids), np.float64)
    # Account state stays float64, while neural input matches float32 weights.
    returns = np.asarray(arrays["bar_log_return"])[ids]
    for row in range(len(ids)):
        if reset[row]:
            state = initial_portfolio(1, "cpu", cfg, dtype=torch.float64)
        action = local.act_from_context(encoded[row:row+1], inventory_features(state).float(), deterministic=True)
        requested = action["request"].double()
        requests[row] = float(requested[0])
        state, _ = advance_portfolio(state, requested, torch.as_tensor(returns[row:row+1]), cfg)
    return requests


def ledger_frame(arrays, ids, requests, fold, seed, arm, stage, split):
    return pd.DataFrame({"fold": fold, "seed": seed, "arm": arm, "stage": stage,
        "split": split, "decision_id": np.asarray(arrays["decision_id"])[ids],
        "timestamp": pd.to_datetime(np.asarray(arrays["index_ns"])[ids], utc=True),
        "bar_log_return": np.asarray(arrays["bar_log_return"])[ids], "requested_position": requests})


def validation_score(model, arrays, ids, fold_data, cfg, latent, norm):
    requests = predict_requests(model, arrays, ids, fold_data, cfg, latent, norm)
    executed = evaluation.execute_positions(np.asarray(arrays["bar_log_return"])[ids], requests,
        pd.to_datetime(np.asarray(arrays["index_ns"])[ids], utc=True), cfg=cfg)
    metrics = evaluation.score_path(executed, cfg)
    score = float(metrics["reward_total"])
    if not np.isfinite(score):
        raise RuntimeError("nonfinite validation objective")
    return score, metrics


def _save_checkpoint(path, model, registration_sha, step, stage):
    temporary = Path(str(path) + ".tmp")
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metadata": model.checkpoint_metadata(), "registration_sha256": registration_sha,
        "stage": stage, "selected_step": step}, temporary)
    temporary.replace(path)


def _load_checkpoint(path, model, registration_sha):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if obj["registration_sha256"] != registration_sha:
        raise ValueError("policy checkpoint belongs to another registration")
    model.load_state_dict(obj["state_dict"])
    return obj


def _optimizer(model, cfg):
    critic = list(model.critic_head.parameters())
    critic_ids = {id(p) for p in critic}
    actor = [p for p in model.parameters() if id(p) not in critic_ids]
    if len(actor) + len(critic) != len(list(model.parameters())):
        raise ValueError("PPO optimizer groups overlap")
    return torch.optim.AdamW([
        {"params": actor, "lr": cfg["rl_training"]["learning_rate"]},
        {"params": critic, "lr": cfg["rl_training"]["critic_learning_rate"]}], weight_decay=0.)


def _complete_fit(path, registration_sha):
    if not path.exists():
        return None
    obj = json.loads(path.read_text())
    if not obj.get("complete") or obj.get("registration_sha256") != registration_sha:
        raise ValueError("incomplete or foreign policy fit marker")
    for filename, expected in obj["artifact_sha256"].items():
        if sha(path.parent / filename) != expected:
            raise ValueError("frozen policy fit artifact changed")
    return obj


def train_policy(cfg, fold, seed, arm, registration_sha):
    root = Path(cfg["output_dir"]) / "neural" / f"fold_{fold:02d}_{arm}_s{seed}"
    root.mkdir(parents=True, exist_ok=True)
    prior_fit = _complete_fit(root / "fit.json", registration_sha)
    if prior_fit:
        return prior_fit
    arrays, fold_data = data.load_arrays(cfg), data.fold_arrays(cfg, fold)
    latent, latent_norm, latent_dim = _latent_data(cfg, fold, seed, arm)
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    device = cfg["device"]
    model = PolicyActorCritic(arrays["x_grid"].shape[1], arrays["f"].shape[1], latent_dim, cfg, arm).to(device)
    initial_state_sha = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    train = fold_data["train"]
    teacher_ids = train[np.asarray(arrays["teacher_valid"])[train]]
    if not len(teacher_ids):
        raise ValueError("no purged Train supervision for BC")
    settings = cfg["bc_training"]
    panel = selection_panel(fold_data["earlystop"], settings["selection_rows"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    best, best_step, progress_step, history = -np.inf, 0, 0, []
    started = time.monotonic()
    for step in range(1, int(settings["max_steps"]) + 1):
        model.train()
        ids = rng.choice(teacher_ids, int(settings["batch_size"]), replace=True)
        current = rng.uniform(cfg["accounting"]["position_min"], cfg["accounting"]["position_max"], len(ids))
        inventory = np.zeros((len(ids), 6), np.float32)
        inventory[:, 0] = inventory[:, 1] = current
        target = evaluation.teacher_target(np.asarray(arrays["future_path_log_returns"])[ids], current, cfg)
        x, f, z = _tensors(_inputs(arrays, ids, fold_data, cfg, arm, latent, latent_norm, latent_dim), device)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.bc_loss(x, f, z, torch.from_numpy(inventory).to(device), torch.from_numpy(np.asarray(target, np.float32)).to(device))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite BC loss")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
        coverage = model.state_gradient_coverage() if step == 1 else None
        if coverage is not None:
            required = ["history", "policy"] + (["forecast"] if arm in ("B", "D") else []) + (["latent"] if arm in ("C", "D") else [])
            if not all(coverage[k] for k in required):
                raise RuntimeError("missing BC active observation gradients")
            bc_coverage = coverage
        optimizer.step()
        if step % int(settings["validation_every"]) == 0 or step == int(settings["max_steps"]):
            score, metrics = validation_score(model, arrays, panel, fold_data, cfg, latent, latent_norm)
            improved = score > best
            substantial = score > best + float(settings["relative_improvement"]) * max(abs(best) if np.isfinite(best) else 1., 1e-6)
            if improved:
                best, best_step = score, step
                _save_checkpoint(root / "bc.pt", model, registration_sha, step, "bc")
            if substantial:
                progress_step = step
            record = {"stage": "bc", "step": step, "train_loss": float(loss.detach().cpu()),
                "gradient_norm": float(gradient.detach().cpu()), "val_reward": score, "best_step": best_step,
                "val_metrics": metrics, "elapsed_seconds": time.monotonic() - started}
            history.append(record)
            write_json(root / "progress.json", record)
            print(json.dumps({"job": root.name, **record}), flush=True)
            if step >= settings["min_steps"] and step - progress_step >= settings["patience_steps"]:
                break
    bc_steps = step
    _load_checkpoint(root / "bc.pt", model, registration_sha)
    bc_selection = {"best_step": best_step, "steps": bc_steps, "val_reward": best, "gradient_coverage": bc_coverage}
    prior = freeze_bc_prior(model)
    optimizer = _optimizer(model, cfg)
    settings = cfg["rl_training"]
    panel = selection_panel(fold_data["earlystop"], settings["selection_rows"])
    starts = contiguous_starts(train, arrays["index_ns"], int(settings["rollout_horizon"]))
    if not len(starts):
        raise ValueError("no contiguous Train-only market rollouts available")
    best, best_step, progress_step = -np.inf, 0, 0
    rl_pg_steps, changed_steps, transitions = 0, 0, 0
    for update in range(1, int(settings["max_updates"]) + 1):
        horizon, batch = int(settings["rollout_horizon"]), int(settings["rollout_batch_size"])
        offsets = rng.choice(starts, batch, replace=True)
        ids = train[offsets[:, None] + np.arange(horizon + 1)[None, :]]
        flat = ids.ravel()
        values = _inputs(arrays, flat, fold_data, cfg, arm, latent, latent_norm, latent_dim)
        x, f, z = (item.reshape((batch, horizon + 1) + item.shape[1:]) for item in _tensors(values, device))
        bar_returns = torch.from_numpy(np.asarray(arrays["bar_log_return"])[ids[:, :-1]].astype(np.float32)).to(device)
        rollout = collect_history_rollout(model, x, f, z, bar_returns, cfg)
        diagnostics = ppo_update(model, prior, optimizer, rollout, cfg)
        if not all(np.isfinite(value) for value in diagnostics.values() if isinstance(value, (float, int))):
            raise RuntimeError("nonfinite actual PPO diagnostics")
        rl_pg_steps += int(diagnostics["real_policy_gradient_steps"])
        changed_steps += int(diagnostics["changed_policy_steps"])
        transitions += int(diagnostics["transitions"])
        if update == 1:
            rl_coverage = model.state_gradient_coverage()
            if not (rl_coverage["critic"] and diagnostics["real_policy_gradient_steps"] > 0 and diagnostics["changed_policy_steps"] > 0):
                raise RuntimeError("actual policy-gradient and critic updates required")
            write_json(root / "first_rl_update.json", {"update": update, "diagnostics": diagnostics,
                "gradient_coverage": rl_coverage, "registration_sha256": registration_sha})
        if update % int(settings["validation_every"]) == 0 or update == int(settings["max_updates"]):
            score, metrics = validation_score(model, arrays, panel, fold_data, cfg, latent, latent_norm)
            substantial = score > best + float(settings["relative_improvement"]) * max(abs(best) if np.isfinite(best) else 1., 1e-6)
            if score > best:
                best, best_step = score, update
                _save_checkpoint(root / "rl.pt", model, registration_sha, update, "rl")
            if substantial:
                progress_step = update
            record = {"stage": "rl", "update": update, "val_reward": score, "best_update": best_step,
                "val_metrics": metrics, "diagnostics": diagnostics, "transitions_total": transitions,
                "elapsed_seconds": time.monotonic() - started}
            history.append(record)
            write_json(root / "progress.json", record)
            print(json.dumps({"job": root.name, **record}), flush=True)
            if update >= settings["min_updates"] and update - progress_step >= settings["patience_updates"]:
                break
    if best_step < 1 or rl_pg_steps == 0 or changed_steps == 0:
        raise RuntimeError("RL selection must represent a genuinely trained policy")
    rl_selection = {"best_update": best_step, "updates": update, "val_reward": best,
        "real_policy_gradient_steps": rl_pg_steps, "changed_policy_steps": changed_steps,
        "historical_transitions": transitions, "gradient_coverage": rl_coverage,
        "bc_fallback_allowed": False, "rollout_source": "actual_historical_Train_segments"}
    panels = []
    val_ids = fold_data["val"]
    for stage in ("bc", "rl"):
        _load_checkpoint(root / f"{stage}.pt", model, registration_sha)
        requests = predict_requests(model, arrays, val_ids, fold_data, cfg, latent, latent_norm)
        panels.append(ledger_frame(arrays, val_ids, requests, fold, seed, arm, stage, "val"))
    pd.concat(panels, ignore_index=True).to_parquet(root / "validation_predictions.parquet", index=False)
    write_json(root / "training_history.json", history)
    fit = {"complete": True, "registration_sha256": registration_sha, "fold": fold, "seed": seed, "arm": arm,
        "normalizer_source_id": train_source_id(fold_data), "normalizer_fit_split": "train",
        "fold_arrays_sha256": sha(Path(cfg["output_dir"]) / "prepared" / f"fold_{fold:02d}.npz"),
        "bc": bc_selection, "rl": rl_selection, "initial_actor_state_sha256": initial_state_sha,
        "metadata": model.checkpoint_metadata(), "test_used": False, "warm_start": False,
        "wm_fit_sha256": sha(Path(cfg["output_dir"]) / "world_models" / f"fold_{fold:02d}_s{seed}" / "fit.json") if arm in ("C", "D") else None,
        "artifact_sha256": {name: sha(root / name) for name in ("bc.pt", "rl.pt", "validation_predictions.parquet", "training_history.json")},
        "elapsed_seconds": time.monotonic() - started}
    write_json(root / "fit.json", fit)
    return fit


def evaluate_policy(cfg, fold, seed, arm, registration_sha):
    from .wm_policy_ablation_world import encode_test_latents
    from .wm_policy_ablation_benchmark import verify_test_barrier
    base = Path(cfg["output_dir"])
    barrier = verify_test_barrier(cfg)
    if barrier["registration_sha256"] != registration_sha or sha(base / "validation_choices.json") != barrier["validation_choices_sha256"]:
        raise ValueError("complete frozen global barrier required for Test policy inference")
    source = base / "neural" / f"fold_{fold:02d}_{arm}_s{seed}"
    fit = _complete_fit(source / "fit.json", registration_sha)
    if fit is None or barrier["frozen_fit_sha256"].get(str((source / "fit.json").relative_to(base))) != sha(source / "fit.json"):
        raise ValueError("selected policy fit was not frozen at the global Test barrier")
    root = base / "evaluations" / source.name
    root.mkdir(parents=True, exist_ok=True)
    if (root / "evaluation.json").exists():
        obj = json.loads((root / "evaluation.json").read_text())
        if obj["barrier_sha256"] != sha(base / "test_barrier.json") or obj["prediction_sha256"] != sha(root / "predictions.parquet"):
            raise ValueError("Test evaluation changed")
        return obj
    if arm in ("C", "D"):
        encode_test_latents(cfg, fold, seed, registration_sha)
    arrays, fold_data = data.load_arrays(cfg), data.fold_arrays(cfg, fold)
    latent, norm, latent_dim = _latent_data(cfg, fold, seed, arm, test=True)
    model = PolicyActorCritic(arrays["x_grid"].shape[1], arrays["f"].shape[1], latent_dim, cfg, arm).to(cfg["device"])
    ids, panels = fold_data["test"], []
    for stage in ("bc", "rl"):
        _load_checkpoint(source / f"{stage}.pt", model, registration_sha)
        requests = predict_requests(model, arrays, ids, fold_data, cfg, latent, norm)
        panels.append(ledger_frame(arrays, ids, requests, fold, seed, arm, stage, "test"))
    pd.concat(panels, ignore_index=True).to_parquet(root / "predictions.parquet", index=False)
    obj = {"complete": True, "test_report_only": True, "fold": fold, "seed": seed, "arm": arm,
        "barrier_sha256": sha(base / "test_barrier.json"), "prediction_sha256": sha(root / "predictions.parquet"),
        "n_rows": len(ids) * 2, "fit_sha256": sha(source / "fit.json")}
    write_json(root / "evaluation.json", obj)
    return obj
