"""Independent-seed WM/BC/PPO fits and registered continuation adapters.

The original research kernels are imported read-only.  New seeds train their
own WM, BC and initial RL before continuation, using exactly the old budgets.
No report-only Test inference can run before the new complete global barrier.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from . import wm_policy_ablation_data as data
from . import wm_policy_ablation_evaluation as evaluation
from . import wm_policy_ablation_policy as original
from . import wm_policy_ablation_world as world
from . import wm_rl_release_policy as release
from .wm_policy_ablation_models import PolicyActorCritic, PolicyAblationWM
from .wm_policy_ablation_models import freeze_bc_prior, collect_history_rollout, ppo_update

sha, write_json = original.sha, original.write_json


def kernel_config(cfg):
    """Use the registered numerical configuration in a separate writable root."""
    settings = copy.deepcopy(cfg["kernel_config"])
    if Path(settings["output_dir"]).resolve() != Path(cfg["output_dir"]).resolve():
        raise ValueError("new seed kernels must write only inside the new study")
    if settings["policy"]["wm_update_during_policy_training"] is not False:
        raise ValueError("WM must remain frozen during BC and RL")
    return settings


def _registered(cfg, task, registration_sha):
    path = Path(cfg["output_dir"]) / "preregistration.json"
    if not path.exists() or sha(path) != registration_sha:
        raise ValueError("immutable seed-study registration required")
    registration = json.loads(path.read_text())
    if registration.get("config") != cfg or task not in registration.get("jobs", []):
        raise ValueError("worker configuration/task differs from registration")
    return registration


def condition_config(cfg, condition, update=None):
    settings = kernel_config(cfg)
    maximum = int(cfg["continuation"]["max_updates"])
    if maximum != 200 or list(cfg["continuation"]["selection_updates"]) != [125, 150, 175, 200]:
        raise ValueError("continuation must preserve the original fixed 200-update budget")
    update = maximum if update is None else update
    if isinstance(update, bool) or int(update) != update or not 1 <= update <= maximum:
        raise ValueError("continuation update lies outside the registered budget")
    declared = {row["condition"] for row in cfg["condition_definitions"].values()}
    if condition not in declared or condition not in cfg["schedules"]:
        raise ValueError("condition absent from the registered seed-study schedules")
    schedules = cfg["schedules"][condition]
    if (not schedules or schedules[-1]["through_update"] != maximum
            or any(a["through_update"] >= b["through_update"] for a, b in zip(schedules, schedules[1:]))):
        raise ValueError("ordered schedule must exhaust the fixed update budget")
    stage = next(row for row in schedules if update <= row["through_update"])
    settings["accounting"]["position_min"] = float(stage["position_min"])
    settings["rl_training"]["bc_prior_coefficient"] = float(stage["bc_prior_coefficient"])
    if not 0 <= settings["accounting"]["position_min"] <= .5 or not 0 <= settings["rl_training"]["bc_prior_coefficient"] <= .01:
        raise ValueError("only the registered floor/prior relaxation is supported")
    return settings


def _bundle(cfg, task, registration_sha, *, test=False):
    settings = kernel_config(cfg)
    root = Path(cfg["output_dir"])
    source = root / "neural" / f"fold_{task['fold']:02d}_{task['arm']}_s{task['seed']}"
    fit = original._complete_fit(source / "fit.json", registration_sha)
    if (fit is None or fit.get("test_used") is not False or fit.get("warm_start") is not False
            or any(fit.get(key) != task[key] for key in ("fold", "seed", "arm"))):
        raise ValueError("new seed requires its independently trained BC and initial RL")
    if test and task["arm"] == "C":
        _encode_test_latents(cfg, task, registration_sha)
    latent, norm, width = original._latent_data(settings, task["fold"], task["seed"], task["arm"], test=test)
    names = [root / "preregistration.json", root / "prepared/cache_manifest.json",
        root / "prepared" / f"fold_{task['fold']:02d}.npz", source / "fit.json", source / "bc.pt", source / "rl.pt"]
    if task["arm"] == "C":
        directory = root / "world_models" / f"fold_{task['fold']:02d}_s{task['seed']}"
        world._check_fit(directory, registration_sha)
        names += [directory / name for name in ("fit.json", "model.pt", "latent_norm.npz",
            "latent_test.npy" if test else "latent_train_val.npy")]
        if test:
            names.append(directory / "test_latent.json")
    return {"cfg": settings, "arrays": data.load_arrays(settings), "fold": data.fold_arrays(settings, task["fold"]),
        "latent": latent, "latent_norm": norm, "latent_dim": width, "source_directory": source,
        "source_fit": fit, "source_registration_sha256": registration_sha,
        "source_artifact_sha256": {str(path.resolve()): sha(path) for path in names}}


def _model(bundle, task, device):
    return PolicyActorCritic(bundle["arrays"]["x_grid"].shape[1], bundle["arrays"]["f"].shape[1],
        bundle["latent_dim"], bundle["cfg"], task["arm"]).to(device)


def worker(cfg, task, registration_sha):
    """One scheduler task; only Train/Val kernels are reachable here."""
    _registered(cfg, task, registration_sha)
    settings = kernel_config(cfg)
    torch.set_num_threads(int(cfg.get("runtime", {}).get("cpu_threads", 1)))
    if task["seed"] not in cfg["new_seeds"]:
        raise ValueError("old seeds must be reused read-only, never retrained")
    if task["kind"] == "wm":
        return world.train_wm(settings, task["fold"], task["seed"], registration_sha)
    if task["kind"] == "policy":
        if task["arm"] not in ("A", "C"):
            raise ValueError("seed study only trains registered A/C arms")
        return original.train_policy(settings, task["fold"], task["seed"], task["arm"], registration_sha)
    if task["kind"] == "continuation":
        return train_continuation(cfg, task, registration_sha)
    raise ValueError("unknown seed-study training task")


def train_continuation(cfg, task, registration_sha):
    _registered(cfg, task, registration_sha)
    settings = kernel_config(cfg)
    directory = Path(cfg["output_dir"]) / "neural" / task["job"]
    directory.mkdir(parents=True, exist_ok=True)
    prior_fit = original._complete_fit(directory / "fit.json", registration_sha)
    if prior_fit:
        release._verify_source_digests(prior_fit["source_artifact_sha256"])
        return prior_fit
    started = time.monotonic()
    bundle = _bundle(cfg, task, registration_sha)
    arrays, fold = bundle["arrays"], bundle["fold"]
    device = settings["device"]
    random_seed = int(np.random.SeedSequence([int(task["seed"]), int(task["fold"]), ord(task["arm"]), 20261112]).generate_state(1)[0])
    original.seed_everything(random_seed)
    rng = np.random.default_rng(random_seed)
    model = _model(bundle, task, device)
    original._load_checkpoint(bundle["source_directory"] / "rl.pt", model, registration_sha)
    initial_sha = release._state_sha(model)
    prior_model = _model(bundle, task, device)
    original._load_checkpoint(bundle["source_directory"] / "bc.pt", prior_model, registration_sha)
    prior = freeze_bc_prior(prior_model)
    prior_sha = release._state_sha(prior)
    optimizer = original._optimizer(model, settings)
    train = np.asarray(fold["train"], np.int64)
    meta = next(row for row in data.policy_folds(settings) if row["fold"] == int(task["fold"]))
    times = np.asarray(arrays["index_ns"])[train]
    if (not len(train) or np.any(times < pd.Timestamp(meta["train_start"]).value)
            or np.any(times + int(settings["policy"]["purge_bars"]) * data.BAR_NS >= pd.Timestamp(meta["train_end"]).value)):
        raise ValueError("continuation requires strictly purged Train-only market returns")
    horizon, batch = int(settings["rl_training"]["rollout_horizon"]), int(settings["rl_training"]["rollout_batch_size"])
    starts = original.contiguous_starts(train, arrays["index_ns"], horizon)
    if not len(starts):
        raise ValueError("continuation requires contiguous Train rollouts")
    panel = original.selection_panel(fold["earlystop"], cfg["continuation"]["selection_rows"])
    final_cfg = condition_config(cfg, task["condition"])
    candidates = list(cfg["continuation"]["selection_updates"])
    maximum = int(cfg["continuation"]["max_updates"])
    for update in candidates:
        selected_cfg = condition_config(cfg, task["condition"], update)
        if selected_cfg["accounting"] != final_cfg["accounting"] or selected_cfg["rl_training"] != final_cfg["rl_training"]:
            raise ValueError("Val selection must begin after the final release setting")
    best, selected, selected_metrics = -np.inf, None, None
    history, market_digest = [], hashlib.sha256()
    transitions = policy_updates = changed_updates = 0
    for update in range(1, maximum + 1):
        active_cfg = condition_config(cfg, task["condition"], update)
        release._apply(model, active_cfg)
        offsets = rng.choice(starts, batch, replace=True)
        ids = train[offsets[:, None] + np.arange(horizon + 1)[None, :]]
        market_digest.update(np.ascontiguousarray(ids[:, 0], np.int64).tobytes())
        values = original._inputs(arrays, ids.ravel(), fold, active_cfg, task["arm"], bundle["latent"], bundle["latent_norm"], bundle["latent_dim"])
        x, forecasts, latents = (value.reshape((batch, horizon + 1) + value.shape[1:]) for value in original._tensors(values, device))
        returns = torch.from_numpy(np.asarray(arrays["bar_log_return"])[ids[:, :-1]].astype(np.float32)).to(device)
        rollout = collect_history_rollout(model, x, forecasts, latents, returns, active_cfg)
        diagnostics = ppo_update(model, prior, optimizer, rollout, active_cfg)
        if not all(np.isfinite(value) for value in diagnostics.values() if isinstance(value, (float, int))):
            raise RuntimeError("nonfinite continuation PPO diagnostics")
        if update in (1, maximum):
            coverage = model.state_gradient_coverage()
            required = ["history", "policy", "critic"] + (["latent"] if task["arm"] == "C" else [])
            if not all(coverage[name] for name in required):
                raise RuntimeError("all active policy/critic observation paths require finite gradients")
        transitions += int(diagnostics["transitions"])
        policy_updates += int(diagnostics["real_policy_gradient_steps"] > 0)
        changed_updates += int(diagnostics["changed_policy_steps"] > 0)
        record = {"update": update, "position_min": model.position_min,
            "bc_prior_coefficient": active_cfg["rl_training"]["bc_prior_coefficient"],
            "fresh_on_policy_rollout": True, "diagnostics": diagnostics,
            "historical_transitions_total": transitions, "elapsed_seconds": time.monotonic() - started}
        if update in (1, maximum):
            record["gradient_coverage"] = coverage
        if update in candidates:
            release._apply(model, final_cfg)
            score, metrics = original.validation_score(model, arrays, panel, fold, final_cfg, bundle["latent"], bundle["latent_norm"])
            record.update({"validation_score": score, "validation_metrics": metrics})
            if score > best:
                best, selected, selected_metrics = score, update, metrics
                original._save_checkpoint(directory / "final.pt", model, registration_sha, update, "rl_continuation")
            record["selected_update"] = selected
        history.append(record)
        if update == 1 or update % 25 == 0:
            write_json(directory / "progress.json", {**record, "job": task["job"], "condition": task["condition"]})
            print(json.dumps({"job": task["job"], **record}), flush=True)
    if (selected not in candidates or transitions != maximum * batch * horizon or not policy_updates
            or not changed_updates or release._state_sha(prior) != prior_sha):
        raise RuntimeError("fixed budget, genuine PPO updates and immutable BC prior required")
    original._load_checkpoint(directory / "final.pt", model, registration_sha)
    release._apply(model, final_cfg)
    val_ids = np.asarray(fold["val"], np.int64)
    requests = release._predict(model, bundle, val_ids, final_cfg)
    ledger = release._ledger(bundle, val_ids, requests, task, "val", final_cfg)
    full_metrics = evaluation.score_path(ledger, final_cfg)
    ledger.to_parquet(directory / "validation_predictions.parquet", index=False)
    write_json(directory / "training_history.json", history)
    release._verify_source_digests(bundle["source_artifact_sha256"])
    fit = {"complete": True, "registration_sha256": registration_sha, **task,
        "source_artifact_sha256": bundle["source_artifact_sha256"],
        "source_rl_checkpoint_sha256": sha(bundle["source_directory"] / "rl.pt"),
        "source_bc_checkpoint_sha256": sha(bundle["source_directory"] / "bc.pt"),
        "initial_state_sha256": initial_sha, "initial_actor_state_sha256": initial_sha,
        "immutable_bc_prior_state_sha256": prior_sha, "final_selected_state_sha256": release._state_sha(model),
        "warm_start": True, "source_stage": "selected_rl", "optimizer_moments_inherited": False,
        "optimizer_reset_same_for_all_branches": True, "optimizer_reset_at_stage_boundary": False,
        "bc_prior_immutable": True, "continuation_random_seed": random_seed,
        "market_start_sequence_sha256": market_digest.hexdigest(), "updates": maximum, "selected_update": selected,
        "selection_updates": candidates, "validation_metrics": selected_metrics, "full_validation_metrics": full_metrics,
        "selected_validation_score": best, "test_used": False, "wm_frozen": True,
        "normalizer_source_id": original.train_source_id(fold), "normalizer_fit_split": "train",
        "historical_transitions": transitions, "updates_with_policy_gradient": policy_updates,
        "updates_with_policy_parameter_changes": changed_updates, "final_gradient_coverage": coverage,
        "final_accounting": final_cfg["accounting"], "final_bc_prior_coefficient": final_cfg["rl_training"]["bc_prior_coefficient"],
        "checkpoint_metadata": model.checkpoint_metadata(), "elapsed_seconds": time.monotonic() - started,
        "independent_seed_pipeline": "WM then BC then initial historical PPO then continuation",
        "artifact_sha256": {name: sha(directory / name) for name in ("final.pt", "training_history.json", "validation_predictions.parquet")}}
    write_json(directory / "fit.json", fit)
    return fit


def _test_barrier(cfg, registration_sha):
    from .wm_seed_robustness_benchmark import verify_test_barrier
    barrier = verify_test_barrier(cfg)
    if barrier.get("registration_sha256") != registration_sha:
        raise ValueError("the complete seed-study global barrier must precede Test")
    return barrier


def _encode_test_latents(cfg, task, registration_sha):
    barrier = _test_barrier(cfg, registration_sha)
    root = Path(cfg["output_dir"])
    barrier_path = root / "test_barrier.json"
    settings = kernel_config(cfg)
    directory = root / "world_models" / f"fold_{task['fold']:02d}_s{task['seed']}"
    fit_path = directory / "fit.json"
    if barrier["frozen_fit_sha256"].get(str(fit_path.relative_to(root))) != sha(fit_path):
        raise ValueError("independently trained WM must be in the global selection barrier")
    world._check_fit(directory, registration_sha)
    with (directory / "test_encoding.lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        destination, marker = directory / "latent_test.npy", directory / "test_latent.json"
        if marker.exists():
            obj = json.loads(marker.read_text())
            if (obj.get("registration_sha256") != registration_sha or obj.get("barrier_sha256") != sha(barrier_path)
                    or obj.get("checkpoint_sha256") != sha(directory / "model.pt")
                    or obj.get("latent_sha256") != sha(destination)):
                raise ValueError("new-seed Test latent provenance changed")
            return
        arrays, fold = data.load_arrays(settings), data.fold_arrays(settings, task["fold"])
        checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        if checkpoint.get("registration_sha256") != registration_sha:
            raise ValueError("foreign new-seed WM checkpoint")
        model = PolicyAblationWM(arrays["x_grid"].shape[1], settings["wm_model"], checkpoint["scales"]).to(settings["device"])
        model.load_state_dict(checkpoint["state_dict"])
        model.freeze_for_policy()
        world._write_latents(model, arrays, np.asarray(fold["test"], np.int64), fold, settings, destination)
        write_json(marker, {"complete": True, "registration_sha256": registration_sha,
            "barrier_sha256": sha(barrier_path), "checkpoint_sha256": sha(directory / "model.pt"),
            "latent_sha256": sha(destination), "test_selection_allowed": False})


def evaluate_policy(cfg, task, registration_sha):
    """One selected continuation on the fixed historical Test cohort."""
    _registered(cfg, task, registration_sha)
    barrier = _test_barrier(cfg, registration_sha)
    if task["kind"] != "continuation":
        raise ValueError("only preregistered continuation policies receive Test inference")
    root = Path(cfg["output_dir"])
    directory = root / "neural" / task["job"]
    fit = original._complete_fit(directory / "fit.json", registration_sha)
    if fit is None or barrier["frozen_fit_sha256"].get(str((directory / "fit.json").relative_to(root))) != sha(directory / "fit.json"):
        raise ValueError("selected continuation fit must be frozen before Test")
    output = root / "evaluations" / task["job"]
    output.mkdir(parents=True, exist_ok=True)
    marker, path = output / "evaluation.json", output / "executed_ledger.parquet"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get("barrier_sha256") != sha(root / "test_barrier.json") or previous.get("prediction_sha256") != sha(path):
            raise ValueError("new-seed Test result changed")
        release._verify_source_digests(previous["source_artifact_sha256"])
        return previous
    bundle = _bundle(cfg, task, registration_sha, test=True)
    model = _model(bundle, task, bundle["cfg"]["device"])
    original._load_checkpoint(directory / "final.pt", model, registration_sha)
    settings = condition_config(cfg, task["condition"])
    release._apply(model, settings)
    ids = np.asarray(bundle["fold"]["test"], np.int64)
    requests = release._predict(model, bundle, ids, settings)
    ledger = release._ledger(bundle, ids, requests, task, "test", settings)
    release._verify_source_digests(bundle["source_artifact_sha256"])
    ledger.to_parquet(path, index=False)
    result = {"complete": True, "test_report_only": True, "job": task["job"], "fold": task["fold"],
        "arm": task["arm"], "seed": task["seed"], "condition": task["condition"], "n_rows": len(ids),
        "registration_sha256": registration_sha, "barrier_sha256": sha(root / "test_barrier.json"),
        "fit_sha256": sha(directory / "fit.json"), "prediction_sha256": sha(path),
        "metrics": evaluation.score_path(ledger, settings), "final_accounting": settings["accounting"],
        "final_bc_prior_coefficient": settings["rl_training"]["bc_prior_coefficient"],
        "source_artifact_sha256": bundle["source_artifact_sha256"],
        "artifact_sha256": {"executed_ledger.parquet": sha(path)}}
    write_json(marker, result)
    return result
