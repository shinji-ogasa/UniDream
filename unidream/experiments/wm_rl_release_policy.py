"""Registered continuation PPO with isolated position-floor/prior schedules.

Research-only: immutable prior study checkpoints/normalizers/WM states are read,
never rewritten. Every branch restarts optimizer moments, uses the same seeded
market-start stream, and spends its entire registered update budget. Checkpoint
selection begins only after every schedule reaches its final setting.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from . import wm_policy_ablation_benchmark as source_benchmark
from . import wm_policy_ablation_data as data
from . import wm_policy_ablation_evaluation as evaluation
from . import wm_policy_ablation_policy as original
from .wm_policy_ablation_models import PolicyActorCritic, freeze_bc_prior, collect_history_rollout, ppo_update

sha = original.sha
write_json = original.write_json


@lru_cache(maxsize=8)
def _cached_source(path, checksum):
    return source_benchmark.load_config(path)


def source_config(cfg):
    path = str(Path(cfg["source_config"]).resolve())
    return copy.deepcopy(_cached_source(path, sha(path)))


def condition_cfg(cfg, condition, update=None):
    """Return a copied OLD cache config, changing only floor and BC-KL weight."""
    if condition not in cfg["conditions"] or condition not in cfg["schedules"]:
        raise ValueError("condition must be in the registered release grid")
    maximum = int(cfg["continuation"]["max_updates"])
    if update is None:
        update = maximum
    if isinstance(update, bool) or int(update) != update or not 1 <= update <= maximum:
        raise ValueError("update must lie inside the registered continuation budget")
    stages = cfg["schedules"][condition]
    if not stages or stages[-1]["through_update"] != maximum or any(
            a["through_update"] >= b["through_update"] for a, b in zip(stages[:-1], stages[1:])):
        raise ValueError("release schedules must be ordered and exhaust the exact update budget")
    stage = next(value for value in stages if update <= value["through_update"])
    output = source_config(cfg)
    output["accounting"]["position_min"] = float(stage["position_min"])
    output["rl_training"]["bc_prior_coefficient"] = float(stage["bc_prior_coefficient"])
    if not 0 <= output["accounting"]["position_min"] <= .5 or not 0 <= output["rl_training"]["bc_prior_coefficient"] <= .01:
        raise ValueError("only the declared nonnegative floor/prior relaxations are supported")
    return output


def _registered(cfg, task, registration_sha):
    path = Path(cfg["output_dir"]) / "preregistration.json"
    if not path.exists() or sha(path) != registration_sha:
        raise ValueError("new release registration required before source checkpoints or data reads")
    registration = json.loads(path.read_text())
    if registration.get("config") != cfg or task not in registration.get("jobs", []):
        raise ValueError("release configuration and worker must match the immutable registration")
    return registration


def _source_inputs(cfg, task, *, test=False):
    scfg = source_config(cfg)
    source_benchmark.verify(scfg, cache=True)
    barrier = source_benchmark.verify_test_barrier(scfg)
    root = Path(scfg["output_dir"])
    directory = root / "neural" / f"fold_{task['fold']:02d}_{task['arm']}_s{task['seed']}"
    source_reg_sha = sha(root / "preregistration.json")
    fit = original._complete_fit(directory / "fit.json", source_reg_sha)
    relative = str((directory / "fit.json").relative_to(root))
    if fit is None or barrier["frozen_fit_sha256"].get(relative) != sha(directory / "fit.json"):
        raise ValueError("source RL/BC fit was not frozen in the complete original barrier")
    names = [Path(scfg["config_path"]), root / "preregistration.json", root / "test_barrier.json", root / "prepared/cache_manifest.json",
        directory / "fit.json", directory / "rl.pt", directory / "bc.pt", root / "prepared" / f"fold_{task['fold']:02d}.npz"]
    latent, norm, width = original._latent_data(scfg, task["fold"], task["seed"], task["arm"], test=test)
    if task["arm"] == "C":
        wm = root / "world_models" / f"fold_{task['fold']:02d}_s{task['seed']}"
        names += [wm / "fit.json", wm / "latent_norm.npz", wm / ("latent_test.npy" if test else "latent_train_val.npy")]
        if test:
            names.append(wm / "test_latent.json")
    provenance = {str(path.resolve()): sha(path) for path in names}
    return {"cfg": scfg, "arrays": data.load_arrays(scfg), "fold": data.fold_arrays(scfg, task["fold"]),
        "latent": latent, "latent_norm": norm, "latent_dim": width, "source_directory": directory,
        "source_registration_sha256": source_reg_sha, "source_fit": fit, "source_artifact_sha256": provenance}


def _bind_source(registration, bundle):
    sources = registration["sources"]
    source_root = Path(sources["source_root"])
    if sources["source_config"] != bundle["cfg"]:
        raise ValueError("source input configuration differs from the newly registered source")
    registered = {str((source_root/name).resolve()): digest for name, digest in sources["artifact_sha256"].items()}
    for filename, digest in bundle["source_artifact_sha256"].items():
        if filename in registered and registered[filename] != digest:
            raise ValueError("source checkpoint, fold normalization or registration differs from the release binding")
    for filename in ("fit.json", "rl.pt", "bc.pt"):
        path = str((bundle["source_directory"]/filename).resolve())
        if path not in registered or registered[path] != bundle["source_artifact_sha256"][path]:
            raise ValueError("each inherited source checkpoint must be explicitly registered")


def _model(bundle, task, device):
    arrays = bundle["arrays"]
    return PolicyActorCritic(arrays["x_grid"].shape[1], arrays["f"].shape[1], bundle["latent_dim"],
        bundle["cfg"], task["arm"]).to(device)


def _budget(cfg):
    settings = cfg["continuation"]
    maximum, candidates = int(settings["max_updates"]), list(settings["selection_updates"])
    if maximum != 200 or candidates != [125, 150, 175, 200]:
        raise ValueError("fixed200 updates and common125/150/175/200 Val checkpoints required")
    for condition in cfg["conditions"]:
        final = condition_cfg(cfg, condition)
        for update in candidates:
            candidate = condition_cfg(cfg, condition, update)
            if (candidate["accounting"] != final["accounting"]
                    or candidate["rl_training"] != final["rl_training"]):
                raise ValueError("all Val candidates must be evaluated after the final setting is reached")
    return maximum, candidates


def _apply(model, settings):
    model.position_min = float(settings["accounting"]["position_min"])
    model.position_max = float(settings["accounting"]["position_max"])
    model.max_position_step = float(settings["accounting"]["max_position_step"])


def _state_sha(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _verify_source_digests(provenance):
    for filename, expected in provenance.items():
        if sha(filename) != expected:
            raise ValueError("original source changed during or after continuation training")


def _ledger(bundle, ids, requests, task, split, settings):
    arrays = bundle["arrays"]
    ledger = evaluation.execute_positions(np.asarray(arrays["bar_log_return"])[ids], requests,
        pd.to_datetime(np.asarray(arrays["index_ns"])[ids], utc=True),
        np.full(len(ids), int(task["fold"])), settings)
    ledger["decision_id"] = np.asarray(arrays["decision_id"])[ids]
    ledger["condition"], ledger["arm"], ledger["seed"] = task["condition"], task["arm"], int(task["seed"])
    ledger["split"], ledger["stage"] = split, "rl_continuation"
    return ledger


def _predict(model, bundle, ids, settings):
    return original.predict_requests(model, bundle["arrays"], ids, bundle["fold"], settings,
        bundle["latent"], bundle["latent_norm"])


def train_policy(cfg, task, registration_sha):
    registration = _registered(cfg, task, registration_sha)
    maximum, candidates = _budget(cfg)
    root = Path(cfg["output_dir"]) / "neural" / task["job"]
    root.mkdir(parents=True, exist_ok=True)
    prior_fit = original._complete_fit(root / "fit.json", registration_sha)
    if prior_fit:
        _verify_source_digests(prior_fit["source_artifact_sha256"])
        return prior_fit
    started = time.monotonic()
    bundle = _source_inputs(cfg, task)
    _bind_source(registration, bundle)
    source_cfg, arrays, fold = bundle["cfg"], bundle["arrays"], bundle["fold"]
    device = cfg.get("device", source_cfg["device"])
    torch.set_num_threads(int(cfg.get("runtime", {}).get("cpu_threads", 1)))
    random_seed = int(np.random.SeedSequence([int(task["seed"]), int(task["fold"]), ord(task["arm"]), 20261112]).generate_state(1)[0])
    original.seed_everything(random_seed)
    rng = np.random.default_rng(random_seed)
    model = _model(bundle, task, device)
    original._load_checkpoint(bundle["source_directory"] / "rl.pt", model, bundle["source_registration_sha256"])
    initial_actor_sha = _state_sha(model)
    prior_model = _model(bundle, task, device)
    original._load_checkpoint(bundle["source_directory"] / "bc.pt", prior_model, bundle["source_registration_sha256"])
    prior = freeze_bc_prior(prior_model)
    prior_sha = _state_sha(prior)
    optimizer = original._optimizer(model, source_cfg)
    train = np.asarray(fold["train"], np.int64)
    meta = next(row for row in data.policy_folds(source_cfg) if row["fold"] == int(task["fold"]))
    times = np.asarray(arrays["index_ns"])[train]
    if (not len(train) or np.any(times < pd.Timestamp(meta["train_start"]).value)
            or np.any(times + int(source_cfg["policy"]["purge_bars"]) * data.BAR_NS >= pd.Timestamp(meta["train_end"]).value)):
        raise ValueError("all continuation market returns must lie strictly inside purged historical Train")
    horizon = int(source_cfg["rl_training"]["rollout_horizon"])
    batch = int(source_cfg["rl_training"]["rollout_batch_size"])
    starts = original.contiguous_starts(train, arrays["index_ns"], horizon)
    if not len(starts):
        raise ValueError("nonempty contiguous Train-only continuation rollout support required")
    panel = original.selection_panel(fold["earlystop"], cfg["continuation"]["selection_rows"])
    final_cfg = condition_cfg(cfg, task["condition"])
    best, best_update, selected_metrics = -np.inf, None, None
    history, updates_with_policy_gradient, changed_updates = [], 0, 0
    market_start_digest = hashlib.sha256()
    transition_count = 0
    for update in range(1, maximum + 1):
        settings = condition_cfg(cfg, task["condition"], update)
        _apply(model, settings)
        offsets = rng.choice(starts, batch, replace=True)
        ids = train[offsets[:, None] + np.arange(horizon+1)[None, :]]
        market_start_digest.update(np.ascontiguousarray(ids[:, 0], np.int64).tobytes())
        inputs = original._inputs(arrays, ids.ravel(), fold, settings, task["arm"], bundle["latent"], bundle["latent_norm"], bundle["latent_dim"])
        x, forecasts, latents = (value.reshape((batch, horizon+1)+value.shape[1:])
            for value in original._tensors(inputs, device))
        bar_returns = torch.from_numpy(np.asarray(arrays["bar_log_return"])[ids[:, :-1]].astype(np.float32)).to(device)
        rollout = collect_history_rollout(model, x, forecasts, latents, bar_returns, settings)
        diagnostics = ppo_update(model, prior, optimizer, rollout, settings)
        if not all(np.isfinite(value) for value in diagnostics.values() if isinstance(value, (float, int))):
            raise RuntimeError("nonfinite continuation PPO diagnostics")
        required = ["history", "policy", "critic"] + (["latent"] if task["arm"] == "C" else [])
        if update in (1, maximum):
            coverage = model.state_gradient_coverage()
            if not all(coverage[key] for key in required):
                raise RuntimeError("continuation requires finite active observation/policy/critic gradients")
        updates_with_policy_gradient += int(diagnostics["real_policy_gradient_steps"] > 0)
        changed_updates += int(diagnostics["changed_policy_steps"] > 0)
        transition_count += int(diagnostics["transitions"])
        record = {"update": update, "position_min": model.position_min,
            "bc_prior_coefficient": settings["rl_training"]["bc_prior_coefficient"],
            "fresh_on_policy_rollout": True, "diagnostics": diagnostics,
            "elapsed_seconds": time.monotonic()-started, "historical_transitions_total": transition_count}
        if update in (1, maximum):
            record["gradient_coverage"] = coverage
        if update in candidates:
            _apply(model, final_cfg)
            score, metrics = original.validation_score(model, arrays, panel, fold, final_cfg, bundle["latent"], bundle["latent_norm"])
            record.update({"validation_score": score, "validation_metrics": metrics})
            if score > best:
                best, best_update, selected_metrics = score, update, metrics
                original._save_checkpoint(root / "final.pt", model, registration_sha, update, "rl_continuation")
            record["selected_update"] = best_update
        history.append(record)
        if update == 1 or update % 25 == 0:
            write_json(root / "progress.json", {**record, "job": task["job"], "condition": task["condition"]})
            print(json.dumps({"job": task["job"], **record}), flush=True)
    if (best_update not in candidates or transition_count != maximum*batch*horizon
            or updates_with_policy_gradient == 0 or changed_updates == 0 or _state_sha(prior) != prior_sha):
        raise RuntimeError("fixed-budget PPO, selected checkpoint, genuine gradients or immutable BC prior contract failed")
    original._load_checkpoint(root / "final.pt", model, registration_sha)
    _apply(model, final_cfg)
    val_ids = np.asarray(fold["val"], np.int64)
    val_requests = _predict(model, bundle, val_ids, final_cfg)
    val_ledger = _ledger(bundle, val_ids, val_requests, task, "val", final_cfg)
    full_metrics = evaluation.score_path(val_ledger, final_cfg)
    val_ledger.to_parquet(root / "validation_predictions.parquet", index=False)
    write_json(root / "training_history.json", history)
    _verify_source_digests(bundle["source_artifact_sha256"])
    fit = {"complete": True, "registration_sha256": registration_sha, **task,
        "source_artifact_sha256": bundle["source_artifact_sha256"],
        "source_rl_checkpoint_sha256": sha(bundle["source_directory"] / "rl.pt"),
        "source_bc_checkpoint_sha256": sha(bundle["source_directory"] / "bc.pt"),
        "initial_actor_state_sha256": initial_actor_sha, "initial_state_sha256": initial_actor_sha,
        "immutable_bc_prior_state_sha256": prior_sha,
        "final_selected_state_sha256": _state_sha(model), "warm_start": True, "source_stage": "selected_rl",
        "optimizer_moments_inherited": False, "optimizer_reset_same_for_all_branches": True,
        "optimizer_reset_at_stage_boundary": False, "bc_prior_immutable": True,
        "continuation_random_seed": random_seed, "market_start_sequence_sha256": market_start_digest.hexdigest(),
        "updates": maximum, "selected_update": best_update, "selection_updates": candidates,
        "validation_metrics": selected_metrics, "full_validation_metrics": full_metrics,
        "selected_validation_score": best, "test_used": False, "wm_frozen": True,
        "normalizer_source_id": original.train_source_id(fold), "normalizer_fit_split": "train",
        "historical_transitions": transition_count, "updates_with_policy_gradient": updates_with_policy_gradient,
        "updates_with_policy_parameter_changes": changed_updates, "final_gradient_coverage": coverage,
        "final_accounting": final_cfg["accounting"], "final_bc_prior_coefficient": final_cfg["rl_training"]["bc_prior_coefficient"],
        "checkpoint_metadata": model.checkpoint_metadata(), "elapsed_seconds": time.monotonic()-started,
        "artifact_sha256": {name: sha(root/name) for name in ("final.pt", "training_history.json", "validation_predictions.parquet")}}
    write_json(root / "fit.json", fit)
    return fit


def evaluate_policy(cfg, task, registration_sha):
    registration = _registered(cfg, task, registration_sha)
    from .wm_rl_release_benchmark import verify_test_barrier
    barrier = verify_test_barrier(cfg)
    root = Path(cfg["output_dir"])
    if barrier.get("registration_sha256") != registration_sha:
        raise ValueError("complete new continuation global barrier required before Test")
    directory = root / "neural" / task["job"]
    relative = str((directory/"fit.json").relative_to(root))
    fit = original._complete_fit(directory/"fit.json", registration_sha)
    if fit is None or barrier.get("frozen_fit_sha256", {}).get(relative) != sha(directory/"fit.json"):
        raise ValueError("selected continuation fit must be frozen in the global barrier")
    output = root / "evaluations" / task["job"]
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "evaluation.json"
    barrier_sha = sha(root/"selection_barrier.json")
    if marker.exists():
        existing = json.loads(marker.read_text())
        if existing.get("barrier_sha256") != barrier_sha or existing.get("prediction_sha256") != sha(output/"executed_ledger.parquet"):
            raise ValueError("immutable continuation Test result changed")
        _verify_source_digests(existing["source_artifact_sha256"])
        return existing
    bundle = _source_inputs(cfg, task, test=True)
    _bind_source(registration, bundle)
    model = _model(bundle, task, cfg.get("device", bundle["cfg"]["device"]))
    original._load_checkpoint(directory/"final.pt", model, registration_sha)
    settings = condition_cfg(cfg, task["condition"])
    _apply(model, settings)
    ids = np.asarray(bundle["fold"]["test"], np.int64)
    requests = _predict(model, bundle, ids, settings)
    ledger = _ledger(bundle, ids, requests, task, "test", settings)
    _verify_source_digests(bundle["source_artifact_sha256"])
    ledger.to_parquet(output/"executed_ledger.parquet", index=False)
    result = {"complete": True, "test_report_only": True, "job": task["job"], "condition": task["condition"],
        "arm": task["arm"], "fold": task["fold"], "seed": task["seed"], "n_rows": len(ids),
        "barrier_sha256": barrier_sha, "registration_sha256": registration_sha,
        "fit_sha256": sha(directory/"fit.json"), "prediction_sha256": sha(output/"executed_ledger.parquet"),
        "artifact_sha256": {"executed_ledger.parquet": sha(output/"executed_ledger.parquet")},
        "metrics": evaluation.score_path(ledger, settings),
        "final_accounting": settings["accounting"], "final_bc_prior_coefficient": settings["rl_training"]["bc_prior_coefficient"],
        "source_artifact_sha256": bundle["source_artifact_sha256"]}
    write_json(marker, result)
    return result
