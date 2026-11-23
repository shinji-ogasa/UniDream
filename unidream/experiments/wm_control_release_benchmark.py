"""Registered head-only control-release research; leaves all prior studies intact.

The market encoder is fixed. All Train/Val fits finish before any new Test
context is opened. Reports use one cash/units account across the full calendar,
including input gaps and policy-fold transitions. This is not the v31 entry.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
import yaml

from . import wm_policy_ablation_data as old_data
from . import wm_seed_robustness_benchmark as source_study


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/wm_control_release_20261112.yaml"
SEEDS = [7, 19, 42, 3, 11, 23, 31, 53, 67, 89]
CONDITIONS = ["control", "cap15", "cap20", "speed", "free"]
_WORKER_CFG = None
_WORKER_REGISTRATION = None
_VERIFIED_BARRIERS = set()


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_contract(cfg):
    if (cfg["seeds"] != SEEDS or cfg["folds"] != list(range(6)) or cfg["arms"] != ["A", "C"]
            or cfg["conditions"] != CONDITIONS or cfg.get("test_selection_allowed") is not False
            or cfg.get("historical_seen_test") is not True or cfg.get("independently_unused_holdout") is not False
            or cfg.get("encoder_scope") != "frozen_source_floor_gradual_market_encoder"):
        raise ValueError("fixed10-seed, known-Test, frozen-encoder research grid required")
    t, a, s = cfg["training"], cfg["accounting"], cfg["statistics"]
    if (t["max_updates"] != 200 or t["selection_updates"] != [125, 150, 175, 200]
            or t["early_stopping"] is not False or t["validation_scope"] != "full_three_month_calendar"
            or t["trainable"] != ["policy_head", "critic_head", "log_std"]
            or t["rollout_batch_size"] != 16 or t["rollout_horizon"] != 64):
        raise ValueError("same200-update head-only budget and full-Val selection required")
    if (a["position_min"] != 0 or a["position_max"] != 1.12 or a["max_position_step"] != .08
            or a["initial_position"] != 1 or a["cost_rate"] != .00055 or a["borrow_apr"] != .10
            or a["drawdown_penalty"] != .25 or a["annual_bars"] != 35040
            or a["execution_delay_bars"] != 1 or a["initial_entry_cost"] is not False
            or a["terminal_liquidation"] is not False or a["reset_positions_at_segment_boundary"] is not False
            or a["deadband"] != 0):
        raise ValueError("registered continuous cash/units execution and common objective required")
    definitions = dict(zip(CONDITIONS, [
        dict(position_max=1.12, max_position_step=.08, bc_prior_coefficient=.01),
        dict(position_max=1.5, max_position_step=.08, bc_prior_coefficient=.01),
        dict(position_max=2., max_position_step=.08, bc_prior_coefficient=.01),
        dict(position_max=2., max_position_step=.24, bc_prior_coefficient=.01),
        dict(position_max=2., max_position_step=.24, bc_prior_coefficient=0.)]))
    if cfg["condition_definitions"] != definitions:
        raise ValueError("every declared final condition must remain in the study")
    for name in CONDITIONS:
        stages = cfg["schedules"][name]
        if (not stages or stages[-1]["through_update"] != 200
                or any(x["through_update"] >= y["through_update"] for x, y in zip(stages[:-1], stages[1:]))
                or {k: stages[0][k] for k in definitions[name]} != definitions["control"]
                or {k: stages[-1][k] for k in definitions[name]} != definitions[name]):
            raise ValueError("all branches must start together and reach the declared final setting")
        if name != "control" and stages[0]["through_update"] != 50:
            raise ValueError("all branches share the first50 updates")
        if len(stages) > 1 and stages[-2]["through_update"] > 100:
            raise ValueError("Val selection must occur after the final release")
    if (s["expected_market_rows"] != 52608 or s["expected_available_decisions"] != 50038
            or (s["primary_family_size"], s["secondary_family_size"], s["combined_family_size"]) != (13, 40, 53)
            or s["bootstrap_replicates"] != 10000 or s["block_days"] != [7, 14, 30]):
        raise ValueError("fixed support and all53 registered comparisons required")
    r = cfg["runtime"]
    if (r["cpu_threads"] != 1 or not 1 <= r["fit_workers"] <= 16
            or not 1 <= r["encoder_workers"] <= 8 or not 1 <= r["evaluation_workers"] <= 16):
        raise ValueError("bounded CPU/MPS concurrency required")


def _expected_add(snapshot, path, digest=None):
    path = str(Path(path).resolve())
    value = sha(path) if digest is None else digest
    if path in snapshot and snapshot[path] != value:
        raise ValueError("inconsistent immutable source provenance: " + path)
    snapshot[path] = value


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg["config_path"] = str(path)
    for key in ("output_dir", "source_study_config"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    for key in ("raw_market_path", "raw_ohlc_path", "raw_index_ns_path", "decision_available_path"):
        cfg["data"][key] = str((ROOT / cfg["data"][key]).resolve())
    validate_contract(cfg)
    old = source_study.load_config(cfg["source_study_config"])
    old_root = Path(old["output_dir"])
    completed = json.loads((old_root / "completion.json").read_text())
    if not completed.get("complete") or completed.get("evaluated_continuations") != 180:
        raise ValueError("the complete immutable10-seed predecessor study is required")
    old_reg = json.loads((old_root / "preregistration.json").read_text())
    old_barrier = source_study.verify_test_barrier(old)
    source = dict(old_reg["source_sha256"])
    source.update(old_reg["sources"]["artifact_sha256"])
    source.update(old_reg["sources"]["old_release_source_sha256"])
    source.update(old_reg["sources"]["old_policy_source_sha256"])
    source = {str((ROOT / name).resolve()): digest for name, digest in source.items()}
    for name in ("preregistration.json", "completion.json", "test_barrier.json", "prepared/cache_manifest.json"):
        _expected_add(source, old_root / name)
    _expected_add(source, cfg["source_study_config"])
    _expected_add(source, cfg["data"]["raw_market_path"])
    pairs = {}
    for fold in cfg["folds"]:
        for seed in cfg["seeds"]:
            reused = seed in old["reused_seeds"]
            policy_root = Path(old["source_output_dir"] if reused else old["output_dir"])
            release_root = Path(old["release_output_dir"] if reused else old["output_dir"])
            kernel = copy.deepcopy(old["kernel_config"])
            kernel["output_dir"] = str(policy_root)
            initial_reg = sha(release_root / "preregistration.json")
            policy_reg = sha(policy_root / "preregistration.json")
            for arm in cfg["arms"]:
                name = f"fold_{fold:02d}_{arm}_s{seed}"
                initial = release_root / "neural" / (name + "_floor_gradual")
                bc = policy_root / "neural" / name
                fit = json.loads((initial / "fit.json").read_text())
                old_fit_path = str((initial / "fit.json").resolve())
                expected_fit_digest = (old_barrier["reused_fit_sha256"].get(old_fit_path) if reused
                    else old_barrier["frozen_fit_sha256"].get(str((initial / "fit.json").relative_to(old_root))))
                if (not fit.get("complete") or fit.get("test_used") is not False
                        or fit.get("registration_sha256") != initial_reg
                        or fit.get("selected_update") not in [125, 150, 175, 200]
                        or any(fit.get(k) != v for k, v in (("fold", fold), ("seed", seed), ("arm", arm)))
                        or expected_fit_digest != sha(initial / "fit.json")):
                    raise ValueError("a Train/Val-selected immutable floor-gradual source is required")
                for filename, digest in fit["artifact_sha256"].items():
                    _expected_add(source, initial / filename, digest)
                for filename, digest in fit["source_artifact_sha256"].items():
                    _expected_add(source, filename, digest)
                _expected_add(source, initial / "fit.json")
                _expected_add(source, bc / "bc.pt", fit["source_bc_checkpoint_sha256"])
                _expected_add(source, bc / "fit.json")
                fold_path = policy_root / "prepared" / f"fold_{fold:02d}.npz"
                _expected_add(source, fold_path)
                metadata = json.loads((bc / "fit.json").read_text())
                width = int(kernel["wm_model"]["n_categoricals"]) * int(kernel["wm_model"]["n_classes"]) + int(kernel["wm_model"]["d_model"])
                pair = dict(kernel_config=kernel, fold_path=str(fold_path),
                    initial_checkpoint_path=str(initial / "final.pt"), initial_checkpoint_registration_sha256=initial_reg,
                    initial_fit_path=str(initial / "fit.json"), bc_checkpoint_path=str(bc / "bc.pt"),
                    bc_checkpoint_registration_sha256=policy_reg, latent_dim=width,
                    latent_train_val_path=None, latent_test_path=None, latent_norm_path=None,
                    source_fit_registration_sha256=metadata["registration_sha256"])
                if arm == "C":
                    wm = policy_root / "world_models" / f"fold_{fold:02d}_s{seed}"
                    wm_fit = json.loads((wm / "fit.json").read_text())
                    wm_test = json.loads((wm / "test_latent.json").read_text())
                    for key, filename in (("latent_train_val_path", "latent_train_val.npy"),
                                          ("latent_test_path", "latent_test.npy"), ("latent_norm_path", "latent_norm.npz")):
                        pair[key] = str(wm / filename)
                        digest = wm_test["latent_sha256"] if filename == "latent_test.npy" else wm_fit["artifact_sha256"][filename]
                        _expected_add(source, wm / filename, digest)
                    _expected_add(source, wm / "fit.json")
                    _expected_add(source, wm / "test_latent.json")
                pair["source_artifact_sha256"] = {name: source[name] for name in (
                    pair["initial_checkpoint_path"], pair["initial_fit_path"], pair["bc_checkpoint_path"], str(fold_path))}
                for key in ("latent_train_val_path", "latent_test_path", "latent_norm_path"):
                    if pair[key] is not None:
                        pair["source_artifact_sha256"][pair[key]] = source[pair[key]]
                pairs[name] = pair
    cfg["source_pairs"] = pairs
    cfg["source_artifact_sha256"] = source
    cfg["fold_metadata"] = old_data.policy_folds(old["kernel_config"])
    cfg["source_study_output_dir"] = str(old_root)
    return cfg


def fit_jobs(cfg):
    return [dict(job=f"fold_{f:02d}_{a}_s{s}_{c}", fold=f, arm=a, seed=s, condition=c)
            for f in cfg["folds"] for s in cfg["seeds"] for a in cfg["arms"] for c in cfg["conditions"]]


def context_jobs(cfg):
    return [dict(job=f"fold_{f:02d}_{a}_s{s}", fold=f, arm=a, seed=s)
            for f in cfg["folds"] for s in cfg["seeds"] for a in cfg["arms"]]


def evaluation_jobs(cfg):
    return [dict(job=f"{kind}_{a}_{c}_s{s}", path_id=f"{kind}_{a}_{c}_s{s}", kind=kind,
                 arm=a, condition=c, seed=s)
            for kind in ("trained", "frozen") for a in cfg["arms"] for c in cfg["conditions"] for s in cfg["seeds"]]


def source_hashes(cfg):
    paths = [Path(cfg["config_path"])]
    paths += sorted((ROOT / "unidream/experiments").glob("wm_control_release_*.py"))
    paths += sorted((ROOT / "tests").glob("test_wm_control_release_*.py"))
    if len(paths) != 9:
        raise ValueError("exactly four research modules, four tests and one config must be frozen")
    return {str(p): sha(p) for p in paths}


def verify_sources(cfg):
    for filename, digest in cfg["source_artifact_sha256"].items():
        if sha(filename) != digest:
            raise ValueError("immutable predecessor source changed: " + filename)


def register(cfg):
    root = Path(cfg["output_dir"])
    path = root / "preregistration.json"
    if path.exists():
        value = json.loads(path.read_text())
        if value["config"] != cfg or value["source_sha256"] != source_hashes(cfg):
            raise ValueError("registered implementation/configuration cannot be changed")
        return sha(path)
    verify_sources(cfg)
    value = dict(at=now(), config=cfg, source_sha256=source_hashes(cfg), jobs=fit_jobs(cfg),
        context_jobs=context_jobs(cfg), evaluation_jobs=evaluation_jobs(cfg),
        source_artifact_sha256=cfg["source_artifact_sha256"],
        packages=_packages(), python_version=sys.version,
        selection="Train/PPO and full-Val-only checkpoint selection; global Test barrier",
        old_study_immutable=True, mainline_unchanged=True,
        limitation="known historical Test, inherited saved context availability, fixed-APR borrowing; no exchange liquidation model")
    write_json(path, value)
    return sha(path)


def verify(cfg):
    root = Path(cfg["output_dir"])
    value = json.loads((root / "preregistration.json").read_text())
    if (value["config"] != cfg or value["source_sha256"] != source_hashes(cfg)
            or value.get("packages") != _packages() or value.get("python_version") != sys.version):
        raise ValueError("frozen research implementation/config changed")
    return sha(root / "preregistration.json")


def _packages():
    return {k: importlib.metadata.version(k) for k in ("torch", "numpy", "pandas", "PyYAML", "scipy")}


def prepare_data(cfg, registration_sha):
    if verify(cfg) != registration_sha:
        raise ValueError("registration mismatch")
    root = Path(cfg["output_dir"])
    marker = root / "prepared/manifest.json"
    if marker.exists():
        value = json.loads(marker.read_text())
        if value["registration_sha256"] != registration_sha:
            raise ValueError("prepared data belongs to a different registration")
        for filename, digest in value["artifact_sha256"].items():
            if sha(filename) != digest:
                raise ValueError("prepared raw market/calendar changed")
        return value
    raw = pd.read_parquet(cfg["data"]["raw_market_path"], columns=["spot_open", "spot_high", "spot_low", "spot_close"])
    prices = raw.to_numpy(np.float64)
    times = raw.index.astype("int64").to_numpy()
    if np.any(np.diff(times) != old_data.BAR_NS):
        raise ValueError("full regular15-minute market grid required")
    start, end = (pd.Timestamp(cfg["data"][k]).value for k in ("test_start", "test_end_exclusive"))
    test = (times >= start) & (times < end)
    valid = np.isfinite(prices).all(1) & (prices > 0).all(1)
    valid &= (prices[:, 2] <= np.minimum(prices[:, 0], prices[:, 3])) & (prices[:, 1] >= np.maximum(prices[:, 0], prices[:, 3]))
    if test.sum() != 52608 or not valid[test].all():
        raise ValueError("complete52608-price Test calendar required; never interpolate prices")
    available = np.zeros(len(times), bool)
    for f in cfg["folds"]:
        pair = cfg["source_pairs"][f"fold_{f:02d}_A_s{cfg['seeds'][0]}"]
        arrays = old_data.load_arrays(pair["kernel_config"])
        with np.load(pair["fold_path"], allow_pickle=False) as fold:
            for split in ("train", "val", "test"):
                ids = np.asarray(arrays["decision_id"])[fold[split]]
                if not valid[ids].all() or not np.isfinite(prices[ids - 1, 3]).all():
                    raise ValueError("all sampled Train and full Val/Test prices must be valid")
                if split == "val":
                    available[ids] = True
    available[test] = False
    for f in cfg["folds"]:
        pair = cfg["source_pairs"][f"fold_{f:02d}_A_s{cfg['seeds'][0]}"]
        arrays = old_data.load_arrays(pair["kernel_config"])
        with np.load(pair["fold_path"], allow_pickle=False) as fold:
            available[np.asarray(arrays["decision_id"])[fold["test"]]] = True
    if available[test].sum() != cfg["statistics"]["expected_available_decisions"]:
        raise ValueError("immutable saved decision support changed")
    for key, values in (("raw_ohlc_path", prices), ("raw_index_ns_path", times), ("decision_available_path", available)):
        path = Path(cfg["data"][key]); path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, values, allow_pickle=False)
    value = dict(complete=True, registration_sha256=registration_sha, test_calendar_rows=int(test.sum()),
        test_decision_rows=int(available[test].sum()), omitted_price_rows=0,
        first_timestamp=cfg["data"]["test_start"], end_exclusive=cfg["data"]["test_end_exclusive"],
        inherited_decision_support=True, decisions_missing_hold_inventory=True, account_resets=0,
        artifact_sha256={cfg["data"][k]: sha(cfg["data"][k]) for k in
            ("raw_ohlc_path", "raw_index_ns_path", "decision_available_path")})
    write_json(marker, value)
    return value


def checked_fit(cfg, task, registration_sha):
    directory = Path(cfg["output_dir"]) / "neural" / task["job"]
    path = directory / "fit.json"
    fit = json.loads(path.read_text())
    if (not fit.get("complete") or fit.get("registration_sha256") != registration_sha
            or any(fit.get(k) != task[k] for k in ("job", "fold", "arm", "seed", "condition"))
            or fit.get("test_used") is not False or fit.get("selected_update") not in cfg["training"]["selection_updates"]
            or fit.get("encoder_unchanged") is not True or fit.get("head_changed") is not True):
        raise ValueError("complete genuine head-only Train/Val fit required before Test")
    if (fit.get("updates") != cfg["training"]["max_updates"]
            or fit.get("historical_transitions") != cfg["training"]["max_updates"] * cfg["training"]["rollout_batch_size"] * cfg["training"]["rollout_horizon"]
            or fit.get("market_encoders_trainable") is not False or fit.get("wm_frozen") is not True
            or fit.get("bc_prior_immutable") is not True or fit.get("optimizer_moments_inherited") is not False
            or fit.get("optimizer_reset_at_stage_boundary") is not False):
        raise ValueError("complete identical head-only PPO budget and immutable representations required")
    pair = cfg["source_pairs"][f"fold_{task['fold']:02d}_{task['arm']}_s{task['seed']}"]
    for key, path_key in (("source_initial_checkpoint_sha256", "initial_checkpoint_path"),
                           ("source_bc_checkpoint_sha256", "bc_checkpoint_path")):
        if fit.get(key) != pair["source_artifact_sha256"][pair[path_key]]:
            raise ValueError("head continuation must begin from the exact registered source checkpoint")
    for name, digest in fit["artifact_sha256"].items():
        file = Path(name) if Path(name).is_absolute() else directory / name
        if sha(file) != digest:
            raise ValueError("selected head artifact changed")
    return fit


def freeze_test_barrier(cfg):
    registration_sha = verify(cfg)
    root = Path(cfg["output_dir"])
    bindings, paired = {}, {}
    for task in fit_jobs(cfg):
        fit = checked_fit(cfg, task, registration_sha)
        paired.setdefault((task["fold"], task["seed"], task["arm"]), []).append(fit)
        directory = root / "neural" / task["job"]
        for name in ("fit.json", "final.pt"):
            bindings[str((directory / name).resolve())] = sha(directory / name)
    for group in paired.values():
        if {fit["condition"] for fit in group} != set(cfg["conditions"]):
            raise ValueError("all release branches must be fitted for every source pair")
        for key in ("initial_state_sha256", "initial_encoder_sha256", "immutable_bc_prior_state_sha256",
                    "continuation_random_seed", "market_start_sequence_sha256", "context_manifest_sha256"):
            values = [fit.get(key) for fit in group]
            if any(value is None for value in values) or any(value != values[0] for value in values):
                raise ValueError("all branches require the same source state, prior, context and market stream")
    payload = dict(complete=True, registration_sha256=registration_sha, n_training_fits=len(fit_jobs(cfg)),
        frozen_fit_sha256=bindings, test_selection_allowed=False, created_after_all_Train_Val=True)
    path = root / "test_barrier.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if any(prior.get(k) != v for k, v in payload.items()):
            raise ValueError("frozen Test barrier changed")
        return prior
    write_json(path, dict(at=now(), **payload))
    return payload


def verify_test_barrier(cfg, *, full=False):
    root = Path(cfg["output_dir"])
    value = json.loads((root / "test_barrier.json").read_text())
    if (value.get("complete") is not True or value.get("registration_sha256") != verify(cfg)
            or value.get("n_training_fits") != len(fit_jobs(cfg)) or value.get("test_selection_allowed") is not False):
        raise ValueError("all registered Train/Val fits must be frozen before Test access")
    expected = {str((root / "neural" / task["job"] / name).resolve())
                for task in fit_jobs(cfg) for name in ("fit.json", "final.pt")}
    if set(value.get("frozen_fit_sha256", {})) != expected:
        raise ValueError("the exact complete set of registered frozen fits/checkpoints is required")
    token = (str(root.resolve()), value["registration_sha256"], sha(root / "test_barrier.json"))
    if full or token not in _VERIFIED_BARRIERS:
        for filename, digest in value["frozen_fit_sha256"].items():
            if sha(filename) != digest:
                raise ValueError("a frozen model/Val selection changed")
        for task in fit_jobs(cfg):
            checked_fit(cfg, task, value["registration_sha256"])
        _VERIFIED_BARRIERS.add(token)
    return value


def _init_worker(cfg, registration_sha):
    global _WORKER_CFG, _WORKER_REGISTRATION
    _WORKER_CFG, _WORKER_REGISTRATION = cfg, registration_sha
    import torch
    torch.set_num_threads(int(cfg["runtime"]["cpu_threads"]))


def _evaluate_path(cfg, task, registration_sha):
    from . import wm_control_release_training as training
    root = Path(cfg["output_dir"])
    directory = root / "evaluations" / task["job"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "executed_ledger.parquet"
    state, frames, selected = None, [], {}
    times = np.load(cfg["data"]["raw_index_ns_path"], mmap_mode="r")
    for meta in cfg["fold_metadata"]:
        f = int(meta["fold"])
        job = dict(job=f"fold_{f:02d}_{task['arm']}_s{task['seed']}_{task['condition']}",
            fold=f, arm=task["arm"], seed=task["seed"], condition=task["condition"])
        model, settings, cache = training.load_selected(cfg, job, registration_sha, frozen=task["kind"] == "frozen")
        ids = np.flatnonzero((times >= pd.Timestamp(meta["test_start"]).value) &
                            (times < pd.Timestamp(meta["test_end"]).value))
        ledger, state = training.replay(model, cache, ids, settings, initial_state=state)
        ledger["fold"] = f
        frames.append(ledger)
        checkpoint_name = "frozen_control.pt" if task["kind"] == "frozen" else "final.pt"
        selected[str(f)] = sha(root / "neural" / job["job"] / checkpoint_name)
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != cfg["statistics"]["expected_market_rows"] or frame.decision_id.duplicated().any():
        raise ValueError("one complete continuous calendar path required")
    for key in ("kind", "arm", "condition", "seed", "path_id"):
        frame[key] = task[key]
    frame.to_parquet(path, index=False)
    value = dict(complete=True, registration_sha256=registration_sha, barrier_sha256=sha(root / "test_barrier.json"),
        **task, path=str(path.resolve()), sha256=sha(path), initial_equity=1., n_rows=len(frame),
        account_resets=0, selected_checkpoint_sha256=selected, test_used_for_selection=False)
    write_json(directory / "evaluation.json", value)
    return value


def _worker_task(phase, task):
    from . import wm_control_release_training as training
    cfg, reg = _WORKER_CFG, _WORKER_REGISTRATION
    if phase == "contexts_train_val":
        return training.prepare_context(cfg, task, reg, split="train_val")
    if phase == "training":
        return training.train(cfg, task, reg)
    if phase == "contexts_test":
        return training.prepare_context(cfg, task, reg, split="test")
    if phase == "evaluation":
        return _evaluate_path(cfg, task, reg)
    raise ValueError("unknown scheduler phase")


def progress(cfg, phase, completed, total, running, started):
    value = dict(at=now(), phase=phase, completed=completed, total=total, running=running,
        elapsed_seconds=time.monotonic() - started, test_selection_allowed=False,
        independently_unused_holdout=False)
    root = Path(cfg["output_dir"])
    write_json(root / "status.json", value)
    (root / "progress.html").write_text("<!doctype html><meta charset='utf-8'><meta http-equiv='refresh' content='30'>"
        "<title>RL保有量・レバレッジ制御実験</title><h1>5段階×WM有無×10seed×6fold</h1>"
        "<p>600件のhead追加PPO。市場encoder固定、cash/units連続会計。</p><pre>"
        + json.dumps(value, ensure_ascii=False, indent=2) + "</pre>")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def run_pool(cfg, phase, tasks, workers):
    registration_sha = verify(cfg)
    root = Path(cfg["output_dir"])
    started = time.monotonic()
    remaining, completed = [], 0
    for task in tasks:
        marker = root / "scheduler" / phase / (task["job"] + ".json")
        if marker.exists():
            value = json.loads(marker.read_text())
            if value.get("registration_sha256") != registration_sha or value.get("complete") is not True:
                raise ValueError("scheduler completion belongs to a different registration")
            if phase == "training":
                checked_fit(cfg, task, registration_sha)
            completed += 1
        else:
            remaining.append(task)
    progress(cfg, phase, completed, len(tasks), [], started)
    if not remaining:
        return
    iterator = iter(remaining)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(cfg, registration_sha)) as executor:
        active = {}
        def submit():
            task = next(iterator, None)
            if task is not None:
                active[executor.submit(_worker_task, phase, task)] = task
        for _ in range(workers):
            submit()
        while active:
            ready, _ = wait(active, timeout=cfg["runtime"]["status_seconds"], return_when=FIRST_COMPLETED)
            for future in ready:
                task = active.pop(future)
                try:
                    result = future.result()
                except BaseException:
                    write_json(root / "failure.json", dict(at=now(), phase=phase, task=task, traceback=traceback.format_exc()))
                    for other in active:
                        other.cancel()
                    raise
                write_json(root / "scheduler" / phase / (task["job"] + ".json"),
                    dict(complete=True, registration_sha256=registration_sha, result=result))
                completed += 1
                submit()
            progress(cfg, phase, completed, len(tasks), [v["job"] for v in active.values()], started)


def path_manifest(cfg, registration_sha):
    root = Path(cfg["output_dir"])
    verify_test_barrier(cfg, full=True)
    rows = []
    for task in evaluation_jobs(cfg):
        value = json.loads((root / "evaluations" / task["job"] / "evaluation.json").read_text())
        if (not value.get("complete") or value["registration_sha256"] != registration_sha
                or value["barrier_sha256"] != sha(root / "test_barrier.json")
                or value["n_rows"] != 52608 or value["account_resets"] != 0
                or value["test_used_for_selection"] is not False or sha(value["path"]) != value["sha256"]):
            raise ValueError("all200 complete frozen continuous Test paths required")
        rows.append(value)
    value = dict(complete=True, registration_sha256=registration_sha, barrier_sha256=sha(root / "test_barrier.json"),
        paths=rows, raw_arrays={name: dict(path=cfg["data"][key], sha256=sha(cfg["data"][key])) for name, key in
            (("ohlc", "raw_ohlc_path"), ("index_ns", "raw_index_ns_path"), ("decision_available", "decision_available_path"))})
    write_json(root / "path_manifest.json", value)
    return value


def _run_locked(cfg):
    root = Path(cfg["output_dir"])
    registration_sha = register(cfg)
    verify_sources(cfg)
    completion = root / "completion.json"
    if completion.exists():
        value = json.loads(completion.read_text())
        if value.get("complete") is not True or value["registration_sha256"] != registration_sha:
            raise ValueError("invalid completed control-release study")
        for name, digest in value["artifact_sha256"].items():
            if sha(root / name) != digest:
                raise ValueError("completed report changed")
        return value
    prepare_data(cfg, registration_sha)
    run_pool(cfg, "contexts_train_val", context_jobs(cfg), cfg["runtime"]["encoder_workers"])
    run_pool(cfg, "training", fit_jobs(cfg), cfg["runtime"]["fit_workers"])
    prepare_data(cfg, registration_sha)
    verify_sources(cfg)
    freeze_test_barrier(cfg)
    run_pool(cfg, "contexts_test", context_jobs(cfg), cfg["runtime"]["encoder_workers"])
    run_pool(cfg, "evaluation", evaluation_jobs(cfg), cfg["runtime"]["evaluation_workers"])
    path_manifest(cfg, registration_sha)
    from .wm_control_release_evaluation import aggregate
    report = aggregate(cfg, registration_sha)
    prepare_data(cfg, registration_sha)
    verify_sources(cfg)
    verify_test_barrier(cfg, full=True)
    artifacts = {str(p.relative_to(root)): sha(p) for p in (root / "final").iterdir() if p.is_file()}
    value = dict(complete=True, at=now(), registration_sha256=registration_sha, new_training_fits=600,
        n_paths=200, n_market_rows=52608, n_seeds=10, source_artifacts_unchanged=True,
        test_selection_allowed=False, independently_unused_holdout=False, report=report,
        artifact_sha256=artifacts)
    write_json(completion, value)
    progress(cfg, "complete", 600, 600, [], time.monotonic())
    return value


def run(cfg):
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runner.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("this registered campaign already has a running supervisor") from error
        return _run_locked(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--action", choices=["register", "prepare", "run"], default="run")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.action == "register":
        print(register(cfg))
    elif args.action == "prepare":
        print(json.dumps(prepare_data(cfg, register(cfg)), ensure_ascii=False))
    else:
        print(json.dumps(run(cfg), ensure_ascii=False))


if __name__ == "__main__":
    main()
