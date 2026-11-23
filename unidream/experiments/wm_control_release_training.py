"""Frozen market encoders, cached contexts, and head-only continuation PPO.

This separate study reads its selected WM/actor/BC ancestors without changing
them. Contexts for the actor and immutable BC prior are cached independently.
Every calendar bar advances a cash/units account; missing features suppress
new decisions and never reset holdings, pending orders or drawdowns.
"""
from __future__ import annotations

import copy
import fcntl
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as F

from . import wm_control_release_accounting as account
from . import wm_policy_ablation_data as source_data
from . import wm_policy_ablation_policy as source_policy
from .wm_policy_ablation_models import PolicyActorCritic, compute_gae

sha, write_json = source_policy.sha, source_policy.write_json
HEAD_PREFIXES = ("policy_head.", "critic_head.")


def pair_id(task):
    return f"fold_{int(task['fold']):02d}_{task['arm']}_s{int(task['seed'])}"


def _registration(cfg, task, registration_sha):
    root = Path(cfg["output_dir"])
    path = root / "preregistration.json"
    if not path.exists() or sha(path) != registration_sha:
        raise ValueError("immutable control-release registration required before inputs")
    registration = json.loads(path.read_text())
    if registration.get("config") != cfg or pair_id(task) not in cfg["source_pairs"]:
        raise ValueError("worker source pair/configuration differs from registration")
    if "job" in task and (Path(task["job"]).name != task["job"] or task["job"] in (".", "..")):
        raise ValueError("worker job must be one local registered filename")
    return registration


def _barrier(cfg, registration_sha):
    from .wm_control_release_benchmark import verify_test_barrier
    value = verify_test_barrier(cfg)
    if value.get("registration_sha256") != registration_sha:
        raise ValueError("all head selections must be frozen before Test")
    return value


def condition_config(cfg, condition, update=None):
    """Copy only modified settings; large immutable source maps remain shared."""
    if condition not in cfg["schedules"] or condition not in cfg["conditions"]:
        raise ValueError("condition absent from the registered control-release study")
    maximum = int(cfg["training"]["max_updates"])
    update = maximum if update is None else update
    if isinstance(update, bool) or int(update) != update or not 1 <= update <= maximum:
        raise ValueError("update must lie inside the declared continuation budget")
    stages = cfg["schedules"][condition]
    if (not stages or stages[-1]["through_update"] != maximum
            or any(a["through_update"] >= b["through_update"] for a, b in zip(stages, stages[1:]))):
        raise ValueError("strictly ordered schedules must exhaust the exact update budget")
    stage = next(row for row in stages if update <= row["through_update"])
    accounting = {**cfg["accounting"], **stage.get("accounting", {})}
    training = {**cfg["training"], **stage.get("training", {}), **stage.get("rl_training", {})}
    for key in ("position_min", "position_max", "max_position_step"):
        if key in stage:
            accounting[key] = float(stage[key])
    if "bc_prior_coefficient" in stage:
        training["bc_prior_coefficient"] = float(stage["bc_prior_coefficient"])
    accounting["reset_positions_at_segment_boundary"] = False
    result = {**cfg, "accounting": accounting, "training": training, "rl_training": training}
    account._settings(result)
    if not 0 <= float(training.get("bc_prior_coefficient", .01)) <= .01:
        raise ValueError("nonnegative registered BC prior coefficient required")
    return result


def _apply(model, settings):
    for name in ("position_min", "position_max", "max_position_step"):
        setattr(model, name, float(settings["accounting"][name]))


def _is_head(name):
    return name == "log_std" or name.startswith(HEAD_PREFIXES)


def _state_sha(model, *, encoder_only=False, heads_only=False):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if encoder_only and _is_head(name) or heads_only and not _is_head(name):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _head_only(model, trainable=True):
    model.cpu().eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(trainable and _is_head(name))
        parameter.grad = None
    return model


def _cpu_seed(seed):
    """Seed only CPU RNGs; CPU fit workers must not initialize MPS devices."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.random.default_generator.manual_seed(int(seed))


def _load_source(pair, prior=False):
    path_key = "bc_checkpoint_path" if prior else "initial_checkpoint_path"
    reg_key = "bc_checkpoint_registration_sha256" if prior else "initial_checkpoint_registration_sha256"
    path = Path(pair[path_key])
    expected = pair.get("source_artifact_sha256", {}).get(str(path))
    if expected is not None and sha(path) != expected:
        raise ValueError("registered source actor/BC checkpoint changed")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if obj.get("registration_sha256") != pair[reg_key]:
        raise ValueError("source actor/BC checkpoint belongs to another registration")
    metadata = obj["metadata"]
    model = PolicyActorCritic(metadata["x_dim"], metadata["forecast_dim"], metadata["latent_dim"],
                             pair["kernel_config"], metadata["arm"])
    model.load_state_dict(obj["state_dict"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("finite source actor/BC state required")
    return model


def _fold(pair):
    with np.load(pair["fold_path"], allow_pickle=False) as value:
        return {name: value[name] for name in value.files}


@lru_cache(maxsize=4)
def _prices(ohlc_path, index_path):
    return (np.load(ohlc_path, mmap_mode="r", allow_pickle=False),
            np.load(index_path, mmap_mode="r", allow_pickle=False))


def _raw(cfg):
    return _prices(str(cfg["data"]["raw_ohlc_path"]), str(cfg["data"]["raw_index_ns_path"]))


def _price_support(ohlc, raw_ids):
    ids = np.asarray(raw_ids, np.int64)
    if not len(ids) or np.any(ids < 1) or np.any(ids >= len(ohlc)):
        raise ValueError("nonempty raw decisions and previous closes are required")
    bars = np.asarray(ohlc[ids], np.float64)
    previous = np.asarray(ohlc[ids-1, 3], np.float64)
    if (bars.shape != (len(ids), 4) or not np.isfinite(bars).all() or np.any(bars <= 0)
            or not np.isfinite(previous).all() or np.any(previous <= 0)
            or np.any(bars[:, 1] < np.maximum(bars[:, 0], bars[:, 3]))
            or np.any(bars[:, 2] > np.minimum(bars[:, 0], bars[:, 3]))):
        raise ValueError("raw eligible OHLC and predecessor closes must be positive, finite and consistent")


def _source_hashes(pair, split):
    names = [pair[name] for name in ("initial_checkpoint_path", "initial_fit_path", "bc_checkpoint_path", "fold_path")]
    if pair.get("latent_norm_path"):
        names += [pair["latent_norm_path"], pair["latent_test_path" if split == "test" else "latent_train_val_path"]]
    expected = pair.get("source_artifact_sha256", {})
    result = {}
    for filename in names:
        path = str(Path(filename).resolve())
        digest = sha(path)
        if path in expected and expected[path] != digest:
            raise ValueError("source state/normalization/latent cache changed before context encoding")
        result[path] = digest
    return result


def _cache_directory(cfg, task, split):
    return Path(cfg["output_dir"]) / "contexts" / pair_id(task) / split


def _check_cache(manifest, registration_sha, *, verify_arrays=False):
    if (manifest.get("complete") is not True or manifest.get("registration_sha256") != registration_sha
            or manifest.get("actor_and_prior_contexts_separate") is not True):
        raise ValueError("complete registered actor and BC-prior context caches required")
    if verify_arrays:
        directory = Path(manifest["cachepath"])
        for name, digest in manifest["artifact_sha256"].items():
            if sha(directory/name) != digest:
                raise ValueError("immutable context cache changed")
    return manifest


def prepare_context(cfg, task, registration_sha, split="train_val"):
    _registration(cfg, task, registration_sha)
    if split not in ("train_val", "test"):
        raise ValueError("context split must be train_val or report-only test")
    if split == "test":
        _barrier(cfg, registration_sha)
    directory = _cache_directory(cfg, task, split)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/"encoding.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        marker = directory/"manifest.json"
        if marker.exists():
            manifest = _check_cache(json.loads(marker.read_text()), registration_sha, verify_arrays=True)
            if manifest.get("pair_id") != pair_id(task) or manifest.get("split") != split:
                raise ValueError("context cache belongs to another pair or split")
            return manifest
        torch.set_num_threads(int(cfg.get("runtime", {}).get("cpu_threads", 1)))
        pair = cfg["source_pairs"][pair_id(task)]
        provenance = _source_hashes(pair, split)
        settings, fold = pair["kernel_config"], _fold(pair)
        arrays = source_data.load_arrays(settings)
        train, val = (np.asarray(arrays["decision_id"])[fold[name]].astype(np.int64) for name in ("train", "val"))
        ids = (np.asarray(fold["test"], np.int64) if split == "test"
               else np.unique(np.r_[fold["train"], fold["val"]]).astype(np.int64))
        raw_ids = np.asarray(arrays["decision_id"])[ids].astype(np.int64)
        if not len(raw_ids) or np.any(np.diff(raw_ids) <= 0):
            raise ValueError("unique increasing eligible raw decisions required")
        ohlc, raw_times = _raw(cfg)
        _price_support(ohlc, raw_ids)
        if not np.array_equal(np.asarray(arrays["index_ns"])[ids], np.asarray(raw_times)[raw_ids]):
            raise ValueError("source ledger and new account raw timestamps must align exactly")
        actor, prior = _load_source(pair), _load_source(pair, prior=True)
        initial_encoder, prior_encoder = _state_sha(actor, encoder_only=True), _state_sha(prior, encoder_only=True)
        device = cfg.get("runtime", {}).get("encoder_device", cfg.get("device", "mps"))
        actor, prior = actor.to(device).eval(), prior.to(device).eval()
        latent, norm = None, None
        if task["arm"] == "C":
            latent = np.load(pair["latent_test_path" if split == "test" else "latent_train_val_path"], mmap_mode="r", allow_pickle=False)
            with np.load(pair["latent_norm_path"], allow_pickle=False) as obj:
                norm = {name: obj[name] for name in obj.files}
        width = actor.hidden_size + 2*actor.projection_size
        actor_tmp, prior_tmp = directory/"contexts.npy.tmp", directory/"prior_contexts.npy.tmp"
        contexts = np.lib.format.open_memmap(actor_tmp, mode="w+", dtype=np.float32, shape=(len(ids), width))
        prior_contexts = np.lib.format.open_memmap(prior_tmp, mode="w+", dtype=np.float32, shape=(len(ids), width))
        batch = int(cfg["training"].get("context_batch_size", 512))
        with torch.no_grad():
            for start in range(0, len(ids), batch):
                rows = ids[start:start+batch]
                values = source_policy._inputs(arrays, rows, fold, settings, task["arm"], latent, norm, int(pair["latent_dim"]))
                tensors = source_policy._tensors(values, device)
                left, right = actor.encode_context(*tensors).cpu().numpy(), prior.encode_context(*tensors).cpu().numpy()
                if not np.isfinite(left).all() or not np.isfinite(right).all():
                    raise RuntimeError("finite causal actor and BC contexts required")
                contexts[start:start+len(rows)], prior_contexts[start:start+len(rows)] = left, right
        contexts.flush(); prior_contexts.flush()
        del contexts, prior_contexts
        actor_tmp.replace(directory/"contexts.npy"); prior_tmp.replace(directory/"prior_contexts.npy")
        if _state_sha(actor, encoder_only=True) != initial_encoder or _state_sha(prior, encoder_only=True) != prior_encoder:
            raise RuntimeError("market context encoding must not change either frozen encoder")
        for name, value in (("raw_ids", raw_ids), ("train_raw_ids", train), ("validation_raw_ids", val)):
            np.save(directory/(name+".npy"), value, allow_pickle=False)
        names = ["raw_ids.npy", "contexts.npy", "prior_contexts.npy", "train_raw_ids.npy", "validation_raw_ids.npy"]
        manifest = {"complete": True, "registration_sha256": registration_sha, "pair_id": pair_id(task),
            "split": split, "cachepath": str(directory.resolve()), "manifest_path": str(marker.resolve()),
            "paths": {name[:-4]: str((directory/name).resolve()) for name in names},
            "artifact_sha256": {name: sha(directory/name) for name in names}, "source_artifact_sha256": provenance,
            "actor_encoder_sha256": initial_encoder, "prior_encoder_sha256": prior_encoder,
            "actor_and_prior_contexts_separate": True, "n_rows": len(ids), "context_dim": width,
            "normalizer_fit_split": "original policy Train only", "encoder_device": str(device),
            "test_used": split == "test", "test_selection_allowed": False}
        write_json(marker, manifest)
        return manifest


@lru_cache(maxsize=8)
def _context_arrays(manifest_path, manifest_sha):
    manifest = json.loads(Path(manifest_path).read_text())
    return {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in manifest["paths"].items()}


def _cached(cache):
    path = cache["manifest_path"]
    return _context_arrays(path, sha(path))


def _positions(cache_ids, raw_ids):
    raw_ids = np.asarray(raw_ids, np.int64)
    positions = np.searchsorted(cache_ids, raw_ids)
    valid = positions < len(cache_ids)
    valid[valid] &= np.asarray(cache_ids)[positions[valid]] == raw_ids[valid]
    return positions, valid


def _coverage(model):
    result = {}
    for name, module in (("policy", model.policy_head), ("critic", model.critic_head)):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        result[name] = bool(gradients and all(torch.isfinite(g).all() for g in gradients)
            and sum(float(g.abs().sum()) for g in gradients) > 0)
    result["encoder_has_no_gradients"] = all(p.grad is None and not p.requires_grad for n, p in model.named_parameters() if not _is_head(n))
    return result


@torch.no_grad()
def replay(model, contexts_cache, raw_ids, cfg, initial_state=None, initial_price=None, deterministic=True):
    """Run consecutive calendar bars and return (ledger, carried account state)."""
    if not deterministic:
        raise ValueError("report and Val replay require deterministic selected policy heads")
    ids = np.asarray(raw_ids, np.int64)
    if ids.ndim != 1 or not len(ids) or np.any(np.diff(ids) != 1):
        raise ValueError("replay requires the complete consecutive raw calendar range")
    ohlc, index_ns = _raw(cfg)
    _price_support(ohlc, ids)
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("cached-context account replay requires CPU policy heads")
    model.eval()
    cache = _cached(contexts_cache)
    positions, available = _positions(cache["raw_ids"], ids)
    encoded = np.zeros((len(ids), contexts_cache["context_dim"]), np.float32)
    encoded[available] = cache["contexts"][positions[available]]
    tensor = torch.from_numpy(encoded)
    if initial_state is None:
        price = float(ohlc[ids[0]-1, 3]) if initial_price is None else initial_price
        state = account.initial_state(1, "cpu", price, cfg)
    else:
        state = {name: value.detach().cpu().clone() for name, value in initial_state.items()}
        if len(state["cash"]) != 1:
            raise ValueError("report replay carries exactly one account")
        if bool(state["alive"][0]) and not np.isclose(float(state["prev_close"][0]), float(ohlc[ids[0]-1, 3]), rtol=0, atol=1e-8):
            raise ValueError("carried account must continue from the previous raw close")
    rows = []
    for row, raw_id in enumerate(ids):
        alive_before = bool(state["alive"][0])
        previous_equity = float(state["prev_equity"][0])
        if available[row] and alive_before:
            action = model.act_from_context(tensor[row:row+1], account.inventory(state).float(), deterministic=True)
            request = action["request"].double()
        else:
            request = state["requested"].clone()
        market_log = float(np.log(float(ohlc[raw_id, 3])/float(ohlc[raw_id-1, 3])))
        bar = ohlc[raw_id]
        state, diagnostics = account.step(state, request, bar[0], bar[2], bar[3], cfg,
                                          decision_available=bool(available[row]))
        record = {name: (bool(value[0]) if value.dtype == torch.bool else float(value[0])) for name, value in diagnostics.items()}
        record.update(raw_id=int(raw_id), decision_id=int(raw_id), timestamp=pd.Timestamp(int(index_ns[raw_id]), tz="UTC"),
            market_log_return=market_log, bar_log_return=market_log, equity_previous=previous_equity,
            alive_before=alive_before, context_available=bool(available[row]))
        record["unscaled_reward"] = (record["training_log_return"] - market_log
            - float(cfg["accounting"].get("drawdown_penalty", .25))*record["max_dd_increment"]) if alive_before else 0.
        rows.append(record)
    return pd.DataFrame(rows), state


def replay_metrics(ledger):
    if ledger.empty:
        raise ValueError("nonempty account ledger required")
    bankrupt = bool(ledger.bankrupt.any())
    initial = float(ledger.equity_previous.iloc[0])
    terminal = float(ledger.equity.iloc[-1])
    return {"n_rows": len(ledger), "net_total_return": terminal/initial-1 if initial > 0 else 0.,
        "terminal_equity": terminal, "max_drawdown": float(ledger.max_drawdown.max()),
        "bankrupt": bankrupt, "selection_score": -1e30 if bankrupt else float(ledger.unscaled_reward.sum()),
        "reward_total": float(ledger.unscaled_reward.sum()), "mean_exposure": float(ledger.executed_position.mean()),
        "fee_equity_units": float(ledger.fee.sum()), "borrow_equity_units": float(ledger.borrow.sum()),
        "decision_rows": int(ledger.decision_available.sum()), "calendar_account_no_reset": True}


def _contiguous_starts(raw_ids, horizon):
    ids = np.asarray(raw_ids, np.int64)
    boundary = np.flatnonzero(np.r_[True, np.diff(ids) != 1, True])
    candidates = [np.arange(a, b-horizon, dtype=np.int64) for a, b in zip(boundary[:-1], boundary[1:]) if b-a > horizon]
    return np.concatenate(candidates) if candidates else np.empty(0, np.int64)


@torch.no_grad()
def _collect(model, contexts, prior_contexts, ohlc, raw_ids, settings):
    batch, span = raw_ids.shape
    horizon = span-1
    state = account.initial_state(batch, "cpu", np.asarray(ohlc[raw_ids[:, 0]-1, 3]), settings)
    saved = {key: [] for key in ("inventory", "raw_action", "old_log_prob", "old_value", "reward", "terminals", "active")}
    model.eval()
    for step in range(horizon):
        inv = account.inventory(state).float()
        action = model.act_from_context(contexts[:, step], inv)
        was_alive = state["alive"].clone()
        ids = raw_ids[:, step]
        bars = np.asarray(ohlc[ids], np.float64)
        market = torch.from_numpy(np.log(np.asarray(ohlc[ids, 3])/np.asarray(ohlc[ids-1, 3])))
        state, diagnostics = account.step(state, action["request"].double(), bars[:, 0], bars[:, 2], bars[:, 3], settings)
        reward = diagnostics["training_log_return"]-market-float(settings["accounting"].get("drawdown_penalty", .25))*diagnostics["max_dd_increment"]
        reward = torch.where(was_alive, reward, torch.zeros_like(reward))*float(settings["training"].get("reward_scale", 100.))
        for name, value in (("inventory", inv), ("raw_action", action["raw_action"]), ("old_log_prob", action["log_prob"]),
                            ("old_value", action["value"]), ("reward", reward.float()), ("terminals", ~state["alive"]), ("active", was_alive)):
            saved[name].append(value)
    bootstrap = model.heads(contexts[:, -1], account.inventory(state).float())[1]
    stacked = {name: torch.stack(values, 1) for name, values in saved.items()}
    advantages, returns = compute_gae(stacked["reward"].T, stacked["old_value"].T, stacked["terminals"].T,
        bootstrap, float(settings["training"].get("gamma", .99)), float(settings["training"].get("gae_lambda", .95)))
    result = {name: value.flatten(0, 1).detach() for name, value in stacked.items() if name not in ("reward", "terminals")}
    result.update(contexts=contexts[:, :-1].flatten(0, 1), prior_contexts=prior_contexts[:, :-1].flatten(0, 1),
                  advantages=advantages.T.flatten().detach(), returns=returns.T.flatten().detach())
    return {"batch": result, "reward_mean": float(stacked["reward"].mean()), "bankrupt_rollouts": int((~state["alive"]).sum())}


def _optimizer(model, settings):
    return torch.optim.AdamW([
        {"params": [*model.policy_head.parameters(), model.log_std], "lr": float(settings["training"]["learning_rate"])},
        {"params": model.critic_head.parameters(), "lr": float(settings["training"]["critic_learning_rate"])}], weight_decay=0.)


def _ppo(model, prior, optimizer, rollout, settings):
    batch, training = rollout["batch"], settings["training"]
    advantages = batch["advantages"]
    alive = batch["active"].float()
    live_advantages = advantages[batch["active"]]
    advantages = ((advantages-live_advantages.mean())/live_advantages.std(unbiased=False).clamp_min(1e-8))*alive
    n, size = len(advantages), int(training["minibatch_size"])
    epochs, clip = int(training["ppo_epochs"]), float(training["clip_ratio"])
    with torch.no_grad():
        distribution, _ = prior.heads(batch["prior_contexts"], batch["inventory"])
        reference_mean, reference_std = distribution.mean, distribution.stddev
    rows = []
    model.train(); prior.eval()
    for _ in range(epochs):
        for ids in torch.randperm(n).split(size):
            current, values = model.heads(batch["contexts"][ids], batch["inventory"][ids])
            ratio = torch.exp(current.log_prob(batch["raw_action"][ids])-batch["old_log_prob"][ids])
            weights = alive[ids]
            denominator = weights.sum().clamp_min(1)
            policy_loss = -(torch.minimum(ratio*advantages[ids], ratio.clamp(1-clip, 1+clip)*advantages[ids])*weights).sum()/denominator
            critic_loss = F.mse_loss(values, batch["returns"][ids])
            prior_kl = (kl_divergence(current, Normal(reference_mean[ids], reference_std[ids], validate_args=False))*weights).sum()/denominator
            entropy = (current.entropy()*weights).sum()/denominator
            loss = policy_loss + float(training["value_coefficient"])*critic_loss + float(training["bc_prior_coefficient"])*prior_kl-float(training["entropy_coefficient"])*entropy
            actor_parameters = [*model.policy_head.parameters(), model.log_std]
            grads = torch.autograd.grad(policy_loss, actor_parameters, retain_graph=True, allow_unused=True)
            policy_norm = torch.sqrt(sum(gradient.square().sum() for gradient in grads if gradient is not None))
            before = model.policy_head[-1].weight.detach().clone()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(training["gradient_clip"]), error_if_nonfinite=True)
            optimizer.step()
            row = {"policy_loss": float(policy_loss.detach()), "critic_loss": float(critic_loss.detach()),
                "prior_kl": float(prior_kl.detach()), "latent_entropy": float(entropy.detach()),
                "policy_gradient_norm": float(policy_norm.detach()), "total_gradient_norm": float(norm),
                "policy_parameter_delta": float((model.policy_head[-1].weight.detach()-before).norm()),
                "clip_fraction": float(((ratio.detach()-1).abs()>clip).float().mean())}
            if not all(np.isfinite(value) for value in row.values()):
                raise RuntimeError("finite head-only PPO gradients and losses required")
            rows.append(row)
    diagnostics = {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}
    diagnostics.update(optimizer_steps=len(rows), transitions=n,
        real_policy_gradient_steps=sum(row["policy_gradient_norm"]>0 for row in rows),
        changed_policy_steps=sum(row["policy_parameter_delta"]>0 for row in rows),
        reward_mean=rollout["reward_mean"], bankrupt_rollouts=rollout["bankrupt_rollouts"])
    return diagnostics


def _val_range(cfg, task):
    _, times = _raw(cfg)
    settings = cfg["source_pairs"][pair_id(task)]["kernel_config"]
    meta = next(row for row in source_data.policy_folds(settings) if row["fold"] == int(task["fold"]))
    start, end = (pd.Timestamp(meta[name]).value for name in ("val_start", "val_end"))
    return np.arange(np.searchsorted(times, start), np.searchsorted(times, end), dtype=np.int64)


def _fit(cfg, task, registration_sha):
    path = Path(cfg["output_dir"])/"neural"/task["job"]/"fit.json"
    return source_policy._complete_fit(path, registration_sha)


def train(cfg, task, registration_sha):
    _registration(cfg, task, registration_sha)
    torch.set_num_threads(int(cfg.get("runtime", {}).get("cpu_threads", 1)))
    completed = _fit(cfg, task, registration_sha)
    if completed:
        return completed
    started = time.monotonic()
    pair = cfg["source_pairs"][pair_id(task)]
    cache = prepare_context(cfg, task, registration_sha)
    values = _cached(cache)
    model, prior = _head_only(_load_source(pair)), _head_only(_load_source(pair, prior=True), False)
    if _state_sha(model, encoder_only=True) != cache["actor_encoder_sha256"] or _state_sha(prior, encoder_only=True) != cache["prior_encoder_sha256"]:
        raise ValueError("cached actor and BC encoder identities differ")
    initial_sha, encoder_sha, head_sha, prior_sha = _state_sha(model), _state_sha(model, encoder_only=True), _state_sha(model, heads_only=True), _state_sha(prior)
    random_seed = int(np.random.SeedSequence([int(task["seed"]), int(task["fold"]), ord(task["arm"]), int(cfg.get("random_seed", 20261112))]).generate_state(1)[0])
    _cpu_seed(random_seed); rng = np.random.default_rng(random_seed)
    maximum, candidates = int(cfg["training"]["max_updates"]), list(cfg["training"]["selection_updates"])
    final = condition_config(cfg, task["condition"])
    for update in candidates:
        candidate = condition_config(cfg, task["condition"], update)
        if candidate["accounting"] != final["accounting"] or candidate["training"] != final["training"]:
            raise ValueError("all Val candidates must use the final release settings")
    directory = Path(cfg["output_dir"])/"neural"/task["job"]
    directory.mkdir(parents=True, exist_ok=True)
    _apply(model, final)
    source_policy._save_checkpoint(directory/"frozen_control.pt", model, registration_sha, 0, "frozen_weight_control")
    _apply(model, condition_config(cfg, task["condition"], 1))
    optimizer = _optimizer(model, final)
    train_raw = np.asarray(values["train_raw_ids"], np.int64)
    horizon, batch = int(cfg["training"]["rollout_horizon"]), int(cfg["training"]["rollout_batch_size"])
    starts = _contiguous_starts(train_raw, horizon)
    if not len(starts):
        raise ValueError("nonempty contiguous eligible Train rollouts required")
    ohlc, times = _raw(cfg)
    _price_support(ohlc, train_raw)
    fold = _fold(pair)
    meta = next(row for row in source_data.policy_folds(pair["kernel_config"]) if row["fold"] == int(task["fold"]))
    if (np.any(np.asarray(times)[train_raw]<pd.Timestamp(meta["train_start"]).value)
            or np.any(np.asarray(times)[train_raw]+int(pair["kernel_config"]["policy"]["purge_bars"])*source_data.BAR_NS>=pd.Timestamp(meta["train_end"]).value)):
        raise ValueError("all rollout returns must remain strictly inside purged Train")
    history, market_digest = [], hashlib.sha256()
    best, selected, best_metrics = -np.inf, None, None
    transitions = changed_updates = gradient_updates = 0
    val_ids = _val_range(cfg, task)
    for update in range(1, maximum+1):
        active = condition_config(cfg, task["condition"], update)
        _apply(model, active)
        offsets = rng.choice(starts, batch, replace=True)
        ids = train_raw[offsets[:, None]+np.arange(horizon+1)[None, :]]
        market_digest.update(np.ascontiguousarray(ids[:, 0], np.int64).tobytes())
        positions, available = _positions(values["raw_ids"], ids.ravel())
        if not available.all():
            raise ValueError("every rollout/bootstrap decision requires a precomputed causal context")
        contexts = torch.from_numpy(np.asarray(values["contexts"][positions], np.float32)).reshape(batch, horizon+1, -1)
        bc_contexts = torch.from_numpy(np.asarray(values["prior_contexts"][positions], np.float32)).reshape(batch, horizon+1, -1)
        rollout = _collect(model, contexts, bc_contexts, ohlc, ids, active)
        diagnostics = _ppo(model, prior, optimizer, rollout, active)
        transitions += diagnostics["transitions"]
        gradient_updates += int(diagnostics["real_policy_gradient_steps"]>0)
        changed_updates += int(diagnostics["changed_policy_steps"]>0)
        record = {"update": update, "position_max": model.position_max, "max_position_step": model.max_position_step,
            "bc_prior_coefficient": active["training"]["bc_prior_coefficient"], "fresh_on_policy_rollout": True,
            "diagnostics": diagnostics, "historical_transitions_total": transitions, "elapsed_seconds": time.monotonic()-started}
        if update in (1, maximum):
            coverage = _coverage(model)
            if not all(coverage.values()):
                raise RuntimeError("finite active head gradients and absent encoder gradients required")
            record["head_gradient_coverage"] = coverage
        if update in candidates:
            _apply(model, final)
            ledger, _ = replay(model, cache, val_ids, final)
            metrics = replay_metrics(ledger)
            record.update(validation_score=metrics["selection_score"], validation_metrics=metrics)
            if metrics["selection_score"] > best:
                best, selected, best_metrics = metrics["selection_score"], update, metrics
                source_policy._save_checkpoint(directory/"final.pt", model, registration_sha, update, "head_only_continuation")
                ledger.to_parquet(directory/"validation_predictions.parquet", index=False)
            record["selected_update"] = selected
        history.append(record)
        if update == 1 or update % 25 == 0:
            write_json(directory/"progress.json", {"job": task["job"], **record})
    if selected not in candidates or transitions != maximum*batch*horizon or not changed_updates or not gradient_updates:
        raise RuntimeError("complete head-only PPO budget and genuine policy changes required")
    source_policy._load_checkpoint(directory/"final.pt", model, registration_sha)
    _apply(model, final)
    encoder_unchanged = _state_sha(model, encoder_only=True) == encoder_sha
    head_changed = _state_sha(model, heads_only=True) != head_sha
    if not encoder_unchanged or not head_changed or _state_sha(prior) != prior_sha:
        raise RuntimeError("immutable actor/BC encoders and genuine selected head learning required")
    write_json(directory/"training_history.json", history)
    paths = ("final.pt", "frozen_control.pt", "training_history.json", "validation_predictions.parquet")
    fit = {"complete": True, "registration_sha256": registration_sha, **task,
        "selected_update": selected, "updates": maximum, "historical_transitions": transitions,
        "updates_with_policy_gradient": gradient_updates, "updates_with_policy_parameter_changes": changed_updates,
        "encoder_unchanged": encoder_unchanged, "head_changed": head_changed, "market_encoders_trainable": False,
        "initial_state_sha256": initial_sha, "initial_encoder_sha256": encoder_sha, "initial_heads_sha256": head_sha,
        "selected_state_sha256": _state_sha(model), "immutable_bc_prior_state_sha256": prior_sha,
        "continuation_random_seed": random_seed, "market_start_sequence_sha256": market_digest.hexdigest(),
        "context_manifest_path": cache["manifest_path"], "context_manifest_sha256": sha(cache["manifest_path"]),
        "source_initial_checkpoint_sha256": sha(pair["initial_checkpoint_path"]),
        "source_bc_checkpoint_sha256": sha(pair["bc_checkpoint_path"]),
        "normalizer_fit_split": "original policy Train only", "normalizer_source_id": source_policy.train_source_id(fold),
        "selected_validation_score": best, "validation_metrics": best_metrics,
        "validation_calendar_rows": len(val_ids), "validation_selection_scope": "complete consecutive raw Val calendar; carried cash/units/pending/DD",
        "test_used": False, "wm_frozen": True, "bc_prior_immutable": True,
        "optimizer_moments_inherited": False, "optimizer_reset_at_stage_boundary": False,
        "checkpoint_metadata": model.checkpoint_metadata(), "final_accounting": final["accounting"],
        "final_bc_prior_coefficient": final["training"]["bc_prior_coefficient"],
        "head_gradient_coverage": coverage, "elapsed_seconds": time.monotonic()-started,
        "artifact_sha256": {name: sha(directory/name) for name in paths}}
    write_json(directory/"fit.json", fit)
    return fit


def load_selected(cfg, task, registration_sha, frozen=False):
    _registration(cfg, task, registration_sha)
    _barrier(cfg, registration_sha)
    fit = _fit(cfg, task, registration_sha)
    if fit is None or fit.get("test_used") is not False:
        raise ValueError("frozen complete Train/Val head selection required")
    pair = cfg["source_pairs"][pair_id(task)]
    model = _head_only(_load_source(pair), False)
    directory = Path(cfg["output_dir"])/"neural"/task["job"]
    name = "frozen_control.pt" if frozen else "final.pt"
    obj = source_policy._load_checkpoint(directory/name, model, registration_sha)
    final = condition_config(cfg, task["condition"])
    _apply(model, final)
    if _state_sha(model, encoder_only=True) != fit["initial_encoder_sha256"]:
        raise ValueError("selected/frozen control encoder changed")
    if frozen and _state_sha(model) != fit["initial_state_sha256"]:
        raise ValueError("frozen-weight control must keep every original actor/critic weight")
    if any(obj["metadata"].get(key) != final["accounting"][key] for key in ("position_min", "position_max", "max_position_step")):
        raise ValueError("checkpoint action-map metadata differs from final registered constraints")
    cache = prepare_context(cfg, task, registration_sha, "test")
    return model, final, cache
