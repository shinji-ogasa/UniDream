"""Isolated BC, cached-market PPO and continuous-calendar tail-transfer replay.

Producer features must already be chronological out-of-sample for policy Train.
This module never fits a market producer and never chooses a policy on Test.
BC learns its own causal history/projection encoder; PPO freezes that encoder.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Normal, Independent, kl_divergence
from torch.nn import functional as F

from . import wm_tail_transfer_accounting as account
from .wm_policy_ablation_models import compute_gae


VARIANTS = tuple(f"R{i:02d}" for i in range(8))
MARKET_KEYS = ("raw", "old_z", "new_z", "forecasts", "uncertainty", "oracle")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def effective_config(cfg, task):
    result = {**cfg, "policy": dict(cfg.get("policy", {})), "accounting": dict(cfg.get("accounting", {})),
              "perpetual": dict(cfg.get("perpetual", {}))}
    control = task.get("control", {})
    aliases = {"cap": "position_max", "speed": "max_position_step", "account_mode": "mode",
               "mmr": "maintenance_margin_rate", "slippage": "slippage_rate"}
    for key, name in aliases.items():
        if key in control:
            result["accounting"][name] = control[key]
    for key in ("controller", "risk_gate", "objective", "bc_prior"):
        if key in control:
            result["policy"][key] = control[key]
    result["accounting"]["reset_positions_at_segment_boundary"] = False
    result["accounting"].setdefault("mode", "cash_borrow")
    result["policy"].setdefault("controller", "direct")
    result["policy"].setdefault("objective", "net_log_dd")
    result["policy"].setdefault("bc_prior", .01)
    result["policy"].setdefault("risk_gate", False)
    if result["policy"]["objective"] == "episodic_cvar_5pct":
        result["policy"]["rollout_batch_size"] = int(result["policy"].get("cvar_rollout_batch_size", result["policy"].get("cvar_rollout_batch", 64)))
        result["policy"]["ppo_updates"] = int(result["policy"].get("cvar_ppo_updates", 50))
        result["policy"]["selection_updates"] = list(result["policy"].get("cvar_selection_updates", [25, 32, 38, 44, 50]))
    if result["policy"]["controller"] not in ("direct", "hierarchical", "ml_rule"):
        raise ValueError("unknown registered allocation controller")
    if result["policy"]["objective"] not in ("net_log_dd", "net_log", "episodic_cvar_5pct"):
        raise ValueError("unknown registered PPO objective")
    return result


def _market(arrays):
    return arrays.get("market", arrays)


def _raw(arrays, cfg):
    values = _market(arrays)
    key = "raw" if "raw" in values else "input_"+cfg.get("policy", {}).get("source_input_group", "D03")
    if key not in values:
        raise ValueError("causal per-calendar raw market features required")
    return values[key]


def _dims(arrays, cfg):
    values = _market(arrays)
    width = lambda key: int(values[key].shape[1]) if key in values else 1
    return {"raw": int(_raw(arrays, cfg).shape[1]), "z": max(width("old_z"), width("new_z")),
            "f": max(width("forecasts"), width("oracle")), "u": width("uncertainty")}


def _columns(values, ids, width):
    output = np.zeros((len(ids), width), np.float32)
    if values is not None:
        part = np.asarray(values[ids], np.float32)
        if part.ndim != 2 or part.shape[1] > width or not np.isfinite(part).all():
            raise ValueError("finite matched producer features required for active inputs")
        output[:, :part.shape[1]] = part
    return output


class CachedTailPolicy(nn.Module):
    """Equal geometry across R00..R07; only declared inputs are read."""
    def __init__(self, dims, cfg, variant="R04"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("unknown tail-transfer input variant")
        self.variant, self.dims = variant, dict(dims)
        p, a = cfg["policy"], cfg["accounting"]
        self.context = int(p.get("context", 128))
        hidden, projection = int(p.get("hidden_size", 64)), int(p.get("projection_dim", 64))
        self.raw_encoder = nn.Sequential(nn.Linear(dims["raw"], hidden), nn.ELU())
        self.history_encoder = nn.GRU(hidden, hidden, batch_first=True)
        self.history_norm = nn.LayerNorm(hidden)
        self.z_encoder = nn.Sequential(nn.Linear(dims["z"], projection), nn.ELU())
        self.f_encoder = nn.Sequential(nn.Linear(dims["f"], projection), nn.ELU())
        self.u_encoder = nn.Sequential(nn.Linear(dims["u"], projection), nn.ELU())
        self.context_dim = hidden+3*projection+4
        # All policies have the same 14 account-input slots. R00..R05/R07
        # hard-mask the eight expanded slots rather than changing capacity.
        head_hidden = int(p.get("head_hidden", 128))
        self.action_dim = 3 if p.get("controller", "direct") == "hierarchical" else 1
        self.policy_head = nn.Sequential(nn.Linear(self.context_dim+14, head_hidden), nn.ELU(), nn.Linear(head_hidden, self.action_dim))
        self.critic_head = nn.Sequential(nn.Linear(self.context_dim+14, head_hidden), nn.ELU(), nn.Linear(head_hidden, 1))
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -1.))
        self.position_min, self.position_max = float(a.get("position_min", 0.)), float(a.get("position_max", 2.))
        self.max_position_step = float(a.get("max_position_step", .08))
        self.controller = p.get("controller", "direct")
        self.risk_gate = bool(p.get("risk_gate", False))
        self.gate_threshold = float(p.get("uncertainty_gate_threshold", .9))
        self.annual_bars = float(a.get("annual_bars", 35040))
        self.prediction_horizon = float(p.get("prediction_horizon_bars", 16))
        for key in ("raw", "z", "f", "u"):
            size = dims[key]
            self.register_buffer(key+"_mean", torch.zeros(size))
            self.register_buffer(key+"_std", torch.ones(size))
        with torch.no_grad():
            self.policy_head[-1].weight.mul_(.01)
            self.policy_head[-1].bias.zero_()

    def encode_context(self, raw, z, forecasts, uncertainty, risk):
        if raw.ndim != 3 or raw.shape[-1] != self.dims["raw"] or not 0 < raw.shape[1] <= self.context:
            raise ValueError("matched causal raw windows required")
        history, _ = self.history_encoder(self.raw_encoder(((raw-self.raw_mean)/self.raw_std).clamp(-10, 10)))
        h = self.history_norm(history[:, -1])
        z_out = self.z_encoder(((z-self.z_mean)/self.z_std).clamp(-10, 10)) if self.variant in ("R01", "R03", "R04", "R05", "R06") else h.new_zeros((len(h), self.z_encoder[0].out_features))
        f_out = self.f_encoder(((forecasts-self.f_mean)/self.f_std).clamp(-10, 10)) if self.variant in ("R02", "R04", "R05", "R06", "R07") else h.new_zeros((len(h), self.f_encoder[0].out_features))
        u_out = self.u_encoder(((uncertainty-self.u_mean)/self.u_std).clamp(-10, 10)) if self.variant in ("R05", "R06") else h.new_zeros((len(h), self.u_encoder[0].out_features))
        if risk.shape != (len(h), 4):
            raise ValueError("risk context must contain sigma/up/down/uncertainty")
        # Semantic predictions are only exposed to arms already receiving F.
        visible_risk = risk if self.variant in ("R02", "R04", "R05", "R06", "R07") else torch.zeros_like(risk)
        if self.variant not in ("R05", "R06") and not self.risk_gate:
            visible_risk = visible_risk.clone(); visible_risk[:, 3] = 0.
        if self.risk_gate:
            visible_risk = visible_risk.clone(); visible_risk[:, 3] = risk[:, 3]
        return torch.cat((h, z_out, f_out, u_out, visible_risk), -1)

    def heads(self, context, inventory):
        if inventory.shape != (len(context), 14):
            raise ValueError("fourteen padded own-account features required")
        if self.variant != "R06":
            inventory = inventory.clone(); inventory[:, 6:] = 0.
        if self.variant not in ("R05", "R06") and self.risk_gate:
            # Gate-only ablations may use producer uncertainty in the external
            # controller without silently adding it to the learned actor.
            context = context.clone(); context[:, -1] = 0.
        values = torch.cat((context, inventory), -1)
        mean = self.policy_head(values)
        std = self.log_std.clamp(-4, 0).exp().expand_as(mean)
        return Independent(Normal(mean, std, validate_args=False), 1), self.critic_head(values).squeeze(-1)

    def request_from_action(self, raw_action, previous_request, context):
        if raw_action.shape != (len(context), self.action_dim):
            raise ValueError("raw Gaussian action dimensions differ from controller")
        if self.controller == "hierarchical":
            annual_sigma = context[:, -4].clamp_min(1e-5)*math.sqrt(self.annual_bars/self.prediction_horizon)
            target_vol = .8*torch.sigmoid(raw_action[:, 0])
            risk_cap = self.position_max*torch.sigmoid(raw_action[:, 1])
            participation = torch.sigmoid(-raw_action[:, 2])
            desired = torch.minimum(target_vol/annual_sigma, risk_cap)*participation
            delta = (desired-previous_request).clamp(-self.max_position_step, self.max_position_step)
        else:
            delta = self.max_position_step*torch.tanh(raw_action[:, 0])
        request = (previous_request+delta).clamp(self.position_min, self.position_max)
        if self.risk_gate:
            # Apply the deterministic gate inside the raw-action likelihood
            # transformation; never optimize likelihood of a clipped target.
            gated = context[:, -1] > self.gate_threshold
            request = torch.where(gated, (previous_request-self.max_position_step).clamp_min(self.position_min), request)
        return request

    def act_from_context(self, context, inventory, deterministic=False):
        distribution, value = self.heads(context, inventory)
        action = distribution.mean if deterministic else distribution.sample()
        request = self.request_from_action(action, inventory[:, 0], context)
        if self.controller == "ml_rule":
            request = ml_rule_request(context, inventory[:, 0], self)
        return {"raw_action": action, "request": request, "log_prob": distribution.log_prob(action), "value": value}

    def freeze_market(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name == "log_std" or name.startswith(("policy_head.", "critic_head.")))
        self.eval()
        return self


def ml_rule_request(context, previous_request, model):
    sigma = context[:, -4].clamp_min(1e-5)*math.sqrt(model.annual_bars/model.prediction_horizon)
    up, down = context[:, -3], context[:, -2]
    desired = (.1/sigma).clamp(0, model.position_max)
    desired = torch.where(down >= .5, torch.zeros_like(desired), desired)
    desired = torch.where(up > down, desired, desired*.5)
    if model.risk_gate:
        desired = torch.where(context[:, -1] > model.gate_threshold, torch.zeros_like(desired), desired)
    return (previous_request+(desired-previous_request).clamp(-model.max_position_step, model.max_position_step)).clamp(model.position_min, model.position_max)


def _eligible(arrays, ids, cfg, variant):
    ids = np.asarray(ids, np.int64)
    n, context = len(arrays["index_ns"]), int(cfg["policy"].get("context", 128))
    if ids.ndim != 1 or not len(ids) or np.any(ids < context-1) or np.any(ids >= n) or np.any(np.diff(ids) <= 0):
        raise ValueError("nonempty unique chronological raw decision ids with history required")
    available = np.asarray(arrays.get("available", arrays.get("decision_available", np.ones(n, bool))), bool)
    raw_finite = np.isfinite(_raw(arrays, cfg)).all(1)
    if "window_cache" in arrays:
        # The canonical window has its rolling TA restarted inside these128
        # raw bars. Gathering128 precomputed TA point rows would expand the
        # information history and is intentionally avoided on this path.
        valid = available[ids] & raw_finite[ids]
    else:
        invalid_prefix = np.r_[0, np.cumsum(~raw_finite)]
        valid = available[ids] & ((invalid_prefix[ids+1]-invalid_prefix[ids-context+1]) == 0)
    values = _market(arrays)
    used = (["old_z"] if variant == "R01" else ["new_z"] if variant in ("R03", "R04", "R05", "R06") else [])
    used += ["oracle"] if variant == "R07" else ["forecasts"] if variant in ("R02", "R04", "R05", "R06") else []
    used += ["uncertainty"] if variant in ("R05", "R06") else []
    for key in used:
        if key not in values:
            raise ValueError(f"active variant requires producer feature {key}")
        valid &= np.isfinite(values[key][ids]).all(1)
    return ids[valid]


def _risk_values(arrays, ids, required=False):
    keys = ("risk_sigma", "up_probability", "down_probability", "uncertainty_score")
    output = np.zeros((len(ids), 4), np.float32)
    for col, key in enumerate(keys):
        if key in arrays:
            output[:, col] = np.asarray(arrays[key])[ids]
        elif required and col < 3:
            raise ValueError("semantic risk_sigma/up_probability/down_probability required for risk controller")
    if not np.isfinite(output).all() or np.any(output[:, 0] < 0) or np.any((output[:, 1:3] < 0) | (output[:, 1:3] > 1)):
        raise ValueError("finite positive sigma and valid producer probabilities required")
    return output


def _inputs(arrays, ids, model, cfg, device="cpu"):
    ids = np.asarray(ids, np.int64)
    if "window_cache" in arrays:
        raw = np.asarray(arrays["window_cache"][ids], np.float32)
        if raw.shape != (len(ids), model.context, model.dims["raw"]) or not np.isfinite(raw).all():
            raise ValueError("canonical lazy raw-window cache has invalid shape or values")
    else:
        positions = ids[:, None]-model.context+1+np.arange(model.context)[None, :]
        raw = np.asarray(_raw(arrays, cfg)[positions], np.float32)
    values = _market(arrays)
    zkey = "old_z" if model.variant == "R01" else "new_z"
    fkey = "oracle" if model.variant == "R07" else "forecasts"
    z = _columns(values.get(zkey) if model.variant in ("R01", "R03", "R04", "R05", "R06") else None, ids, model.dims["z"])
    f = _columns(values.get(fkey) if model.variant in ("R02", "R04", "R05", "R06", "R07") else None, ids, model.dims["f"])
    u = _columns(values.get("uncertainty") if model.variant in ("R05", "R06") else None, ids, model.dims["u"])
    required = model.controller in ("hierarchical", "ml_rule") or model.risk_gate
    risk = _risk_values(arrays, ids, required)
    if model.risk_gate and "uncertainty_score" not in arrays:
        raise ValueError("explicit uncertainty gate requires a producer-calibrated uncertainty score")
    return tuple(torch.as_tensor(v, device=device) for v in (raw, z, f, u, risk))


def fit_normalizers(model, arrays, train_ids, cfg):
    """Policy-Train-only feature scales; no validation or Test rows are read."""
    values = _market(arrays)
    keys = {"raw": _raw(arrays, cfg), "z": values.get("old_z" if model.variant == "R01" else "new_z"),
            "f": values.get("oracle" if model.variant == "R07" else "forecasts"), "u": values.get("uncertainty")}
    for name, data in keys.items():
        used = name == "raw" or (name == "z" and model.variant in ("R01", "R03", "R04", "R05", "R06")) or (name == "f" and model.variant in ("R02", "R04", "R05", "R06", "R07")) or (name == "u" and model.variant in ("R05", "R06"))
        if not used:
            continue
        if data is None:
            continue
        part = np.asarray(data[np.asarray(train_ids, np.int64)], np.float64)
        if not np.isfinite(part).all():
            raise ValueError("normalizer must use finite policy Train rows")
        width = part.shape[1]
        getattr(model, name+"_mean")[:width].copy_(torch.from_numpy(part.mean(0).astype(np.float32)))
        getattr(model, name+"_std")[:width].copy_(torch.from_numpy(np.maximum(part.std(0), 1e-8).astype(np.float32)))


@torch.no_grad()
def encode_contexts(model, arrays, raw_ids, cfg, destination=None):
    ids = _eligible(arrays, raw_ids, cfg, model.variant)
    model.eval()
    device = next(model.parameters()).device
    if destination is not None:
        Path(destination).mkdir(parents=True, exist_ok=True)
    output = np.empty((len(ids), model.context_dim), np.float32) if destination is None else np.lib.format.open_memmap(Path(destination)/"contexts.npy", mode="w+", dtype=np.float32, shape=(len(ids), model.context_dim))
    batch = int(cfg["policy"].get("context_batch_size", 512))
    for start in range(0, len(ids), batch):
        panel = ids[start:start+batch]
        output[start:start+len(panel)] = model.encode_context(*_inputs(arrays, panel, model, cfg, device)).cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("finite frozen market encodings required")
    if destination is not None:
        np.save(Path(destination)/"context_raw_ids.npy", ids, allow_pickle=False)
        output.flush()
    return {"raw_ids": ids, "contexts": output, "variant": model.variant, "context_dim": model.context_dim}


_TEST_CONTEXT_VERIFIED = {}


def _market_encoder_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name == "log_std" or name.startswith(("policy_head.", "critic_head.")):
            continue
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _context_file_stat(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def encode_test_contexts_cached(model, policy_arrays, raw_ids, cfg, cache_dir):
    """Cache fixed market encodings across BC/RL and accounting sensitivities.

    The report caller verifies the global Test barrier first. This function
    requires the verified Test producer manifest and hashes the actual market
    encoder, semantic risks, availability and requested calendar ids. Policy
    and critic heads, leverage/accounting costs, and own-account features are
    deliberately absent: none of them changes a market encoding.
    """
    producer = policy_arrays.get("producer_manifest")
    if (not isinstance(producer, dict) or not producer.get("complete") or producer.get("split") != "test"
            or not producer.get("registration_sha256") or not isinstance(producer.get("artifact_sha256"), dict)):
        raise ValueError("cached Test contexts require a verified Test producer manifest")
    ids = np.asarray(raw_ids, np.int64)
    if ids.ndim != 1 or not len(ids) or np.any(np.diff(ids) <= 0):
        raise ValueError("cached Test contexts require unique chronological calendar ids")
    n = len(policy_arrays["index_ns"])
    if ids[0] < model.context-1 or ids[-1] >= n:
        raise ValueError("cached Test context calendar ids lack raw history or exceed the calendar")
    available = np.asarray(policy_arrays.get("available", policy_arrays.get("decision_available", np.ones(n, bool))), bool)[ids]
    risk = _risk_values(policy_arrays, ids, required=model.controller in ("hierarchical", "ml_rule") or model.risk_gate)
    signature = {"version": 1, "registration_sha256": producer["registration_sha256"],
                 "producer_artifact_sha256": producer["artifact_sha256"],
                 "encoder_sha256": _market_encoder_digest(model), "variant": model.variant,
                 "dims": model.dims, "context": model.context, "context_dim": model.context_dim,
                 "controller": model.controller, "risk_gate": model.risk_gate,
                 "source_input_group": cfg.get("policy", {}).get("source_input_group", "D03"),
                 "raw_ids_sha256": hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest(),
                 "available_sha256": hashlib.sha256(np.ascontiguousarray(available).tobytes()).hexdigest(),
                 "semantic_risk_sha256": hashlib.sha256(np.ascontiguousarray(risk).tobytes()).hexdigest()}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    root = Path(cache_dir); root.mkdir(parents=True, exist_ok=True)
    directory = root/key
    with (root/(key+".lock")).open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        marker = directory/"manifest.json"
        if not marker.exists():
            if directory.exists():
                raise ValueError("incomplete immutable Test market-context cache")
            with tempfile.TemporaryDirectory(prefix=key+".", dir=root) as temporary:
                working = Path(temporary)
                encoded = encode_contexts(model, policy_arrays, ids, cfg, working)
                manifest = {"complete": True, "key": key, "signature": signature,
                            "n_rows": len(encoded["raw_ids"]), "context_dim": model.context_dim,
                            "artifact_sha256": {name: _sha(working/name) for name in ("contexts.npy", "context_raw_ids.npy")}}
                _json(working/"manifest.json", manifest)
                # Both arrays and their provenance marker become visible in
                # one directory rename; readers never see partially written NPY.
                working.rename(directory)
        manifest = json.loads(marker.read_text())
        if not manifest.get("complete") or manifest.get("key") != key or manifest.get("signature") != signature:
            raise ValueError("Test market-context cache belongs to another frozen encoder/input signature")
        marker_sha = _sha(marker)
        expected_names = {"contexts.npy", "context_raw_ids.npy"}
        if set(manifest.get("artifact_sha256", {})) != expected_names:
            raise ValueError("Test market-context cache array set changed")
        stats = tuple((name, _context_file_stat(directory/name)) for name in sorted(expected_names))
        verified_key = (str(directory.resolve()), marker_sha)
        if _TEST_CONTEXT_VERIFIED.get(verified_key) != stats:
            if any(_sha(directory/name) != expected for name, expected in manifest["artifact_sha256"].items()):
                raise ValueError("frozen Test market-context array changed")
            _TEST_CONTEXT_VERIFIED[verified_key] = stats
        context = np.load(directory/"contexts.npy", mmap_mode="r", allow_pickle=False)
        context_ids = np.load(directory/"context_raw_ids.npy", mmap_mode="r", allow_pickle=False)
        if context.shape != (manifest["n_rows"], model.context_dim) or context_ids.shape != (manifest["n_rows"],):
            raise ValueError("frozen Test market-context dimensions changed")
        return {"raw_ids": context_ids, "contexts": context, "variant": model.variant, "context_dim": model.context_dim,
                "cache_key": key, "cache_manifest_path": str(marker.resolve()), "cache_manifest_sha256": marker_sha}


def _prices(arrays, cfg):
    key = "ohlcperp" if cfg["accounting"].get("mode") == "linear_perp" else "ohlcspot"
    if key not in arrays:
        raise ValueError(f"actual raw trade OHLC required: {key}")
    return arrays[key]


def _initial(arrays, ids, cfg, initial_position=None):
    source = arrays["ohlcmark"] if cfg["accounting"].get("mode") == "linear_perp" else _prices(arrays, cfg)
    state = account.initial_state(len(ids), "cpu", np.asarray(source[np.asarray(ids)-1, 3]), cfg)
    if initial_position is not None:
        positions = torch.as_tensor(initial_position, dtype=torch.float64)
        state["units"] = positions/state["prev_close"]
        state["cash"] = 1.-positions
        state["requested"], state["executed"] = positions.clone(), positions.clone()
    return state


def _advance(arrays, ids, state, request, cfg, available=None):
    ids = np.asarray(ids, np.int64)
    bars = np.asarray(_prices(arrays, cfg)[ids], np.float64)
    kwargs = {"decision_available": available}
    if cfg["accounting"].get("mode") == "linear_perp":
        mark = np.asarray(arrays["ohlcmark"][ids], np.float64)
        kwargs.update(mark_open=mark[:, 0], mark_low=mark[:, 2], mark_close=mark[:, 3],
                      funding_rate=np.asarray(arrays.get("funding_settlement_rate", np.zeros(len(arrays["index_ns"]))))[ids],
                      funding_event_count=np.asarray(arrays.get("funding_event_count", np.zeros(len(arrays["index_ns"]), np.int64)))[ids], funding_event=ids)
    return account.step(state, request, bars[:, 0], bars[:, 2], bars[:, 3], cfg, **kwargs)


@torch.no_grad()
def replay_cached(model, contexts, arrays, raw_ids, cfg, initial_state=None):
    ids = np.asarray(raw_ids, np.int64)
    if ids.ndim != 1 or not len(ids) or ids[0] < 1 or np.any(np.diff(ids) != 1):
        raise ValueError("replay must contain every consecutive raw-calendar bar")
    model.cpu().eval()
    cids, encoded = np.asarray(contexts["raw_ids"], np.int64), contexts["contexts"]
    positions = np.searchsorted(cids, ids)
    available = positions < len(cids)
    available[available] &= cids[positions[available]] == ids[available]
    state = _initial(arrays, ids[:1], cfg) if initial_state is None else {k: v.detach().cpu().clone() for k, v in initial_state.items()}
    if len(state["cash"]) != 1:
        raise ValueError("report replay requires exactly one carried account")
    source = arrays["ohlcmark"] if cfg["accounting"].get("mode") == "linear_perp" else _prices(arrays, cfg)
    if bool(state["alive"][0]) and not np.isclose(state["prev_close"].item(), source[ids[0]-1, 3], atol=1e-8, rtol=0):
        raise ValueError("fold replay must continue from the preceding raw close")
    records = []
    for row, raw_id in enumerate(ids):
        previous, active = state["prev_equity"].item(), bool(state["alive"][0])
        if available[row] and active:
            context = torch.from_numpy(np.asarray(encoded[positions[row]:positions[row]+1], np.float32))
            action = model.act_from_context(context, account.inventory(state, cfg, expanded=True).float(), deterministic=True)
            request = action["request"].double()
            raw_action = action["raw_action"][0].tolist()
        else:
            request, raw_action = state["requested"], [float("nan")]*model.action_dim
        state, diag = _advance(arrays, [raw_id], state, request, cfg, bool(available[row]))
        record = {k: bool(v[0]) if v.dtype == torch.bool else float(v[0]) for k, v in diag.items()}
        record.update(raw_id=int(raw_id), timestamp=pd.Timestamp(int(arrays["index_ns"][raw_id]), tz="UTC"),
                      equity_previous=previous, context_available=bool(available[row]), alive_before=active,
                      raw_action=raw_action)
        records.append(record)
    return pd.DataFrame(records), state


def path_score(ledger, cfg):
    if ledger.empty:
        raise ValueError("nonempty chronological validation ledger required")
    logs = ledger.training_log_return.to_numpy(float)
    penalty = float(cfg["accounting"].get("drawdown_penalty", .25)) if cfg["policy"].get("objective", "net_log_dd") == "net_log_dd" else 0.
    score = float(logs.sum()-penalty*ledger.max_dd_increment.sum())
    cvar_episodes = 0
    if cfg["policy"].get("objective") == "episodic_cvar_5pct":
        horizon = int(cfg["policy"].get("rollout_horizon", 64))
        cvar_episodes = len(logs)//horizon
        if cvar_episodes < 1:
            raise ValueError("CVaR selector needs complete nonoverlapping validation episodes")
        episode_returns = logs[:cvar_episodes*horizon].reshape(cvar_episodes, horizon).sum(1)
        score = float(lower_tail_cvar(torch.from_numpy(episode_returns)))
    return {"selection_score": score, "net_total_return": float(ledger.equity.iloc[-1]/ledger.equity_previous.iloc[0]-1),
            "max_drawdown": float(ledger.max_drawdown.max()), "liquidated": bool(ledger.liquidation_event.any()),
            "bankrupt": bool(ledger.bankrupt.any()), "n_rows": len(ledger), "n_decisions": int(ledger.decision_available.sum()),
            "cvar_validation_episodes": cvar_episodes}


@torch.no_grad()
def _bc_teacher(arrays, train_ids, cfg, rng):
    horizon = int(cfg["policy"].get("prediction_horizon_bars", 16))
    # Targets terminate strictly inside policy Train, including next-open lag.
    upper = int(np.asarray(train_ids)[-1])+1
    ids = np.asarray(train_ids, np.int64)
    ids = ids[ids+horizon+1 < upper]
    maximum = int(cfg["policy"].get("teacher_max_rows", 8192))
    if len(ids) > maximum:
        ids = np.sort(rng.choice(ids, maximum, replace=False))
    if not len(ids):
        raise ValueError("policy Train is too short for execution-aware BC labels")
    current_grid = np.linspace(float(cfg["accounting"].get("position_min", 0.)), float(cfg["accounting"].get("position_max", 2.)), 7)
    deltas = np.linspace(-1, 1, 5)*float(cfg["accounting"].get("max_position_step", .08))
    target = np.clip(current_grid[:, None]+deltas[None, :], cfg["accounting"].get("position_min", 0.), cfg["accounting"].get("position_max", 2.))
    scores = np.empty((len(ids), len(current_grid), len(deltas)), np.float32)
    for start in range(0, len(ids), 256):
        panel = ids[start:start+256]
        raw = np.repeat(panel, len(current_grid)*len(deltas))
        current = np.tile(np.repeat(current_grid, len(deltas)), len(panel))
        requested = torch.from_numpy(np.tile(target.reshape(-1), len(panel)))
        state = _initial(arrays, raw, cfg, current)
        for step in range(horizon+1):
            state, _ = _advance(arrays, raw+step, state, requested, cfg, available=(step == 0))
        utility = state["log_wealth"]-float(cfg["accounting"].get("drawdown_penalty", .25))*state["max_drawdown"]
        utility = torch.where(torch.isfinite(utility), utility, torch.full_like(utility, -100.))
        scores[start:start+len(panel)] = utility.numpy().reshape(len(panel), len(current_grid), len(deltas))
    best = np.take_along_axis(np.broadcast_to(target, scores.shape), scores.argmax(-1)[..., None], axis=-1)[..., 0]
    return ids, current_grid, best


def _save(path, model, cfg, task, regsha, stage, step):
    obj = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "dims": model.dims,
           "config": cfg, "task": task, "variant": model.variant, "registration_sha256": regsha,
           "stage": stage, "step": int(step)}
    temporary = Path(str(path)+".tmp")
    torch.save(obj, temporary); temporary.replace(path)


def load_policy(path, cfg=None, registration_sha=None):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if registration_sha is not None and obj["registration_sha256"] != registration_sha:
        raise ValueError("policy checkpoint belongs to another registered study")
    settings = obj["config"] if cfg is None else effective_config(cfg, obj["task"])
    model = CachedTailPolicy(obj["dims"], settings, obj["variant"])
    model.load_state_dict(obj["state_dict"], strict=True)
    return model.cpu().eval(), obj


def _context_rows(cache, ids):
    locations = np.searchsorted(cache["raw_ids"], ids)
    if np.any(locations >= len(cache["raw_ids"])) or not np.array_equal(cache["raw_ids"][locations], ids):
        raise ValueError("all rollout observations need frozen active market contexts")
    return torch.from_numpy(np.asarray(cache["contexts"][locations], np.float32))


def _starts(ids, horizon):
    boundaries = np.flatnonzero(np.r_[True, np.diff(ids) != 1, True])
    chunks = [np.arange(a, b-horizon, dtype=np.int64) for a, b in zip(boundaries[:-1], boundaries[1:]) if b-a > horizon]
    return np.concatenate(chunks) if chunks else np.empty(0, np.int64)


def lower_tail_cvar(values, alpha=.05):
    """Exact empirical lower-tail mean, with a fractional boundary sample."""
    if values.ndim != 1 or not len(values) or not 0 < alpha <= 1:
        raise ValueError("CVaR requires a nonempty episode-return vector and valid alpha")
    sorted_values = torch.sort(values).values
    mass = alpha*len(values)
    whole = int(math.floor(mass))
    result = sorted_values[:whole].sum()
    if whole < len(values):
        result = result+(mass-whole)*sorted_values[whole]
    return result/mass


@torch.no_grad()
def collect_rollout(model, prior, cache, arrays, raw_ids, cfg):
    batch, span = raw_ids.shape
    horizon = span-1
    context = _context_rows(cache, raw_ids)
    state = _initial(arrays, raw_ids[:, 0], cfg)
    saved = {k: [] for k in ("inventory", "raw_action", "old_log_prob", "old_value", "reward", "terminals", "active", "net_log")}
    model.eval()
    for step in range(horizon):
        inv = account.inventory(state, cfg, expanded=True).float()
        action = model.act_from_context(context[:, step], inv)
        active = state["alive"].clone()
        state, diag = _advance(arrays, raw_ids[:, step], state, action["request"].double(), cfg)
        objective = cfg["policy"].get("objective", "net_log_dd")
        reward = diag["training_log_return"]
        if objective == "net_log_dd":
            reward = reward-float(cfg["accounting"].get("drawdown_penalty", .25))*diag["max_dd_increment"]
        reward = torch.where(active, reward, torch.zeros_like(reward))*float(cfg["policy"].get("reward_scale", 100.))
        for key, value in (("inventory", inv), ("raw_action", action["raw_action"]), ("old_log_prob", action["log_prob"]),
                           ("old_value", action["value"]), ("reward", reward.float()), ("terminals", ~state["alive"]),
                           ("active", active), ("net_log", diag["training_log_return"])):
            saved[key].append(value)
    stacked = {k: torch.stack(v, 1) for k, v in saved.items()}
    bootstrap = model.heads(context[:, -1], account.inventory(state, cfg, expanded=True).float())[1]
    p = cfg["policy"]
    advantages, returns = compute_gae(stacked["reward"].T, stacked["old_value"].T, stacked["terminals"].T,
                                      bootstrap, float(p.get("gamma", .99)), float(p.get("gae_lambda", .95)))
    episodes = stacked["net_log"].sum(1).float()
    cvar = lower_tail_cvar(episodes)
    if p.get("objective") == "episodic_cvar_5pct":
        # Rockafellar-Uryasev envelope gradient for lower-return CVaR:
        # E[(episode_return-q)*1(return<=q)*score]/alpha. This uses genuine
        # full-rollout returns, not the worst individual bars.
        q = torch.sort(episodes).values[max(0, math.ceil(.05*len(episodes))-1)]
        episodic_adv = torch.where(episodes <= q, (episodes-q)/.05, torch.zeros_like(episodes))
        advantages = episodic_adv[None, :].expand(horizon, batch)*float(p.get("reward_scale", 100.))
    result = {k: value.flatten(0, 1).detach() for k, value in stacked.items() if k not in ("reward", "terminals", "net_log")}
    result.update(contexts=context[:, :-1].flatten(0, 1), advantages=advantages.T.flatten().detach(), returns=returns.T.flatten().detach())
    return {"batch": result, "episode_lower5_cvar": float(cvar), "episodes": batch,
            "tail_effective_mass": .05*batch, "reward_mean": float(stacked["reward"].mean()),
            "liquidated_rollouts": int((~state["alive"]).sum())}


def ppo_update(model, prior, optimizer, rollout, cfg):
    data, p = rollout["batch"], cfg["policy"]
    active = data["active"].float()
    live = data["advantages"][data["active"]]
    if not len(live):
        raise ValueError("rollout contains no live policy decisions")
    advantage = ((data["advantages"]-live.mean())/live.std(unbiased=False).clamp_min(1e-8))*active
    with torch.no_grad():
        reference = prior.heads(data["contexts"], data["inventory"])[0]
    diagnostics = []
    model.train(); prior.eval()
    for _ in range(int(p.get("ppo_epochs", 2))):
        for ids in torch.randperm(len(active)).split(int(p.get("minibatch_size", 128))):
            current, value = model.heads(data["contexts"][ids], data["inventory"][ids])
            ratio = (current.log_prob(data["raw_action"][ids])-data["old_log_prob"][ids]).exp()
            weight, denominator = active[ids], active[ids].sum().clamp_min(1)
            clipped = ratio.clamp(1-float(p.get("clip_ratio", .2)), 1+float(p.get("clip_ratio", .2)))
            actor = -(torch.minimum(ratio*advantage[ids], clipped*advantage[ids])*weight).sum()/denominator
            critic = ((value-data["returns"][ids]).square()*weight).sum()/denominator
            ref = Independent(Normal(reference.base_dist.loc[ids], reference.base_dist.scale[ids], validate_args=False), 1)
            prior_kl = (kl_divergence(current, ref)*weight).sum()/denominator
            entropy = (current.entropy()*weight).sum()/denominator
            loss = actor+float(p.get("value_coefficient", .5))*critic+float(p.get("bc_prior", .01))*prior_kl-float(p.get("entropy_coefficient", .001))*entropy
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite tail-transfer PPO objective")
            before = model.policy_head[-1].weight.detach().clone()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], float(p.get("gradient_clip", 10.)), error_if_nonfinite=True)
            optimizer.step()
            diagnostics.append((float(actor.detach()), float(critic.detach()), float(prior_kl.detach()), float(norm), float((model.policy_head[-1].weight-before).norm().detach())))
    mean = np.mean(diagnostics, axis=0)
    return {"policy_loss": float(mean[0]), "critic_loss": float(mean[1]), "prior_kl": float(mean[2]),
            "gradient_norm": float(mean[3]), "policy_parameter_delta": float(mean[4]), "optimizer_steps": len(diagnostics),
            "transitions": len(active), "episode_lower5_cvar": rollout["episode_lower5_cvar"],
            "tail_effective_mass": rollout["tail_effective_mass"], "liquidated_rollouts": rollout["liquidated_rollouts"]}


def train_cached(cfg, task, arrays, out, registration_sha):
    """Fresh BC then head-only PPO. Only policy Train/full outer Val are read."""
    cfg = effective_config(cfg, task)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    marker = out/"fit.json"
    if marker.exists():
        fit = json.loads(marker.read_text())
        if fit.get("registration_sha256") != registration_sha or fit.get("task") != task or not fit.get("complete"):
            raise ValueError("foreign or incomplete policy fit marker")
        if any(_sha(out/name) != value for name, value in fit["artifact_sha256"].items()):
            raise ValueError("immutable policy fit artifacts changed")
        return fit
    torch.set_num_threads(int(cfg.get("runtime", {}).get("cpu_threads", 1)))
    torch.use_deterministic_algorithms(True)
    seed = int(task["seed"])
    random.seed(seed); np.random.seed(seed); torch.random.default_generator.manual_seed(seed)
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    variant = task.get("policy_variant", "R04")
    train_ids = _eligible(arrays, arrays["train_ids"], cfg, variant)
    val_ids = np.asarray(arrays["val_ids"], np.int64)
    if not len(train_ids) or not len(val_ids) or np.any(np.diff(val_ids) != 1) or train_ids[-1] >= val_ids[0]:
        raise ValueError("strictly chronological policy Train and full-calendar Val required")
    if "producer_label_end_ns" in arrays and int(arrays["producer_label_end_ns"]) >= int(arrays["index_ns"][train_ids[0]]):
        raise ValueError("producer selection/calibration must finish before policy Train")
    model = CachedTailPolicy(_dims(arrays, cfg), cfg, variant).cpu()
    initial_state_sha = hashlib.sha256(b"".join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    fit_normalizers(model, arrays, train_ids, cfg)
    teacher_ids, current_grid, targets = _bc_teacher(arrays, train_ids, cfg, rng)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["policy"].get("bc_learning_rate", .0003)), weight_decay=0.)
    bc_history = []
    for step in range(1, int(cfg["policy"].get("bc_steps", 400))+1):
        rows = rng.integers(0, len(teacher_ids), int(cfg["policy"].get("bc_batch_size", 64)))
        col = rng.integers(0, len(current_grid), len(rows))
        ids = teacher_ids[rows]
        state = _initial(arrays, ids, cfg, current_grid[col])
        inv = account.inventory(state, cfg, expanded=True).float()
        model.train()
        context = model.encode_context(*_inputs(arrays, ids, model, cfg))
        distribution, _ = model.heads(context, inv)
        request = model.request_from_action(distribution.mean, inv[:, 0], context)
        loss = ((request-torch.from_numpy(targets[rows, col]))/model.max_position_step).square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite BC loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["policy"].get("gradient_clip", 10.)), error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == int(cfg["policy"].get("bc_steps", 400)):
            bc_history.append({"step": step, "bc_loss": float(loss.detach()), "elapsed_seconds": time.monotonic()-started})
            _json(out/"progress.json", {"stage": "bc", "step": step, "elapsed_seconds": time.monotonic()-started})
    _save(out/"bc.pt", model, cfg, task, registration_sha, "BC", int(cfg["policy"].get("bc_steps", 400)))
    model.freeze_market()
    prior = copy.deepcopy(model).requires_grad_(False).eval()
    market_hash = hashlib.sha256(b"".join(v.detach().numpy().tobytes() for k, v in model.state_dict().items() if not k.startswith(("policy_head.", "critic_head.")) and k != "log_std")).hexdigest()
    all_context_ids = np.unique(np.r_[train_ids, _eligible(arrays, val_ids, cfg, variant)])
    cache = encode_contexts(model, arrays, all_context_ids, cfg, out)
    bc_ledger, _ = replay_cached(model, cache, arrays, val_ids, cfg)
    bc_metrics = path_score(bc_ledger, cfg)
    bc_ledger.to_parquet(out/"bc_validation.parquet", index=False)
    p = cfg["policy"]
    optimizer = torch.optim.AdamW([{"params": [*model.policy_head.parameters(), model.log_std], "lr": float(p.get("actor_learning_rate", .0001))},
                                   {"params": model.critic_head.parameters(), "lr": float(p.get("critic_learning_rate", .0003))}], weight_decay=0.)
    starts = _starts(train_ids, int(p.get("rollout_horizon", 64)))
    if not len(starts):
        raise ValueError("continuous policy Train rollouts required")
    best, selected_step, history, updates = -math.inf, 0, [], int(p.get("ppo_updates", 200))
    # BC-only and fixed ML rules have no policy-gradient fit. Their actual
    # consumed update count is explicit rather than claiming a 200-update run.
    if task.get("stage") == "BC" or model.controller == "ml_rule":
        updates = 0
        _save(out/"selected.pt", model, cfg, task, registration_sha, "BC" if task.get("stage") == "BC" else "ML_rule", 0)
        bc_ledger.to_parquet(out/"selected_validation.parquet", index=False)
        best = bc_metrics["selection_score"]
    for update in range(1, updates+1):
        chosen = rng.choice(starts, int(p.get("rollout_batch_size", 16)), replace=True)
        raw_ids = train_ids[chosen[:, None]+np.arange(int(p.get("rollout_horizon", 64))+1)[None, :]]
        rollout = collect_rollout(model, prior, cache, arrays, raw_ids, cfg)
        diagnostics = ppo_update(model, prior, optimizer, rollout, cfg)
        diagnostics["update"] = update
        if update in p.get("selection_updates", [100, 125, 150, 175, 200]) or update == updates:
            ledger, _ = replay_cached(model, cache, arrays, val_ids, cfg)
            metrics = path_score(ledger, cfg)
            diagnostics["validation"] = metrics
            if metrics["selection_score"] > best:
                best, selected_step = metrics["selection_score"], update
                _save(out/"selected.pt", model, cfg, task, registration_sha, "RL", update)
                ledger.to_parquet(out/"selected_validation.parquet", index=False)
        history.append(diagnostics)
        if update == 1 or update % 25 == 0 or update == updates:
            _json(out/"progress.json", {"stage": "PPO", "update": update, "selected_update": selected_step, "elapsed_seconds": time.monotonic()-started})
    final_market_hash = hashlib.sha256(b"".join(v.detach().numpy().tobytes() for k, v in model.state_dict().items() if not k.startswith(("policy_head.", "critic_head.")) and k != "log_std")).hexdigest()
    if final_market_hash != market_hash:
        raise RuntimeError("PPO changed the frozen market representation")
    _json(out/"history.json", {"bc": bc_history, "ppo": history})
    artifacts = ["bc.pt", "selected.pt", "contexts.npy", "context_raw_ids.npy", "history.json", "bc_validation.parquet", "selected_validation.parquet"]
    fit = {"complete": True, "registration_sha256": registration_sha, "task": task,
           "selected_update": selected_step, "actual_ppo_updates": updates, "actual_bc_steps": int(p.get("bc_steps", 400)),
           "bc_validation": bc_metrics, "selected_validation_score": best,
           "market_encoder_frozen_during_ppo": True, "market_encoder_sha256": market_hash,
           "initial_model_sha256": initial_state_sha,
           "policy_train_ids_sha256": hashlib.sha256(np.ascontiguousarray(train_ids).tobytes()).hexdigest(),
           "bc_teacher_ids_sha256": hashlib.sha256(np.ascontiguousarray(teacher_ids).tobytes()).hexdigest(),
           "actual_ppo_transitions": sum(row["transitions"] for row in history),
           "actual_ppo_optimizer_steps": sum(row["optimizer_steps"] for row in history),
           "ppo_objective": p["objective"], "cvar_threshold_method": "ceil_order_statistic_empirical_left_quantile",
           "cvar_return_unit": "whole_rollout_net_log_return", "cvar_alpha": .05,
           "configured_rollout_batch_size": int(p.get("rollout_batch_size", 16)),
           "configured_rollout_horizon": int(p.get("rollout_horizon", 64)),
           "policy_train_rows": len(train_ids), "validation_calendar_rows": len(val_ids), "cached_market_rows": len(cache["raw_ids"]),
           "producer_policy_chronology": "producer_fit10months_validate2months_policyTrain12months",
           "oracle_diagnostic": variant == "R07", "test_read": False, "elapsed_seconds": time.monotonic()-started,
           "artifact_sha256": {name: _sha(out/name) for name in artifacts}}
    _json(marker, fit)
    return fit
