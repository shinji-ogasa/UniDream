"""A bounded, gap-preserving WM -> causal-input teacher BC -> imagination AC fork.

No selector, economic validation, oracle relabel or mainline resume occurs here.
The WM is fitted on T, so its T predictions are in-sample; causal input windows
are not a claim of out-of-fold predictions. No predictive checkpoint selection
occurs in this screen: WM endpoint700 is fixed.
All three AC endpoints start from the identical frozen BC.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from unidream.actor_critic.bc_pretrain import BCPretrainer
from unidream.actor_critic.critic import Critic
from unidream.actor_critic.imagination_ac import ImagACTrainer
from unidream.experiments.bc_setup import prepare_bc_setup
from unidream.experiments.checkpointing import atomic_torch_save
from unidream.experiments.overlay_teacher import apply_benchmark_overlay_teacher
from unidream.experiments.wm_rl_inputs import MarketSequenceDataset, _frame, _mask, sequence_masks
from unidream.experiments.wm_rl_policy import encode_fixed_context
from unidream.world_model.train_wm import WorldModelTrainer, build_ensemble

HEADS = ("return", "vol", "drawdown", "crash", "drawdown_excess",
         "position_utility", "overweight_advantage", "recovery")
AC_ARMS = {
    "ac_decay_dd25": {"alpha_final": .05, "relative_dd_coef": 2.5},
    "ac_anchor_dd25": {"alpha_final": .35, "relative_dd_coef": 2.5},
    "ac_anchor_dd50": {"alpha_final": .35, "relative_dd_coef": 5.0},
}


def _digest(value):
    array = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode()); h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def contiguous_segments(mask):
    """Return original-grid half-open runs; never bridge a false row."""
    mask = np.asarray(mask)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("mask must be a one-dimensional boolean array")
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))


def normalize_auxiliary(raw, *, fit_mask, available, names, ac_cfg):
    """Old predictive-state float32 normalization arithmetic, fitted on T only."""
    raw = np.asarray(raw)
    n = len(raw)
    for mask in (fit_mask, available):
        if np.asarray(mask).dtype != np.bool_ or np.asarray(mask).shape != (n,):
            raise ValueError("auxiliary masks must be strict aligned booleans")
    if raw.shape != (n, 42) or len(names) != 42 or len(set(names)) != 42:
        raise ValueError("exactly 42 ordered predictive features required")
    selected = np.asarray(fit_mask) & np.asarray(available)
    if not selected.any() or not np.isfinite(raw[available]).all():
        raise ValueError("finite auxiliary support and nonempty T required")
    train = np.asarray(raw[selected], dtype=np.float32)
    mean, std = train.mean(axis=0), train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("nonfinite T auxiliary normalizer")
    standardize = ac_cfg.get("wm_predictive_state_standardize", True)
    if not isinstance(standardize, bool):
        raise ValueError("wm_predictive_state_standardize must be boolean")
    clip, scale = float(ac_cfg.get("wm_predictive_state_clip", 5.0)), float(ac_cfg.get("wm_predictive_state_scale", 1.0))
    if not np.isfinite([clip, scale]).all() or clip <= 0 or scale <= 0:
        raise ValueError("positive finite auxiliary clip/scale required")
    values = raw[available].astype(np.float32)
    if standardize:
        values = (values - mean) / std
    values = (np.clip(values, -clip, clip) * scale).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite normalized auxiliary values")
    result = np.full((n, 42), np.nan, dtype=np.float32)
    result[available] = values
    normalizer = {"feature_names": list(names), "mean": mean.tolist(), "std": std.tolist(),
        "standardize": standardize, "clip": clip, "scale": scale, "arithmetic": "float32_mean_std_ddof0_std_lt1e-6_to1",
        "fit_count": int(selected.sum()), "fit_mask_sha256": _digest(selected),
        "fit_values_sha256": _digest(train)}
    return result, normalizer


def build_segmented_teacher(*, returns, auxiliary, eligible, cfg):
    """Apply the existing low-frequency teacher separately to each T run."""
    bc, ac = cfg.get("bc", {}), cfg.get("ac", {})
    if bc.get("benchmark_overlay_teacher_mode") != "lowfreq_wm_overlay" or not bc.get("benchmark_overlay_teacher", False):
        raise ValueError("enabled lowfreq_wm_overlay teacher required")
    if bc.get("benchmark_overlay_lowfreq_base") != 0 or bc.get("benchmark_overlay_lowfreq_min_hold") != 32:
        raise ValueError("registered teacher requires base0/min_hold32")
    n = len(eligible)
    if np.shape(auxiliary) != (n, 42) or np.shape(returns) != (n,):
        raise ValueError("aligned teacher data required")
    if not np.isfinite(auxiliary[eligible]).all() or not np.isfinite(returns[eligible]).all():
        raise ValueError("teacher support must be finite")
    positions = np.full(n, np.nan, dtype=np.float32)
    for start, end in contiguous_segments(eligible):
        positions[start:end] = apply_benchmark_overlay_teacher(
            np.ones(end - start, dtype=np.float32), bc_cfg=bc, ac_cfg=ac,
            reward_cfg=cfg.get("reward", {}), advantage_values=auxiliary[start:end], returns=returns[start:end])
    if not np.isfinite(positions[eligible]).all():
        raise ValueError("nonfinite teacher positions")
    return positions


def segmented_controller_states(actor, positions, eligible):
    states = np.full((len(eligible), 4), np.nan, dtype=np.float32)
    if actor.inventory_dim != 4:
        raise ValueError("registered four-state controller required")
    for start, end in contiguous_segments(eligible):
        states[start:end] = actor.controller_states_from_positions(positions[start:end])
    return states


def chunk_origins_and_targets(positions, states, eligible, *, chunk_size=4):
    """Existing first-switch chunk target, all input features at chunk origin.

    Chunk future teacher labels are supervised targets, not future inputs.
    Each segment drops its own incomplete tail. No cross-gap chunk is created.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("positive integer chunk size required")
    origins, target_origins = [], []
    for start, end in contiguous_segments(eligible):
        for origin in range(start, end - chunk_size + 1, chunk_size):
            target = positions[origin:origin + chunk_size]
            switched = np.flatnonzero(np.abs(target - (states[origin, 0] + 1)) > 1e-8)
            origins.append(origin)
            target_origins.append(origin + (int(switched[0]) if len(switched) else 0))
    if not origins:
        raise ValueError("no complete BC chunks")
    return np.asarray(origins, dtype=np.int64), np.asarray(target_origins, dtype=np.int64)


def bind_controller_anchor_bank(trainer, *, z, h, positions, states, auxiliary, origins):
    """Replace legacy compressed-series inventory with original-grid T states."""
    if np.shape(states) != (len(origins), 4) or not np.isfinite(states).all():
        raise ValueError("finite full controller bank required")
    trainer.set_oracle_data(z, h, positions, regime_probs=None, advantage_values=auxiliary)
    trainer._oracle_inventory = torch.as_tensor(states, dtype=torch.float32, device=trainer.device)
    traded = np.abs((positions - 1) - states[:, 0]) > 1e-8
    n_pos, n_neg = int(traded.sum()), int((~traded).sum())
    trainer._oracle_trade_pos_weight = (torch.tensor(n_neg / n_pos, device=trainer.device, dtype=torch.float32)
                                         if n_pos and n_neg else None)
    size = int(trainer.nn_anchor_bank_size)
    anchor_ix = (np.arange(len(origins), dtype=np.int64) if len(origins) <= size
                 else np.linspace(0, len(origins) - 1, num=size, dtype=np.int64)) if size > 0 else np.empty(0, dtype=np.int64)
    trainer._oracle_anchor_inventory = (trainer._oracle_inventory[torch.as_tensor(anchor_ix, device=trainer.device)]
                                         if len(anchor_ix) else None)
    return {"origin_indices": origins.tolist(), "anchor_origin_indices": origins[anchor_ix].tolist(),
            "controller_states_sha256": _digest(states), "trade_positive": n_pos, "trade_negative": n_neg,
            "inventory_override": "original_grid_segmented_full4_not_compressed_previous_target"}


def _validate_cfg(cfg, regime_probs):
    if regime_probs is not None:
        raise ValueError("this bounded matrix explicitly uses regime_dim0")
    cfg = copy.deepcopy(cfg)
    wm, ac, bc = (cfg.setdefault(name, {}) for name in ("world_model", "ac", "bc"))
    required = ((wm, "reward_mode", "market_log_return"), (wm, "action_context", "actionless"),
        (wm, "train_sequence_length", 128), (wm, "batch_size", 32), (wm, "max_steps", 700),
        (wm, "max_seq_len", 128), (wm, "idm_scale", 0), (ac, "reward_objective", "benchmark_absolute_constraint"),
        (ac, "controller_state_dim", 4), (ac, "regime_dim", 0), (ac, "horizon", 4),
        (ac, "max_steps", 300), (ac, "alpha_init", .35), (bc, "n_epochs", 5), (bc, "batch_size", 512))
    for section, key, value in required:
        if section.get(key) != value or isinstance(section.get(key), bool):
            raise ValueError(f"registered trainer requires {key}={value!r}")
    if cfg.get("data", {}).get("seq_len") != 64:
        raise ValueError("fixed data.seq_len64 required")
    if ac.get("use_wm_predictive_state") is not True or tuple(ac.get("wm_predictive_state_heads", ())) != HEADS:
        raise ValueError("registered ordered eight heads required")
    if bc.get("sirl_hidden", 0) != 0 or bc.get("class_balanced", False) or bc.get("path_aux_coef", 0) != 0 or bc.get("self_condition_prob", 0) != 0:
        raise ValueError("bounded BC does not support SIRL/class weighting/path/self-conditioning")
    if bc.get("transition_advantage_relabel", False) or bc.get("transition_route_labels", False) or bc.get("sample_quality_mode", "none") != "none":
        raise ValueError("hindsight relabel or outcome sample-quality is forbidden")
    if ac.get("online_wm_interval", 0) != 0 or ac.get("critic_pretrain_steps", 0) != 0:
        raise ValueError("registered arms have no online WM or critic-only stage")
    if float(wm.get("reward_scale", 0)) <= 0:
        raise ValueError("market reward training must be active")
    wm["require_target_gradient_coverage"] = True
    ac["advantage_dim"] = 42
    return cfg


def train_wm_bc_ac(*, features, returns, feature_eligible, target_eligible,
                   train_mask, cfg, output_dir, seed=7, device="cpu",
                   wm_val_mask=None, regime_probs=None):
    """Train one WM+BC and the three fixed AC endpoints in a new directory.

    ``features`` are already normalized by the caller using T only. The caller
    controls the physical data extent. ``returns`` are separate raw labels.
    This wrapper never selects a checkpoint or opens later files; wm_val_mask
    must be None for the registered fixed endpoint700 screen.
    """
    cfg = _validate_cfg(cfg, regime_probs)
    if isinstance(seed, bool) or seed != 7:
        raise ValueError("registered seed7 required")
    index = _frame(features)
    n = len(features)
    fm, tm, train = (_mask(v, index, name) for v, name in (
        (feature_eligible, "feature_eligible"), (target_eligible, "target_eligible"), (train_mask, "train_mask")))
    if not isinstance(returns, pd.Series) or not returns.index.equals(index):
        raise ValueError("returns must be a full-grid aligned Series")
    train_rows = train & fm & tm
    if not train_rows.any():
        raise ValueError("no eligible training rows")
    if wm_val_mask is not None:
        raise ValueError("fixed endpoint700 screen forbids WM validation checkpoint selection")
    val = None
    dataset = MarketSequenceDataset(features, feature_eligible=fm, target_eligible=tm,
        row_mask=train, seq_len=128, returns=returns)
    val_dataset = None
    if len(dataset) == 0 or (val_dataset is not None and len(val_dataset) == 0):
        raise ValueError("no full128 eligible WM train/validation sequences")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "resolved_training_config.json", cfg)
    _seed(seed)
    ensemble = build_ensemble(features.shape[1], cfg)
    wm = WorldModelTrainer(ensemble, cfg, device=device)
    wm_log = wm.train_on_dataset(dataset, val_dataset=val_dataset, max_steps=700,
        checkpoint_path=str(output / "world_model.pt"),
        coverage_context={"phase": "bounded_market_wm", "seed": seed})
    if wm.global_step != 700:
        raise RuntimeError("WM must execute exactly700 training updates")
    if not wm.target_gradient_coverage_passes():
        raise RuntimeError("all active WM target/gradient rows must pass")
    names = wm.predictive_feature_names()
    if len(names) != 42:
        raise ValueError("WM must expose exactly42 active auxiliary predictions")
    ensemble.eval()
    for parameter in ensemble.parameters():
        parameter.requires_grad_(False)
    encoded = encode_fixed_context(ensemble, features.to_numpy(), index, fm,
        context_length=64, batch_size=int(cfg.get("run", {}).get("encode_batch_size", 64)), device=device)
    available = encoded["available"]
    ix = np.flatnonzero(available)
    if not len(ix):
        raise ValueError("no fixed64 encoded origins")
    predictions = wm.predict_auxiliary_from_encoded(encoded["z"][ix], encoded["h"][ix],
        features.iloc[ix].to_numpy(), batch_size=int(cfg.get("run", {}).get("auxiliary_batch_size", 8192)))
    if tuple(predictions) != HEADS:
        raise ValueError("WM auxiliary head inventory/order mismatch")
    raw = np.full((n, 42), np.nan, dtype=np.float32)
    raw[ix] = np.concatenate([predictions[name] for name in HEADS], axis=1)
    auxiliary, normalizer = normalize_auxiliary(raw, fit_mask=train_rows, available=available, names=names, ac_cfg=cfg["ac"])
    teacher_mask = train_rows & available
    return_values = np.full(n, np.nan, dtype=np.float64)
    selected_returns = np.asarray(returns.to_numpy()[teacher_mask])
    if selected_returns.dtype.kind not in "fiu" or not np.isfinite(selected_returns).all():
        raise ValueError("selected teacher returns must be finite real numbers")
    return_values[teacher_mask] = selected_returns
    positions = build_segmented_teacher(returns=return_values, auxiliary=auxiliary, eligible=teacher_mask, cfg=cfg)
    origins = np.flatnonzero(teacher_mask)
    action_values = np.asarray(cfg.get("actions", {}).get("values", [0, .5, 1, 1.25]), dtype=np.float32)
    setup = prepare_bc_setup(ensemble=ensemble, oracle_action_values=action_values,
        oracle_positions=positions[origins], oracle_values=None, train_regime_probs=None,
        outcome_edge=None, ac_cfg=cfg["ac"], bc_cfg=cfg["bc"], reward_cfg=cfg["reward"],
        oracle_teacher_mode="lowfreq_wm_overlay")
    actor = setup["actor"].to(device)
    states = segmented_controller_states(actor, positions, teacher_mask)
    # prepare_bc_setup builds a compressed transition table. Replace it with
    # original-segment pre-action states before it can influence BC or inference.
    current = actor.target_indices(torch.tensor(states[origins, 0] + 1)).cpu().numpy()
    target = actor.target_indices(torch.tensor(positions[origins])).cpu().numpy()
    support = np.zeros((1, len(action_values), len(action_values)), dtype=np.float32)
    np.add.at(support, (np.zeros(len(origins), dtype=int), current, target), 1)
    actor.support_transition_counts = support
    bc_origins, bc_target_origins = chunk_origins_and_targets(positions, states, teacher_mask,
        chunk_size=int(cfg["bc"].get("chunk_size", 4)))
    bc_kwargs = {key: value for key, value in cfg["bc"].items()
                 if key in inspect.signature(BCPretrainer).parameters and key not in {"actor", "z_dim", "h_dim", "device"}}
    bc = BCPretrainer(actor, ensemble.get_z_dim(), ensemble.get_d_model(), device=device, **bc_kwargs)
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    bz, bh, bs, ba = (tensor(value[bc_origins]) for value in (encoded["z"], encoded["h"], states, auxiliary))
    bp = tensor(positions[bc_target_origins])
    traded = torch.abs((bp - 1) - bs[:, 0]) > 1e-8
    positive, negative = int(traded.sum()), int((~traded).sum())
    pos_weight = tensor(negative / positive) if positive and negative else None
    bc_log = []
    actor.train()
    for epoch in range(5):
        order = torch.randperm(len(bc_origins), device=device)
        total = 0.0
        for start in range(0, len(order), 512):
            batch = order[start:start + 512]
            loss = bc._bc_loss(bz[batch], bh[batch], bp[batch], inventory=bs[batch],
                advantage=ba[batch], trade_pos_weight=pos_weight)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite BC loss")
            bc.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 10)
            bc.optimizer.step()
            total += float(loss.detach()) * len(batch)
        bc_log.append({"epoch": epoch + 1, "loss": total / len(order)})
        print(f"[MaskedBC] epoch={epoch + 1}/5 loss={bc_log[-1]['loss']:.8g}", flush=True)
    actor.eval()
    atomic_torch_save(copy.deepcopy(actor).cpu(), str(output / "bc_actor_full.pt"))
    atomic_torch_save({"actor": actor.state_dict(), "optimizer": bc.optimizer.state_dict()}, str(output / "bc.pt"))
    # A fixed64 latent window consists of 63 previous cached z rows + z0.
    # Cached z rows themselves require64 feature rows: conservative extra warmup.
    ac_origin_mask = sequence_masks(index, feature_eligible=teacher_mask, seq_len=64)["endpoint_eligible"]
    ac_origins = np.flatnonzero(ac_origin_mask)
    if not len(ac_origins):
        raise ValueError("no contiguous fixed64 latent context for AC")
    ac_actors, ac_logs, bank_meta, arm_configs = {}, {}, {}, {}
    for arm, override in AC_ARMS.items():
        _seed(seed)
        arm_cfg = copy.deepcopy(cfg); arm_cfg["ac"].update(override)
        arm_configs[arm] = arm_cfg["ac"]
        arm_actor = copy.deepcopy(actor).to(device).train()
        critic = Critic(ensemble.get_z_dim(), ensemble.get_d_model(),
            hidden_dim=cfg["ac"].get("critic_hidden", 256), n_layers=cfg["ac"].get("ac_layers", 2),
            n_bins=len(ensemble.get_bins()), ema_decay=cfg["ac"].get("ema_decay", .98))
        trainer = ImagACTrainer(arm_actor, critic, ensemble, arm_cfg, device=device)
        bank_meta[arm] = bind_controller_anchor_bank(trainer, z=encoded["z"][origins], h=encoded["h"][origins],
            positions=positions[origins], states=states[origins], auxiliary=auxiliary[origins], origins=origins)
        trainer.checkpoint_metadata = {"experiment_fork": True, "source": "same_frozen_WM_and_BC",
            "arm": arm, "seed": seed, "selection": "endpoint300_no_financial_selection"}
        logs = []
        rng = np.random.default_rng(seed)
        for step in range(300):
            picked = rng.choice(ac_origins, size=min(int(cfg["ac"].get("batch_size", 512)), len(ac_origins)), replace=True)
            past_ix = picked[:, None] + np.arange(-63, 0)[None, :]
            log = trainer.train_step(tensor(encoded["z"][picked]), tensor(encoded["h"][picked]),
                past_zs=tensor(encoded["z"][past_ix]), past_as=torch.ones((len(picked), 63, 1), device=device),
                regime0=None, advantage0=tensor(auxiliary[picked]), controller_state0=tensor(states[picked]))
            if any(not np.isfinite(value) for value in log.values() if isinstance(value, (int, float))):
                raise RuntimeError("nonfinite AC log")
            logs.append(log)
            if (step + 1) % 50 == 0:
                print(f"[MaskedAC] {arm} step={step + 1}/300 actor_loss={log['actor_loss']:.8g}", flush=True)
        trainer.save(str(output / f"{arm}.pt"))
        arm_actor.eval()
        atomic_torch_save(copy.deepcopy(arm_actor).cpu(), str(output / f"{arm}_actor_full.pt"))
        ac_actors[arm], ac_logs[arm] = arm_actor, logs
    provenance = {"seed": seed, "fork": "same_frozen_WM_BC_three_independent_AC_endpoints",
        "features": list(features.columns), "timestamp_ns_sha256": _digest(index.asi8), "feature_mask_sha256": _digest(fm), "target_mask_sha256": _digest(tm),
        "train_mask_sha256": _digest(train), "wm_val_mask_sha256": None if val is None else _digest(val),
        "train_features_sha256": _digest(features.to_numpy()[train_rows]),
        "train_returns_sha256": _digest(returns.to_numpy()[train_rows]),
        "wm_train_sequences": len(dataset), "wm_val_sequences": None if val_dataset is None else len(val_dataset),
        "wm_train_sequence_length": 128, "wm_executed_steps": int(wm.global_step), "inference_context_length": 64,
        "wm_validation_selection": "predictive_loss_only" if val is not None else "endpoint700",
        "teacher_type": "lowfreq_wm_overlay_T_fitted_in_sample_causal_inputs_no_hindsight_relabel",
        "teacher_segments": contiguous_segments(teacher_mask), "teacher_rows": len(origins),
        "bc_chunks": len(bc_origins), "bc_chunk_size": int(cfg["bc"].get("chunk_size", 4)),
        "bc_chunk_inputs": "origin_z_h_full4_and42aux; first_switch_teacher_target",
        "ac_origin_count": len(ac_origins), "ac_origins_sha256": _digest(ac_origins),
        "ac_context": "63_previous_cached_eligible_T_z_plus_z0_no_gaps_extra63warmup; fixed64_at_each_imagined_step",
        "regime_dim": 0, "controller_state_dim": 4, "auxiliary_dim": 42,
        "arms": arm_configs, "anchor_banks": bank_meta}
    _json(output / "training_provenance.json", provenance)
    _json(output / "auxiliary_normalizer.json", normalizer)
    _json(output / "training_logs.json", {"wm": wm_log, "bc": bc_log, "ac": ac_logs})
    np.savez_compressed(output / "training_support.npz", teacher_positions=positions,
        controller_states=states, teacher_mask=teacher_mask, bc_origins=bc_origins,
        bc_target_origins=bc_target_origins, ac_origins=ac_origins)
    artifacts = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(output.iterdir()) if path.is_file()}
    _json(output / "artifacts.json", artifacts)
    return {"ensemble": ensemble, "wm_trainer": wm, "bc_actor": actor, "ac_actors": ac_actors,
            "encoded": encoded, "auxiliary": {"raw": raw, "standardized": auxiliary, "normalizer": normalizer},
            "teacher": {"positions": positions, "states": states, "available": teacher_mask},
            "training_provenance": provenance, "artifacts": artifacts}


__all__ = ["AC_ARMS", "train_wm_bc_ac", "normalize_auxiliary", "contiguous_segments",
           "build_segmented_teacher", "segmented_controller_states", "chunk_origins_and_targets",
           "bind_controller_anchor_bank"]
