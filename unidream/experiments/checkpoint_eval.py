"""Shared checkpoint loading helpers for leak-free fold evaluation."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from unidream.cli.train import _action_stats, _fmt_action_stats, _forward_window_stats, _ts
from unidream.data.dataset import WFODataset
from unidream.experiments.bc_setup import prepare_bc_setup
from unidream.experiments.bc_stage import build_bc_trainer
from unidream.experiments.fold_inputs import prepare_fold_inputs
from unidream.experiments.predictive_state import build_wm_predictive_state_bundle
from unidream.experiments.run_config import checkpoint_semantic_fingerprint
from unidream.world_model.train_wm import WorldModelTrainer, build_ensemble


@dataclass(frozen=True)
class CheckpointRunSpec:
    label: str
    checkpoint_dir: str
    use_ac: bool
    ac_filename: str = "ac.pt"


def parse_checkpoint_run_spec(spec: str) -> CheckpointRunSpec:
    parts = spec.split("=", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Run spec must be label=checkpoint_dir[:bc|:ac], got: {spec}")
    label, raw_path = parts
    mode = "ac"
    if raw_path.endswith(":bc"):
        raw_path = raw_path[:-3]
        mode = "bc"
    elif raw_path.endswith(":ac"):
        raw_path = raw_path[:-3]
        mode = "ac"
    ac_filename = "ac.pt"
    if "@" in raw_path:
        raw_path, ac_filename = raw_path.rsplit("@", 1)
        if not raw_path or not ac_filename:
            raise ValueError(f"Run spec has invalid checkpoint file override: {spec}")
    return CheckpointRunSpec(label=label, checkpoint_dir=raw_path, use_ac=(mode == "ac"), ac_filename=ac_filename)


def load_fold_model_context(
    *,
    fold_idx: int,
    dataset: WFODataset,
    cfg: dict[str, Any],
    checkpoint_dir: Path,
    device: str,
    benchmark_position: float,
) -> dict[str, Any]:
    """Load WM/BC artifacts and encoded fold state for evaluation CLIs."""
    ac_cfg = copy.deepcopy(cfg["ac"])
    bc_cfg = cfg["bc"]
    reward_cfg = cfg["reward"]
    fold_inputs = prepare_fold_inputs(
        fold_idx=fold_idx,
        wfo_dataset=dataset,
        cfg=cfg,
        costs_cfg=cfg["costs"],
        ac_cfg=ac_cfg,
        bc_cfg=bc_cfg,
        reward_cfg=reward_cfg,
        action_stats_fn=_action_stats,
        format_action_stats_fn=_fmt_action_stats,
        benchmark_position=benchmark_position,
        forward_window_stats_fn=_forward_window_stats,
        log_ts=_ts,
    )
    cfg_model = copy.deepcopy(cfg)
    if fold_inputs["train_regime_probs"] is not None:
        cfg_model.setdefault("world_model", {})["regime_dim"] = int(fold_inputs["train_regime_probs"].shape[1])
    ensemble = build_ensemble(dataset.obs_dim, cfg_model)
    wm_trainer = WorldModelTrainer(ensemble, cfg_model, device=device)
    fold_dir = checkpoint_dir / f"fold_{fold_idx}"
    wm_path = fold_dir / "world_model.pt"
    bc_path = fold_dir / "bc_actor.pt"
    if not wm_path.exists() or not bc_path.exists():
        raise FileNotFoundError(f"missing fold checkpoint under {fold_dir}")
    wm_trainer.load(str(wm_path))
    seq_len = int(cfg["data"]["seq_len"])
    encoded_train = wm_trainer.encode_sequence(dataset.train_features, actions=None, seq_len=seq_len)
    encoded_val = wm_trainer.encode_sequence(dataset.val_features, actions=None, seq_len=seq_len)
    encoded_test = wm_trainer.encode_sequence(dataset.test_features, actions=None, seq_len=seq_len)
    predictive = build_wm_predictive_state_bundle(
        wm_trainer=wm_trainer,
        wfo_dataset=dataset,
        z_train=encoded_train["z"],
        h_train=encoded_train["h"],
        seq_len=seq_len,
        ac_cfg=ac_cfg,
        log_ts=_ts,
    )
    train_advantage = fold_inputs["train_advantage_values"]
    val_advantage = fold_inputs["val_advantage_values"]
    test_advantage = fold_inputs["test_advantage_values"]
    if predictive is not None:
        ac_cfg["advantage_conditioned"] = True
        ac_cfg["advantage_dim"] = int(predictive["train"].shape[1])
        train_advantage = predictive["train"]
        val_advantage = predictive["val"]
        test_advantage = predictive["test"]
    bc_setup = prepare_bc_setup(
        ensemble=ensemble,
        oracle_action_values=fold_inputs["oracle_bundle"]["oracle_action_values"],
        oracle_positions=fold_inputs["oracle_positions"],
        oracle_values=fold_inputs["oracle_bundle"]["oracle_values"],
        train_regime_probs=fold_inputs["train_regime_probs"],
        outcome_edge=fold_inputs["outcome_edge"],
        ac_cfg=ac_cfg,
        bc_cfg=bc_cfg,
        reward_cfg=reward_cfg,
        oracle_teacher_mode=fold_inputs["oracle_bundle"]["oracle_teacher_mode"],
    )
    actor = bc_setup["actor"]
    bc_trainer = build_bc_trainer(
        actor=actor,
        ensemble=ensemble,
        bc_cfg=bc_cfg,
        oracle_cfg=fold_inputs["oracle_cfg"],
        ac_cfg=ac_cfg,
        reward_cfg=reward_cfg,
        device=device,
    )
    bc_trainer.load(str(bc_path))
    actor.eval()
    return {
        "fold_inputs": fold_inputs,
        "wm_trainer": wm_trainer,
        "actor": actor,
        "encoded_train": encoded_train,
        "encoded_val": encoded_val,
        "encoded_test": encoded_test,
        "predictive_bundle": predictive,
        "train_advantage": train_advantage,
        "val_advantage": val_advantage,
        "test_advantage": test_advantage,
        "wm_hash": checkpoint_semantic_fingerprint(wm_path),
        "bc_hash": checkpoint_semantic_fingerprint(bc_path),
    }


def load_actor_state_checkpoint(actor: Any, path: Path, device: str) -> None:
    """Load a BC/AC actor state while allowing known optional legacy heads."""
    ckpt = torch.load(path, map_location=torch.device(device), weights_only=False)
    state = ckpt.get("actor", ckpt)
    incompatible = actor.load_state_dict(state, strict=False)
    optional_missing = {
        "execution_head.weight",
        "execution_head.bias",
        "residual_head_a.weight",
        "residual_head_b.weight",
        "residual_head_a.bias",
        "residual_head_b.bias",
        "regime_mode_gate_head.weight",
        "route_head.weight",
        "route_head.bias",
        "route_delta_head.weight",
        "route_delta_head.bias",
        "route_active_head.weight",
        "route_active_head.bias",
        "route_active_class_head.weight",
        "route_active_class_head.bias",
        "route_advantage_gate.weight",
        "benchmark_overweight_sizing_adapter.weight",
        "benchmark_overweight_sizing_adapter.bias",
        "inventory_recovery_head.weight",
        "inventory_recovery_head.bias",
    }
    missing = [key for key in incompatible.missing_keys if key not in optional_missing]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"Actor checkpoint incompatibility while loading {path}: missing={missing}, unexpected={unexpected}")


def load_inference_run_context(
    *,
    run: CheckpointRunSpec,
    split: Any,
    dataset: WFODataset,
    cfg: dict[str, Any],
    device: str,
    benchmark_position: float,
) -> dict[str, Any]:
    context = load_fold_model_context(
        fold_idx=int(split.fold_idx),
        dataset=dataset,
        cfg=cfg,
        checkpoint_dir=Path(run.checkpoint_dir),
        device=device,
        benchmark_position=benchmark_position,
    )
    if run.use_ac:
        ac_path = Path(run.checkpoint_dir) / f"fold_{int(split.fold_idx)}" / run.ac_filename
        if not ac_path.exists():
            raise FileNotFoundError(f"{run.label}: requested AC actor but missing {ac_path}")
        load_actor_state_checkpoint(context["actor"], ac_path, device)
    adjust_grid = cfg.get("ac", {}).get("val_adjust_rate_scale_grid") or [1.0]
    context["actor"].infer_adjust_rate_scale = float(adjust_grid[0])
    context["actor"].infer_advantage_level = float(cfg.get("ac", {}).get("infer_advantage_level", 0.0))
    context["actor"].eval()
    return context
