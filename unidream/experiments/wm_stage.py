from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from unidream.data.dataset import SequenceDataset
from unidream.world_model.train_wm import (
    WorldModelTrainer,
    build_ensemble,
    world_model_uses_dataset_actions,
)
from .chronological_oof import (
    ConditionalPathBlocked,
    conditional_path_or_artifact_enabled,
    conditional_runtime_config,
)
from .conditional_teacher import require_authenticated_conditional_teacher_context


def prepare_world_model_stage(
    *,
    obs_dim: int,
    cfg: dict,
    device: str,
    wm_path: str,
    wfo_dataset,
    oracle_positions,
    val_oracle_positions,
    train_returns,
    train_regime_probs=None,
    val_regime_probs=None,
    checkpoint_metadata: dict | None = None,
    conditional_teacher_context=None,
    log_ts,
) -> tuple:
    effective_cfg = conditional_runtime_config(cfg, cfg.get("world_model", {}))
    if conditional_path_or_artifact_enabled(effective_cfg):
        # A strict conditional run may enter the stage only with an
        # authenticated OOF-derived teacher.  The context is checked against
        # the complete config (not the stage-local mapping) so expected hashes
        # and the canonical action contract cannot disappear at this boundary.
        context = require_authenticated_conditional_teacher_context(
            conditional_teacher_context,
            config=cfg,
            caller="prepare_world_model_stage",
        )
        oracle_positions = context.train_positions
        val_oracle_positions = context.val_positions
    cfg_local = deepcopy(cfg)
    if train_regime_probs is not None:
        cfg_local.setdefault("world_model", {})["regime_dim"] = int(train_regime_probs.shape[1])
    ensemble = build_ensemble(obs_dim, cfg_local)
    wm_trainer = WorldModelTrainer(ensemble, cfg_local, device=device)
    wm_trainer.checkpoint_metadata = dict(checkpoint_metadata or {})

    print(f"\n[{log_ts()}] [Step 2] World Model Training...")
    use_dataset_actions = world_model_uses_dataset_actions(cfg_local)
    train_actions = (
        oracle_positions[: len(wfo_dataset.train_features)]
        if use_dataset_actions and oracle_positions is not None
        else None
    )
    val_actions = (
        val_oracle_positions[: len(wfo_dataset.val_features)]
        if use_dataset_actions and val_oracle_positions is not None
        else None
    )
    train_ds_with_actions = SequenceDataset(
        wfo_dataset.train_features,
        seq_len=cfg.get("data", {}).get("seq_len", 64),
        actions=train_actions,
        returns=train_returns,
        regime_probs=train_regime_probs[: len(wfo_dataset.train_features)] if train_regime_probs is not None else None,
    )
    val_ds = SequenceDataset(
        wfo_dataset.val_features,
        seq_len=cfg.get("data", {}).get("seq_len", 64),
        actions=val_actions,
        returns=wfo_dataset.val_returns,
        regime_probs=val_regime_probs[: len(wfo_dataset.val_features)] if val_regime_probs is not None else None,
    )
    train_kwargs = {
        "val_dataset": val_ds,
        "checkpoint_path": wm_path,
    }
    # Keep the historical default when callers do not opt in, while allowing
    # an experiment to pin validation patience explicitly.  This matters for
    # a declared training budget: otherwise the trainer's implicit patience
    # can stop a run far below max_steps.
    world_model_cfg = effective_cfg.get("world_model")
    if not isinstance(world_model_cfg, Mapping):
        # ``conditional_runtime_config(full, section)`` returns the section
        # itself; callers that pass only a full config return the nested
        # world_model mapping.  Support both shapes without weakening the
        # strict conditional gate.
        world_model_cfg = effective_cfg
    if "patience" in world_model_cfg:
        patience = int(world_model_cfg["patience"])
        if patience <= 0:
            raise ValueError("world_model.patience must be positive when configured")
        train_kwargs["patience"] = patience
    wm_trainer.train_on_dataset(train_ds_with_actions, **train_kwargs)
    return ensemble, wm_trainer
