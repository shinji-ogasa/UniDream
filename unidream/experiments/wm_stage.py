from __future__ import annotations

from copy import deepcopy

from unidream.data.dataset import SequenceDataset
from unidream.world_model.train_wm import WorldModelTrainer, build_ensemble


def prepare_world_model_stage(
    *,
    fold_idx: int,
    obs_dim: int,
    cfg: dict,
    device: str,
    has_wm: bool,
    wm_path: str,
    wfo_dataset,
    oracle_positions,
    val_oracle_positions,
    train_returns,
    train_regime_probs=None,
    val_regime_probs=None,
    log_ts,
) -> tuple:
    cfg_local = deepcopy(cfg)
    if train_regime_probs is not None:
        cfg_local.setdefault("world_model", {})["regime_dim"] = int(train_regime_probs.shape[1])
    ensemble = build_ensemble(obs_dim, cfg_local)
    wm_trainer = WorldModelTrainer(ensemble, cfg_local, device=device)

    if has_wm:
        print(f"\n[{log_ts()}] [Step 2] World Model - loading checkpoint: {wm_path}")
        wm_trainer.load(wm_path)
        return ensemble, wm_trainer

    print(f"\n[{log_ts()}] [Step 2] World Model Training...")
    init_checkpoint = cfg.get("world_model", {}).get("init_checkpoint")
    if init_checkpoint:
        init_checkpoint = str(init_checkpoint).format(fold=fold_idx, fold_idx=fold_idx)
        print(f"[{log_ts()}] [WM] Initializing from checkpoint: {init_checkpoint}")
        wm_trainer.load(init_checkpoint)
    train_ds_with_actions = SequenceDataset(
        wfo_dataset.train_features,
        seq_len=cfg.get("data", {}).get("seq_len", 64),
        actions=oracle_positions[: len(wfo_dataset.train_features)],
        returns=train_returns,
        regime_probs=train_regime_probs[: len(wfo_dataset.train_features)] if train_regime_probs is not None else None,
    )
    val_ds = SequenceDataset(
        wfo_dataset.val_features,
        seq_len=cfg.get("data", {}).get("seq_len", 64),
        actions=val_oracle_positions[: len(wfo_dataset.val_features)],
        returns=wfo_dataset.val_returns,
        regime_probs=val_regime_probs[: len(wfo_dataset.val_features)] if val_regime_probs is not None else None,
    )
    wm_trainer.train_on_dataset(
        train_ds_with_actions,
        val_dataset=val_ds,
        checkpoint_path=wm_path,
    )
    return ensemble, wm_trainer
