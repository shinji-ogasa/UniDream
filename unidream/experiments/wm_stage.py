from __future__ import annotations

from unidream.data.dataset import SequenceDataset
from unidream.world_model.train_wm import WorldModelTrainer, build_ensemble


def prepare_world_model_stage(
    *,
    obs_dim: int,
    cfg: dict,
    device: str,
    has_wm: bool,
    wm_path: str,
    wfo_dataset,
    oracle_positions,
    val_oracle_positions,
    train_returns,
    log_ts,
) -> tuple:
    ensemble = build_ensemble(obs_dim, cfg)
    wm_trainer = WorldModelTrainer(ensemble, cfg, device=device)

    if has_wm:
        print(f"\n[{log_ts()}] [Step 2] World Model - loading checkpoint: {wm_path}")
        wm_trainer.load(wm_path)
        return ensemble, wm_trainer

    print(f"\n[{log_ts()}] [Step 2] World Model Training...")
    train_ds_with_actions = SequenceDataset(
        wfo_dataset.train_features,
        seq_len=cfg.get("data", {}).get("seq_len", 64),
        actions=oracle_positions[: len(wfo_dataset.train_features)],
        returns=train_returns,
    )
    val_ds = SequenceDataset(
        wfo_dataset.val_features,
        seq_len=cfg.get("data", {}).get("seq_len", 64),
        actions=val_oracle_positions[: len(wfo_dataset.val_features)],
        returns=wfo_dataset.val_returns,
    )
    wm_trainer.train_on_dataset(
        train_ds_with_actions,
        val_dataset=val_ds,
        checkpoint_path=wm_path,
    )
    return ensemble, wm_trainer
