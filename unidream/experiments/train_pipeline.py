from __future__ import annotations

from unidream.data.dataset import WFODataset

from .runtime import set_seed


def run_wfo_folds(
    *,
    features_df,
    raw_returns,
    splits,
    data_cfg: dict,
    cfg: dict,
    device: str,
    checkpoint_dir: str,
    run_fold_fn,
    seed: int | None = None,
) -> dict:
    fold_results = {}
    for split in splits:
        if seed is not None:
            set_seed(seed)
        wfo_ds = WFODataset(
            features_df,
            raw_returns,
            split,
            seq_len=data_cfg.get("seq_len", 64),
        )
        result = run_fold_fn(
            fold_idx=split.fold_idx,
            wfo_dataset=wfo_ds,
            cfg=cfg,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
        fold_results[split.fold_idx] = result
    return fold_results
