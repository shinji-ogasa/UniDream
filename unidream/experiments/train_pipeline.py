from __future__ import annotations

from unidream.data.dataset import WFODataset


def run_wfo_folds(
    *,
    features_df,
    raw_returns,
    splits,
    data_cfg: dict,
    cfg: dict,
    device: str,
    checkpoint_dir: str,
    resume: bool,
    start_from: str,
    stop_after: str,
    run_fold_fn,
) -> dict:
    fold_results = {}
    for split in splits:
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
            resume=resume,
            start_from=start_from,
            stop_after=stop_after,
        )
        fold_results[split.fold_idx] = result
    return fold_results
