from __future__ import annotations

from pathlib import Path

import pandas as pd

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
    run_manifest: dict | None = None,
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
            run_manifest=run_manifest,
        )
        if cfg.get("logging", {}).get("save_test_path", False):
            test_index = features_df.index[
                (features_df.index >= split.test_start) & (features_df.index < split.test_end)
            ]
            frame = pd.DataFrame({
                "position": result["positions"],
                "market_log_return": result["test_returns"],
                "net_log_return": result["metrics"].pnl_series,
            }, index=test_index)
            dest = Path(checkpoint_dir) / f"fold_{split.fold_idx}" / "test_path.parquet"
            temp = dest.with_suffix(".tmp")
            frame.to_parquet(temp)
            temp.replace(dest)
        fold_results[split.fold_idx] = result
    return fold_results
