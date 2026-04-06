from __future__ import annotations

from unidream.data.dataset import get_wfo_splits


def build_wfo_splits(features_df, data_cfg: dict):
    return get_wfo_splits(
        features_df,
        train_years=data_cfg.get("train_years", 2),
        val_months=data_cfg.get("val_months", 3),
        test_months=data_cfg.get("test_months", 3),
    )


def select_wfo_splits(splits, folds_arg: str | None):
    if not folds_arg:
        return splits, None

    selected_folds = sorted(
        {
            int(token.strip())
            for token in folds_arg.split(",")
            if token.strip()
        }
    )
    if not selected_folds:
        raise ValueError("--folds must contain at least one fold index")

    selected = [split for split in splits if split.fold_idx in selected_folds]
    if len(selected) == 0:
        raise ValueError(
            f"--folds selected {selected_folds}, but no matching folds were found in this dataset"
        )
    return selected, selected_folds
