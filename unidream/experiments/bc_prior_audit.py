from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from unidream.actor_critic.imagination_ac import _action_stats
from unidream.data.dataset import WFODataset
from unidream.data.oracle import _forward_window_stats

from .bc_setup import prepare_bc_setup
from .bc_stage import build_bc_trainer
from .fold_inputs import prepare_fold_inputs
from .runtime import resolve_costs
from .teacher_audit import load_audit_features
from .wfo_runtime import build_wfo_splits, select_wfo_splits
from .wm_stage import prepare_world_model_stage


def _format_action_stats(stats: dict[str, float]) -> str:
    return (
        f"long={stats['long']:.1%} short={stats['short']:.1%} flat={stats['flat']:.1%} "
        f"mean={stats['mean']:+.3f} switches={stats['switches']} avg_hold={stats['avg_hold']:.1f}b"
    )


def _subset_rows(
    *,
    config_name: str,
    fold_idx: int,
    split_name: str,
    regime_probs: np.ndarray | None,
    teacher_positions: np.ndarray,
    bc_positions: np.ndarray,
    benchmark_position: float,
) -> list[dict]:
    rows: list[dict] = []

    def _one_row(regime_label: str, teacher_subset: np.ndarray, bc_subset: np.ndarray) -> dict[str, float | int | str]:
        teacher_stats = _action_stats(teacher_subset, benchmark_position=benchmark_position)
        bc_stats = _action_stats(bc_subset, benchmark_position=benchmark_position)
        teacher_short = teacher_subset < benchmark_position - 1e-6
        bc_short = bc_subset < benchmark_position - 1e-6
        teacher_flat = np.isclose(teacher_subset, benchmark_position, atol=1e-6)
        bc_flat = np.isclose(bc_subset, benchmark_position, atol=1e-6)
        return {
            "config": config_name,
            "fold": int(fold_idx),
            "split": split_name,
            "regime": regime_label,
            "n_bars": int(len(teacher_subset)),
            "teacher_long_ratio": float(teacher_stats["long"]),
            "teacher_short_ratio": float(teacher_stats["short"]),
            "teacher_flat_ratio": float(teacher_stats["flat"]),
            "teacher_mean_overlay": float(teacher_stats["mean"]),
            "bc_long_ratio": float(bc_stats["long"]),
            "bc_short_ratio": float(bc_stats["short"]),
            "bc_flat_ratio": float(bc_stats["flat"]),
            "bc_mean_overlay": float(bc_stats["mean"]),
            "mean_abs_gap": float(np.mean(np.abs(teacher_subset - bc_subset))),
            "rmse_gap": float(np.sqrt(np.mean(np.square(teacher_subset - bc_subset)))),
            "short_mismatch": float(np.mean(teacher_short != bc_short)),
            "flat_mismatch": float(np.mean(teacher_flat != bc_flat)),
            "teacher_turnover": float(teacher_stats["turnover"]),
            "bc_turnover": float(bc_stats["turnover"]),
        }

    rows.append(_one_row("all", teacher_positions, bc_positions))
    if regime_probs is None:
        return rows

    regime_idx = np.argmax(regime_probs, axis=1)
    for rid in range(regime_probs.shape[1]):
        mask = regime_idx == rid
        if not np.any(mask):
            continue
        rows.append(_one_row(f"regime_{rid}", teacher_positions[mask], bc_positions[mask]))
    return rows


def run_bc_prior_audit(
    *,
    cfg: dict,
    config_name: str,
    symbol: str,
    interval: str,
    start: str,
    end: str,
    cache_dir: str,
    raw_cache_dir: str | None,
    checkpoint_dir: str,
    folds_arg: str | None,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_cfg = cfg.get("data", {})
    zscore_window = int(cfg.get("normalization", {}).get("zscore_window_days", 60))
    extra_series_mode = str(cfg.get("data", {}).get("extra_series_mode", "derived"))
    extra_series_include = cfg.get("data", {}).get("extra_series_include")
    include_funding = bool(cfg.get("data", {}).get("include_funding", True))
    include_oi = bool(cfg.get("data", {}).get("include_oi", True))
    include_mark = bool(cfg.get("data", {}).get("include_mark", True))
    cache_tag = f"{symbol}_{interval}_{start}_{end}_z{zscore_window}_v2"
    features_df, raw_returns = load_audit_features(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        zscore_window=zscore_window,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        raw_cache_dir=raw_cache_dir,
        extra_series_mode=extra_series_mode,
        extra_series_include=extra_series_include,
        include_funding=include_funding,
        include_oi=include_oi,
        include_mark=include_mark,
    )

    resolved_cfg, _ = resolve_costs(cfg)
    splits = build_wfo_splits(features_df, data_cfg)
    selected_splits, _ = select_wfo_splits(splits, folds_arg)
    seq_len = data_cfg.get("seq_len", 64)
    benchmark_position = float(resolved_cfg.get("reward", {}).get("benchmark_position", 1.0))

    detail_rows: list[dict] = []

    for split in selected_splits:
        fold_ckpt = os.path.join(checkpoint_dir, f"fold_{split.fold_idx}")
        wm_path = os.path.join(fold_ckpt, "world_model.pt")
        bc_path = os.path.join(fold_ckpt, "bc_actor.pt")
        if not (os.path.exists(wm_path) and os.path.exists(bc_path)):
            raise FileNotFoundError(f"missing checkpoints for fold {split.fold_idx}: {fold_ckpt}")

        wfo_dataset = WFODataset(features_df, raw_returns, split, seq_len=seq_len)
        fold_inputs = prepare_fold_inputs(
            wfo_dataset=wfo_dataset,
            cfg=resolved_cfg,
            costs_cfg=resolved_cfg.get("costs", {}),
            ac_cfg=resolved_cfg.get("ac", {}),
            bc_cfg=resolved_cfg.get("bc", {}),
            reward_cfg=resolved_cfg.get("reward", {}),
            action_stats_fn=_action_stats,
            format_action_stats_fn=_format_action_stats,
            benchmark_position=benchmark_position,
            forward_window_stats_fn=_forward_window_stats,
            log_ts=lambda: "audit",
        )

        ensemble, wm_trainer = prepare_world_model_stage(
            obs_dim=wfo_dataset.train_features.shape[1],
            cfg=resolved_cfg,
            device=device,
            has_wm=True,
            wm_path=wm_path,
            wfo_dataset=wfo_dataset,
            oracle_positions=fold_inputs["oracle_positions"],
            val_oracle_positions=fold_inputs["val_oracle_positions"],
            train_returns=fold_inputs["train_returns"],
            train_regime_probs=fold_inputs["train_regime_probs"],
            val_regime_probs=fold_inputs["val_regime_probs"],
            log_ts=lambda: "audit",
        )

        bc_setup = prepare_bc_setup(
            ensemble=ensemble,
            oracle_action_values=fold_inputs["oracle_bundle"]["oracle_action_values"],
            oracle_positions=fold_inputs["oracle_positions"],
            oracle_values=fold_inputs["oracle_bundle"]["oracle_values"],
            train_regime_probs=fold_inputs["train_regime_probs"],
            outcome_edge=fold_inputs["outcome_edge"],
            ac_cfg=resolved_cfg.get("ac", {}),
            bc_cfg=resolved_cfg.get("bc", {}),
            reward_cfg=resolved_cfg.get("reward", {}),
            oracle_teacher_mode=fold_inputs["oracle_bundle"]["oracle_teacher_mode"],
        )
        actor = bc_setup["actor"]
        bc_trainer = build_bc_trainer(
            actor=actor,
            ensemble=ensemble,
            bc_cfg=resolved_cfg.get("bc", {}),
            oracle_cfg=fold_inputs["oracle_cfg"],
            ac_cfg=resolved_cfg.get("ac", {}),
            reward_cfg=resolved_cfg.get("reward", {}),
            device=device,
        )
        bc_trainer.load(bc_path)

        split_specs = [
            ("train", wfo_dataset.train_features, fold_inputs["oracle_positions"], fold_inputs["train_regime_probs"]),
            ("val", wfo_dataset.val_features, fold_inputs["val_oracle_positions"], fold_inputs["val_regime_probs"]),
        ]
        for split_name, split_features, teacher_positions, regime_probs in split_specs:
            enc = wm_trainer.encode_sequence(split_features, seq_len=seq_len)
            bc_positions = actor.predict_positions(
                enc["z"],
                enc["h"],
                regime_np=regime_probs,
                device=device,
            )
            t_min = min(len(teacher_positions), len(bc_positions))
            detail_rows.extend(
                _subset_rows(
                    config_name=config_name,
                    fold_idx=split.fold_idx,
                    split_name=split_name,
                    regime_probs=regime_probs[:t_min] if regime_probs is not None else None,
                    teacher_positions=np.asarray(teacher_positions[:t_min], dtype=np.float32),
                    bc_positions=np.asarray(bc_positions[:t_min], dtype=np.float32),
                    benchmark_position=benchmark_position,
                )
            )

    detail_df = pd.DataFrame(detail_rows).sort_values(["fold", "split", "regime"]).reset_index(drop=True)
    summary_df = (
        detail_df.groupby(["config", "split", "regime"], dropna=False)
        .agg(
            n_bars=("n_bars", "sum"),
            teacher_short_ratio=("teacher_short_ratio", "mean"),
            teacher_flat_ratio=("teacher_flat_ratio", "mean"),
            bc_short_ratio=("bc_short_ratio", "mean"),
            bc_flat_ratio=("bc_flat_ratio", "mean"),
            mean_abs_gap=("mean_abs_gap", "mean"),
            rmse_gap=("rmse_gap", "mean"),
            short_mismatch=("short_mismatch", "mean"),
            flat_mismatch=("flat_mismatch", "mean"),
            teacher_turnover=("teacher_turnover", "mean"),
            bc_turnover=("bc_turnover", "mean"),
        )
        .reset_index()
        .sort_values(["split", "regime"])
    )

    out_dir = Path(checkpoint_dir) / "bc_prior_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / f"{config_name}_bc_prior_audit_summary.csv", index=False)
    detail_df.to_csv(out_dir / f"{config_name}_bc_prior_audit_detail.csv", index=False)
    return summary_df, detail_df
