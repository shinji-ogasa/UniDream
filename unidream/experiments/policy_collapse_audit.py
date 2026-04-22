from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from unidream.actor_critic.imagination_ac import _action_stats
from unidream.data.dataset import WFODataset
from unidream.device import resolve_device
from unidream.data.oracle import _forward_window_stats

from .bc_setup import prepare_bc_setup
from .bc_stage import build_bc_trainer
from .fold_inputs import prepare_fold_inputs
from .runtime import resolve_costs
from .teacher_audit import _write_audit_csvs, load_audit_features
from .wfo_runtime import build_wfo_splits, select_wfo_splits
from .wm_stage import prepare_world_model_stage


def _format_action_stats(stats: dict[str, float]) -> str:
    return (
        f"long={stats['long']:.1%} short={stats['short']:.1%} flat={stats['flat']:.1%} "
        f"mean={stats['mean']:+.3f} switches={stats['switches']} avg_hold={stats['avg_hold']:.1f}b"
    )


def _rollout_head_metrics(
    *,
    actor,
    z: np.ndarray,
    h: np.ndarray,
    regime_probs: np.ndarray | None,
    advantage_values: np.ndarray | None,
    device: str,
) -> dict[str, np.ndarray | float]:
    dev = torch.device(resolve_device(device))
    z_t = torch.as_tensor(z, dtype=torch.float32, device=dev)
    h_t = torch.as_tensor(h, dtype=torch.float32, device=dev)
    regime_t = None
    if regime_probs is not None:
        regime_t = torch.as_tensor(regime_probs, dtype=torch.float32, device=dev)
    advantage_t = None
    if advantage_values is not None:
        advantage_t = torch.as_tensor(advantage_values, dtype=torch.float32, device=dev)
        if advantage_t.ndim == 1:
            advantage_t = advantage_t.unsqueeze(-1)

    benchmark = float(getattr(actor, "benchmark_position", 1.0))
    target_values = actor._target_values_tensor(dev, torch.float32)
    short_mask = target_values < benchmark - 1e-6
    baseline_idx = int(actor.target_indices(torch.tensor([benchmark], dtype=torch.float32, device=dev)).item())

    controller_state = torch.zeros(1, actor.inventory_dim, dtype=torch.float32, device=dev)
    positions: list[float] = []
    trade_probs: list[float] = []
    target_entropy: list[float] = []
    short_mass: list[float] = []
    baseline_mass: list[float] = []
    target_mean_overlay: list[float] = []
    target_std_vals: list[float] = []

    actor.eval()
    with torch.no_grad():
        for i in range(len(z)):
            reg_i = regime_t[i : i + 1] if regime_t is not None else None
            adv_i = advantage_t[i : i + 1] if advantage_t is not None else None
            trade_logits, target_logits, target_mean, target_std, _band, _current = actor.controller_outputs_full(
                z_t[i : i + 1],
                h_t[i : i + 1],
                inventory=controller_state,
                regime=reg_i,
                advantage=adv_i,
            )
            probs = F.softmax(target_logits, dim=-1).squeeze(0)
            entropy = float((-(probs * probs.clamp_min(1e-9).log()).sum()).item())
            short_mass.append(float(probs[short_mask].sum().item()))
            baseline_mass.append(float(probs[baseline_idx].item()))
            execution_prob = torch.sigmoid(trade_logits)
            if actor._use_separate_execution_head():
                execution_prob = torch.sigmoid(
                    actor.execution_logits(
                        z_t[i : i + 1],
                        h_t[i : i + 1],
                        inventory=controller_state,
                        regime=reg_i,
                        advantage=adv_i,
                    )
                )
            trade_probs.append(float(execution_prob.item()))
            target_entropy.append(entropy)
            target_mean_overlay.append(float(target_mean.item()))
            target_std_vals.append(float(target_std.item()))

            next_position = actor.act_greedy(
                z_t[i : i + 1],
                h_t[i : i + 1],
                inventory=controller_state,
                regime=reg_i,
                advantage=adv_i,
            )
            positions.append(float(next_position.item()))
            controller_state = actor.update_controller_state(controller_state, next_position)

    return {
        "positions": np.asarray(positions, dtype=np.float32),
        "trade_probs": np.asarray(trade_probs, dtype=np.float32),
        "target_entropy": np.asarray(target_entropy, dtype=np.float32),
        "short_mass": np.asarray(short_mass, dtype=np.float32),
        "baseline_mass": np.asarray(baseline_mass, dtype=np.float32),
        "target_mean_overlay": np.asarray(target_mean_overlay, dtype=np.float32),
        "target_std": np.asarray(target_std_vals, dtype=np.float32),
    }


def _subset_rows(
    *,
    config_name: str,
    fold_idx: int,
    split_name: str,
    regime_probs: np.ndarray | None,
    teacher_positions: np.ndarray,
    bc_positions: np.ndarray,
    head_metrics: dict[str, np.ndarray | float],
    benchmark_position: float,
) -> list[dict]:
    rows: list[dict] = []

    def _row(label: str, mask: np.ndarray | None) -> dict:
        if mask is None:
            t = teacher_positions
            p = bc_positions
            trade_probs = head_metrics["trade_probs"]
            target_entropy = head_metrics["target_entropy"]
            short_mass = head_metrics["short_mass"]
            baseline_mass = head_metrics["baseline_mass"]
            target_mean_overlay = head_metrics["target_mean_overlay"]
            target_std = head_metrics["target_std"]
        else:
            t = teacher_positions[mask]
            p = bc_positions[mask]
            trade_probs = head_metrics["trade_probs"][mask]
            target_entropy = head_metrics["target_entropy"][mask]
            short_mass = head_metrics["short_mass"][mask]
            baseline_mass = head_metrics["baseline_mass"][mask]
            target_mean_overlay = head_metrics["target_mean_overlay"][mask]
            target_std = head_metrics["target_std"][mask]

        teacher_stats = _action_stats(t, benchmark_position=benchmark_position)
        bc_stats = _action_stats(p, benchmark_position=benchmark_position)
        return {
            "config": config_name,
            "fold": int(fold_idx),
            "split": split_name,
            "regime": label,
            "n_bars": int(len(p)),
            "teacher_short_ratio": float(teacher_stats["short"]),
            "teacher_flat_ratio": float(teacher_stats["flat"]),
            "bc_short_ratio": float(bc_stats["short"]),
            "bc_flat_ratio": float(bc_stats["flat"]),
            "trade_prob_mean": float(np.mean(trade_probs)),
            "trade_prob_high_ratio": float(np.mean(trade_probs >= 0.80)),
            "target_entropy_mean": float(np.mean(target_entropy)),
            "short_target_mass_mean": float(np.mean(short_mass)),
            "baseline_target_mass_mean": float(np.mean(baseline_mass)),
            "target_mean_overlay_mean": float(np.mean(target_mean_overlay)),
            "target_mean_overlay_std": float(np.std(target_mean_overlay)),
            "target_std_mean": float(np.mean(target_std)),
            "teacher_to_bc_mean_abs_gap": float(np.mean(np.abs(t - p))),
        }

    rows.append(_row("all", None))
    if regime_probs is None:
        return rows
    regime_idx = np.argmax(regime_probs, axis=1)
    for rid in range(regime_probs.shape[1]):
        mask = regime_idx == rid
        if np.any(mask):
            rows.append(_row(f"regime_{rid}", mask))
    return rows


def run_policy_collapse_audit(
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
    split_filter: tuple[str, ...] = ("val",),
    max_bars: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_cfg = cfg.get("data", {})
    zscore_window = int(cfg.get("normalization", {}).get("zscore_window_days", 60))
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
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
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
            (
                "train",
                wfo_dataset.train_features,
                fold_inputs["oracle_positions"],
                fold_inputs["train_regime_probs"],
                fold_inputs.get("train_advantage_values"),
            ),
            (
                "val",
                wfo_dataset.val_features,
                fold_inputs["val_oracle_positions"],
                fold_inputs["val_regime_probs"],
                fold_inputs.get("val_advantage_values"),
            ),
        ]
        for split_name, split_features, teacher_positions, regime_probs, advantage_values in split_specs:
            if split_name not in split_filter:
                continue
            if max_bars is not None and len(split_features) > max_bars:
                split_features = split_features[-max_bars:]
                teacher_positions = teacher_positions[-max_bars:]
                if regime_probs is not None:
                    regime_probs = regime_probs[-max_bars:]
                if advantage_values is not None:
                    advantage_values = advantage_values[-max_bars:]

            enc = wm_trainer.encode_sequence(split_features, seq_len=seq_len)
            head_metrics = _rollout_head_metrics(
                actor=actor,
                z=enc["z"],
                h=enc["h"],
                regime_probs=regime_probs,
                advantage_values=advantage_values,
                device=device,
            )
            bc_positions = head_metrics["positions"]
            t_min = min(len(teacher_positions), len(bc_positions))
            trimmed_metrics = {
                k: (v[:t_min] if isinstance(v, np.ndarray) else v)
                for k, v in head_metrics.items()
            }
            detail_rows.extend(
                _subset_rows(
                    config_name=config_name,
                    fold_idx=split.fold_idx,
                    split_name=split_name,
                    regime_probs=regime_probs[:t_min] if regime_probs is not None else None,
                    teacher_positions=np.asarray(teacher_positions[:t_min], dtype=np.float32),
                    bc_positions=np.asarray(bc_positions[:t_min], dtype=np.float32),
                    head_metrics=trimmed_metrics,
                    benchmark_position=benchmark_position,
                )
            )

    detail_df = pd.DataFrame(detail_rows).sort_values(["config", "fold", "split", "regime"]).reset_index(drop=True)
    summary_df = (
        detail_df.groupby(["config", "split", "regime"], dropna=False)
        .agg(
            n_bars=("n_bars", "sum"),
            teacher_short_ratio=("teacher_short_ratio", "mean"),
            teacher_flat_ratio=("teacher_flat_ratio", "mean"),
            bc_short_ratio=("bc_short_ratio", "mean"),
            bc_flat_ratio=("bc_flat_ratio", "mean"),
            trade_prob_mean=("trade_prob_mean", "mean"),
            trade_prob_high_ratio=("trade_prob_high_ratio", "mean"),
            target_entropy_mean=("target_entropy_mean", "mean"),
            short_target_mass_mean=("short_target_mass_mean", "mean"),
            baseline_target_mass_mean=("baseline_target_mass_mean", "mean"),
            target_mean_overlay_mean=("target_mean_overlay_mean", "mean"),
            target_mean_overlay_std=("target_mean_overlay_std", "mean"),
            target_std_mean=("target_std_mean", "mean"),
            teacher_to_bc_mean_abs_gap=("teacher_to_bc_mean_abs_gap", "mean"),
        )
        .reset_index()
        .sort_values(["config", "split", "regime"])
    )

    out_dir = Path(checkpoint_dir) / "policy_collapse_audit"
    _write_audit_csvs(out_dir, config_name, "policy_collapse_audit", summary_df, detail_df)
    return summary_df, detail_df
