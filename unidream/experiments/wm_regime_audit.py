from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from unidream.actor_critic.imagination_ac import _action_stats
from unidream.data.dataset import WFODataset
from unidream.data.oracle import _forward_window_stats

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


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(labels), n_classes), dtype=np.float32)
    out[np.arange(len(labels)), labels.astype(int)] = 1.0
    return out


def _fit_ridge_multiclass(x: np.ndarray, labels: np.ndarray, ridge: float) -> np.ndarray:
    n_classes = int(labels.max()) + 1
    y = _one_hot(labels, n_classes)
    x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1)
    gram = x_aug.T @ x_aug
    reg = ridge * np.eye(gram.shape[0], dtype=np.float32)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(gram + reg, x_aug.T @ y)
    return weights


def _predict_ridge_multiclass(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1)
    logits = x_aug @ weights
    return np.argmax(logits, axis=1)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s: list[float] = []
    for cls in range(n_classes):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        f1s.append(f1)
    return float(np.mean(f1s))


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls: list[float] = []
    for cls in range(n_classes):
        mask = y_true == cls
        if not np.any(mask):
            continue
        recalls.append(float(np.mean(y_pred[mask] == cls)))
    return float(np.mean(recalls)) if recalls else 0.0


def _latent_matrix(enc: dict[str, np.ndarray]) -> np.ndarray:
    z = np.asarray(enc["z"], dtype=np.float32)
    h = np.asarray(enc["h"], dtype=np.float32)
    return np.concatenate([z, h], axis=1)


def _probe_rows(
    *,
    config_name: str,
    fold_idx: int,
    split_name: str,
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int | str]:
    n_classes = int(max(np.max(y_true), np.max(y_pred))) + 1 if len(y_true) else 0
    return {
        "config": config_name,
        "fold": int(fold_idx),
        "split": split_name,
        "task": task,
        "n_samples": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "balanced_accuracy": _balanced_accuracy(y_true, y_pred, n_classes) if len(y_true) else 0.0,
        "macro_f1": _macro_f1(y_true, y_pred, n_classes) if len(y_true) else 0.0,
    }


def run_wm_regime_audit(
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
    ridge: float = 1e-2,
    split_filter: tuple[str, ...] = ("train", "val"),
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
        if not os.path.exists(wm_path):
            raise FileNotFoundError(f"missing WM checkpoint for fold {split.fold_idx}: {fold_ckpt}")

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

        _, wm_trainer = prepare_world_model_stage(
            obs_dim=wfo_dataset.train_features.shape[1],
            cfg=resolved_cfg,
            device=device,
            has_wm=True,
            wm_path=wm_path,
            wfo_dataset=wfo_dataset,
            oracle_positions=fold_inputs["oracle_positions"],
            val_oracle_positions=fold_inputs["val_oracle_positions"],
            train_returns=fold_inputs["train_returns"],
            log_ts=lambda: "audit",
        )

        split_specs = [
            ("train", wfo_dataset.train_features, fold_inputs["train_regime_probs"]),
            ("val", wfo_dataset.val_features, fold_inputs["val_regime_probs"]),
        ]

        train_pack: dict[str, np.ndarray] | None = None
        for split_name, split_features, regime_probs in split_specs:
            if split_name not in split_filter:
                continue
            if max_bars is not None and len(split_features) > max_bars:
                split_features = split_features[-max_bars:]
                regime_probs = regime_probs[-max_bars:]
            enc = wm_trainer.encode_sequence(split_features, seq_len=seq_len)
            x = _latent_matrix(enc)
            y = np.argmax(np.asarray(regime_probs), axis=1).astype(np.int64)
            n = min(len(x), len(y))
            x = x[:n]
            y = y[:n]
            if train_pack is None:
                train_pack = {"x": x, "y": y}
            current_weights = _fit_ridge_multiclass(train_pack["x"], train_pack["y"], ridge)
            y_pred = _predict_ridge_multiclass(x, current_weights)
            detail_rows.append(
                _probe_rows(
                    config_name=config_name,
                    fold_idx=split.fold_idx,
                    split_name=split_name,
                    task="current_regime",
                    y_true=y,
                    y_pred=y_pred,
                )
            )

            if len(x) > 1:
                x_next = x[:-1]
                y_next = y[1:]
                train_x_next = train_pack["x"][:-1] if len(train_pack["x"]) > 1 else train_pack["x"]
                train_y_next = train_pack["y"][1:] if len(train_pack["y"]) > 1 else train_pack["y"]
                next_weights = _fit_ridge_multiclass(train_x_next, train_y_next, ridge)
                y_next_pred = _predict_ridge_multiclass(x_next, next_weights)
                detail_rows.append(
                    _probe_rows(
                        config_name=config_name,
                        fold_idx=split.fold_idx,
                        split_name=split_name,
                        task="next_regime",
                        y_true=y_next,
                        y_pred=y_next_pred,
                    )
                )

    detail_df = pd.DataFrame(detail_rows).sort_values(["fold", "split", "task"]).reset_index(drop=True)
    summary_df = (
        detail_df.groupby(["config", "split", "task"], dropna=False)
        .agg(
            n_samples=("n_samples", "sum"),
            accuracy=("accuracy", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
        )
        .reset_index()
        .sort_values(["split", "task"])
    )

    out_dir = Path(checkpoint_dir) / "wm_regime_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / f"{config_name}_wm_regime_audit_summary.csv", index=False)
    detail_df.to_csv(out_dir / f"{config_name}_wm_regime_audit_detail.csv", index=False)
    return summary_df, detail_df
