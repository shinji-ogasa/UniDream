from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from unidream.actor_critic.imagination_ac import _action_stats
from unidream.data.dataset import WFODataset
from unidream.data.oracle import _forward_window_stats
from unidream.device import add_device_argument, resolve_device
from unidream.eval.backtest import compute_pnl
from unidream.experiments.bc_setup import prepare_bc_setup
from unidream.experiments.bc_stage import build_bc_trainer
from unidream.experiments.fold_inputs import prepare_fold_inputs
from unidream.experiments.fold_runtime import prepare_fold_runtime
from unidream.experiments.oracle_post import apply_oracle_postprocess
from unidream.experiments.oracle_stage import compute_base_oracle
from unidream.experiments.oracle_teacher import compute_teacher_oracle
from unidream.experiments.predictive_state import build_wm_predictive_state_bundle
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.transition_advantage import (
    ROUTE_NAMES,
    compute_route_targets,
    compute_transition_advantage,
    config_from_dict as transition_advantage_config_from_dict,
    current_positions_from_path,
)
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.experiments.wm_stage import prepare_world_model_stage


EXPOSURE_ROUTE_NAMES = ("neutral", "de_risk", "overweight")


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fmt_action_stats(stats: dict) -> str:
    return (
        f"long={stats['long']:.0%} short={stats['short']:.0%} flat={stats['flat']:.0%} "
        f"mean={stats['mean']:+.3f} switches={stats['switches']} "
        f"avg_hold={stats['avg_hold']:.1f}b turnover={stats['turnover']:.2f}"
    )


def _benchmark_position_value(cfg: dict) -> float:
    return float(cfg.get("reward", {}).get("benchmark_position", 1.0))


def _benchmark_positions(length: int, cfg: dict) -> np.ndarray:
    return np.full(length, _benchmark_position_value(cfg), dtype=np.float64)


def _safe_float(x) -> float | None:
    if x is None:
        return None
    v = float(x)
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    route_names: tuple[str, ...],
) -> tuple[float, list[dict]]:
    rows = []
    f1s = []
    n_classes = len(route_names)
    for cls in range(n_classes):
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        f1s.append(f1)
        rows.append(
            {
                "route": route_names[cls],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int((y_true == cls).sum()),
                "pred_count": int((y_pred == cls).sum()),
            }
        )
    return float(np.mean(f1s)), rows


def _ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    out = 0.0
    total = max(len(conf), 1)
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        mask = (conf >= lo) & (conf < hi if i < bins - 1 else conf <= hi)
        if not mask.any():
            continue
        out += float(mask.sum()) / total * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return out


def _predict_route_probs(
    actor,
    *,
    z: np.ndarray,
    h: np.ndarray,
    positions: np.ndarray,
    regime: np.ndarray | None,
    advantage: np.ndarray | None,
    device: str,
    batch_size: int = 4096,
) -> np.ndarray:
    dev = torch.device(resolve_device(device))
    states = actor.controller_states_from_positions(positions[: len(z)])
    probs = []
    actor.eval()
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            end = min(start + batch_size, len(z))
            z_t = torch.tensor(z[start:end], dtype=torch.float32, device=dev)
            h_t = torch.tensor(h[start:end], dtype=torch.float32, device=dev)
            inv_t = torch.tensor(states[start:end], dtype=torch.float32, device=dev)
            reg_t = (
                torch.tensor(regime[start:end], dtype=torch.float32, device=dev)
                if regime is not None
                else None
            )
            adv_t = (
                torch.tensor(advantage[start:end], dtype=torch.float32, device=dev)
                if advantage is not None
                else None
            )
            logits = actor.route_logits(z_t, h_t, inventory=inv_t, regime=reg_t, advantage=adv_t)
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0).astype(np.float32)


def _evaluate_split(
    *,
    name: str,
    probs: np.ndarray,
    labels: np.ndarray,
    route_advantage: np.ndarray,
    returns: np.ndarray,
    positions: np.ndarray,
    cfg: dict,
    costs_cfg: dict,
) -> dict:
    n = min(len(probs), len(labels), len(route_advantage), len(returns), len(positions))
    probs = probs[:n]
    labels = labels[:n].astype(np.int64)
    route_advantage = route_advantage[:n].astype(np.float64)
    route_names = EXPOSURE_ROUTE_NAMES if probs.shape[-1] == 3 else ROUTE_NAMES
    if probs.shape[-1] == 3:
        mapped = np.zeros_like(labels)
        mapped[labels == 1] = 1
        mapped[labels == 3] = 2
        labels = mapped
    pred = np.argmax(probs, axis=1).astype(np.int64)
    conf = probs.max(axis=1)
    correct = pred == labels
    n_routes = len(route_names)
    ce = -np.log(np.clip(probs[np.arange(n), labels], 1e-8, 1.0)).mean()
    macro_f1, route_rows = _macro_f1(labels, pred, route_names)
    active_true = labels != 0
    active_pred = pred != 0
    neutral_true = labels == 0
    active_recall = float(((active_true) & (active_pred)).sum() / max(active_true.sum(), 1))
    false_active_rate = float(((neutral_true) & (active_pred)).sum() / max(neutral_true.sum(), 1))
    neutral_precision = float(((labels == 0) & (pred == 0)).sum() / max((pred == 0).sum(), 1))
    active_prob = 1.0 - probs[:, 0]
    top_active_adv = None
    if active_pred.any():
        pred_active_scores = active_prob[active_pred]
        threshold = np.quantile(pred_active_scores, 0.90)
        mask = active_pred & (active_prob >= threshold)
        if mask.any():
            top_active_adv = float(np.mean(route_advantage[mask]))
    pred_route_adv = []
    for idx, route in enumerate(route_names):
        mask = pred == idx
        pred_route_adv.append(
            {
                "route": route,
                "pred_rate": float(mask.mean()) if n else 0.0,
                "mean_label_advantage": float(np.mean(route_advantage[mask])) if mask.any() else 0.0,
            }
        )

    pnl = compute_pnl(
        returns[:n],
        positions[:n],
        spread_bps=costs_cfg.get("spread_bps", 3.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0003),
        slippage_bps=costs_cfg.get("slippage_bps", 1.0),
    )
    route_pnl = []
    for idx, route in enumerate(route_names):
        mask = pred == idx
        route_pnl.append(
            {
                "route": route,
                "mean_pnl": float(np.mean(pnl[mask])) if mask.any() else 0.0,
                "sum_pnl": float(np.sum(pnl[mask])) if mask.any() else 0.0,
            }
        )

    return {
        "split": name,
        "n": int(n),
        "ce": float(ce),
        "accuracy": float(correct.mean()) if n else 0.0,
        "macro_f1": macro_f1,
        "active_recall": active_recall,
        "false_active_rate": false_active_rate,
        "neutral_precision": neutral_precision,
        "ece": _ece(conf, correct),
        "top_decile_pred_active_advantage": _safe_float(top_active_adv),
        "routes": route_rows,
        "pred_route_advantage": pred_route_adv,
        "pred_route_pnl": route_pnl,
    }


def _test_route_targets(
    *,
    wfo_dataset,
    cfg: dict,
    costs_cfg: dict,
    bc_cfg: dict,
    ac_cfg: dict,
    reward_cfg: dict,
    train_returns: np.ndarray,
    oracle_action_values,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    oracle_cfg = cfg.get("oracle", {})
    base = compute_base_oracle(
        train_returns=train_returns,
        val_returns=wfo_dataset.test_returns,
        oracle_cfg=oracle_cfg,
        reward_cfg=reward_cfg,
        costs_cfg=costs_cfg,
        default_action_values=cfg.get("actions", {}).get("values"),
    )
    teacher = compute_teacher_oracle(
        teacher_mode=base["oracle_teacher_mode"],
        base_oracle_positions=base["oracle_positions"],
        base_val_oracle_positions=base["val_oracle_positions"],
        base_oracle_values=base["oracle_values"],
        train_returns=train_returns,
        val_returns=wfo_dataset.test_returns,
        train_features=wfo_dataset.train_features,
        val_features=wfo_dataset.test_features,
        feature_columns=getattr(wfo_dataset, "feature_columns", []),
        oracle_action_values=oracle_action_values,
        oracle_cfg=oracle_cfg,
        ac_cfg=ac_cfg,
        oracle_benchmark_position=base["oracle_benchmark_position"],
    )
    train_pos, test_pos, outcome_edge = apply_oracle_postprocess(
        oracle_positions=teacher["oracle_positions"],
        val_oracle_positions=teacher["val_oracle_positions"],
        oracle_action_values=oracle_action_values,
        oracle_cfg=oracle_cfg,
        ac_cfg=ac_cfg,
        bc_cfg=bc_cfg,
        oracle_reward_mode=base["oracle_reward_mode"],
        oracle_benchmark_position=base["oracle_benchmark_position"],
        oracle_teacher_mode=base["oracle_teacher_mode"],
        train_returns=train_returns,
        forward_window_stats_fn=_forward_window_stats,
    )
    del train_pos, outcome_edge
    ta_cfg = transition_advantage_config_from_dict(
        bc_cfg,
        costs_cfg=costs_cfg,
        benchmark_position=base["oracle_benchmark_position"],
        default_actions=oracle_action_values,
    )
    current = current_positions_from_path(test_pos, base["oracle_benchmark_position"])
    bundle = compute_transition_advantage(wfo_dataset.test_returns, current, ta_cfg)
    route_margin = bc_cfg.get(
        "transition_route_margins",
        bc_cfg.get("transition_route_margin", bc_cfg.get("transition_advantage_margin", ta_cfg.margin)),
    )
    route_bundle = compute_route_targets(
        bundle,
        tau=float(bc_cfg.get("route_adv_tau", 0.001)),
        label_smoothing=float(bc_cfg.get("route_label_smoothing", 0.05)),
        margin=route_margin,
        route_penalties=bc_cfg.get("transition_route_penalties"),
    )
    return (
        np.asarray(test_pos, dtype=np.float32),
        route_bundle["route_labels"],
        route_bundle["route_soft_labels"],
        route_bundle["route_advantage"],
    )


def _write_md(path: str, *, config: str, fold_results: dict) -> None:
    lines = [
        "# Route Probe Results",
        "",
        f"Config: `{config}`",
        "",
    ]
    for fold, split_rows in fold_results.items():
        lines.extend(["", f"## Fold {fold}", ""])
        lines.append("| split | CE | Acc | Macro-F1 | Active Recall | De-risk Recall | Recovery Recall | Overweight Recall | False Active | Neutral Precision | ECE | Top Active Adv |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for split_name, row in split_rows.items():
            by_route = {r["route"]: r for r in row["routes"]}
            def recall(name: str) -> str:
                route = by_route.get(name)
                return "NA" if route is None else f"{route['recall']:.3f}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        split_name,
                        f"{row['ce']:.4f}",
                        f"{row['accuracy']:.3f}",
                        f"{row['macro_f1']:.3f}",
                        f"{row['active_recall']:.3f}",
                        recall("de_risk"),
                        recall("recovery"),
                        recall("overweight"),
                        f"{row['false_active_rate']:.3f}",
                        f"{row['neutral_precision']:.3f}",
                        f"{row['ece']:.3f}",
                        "NA" if row["top_decile_pred_active_advantage"] is None else f"{row['top_decile_pred_active_advantage']:.6f}",
                    ]
                )
                + " |"
            )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.route_probe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--folds", default="4")
    parser.add_argument("--output-json", default="documents/route_probe.json")
    parser.add_argument("--output-md", default="documents/route_probe.md")
    add_device_argument(parser)
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    set_seed(args.seed)

    cfg = load_config(args.config)
    cfg, _active_profile = resolve_costs(cfg, None)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = cfg.get("logging", {}).get("checkpoint_dir", "checkpoints")
    data_cfg = cfg.get("data", {})
    symbol = data_cfg.get("symbol", "BTCUSDT")
    interval = data_cfg.get("interval", "15m")
    cache_dir = "checkpoints/data_cache"
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}_v2"
    features_df, raw_returns = load_training_features(
        symbol=symbol,
        interval=interval,
        start=args.start,
        end=args.end,
        zscore_window=zscore_window,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )
    splits, _selected = select_wfo_splits(build_wfo_splits(features_df, data_cfg), args.folds)
    costs_cfg = cfg.get("costs", {})
    reward_cfg = cfg.get("reward", {})
    all_results: dict[str, dict] = {}
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)

    for split in splits:
        wfo_dataset = WFODataset(
            features_df,
            raw_returns,
            split,
            seq_len=cfg.get("data", {}).get("seq_len", 64),
        )
        fold_runtime = prepare_fold_runtime(
            fold_idx=split.fold_idx,
            checkpoint_dir=args.checkpoint_dir,
            ac_cfg=cfg.get("ac", {}),
            resume=False,
            start_from="test",
            stop_after="test",
        )
        if not fold_runtime["has_wm_ckpt"] or not fold_runtime["has_bc_ckpt"]:
            raise FileNotFoundError(f"Missing WM/BC checkpoint for fold {split.fold_idx}: {fold_runtime}")

        fold_inputs = prepare_fold_inputs(
            wfo_dataset=wfo_dataset,
            cfg=cfg,
            costs_cfg=costs_cfg,
            ac_cfg=cfg.get("ac", {}),
            bc_cfg=cfg.get("bc", {}),
            reward_cfg=reward_cfg,
            action_stats_fn=_action_stats,
            format_action_stats_fn=_fmt_action_stats,
            benchmark_position=_benchmark_position_value(cfg),
            forward_window_stats_fn=_forward_window_stats,
            log_ts=_ts,
        )
        train_returns = fold_inputs["train_returns"]
        oracle_cfg = fold_inputs["oracle_cfg"]
        oracle_bundle = fold_inputs["oracle_bundle"]
        oracle_positions = fold_inputs["oracle_positions"]
        val_oracle_positions = fold_inputs["val_oracle_positions"]
        train_regime_probs = fold_inputs["train_regime_probs"]
        val_regime_probs = fold_inputs["val_regime_probs"]
        test_regime_probs = fold_inputs["test_regime_probs"]

        ensemble, wm_trainer = prepare_world_model_stage(
            fold_idx=split.fold_idx,
            obs_dim=wfo_dataset.obs_dim,
            cfg=cfg,
            device=args.device,
            has_wm=True,
            wm_path=fold_runtime["wm_path"],
            wfo_dataset=wfo_dataset,
            oracle_positions=oracle_positions,
            val_oracle_positions=val_oracle_positions,
            train_returns=train_returns,
            train_regime_probs=train_regime_probs,
            val_regime_probs=val_regime_probs,
            log_ts=_ts,
        )
        seq_len = cfg.get("data", {}).get("seq_len", 64)
        enc_train = wm_trainer.encode_sequence(wfo_dataset.train_features, actions=None, seq_len=seq_len)
        enc_val = wm_trainer.encode_sequence(wfo_dataset.val_features, actions=None, seq_len=seq_len)
        enc_test = wm_trainer.encode_sequence(wfo_dataset.test_features, actions=None, seq_len=seq_len)

        train_advantage_values = fold_inputs.get("train_advantage_values")
        val_advantage_values = fold_inputs.get("val_advantage_values")
        test_advantage_values = fold_inputs.get("test_advantage_values")
        predictive_bundle = build_wm_predictive_state_bundle(
            wm_trainer=wm_trainer,
            wfo_dataset=wfo_dataset,
            z_train=enc_train["z"],
            h_train=enc_train["h"],
            seq_len=seq_len,
            ac_cfg=cfg.get("ac", {}),
            log_ts=_ts,
        )
        if predictive_bundle is not None:
            cfg["ac"]["advantage_conditioned"] = True
            cfg["ac"]["advantage_dim"] = int(predictive_bundle["train"].shape[1])
            train_advantage_values = predictive_bundle["train"]
            val_advantage_values = predictive_bundle["val"]
            test_advantage_values = predictive_bundle["test"]

        bc_setup = prepare_bc_setup(
            ensemble=ensemble,
            oracle_action_values=oracle_bundle["oracle_action_values"],
            oracle_positions=oracle_positions,
            oracle_values=oracle_bundle["oracle_values"],
            train_regime_probs=train_regime_probs,
            outcome_edge=fold_inputs["outcome_edge"],
            ac_cfg=cfg.get("ac", {}),
            bc_cfg=cfg.get("bc", {}),
            reward_cfg=reward_cfg,
            oracle_teacher_mode=oracle_bundle["oracle_teacher_mode"],
        )
        actor = bc_setup["actor"]
        trainer = build_bc_trainer(
            actor=actor,
            ensemble=ensemble,
            bc_cfg=cfg.get("bc", {}),
            oracle_cfg=oracle_cfg,
            ac_cfg=cfg.get("ac", {}),
            reward_cfg=reward_cfg,
            device=args.device,
        )
        trainer.load(fold_runtime["bc_path"])

        test_positions, test_route_labels, _test_route_soft, test_route_adv = _test_route_targets(
            wfo_dataset=wfo_dataset,
            cfg=cfg,
            costs_cfg=costs_cfg,
            bc_cfg=cfg.get("bc", {}),
            ac_cfg=cfg.get("ac", {}),
            reward_cfg=reward_cfg,
            train_returns=train_returns,
            oracle_action_values=oracle_bundle["oracle_action_values"],
        )

        split_results = {}
        split_specs = {
            "train": (
                enc_train,
                oracle_positions,
                fold_inputs.get("train_route_labels"),
                fold_inputs.get("train_route_advantage"),
                train_regime_probs,
                train_advantage_values,
                wfo_dataset.train_returns,
            ),
            "val": (
                enc_val,
                val_oracle_positions,
                fold_inputs.get("val_route_labels"),
                fold_inputs.get("val_route_advantage"),
                val_regime_probs,
                val_advantage_values,
                wfo_dataset.val_returns,
            ),
            "test": (
                enc_test,
                test_positions,
                test_route_labels,
                test_route_adv,
                test_regime_probs,
                test_advantage_values,
                wfo_dataset.test_returns,
            ),
        }
        for split_name, (enc, pos, labels, adv, regime, adv_values, rets) in split_specs.items():
            if labels is None or adv is None:
                continue
            n = min(len(enc["z"]), len(labels), len(pos))
            probs = _predict_route_probs(
                actor,
                z=enc["z"][:n],
                h=enc["h"][:n],
                positions=np.asarray(pos[:n], dtype=np.float32),
                regime=regime[:n] if regime is not None else None,
                advantage=adv_values[:n] if adv_values is not None else None,
                device=args.device,
            )
            split_results[split_name] = _evaluate_split(
                name=split_name,
                probs=probs,
                labels=np.asarray(labels[:n]),
                route_advantage=np.asarray(adv[:n]),
                returns=np.asarray(rets[:n]),
                positions=np.asarray(pos[:n]),
                cfg=cfg,
                costs_cfg=costs_cfg,
            )
        all_results[str(split.fold_idx)] = split_results

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, config=args.config, fold_results=all_results)
    print(f"[RouteProbe] wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
