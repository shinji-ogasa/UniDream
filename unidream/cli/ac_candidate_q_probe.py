from __future__ import annotations

import argparse
import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch

from unidream.actor_critic.imagination_ac import _action_stats
from unidream.actor_critic.state_action_critic import (
    CandidateQTrainConfig,
    build_action_feature_matrix,
    build_state_matrix,
    evaluate_candidate_q,
    nearest_candidate_indices,
    predict_candidate_q,
    train_candidate_q_ensemble,
)
from unidream.data.dataset import WFODataset
from unidream.data.oracle import _forward_window_stats
from unidream.device import add_device_argument, resolve_device
from unidream.eval.backtest import Backtest
from unidream.experiments.bc_setup import prepare_bc_setup
from unidream.experiments.bc_stage import build_bc_trainer
from unidream.experiments.fold_inputs import prepare_fold_inputs
from unidream.experiments.fold_runtime import prepare_fold_runtime
from unidream.experiments.predictive_state import build_wm_predictive_state_bundle
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.transition_advantage import (
    compute_transition_advantage,
    config_from_dict as transition_advantage_config_from_dict,
    current_positions_from_path,
)
from unidream.experiments.val_selector_stage import run_val_selector_stage
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.experiments.wm_stage import prepare_world_model_stage
from unidream.cli.train import (
    _candidate_to_text,
    _select_policy_candidate,
    _selector_candidate,
    _selector_cfg,
)


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


def _parse_candidates(raw: str | None, cfg: dict) -> np.ndarray:
    if raw:
        vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        vals = list(cfg.get("bc", {}).get("transition_candidate_actions", []))
        vals += list(cfg.get("oracle", {}).get("action_values", []))
        vals += [0.0, 1.0, 1.05, 1.10]
    vals = sorted({round(float(v), 6) for v in vals})
    return np.asarray(vals, dtype=np.float32)


@dataclass(frozen=True)
class CandidateSpec:
    labels: list[str]
    positions: np.ndarray
    anchor_idx: np.ndarray


def _transition_unit_cost(costs_cfg: dict) -> float:
    return float(
        (costs_cfg.get("spread_bps", 3.0) / 10000.0) / 2.0
        + costs_cfg.get("fee_rate", 0.0003)
        + costs_cfg.get("slippage_bps", 1.0) / 10000.0
    )


def _rolling_sum(x: np.ndarray, horizon: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    if horizon <= 0 or len(x) < horizon:
        return out
    csum = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64))])
    out[: len(x) - horizon + 1] = csum[horizon:] - csum[:-horizon]
    return out


def _rolling_vol(x: np.ndarray, horizon: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    if horizon <= 1 or len(x) < horizon:
        return out
    sum_x = _rolling_sum(x, horizon)
    sum_x2 = _rolling_sum(np.asarray(x, dtype=np.float64) ** 2, horizon)
    mean = sum_x / horizon
    var = np.maximum(sum_x2 / horizon - mean * mean, 0.0)
    out[: len(x) - horizon + 1] = np.sqrt(var[: len(x) - horizon + 1]) * np.sqrt(horizon)
    return out


def _rolling_drawdown_dynamic(returns: np.ndarray, overlay: np.ndarray, horizon: int) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64)
    overlay = np.asarray(overlay, dtype=np.float64)
    out = np.full(overlay.shape, np.nan, dtype=np.float64)
    if horizon <= 0 or len(returns) < horizon:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(returns, horizon)
    n = windows.shape[0]
    path = overlay[:n, :, None] * windows[:, None, :]
    cums = np.cumsum(path, axis=2)
    peak = np.maximum.accumulate(np.maximum(cums, 0.0), axis=2)
    out[:n] = -np.min(cums - peak, axis=2)
    return out


def _build_candidate_spec(
    *,
    mode: str,
    fixed_actions: np.ndarray,
    positions: np.ndarray,
    current: np.ndarray,
    cfg: dict,
    benchmark_position: float,
) -> CandidateSpec:
    mode = str(mode).lower()
    positions = np.asarray(positions, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    n = min(len(positions), len(current))
    positions = positions[:n]
    current = current[:n]
    if mode == "fixed":
        actions = np.asarray(fixed_actions, dtype=np.float32)
        return CandidateSpec(
            labels=[f"{x:.2f}" for x in actions],
            positions=np.broadcast_to(actions[None, :], (n, len(actions))).astype(np.float32),
            anchor_idx=nearest_candidate_indices(positions, actions),
        )
    ac_cfg = cfg.get("ac", {})
    abs_min = float(ac_cfg.get("abs_min_position", 0.0))
    abs_max = float(ac_cfg.get("abs_max_position", 1.25))
    step = float(ac_cfg.get("candidate_q_residual_step", 0.05))
    mild_ow = [float(x) for x in ac_cfg.get("candidate_q_overweight_actions", [1.05, 1.10, 1.25])]
    cols = [
        positions,
        positions - step,
        positions + step,
        current,
        np.full(n, benchmark_position, dtype=np.float32),
    ]
    labels = ["bc", f"bc_minus_{step:.2f}", f"bc_plus_{step:.2f}", "hold_current", "benchmark"]
    for ow in mild_ow:
        cols.append(np.full(n, ow, dtype=np.float32))
        labels.append(f"ow_{ow:.2f}")
    matrix = np.stack(cols, axis=1).astype(np.float32)
    matrix = np.clip(matrix, abs_min, abs_max)
    return CandidateSpec(labels=labels, positions=matrix, anchor_idx=np.zeros(n, dtype=np.int64))


def _compute_dynamic_candidate_values(
    *,
    returns: np.ndarray,
    current: np.ndarray,
    candidate_positions: np.ndarray,
    cfg: dict,
    costs_cfg: dict,
    benchmark_position: float,
) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    candidates = np.asarray(candidate_positions, dtype=np.float64)
    n = min(len(returns), len(current), len(candidates))
    returns = returns[:n]
    current = current[:n]
    candidates = candidates[:n]
    bc_cfg = cfg.get("bc", {})
    horizons = tuple(int(h) for h in bc_cfg.get("transition_advantage_horizons", [4, 8, 16, 32]))
    raw_weights = bc_cfg.get("transition_advantage_horizon_weights")
    weights = np.asarray(raw_weights if raw_weights is not None else np.ones(len(horizons)), dtype=np.float64)
    if weights.size != len(horizons):
        weights = np.ones(len(horizons), dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    overlay = candidates - float(benchmark_position)
    trade_delta = np.abs(candidates - current[:, None])
    unit_cost = _transition_unit_cost(costs_cfg)
    values_h = np.full((n, candidates.shape[1], len(horizons)), np.nan, dtype=np.float64)
    for hi, horizon in enumerate(horizons):
        sum_ret = _rolling_sum(returns, horizon)
        vol = _rolling_vol(returns, horizon)
        dd = _rolling_drawdown_dynamic(returns, overlay, horizon)
        valid = np.isfinite(sum_ret)
        value = overlay * sum_ret[:, None]
        value = value - unit_cost * trade_delta
        value = value - float(bc_cfg.get("transition_turnover_penalty_coef", 0.0)) * trade_delta
        value = value - float(bc_cfg.get("transition_volatility_penalty_coef", 0.10)) * np.nan_to_num(vol, nan=0.0)[:, None] * np.abs(overlay)
        value = value - float(bc_cfg.get("transition_drawdown_penalty_coef", 0.25)) * np.nan_to_num(dd, nan=0.0)
        value = value - float(bc_cfg.get("transition_leverage_penalty_coef", 0.0)) * np.maximum(candidates - benchmark_position, 0.0) * horizon
        value = value - float(bc_cfg.get("transition_short_penalty_coef", 0.0)) * np.maximum(-candidates, 0.0) * horizon
        value[~valid, :] = np.nan
        values_h[:, :, hi] = value
    finite = np.isfinite(values_h)
    weighted = np.where(finite, values_h, 0.0) * weights.reshape(1, 1, -1)
    denom = np.sum(finite * weights.reshape(1, 1, -1), axis=2)
    return np.divide(weighted.sum(axis=2), denom, out=np.full(candidates.shape, np.nan), where=denom > 0).astype(np.float32)


def _finite_float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return _finite_float(obj)
    if isinstance(obj, float):
        return _finite_float(obj)
    return obj


def _metric_str(v, digits: int = 4) -> str:
    if v is None:
        return "NA"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def _prepare_transition_values(
    *,
    returns: np.ndarray,
    positions: np.ndarray,
    cfg: dict,
    costs_cfg: dict,
    candidate_actions: np.ndarray,
    benchmark_position: float,
    candidate_mode: str,
) -> tuple[np.ndarray, np.ndarray, CandidateSpec]:
    current = current_positions_from_path(positions, benchmark_position)
    spec = _build_candidate_spec(
        mode=candidate_mode,
        fixed_actions=candidate_actions,
        positions=positions,
        current=current,
        cfg=cfg,
        benchmark_position=benchmark_position,
    )
    if str(candidate_mode).lower() != "fixed":
        values = _compute_dynamic_candidate_values(
            returns=returns,
            current=current,
            candidate_positions=spec.positions,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
        )
        n = min(len(current), len(values), len(spec.positions))
        return current[:n].astype(np.float32), values[:n], CandidateSpec(
            labels=spec.labels,
            positions=spec.positions[:n],
            anchor_idx=spec.anchor_idx[:n],
        )
    bc_cfg = deepcopy(cfg.get("bc", {}))
    bc_cfg["transition_candidate_actions"] = [float(x) for x in candidate_actions]
    ta_cfg = transition_advantage_config_from_dict(
        bc_cfg,
        costs_cfg=costs_cfg,
        benchmark_position=benchmark_position,
        default_actions=candidate_actions,
    )
    bundle = compute_transition_advantage(returns, current, ta_cfg)
    return current.astype(np.float32), np.asarray(bundle["values"], dtype=np.float32), spec


def _split_payload(
    *,
    actor,
    enc: dict,
    returns: np.ndarray,
    regime: np.ndarray | None,
    advantage: np.ndarray | None,
    candidate_actions: np.ndarray,
    candidate_mode: str,
    cfg: dict,
    costs_cfg: dict,
    benchmark_position: float,
    device: str,
) -> dict:
    n = len(enc["z"])
    if regime is not None:
        n = min(n, len(regime))
    if advantage is not None:
        n = min(n, len(advantage))
    n = min(n, len(returns))
    z = enc["z"][:n]
    h = enc["h"][:n]
    reg = regime[:n] if regime is not None else None
    adv = advantage[:n] if advantage is not None else None
    positions = actor.predict_positions(z, h, regime_np=reg, advantage_np=adv, device=device)
    current, values, candidate_spec = _prepare_transition_values(
        returns=np.asarray(returns[:n], dtype=np.float32),
        positions=positions[:n],
        cfg=cfg,
        costs_cfg=costs_cfg,
        candidate_actions=candidate_actions,
        benchmark_position=benchmark_position,
        candidate_mode=candidate_mode,
    )
    states = actor.controller_states_from_positions(positions[:n])
    state_matrix = build_state_matrix(z=z, h=h, inventory=states, regime=reg, advantage=adv)
    n = min(len(state_matrix), len(current), len(values), len(positions))
    action_features = build_action_feature_matrix(
        candidate_actions=candidate_spec.positions[:n],
        current_positions=current[:n],
        benchmark_position=benchmark_position,
    )
    anchor_idx = candidate_spec.anchor_idx[:n]
    baseline_stats = _action_stats(positions[:n], benchmark_position=benchmark_position)
    baseline_metrics = Backtest(
        np.asarray(returns[:n], dtype=np.float64),
        positions[:n],
        spread_bps=costs_cfg.get("spread_bps", 3.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0003),
        slippage_bps=costs_cfg.get("slippage_bps", 1.0),
        interval=cfg.get("data", {}).get("interval", "15m"),
        benchmark_positions=_benchmark_positions(n, cfg),
    ).run()
    return {
        "state": state_matrix[:n],
        "action_features": action_features[:n],
        "values": values[:n],
        "anchor_idx": anchor_idx[:n],
        "current": current[:n],
        "candidate_actions": candidate_spec.positions[:n],
        "candidate_labels": candidate_spec.labels,
        "positions": positions[:n],
        "baseline_stats": baseline_stats,
        "baseline_metrics": baseline_metrics.to_dict(),
    }


def _evaluate_variant(
    *,
    name: str,
    train_payload: dict,
    split_payloads: dict[str, dict],
    benchmark_position: float,
    cfg: CandidateQTrainConfig,
    device: str,
    reduce: str,
    target_mode: str = "value",
) -> dict:
    train_values = np.asarray(train_payload["values"], dtype=np.float32)
    if target_mode == "anchor_advantage":
        row = np.arange(len(train_values))
        anchor = np.asarray(train_payload["anchor_idx"], dtype=np.int64).clip(0, train_values.shape[1] - 1)
        anchor_values = train_values[row, anchor]
        train_values = train_values - anchor_values[:, None]
    models, meta = train_candidate_q_ensemble(
        train_state=train_payload["state"],
        train_action_features=train_payload["action_features"],
        train_values=train_values,
        train_anchor_idx=train_payload["anchor_idx"],
        cfg=cfg,
        device=device,
    )
    split_rows = {}
    for split_name, payload in split_payloads.items():
        q = predict_candidate_q(
            models,
            state=payload["state"],
            action_features=payload["action_features"],
            target_mean=meta["target_mean"],
            target_std=meta["target_std"],
            device=device,
            reduce=reduce,
        )
        if target_mode == "anchor_advantage":
            values = np.asarray(payload["values"], dtype=np.float32)
            row = np.arange(len(values))
            anchor = np.asarray(payload["anchor_idx"], dtype=np.int64).clip(0, values.shape[1] - 1)
            q = q + values[row, anchor][:, None]
        split_rows[split_name] = evaluate_candidate_q(
            q_pred=q,
            values=payload["values"],
            candidate_actions=payload["candidate_actions"],
            anchor_idx=payload["anchor_idx"],
            current_positions=payload["current"],
            benchmark_position=benchmark_position,
        )
    return {"name": name, "target_mode": target_mode, "train_meta": meta, "splits": split_rows}


def _write_md(path: str, *, config: str, fold_results: dict, candidate_mode: str, candidate_labels: list[str]) -> None:
    lines = [
        "# AC Candidate State-Action Critic Probe",
        "",
        f"Config: `{config}`",
        f"Candidate mode: `{candidate_mode}`",
        f"Candidate labels: `{', '.join(candidate_labels)}`",
        "",
        "## Summary",
        "",
        "This probe trains candidate Q(s, position_candidate) from realized transition values. Actor updates are not performed.",
        "",
    ]
    for fold, result in fold_results.items():
        lines.extend([f"## Fold {fold}", ""])
        lines.append("### Baseline Policy")
        lines.append("")
        lines.append("| split | AlphaEx pt/yr | SharpeDelta | MaxDDDelta pt | long | short | flat | turnover |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for split_name, base in result["baseline"].items():
            m = base["metrics"]
            s = base["stats"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        split_name,
                        _metric_str(100.0 * (m.get("alpha_excess") or 0.0), 2),
                        _metric_str(m.get("sharpe_delta"), 3),
                        _metric_str(100.0 * (m.get("maxdd_delta") or 0.0), 2),
                        _metric_str(s.get("long"), 3),
                        _metric_str(s.get("short"), 3),
                        _metric_str(s.get("flat"), 3),
                        _metric_str(s.get("turnover"), 2),
                    ]
                )
                + " |"
            )
        lines.extend(["", "### Candidate Q Metrics", ""])
        lines.append("| variant | split | RMSE | flat corr | row Spearman | top1 match | selected adv vs anchor | best possible adv | top-decile selected adv | selected short | selected flat | selected long | mean abs delta |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for variant in result["variants"]:
            for split_name, row in variant["splits"].items():
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            variant["name"],
                            split_name,
                            _metric_str(row.get("rmse"), 6),
                            _metric_str(row.get("flat_pearson"), 4),
                            _metric_str(row.get("row_spearman"), 4),
                            _metric_str(row.get("top1_best_match"), 3),
                            _metric_str(row.get("selected_realized_adv_vs_anchor"), 6),
                            _metric_str(row.get("best_possible_adv_vs_anchor"), 6),
                            _metric_str(row.get("top_decile_selected_realized_adv_vs_anchor"), 6),
                            _metric_str(row.get("selected_short_rate"), 3),
                            _metric_str(row.get("selected_flat_rate"), 3),
                            _metric_str(row.get("selected_long_rate"), 3),
                            _metric_str(row.get("selected_mean_abs_delta"), 4),
                        ]
                    )
                    + " |"
                )
        lines.extend(["", "### Reading", ""])
        lines.append("- Positive row Spearman means the critic can rank candidate actions inside a state.")
        lines.append("- `selected adv vs anchor` should be positive before actor residual updates are allowed.")
        lines.append("- High selected short/long concentration indicates one-sided Q bias.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.ac_candidate_q_probe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--folds", default="4")
    parser.add_argument("--candidate-actions", default=None)
    parser.add_argument("--candidate-mode", choices=("dynamic", "fixed"), default="dynamic")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ensemble-size", type=int, default=2)
    parser.add_argument("--output-json", default="documents/ac_candidate_q_probe.json")
    parser.add_argument("--output-md", default="documents/ac_candidate_q_probe.md")
    add_device_argument(parser)
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    set_seed(args.seed)

    cfg = load_config(args.config)
    cfg, _active_profile = resolve_costs(cfg, None)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = cfg.get("logging", {}).get("checkpoint_dir", "checkpoints")
    candidate_actions = _parse_candidates(args.candidate_actions, cfg)
    candidate_labels = [f"{x:.2f}" for x in candidate_actions]
    benchmark_position = _benchmark_position_value(cfg)

    data_cfg = cfg.get("data", {})
    symbol = data_cfg.get("symbol", "BTCUSDT")
    interval = data_cfg.get("interval", "15m")
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_dir = "checkpoints/data_cache"
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
        print(f"[CandidateQ] fold={split.fold_idx} mode={args.candidate_mode} fixed_candidates={candidate_actions.tolist()}")
        wfo_dataset = WFODataset(features_df, raw_returns, split, seq_len=cfg.get("data", {}).get("seq_len", 64))
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
            benchmark_position=benchmark_position,
            forward_window_stats_fn=_forward_window_stats,
            log_ts=_ts,
        )
        ensemble, wm_trainer = prepare_world_model_stage(
            fold_idx=split.fold_idx,
            obs_dim=wfo_dataset.obs_dim,
            cfg=cfg,
            device=args.device,
            has_wm=True,
            wm_path=fold_runtime["wm_path"],
            wfo_dataset=wfo_dataset,
            oracle_positions=fold_inputs["oracle_positions"],
            val_oracle_positions=fold_inputs["val_oracle_positions"],
            train_returns=fold_inputs["train_returns"],
            train_regime_probs=fold_inputs["train_regime_probs"],
            val_regime_probs=fold_inputs["val_regime_probs"],
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
            oracle_action_values=fold_inputs["oracle_bundle"]["oracle_action_values"],
            oracle_positions=fold_inputs["oracle_positions"],
            oracle_values=fold_inputs["oracle_bundle"]["oracle_values"],
            train_regime_probs=fold_inputs["train_regime_probs"],
            outcome_edge=fold_inputs["outcome_edge"],
            ac_cfg=cfg.get("ac", {}),
            bc_cfg=cfg.get("bc", {}),
            reward_cfg=reward_cfg,
            oracle_teacher_mode=fold_inputs["oracle_bundle"]["oracle_teacher_mode"],
        )
        actor = bc_setup["actor"]
        trainer = build_bc_trainer(
            actor=actor,
            ensemble=ensemble,
            bc_cfg=cfg.get("bc", {}),
            oracle_cfg=fold_inputs["oracle_cfg"],
            ac_cfg=cfg.get("ac", {}),
            reward_cfg=reward_cfg,
            device=args.device,
        )
        trainer.load(fold_runtime["bc_path"])

        run_val_selector_stage(
            actor=actor,
            wm_trainer=wm_trainer,
            wfo_dataset=wfo_dataset,
            seq_len=seq_len,
            val_regime_probs=fold_inputs["val_regime_probs"],
            val_advantage_values=val_advantage_values,
            device=args.device,
            cfg=cfg,
            ac_cfg=cfg.get("ac", {}),
            costs_cfg=costs_cfg,
            backtest_cls=Backtest,
            action_stats_fn=_action_stats,
            selector_cfg_fn=_selector_cfg,
            selector_candidate_fn=_selector_candidate,
            select_policy_candidate_fn=_select_policy_candidate,
            candidate_to_text_fn=_candidate_to_text,
            benchmark_positions_fn=lambda length: _benchmark_positions(length, cfg),
            benchmark_position=benchmark_position,
        )

        payloads = {
            "train": _split_payload(
                actor=actor,
                enc=enc_train,
                returns=wfo_dataset.train_returns,
                regime=fold_inputs["train_regime_probs"],
                advantage=train_advantage_values,
                candidate_actions=candidate_actions,
                candidate_mode=args.candidate_mode,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
                device=args.device,
            ),
            "val": _split_payload(
                actor=actor,
                enc=enc_val,
                returns=wfo_dataset.val_returns,
                regime=fold_inputs["val_regime_probs"],
                advantage=val_advantage_values,
                candidate_actions=candidate_actions,
                candidate_mode=args.candidate_mode,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
                device=args.device,
            ),
            "test": _split_payload(
                actor=actor,
                enc=enc_test,
                returns=wfo_dataset.test_returns,
                regime=fold_inputs["test_regime_probs"],
                advantage=test_advantage_values,
                candidate_actions=candidate_actions,
                candidate_mode=args.candidate_mode,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
                device=args.device,
            ),
        }
        candidate_labels = payloads["train"].get("candidate_labels", candidate_labels)
        variants = [
            (
                "mse_ensemble_mean",
                CandidateQTrainConfig(
                    steps=args.steps,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden,
                    n_layers=args.layers,
                    ensemble_size=args.ensemble_size,
                    seed=args.seed,
                    cql_lite_coef=0.0,
                ),
                "mean",
                "value",
            ),
            (
                "cql_lite_minq",
                CandidateQTrainConfig(
                    steps=args.steps,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden,
                    n_layers=args.layers,
                    ensemble_size=args.ensemble_size,
                    seed=args.seed + 100,
                    cql_lite_coef=0.05,
                    cql_temperature=1.0,
                ),
                "min",
                "value",
            ),
            (
                "anchor_adv_mse_mean",
                CandidateQTrainConfig(
                    steps=args.steps,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden,
                    n_layers=args.layers,
                    ensemble_size=args.ensemble_size,
                    seed=args.seed + 200,
                    cql_lite_coef=0.0,
                ),
                "mean",
                "anchor_advantage",
            ),
            (
                "anchor_adv_cql_minq",
                CandidateQTrainConfig(
                    steps=args.steps,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden,
                    n_layers=args.layers,
                    ensemble_size=args.ensemble_size,
                    seed=args.seed + 300,
                    cql_lite_coef=0.01,
                    cql_temperature=1.0,
                ),
                "min",
                "anchor_advantage",
            ),
            (
                "anchor_rank_ce_mean",
                CandidateQTrainConfig(
                    steps=args.steps,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden,
                    n_layers=args.layers,
                    ensemble_size=args.ensemble_size,
                    seed=args.seed + 400,
                    cql_lite_coef=0.0,
                    rank_ce_coef=0.5,
                    rank_tau=0.25,
                ),
                "mean",
                "anchor_advantage",
            ),
            (
                "anchor_rank_ce_cql_minq",
                CandidateQTrainConfig(
                    steps=args.steps,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden,
                    n_layers=args.layers,
                    ensemble_size=args.ensemble_size,
                    seed=args.seed + 500,
                    cql_lite_coef=0.01,
                    cql_temperature=1.0,
                    rank_ce_coef=0.5,
                    rank_tau=0.25,
                ),
                "min",
                "anchor_advantage",
            ),
            (
                "anchor_margin_rank_m010_mean",
                CandidateQTrainConfig(
                    steps=args.steps,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden,
                    n_layers=args.layers,
                    ensemble_size=args.ensemble_size,
                    seed=args.seed + 600,
                    cql_lite_coef=0.0,
                    rank_ce_coef=0.5,
                    rank_tau=0.5,
                    rank_target_mode="margin_best",
                    rank_margin=0.10,
                ),
                "mean",
                "anchor_advantage",
            ),
            (
                "anchor_margin_rank_m025_mean",
                CandidateQTrainConfig(
                    steps=args.steps,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden,
                    n_layers=args.layers,
                    ensemble_size=args.ensemble_size,
                    seed=args.seed + 700,
                    cql_lite_coef=0.0,
                    rank_ce_coef=0.5,
                    rank_tau=0.5,
                    rank_target_mode="margin_best",
                    rank_margin=0.25,
                ),
                "mean",
                "anchor_advantage",
            ),
        ]
        variant_rows = []
        for name, train_cfg, reduce, target_mode in variants:
            print(f"[CandidateQ] training variant={name} reduce={reduce}")
            variant_rows.append(
                _evaluate_variant(
                    name=name,
                    train_payload=payloads["train"],
                    split_payloads=payloads,
                    benchmark_position=benchmark_position,
                    cfg=train_cfg,
                    device=args.device,
                    reduce=reduce,
                    target_mode=target_mode,
                )
            )
            test_row = variant_rows[-1]["splits"]["test"]
            print(
                "[CandidateQ] "
                f"{name} test row_spearman={test_row['row_spearman']:.4f} "
                f"selected_adv={test_row['selected_realized_adv_vs_anchor']:.6f} "
                f"short={test_row['selected_short_rate']:.1%} flat={test_row['selected_flat_rate']:.1%} "
                f"long={test_row['selected_long_rate']:.1%}"
            )
        all_results[str(split.fold_idx)] = {
            "candidate_actions": candidate_actions.tolist(),
            "candidate_mode": args.candidate_mode,
            "candidate_labels": candidate_labels,
            "baseline": {
                split_name: {
                    "stats": payload["baseline_stats"],
                    "metrics": payload["baseline_metrics"],
                }
                for split_name, payload in payloads.items()
            },
            "variants": variant_rows,
        }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(all_results), f, ensure_ascii=False, indent=2)
    _write_md(
        args.output_md,
        config=args.config,
        fold_results=all_results,
        candidate_mode=args.candidate_mode,
        candidate_labels=candidate_labels,
    )
    print(f"[CandidateQ] wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
