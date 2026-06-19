"""UniDream training CLI.

SPEC.md の実装順序に従って以下を実行する:
  1. データ取得・特徴量計算
  2. WFO 分割
  3. train 期間で Hindsight Oracle 計算
  4. Transformer 世界モデル学習
  5. BC 初期化（Actor を oracle に模倣させる）
  6. Imagination AC fine-tune
  7. test 期間バックテスト
  8. PBO / Deflated Sharpe による過学習検出
  9. HMM レジーム別メトリクス

Usage:
    python -m unidream.cli.train --config configs/trading.yaml --seed 7 --device cuda
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import torch

from unidream.data.download import (
    fetch_binance_ohlcv,
    fetch_funding_rate,
    fetch_open_interest_hist,
    fetch_mark_price_klines,
)
from unidream.data.features import compute_features, get_raw_returns
from unidream.data.oracle import _forward_window_stats, oracle_to_dataset
from unidream.data.dataset import WFODataset, SequenceDataset
from unidream.actor_critic.imagination_ac import _action_stats, _fmt_action_stats, _ac_alerts_ascii as _ac_alerts
from unidream.device import DEVICE_HELP, resolve_device
from unidream.eval.backtest import Backtest, pnl_attribution
from unidream.experiments.runtime import load_config, resolve_costs, set_seed
from unidream.experiments.fold_runtime import resolve_ac_max_steps
from unidream.experiments.fold_inputs import prepare_fold_inputs
from unidream.experiments.train_app import run_training_app
from unidream.experiments.m2 import (
    benchmark_position_value as _benchmark_position_value,
    directional_collapse as _directional_collapse,
    format_m2_scorecard as _format_m2_scorecard,
    m2_scorecard as _m2_scorecard,
)
from unidream.experiments.ac_stage import run_ac_stage
from unidream.experiments.bc_setup import prepare_bc_setup
from unidream.experiments.bc_stage import run_bc_stage
from unidream.experiments.test_stage import run_test_stage
from unidream.experiments.val_selector_stage import run_val_selector_stage
from unidream.experiments.wm_stage import prepare_world_model_stage
from unidream.experiments.predictive_state import build_wm_predictive_state_bundle
from unidream.experiments.run_config import configure_determinism, load_training_run_config


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _benchmark_positions(length: int, cfg: dict) -> np.ndarray:
    benchmark_pos = cfg.get("reward", {}).get("benchmark_position", 1.0)
    return np.full(length, benchmark_pos, dtype=np.float64)


def _policy_score(metrics, stats: dict, benchmark_position: float = 0.0) -> tuple[float, str]:
    alpha_excess = 100.0 * (metrics.alpha_excess or 0.0)
    sharpe_delta = metrics.sharpe_delta or 0.0
    score = 2.0 * alpha_excess + 5.0 * sharpe_delta
    penalty = 0.0
    overlay_mode = abs(float(benchmark_position)) > 1e-8
    directional_collapse = (
        max(stats["long"], stats["short"]) >= 0.80
        and stats["switches"] <= 5
        and stats["turnover"] < 1.0
    )
    if alpha_excess < 0.0:
        penalty += 100.0 + 0.5 * abs(alpha_excess)
    if overlay_mode:
        score -= 5.0 * stats["turnover"]
    if not overlay_mode and stats["flat"] >= 0.50:
        penalty += 30.0
    if not overlay_mode and stats["flat"] >= 0.80:
        penalty += 100.0
    if directional_collapse and stats["long"] >= 0.85:
        penalty += 120.0
    if directional_collapse and stats["short"] >= 0.85:
        penalty += 120.0
    if directional_collapse or (not overlay_mode and stats["flat"] >= 0.80):
        penalty += 200.0
    if stats["avg_hold"] < 2.0:
        penalty += 10.0
    if not overlay_mode and stats["switches"] == 0:
        penalty += 25.0
    score -= penalty
    label = (
        f"alpha={alpha_excess:+.2f}pt sharpeΔ={sharpe_delta:+.3f} score={score:.3f} "
        f"long={stats['long']:.0%} short={stats['short']:.0%} flat={stats['flat']:.0%}"
    )
    return score, label


def _selector_cfg(ac_cfg: dict) -> dict:
    return {
        "reject_alpha_floor_pt": float(ac_cfg.get("selector_reject_alpha_floor_pt", -25.0)),
        "reject_sharpe_floor": float(ac_cfg.get("selector_reject_sharpe_floor", -1.0)),
        "reject_maxdd_worse_pt": float(ac_cfg.get("selector_reject_maxdd_worse_pt", 5.0)),
        "reject_win_rate_floor": float(ac_cfg.get("selector_reject_win_rate_floor", 0.35)),
        "win_rate_metric": str(ac_cfg.get("selector_win_rate_metric", "period")),
        "max_turnover": float(ac_cfg.get("selector_max_turnover", 8.0)),
        "min_avg_hold": float(ac_cfg.get("selector_min_avg_hold", 3.0)),
        "max_directional_ratio": float(ac_cfg.get("selector_max_directional_ratio", 1.01)),
        "directional_penalty_coef": float(ac_cfg.get("selector_directional_penalty_coef", 80.0)),
        "directional_soft_limit": float(ac_cfg.get("selector_directional_soft_limit", 0.90)),
        "confirm_alpha_tol_pt": float(ac_cfg.get("selector_confirm_alpha_tol_pt", 5.0)),
        "confirm_sharpe_tol": float(ac_cfg.get("selector_confirm_sharpe_tol", 0.20)),
        "confirm_score_tol": float(ac_cfg.get("selector_confirm_score_tol", 15.0)),
        "turnover_score_coef": float(ac_cfg.get("selector_turnover_score_coef", 3.0)),
        "maxdd_score_coef": float(ac_cfg.get("selector_maxdd_score_coef", 50.0)),
        "maxdd_worse_score_coef": float(ac_cfg.get("selector_maxdd_worse_score_coef", 1.5)),
        "maxdd_improve_score_coef": float(ac_cfg.get("selector_maxdd_improve_score_coef", 0.5)),
        "win_rate_score_coef": float(ac_cfg.get("selector_win_rate_score_coef", 20.0)),
        "alpha_score_coef": float(ac_cfg.get("selector_alpha_score_coef", 2.0)),
        "sharpe_score_coef": float(ac_cfg.get("selector_sharpe_score_coef", 5.0)),
        "turnover_target": float(ac_cfg.get("selector_turnover_target", ac_cfg.get("selector_max_turnover", 8.0))),
        "turnover_excess_score_coef": float(ac_cfg.get("selector_turnover_excess_score_coef", 0.0)),
        "period_win_bonus_coef": float(ac_cfg.get("selector_period_win_bonus_coef", 0.0)),
        "max_long_rate": float(ac_cfg.get("selector_max_long_rate", 1.0)),
        "max_short_rate": float(ac_cfg.get("selector_max_short_rate", 1.0)),
        "hard_maxdd_delta_pt": float(ac_cfg.get("selector_hard_maxdd_delta_pt", float("inf"))),
        "near_best_tiebreak": str(ac_cfg.get("selector_near_best_tiebreak", "conservative")),
        "m2_bonus": float(ac_cfg.get("selector_m2_bonus", 15.0)),
        "stretch_bonus": float(ac_cfg.get("selector_stretch_bonus", 5.0)),
        "active_alpha_min_pt": float(ac_cfg.get("selector_active_alpha_min_pt", 8.0)),
        "active_sharpe_min": float(ac_cfg.get("selector_active_sharpe_min", 0.05)),
        "active_maxdd_worse_pt": float(ac_cfg.get("selector_active_maxdd_worse_pt", 0.0)),
        "active_min_win_rate": float(ac_cfg.get("selector_active_min_win_rate", 0.48)),
        "active_score_margin": float(ac_cfg.get("selector_active_score_margin", 5.0)),
    }


def _is_benchmark_hold(stats: dict, benchmark_position: float) -> bool:
    overlay_mode = abs(float(benchmark_position)) > 1e-8
    return overlay_mode and stats["flat"] >= 0.95 and stats["switches"] == 0


def _selector_candidate(
    candidate: dict[str, float],
    metrics,
    stats: dict,
    benchmark_position: float,
    selector_cfg: dict,
    cfg: dict | None = None,
) -> dict:
    alpha_excess_pt = 100.0 * float(metrics.alpha_excess or 0.0)
    sharpe_delta = float(metrics.sharpe_delta or 0.0)
    max_dd = abs(float(metrics.max_drawdown or 0.0))
    maxdd_delta_pt = 100.0 * float(metrics.maxdd_delta or 0.0)
    win_rate_vs_bh = float(metrics.win_rate_vs_bh or 0.0)
    period_win_raw = getattr(metrics, "period_win_rate_vs_bh", None)
    period_win_rate_vs_bh = win_rate_vs_bh if period_win_raw is None else float(period_win_raw)
    selector_win_rate = period_win_rate_vs_bh if selector_cfg["win_rate_metric"] == "period" else win_rate_vs_bh
    overlay_mode = abs(float(benchmark_position)) > 1e-8
    benchmark_hold = _is_benchmark_hold(stats, benchmark_position)
    directional_ratio = max(stats["long"], stats["short"])
    directional_collapse = _directional_collapse(stats)
    scorecard = _m2_scorecard(metrics, stats, cfg or {})
    reject_reason = None

    if not benchmark_hold:
        if alpha_excess_pt <= selector_cfg["reject_alpha_floor_pt"]:
            reject_reason = f"alpha<{selector_cfg['reject_alpha_floor_pt']:.1f}"
        elif sharpe_delta <= selector_cfg["reject_sharpe_floor"]:
            reject_reason = f"sharpeΔ<{selector_cfg['reject_sharpe_floor']:.2f}"
        elif maxdd_delta_pt > selector_cfg["reject_maxdd_worse_pt"]:
            reject_reason = f"maxddΔ>{selector_cfg['reject_maxdd_worse_pt']:.1f}pt"
        elif selector_win_rate < selector_cfg["reject_win_rate_floor"]:
            reject_reason = f"win<{selector_cfg['reject_win_rate_floor']:.0%}"
        elif stats["turnover"] > selector_cfg["max_turnover"]:
            reject_reason = f"turnover>{selector_cfg['max_turnover']:.2f}"
        elif stats["long"] > selector_cfg["max_long_rate"]:
            reject_reason = f"long>{selector_cfg['max_long_rate']:.0%}"
        elif stats["short"] > selector_cfg["max_short_rate"]:
            reject_reason = f"short>{selector_cfg['max_short_rate']:.0%}"
        elif maxdd_delta_pt > selector_cfg["hard_maxdd_delta_pt"]:
            reject_reason = f"hard_maxddΔ>{selector_cfg['hard_maxdd_delta_pt']:.1f}pt"
        elif stats["avg_hold"] < selector_cfg["min_avg_hold"]:
            reject_reason = f"avg_hold<{selector_cfg['min_avg_hold']:.1f}"
        elif directional_collapse or not scorecard["collapse_guard_pass"]:
            reject_reason = "collapse_guard"
        elif directional_ratio >= selector_cfg["max_directional_ratio"]:
            reject_reason = f"one_sided>{selector_cfg['max_directional_ratio']:.2f}"
        elif (not overlay_mode) and stats["flat"] >= 0.80:
            reject_reason = "flat_collapse"

    directional_penalty = 0.0
    if not benchmark_hold:
        directional_penalty = selector_cfg["directional_penalty_coef"] * max(
            0.0, directional_ratio - selector_cfg["directional_soft_limit"]
        )

    turnover = float(stats["turnover"])
    turnover_penalty = selector_cfg["turnover_score_coef"] * turnover
    turnover_penalty += selector_cfg["turnover_excess_score_coef"] * max(
        0.0,
        turnover - selector_cfg["turnover_target"],
    )
    period_bonus = selector_cfg["period_win_bonus_coef"] * max(0.0, selector_win_rate - 0.5)
    score = (
        selector_cfg["alpha_score_coef"] * alpha_excess_pt
        + selector_cfg["sharpe_score_coef"] * sharpe_delta
        - turnover_penalty
        - selector_cfg["maxdd_score_coef"] * max_dd
        - selector_cfg["maxdd_worse_score_coef"] * max(0.0, maxdd_delta_pt)
        + selector_cfg["maxdd_improve_score_coef"] * max(0.0, -maxdd_delta_pt)
        + selector_cfg["win_rate_score_coef"] * (selector_win_rate - 0.5)
        + period_bonus
        - directional_penalty
    )
    if scorecard["m2_pass"]:
        score += selector_cfg["m2_bonus"]
    elif scorecard["stretch_hit"]:
        score += selector_cfg["stretch_bonus"]
    if benchmark_hold:
        score += 0.5
    if reject_reason is not None:
        score -= 500.0

    label = (
        f"alpha={alpha_excess_pt:+.2f}pt sharpeΔ={sharpe_delta:+.3f} "
        f"score={score:.3f} long={stats['long']:.0%} short={stats['short']:.0%} "
        f"flat={stats['flat']:.0%} turnover={turnover:.2f}"
    )
    label = label.replace(
        " score=",
        f" maxddΔ={maxdd_delta_pt:+.2f}pt "
        f"barwin={win_rate_vs_bh:.1%} periodwin={period_win_rate_vs_bh:.1%} score=",
        1,
    )
    label += f" M2={'pass' if scorecard['m2_pass'] else 'miss'}"
    if reject_reason is not None:
        label += f" reject={reject_reason}"

    return {
        "candidate": candidate,
        "score": float(score),
        "label": label,
        "alpha_excess_pt": alpha_excess_pt,
        "sharpe_delta": sharpe_delta,
        "max_drawdown": max_dd,
        "maxdd_delta_pt": maxdd_delta_pt,
        "win_rate_vs_bh": win_rate_vs_bh,
        "period_win_rate_vs_bh": period_win_rate_vs_bh,
        "selector_win_rate": selector_win_rate,
        "stats": stats,
        "reject_reason": reject_reason,
        "benchmark_hold": benchmark_hold,
        "scorecard": scorecard,
    }


def _candidate_to_text(candidate: dict[str, float]) -> str:
    scale = float(candidate["scale"])
    adv = float(candidate["adv"])
    return f"scale={scale:.3f} adv={adv:.2f}"


def _select_policy_candidate(candidates: list[dict], selector_cfg: dict) -> dict:
    valid = [c for c in candidates if c["reject_reason"] is None]
    pool = valid if valid else candidates
    benchmark_hold = next((c for c in pool if c["benchmark_hold"]), None)
    best = max(pool, key=lambda c: c["score"])
    if benchmark_hold is not None and not best["benchmark_hold"]:
        active_is_strong = (
            best["alpha_excess_pt"] >= selector_cfg["active_alpha_min_pt"]
            and best["maxdd_delta_pt"] <= selector_cfg["active_maxdd_worse_pt"]
            and best["selector_win_rate"] >= selector_cfg["active_min_win_rate"]
            and best["score"] >= benchmark_hold["score"] + selector_cfg["active_score_margin"]
            and (
                best["scorecard"]["m2_pass"]
                or (
                    best["scorecard"]["stretch_hit"]
                    and best["sharpe_delta"] >= selector_cfg["active_sharpe_min"]
                )
            )
        )
        if not active_is_strong:
            return benchmark_hold
    alpha_floor = best["alpha_excess_pt"] - selector_cfg["confirm_alpha_tol_pt"]
    sharpe_floor = best["sharpe_delta"] - selector_cfg["confirm_sharpe_tol"]
    score_floor = best["score"] - selector_cfg["confirm_score_tol"]
    near_best = [
        c for c in pool
        if c["score"] >= score_floor
        and c["alpha_excess_pt"] >= alpha_floor
        and c["sharpe_delta"] >= sharpe_floor
    ]
    if not near_best:
        near_best = [best]
    tiebreak_mode = selector_cfg.get("near_best_tiebreak", "conservative")
    if tiebreak_mode == "balanced":
        key_fn = lambda c: (
            -c["sharpe_delta"],
            c["maxdd_delta_pt"],
            -c["selector_win_rate"],
            c["stats"]["turnover"],
            -c["score"],
        )
    elif tiebreak_mode == "score":
        key_fn = lambda c: (
            -c["score"],
            -c["sharpe_delta"],
            c["maxdd_delta_pt"],
            c["stats"]["turnover"],
        )
    else:
        key_fn = lambda c: (
            0 if c["benchmark_hold"] else 1,
            c["stats"]["turnover"],
            c["maxdd_delta_pt"],
            -c["selector_win_rate"],
            -c["score"],
        )
    chosen = min(near_best, key=key_fn)
    return chosen
def run_fold(
    fold_idx: int,
    wfo_dataset: WFODataset,
    cfg: dict,
    device: str,
    checkpoint_dir: str,
) -> dict:
    """Train and evaluate one WFO fold from scratch."""
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx}: train {wfo_dataset.split.train_start.date()} → "
          f"{wfo_dataset.split.train_end.date()} | "
          f"test {wfo_dataset.split.test_start.date()} → "
          f"{wfo_dataset.split.test_end.date()}")
    print(f"{'='*60}")

    ac_cfg = cfg.get("ac", {})
    bc_cfg = cfg.get("bc", {})
    wm_cfg = cfg.get("world_model", {})
    costs_cfg = cfg.get("costs", {})

    obs_dim = wfo_dataset.obs_dim
    seq_len = cfg.get("data", {}).get("seq_len", 64)

    fold_ckpt_dir = os.path.join(checkpoint_dir, f"fold_{fold_idx}")
    os.makedirs(fold_ckpt_dir, exist_ok=True)
    wm_path = os.path.join(fold_ckpt_dir, "world_model.pt")
    bc_path = os.path.join(fold_ckpt_dir, "bc_actor.pt")
    ac_path = os.path.join(fold_ckpt_dir, "ac.pt")
    ac_max_steps_cfg = resolve_ac_max_steps(ac_cfg)

    reward_cfg = cfg.get("reward", {})
    fold_inputs = prepare_fold_inputs(
        fold_idx=fold_idx,
        wfo_dataset=wfo_dataset,
        cfg=cfg,
        costs_cfg=costs_cfg,
        ac_cfg=ac_cfg,
        bc_cfg=bc_cfg,
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
    oracle_action_values = oracle_bundle["oracle_action_values"]
    oracle_min_hold = oracle_bundle["oracle_min_hold"]
    oracle_soft_temp = oracle_bundle["oracle_soft_temp"]
    oracle_reward_mode = oracle_bundle["oracle_reward_mode"]
    oracle_benchmark_position = oracle_bundle["oracle_benchmark_position"]
    oracle_teacher_mode = oracle_bundle["oracle_teacher_mode"]
    oracle_values = oracle_bundle["oracle_values"]
    oracle_soft_labels = oracle_bundle["oracle_soft_labels"]
    oracle_positions = fold_inputs["oracle_positions"]
    val_oracle_positions = fold_inputs["val_oracle_positions"]
    outcome_edge = fold_inputs["outcome_edge"]
    hmm_det = fold_inputs["hmm_det"]
    regime_dim = fold_inputs["regime_dim"]
    train_regime_probs = fold_inputs["train_regime_probs"]
    val_regime_probs = fold_inputs["val_regime_probs"]
    test_regime_probs = fold_inputs["test_regime_probs"]
    train_advantage_values = fold_inputs.get("train_advantage_values")
    val_advantage_values = fold_inputs.get("val_advantage_values")
    test_advantage_values = fold_inputs.get("test_advantage_values")
    train_route_labels = fold_inputs.get("train_route_labels")
    train_route_soft_labels = fold_inputs.get("train_route_soft_labels")
    train_route_advantage = fold_inputs.get("train_route_advantage")

    # --------- Step 2: 世界モデル学習 ---------
    ensemble, wm_trainer = prepare_world_model_stage(
        fold_idx=fold_idx,
        obs_dim=obs_dim,
        cfg=cfg,
        device=device,
        has_wm=False,
        wm_path=wm_path,
        wfo_dataset=wfo_dataset,
        oracle_positions=oracle_positions,
        val_oracle_positions=val_oracle_positions,
        train_returns=train_returns,
        train_regime_probs=train_regime_probs,
        val_regime_probs=val_regime_probs,
        log_ts=_ts,
    )

    # train 期間の全シーケンスをエンコード
    # BC 学習・評価の一貫性のため no-action（flat）context で encode する。
    # oracle action context で h を作ると「未来情報リーク入り h」になり、
    # val/test（no-action h）との分布ギャップが大きすぎて BC が崩壊する。
    # z はエンコーダ出力で action-independent なため、どちらの encode でも同一。
    encoded = wm_trainer.encode_sequence(
        wfo_dataset.train_features,
        actions=None,   # no oracle context → BC/val/test で h 分布を一致させる
        seq_len=seq_len,
    )
    z_train = encoded["z"]
    h_train = encoded["h"]

    predictive_bundle = build_wm_predictive_state_bundle(
        wm_trainer=wm_trainer,
        wfo_dataset=wfo_dataset,
        z_train=z_train,
        h_train=h_train,
        seq_len=seq_len,
        ac_cfg=ac_cfg,
        log_ts=_ts,
    )
    if predictive_bundle is not None:
        ac_cfg["advantage_conditioned"] = True
        ac_cfg["advantage_dim"] = int(predictive_bundle["train"].shape[1])
        train_advantage_values = predictive_bundle["train"]
        val_advantage_values = predictive_bundle["val"]
        test_advantage_values = predictive_bundle["test"]

    # --------- Step 3: BC 初期化 ---------
    bc_setup = prepare_bc_setup(
        ensemble=ensemble,
        oracle_action_values=oracle_action_values,
        oracle_positions=oracle_positions,
        oracle_values=oracle_values,
        train_regime_probs=train_regime_probs,
        outcome_edge=outcome_edge,
        ac_cfg=ac_cfg,
        bc_cfg=bc_cfg,
        reward_cfg=reward_cfg,
        oracle_teacher_mode=oracle_teacher_mode,
    )
    actor = bc_setup["actor"]
    bc_sample_quality = bc_setup["bc_sample_quality"]
    bc_advantage_values = (
        train_advantage_values if train_advantage_values is not None else bc_setup["bc_advantage_values"]
    )

    bc_trainer = run_bc_stage(
        actor=actor,
        ensemble=ensemble,
        bc_cfg=bc_cfg,
        oracle_cfg=oracle_cfg,
        ac_cfg=ac_cfg,
        reward_cfg=reward_cfg,
        device=device,
        bc_path=bc_path,
        z_train=z_train,
        h_train=h_train,
        oracle_positions=oracle_positions,
        train_regime_probs=train_regime_probs,
        oracle_soft_labels=oracle_soft_labels,
        bc_sample_quality=bc_sample_quality,
        bc_advantage_values=bc_advantage_values,
        train_returns=train_returns,
        train_route_labels=train_route_labels,
        train_route_soft_labels=train_route_soft_labels,
        train_route_advantage=train_route_advantage,
        log_ts=_ts,
    )

    # --------- Step 4: Imagination AC Fine-tune ---------
    # no-action h（z_train/h_train）を AC でもそのまま使う。
    # BC-action re-encode は廃止: test/val も no-action h のため、
    # re-encode すると AC 開始状態が no-action h から外れ、
    # BC 正則化（_oracle_z/_oracle_h = no-action h）と AC gradient が
    # 異なる h 分布上で計算されて矛盾した gradient が生じる。
    # no-action h に統一することで BC train/AC train/val/test が一貫する。

    ac_trainer = run_ac_stage(
        actor=actor,
        ensemble=ensemble,
        cfg=cfg,
        ac_cfg=ac_cfg,
        wm_cfg=wm_cfg,
        costs_cfg=costs_cfg,
        device=device,
        ac_path=ac_path,
        z_train=z_train,
        h_train=h_train,
        oracle_positions=oracle_positions,
        train_regime_probs=train_regime_probs,
        train_advantage_values=train_advantage_values,
        wfo_dataset=wfo_dataset,
        wm_trainer=wm_trainer,
        seq_len=seq_len,
        val_regime_probs=val_regime_probs,
        val_advantage_values=val_advantage_values,
        val_oracle_positions=val_oracle_positions,
        ac_max_steps_cfg=ac_max_steps_cfg,
        log_ts=_ts,
        backtest_cls=Backtest,
        pnl_attribution_fn=pnl_attribution,
        action_stats_fn=_action_stats,
        format_action_stats_fn=_fmt_action_stats,
        ac_alerts_fn=_ac_alerts,
        benchmark_positions_fn=lambda length: _benchmark_positions(length, cfg),
        benchmark_position=_benchmark_position_value(cfg),
        policy_score_fn=_policy_score,
        sequence_dataset_cls=SequenceDataset,
    )

    run_val_selector_stage(
        actor=actor,
        wm_trainer=wm_trainer,
        wfo_dataset=wfo_dataset,
        seq_len=seq_len,
        val_regime_probs=val_regime_probs,
        val_advantage_values=val_advantage_values,
        device=device,
        cfg=cfg,
        ac_cfg=ac_cfg,
        costs_cfg=costs_cfg,
        backtest_cls=Backtest,
        action_stats_fn=_action_stats,
        selector_cfg_fn=_selector_cfg,
        selector_candidate_fn=_selector_candidate,
        select_policy_candidate_fn=_select_policy_candidate,
        candidate_to_text_fn=_candidate_to_text,
        benchmark_positions_fn=lambda length: _benchmark_positions(length, cfg),
        benchmark_position=_benchmark_position_value(cfg),
    )

    # --------- Step 5: Test バックテスト ---------
    test_result = run_test_stage(
        actor=actor,
        wm_trainer=wm_trainer,
        wfo_dataset=wfo_dataset,
        seq_len=seq_len,
        test_regime_probs=test_regime_probs,
        test_advantage_values=test_advantage_values,
        device=device,
        cfg=cfg,
        costs_cfg=costs_cfg,
        backtest_cls=Backtest,
        pnl_attribution_fn=pnl_attribution,
        action_stats_fn=_action_stats,
        format_action_stats_fn=_fmt_action_stats,
        ac_alerts_fn=_ac_alerts,
        benchmark_positions_fn=lambda length: _benchmark_positions(length, cfg),
        benchmark_position=_benchmark_position_value(cfg),
        m2_scorecard_fn=_m2_scorecard,
        format_m2_scorecard_fn=_format_m2_scorecard,
        log_ts=_ts,
        fold_idx=fold_idx,
        override_positions=None,
        override_policy_name=None,
    )
    test_result["fold"] = fold_idx
    return test_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m unidream.cli.train",
        description="Strict reproducible UniDream WM -> BC -> AC -> Test pipeline",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True, help="Self-contained training YAML")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True, help=DEVICE_HELP)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    cfg = load_config(args.config)
    cfg, active_cost_profile = resolve_costs(cfg)
    try:
        run_cfg = load_training_run_config(cfg)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    configure_determinism(args.seed)
    set_seed(args.seed)
    run_training_app(
        config_path=args.config,
        cfg=cfg,
        run_cfg=run_cfg,
        seed=args.seed,
        device=args.device,
        active_cost_profile=active_cost_profile,
        run_fold_fn=run_fold,
        format_m2_scorecard_fn=_format_m2_scorecard,
    )


if __name__ == "__main__":
    main()
