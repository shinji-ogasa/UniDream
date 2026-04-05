"""UniDream メイン学習スクリプト.

SPEC.md の実装順序に従って以下を実行する:
  1. データ取得・特徴量計算
  2. WFO 分割
  3. train 期間で Hindsight Oracle 計算
  4. BC 初期化（Actor を oracle に模倣させる）
  5. Transformer 世界モデル学習
  6. Imagination AC fine-tune
  7. test 期間バックテスト
  8. PBO / Deflated Sharpe による過学習検出
  9. HMM レジーム別メトリクス

Usage:
    python train.py [--config configs/trading.yaml] [--symbol BTCUSDT]
                    [--start 2018-01-01] [--end 2024-01-01]
                    [--device cuda] [--seed 42] [--resume]
"""
from __future__ import annotations

import argparse
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml

from unidream.data.download import fetch_binance_ohlcv, fetch_funding_rate, fetch_open_interest_hist
from unidream.data.features import compute_features, get_raw_returns, augment_with_rebound_features
from unidream.data.oracle import (
    _forward_window_stats,
    hindsight_oracle_dp,
    hindsight_signal_teacher,
    oracle_to_dataset,
    smooth_aim_positions,
    ACTIONS as _ACTIONS,
)
from unidream.data.dataset import get_wfo_splits, WFODataset, SequenceDataset
from unidream.world_model.train_wm import WorldModelTrainer, build_ensemble
from unidream.actor_critic.actor import Actor
from unidream.actor_critic.critic import Critic
from unidream.actor_critic.bc_pretrain import BCPretrainer
from unidream.actor_critic.imagination_ac import ImagACTrainer, _action_stats, _fmt_action_stats, _ac_alerts
from unidream.eval.backtest import Backtest, pnl_attribution
from unidream.eval.wfo import aggregate_wfo_results
from unidream.eval.pbo import compute_pbo, deflated_sharpe
from unidream.eval.regime import RegimeDetector, regime_metrics, print_regime_report


_CACHE_STALE_DAYS = 7  # キャッシュがこの日数以上古ければ再取得


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _cache_is_fresh(path: str, stale_days: int = _CACHE_STALE_DAYS) -> bool:
    """キャッシュファイルが存在し、stale_days 以内に更新されていれば True."""
    if not os.path.exists(path):
        return False
    import time
    age_days = (time.time() - os.path.getmtime(path)) / 86400
    return age_days < stale_days


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_PIPELINE_STAGES = ("wm", "bc", "ac", "test")
_STAGE_TO_INDEX = {stage: idx for idx, stage in enumerate(_PIPELINE_STAGES)}


def _stage_idx(stage: str) -> int:
    return _STAGE_TO_INDEX[stage]


def _benchmark_positions(length: int, cfg: dict) -> np.ndarray:
    benchmark_pos = cfg.get("reward", {}).get("benchmark_position", 1.0)
    return np.full(length, benchmark_pos, dtype=np.float64)


def _benchmark_position_value(cfg: dict) -> float:
    return float(cfg.get("reward", {}).get("benchmark_position", 1.0))


def _teacher_outcome_edge(
    returns: np.ndarray,
    positions: np.ndarray,
    benchmark_position: float,
    horizons: tuple[int, ...],
    horizon_weights: tuple[float, ...],
) -> np.ndarray:
    weighted_forward = np.zeros(len(returns), dtype=np.float32)
    total_weight = 0.0
    for horizon, weight in zip(horizons, horizon_weights):
        if weight == 0.0:
            continue
        fwd_mean, _ = _forward_window_stats(np.asarray(returns, dtype=np.float32), int(horizon))
        weighted_forward += float(weight) * fwd_mean.astype(np.float32)
        total_weight += float(weight)
    if total_weight > 0.0:
        weighted_forward /= total_weight
    action_edge = (np.asarray(positions, dtype=np.float32) - float(benchmark_position)) * weighted_forward
    return np.clip(action_edge, 0.0, None).astype(np.float32)


def _goal_cfg(cfg: dict) -> dict:
    targets = cfg.get("targets", {})
    return {
        "alpha_excess_pt": float(targets.get("alpha_excess_pt", 5.0)),
        "sharpe_delta": float(targets.get("sharpe_delta", 0.20)),
        "maxdd_delta_pt": float(targets.get("maxdd_delta_pt", -10.0)),
        "win_rate_vs_bh": float(targets.get("win_rate_vs_bh", 0.60)),
        "stretch_alpha_excess_pt": float(targets.get("stretch_alpha_excess_pt", 8.0)),
        "stretch_maxdd_delta_pt": float(targets.get("stretch_maxdd_delta_pt", -15.0)),
    }


def _collapse_guard(stats: dict, benchmark_position: float) -> tuple[bool, list[str]]:
    overlay_mode = abs(float(benchmark_position)) > 1e-8
    reasons: list[str] = []
    if _directional_collapse(stats):
        reasons.append("directional_collapse")
    if not overlay_mode and stats["flat"] >= 0.80:
        reasons.append("flat_collapse")
    return len(reasons) == 0, reasons


def _m2_scorecard(metrics, stats: dict, cfg: dict) -> dict:
    goals = _goal_cfg(cfg)
    benchmark_position = _benchmark_position_value(cfg)
    alpha_excess_pt = 100.0 * float(metrics.alpha_excess or 0.0)
    sharpe_delta = float(metrics.sharpe_delta or 0.0)
    maxdd_delta_pt = 100.0 * float(metrics.maxdd_delta or 0.0)
    win_rate_vs_bh = float(metrics.win_rate_vs_bh or 0.0)
    collapse_pass, collapse_reasons = _collapse_guard(stats, benchmark_position)
    required = {
        "alpha_excess": alpha_excess_pt >= goals["alpha_excess_pt"],
        "sharpe_delta": sharpe_delta >= goals["sharpe_delta"],
        "maxdd_delta": maxdd_delta_pt <= goals["maxdd_delta_pt"],
        "win_rate_vs_bh": win_rate_vs_bh >= goals["win_rate_vs_bh"],
        "collapse_guard": collapse_pass,
    }
    stretch = {
        "alpha_excess": alpha_excess_pt >= goals["stretch_alpha_excess_pt"],
        "maxdd_delta": maxdd_delta_pt <= goals["stretch_maxdd_delta_pt"],
    }
    m2_pass = all(required.values())
    stretch_hit = any(stretch.values())
    return {
        "alpha_excess_pt": alpha_excess_pt,
        "sharpe_delta": sharpe_delta,
        "maxdd_delta_pt": maxdd_delta_pt,
        "win_rate_vs_bh": win_rate_vs_bh,
        "collapse_guard_pass": collapse_pass,
        "collapse_guard_reasons": collapse_reasons,
        "required": required,
        "stretch": stretch,
        "m2_pass": m2_pass,
        "stretch_hit": stretch_hit,
    }


def _format_m2_scorecard(scorecard: dict) -> str:
    guard = "pass" if scorecard["collapse_guard_pass"] else ",".join(scorecard["collapse_guard_reasons"])
    m2_state = "PASS" if scorecard["m2_pass"] else "MISS"
    stretch_state = "hit" if scorecard["stretch_hit"] else "miss"
    return (
        f"M2={m2_state} stretch={stretch_state} "
        f"alpha={scorecard['alpha_excess_pt']:+.2f}pt "
        f"sharpeΔ={scorecard['sharpe_delta']:+.3f} "
        f"maxddΔ={scorecard['maxdd_delta_pt']:+.2f}pt "
        f"win={scorecard['win_rate_vs_bh']:.1%} "
        f"guard={guard}"
    )


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
        "m2_bonus": float(ac_cfg.get("selector_m2_bonus", 15.0)),
        "stretch_bonus": float(ac_cfg.get("selector_stretch_bonus", 5.0)),
        "active_alpha_min_pt": float(ac_cfg.get("selector_active_alpha_min_pt", 8.0)),
        "active_sharpe_min": float(ac_cfg.get("selector_active_sharpe_min", 0.05)),
        "active_maxdd_worse_pt": float(ac_cfg.get("selector_active_maxdd_worse_pt", 0.0)),
        "active_min_win_rate": float(ac_cfg.get("selector_active_min_win_rate", 0.48)),
        "active_score_margin": float(ac_cfg.get("selector_active_score_margin", 5.0)),
    }


def _directional_collapse(stats: dict) -> bool:
    return (
        max(stats["long"], stats["short"]) >= 0.80
        and stats["switches"] <= 5
        and stats["turnover"] < 1.0
    )


def _is_benchmark_hold(stats: dict, benchmark_position: float) -> bool:
    overlay_mode = abs(float(benchmark_position)) > 1e-8
    return overlay_mode and stats["flat"] >= 0.95 and stats["switches"] == 0


def _selector_candidate(
    candidate: float,
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
        elif win_rate_vs_bh < selector_cfg["reject_win_rate_floor"]:
            reject_reason = f"win<{selector_cfg['reject_win_rate_floor']:.0%}"
        elif stats["turnover"] > selector_cfg["max_turnover"]:
            reject_reason = f"turnover>{selector_cfg['max_turnover']:.2f}"
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

    score = (
        2.0 * alpha_excess_pt
        + 5.0 * sharpe_delta
        - selector_cfg["turnover_score_coef"] * float(stats["turnover"])
        - selector_cfg["maxdd_score_coef"] * max_dd
        - selector_cfg["maxdd_worse_score_coef"] * max(0.0, maxdd_delta_pt)
        + selector_cfg["maxdd_improve_score_coef"] * max(0.0, -maxdd_delta_pt)
        + selector_cfg["win_rate_score_coef"] * (win_rate_vs_bh - 0.5)
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
        f"flat={stats['flat']:.0%}"
    )
    label = label.replace(
        " score=",
        f" maxddΔ={maxdd_delta_pt:+.2f}pt win={win_rate_vs_bh:.1%} score=",
        1,
    )
    label += f" M2={'pass' if scorecard['m2_pass'] else 'miss'}"
    if reject_reason is not None:
        label += f" reject={reject_reason}"

    return {
        "candidate": float(candidate),
        "score": float(score),
        "label": label,
        "alpha_excess_pt": alpha_excess_pt,
        "sharpe_delta": sharpe_delta,
        "max_drawdown": max_dd,
        "maxdd_delta_pt": maxdd_delta_pt,
        "win_rate_vs_bh": win_rate_vs_bh,
        "stats": stats,
        "reject_reason": reject_reason,
        "benchmark_hold": benchmark_hold,
        "scorecard": scorecard,
    }


def _select_policy_candidate(candidates: list[dict], selector_cfg: dict) -> dict:
    valid = [c for c in candidates if c["reject_reason"] is None]
    pool = valid if valid else candidates
    benchmark_hold = next((c for c in pool if c["benchmark_hold"]), None)
    best = max(pool, key=lambda c: c["score"])
    if benchmark_hold is not None and not best["benchmark_hold"]:
        active_is_strong = (
            best["alpha_excess_pt"] >= selector_cfg["active_alpha_min_pt"]
            and best["maxdd_delta_pt"] <= selector_cfg["active_maxdd_worse_pt"]
            and best["win_rate_vs_bh"] >= selector_cfg["active_min_win_rate"]
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
    chosen = min(
        near_best,
        key=lambda c: (
            0 if c["benchmark_hold"] else 1,
            c["stats"]["turnover"],
            c["maxdd_delta_pt"],
            -c["win_rate_vs_bh"],
            -c["score"],
        ),
    )
    return chosen


def resolve_costs(cfg: dict, cost_profile: str | None = None) -> tuple[dict, str]:
    """Resolve trading costs from either legacy `costs` or named `cost_profiles`."""
    resolved_cfg = dict(cfg)
    profile_name = cost_profile or cfg.get("cost_profile") or "default"
    profiles = cfg.get("cost_profiles")

    if profiles:
        if profile_name == "default":
            profile_name = "base" if "base" in profiles else next(iter(profiles))
        if profile_name not in profiles:
            available = ", ".join(profiles.keys())
            raise KeyError(f"Unknown cost profile '{profile_name}'. Available: {available}")
        resolved_cfg["costs"] = dict(profiles[profile_name])
        resolved_cfg["cost_profile"] = profile_name
    else:
        resolved_cfg["costs"] = dict(cfg.get("costs", {}))
        resolved_cfg["cost_profile"] = profile_name

    return resolved_cfg, resolved_cfg["cost_profile"]


def run_fold(
    fold_idx: int,
    wfo_dataset: WFODataset,
    cfg: dict,
    device: str,
    checkpoint_dir: str,
    resume: bool = False,
    start_from: str = "wm",
    stop_after: str = "test",
) -> dict:
    """1 WFO fold の学習・評価を実行する.

    resume=True の場合、保存済みチェックポイントから再開する:
      - ac.pt が存在 → fold 全体をスキップ（バックテストのみ再実行）
      - bc_actor.pt が存在 → WM・BC をロードして AC から再開
      - world_model.pt が存在 → WM をロードして BC から再開

    Returns:
        {"fold": fold_idx, "metrics": BacktestMetrics, "positions": np.ndarray}
    """
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

    start_idx = _stage_idx(start_from)
    stop_idx = _stage_idx(stop_after)
    has_wm_ckpt = os.path.exists(wm_path)
    has_bc_ckpt = os.path.exists(bc_path)
    has_ac_ckpt = os.path.exists(ac_path)
    has_wm = has_wm_ckpt and (resume or start_idx > _stage_idx("wm"))
    has_bc = has_bc_ckpt and (resume or start_idx > _stage_idx("bc"))
    has_ac = has_ac_ckpt and (resume or start_idx > _stage_idx("ac"))

    if start_idx > _stage_idx("wm") and not has_wm_ckpt:
        raise FileNotFoundError(
            f"Fold {fold_idx}: missing WM checkpoint for --start-from {start_from}: {wm_path}"
        )
    if start_from == "ac" and not has_bc_ckpt:
        raise FileNotFoundError(
            f"Fold {fold_idx}: missing BC checkpoint for --start-from ac: {bc_path}"
        )
    if start_from == "test" and not (has_bc_ckpt or has_ac_ckpt):
        raise FileNotFoundError(
            f"Fold {fold_idx}: --start-from test requires {bc_path} or {ac_path}"
        )

    # --------- Step 1: Hindsight Oracle ---------
    print(f"\n[{_ts()}] [Step 1] Hindsight Oracle DP...")
    train_returns = wfo_dataset.train_returns
    oracle_cfg = cfg.get("oracle", {})
    reward_cfg = cfg.get("reward", {})
    oracle_action_values = np.asarray(
        oracle_cfg.get("action_values", cfg.get("actions", {}).get("values", _ACTIONS)),
        dtype=np.float32,
    )
    oracle_min_hold = oracle_cfg.get("min_hold", 0)
    oracle_soft_temp = oracle_cfg.get("soft_label_temp", 0.0)
    oracle_reward_mode = reward_cfg.get("mode", "absolute")
    oracle_benchmark_position = reward_cfg.get("benchmark_position", 1.0)
    oracle_teacher_mode = str(oracle_cfg.get("teacher_mode", "dp"))
    oracle_actions, oracle_values, oracle_soft_labels = hindsight_oracle_dp(
        train_returns,
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
        discount=oracle_cfg.get("discount", 1.0),
        min_hold=oracle_min_hold,
        soft_label_temp=oracle_soft_temp,
        reward_mode=oracle_reward_mode,
        benchmark_position=oracle_benchmark_position,
        action_values=oracle_action_values,
    )
    print(f"  Oracle computed: {len(oracle_actions)} steps, "
          f"mean value={oracle_values.mean():.4f}")
    print(f"  Oracle objective: {oracle_reward_mode} (benchmark={oracle_benchmark_position:+.2f})")
    _oracle_pos = oracle_action_values[oracle_actions]
    _oracle_s = _action_stats(_oracle_pos, benchmark_position=_benchmark_position_value(cfg))
    print(f"  Oracle dist: {_fmt_action_stats(_oracle_s)}")

    # Val oracle actions（分布比較・WM 学習に使用）
    val_oracle_actions, _, _ = hindsight_oracle_dp(
        wfo_dataset.val_returns,
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
        discount=oracle_cfg.get("discount", 1.0),
        min_hold=oracle_min_hold,
        reward_mode=oracle_reward_mode,
        benchmark_position=oracle_benchmark_position,
        action_values=oracle_action_values,
    )
    oracle_positions = oracle_action_values[oracle_actions].astype(np.float32)
    val_oracle_positions = oracle_action_values[val_oracle_actions].astype(np.float32)
    if oracle_teacher_mode == "signal_aim":
        abs_min = ac_cfg.get("abs_min_position", float(np.min(oracle_action_values)))
        abs_max = ac_cfg.get("abs_max_position", float(np.max(oracle_action_values)))
        teacher_horizons = tuple(oracle_cfg.get("signal_horizons", [4, 16, 64]))
        teacher_weights = tuple(oracle_cfg.get("signal_horizon_weights", [0.2, 0.3, 0.5]))
        oracle_positions, oracle_signal = hindsight_signal_teacher(
            train_returns,
            benchmark_position=oracle_benchmark_position,
            min_position=oracle_cfg.get("signal_floor_position", abs_min),
            max_position=oracle_cfg.get("signal_ceiling_position", abs_max),
            horizons=teacher_horizons,
            horizon_weights=teacher_weights,
            signal_scale=oracle_cfg.get("signal_scale", 1.5),
            signal_deadzone=oracle_cfg.get("signal_deadzone", 0.1),
            signal_clip=oracle_cfg.get("signal_clip", 4.0),
            downside_horizon=oracle_cfg.get("signal_downside_horizon", 16),
            downside_weight=oracle_cfg.get("signal_downside_weight", 0.0),
        )
        val_oracle_positions, val_oracle_signal = hindsight_signal_teacher(
            wfo_dataset.val_returns,
            benchmark_position=oracle_benchmark_position,
            min_position=oracle_cfg.get("signal_floor_position", abs_min),
            max_position=oracle_cfg.get("signal_ceiling_position", abs_max),
            horizons=teacher_horizons,
            horizon_weights=teacher_weights,
            signal_scale=oracle_cfg.get("signal_scale", 1.5),
            signal_deadzone=oracle_cfg.get("signal_deadzone", 0.1),
            signal_clip=oracle_cfg.get("signal_clip", 4.0),
            downside_horizon=oracle_cfg.get("signal_downside_horizon", 16),
            downside_weight=oracle_cfg.get("signal_downside_weight", 0.0),
        )
        oracle_values = oracle_signal.astype(np.float32)
        print(
            "  Signal teacher: "
            f"horizons={teacher_horizons} floor={oracle_cfg.get('signal_floor_position', abs_min):+.2f} "
            f"scale={oracle_cfg.get('signal_scale', 1.5):.2f}"
        )
    if oracle_cfg.get("use_aim_targets", False):
        abs_min = ac_cfg.get("abs_min_position", float(np.min(oracle_action_values)))
        abs_max = ac_cfg.get("abs_max_position", float(np.max(oracle_action_values)))
        oracle_positions = smooth_aim_positions(
            oracle_positions,
            max_step=oracle_cfg.get("aim_max_step", 0.25),
            band=oracle_cfg.get("aim_band", 0.0),
            initial_position=oracle_benchmark_position if oracle_reward_mode == "excess_bh" else 0.0,
            min_position=abs_min,
            max_position=abs_max,
            benchmark_position=oracle_benchmark_position if oracle_reward_mode == "excess_bh" else 0.0,
            underweight_confirm_bars=oracle_cfg.get("aim_underweight_confirm_bars", 0),
            underweight_min_scale=oracle_cfg.get("aim_underweight_min_scale", 0.0),
            underweight_floor_position=oracle_cfg.get("aim_underweight_floor_position"),
            underweight_step_scale=oracle_cfg.get("aim_underweight_step_scale", 1.0),
        ).astype(np.float32)
        val_oracle_positions = smooth_aim_positions(
            val_oracle_positions,
            max_step=oracle_cfg.get("aim_max_step", 0.25),
            band=oracle_cfg.get("aim_band", 0.0),
            initial_position=oracle_benchmark_position if oracle_reward_mode == "excess_bh" else 0.0,
            min_position=abs_min,
            max_position=abs_max,
            benchmark_position=oracle_benchmark_position if oracle_reward_mode == "excess_bh" else 0.0,
            underweight_confirm_bars=oracle_cfg.get("aim_underweight_confirm_bars", 0),
            underweight_min_scale=oracle_cfg.get("aim_underweight_min_scale", 0.0),
            underweight_floor_position=oracle_cfg.get("aim_underweight_floor_position"),
            underweight_step_scale=oracle_cfg.get("aim_underweight_step_scale", 1.0),
        ).astype(np.float32)
        _aim_s = _action_stats(oracle_positions, benchmark_position=_benchmark_position_value(cfg))
        print(f"  Oracle aim dist: {_fmt_action_stats(_aim_s)}")

    outcome_edge = None
    bc_quality_mode = str(bc_cfg.get("sample_quality_mode", "none")).lower()
    if oracle_teacher_mode == "signal_aim" and (
        bc_quality_mode in {"outcome_edge", "outcome_edge_relabel"} or bc_cfg.get("outcome_relabel_bad_to_benchmark", False)
    ):
        outcome_horizons = tuple(oracle_cfg.get("signal_horizons", [4, 16, 64]))
        outcome_weights = tuple(oracle_cfg.get("signal_horizon_weights", [0.2, 0.3, 0.5]))
        outcome_edge = _teacher_outcome_edge(
            train_returns,
            oracle_positions,
            benchmark_position=oracle_benchmark_position,
            horizons=outcome_horizons,
            horizon_weights=outcome_weights,
        )
        if bc_cfg.get("outcome_relabel_bad_to_benchmark", False):
            positive_edge = outcome_edge[outcome_edge > 0.0]
            relabel_quantile = float(np.clip(bc_cfg.get("outcome_relabel_quantile", 0.25), 0.0, 0.99))
            relabel_floor = float(np.quantile(positive_edge, relabel_quantile)) if positive_edge.size > 0 else 0.0
            bad_underweight = (oracle_positions < oracle_benchmark_position - 1e-6) & (outcome_edge <= relabel_floor)
            if bad_underweight.any():
                oracle_positions = oracle_positions.copy()
                oracle_positions[bad_underweight] = oracle_benchmark_position
                print(
                    "  Outcome relabel: "
                    f"{int(bad_underweight.sum())} weak underweight targets -> benchmark "
                    f"(q={relabel_quantile:.2f})"
                )

    # --------- HMM レジーム事後確率（Actor 入力用）---------
    # Actor 生成前に計算して regime_dim を確定する
    n_states = cfg.get("eval", {}).get("hmm_n_states", 3)
    hmm_det = None
    regime_dim = 0
    train_regime_probs = None
    val_regime_probs = None
    test_regime_probs = None
    try:
        from unidream.eval.regime import RegimeDetector
        hmm_det = RegimeDetector(n_states=n_states)
        hmm_det.fit(wfo_dataset.train_returns)  # 内部で平均リターン昇順にソート
        train_regime_probs = hmm_det.predict_proba(wfo_dataset.train_returns).astype(np.float32)
        val_regime_probs = hmm_det.predict_proba(wfo_dataset.val_returns).astype(np.float32)
        test_regime_probs = hmm_det.predict_proba(wfo_dataset.test_returns).astype(np.float32)
        regime_dim = n_states
        print(f"[Regime] HMM fitted, regime_dim={regime_dim}")
    except Exception as e:
        print(f"[Regime] HMM skipped: {e}")

    # --------- Step 2: 世界モデル学習 ---------
    ensemble = build_ensemble(obs_dim, cfg)
    wm_trainer = WorldModelTrainer(ensemble, cfg, device=device)

    if has_wm:
        print(f"\n[{_ts()}] [Step 2] World Model - loading checkpoint: {wm_path}")
        wm_trainer.load(wm_path)
    else:
        print(f"\n[{_ts()}] [Step 2] World Model Training...")
        train_ds_with_actions = SequenceDataset(
            wfo_dataset.train_features,
            seq_len=seq_len,
            actions=oracle_positions[:len(wfo_dataset.train_features)],
            returns=train_returns,
        )
        val_ds = SequenceDataset(
            wfo_dataset.val_features,
            seq_len=seq_len,
            actions=val_oracle_positions[:len(wfo_dataset.val_features)],
            returns=wfo_dataset.val_returns,
        )
        wm_trainer.train_on_dataset(
            train_ds_with_actions,
            val_dataset=val_ds,
            checkpoint_path=wm_path,
        )

    if stop_after == "wm":
        print(f"\n[{_ts()}] [Stop] Requested stop after WM")
        return {"fold": fold_idx, "completed_stage": "wm"}

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

    # --------- Step 3: BC 初期化 ---------
    actor = Actor(
        z_dim=ensemble.get_z_dim(),
        h_dim=ensemble.get_d_model(),
        act_dim=int(len(oracle_action_values)),
        hidden_dim=ac_cfg.get("actor_hidden", 256),
        n_layers=ac_cfg.get("ac_layers", 2),
        regime_dim=regime_dim,
        dropout_p=ac_cfg.get("actor_dropout", 0.0),
        inventory_dim=ac_cfg.get("controller_state_dim", 1),
    )
    actor.target_values = oracle_action_values.astype(np.float32)
    actor.benchmark_position = reward_cfg.get("benchmark_position", 1.0)
    actor.baseline_target_index = int(np.argmin(np.abs(oracle_action_values - reward_cfg.get("benchmark_position", 1.0))))
    actor.abs_min_position = ac_cfg.get("abs_min_position", -1.0)
    actor.abs_max_position = ac_cfg.get("abs_max_position", 1.0)
    actor.infer_temperature = ac_cfg.get("infer_temperature", 1.0)
    actor.infer_gap_boost = ac_cfg.get("infer_gap_boost", 0.0)
    actor.infer_adjust_rate_scale = ac_cfg.get("infer_adjust_rate_scale", 1.0)
    actor.adjustment_temperature = ac_cfg.get("adjustment_temperature", 0.25)
    actor.max_position_step = ac_cfg.get("max_position_step", 10.0)
    actor.min_band = ac_cfg.get("min_band", 0.02)
    actor.max_band = ac_cfg.get("max_band", 0.20)
    actor.min_target_std = ac_cfg.get("min_target_std", 0.05)
    actor.max_target_std = ac_cfg.get("max_target_std", 0.35)
    actor.hold_state_scale = ac_cfg.get("hold_state_scale", 64.0)
    actor.trade_state_eps = ac_cfg.get("trade_state_eps", 1e-6)
    actor.infer_quantize_step = ac_cfg.get("infer_quantize_step", 0.0)
    actor.use_residual_controller = bool(ac_cfg.get("residual_controller", False))
    actor.residual_min_overlay = ac_cfg.get(
        "residual_min_overlay",
        ac_cfg.get("abs_min_position", -1.0) - reward_cfg.get("benchmark_position", 1.0),
    )
    actor.residual_max_overlay = ac_cfg.get("residual_max_overlay", 0.0)
    actor.regime_overlay_caps = ac_cfg.get("regime_overlay_caps")
    actor.infer_bootstrap_target_prob = ac_cfg.get("infer_bootstrap_target_prob", 0.0)
    actor.infer_bootstrap_target_std = ac_cfg.get("infer_bootstrap_target_std", 0.0)
    actor.infer_bootstrap_trade_signal = ac_cfg.get("infer_bootstrap_trade_signal", 0.0)
    actor.infer_bootstrap_baseline_margin = ac_cfg.get("infer_bootstrap_baseline_margin", 0.0)
    actor.infer_underweight_adjust_scale = ac_cfg.get("infer_underweight_adjust_scale", 1.0)
    actor.infer_support_min_count = ac_cfg.get("infer_support_min_count", 0.0)
    actor.infer_support_min_ratio = ac_cfg.get("infer_support_min_ratio", 0.0)
    actor.infer_min_trade_floor = ac_cfg.get("infer_min_trade_floor", 0.0)
    actor.infer_min_trade_gap = ac_cfg.get("infer_min_trade_gap", 0.0)
    actor.infer_min_trade_scale = ac_cfg.get("infer_min_trade_scale", 0.0)
    actor.support_transition_counts = None
    if actor.use_residual_controller:
        residual_min = float(actor.residual_min_overlay)
        residual_max = float(actor.residual_max_overlay)
        init_overlay = float(ac_cfg.get("residual_init_overlay", 0.0))
        if residual_max > residual_min + 1e-6:
            init_frac = np.clip((init_overlay - residual_min) / (residual_max - residual_min), 1e-4, 1.0 - 1e-4)
            residual_bias = float(np.log(init_frac / (1.0 - init_frac)))
            torch.nn.init.constant_(actor.residual_head.bias, residual_bias)
            torch.nn.init.zeros_(actor.residual_head.weight)
    try:
        current_abs_positions = np.empty_like(oracle_positions)
        current_abs_positions[0] = reward_cfg.get("benchmark_position", 1.0)
        if len(oracle_positions) > 1:
            current_abs_positions[1:] = oracle_positions[:-1]
        current_idx = actor.target_indices(torch.tensor(current_abs_positions, dtype=torch.float32)).cpu().numpy()
        next_idx = actor.target_indices(torch.tensor(oracle_positions, dtype=torch.float32)).cpu().numpy()
        if train_regime_probs is not None:
            regime_idx = np.argmax(train_regime_probs[:len(next_idx)], axis=1).astype(np.int64)
            n_regimes = train_regime_probs.shape[1]
        else:
            regime_idx = np.zeros(len(next_idx), dtype=np.int64)
            n_regimes = 1
        support_counts = np.zeros((n_regimes, len(oracle_action_values), len(oracle_action_values)), dtype=np.float32)
        np.add.at(support_counts, (regime_idx, current_idx, next_idx), 1.0)
        actor.support_transition_counts = support_counts
    except Exception as e:
        print(f"[SPIBB] support table skipped: {e}")

    bc_sample_quality = None
    bc_quality_mode = str(bc_cfg.get("sample_quality_mode", "none")).lower()
    if oracle_teacher_mode == "signal_aim" and bc_quality_mode != "none":
        signal_values = np.asarray(oracle_values, dtype=np.float32)
        if bc_quality_mode == "abs_signal":
            bc_sample_quality = np.abs(signal_values)
        elif bc_quality_mode == "underweight_edge":
            benchmark_position = float(reward_cfg.get("benchmark_position", 1.0))
            underweight_size = np.clip(benchmark_position - np.asarray(oracle_positions, dtype=np.float32), 0.0, None)
            negative_signal = np.clip(-signal_values, 0.0, None)
            raw_edge = underweight_size * negative_signal
            positive_edge = raw_edge[raw_edge > 0.0]
            if positive_edge.size > 0:
                edge_quantile = float(np.clip(bc_cfg.get("sample_quality_quantile", 0.75), 0.0, 0.99))
                edge_floor = float(np.quantile(positive_edge, edge_quantile))
                edge_scale = float(np.quantile(positive_edge, 0.90)) - edge_floor
                edge_scale = max(edge_scale, 1e-6)
                bc_sample_quality = np.clip((raw_edge - edge_floor) / edge_scale, 0.0, bc_cfg.get("sample_quality_clip", 4.0))
            else:
                bc_sample_quality = np.zeros_like(raw_edge, dtype=np.float32)
        elif bc_quality_mode in {"outcome_edge", "outcome_edge_relabel"}:
            raw_edge = outcome_edge if outcome_edge is not None else np.zeros_like(signal_values, dtype=np.float32)
            positive_edge = raw_edge[raw_edge > 0.0]
            if positive_edge.size > 0:
                edge_quantile = float(np.clip(bc_cfg.get("sample_quality_quantile", 0.75), 0.0, 0.99))
                edge_floor = float(np.quantile(positive_edge, edge_quantile))
                edge_scale = float(np.quantile(positive_edge, 0.90)) - edge_floor
                edge_scale = max(edge_scale, 1e-6)
                bc_sample_quality = np.clip((raw_edge - edge_floor) / edge_scale, 0.0, bc_cfg.get("sample_quality_clip", 4.0))
            else:
                bc_sample_quality = np.zeros_like(raw_edge, dtype=np.float32)

    if has_bc:
        print(f"\n[{_ts()}] [Step 3] BC - loading checkpoint: {bc_path}")
        bc_trainer = BCPretrainer(
            actor=actor,
            z_dim=ensemble.get_z_dim(),
            h_dim=ensemble.get_d_model(),
            target_aux_coef=bc_cfg.get("target_aux_coef", 1.0),
            trade_aux_coef=bc_cfg.get("trade_aux_coef", 0.5),
            band_aux_coef=bc_cfg.get("band_aux_coef", 0.25),
            execution_aux_coef=bc_cfg.get("execution_aux_coef", 0.0),
            path_aux_coef=bc_cfg.get("path_aux_coef", 0.0),
            path_horizon=bc_cfg.get("path_horizon", 1),
            path_position_coef=bc_cfg.get("path_position_coef", 1.0),
            path_turnover_coef=bc_cfg.get("path_turnover_coef", 0.0),
            path_shortfall_coef=bc_cfg.get("path_shortfall_coef", 0.0),
            soft_trade_targets=bc_cfg.get("soft_trade_targets", True),
            trade_target_scale=bc_cfg.get("trade_target_scale"),
            self_condition_prob=bc_cfg.get("self_condition_prob", 0.0),
            self_condition_interval=bc_cfg.get("self_condition_interval", 1),
            self_condition_warmup_epochs=bc_cfg.get("self_condition_warmup_epochs", 0),
            self_condition_mode=bc_cfg.get("self_condition_mode", "mix"),
            self_condition_blend=bc_cfg.get("self_condition_blend", 0.0),
            self_condition_max_position_gap=bc_cfg.get("self_condition_max_position_gap"),
            self_condition_max_underweight_gap=bc_cfg.get("self_condition_max_underweight_gap"),
            self_condition_relabel_step=bc_cfg.get("self_condition_relabel_step"),
            self_condition_relabel_band=bc_cfg.get("self_condition_relabel_band", 0.0),
            relabel_aim_max_step=oracle_cfg.get("aim_max_step", 0.125),
            relabel_aim_band=oracle_cfg.get("aim_band", 0.0),
            relabel_min_position=ac_cfg.get("abs_min_position", -1.0),
            relabel_max_position=ac_cfg.get("abs_max_position", 1.0),
            relabel_benchmark_position=reward_cfg.get("benchmark_position", 0.0),
            relabel_underweight_confirm_bars=oracle_cfg.get("aim_underweight_confirm_bars", 0),
            relabel_underweight_min_scale=oracle_cfg.get("aim_underweight_min_scale", 0.0),
            relabel_underweight_floor_position=oracle_cfg.get("aim_underweight_floor_position"),
            relabel_underweight_step_scale=oracle_cfg.get("aim_underweight_step_scale", 1.0),
            residual_target_coef=bc_cfg.get("residual_target_coef", 1.0),
            residual_aux_ce_coef=bc_cfg.get("residual_aux_ce_coef", 0.0),
            sample_quality_coef=bc_cfg.get("sample_quality_coef", 0.0),
            sample_quality_clip=bc_cfg.get("sample_quality_clip", 4.0),
            device=device,
        )
        bc_trainer.load(bc_path)
    else:
        if start_idx <= _stage_idx("bc"):
            print(f"\n[{_ts()}] [Step 3] BC Pre-training...")
            bc_trainer = BCPretrainer(
                actor=actor,
                z_dim=ensemble.get_z_dim(),
                h_dim=ensemble.get_d_model(),
                lr=bc_cfg.get("lr", 3e-4),
                batch_size=bc_cfg.get("batch_size", 256),
                n_epochs=bc_cfg.get("n_epochs", 5),
                sirl_hidden=bc_cfg.get("sirl_hidden", 128),
                label_smoothing=bc_cfg.get("label_smoothing", 0.0),
                entropy_coef=bc_cfg.get("entropy_coef", 0.0),
                chunk_size=bc_cfg.get("chunk_size", 1),
                class_balanced=bc_cfg.get("class_balanced", False),
                target_aux_coef=bc_cfg.get("target_aux_coef", 1.0),
                trade_aux_coef=bc_cfg.get("trade_aux_coef", 0.5),
                band_aux_coef=bc_cfg.get("band_aux_coef", 0.25),
                execution_aux_coef=bc_cfg.get("execution_aux_coef", 0.0),
                path_aux_coef=bc_cfg.get("path_aux_coef", 0.0),
                path_horizon=bc_cfg.get("path_horizon", 1),
                path_position_coef=bc_cfg.get("path_position_coef", 1.0),
                path_turnover_coef=bc_cfg.get("path_turnover_coef", 0.0),
                path_shortfall_coef=bc_cfg.get("path_shortfall_coef", 0.0),
                soft_trade_targets=bc_cfg.get("soft_trade_targets", True),
                trade_target_scale=bc_cfg.get("trade_target_scale"),
                self_condition_prob=bc_cfg.get("self_condition_prob", 0.0),
                self_condition_interval=bc_cfg.get("self_condition_interval", 1),
                self_condition_warmup_epochs=bc_cfg.get("self_condition_warmup_epochs", 0),
                self_condition_mode=bc_cfg.get("self_condition_mode", "mix"),
                self_condition_blend=bc_cfg.get("self_condition_blend", 0.0),
                self_condition_max_position_gap=bc_cfg.get("self_condition_max_position_gap"),
                self_condition_max_underweight_gap=bc_cfg.get("self_condition_max_underweight_gap"),
                self_condition_relabel_step=bc_cfg.get("self_condition_relabel_step"),
                self_condition_relabel_band=bc_cfg.get("self_condition_relabel_band", 0.0),
                relabel_aim_max_step=oracle_cfg.get("aim_max_step", 0.125),
                relabel_aim_band=oracle_cfg.get("aim_band", 0.0),
                relabel_min_position=ac_cfg.get("abs_min_position", -1.0),
                relabel_max_position=ac_cfg.get("abs_max_position", 1.0),
                relabel_benchmark_position=reward_cfg.get("benchmark_position", 0.0),
                relabel_underweight_confirm_bars=oracle_cfg.get("aim_underweight_confirm_bars", 0),
                relabel_underweight_min_scale=oracle_cfg.get("aim_underweight_min_scale", 0.0),
                relabel_underweight_floor_position=oracle_cfg.get("aim_underweight_floor_position"),
                relabel_underweight_step_scale=oracle_cfg.get("aim_underweight_step_scale", 1.0),
                residual_target_coef=bc_cfg.get("residual_target_coef", 1.0),
                residual_aux_ce_coef=bc_cfg.get("residual_aux_ce_coef", 0.0),
                sample_quality_coef=bc_cfg.get("sample_quality_coef", 0.0),
                sample_quality_clip=bc_cfg.get("sample_quality_clip", 4.0),
                device=device,
            )
            T_enc = min(len(z_train), len(oracle_positions))
            bc_trainer.train(
                z=z_train[:T_enc],
                h=h_train[:T_enc],
                oracle_positions=oracle_positions[:T_enc],
                regime_probs=train_regime_probs[:T_enc] if train_regime_probs is not None else None,
                soft_labels=oracle_soft_labels[:T_enc] if oracle_soft_labels is not None else None,
                sample_quality=bc_sample_quality[:T_enc] if bc_sample_quality is not None else None,
            )
            bc_trainer.save(bc_path)
        else:
            print(f"\n[{_ts()}] [Step 3] BC - skipped (AC checkpoint will provide actor weights)")

    if stop_after == "bc":
        print(f"\n[{_ts()}] [Stop] Requested stop after BC")
        return {"fold": fold_idx, "completed_stage": "bc"}

    # --------- Step 4: Imagination AC Fine-tune ---------
    # no-action h（z_train/h_train）を AC でもそのまま使う。
    # BC-action re-encode は廃止: test/val も no-action h のため、
    # re-encode すると AC 開始状態が no-action h から外れ、
    # BC 正則化（_oracle_z/_oracle_h = no-action h）と AC gradient が
    # 異なる h 分布上で計算されて矛盾した gradient が生じる。
    # no-action h に統一することで BC train/AC train/val/test が一貫する。

    ac_requested = stop_idx >= _stage_idx("ac") or has_ac
    if ac_requested:
        critic = Critic(
            z_dim=ensemble.get_z_dim(),
            h_dim=ensemble.get_d_model(),
            hidden_dim=ac_cfg.get("critic_hidden", 256),
            n_layers=ac_cfg.get("ac_layers", 2),
            n_bins=wm_cfg.get("n_bins", 255),
            ema_decay=ac_cfg.get("ema_decay", 0.98),
        )
        ac_trainer = ImagACTrainer(
            actor=actor,
            critic=critic,
            ensemble=ensemble,
            cfg=cfg,
            device=device,
        )
        if has_ac:
            ac_trainer.load(ac_path)

        if start_idx <= _stage_idx("ac") or has_ac:
            # oracle データは resume 時も必要（BC 損失計算用）
            # z/h は oracle エンコード（z は obs のみなので同一、h は BC エンコードとは別途保持）
            T_enc = min(len(z_train), len(oracle_positions))
            ac_trainer.set_oracle_data(
                z=z_train[:T_enc],
                h=h_train[:T_enc],
                oracle_positions=oracle_positions[:T_enc],
                regime_probs=train_regime_probs[:T_enc] if train_regime_probs is not None else None,
            )

            encoded_list = [{
                "z": z_train,
                "h": h_train,
                "regime": train_regime_probs if train_regime_probs is not None else None,
            }]

            # Val encoding を no-action（flat）で一度だけ固定する。
            # test と同じ single-pass encoding に統一することで val/test の比較を整合させる。
            # 自己参照ループ（AC の自身の予測を context に使う → val Sharpe 過大）は
            # ここで固定した z_val_fixed/h_val_fixed を AC 学習中ずっと使い回すことで回避する。
            val_features_arr = wfo_dataset.val_features
            val_returns_arr = wfo_dataset.val_returns
            if len(val_features_arr) > 0:
                _enc_val_fixed = wm_trainer.encode_sequence(val_features_arr, seq_len=seq_len)
                z_val_fixed = _enc_val_fixed["z"]
                h_val_fixed = _enc_val_fixed["h"]
            else:
                z_val_fixed = h_val_fixed = None

            # Val backtest function - used for AC checkpoint selection
            def _val_eval() -> tuple[float, str]:
                if z_val_fixed is None:
                    return -float("inf"), "raw=-inf score=-inf"
                pos = actor.predict_positions(
                    z_val_fixed, h_val_fixed, regime_np=val_regime_probs, device=device
                )
                T_min = min(len(val_returns_arr), len(pos))
                metrics = Backtest(
                    val_returns_arr[:T_min], pos[:T_min],
                    spread_bps=costs_cfg.get("spread_bps", 5.0),
                    fee_rate=costs_cfg.get("fee_rate", 0.0004),
                    slippage_bps=costs_cfg.get("slippage_bps", 2.0),
                    interval=cfg.get("data", {}).get("interval", "15m"),
                    benchmark_positions=_benchmark_positions(T_min, cfg),
                ).run()
                stats = _action_stats(pos[:T_min], benchmark_position=_benchmark_position_value(cfg))
                return _policy_score(
                    metrics,
                    stats,
                    benchmark_position=_benchmark_position_value(cfg),
                )

            ac_max_steps = ac_cfg.get("max_steps", 200_000)
            if ac_trainer.global_step >= ac_max_steps:
                print(f"\n[{_ts()}] [Step 4] AC - already complete (step={ac_trainer.global_step})")
            else:
                bc_val_sharpe = -float("inf")
                if has_ac:
                    print(f"\n[{_ts()}] [Step 4] AC - resuming from step {ac_trainer.global_step}/{ac_max_steps}")
                else:
                    print(f"\n[{_ts()}] [Step 4] Imagination AC Fine-tuning...")

                    # BC-only val score (checkpoint selection のベースライン)
                    bc_val_sharpe, bc_val_label = _val_eval()
                    print(f"[AC] BC-only val score: {bc_val_label}")
                    if z_val_fixed is not None:
                        _bc_pos = actor.predict_positions(
                            z_val_fixed, h_val_fixed, regime_np=val_regime_probs, device=device
                        )
                        _bc_T = min(len(val_returns_arr), len(_bc_pos))
                        _bc_m = Backtest(
                            val_returns_arr[:_bc_T], _bc_pos[:_bc_T],
                            spread_bps=costs_cfg.get("spread_bps", 5.0),
                            fee_rate=costs_cfg.get("fee_rate", 0.0004),
                            slippage_bps=costs_cfg.get("slippage_bps", 2.0),
                            interval=cfg.get("data", {}).get("interval", "15m"),
                            benchmark_positions=_benchmark_positions(_bc_T, cfg),
                        ).run()
                        _bc_attr = pnl_attribution(
                            val_returns_arr[:_bc_T], _bc_pos[:_bc_T],
                            spread_bps=costs_cfg.get("spread_bps", 5.0),
                            fee_rate=costs_cfg.get("fee_rate", 0.0004),
                            slippage_bps=costs_cfg.get("slippage_bps", 2.0),
                        )
                        _bc_s = _action_stats(_bc_pos[:_bc_T], benchmark_position=_benchmark_position_value(cfg))
                        print(f"  BC val dist: {_fmt_action_stats(_bc_s)}")
                        print(f"  BC val: TotalRet={_bc_m.total_return:.3f}  "
                              f"AlphaExcess={100.0 * (_bc_m.alpha_excess or 0.0):+.2f}pt  "
                              f"long={_bc_attr['long_gross']:+.4f}  "
                              f"short={_bc_attr['short_gross']:+.4f}  "
                              f"cost={_bc_attr['cost_total']:.4f}")
                        _oracle_val_pos = val_oracle_positions[:_bc_T]
                        _oracle_val_s = _action_stats(_oracle_val_pos, benchmark_position=_benchmark_position_value(cfg))
                        print(f"  Oracle val dist: {_fmt_action_stats(_oracle_val_s)}")
                        _ac_alerts("BC-val", _bc_s)

                    critic_pretrain_steps = ac_cfg.get("critic_pretrain_steps", 0)
                    if critic_pretrain_steps > 0:
                        ac_trainer.pretrain_critic(
                            encoded_sequences=encoded_list,
                            n_steps=critic_pretrain_steps,
                            batch_size=ac_cfg.get("batch_size", 32),
                        )

                interval = cfg.get("data", {}).get("interval", "15m")
                bars_per_day = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}.get(interval, 96)
                online_wm_window = ac_cfg.get("online_wm_window_days", 30) * bars_per_day
                online_wm_steps_val = ac_cfg.get("online_wm_steps", 0)

                def _online_wm_cb(step: int) -> None:
                    if online_wm_steps_val <= 0:
                        return
                    T_train = len(wfo_dataset.train_features)
                    window_start = max(0, T_train - online_wm_window)
                    recent_feat = wfo_dataset.train_features[window_start:]
                    recent_returns = wfo_dataset.train_returns[window_start:]
                    recent_regime = (
                        train_regime_probs[window_start:T_train]
                        if train_regime_probs is not None else None
                    )
                    enc_recent = wm_trainer.encode_sequence(recent_feat, seq_len=seq_len)
                    recent_pos = actor.predict_positions(
                        enc_recent["z"], enc_recent["h"],
                        regime_np=recent_regime, device=device,
                    )
                    recent_ds = SequenceDataset(
                        recent_feat, seq_len=seq_len,
                        actions=recent_pos[:len(recent_feat)],
                        returns=recent_returns[:len(recent_feat)],
                    )
                    if len(recent_ds) < 2:
                        return
                    wm_trainer.ensemble.train()
                    wm_trainer.train_on_dataset(recent_ds, max_steps=online_wm_steps_val, checkpoint_path=None)
                    wm_trainer.ensemble.eval()

                online_wm_callback = _online_wm_cb if online_wm_steps_val > 0 else None

                ac_trainer.train(
                    encoded_sequences=encoded_list,
                    batch_size=ac_cfg.get("batch_size", 32),
                    checkpoint_path=ac_path,
                    val_eval_fn=_val_eval,
                    val_baseline_sharpe=bc_val_sharpe,
                    online_wm_callback=online_wm_callback,
                )
                ac_trainer.save(ac_path)
        else:
            print(f"\n[{_ts()}] [Step 4] AC - skipped (BC actor only for test)")

    if stop_after == "ac":
        print(f"\n[{_ts()}] [Stop] Requested stop after AC")
        return {"fold": fold_idx, "completed_stage": "ac"}

    adjust_scale_grid = ac_cfg.get("val_adjust_rate_scale_grid", [])
    if len(adjust_scale_grid) > 0:
        val_features_arr = wfo_dataset.val_features
        val_returns_arr = wfo_dataset.val_returns
        if len(val_features_arr) > 0:
            enc_val = wm_trainer.encode_sequence(val_features_arr, seq_len=seq_len)
            best_scale = float(getattr(actor, "infer_adjust_rate_scale", 1.0))
            best_label = "score=-inf"
            original_scale = best_scale
            selector_cfg = _selector_cfg(ac_cfg)
            selector_candidates = []
            for candidate in adjust_scale_grid:
                actor.infer_adjust_rate_scale = float(candidate)
                pos = actor.predict_positions(
                    enc_val["z"], enc_val["h"], regime_np=val_regime_probs, device=device
                )
                T_min = min(len(val_returns_arr), len(pos))
                metrics = Backtest(
                    val_returns_arr[:T_min], pos[:T_min],
                    spread_bps=costs_cfg.get("spread_bps", 5.0),
                    fee_rate=costs_cfg.get("fee_rate", 0.0004),
                    slippage_bps=costs_cfg.get("slippage_bps", 2.0),
                    interval=cfg.get("data", {}).get("interval", "15m"),
                    benchmark_positions=_benchmark_positions(T_min, cfg),
                ).run()
                stats = _action_stats(pos[:T_min], benchmark_position=_benchmark_position_value(cfg))
                candidate_rec = _selector_candidate(
                    float(candidate),
                    metrics,
                    stats,
                    benchmark_position=_benchmark_position_value(cfg),
                    selector_cfg=selector_cfg,
                    cfg=cfg,
                )
                selector_candidates.append(candidate_rec)
                print(f"  [ValAdj] scale={float(candidate):.3f} {candidate_rec['label']}")
            chosen = _select_policy_candidate(selector_candidates, selector_cfg)
            best_scale = float(chosen["candidate"])
            best_label = chosen["label"]
            actor.infer_adjust_rate_scale = best_scale
            print(
                f"  [ValAdj] selected scale={best_scale:.3f} "
                f"(default={original_scale:.3f}) {best_label}"
            )

    # --------- Step 5: Test バックテスト ---------
    print(f"\n[{_ts()}] [Step 5] Test Backtest...")
    test_features = wfo_dataset.test_dataset().features.numpy()
    test_returns = wfo_dataset.test_returns

    # Single-pass encoding: no-action context（flat）で encode して actor を適用する。
    # 2-pass encoding はノイジーな BC 予測を context に混入させ switching を増幅するため、
    # val と同じ no-action single-pass に統一して安定性を確保する。
    enc_test = wm_trainer.encode_sequence(test_features, seq_len=seq_len)
    positions = actor.predict_positions(
        enc_test["z"], enc_test["h"], regime_np=test_regime_probs, device=device
    )

    T_min = min(len(test_returns), len(positions))
    bt = Backtest(
        test_returns[:T_min],
        positions[:T_min],
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
        interval=cfg.get("data", {}).get("interval", "15m"),
        benchmark_positions=_benchmark_positions(T_min, cfg),
    )
    metrics = bt.run()

    _test_attr = pnl_attribution(
        test_returns[:T_min], positions[:T_min],
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
    )
    _test_s = _action_stats(positions[:T_min], benchmark_position=_benchmark_position_value(cfg))
    _test_scorecard = _m2_scorecard(metrics, _test_s, cfg)
    print(f"  Sharpe:   {metrics.sharpe:.3f}")
    print(f"  Sortino:  {metrics.sortino:.3f}")
    print(f"  MaxDD:    {metrics.max_drawdown:.3f}")
    print(f"  Calmar:   {metrics.calmar:.3f}")
    print(f"  TotalRet: {metrics.total_return:.4f}")
    if metrics.alpha_excess is not None:
        print(f"  AlphaEx:  {100.0 * metrics.alpha_excess:+.2f} pt/yr")
        print(f"  SharpeΔ:  {(metrics.sharpe_delta or 0.0):+.3f}")
    if metrics.alpha_excess is not None:
        print(f"  MaxDDΔ:   {100.0 * (metrics.maxdd_delta or 0.0):+.2f} pt")
        print(f"  WinRate:  {(metrics.win_rate_vs_bh or 0.0):.1%}")
        print(f"  Score:    {_format_m2_scorecard(_test_scorecard)}")
    print(f"  PnL attr: long={_test_attr['long_gross']:+.4f}  "
          f"short={_test_attr['short_gross']:+.4f}  "
          f"cost={_test_attr['cost_total']:.4f}  "
          f"net={_test_attr['net_total']:+.4f}")
    print(f"  Test dist: {_fmt_action_stats(_test_s)}")
    _ac_alerts("test", _test_s)

    return {
        "fold": fold_idx,
        "metrics": metrics,
        "scorecard": _test_scorecard,
        "positions": positions[:T_min],
        "test_returns": test_returns[:T_min],
        "completed_stage": "test",
    }


def main():
    parser = argparse.ArgumentParser(description="UniDream Training Script")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--symbol", default=None, help="Binance symbol (overrides config)")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="保存済みチェックポイントから再開する")
    parser.add_argument(
        "--start-from",
        default="wm",
        choices=_PIPELINE_STAGES,
        help="Start pipeline from this stage",
    )
    parser.add_argument(
        "--stop-after",
        default="test",
        choices=_PIPELINE_STAGES,
        help="Stop pipeline after this stage",
    )
    parser.add_argument(
        "--cost-profile",
        default=None,
        help="Named cost profile from config cost_profiles (for example: base, stress)",
    )
    parser.add_argument(
        "--folds",
        default=None,
        help="Comma-separated fold indices to run (for example: 0,1,4)",
    )
    args = parser.parse_args()

    if _stage_idx(args.start_from) > _stage_idx(args.stop_after):
        parser.error("--start-from must be earlier than or equal to --stop-after")

    cfg = load_config(args.config)
    cfg, active_cost_profile = resolve_costs(cfg, args.cost_profile)
    set_seed(args.seed)

    symbol = args.symbol or cfg.get("data", {}).get("symbol", "BTCUSDT")
    interval = cfg.get("data", {}).get("interval", "15m")

    print(f"UniDream Training | {symbol} {interval} | {args.start} → {args.end}")
    print(f"Device: {args.device} | Seed: {args.seed} | Resume: {args.resume}")
    print(f"Stages: {args.start_from} -> {args.stop_after}")
    costs_cfg = cfg.get("costs", {})
    total_cost_bps = (
        costs_cfg.get("spread_bps", 0.0) / 2
        + costs_cfg.get("fee_rate", 0.0) * 10000
        + costs_cfg.get("slippage_bps", 0.0)
    )
    print(
        "Costs: "
        f"profile={active_cost_profile} | "
        f"spread={costs_cfg.get('spread_bps', 0.0):.2f}bps "
        f"fee={costs_cfg.get('fee_rate', 0.0) * 10000:.2f}bps "
        f"slip={costs_cfg.get('slippage_bps', 0.0):.2f}bps "
        f"=> one-way Δpos=1 cost={total_cost_bps:.2f}bps"
    )

    # --------- データ取得・特徴量計算（キャッシュ対応）---------
    cache_dir = os.path.join(args.checkpoint_dir, "data_cache")
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}_v2"
    features_cache = os.path.join(cache_dir, f"{cache_tag}_features.parquet")
    returns_cache = os.path.join(cache_dir, f"{cache_tag}_returns.parquet")

    if _cache_is_fresh(features_cache) and _cache_is_fresh(returns_cache):
        print("\n[Data] Loading cached features...")
        features_df = pd.read_parquet(features_cache)
        raw_returns = pd.read_parquet(returns_cache).squeeze()
        print(f"  Cached: {features_df.shape} | obs_dim={features_df.shape[1]}")
    else:
        print("\n[Data] Fetching OHLCV...")
        df = fetch_binance_ohlcv(symbol, interval, args.start, args.end)
        print(f"  Raw data: {len(df)} bars ({df.index[0]} → {df.index[-1]})")

        # Futures 追加データ（funding rate, OI）
        funding_df = None
        oi_df = None
        try:
            print("[Data] Fetching funding rate...")
            funding_df = fetch_funding_rate(symbol, args.start, args.end)
            print(f"  Funding rate: {len(funding_df)} records")
        except Exception as e:
            print(f"  Funding rate skipped: {e}")
        try:
            print("[Data] Fetching open interest...")
            oi_df = fetch_open_interest_hist(symbol, interval, args.start, args.end)
            print(f"  Open interest: {len(oi_df)} records")
        except Exception as e:
            print(f"  Open interest skipped: {e}")

        print("[Data] Computing features...")
        features_df = compute_features(
            df,
            zscore_window_days=cfg.get("normalization", {}).get("zscore_window_days", 60),
            interval=interval,
            funding_df=funding_df,
            oi_df=oi_df,
        )
        raw_returns = get_raw_returns(df)
        common_idx = features_df.index.intersection(raw_returns.index)
        features_df = features_df.loc[common_idx]
        raw_returns = raw_returns.loc[common_idx]
        print(f"  Features: {features_df.shape} | obs_dim={features_df.shape[1]}")

        # キャッシュ保存
        os.makedirs(cache_dir, exist_ok=True)
        features_df.to_parquet(features_cache)
        raw_returns.to_frame().to_parquet(returns_cache)
        print(f"  Cached to {cache_dir}")

    # --------- WFO 分割 ---------
    feature_extras_cfg = cfg.get("feature_extras", {})
    if feature_extras_cfg.get("rebound_v1", False):
        features_df = augment_with_rebound_features(
            features_df,
            raw_returns,
            zscore_window_days=cfg.get("normalization", {}).get("zscore_window_days", 60),
            interval=interval,
            windows_hours=feature_extras_cfg.get("rebound_windows_hours", [24, 72]),
        )
        raw_returns = raw_returns.loc[features_df.index]
        print(f"[Data] Rebound features added -> {features_df.shape}")

    print("[Data] WFO splits...")
    data_cfg = cfg.get("data", {})
    splits = get_wfo_splits(
        features_df,
        train_years=data_cfg.get("train_years", 2),
        val_months=data_cfg.get("val_months", 3),
        test_months=data_cfg.get("test_months", 3),
    )
    print(f"  {len(splits)} folds")

    if len(splits) == 0:
        print("ERROR: WFO splits が空です。データ期間が短すぎます。")
        return

    if args.folds:
        selected_folds = sorted(
            {
                int(token.strip())
                for token in args.folds.split(",")
                if token.strip()
            }
        )
        if not selected_folds:
            parser.error("--folds must contain at least one fold index")
        splits = [split for split in splits if split.fold_idx in selected_folds]
        if len(splits) == 0:
            parser.error(
                f"--folds selected {selected_folds}, but no matching folds were found in this dataset"
            )
        print(f"  Running selected folds only: {selected_folds}")

    # --------- 各 Fold の学習・評価 ---------
    fold_results = {}
    for split in splits:
        wfo_ds = WFODataset(
            features_df,
            raw_returns,
            split,
            seq_len=data_cfg.get("seq_len", 64),
        )
        result = run_fold(
            fold_idx=split.fold_idx,
            wfo_dataset=wfo_ds,
            cfg=cfg,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            start_from=args.start_from,
            stop_after=args.stop_after,
        )
        fold_results[split.fold_idx] = result

    if args.stop_after != "test":
        print("\n" + "="*60)
        print("Stage Summary")
        print("="*60)
        for fold_idx, r in fold_results.items():
            print(f"  Fold {fold_idx}: completed_stage={r.get('completed_stage', args.stop_after)}")
        return

    # --------- PBO / Deflated Sharpe ---------
    # 注意: PBO は「fold を戦略候補扱いした IS/OOS 分割の簡略版」。
    #       標準 CSCV-PBO（Bailey & Lopez de Prado 2014）ではない。
    #       複数モデル構成を比較する際は CSCV 版に差し替えること。
    # 注意: DSR は n_trials=1（ハイパーパラメータ探索なし）のため、
    #       多重比較補正なしの「best fold Sharpe の t 統計量」として機能する。
    print("\n[Eval] Overfitting Diagnostics (simplified)...")
    pnl_list = [r["metrics"].pnl_series for r in fold_results.values()]
    eval_cfg = cfg.get("eval", {})
    pbo = compute_pbo(
        pnl_list,
        n_combinations=eval_cfg.get("pbo_n_trials"),
    )
    print(f"  PBO (simplified): {pbo:.4f} (< 0.5 が望ましい; fold IS/OOS split 版)")

    all_sharpes = [r["metrics"].sharpe for r in fold_results.values()]
    best_sharpe = max(all_sharpes)
    # n_trials=1: ハイパーパラメータ探索を行っていないため多重比較補正なし
    n_trials = 1
    T_avg = int(np.mean([len(r["metrics"].pnl_series) for r in fold_results.values()]))
    dsr = deflated_sharpe(best_sharpe, n_trials=n_trials, T=T_avg)
    dsr_str = f"{dsr:.4f}" if np.isfinite(dsr) else f"N/A ({dsr}, fold 数不足)"
    print(f"  Sharpe t-stat (DSR, n_trials=1): {dsr_str} (> 0 が望ましい)")

    # --------- レジーム別メトリクス ---------
    print("\n[Eval] Regime Analysis...")
    all_test_returns = np.concatenate([r["test_returns"] for r in fold_results.values()])
    all_positions = np.concatenate([r["positions"] for r in fold_results.values()])

    try:
        detector = RegimeDetector(n_states=cfg.get("eval", {}).get("hmm_n_states", 3))
        regimes = detector.fit_predict(all_test_returns)
        regime_results = regime_metrics(
            all_test_returns,
            all_positions,
            regimes,
            n_states=detector.n_states,
            interval=interval,
            spread_bps=cfg.get("costs", {}).get("spread_bps", 5.0),
            fee_rate=cfg.get("costs", {}).get("fee_rate", 0.0004),
            slippage_bps=cfg.get("costs", {}).get("slippage_bps", 2.0),
        )
        print_regime_report(regime_results, detector)
    except Exception as e:
        print(f"  Regime analysis skipped: {e}")

    # --------- サマリー ---------
    scorecards = [r["scorecard"] for r in fold_results.values() if "scorecard" in r]
    aggregate_scorecard = None
    if scorecards:
        aggregate_scorecard = {
            "alpha_excess_pt": float(np.mean([s["alpha_excess_pt"] for s in scorecards])),
            "sharpe_delta": float(np.mean([s["sharpe_delta"] for s in scorecards])),
            "maxdd_delta_pt": float(np.mean([s["maxdd_delta_pt"] for s in scorecards])),
            "win_rate_vs_bh": float(np.mean([s["win_rate_vs_bh"] for s in scorecards])),
            "collapse_guard_pass": all(s["collapse_guard_pass"] for s in scorecards),
            "collapse_guard_reasons": sorted(
                {reason for s in scorecards for reason in s["collapse_guard_reasons"]}
            ),
            "required": {
                "alpha_excess": all(s["required"]["alpha_excess"] for s in scorecards),
                "sharpe_delta": all(s["required"]["sharpe_delta"] for s in scorecards),
                "maxdd_delta": all(s["required"]["maxdd_delta"] for s in scorecards),
                "win_rate_vs_bh": all(s["required"]["win_rate_vs_bh"] for s in scorecards),
                "collapse_guard": all(s["required"]["collapse_guard"] for s in scorecards),
            },
            "stretch": {
                "alpha_excess": any(s["stretch"]["alpha_excess"] for s in scorecards),
                "maxdd_delta": any(s["stretch"]["maxdd_delta"] for s in scorecards),
            },
        }
        aggregate_scorecard["m2_pass"] = all(aggregate_scorecard["required"].values())
        aggregate_scorecard["stretch_hit"] = any(aggregate_scorecard["stretch"].values())

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for fold_idx, r in fold_results.items():
        m = r["metrics"]
        scorecard = r.get("scorecard")
        extra = f" | {_format_m2_scorecard(scorecard)}" if scorecard is not None else ""
        print(f"  Fold {fold_idx}: Sharpe={m.sharpe:.3f}, MaxDD={m.max_drawdown:.3f}, "
              f"Calmar={m.calmar:.3f}, TotalRet={m.total_return:.4f}{extra}")
    print(f"  Mean Sharpe: {np.mean(all_sharpes):.3f}")
    if aggregate_scorecard is not None:
        print(f"  Aggregate M2: {_format_m2_scorecard(aggregate_scorecard)}")
    dsr_summary = f"{dsr:.4f}" if np.isfinite(dsr) else "N/A"
    print(f"  PBO (simplified): {pbo:.4f} | Sharpe t-stat: {dsr_summary}")


if __name__ == "__main__":
    main()
