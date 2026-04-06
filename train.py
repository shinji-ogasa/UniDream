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
from datetime import datetime

import numpy as np
import torch

from unidream.data.download import (
    fetch_binance_ohlcv,
    fetch_funding_rate,
    fetch_open_interest_hist,
    fetch_mark_price_klines,
)
from unidream.data.features import compute_features, get_raw_returns, augment_with_rebound_features
from unidream.data.oracle import (
    _forward_window_stats,
    oracle_to_dataset,
    ACTIONS as _ACTIONS,
)
from unidream.data.dataset import WFODataset, SequenceDataset
from unidream.actor_critic.imagination_ac import _action_stats, _fmt_action_stats, _ac_alerts
from unidream.eval.backtest import Backtest, pnl_attribution
from unidream.eval.regime import RegimeDetector, regime_metrics, print_regime_report
from unidream.experiments.runtime import (
    load_config,
    load_training_features,
    resolve_costs,
    set_seed,
)
from unidream.experiments.fold_runtime import PIPELINE_STAGES, prepare_fold_runtime, stage_idx
from unidream.experiments.oracle_post import apply_oracle_postprocess
from unidream.experiments.oracle_stage import compute_base_oracle
from unidream.experiments.regime_runtime import fit_fold_regimes
from unidream.experiments.oracle_teacher import compute_teacher_oracle
from unidream.experiments.train_pipeline import run_wfo_folds
from unidream.experiments.ac_stage import run_ac_stage
from unidream.experiments.bc_setup import prepare_bc_setup
from unidream.experiments.bc_stage import run_bc_stage
from unidream.experiments.test_stage import run_test_stage
from unidream.experiments.val_selector_stage import run_val_selector_stage
from unidream.experiments.wm_stage import prepare_world_model_stage
from unidream.experiments.train_reporting import (
    aggregate_scorecards,
    compute_overfitting_diagnostics,
    print_stage_summary,
    print_training_summary,
)
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _benchmark_positions(length: int, cfg: dict) -> np.ndarray:
    benchmark_pos = cfg.get("reward", {}).get("benchmark_position", 1.0)
    return np.full(length, benchmark_pos, dtype=np.float64)


def _benchmark_position_value(cfg: dict) -> float:
    return float(cfg.get("reward", {}).get("benchmark_position", 1.0))




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
        "candidate": candidate,
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


def _candidate_to_text(candidate) -> str:
    if isinstance(candidate, dict):
        scale = float(candidate.get("scale", 0.0))
        adv = float(candidate.get("adv", 0.0))
        return f"scale={scale:.3f} adv={adv:.2f}"
    return f"scale={float(candidate):.3f}"


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

    fold_runtime = prepare_fold_runtime(
        fold_idx=fold_idx,
        checkpoint_dir=checkpoint_dir,
        ac_cfg=ac_cfg,
        resume=resume,
        start_from=start_from,
        stop_after=stop_after,
    )
    fold_ckpt_dir = fold_runtime["fold_ckpt_dir"]
    wm_path = fold_runtime["wm_path"]
    bc_path = fold_runtime["bc_path"]
    ac_path = fold_runtime["ac_path"]
    ac_max_steps_cfg = fold_runtime["ac_max_steps_cfg"]
    ignore_ac_ckpt = fold_runtime["ignore_ac_ckpt"]
    start_idx = fold_runtime["start_idx"]
    stop_idx = fold_runtime["stop_idx"]
    has_wm_ckpt = fold_runtime["has_wm_ckpt"]
    has_bc_ckpt = fold_runtime["has_bc_ckpt"]
    has_ac_ckpt = fold_runtime["has_ac_ckpt"]
    has_wm = fold_runtime["has_wm"]
    has_bc = fold_runtime["has_bc"]
    has_ac = fold_runtime["has_ac"]

    if start_idx > stage_idx("wm") and not has_wm_ckpt:
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
    oracle_bundle = compute_base_oracle(
        train_returns=train_returns,
        val_returns=wfo_dataset.val_returns,
        oracle_cfg=oracle_cfg,
        reward_cfg=reward_cfg,
        costs_cfg=costs_cfg,
        default_action_values=cfg.get("actions", {}).get("values", _ACTIONS),
    )
    oracle_action_values = oracle_bundle["oracle_action_values"]
    oracle_min_hold = oracle_bundle["oracle_min_hold"]
    oracle_soft_temp = oracle_bundle["oracle_soft_temp"]
    oracle_reward_mode = oracle_bundle["oracle_reward_mode"]
    oracle_benchmark_position = oracle_bundle["oracle_benchmark_position"]
    oracle_teacher_mode = oracle_bundle["oracle_teacher_mode"]
    oracle_actions = oracle_bundle["oracle_actions"]
    oracle_values = oracle_bundle["oracle_values"]
    oracle_soft_labels = oracle_bundle["oracle_soft_labels"]
    print(f"  Oracle computed: {len(oracle_actions)} steps, "
          f"mean value={oracle_values.mean():.4f}")
    print(f"  Oracle objective: {oracle_reward_mode} (benchmark={oracle_benchmark_position:+.2f})")
    _oracle_pos = oracle_action_values[oracle_actions]
    _oracle_s = _action_stats(_oracle_pos, benchmark_position=_benchmark_position_value(cfg))
    print(f"  Oracle dist: {_fmt_action_stats(_oracle_s)}")

    # Val oracle actions（分布比較・WM 学習に使用）
    oracle_positions = oracle_bundle["oracle_positions"]
    val_oracle_positions = oracle_bundle["val_oracle_positions"]
    teacher_bundle = compute_teacher_oracle(
        teacher_mode=oracle_teacher_mode,
        base_oracle_positions=oracle_positions,
        base_val_oracle_positions=val_oracle_positions,
        base_oracle_values=oracle_values,
        train_returns=train_returns,
        val_returns=wfo_dataset.val_returns,
        train_features=wfo_dataset.train_features,
        val_features=wfo_dataset.val_features,
        feature_columns=getattr(wfo_dataset, "feature_columns", []),
        oracle_action_values=oracle_action_values,
        oracle_cfg=oracle_cfg,
        ac_cfg=ac_cfg,
        oracle_benchmark_position=oracle_benchmark_position,
    )
    oracle_positions = teacher_bundle["oracle_positions"]
    val_oracle_positions = teacher_bundle["val_oracle_positions"]
    if teacher_bundle["oracle_values"] is not None:
        oracle_values = teacher_bundle["oracle_values"]
    if teacher_bundle["teacher_message"]:
        print(teacher_bundle["teacher_message"])
    oracle_positions, val_oracle_positions, outcome_edge = apply_oracle_postprocess(
        oracle_positions=oracle_positions,
        val_oracle_positions=val_oracle_positions,
        oracle_action_values=oracle_action_values,
        oracle_cfg=oracle_cfg,
        ac_cfg=ac_cfg,
        bc_cfg=bc_cfg,
        oracle_reward_mode=oracle_reward_mode,
        oracle_benchmark_position=oracle_benchmark_position,
        oracle_teacher_mode=oracle_teacher_mode,
        train_returns=train_returns,
        forward_window_stats_fn=_forward_window_stats,
    )
    if oracle_cfg.get("use_aim_targets", False):
        _aim_s = _action_stats(oracle_positions, benchmark_position=_benchmark_position_value(cfg))
        print(f"  Oracle aim dist: {_fmt_action_stats(_aim_s)}")
    if bc_cfg.get("outcome_relabel_bad_to_benchmark", False) and outcome_edge is not None:
        relabeled = int(np.sum(oracle_positions >= oracle_benchmark_position - 1e-6))
        print(f"  Outcome relabel active; benchmark-or-higher targets={relabeled}")

    # --------- HMM レジーム事後確率（Actor 入力用）---------
    # Actor 生成前に計算して regime_dim を確定する
    n_states = cfg.get("eval", {}).get("hmm_n_states", 3)
    hmm_det = None
    regime_dim = 0
    train_regime_probs = None
    val_regime_probs = None
    test_regime_probs = None
    try:
        regime_bundle = fit_fold_regimes(
            train_returns=wfo_dataset.train_returns,
            val_returns=wfo_dataset.val_returns,
            test_returns=wfo_dataset.test_returns,
            n_states=n_states,
        )
        hmm_det = regime_bundle["detector"]
        regime_dim = regime_bundle["regime_dim"]
        train_regime_probs = regime_bundle["train_regime_probs"]
        val_regime_probs = regime_bundle["val_regime_probs"]
        test_regime_probs = regime_bundle["test_regime_probs"]
        print(f"[Regime] HMM fitted, regime_dim={regime_dim}")
    except Exception as e:
        print(f"[Regime] HMM skipped: {e}")

    # --------- Step 2: 世界モデル学習 ---------
    ensemble, wm_trainer = prepare_world_model_stage(
        obs_dim=obs_dim,
        cfg=cfg,
        device=device,
        has_wm=has_wm,
        wm_path=wm_path,
        wfo_dataset=wfo_dataset,
        oracle_positions=oracle_positions,
        val_oracle_positions=val_oracle_positions,
        train_returns=train_returns,
        log_ts=_ts,
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
    bc_advantage_values = bc_setup["bc_advantage_values"]

    bc_trainer = run_bc_stage(
        actor=actor,
        ensemble=ensemble,
        bc_cfg=bc_cfg,
        oracle_cfg=oracle_cfg,
        ac_cfg=ac_cfg,
        reward_cfg=reward_cfg,
        device=device,
        has_bc=has_bc,
        start_idx=start_idx,
        bc_stage_idx=stage_idx("bc"),
        bc_path=bc_path,
        z_train=z_train,
        h_train=h_train,
        oracle_positions=oracle_positions,
        train_regime_probs=train_regime_probs,
        oracle_soft_labels=oracle_soft_labels,
        bc_sample_quality=bc_sample_quality,
        bc_advantage_values=bc_advantage_values,
        log_ts=_ts,
    )

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

    ac_trainer = run_ac_stage(
        actor=actor,
        ensemble=ensemble,
        cfg=cfg,
        ac_cfg=ac_cfg,
        wm_cfg=wm_cfg,
        costs_cfg=costs_cfg,
        device=device,
        has_ac=has_ac,
        ac_path=ac_path,
        z_train=z_train,
        h_train=h_train,
        oracle_positions=oracle_positions,
        train_regime_probs=train_regime_probs,
        wfo_dataset=wfo_dataset,
        wm_trainer=wm_trainer,
        seq_len=seq_len,
        val_regime_probs=val_regime_probs,
        val_oracle_positions=val_oracle_positions,
        start_idx=start_idx,
        stop_idx=stop_idx,
        ac_stage_idx=stage_idx("ac"),
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

    if stop_after == "ac":
        print(f"\n[{_ts()}] [Stop] Requested stop after AC")
        return {"fold": fold_idx, "completed_stage": "ac"}

    run_val_selector_stage(
        actor=actor,
        wm_trainer=wm_trainer,
        wfo_dataset=wfo_dataset,
        seq_len=seq_len,
        val_regime_probs=val_regime_probs,
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
    )
    test_result["fold"] = fold_idx
    return test_result


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
        choices=PIPELINE_STAGES,
        help="Start pipeline from this stage",
    )
    parser.add_argument(
        "--stop-after",
        default="test",
        choices=PIPELINE_STAGES,
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

    if stage_idx(args.start_from) > stage_idx(args.stop_after):
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
    features_df, raw_returns = load_training_features(
        symbol=symbol,
        interval=interval,
        start=args.start,
        end=args.end,
        zscore_window=zscore_window,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
    )

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
    splits = build_wfo_splits(features_df, data_cfg)
    print(f"  {len(splits)} folds")

    if len(splits) == 0:
        print("ERROR: WFO splits が空です。データ期間が短すぎます。")
        return

    try:
        splits, selected_folds = select_wfo_splits(splits, args.folds)
    except ValueError as exc:
        parser.error(str(exc))
    if selected_folds is not None:
        print(f"  Running selected folds only: {selected_folds}")

    # --------- 各 Fold の学習・評価 ---------
    fold_results = run_wfo_folds(
        features_df=features_df,
        raw_returns=raw_returns,
        splits=splits,
        data_cfg=data_cfg,
        cfg=cfg,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        start_from=args.start_from,
        stop_after=args.stop_after,
        run_fold_fn=run_fold,
    )

    if args.stop_after != "test":
        print_stage_summary(fold_results, args.stop_after)
        return

    # --------- PBO / Deflated Sharpe ---------
    # 注意: PBO は「fold を戦略候補扱いした IS/OOS 分割の簡略版」。
    #       標準 CSCV-PBO（Bailey & Lopez de Prado 2014）ではない。
    #       複数モデル構成を比較する際は CSCV 版に差し替えること。
    # 注意: DSR は n_trials=1（ハイパーパラメータ探索なし）のため、
    #       多重比較補正なしの「best fold Sharpe の t 統計量」として機能する。
    print("\n[Eval] Overfitting Diagnostics (simplified)...")
    eval_cfg = cfg.get("eval", {})
    pbo, dsr, all_sharpes = compute_overfitting_diagnostics(fold_results, eval_cfg)
    print(f"  PBO (simplified): {pbo:.4f} (< 0.5 is better)")
    dsr_str = f"{dsr:.4f}" if np.isfinite(dsr) else f"N/A ({dsr})"
    print(f"  Sharpe t-stat (DSR, n_trials=1): {dsr_str} (> 0 is better)")

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
    aggregate_scorecard = aggregate_scorecards(scorecards)
    print_training_summary(
        fold_results,
        all_sharpes,
        aggregate_scorecard,
        pbo,
        dsr,
        _format_m2_scorecard,
    )


if __name__ == "__main__":
    main()
