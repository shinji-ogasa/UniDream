"""Validation policy selection for the current B&H-relative actor."""
from __future__ import annotations

import numpy as np

from unidream.experiments.m2 import (
    benchmark_position_value,
    directional_collapse,
    m2_scorecard,
)


def benchmark_positions(length: int, cfg: dict) -> np.ndarray:
    return np.full(length, benchmark_position_value(cfg), dtype=np.float64)


def policy_score(metrics, stats: dict, benchmark_position: float = 0.0) -> tuple[float, str]:
    alpha_excess = 100.0 * (metrics.alpha_excess or 0.0)
    sharpe_delta = metrics.sharpe_delta or 0.0
    score = 2.0 * alpha_excess + 5.0 * sharpe_delta
    penalty = 0.0
    overlay_mode = abs(float(benchmark_position)) > 1e-8
    collapsed = (
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
    if collapsed and stats["long"] >= 0.85:
        penalty += 120.0
    if collapsed and stats["short"] >= 0.85:
        penalty += 120.0
    if collapsed or (not overlay_mode and stats["flat"] >= 0.80):
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


def selector_config(ac_cfg: dict) -> dict:
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
    return (
        abs(float(benchmark_position)) > 1e-8
        and stats["flat"] >= 0.95
        and stats["switches"] == 0
    )


def selector_candidate(
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
    collapsed = directional_collapse(stats)
    scorecard = m2_scorecard(metrics, stats, cfg or {})
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
        elif collapsed or not scorecard["collapse_guard_pass"]:
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
    turnover_penalty = selector_cfg["turnover_score_coef"] * float(stats["turnover"])
    turnover_penalty += selector_cfg["turnover_excess_score_coef"] * max(
        0.0, float(stats["turnover"]) - selector_cfg["turnover_target"]
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
        f"maxddΔ={maxdd_delta_pt:+.2f}pt barwin={win_rate_vs_bh:.1%} "
        f"periodwin={period_win_rate_vs_bh:.1%} score={score:.3f} "
        f"long={stats['long']:.0%} short={stats['short']:.0%} "
        f"flat={stats['flat']:.0%} turnover={stats['turnover']:.2f} "
        f"M2={'pass' if scorecard['m2_pass'] else 'miss'}"
    )
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


def candidate_to_text(candidate: dict[str, float]) -> str:
    return f"scale={float(candidate['scale']):.3f} adv={float(candidate['adv']):.2f}"


def select_policy_candidate(candidates: list[dict], selector_cfg: dict) -> dict:
    valid = [candidate for candidate in candidates if candidate["reject_reason"] is None]
    pool = valid if valid else candidates
    benchmark_hold = next((candidate for candidate in pool if candidate["benchmark_hold"]), None)
    best = max(pool, key=lambda candidate: candidate["score"])
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
        candidate
        for candidate in pool
        if candidate["score"] >= score_floor
        and candidate["alpha_excess_pt"] >= alpha_floor
        and candidate["sharpe_delta"] >= sharpe_floor
    ] or [best]
    mode = selector_cfg.get("near_best_tiebreak", "conservative")
    if mode == "balanced":
        key_fn = lambda candidate: (
            -candidate["sharpe_delta"], candidate["maxdd_delta_pt"],
            -candidate["selector_win_rate"], candidate["stats"]["turnover"],
            -candidate["score"],
        )
    elif mode == "score":
        key_fn = lambda candidate: (
            -candidate["score"], -candidate["sharpe_delta"],
            candidate["maxdd_delta_pt"], candidate["stats"]["turnover"],
        )
    else:
        key_fn = lambda candidate: (
            0 if candidate["benchmark_hold"] else 1,
            candidate["stats"]["turnover"], candidate["maxdd_delta_pt"],
            -candidate["selector_win_rate"], -candidate["score"],
        )
    return min(near_best, key=key_fn)
