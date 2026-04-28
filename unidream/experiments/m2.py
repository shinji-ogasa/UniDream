from __future__ import annotations

import numpy as np


def benchmark_position_value(cfg: dict) -> float:
    return float(cfg.get("reward", {}).get("benchmark_position", 1.0))


def directional_collapse(stats: dict) -> bool:
    return (
        max(stats["long"], stats["short"]) >= 0.80
        and stats["switches"] <= 5
        and stats["turnover"] < 1.0
    )


def collapse_guard(stats: dict, benchmark_position: float) -> tuple[bool, list[str]]:
    overlay_mode = abs(float(benchmark_position)) > 1e-8
    reasons: list[str] = []
    if directional_collapse(stats):
        reasons.append("directional_collapse")
    if not overlay_mode and stats["flat"] >= 0.80:
        reasons.append("flat_collapse")
    return len(reasons) == 0, reasons


def goal_cfg(cfg: dict) -> dict:
    targets = cfg.get("targets", {})
    return {
        "alpha_excess_pt": float(targets.get("alpha_excess_pt", 5.0)),
        "sharpe_delta": float(targets.get("sharpe_delta", 0.20)),
        "maxdd_delta_pt": float(targets.get("maxdd_delta_pt", -10.0)),
        "win_rate_vs_bh": float(targets.get("win_rate_vs_bh", 0.60)),
        "stretch_alpha_excess_pt": float(targets.get("stretch_alpha_excess_pt", 8.0)),
        "stretch_maxdd_delta_pt": float(targets.get("stretch_maxdd_delta_pt", -15.0)),
    }


def m2_scorecard(metrics, stats: dict, cfg: dict) -> dict:
    goals = goal_cfg(cfg)
    benchmark_position = benchmark_position_value(cfg)
    alpha_excess_pt = 100.0 * float(metrics.alpha_excess or 0.0)
    sharpe_delta = float(metrics.sharpe_delta or 0.0)
    maxdd_delta_pt = 100.0 * float(metrics.maxdd_delta or 0.0)
    win_rate_vs_bh = float(metrics.win_rate_vs_bh or 0.0)
    period_win_raw = getattr(metrics, "period_win_rate_vs_bh", None)
    period_win_rate_vs_bh = win_rate_vs_bh if period_win_raw is None else float(period_win_raw)
    upside_capture = getattr(metrics, "upside_capture", None)
    downside_capture = getattr(metrics, "downside_capture", None)
    collapse_pass, collapse_reasons = collapse_guard(stats, benchmark_position)
    required = {
        "alpha_excess": alpha_excess_pt >= goals["alpha_excess_pt"],
        "sharpe_delta": sharpe_delta >= goals["sharpe_delta"],
        "maxdd_delta": maxdd_delta_pt <= goals["maxdd_delta_pt"],
        "win_rate_vs_bh": period_win_rate_vs_bh >= goals["win_rate_vs_bh"],
        "collapse_guard": collapse_pass,
    }
    stretch = {
        "alpha_excess": alpha_excess_pt >= goals["stretch_alpha_excess_pt"],
        "maxdd_delta": maxdd_delta_pt <= goals["stretch_maxdd_delta_pt"],
    }
    return {
        "alpha_excess_pt": alpha_excess_pt,
        "sharpe_delta": sharpe_delta,
        "maxdd_delta_pt": maxdd_delta_pt,
        "win_rate_vs_bh": win_rate_vs_bh,
        "period_win_rate_vs_bh": period_win_rate_vs_bh,
        "upside_capture": None if upside_capture is None else float(upside_capture),
        "downside_capture": None if downside_capture is None else float(downside_capture),
        "collapse_guard_pass": collapse_pass,
        "collapse_guard_reasons": collapse_reasons,
        "required": required,
        "stretch": stretch,
        "m2_pass": all(required.values()),
        "stretch_hit": any(stretch.values()),
    }


def format_m2_scorecard(scorecard: dict) -> str:
    guard = "pass" if scorecard["collapse_guard_pass"] else ",".join(scorecard["collapse_guard_reasons"])
    m2_state = "PASS" if scorecard["m2_pass"] else "MISS"
    stretch_state = "hit" if scorecard["stretch_hit"] else "miss"
    return (
        f"M2={m2_state} stretch={stretch_state} "
        f"alpha={scorecard['alpha_excess_pt']:+.2f}pt "
        f"sharpeΔ={scorecard['sharpe_delta']:+.3f} "
        f"maxddΔ={scorecard['maxdd_delta_pt']:+.2f}pt "
        f"barwin={scorecard['win_rate_vs_bh']:.1%} "
        f"periodwin={scorecard['period_win_rate_vs_bh']:.1%} "
        f"guard={guard}"
    )
