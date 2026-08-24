"""Policy statistics shared by training, evaluation, and reporting.

This module is deliberately model-agnostic.  It contains only calculations
about the emitted position path, so CLIs and experiment stages do not need to
import the Actor-Critic training implementation.
"""
from __future__ import annotations

import numpy as np


def action_stats(positions: np.ndarray, benchmark_position: float = 0.0) -> dict:
    """Summarize an absolute position path relative to its benchmark."""
    total = max(len(positions), 1)
    active_eps = 0.05
    overlay = np.asarray(positions, dtype=np.float64) - float(benchmark_position)
    delta = np.abs(np.diff(overlay)) if total > 1 else np.zeros(0, dtype=np.float64)
    counts = {
        "long": int((overlay > active_eps).sum()),
        "short": int((overlay < -active_eps).sum()),
        "flat": int((np.abs(overlay) <= active_eps).sum()),
    }
    long_rate = counts["long"] / total
    short_rate = counts["short"] / total
    flat_rate = counts["flat"] / total
    mean_overlay = float(np.mean(overlay)) if total > 0 else 0.0
    turnover = float(delta.sum()) if delta.size > 0 else 0.0
    nonzero_delta = delta[delta > 1e-8]
    step_ref = float(np.quantile(nonzero_delta, 0.90)) if nonzero_delta.size > 0 else active_eps
    step_ref = max(step_ref, active_eps)
    hard_switches = int((delta > active_eps).sum()) if delta.size > 0 else 0
    flow_switches = int(np.rint(turnover / step_ref)) if turnover > 0.0 else 0
    switches = max(hard_switches, flow_switches)
    avg_hold = total / max(switches, 1)
    return {
        "long": long_rate,
        "short": short_rate,
        "flat": flat_rate,
        "mean": mean_overlay,
        "switches": switches,
        "avg_hold": avg_hold,
        "counts": counts,
        "turnover": turnover,
        "step_ref": step_ref,
        "mean_abs_delta": turnover / max(total - 1, 1),
    }


def format_action_stats(stats: dict) -> str:
    return (
        f"long={stats['long']:.0%} short={stats['short']:.0%} flat={stats['flat']:.0%} "
        f"mean={stats['mean']:+.3f} switches={stats['switches']} "
        f"avg_hold={stats['avg_hold']:.1f}b turnover={stats['turnover']:.2f}"
    )
