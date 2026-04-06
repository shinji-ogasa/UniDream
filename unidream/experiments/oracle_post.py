from __future__ import annotations

import numpy as np

from unidream.data.oracle import smooth_aim_positions


def teacher_outcome_edge(
    returns: np.ndarray,
    positions: np.ndarray,
    benchmark_position: float,
    horizons: tuple[int, ...],
    horizon_weights: tuple[float, ...],
    forward_window_stats_fn,
) -> np.ndarray:
    weighted_forward = np.zeros(len(returns), dtype=np.float32)
    total_weight = 0.0
    for horizon, weight in zip(horizons, horizon_weights):
        if weight == 0.0:
            continue
        fwd_mean, _ = forward_window_stats_fn(np.asarray(returns, dtype=np.float32), int(horizon))
        weighted_forward += float(weight) * fwd_mean.astype(np.float32)
        total_weight += float(weight)
    if total_weight > 0.0:
        weighted_forward /= total_weight
    action_edge = (np.asarray(positions, dtype=np.float32) - float(benchmark_position)) * weighted_forward
    return np.clip(action_edge, 0.0, None).astype(np.float32)


def apply_oracle_postprocess(
    *,
    oracle_positions: np.ndarray,
    val_oracle_positions: np.ndarray,
    oracle_action_values: np.ndarray,
    oracle_cfg: dict,
    ac_cfg: dict,
    bc_cfg: dict,
    oracle_reward_mode: str,
    oracle_benchmark_position: float,
    oracle_teacher_mode: str,
    train_returns: np.ndarray,
    forward_window_stats_fn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if oracle_cfg.get("use_aim_targets", False):
        abs_min = ac_cfg.get("abs_min_position", float(np.min(oracle_action_values)))
        abs_max = ac_cfg.get("abs_max_position", float(np.max(oracle_action_values)))
        common_kwargs = {
            "max_step": oracle_cfg.get("aim_max_step", 0.25),
            "band": oracle_cfg.get("aim_band", 0.0),
            "initial_position": oracle_benchmark_position if oracle_reward_mode == "excess_bh" else 0.0,
            "min_position": abs_min,
            "max_position": abs_max,
            "benchmark_position": oracle_benchmark_position if oracle_reward_mode == "excess_bh" else 0.0,
            "underweight_confirm_bars": oracle_cfg.get("aim_underweight_confirm_bars", 0),
            "underweight_min_scale": oracle_cfg.get("aim_underweight_min_scale", 0.0),
            "underweight_floor_position": oracle_cfg.get("aim_underweight_floor_position"),
            "underweight_step_scale": oracle_cfg.get("aim_underweight_step_scale", 1.0),
        }
        oracle_positions = smooth_aim_positions(oracle_positions, **common_kwargs).astype(np.float32)
        val_oracle_positions = smooth_aim_positions(val_oracle_positions, **common_kwargs).astype(np.float32)

    outcome_edge = None
    bc_quality_mode = str(bc_cfg.get("sample_quality_mode", "none")).lower()
    if oracle_teacher_mode == "signal_aim" and (
        bc_quality_mode in {"outcome_edge", "outcome_edge_relabel"} or bc_cfg.get("outcome_relabel_bad_to_benchmark", False)
    ):
        outcome_horizons = tuple(oracle_cfg.get("signal_horizons", [4, 16, 64]))
        outcome_weights = tuple(oracle_cfg.get("signal_horizon_weights", [0.2, 0.3, 0.5]))
        outcome_edge = teacher_outcome_edge(
            train_returns,
            oracle_positions,
            benchmark_position=oracle_benchmark_position,
            horizons=outcome_horizons,
            horizon_weights=outcome_weights,
            forward_window_stats_fn=forward_window_stats_fn,
        )
        if bc_cfg.get("outcome_relabel_bad_to_benchmark", False):
            positive_edge = outcome_edge[outcome_edge > 0.0]
            relabel_quantile = float(np.clip(bc_cfg.get("outcome_relabel_quantile", 0.25), 0.0, 0.99))
            relabel_floor = float(np.quantile(positive_edge, relabel_quantile)) if positive_edge.size > 0 else 0.0
            bad_underweight = (oracle_positions < oracle_benchmark_position - 1e-6) & (outcome_edge <= relabel_floor)
            if bad_underweight.any():
                oracle_positions = oracle_positions.copy()
                oracle_positions[bad_underweight] = oracle_benchmark_position

    return oracle_positions, val_oracle_positions, outcome_edge
