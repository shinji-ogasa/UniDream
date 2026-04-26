from __future__ import annotations

import numpy as np

from unidream.data.oracle import hindsight_signal_teacher


def compute_teacher_oracle(
    *,
    teacher_mode: str,
    base_oracle_positions: np.ndarray,
    base_val_oracle_positions: np.ndarray,
    base_oracle_values: np.ndarray,
    train_returns: np.ndarray,
    val_returns: np.ndarray,
    train_features,
    val_features,
    feature_columns,
    oracle_action_values: np.ndarray,
    oracle_cfg: dict,
    ac_cfg: dict,
    oracle_benchmark_position: float,
) -> dict:
    oracle_positions = np.asarray(base_oracle_positions, dtype=np.float32)
    val_oracle_positions = np.asarray(base_val_oracle_positions, dtype=np.float32)
    oracle_values = np.asarray(base_oracle_values, dtype=np.float32)
    teacher_message = None

    abs_min = ac_cfg.get("abs_min_position", float(np.min(oracle_action_values)))
    abs_max = ac_cfg.get("abs_max_position", float(np.max(oracle_action_values)))
    if teacher_mode in ("dp", "base", "hindsight"):
        teacher_message = "  Oracle teacher: base DP"
    elif teacher_mode == "signal_aim":
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
        val_oracle_positions, _ = hindsight_signal_teacher(
            val_returns,
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
        teacher_message = (
            "  Signal teacher: "
            f"horizons={teacher_horizons} floor={oracle_cfg.get('signal_floor_position', abs_min):+.2f} "
            f"scale={oracle_cfg.get('signal_scale', 1.5):.2f}"
        )
    else:
        raise ValueError(
            f"Unsupported oracle.teacher_mode '{teacher_mode}'. "
            "Phase 8 mainline supports only signal_aim or base DP."
        )

    return {
        "oracle_positions": np.asarray(oracle_positions, dtype=np.float32),
        "val_oracle_positions": np.asarray(val_oracle_positions, dtype=np.float32),
        "oracle_values": np.asarray(oracle_values, dtype=np.float32),
        "teacher_message": teacher_message,
    }
