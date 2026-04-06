from __future__ import annotations

import numpy as np

from unidream.data.oracle import (
    feature_dual_teacher,
    feature_linear_teacher,
    feature_stress_teacher,
    fit_linear_future_signal,
    hindsight_signal_teacher,
)


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
    teacher_columns = tuple(feature_columns or [])

    if teacher_mode == "signal_aim":
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
    elif teacher_mode == "feature_stress":
        oracle_positions, oracle_signal = feature_stress_teacher(
            train_features,
            feature_columns=teacher_columns,
            benchmark_position=oracle_benchmark_position,
            min_position=oracle_cfg.get("stress_floor_position", abs_min),
            max_position=oracle_cfg.get("stress_ceiling_position", abs_max),
            fast_vol_col=oracle_cfg.get("stress_fast_vol_col", "rv_16"),
            slow_vol_col=oracle_cfg.get("stress_slow_vol_col", "rv_96"),
            shock_col=oracle_cfg.get("stress_shock_col", "atr_norm_ret"),
            drift_col=oracle_cfg.get("stress_drift_col", "open_ret"),
            macd_col=oracle_cfg.get("stress_macd_col", "macd"),
            macd_signal_col=oracle_cfg.get("stress_macd_signal_col", "macd_signal"),
            funding_col=oracle_cfg.get("stress_funding_col", "funding_rate"),
            fast_vol_threshold=oracle_cfg.get("stress_fast_vol_threshold", 0.8),
            slow_vol_threshold=oracle_cfg.get("stress_slow_vol_threshold", 0.8),
            shock_threshold=oracle_cfg.get("stress_shock_threshold", 0.8),
            drift_threshold=oracle_cfg.get("stress_drift_threshold", 0.2),
            trend_threshold=oracle_cfg.get("stress_trend_threshold", 0.1),
            funding_threshold=oracle_cfg.get("stress_funding_threshold", -0.5),
            fast_vol_weight=oracle_cfg.get("stress_fast_vol_weight", 0.45),
            slow_vol_weight=oracle_cfg.get("stress_slow_vol_weight", 0.25),
            shock_weight=oracle_cfg.get("stress_shock_weight", 0.20),
            drift_weight=oracle_cfg.get("stress_drift_weight", 0.10),
            trend_weight=oracle_cfg.get("stress_trend_weight", 0.15),
            funding_weight=oracle_cfg.get("stress_funding_weight", 0.05),
            entry_threshold=oracle_cfg.get("stress_entry_threshold", 0.15),
            signal_scale=oracle_cfg.get("stress_signal_scale", 1.0),
        )
        val_oracle_positions, _ = feature_stress_teacher(
            val_features,
            feature_columns=teacher_columns,
            benchmark_position=oracle_benchmark_position,
            min_position=oracle_cfg.get("stress_floor_position", abs_min),
            max_position=oracle_cfg.get("stress_ceiling_position", abs_max),
            fast_vol_col=oracle_cfg.get("stress_fast_vol_col", "rv_16"),
            slow_vol_col=oracle_cfg.get("stress_slow_vol_col", "rv_96"),
            shock_col=oracle_cfg.get("stress_shock_col", "atr_norm_ret"),
            drift_col=oracle_cfg.get("stress_drift_col", "open_ret"),
            macd_col=oracle_cfg.get("stress_macd_col", "macd"),
            macd_signal_col=oracle_cfg.get("stress_macd_signal_col", "macd_signal"),
            funding_col=oracle_cfg.get("stress_funding_col", "funding_rate"),
            fast_vol_threshold=oracle_cfg.get("stress_fast_vol_threshold", 0.8),
            slow_vol_threshold=oracle_cfg.get("stress_slow_vol_threshold", 0.8),
            shock_threshold=oracle_cfg.get("stress_shock_threshold", 0.8),
            drift_threshold=oracle_cfg.get("stress_drift_threshold", 0.2),
            trend_threshold=oracle_cfg.get("stress_trend_threshold", 0.1),
            funding_threshold=oracle_cfg.get("stress_funding_threshold", -0.5),
            fast_vol_weight=oracle_cfg.get("stress_fast_vol_weight", 0.45),
            slow_vol_weight=oracle_cfg.get("stress_slow_vol_weight", 0.25),
            shock_weight=oracle_cfg.get("stress_shock_weight", 0.20),
            drift_weight=oracle_cfg.get("stress_drift_weight", 0.10),
            trend_weight=oracle_cfg.get("stress_trend_weight", 0.15),
            funding_weight=oracle_cfg.get("stress_funding_weight", 0.05),
            entry_threshold=oracle_cfg.get("stress_entry_threshold", 0.15),
            signal_scale=oracle_cfg.get("stress_signal_scale", 1.0),
        )
        oracle_values = oracle_signal.astype(np.float32)
        teacher_message = (
            "  Feature stress teacher: "
            f"floor={oracle_cfg.get('stress_floor_position', abs_min):+.2f} "
            f"entry={oracle_cfg.get('stress_entry_threshold', 0.15):.2f} "
            f"scale={oracle_cfg.get('stress_signal_scale', 1.0):.2f}"
        )
    elif teacher_mode == "feature_dual":
        oracle_positions, oracle_signal = feature_dual_teacher(
            train_features,
            feature_columns=teacher_columns,
            benchmark_position=oracle_benchmark_position,
            min_position=oracle_cfg.get("dual_floor_position", abs_min),
            max_position=oracle_cfg.get("dual_ceiling_position", abs_max),
            fast_vol_col=oracle_cfg.get("dual_fast_vol_col", "rv_16"),
            slow_vol_col=oracle_cfg.get("dual_slow_vol_col", "rv_96"),
            shock_col=oracle_cfg.get("dual_shock_col", "atr_norm_ret"),
            drift_col=oracle_cfg.get("dual_drift_col", "open_ret"),
            rsi_col=oracle_cfg.get("dual_rsi_col", "RSI_14"),
            macd_col=oracle_cfg.get("dual_macd_col", "macd"),
            macd_signal_col=oracle_cfg.get("dual_macd_signal_col", "macd_signal"),
            funding_col=oracle_cfg.get("dual_funding_col", "funding_rate"),
            short_fast_vol_threshold=oracle_cfg.get("dual_short_fast_vol_threshold", 0.8),
            short_slow_vol_threshold=oracle_cfg.get("dual_short_slow_vol_threshold", 0.8),
            short_shock_threshold=oracle_cfg.get("dual_short_shock_threshold", 0.8),
            short_drift_threshold=oracle_cfg.get("dual_short_drift_threshold", 0.2),
            short_trend_threshold=oracle_cfg.get("dual_short_trend_threshold", 0.1),
            long_slow_vol_threshold=oracle_cfg.get("dual_long_slow_vol_threshold", -0.6),
            long_rsi_threshold=oracle_cfg.get("dual_long_rsi_threshold", 0.8),
            long_macd_threshold=oracle_cfg.get("dual_long_macd_threshold", 0.8),
            long_drift_threshold=oracle_cfg.get("dual_long_drift_threshold", 0.8),
            long_funding_threshold=oracle_cfg.get("dual_long_funding_threshold", 0.5),
            short_scale=oracle_cfg.get("dual_short_scale", 0.75),
            long_scale=oracle_cfg.get("dual_long_scale", 0.75),
            short_entry_threshold=oracle_cfg.get("dual_short_entry_threshold", 0.25),
            long_entry_threshold=oracle_cfg.get("dual_long_entry_threshold", 0.20),
        )
        val_oracle_positions, _ = feature_dual_teacher(
            val_features,
            feature_columns=teacher_columns,
            benchmark_position=oracle_benchmark_position,
            min_position=oracle_cfg.get("dual_floor_position", abs_min),
            max_position=oracle_cfg.get("dual_ceiling_position", abs_max),
            fast_vol_col=oracle_cfg.get("dual_fast_vol_col", "rv_16"),
            slow_vol_col=oracle_cfg.get("dual_slow_vol_col", "rv_96"),
            shock_col=oracle_cfg.get("dual_shock_col", "atr_norm_ret"),
            drift_col=oracle_cfg.get("dual_drift_col", "open_ret"),
            rsi_col=oracle_cfg.get("dual_rsi_col", "RSI_14"),
            macd_col=oracle_cfg.get("dual_macd_col", "macd"),
            macd_signal_col=oracle_cfg.get("dual_macd_signal_col", "macd_signal"),
            funding_col=oracle_cfg.get("dual_funding_col", "funding_rate"),
            short_fast_vol_threshold=oracle_cfg.get("dual_short_fast_vol_threshold", 0.8),
            short_slow_vol_threshold=oracle_cfg.get("dual_short_slow_vol_threshold", 0.8),
            short_shock_threshold=oracle_cfg.get("dual_short_shock_threshold", 0.8),
            short_drift_threshold=oracle_cfg.get("dual_short_drift_threshold", 0.2),
            short_trend_threshold=oracle_cfg.get("dual_short_trend_threshold", 0.1),
            long_slow_vol_threshold=oracle_cfg.get("dual_long_slow_vol_threshold", -0.6),
            long_rsi_threshold=oracle_cfg.get("dual_long_rsi_threshold", 0.8),
            long_macd_threshold=oracle_cfg.get("dual_long_macd_threshold", 0.8),
            long_drift_threshold=oracle_cfg.get("dual_long_drift_threshold", 0.8),
            long_funding_threshold=oracle_cfg.get("dual_long_funding_threshold", 0.5),
            short_scale=oracle_cfg.get("dual_short_scale", 0.75),
            long_scale=oracle_cfg.get("dual_long_scale", 0.75),
            short_entry_threshold=oracle_cfg.get("dual_short_entry_threshold", 0.25),
            long_entry_threshold=oracle_cfg.get("dual_long_entry_threshold", 0.20),
        )
        oracle_values = oracle_signal.astype(np.float32)
        teacher_message = (
            "  Feature dual teacher: "
            f"range=[{oracle_cfg.get('dual_floor_position', abs_min):+.2f},"
            f"{oracle_cfg.get('dual_ceiling_position', abs_max):+.2f}] "
            f"entries(short={oracle_cfg.get('dual_short_entry_threshold', 0.25):.2f},"
            f" long={oracle_cfg.get('dual_long_entry_threshold', 0.20):.2f})"
        )
    elif teacher_mode == "feature_ridge":
        ridge_horizons = tuple(oracle_cfg.get("ridge_horizons", [4, 16, 64]))
        ridge_weights = tuple(oracle_cfg.get("ridge_horizon_weights", [0.2, 0.3, 0.5]))
        signal_model = fit_linear_future_signal(
            train_features,
            train_returns,
            horizons=ridge_horizons,
            horizon_weights=ridge_weights,
            ridge_alpha=oracle_cfg.get("ridge_alpha", 1.0),
            feature_clip=oracle_cfg.get("ridge_feature_clip", 5.0),
            signal_clip=oracle_cfg.get("ridge_signal_clip", 4.0),
        )
        oracle_positions, oracle_signal = feature_linear_teacher(
            train_features,
            signal_model,
            benchmark_position=oracle_benchmark_position,
            min_position=oracle_cfg.get("ridge_floor_position", abs_min),
            max_position=oracle_cfg.get("ridge_ceiling_position", abs_max),
            pos_deadzone=oracle_cfg.get("ridge_pos_deadzone", 0.10),
            neg_deadzone=oracle_cfg.get("ridge_neg_deadzone", 0.10),
            pos_scale=oracle_cfg.get("ridge_pos_scale", 1.0),
            neg_scale=oracle_cfg.get("ridge_neg_scale", 1.0),
            feature_clip=oracle_cfg.get("ridge_feature_clip", 5.0),
        )
        val_oracle_positions, _ = feature_linear_teacher(
            val_features,
            signal_model,
            benchmark_position=oracle_benchmark_position,
            min_position=oracle_cfg.get("ridge_floor_position", abs_min),
            max_position=oracle_cfg.get("ridge_ceiling_position", abs_max),
            pos_deadzone=oracle_cfg.get("ridge_pos_deadzone", 0.10),
            neg_deadzone=oracle_cfg.get("ridge_neg_deadzone", 0.10),
            pos_scale=oracle_cfg.get("ridge_pos_scale", 1.0),
            neg_scale=oracle_cfg.get("ridge_neg_scale", 1.0),
            feature_clip=oracle_cfg.get("ridge_feature_clip", 5.0),
        )
        oracle_values = oracle_signal.astype(np.float32)
        teacher_message = (
            "  Feature ridge teacher: "
            f"range=[{oracle_cfg.get('ridge_floor_position', abs_min):+.2f},"
            f"{oracle_cfg.get('ridge_ceiling_position', abs_max):+.2f}] "
            f"ridge={oracle_cfg.get('ridge_alpha', 1.0):.2f}"
        )

    return {
        "oracle_positions": np.asarray(oracle_positions, dtype=np.float32),
        "val_oracle_positions": np.asarray(val_oracle_positions, dtype=np.float32),
        "oracle_values": np.asarray(oracle_values, dtype=np.float32),
        "teacher_message": teacher_message,
    }
