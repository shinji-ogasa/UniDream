from __future__ import annotations

import numpy as np

from unidream.data.oracle import hindsight_oracle_dp


def compute_base_oracle(
    *,
    train_returns,
    val_returns,
    oracle_cfg: dict,
    reward_cfg: dict,
    costs_cfg: dict,
    default_action_values,
) -> dict:
    oracle_action_values = np.asarray(
        oracle_cfg.get("action_values", default_action_values),
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
    val_oracle_actions, _, _ = hindsight_oracle_dp(
        val_returns,
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
        discount=oracle_cfg.get("discount", 1.0),
        min_hold=oracle_min_hold,
        reward_mode=oracle_reward_mode,
        benchmark_position=oracle_benchmark_position,
        action_values=oracle_action_values,
    )

    return {
        "oracle_action_values": oracle_action_values,
        "oracle_min_hold": oracle_min_hold,
        "oracle_soft_temp": oracle_soft_temp,
        "oracle_reward_mode": oracle_reward_mode,
        "oracle_benchmark_position": oracle_benchmark_position,
        "oracle_teacher_mode": oracle_teacher_mode,
        "oracle_actions": oracle_actions,
        "oracle_values": oracle_values,
        "oracle_soft_labels": oracle_soft_labels,
        "val_oracle_actions": val_oracle_actions,
        "oracle_positions": oracle_action_values[oracle_actions].astype(np.float32),
        "val_oracle_positions": oracle_action_values[val_oracle_actions].astype(np.float32),
    }
